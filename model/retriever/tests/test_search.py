"""Search driver invariants: run_name tất định + order-independent + canonicalize knob phụ,
random reproducible/capped, grid vét cạn + dedupe, run_search resume (skip existing).
Không cần train-data/torch."""
import search


def test_run_name_order_independent():
    a = search.deterministic_run_name({"epochs": 2, "train_hist_len": 64})
    b = search.deterministic_run_name({"train_hist_len": 64, "epochs": 2})
    assert a == b, "run_name phải độc lập thứ tự dict (sorted theo key)"


def test_canonicalize_drops_disabled_children():
    c = search.canonicalize({"use_synopsis": False, "synopsis_dim": 64, "epochs": 2})
    assert c == {"use_synopsis": False, "epochs": 2}, "tắt synopsis -> bỏ synopsis_dim"
    # 2 config chỉ khác knob con vô nghĩa -> CÙNG run_name (dedupe, không chạy lại model giống hệt)
    n48 = search.deterministic_run_name({"use_synopsis": False, "synopsis_dim": 48})
    n64 = search.deterministic_run_name({"use_synopsis": False, "synopsis_dim": 64})
    assert n48 == n64
    # bật synopsis -> synopsis_dim có nghĩa -> tên khác nhau
    assert (search.deterministic_run_name({"use_synopsis": True, "synopsis_dim": 48})
            != search.deterministic_run_name({"use_synopsis": True, "synopsis_dim": 64}))


def test_random_reproducible_and_capped():
    space = {"train_hist_len": [32, 64, 96], "logq_alpha": [1.0, 0.75], "optimizer": ["adam", "adamw"]}
    n1 = [name for name, _ in search.iter_configs(space, "random", n=5, seed=0)]
    n2 = [name for name, _ in search.iter_configs(space, "random", n=5, seed=0)]
    assert n1 == n2, "cùng seed -> cùng dãy config"
    assert len(n1) == 5 and len(set(n1)) == 5, "đúng n bộ, không trùng"


def test_grid_exhaustive_with_fixed_and_dedupe():
    space = {"use_synopsis": [False, True], "synopsis_dim": [48, 64]}
    cfgs = list(search.iter_configs(space, "grid", fixed={"epochs": 2}))
    names = [name for name, _ in cfgs]
    # canonical combos: syn=False (sd bỏ -> 1) + syn=True×{48,64} (2) = 3
    assert len(names) == len(set(names)) == 3
    assert all(ov.get("epochs") == 2 for _, ov in cfgs), "fixed áp cho mọi config"


def test_run_search_skips_existing():
    calls = []
    search.run_search(lambda name, **ov: calls.append(name),
                      [("a", {}), ("b", {}), ("c", {})],
                      exists_fn=lambda nm: nm == "b")
    assert calls == ["a", "c"], "phải bỏ run đã có trên Drive (b), chạy a + c (resume)"


def test_run_search_dry_run_no_call():
    calls = []
    res = search.run_search(lambda name, **ov: calls.append(name),
                            [("a", {"epochs": 2})], dry_run=True)
    assert calls == [] and len(res) == 1, "dry_run chỉ in tên, không gọi callback"
