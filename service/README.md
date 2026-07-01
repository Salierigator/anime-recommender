cd service/backend
&&
MOCK_MODE=0 ../../venv/bin/uvicorn app.main:app --reload --port 8000

cd service/frontend
&&
npm run dev