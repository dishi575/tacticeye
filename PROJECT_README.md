# TacticEye - run everything

Terminal 1 (backend):
    cd backend
    pip install -r requirements.txt
    uvicorn app.main:app --reload          # http://127.0.0.1:8000/docs

Terminal 2 (frontend):
    cd frontend
    npm install
    npm run dev                            # http://localhost:3000

Frontend reads the API URL from frontend/.env.local (NEXT_PUBLIC_API_URL, default http://127.0.0.1:8000).
Tests: cd backend && pip install -r requirements-dev.txt && python -m pytest
