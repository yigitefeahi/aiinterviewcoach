.PHONY: test lint typecheck run-backend run-frontend

test:
	cd backend && python3 -m pytest -q

lint:
	cd frontend && npm run lint

typecheck:
	cd backend && python3 -m mypy --config-file mypy.ini

run-backend:
	cd backend && python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000

run-frontend:
	cd frontend && npm run dev
