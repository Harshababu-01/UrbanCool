"""FastAPI application for the UrbanCool data backend."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes_advisor import router as advisor_router
from backend.app.api.routes_cells import router as cells_router
from backend.app.api.routes_statistics import router as statistics_router


app = FastAPI(
	title="UrbanCool API",
	description="REST API for real Coimbatore heat-risk and cooling recommendation data.",
	version="1.0.0",
)
app.add_middleware(
	CORSMiddleware,
	allow_origins=[
		"http://localhost:3000",
		"http://localhost:5173",
		"http://127.0.0.1:3000",
		"http://127.0.0.1:5173",
	],
	allow_credentials=True,
	allow_methods=["GET", "POST"],
	allow_headers=["*"],
)
app.include_router(cells_router)
app.include_router(statistics_router)
app.include_router(advisor_router)


@app.get("/api/health")
def health():
	return {"status": "ok", "service": "UrbanCool API"}
