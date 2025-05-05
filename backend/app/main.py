from fastapi import FastAPI
from app.api.v1.routes import router
from app.core.constants import TICKERS
from typing import List
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router=router, prefix="/api/v1")

@app.get("/tickers",response_model=List[str])
async def get_supported_tickers():
    return TICKERS

@app.get("/health")
async def health_check():
    return {"status":"health", "model_version":"1.0.01"}