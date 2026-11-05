from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import config
from .db import Base, engine
from .routers import auth, projects, tasks


@asynccontextmanager
async def lifespan(_app: FastAPI):
    config.warn_insecure_defaults()
    Base.metadata.create_all(engine)
    yield


app = FastAPI(title="Room3D API", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Range", "Accept-Ranges", "Content-Length"],
)

app.include_router(auth.router)
app.include_router(projects.router)
app.include_router(tasks.router)


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok", "auth_required": config.AUTH_REQUIRED}
