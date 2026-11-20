from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from . import config
from .db import Base, engine, migrate
from .routers import auth, edits, gpu, projects, tasks
from .services import dispatcher


@asynccontextmanager
async def lifespan(_app: FastAPI):
    config.warn_insecure_defaults()
    Base.metadata.create_all(engine)
    migrate(engine)
    dispatcher.start()
    yield
    dispatcher.stop()


app = FastAPI(title="Room3D API", version="2.0", lifespan=lifespan)

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
app.include_router(edits.router)
app.include_router(gpu.router)


@app.get("/health", tags=["meta"])
def health():
    from .services.gpu_client import load_config
    return {"status": "ok", "auth_required": config.AUTH_REQUIRED, "gpu_configured": bool(load_config().get("url"))}


# The website itself (static files, hash-routed SPA). Mounted last so API routes win.
if config.FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=config.FRONTEND_DIR, html=True), name="frontend")
