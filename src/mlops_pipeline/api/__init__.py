from fastapi import FastAPI

from .routes import router


def create_app() -> FastAPI:
    """Instantiate FastAPI application and register routers."""
    app = FastAPI(title="MLOps Pipeline API")
    app.include_router(router)
    return app


app = create_app()

