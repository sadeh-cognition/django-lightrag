"""
URL configuration for core.
"""

from django.urls import path
from ninja import NinjaAPI

from .views import router as lightrag_router

api = NinjaAPI(
    title="LightRAG API",
    version="1.0.0",
    description="LightRAG Django API for RAG operations",
    docs_url="/docs/",
)

api.add_router("/lightrag", lightrag_router)

urlpatterns = [
    path("api/", api.urls),
]
