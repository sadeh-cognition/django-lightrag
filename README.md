# LightRAG Django App

LightRAG as a reusable Django app, plus a small example project. It exposes a REST API (via `django-ninja`) and management commands for ingesting and querying documents with a graph + vector hybrid backend.

**Highlights**
- Django app: `djangorag.lightrag_app`
- REST API: `/api/lightrag/...` with interactive docs at `/api/docs/`
- Management commands for ingest/list/query
- Storage: LadybugDB for graph + ChromaDB for vectors

## Requirements
- Python `>=3.10`
- Django `4.2.x`
- ChromaDB and LadybugDB (`chromadb`, `real-ladybug`)
- An embeddings provider reachable at `LIGHTRAG_EMBEDDING_BASE_URL`

## Install
This repo is packaged with `pyproject.toml`.

```bash
pip install -e .
```

For dev tools:

```bash
pip install -e ".[dev]"
```

Notes:
- `embed-gen` is referenced as a local path dependency in `pyproject.toml`. If you do not have it at that path, update the dependency or install it separately.

## Configure
Set these in your Django settings (environment variables are supported for LightRAG):

```python
# settings.py
LIGHTRAG = {
    "EMBEDDING_PROVIDER": "LMStudio",
    "EMBEDDING_MODEL": "text-embedding-embeddinggemma-300m",
    "EMBEDDING_BASE_URL": "http://localhost:1234",
}

# Optional storage settings
CHROMADB_IN_MEMORY = False
CHROMADB_DIR = "chromadb_storage"  # required when not in-memory

LADYBUGDB = {
    "DATABASE_PATH": "ladybugdb.lbug",
    "IN_MEMORY": False,
}
```

## Wire Up URLs
The app ships with its own URL config. In your project `urls.py`, include it:

```python
from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("djangorag.lightrag_app.urls")),
]
```

This provides:
- API base: `http://localhost:8001/api/lightrag/`
- Docs: `http://localhost:8001/api/docs/`

## Database
Run migrations before first use:

```bash
python manage.py migrate
```

## Run
```bash
python manage.py runserver 8001
```

## API Usage
All endpoints are namespaced under `/api/lightrag`.

- `POST /api/lightrag/documents/ingest`
- `GET /api/lightrag/documents`
- `POST /api/lightrag/query`
- `DELETE /api/lightrag/documents/{document_id}`
- `GET /api/lightrag/entities`
- `GET /api/lightrag/relations`
- `GET /api/lightrag/health`

Example ingest:

```bash
curl -X POST http://localhost:8001/api/lightrag/documents/ingest \
  -H 'Content-Type: application/json' \
  -d '{"title": "LightRAG Intro", "content": "LightRAG blends knowledge graphs with vector search.", "metadata": {"source": "readme"}}'
```

Example query:

```bash
curl -X POST http://localhost:8001/api/lightrag/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "What is LightRAG?", "param": {"mode": "hybrid", "top_k": 5}}'
```

## Management Commands
```bash
python manage.py ingest_document --content "LightRAG is a retrieval-augmented generation framework." --title "Intro"
python manage.py list_documents --format table
python manage.py query_rag "What is LightRAG?" --mode hybrid --top-k 5 --include-sources
```

## Troubleshooting
- If `chromadb` is not installed, install it or set `CHROMADB_IN_MEMORY=True` and ensure a compatible ChromaDB version.
- If `real_ladybug` is missing, install `real-ladybug` and verify LadybugDB can create its file.
- If embeddings fail, confirm `LIGHTRAG_EMBEDDING_BASE_URL` points to a running embeddings server.
