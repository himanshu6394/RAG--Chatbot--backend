from fastapi import FastAPI, UploadFile, File, HTTPException
from .api.routes.upload import upload_router
from .api.routes.query import query_router
from .api.routes.theme import theme_router
from .api.routes.debug import debug_router

app = FastAPI(title="Document Chatbot")
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Welcome to Document Chatbot API"}

app.include_router(upload_router, prefix="/upload")
app.include_router(query_router, prefix="/query")
app.include_router(theme_router, prefix="/theme")
app.include_router(debug_router, prefix="/debug")