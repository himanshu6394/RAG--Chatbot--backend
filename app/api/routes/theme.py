from fastapi import APIRouter
from fastapi.responses import JSONResponse
from app.services.vector_store import search_similar
from app.services.synthesis import summarize_themes

theme_router = APIRouter()

@theme_router.get("/")
def summarize_question(q: str):
    try:
        answers = search_similar(q)
        if isinstance(answers, dict):
            answers = [answers]
        summary = summarize_themes(answers)
        return {
            "question": q,
            "themes": summary
        }
    except Exception as ex:
        return JSONResponse(
            status_code=500,
            content={"error": f"An error occurred while summarizing the themes : {str(ex)}", }
        )
