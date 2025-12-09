from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.ocr_extract import ocr_and_extract

router = APIRouter()

@router.post("/upload_and_analyze")
async def upload_and_analyze(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {e}")

    try:
        result = ocr_and_extract(file_bytes, file.filename)
        return result
    except Exception as e:
        return {
            "file": file.filename,
            "error": "Pipeline failed unexpectedly",
            "details": repr(e),
            "status": "failed",
        }
