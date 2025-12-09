# app/api/camera_predict_api.py
import os
import io
import json
import logging
from typing import Optional, Tuple, Any

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

logger = logging.getLogger("camera_predict_api")
logger.addHandler(logging.NullHandler())

# Optional OCR helpers (from app.ml.ocr_utils)
try:
    from app.ml.ocr_utils import image_bytes_to_ocr_text, extract_tests_from_text, preprocess_for_ocr
except Exception:
    image_bytes_to_ocr_text = None
    extract_tests_from_text = None
    preprocess_for_ocr = None

MODEL_DIR = os.getenv("MODEL_DIR", "app/ml/hf_out")
predictor = None  # lazy-loaded

app = FastAPI(title="Camera -> Lab -> Disease Predictor")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PRELOAD_PREDICTOR = os.getenv("PRELOAD_PREDICTOR", "0").lower() not in ("0", "false", "no")


async def _lazy_load_predictor():
    global predictor
    if predictor is None:
        # Import lazily to avoid heavy import at startup if not needed
        from app.ml.predict_hf import Predictor  # type: ignore
        predictor = Predictor(model_dir=MODEL_DIR)
    return predictor


if PRELOAD_PREDICTOR:
    try:
        import asyncio

        loop = asyncio.get_event_loop()
        loop.run_until_complete(_lazy_load_predictor())
        logger.info("Predictor preloaded")
    except Exception:
        logger.exception("Predictor preload failed; will lazy-load on first request")


def _local_pytesseract_ocr(image_bytes: bytes) -> Tuple[str, Optional[str]]:
    """
    Simple pytesseract fallback: use preprocess_for_ocr if available, then run pytesseract.
    Returns tuple (text, optional_error_string)
    """
    try:
        from PIL import Image, ImageOps
        import pytesseract
    except Exception as e:
        return "", f"pillow_or_tesseract_import_error: {e}"

    pil = None
    try:
        if preprocess_for_ocr is not None:
            try:
                proc = preprocess_for_ocr(image_bytes)
                # preprocess_for_ocr may return a numpy array (grayscale) or BGR ndarray
                import numpy as _np
                if isinstance(proc, _np.ndarray):
                    if proc.ndim == 2:
                        pil = Image.fromarray(proc).convert("RGB")
                    elif proc.ndim == 3 and proc.shape[2] == 3:
                        import cv2 as _cv  # local import
                        pil = Image.fromarray(_cv.cvtColor(proc, _cv.COLOR_BGR2RGB))
                elif isinstance(proc, Image.Image):
                    pil = proc.convert("RGB")
            except Exception:
                pil = None
        if pil is None:
            pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # slight autocontrast to improve recognition of low-contrast photos
        pil = ImageOps.autocontrast(pil, cutoff=1)

        # numeric-first quick attempt (good for lab values)
        try:
            cfg_num = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.%/—,-'
            txt = pytesseract.image_to_string(pil, config=cfg_num)
            if txt and any(ch.isdigit() for ch in txt):
                return txt, None
        except Exception:
            pass

        # generic full attempt
        txt = pytesseract.image_to_string(pil)
        return txt or "", None
    except Exception as e:
        return "", f"local_pytesseract_error: {e}"


def _normalize_label_for_kb(label: Optional[str]) -> str:
    if not label:
        return ""
    l = label.strip().lower()
    return l.replace(" ", "_").replace("-", "_")


@app.post("/api/predict_camera")
async def predict_camera(
    image: UploadFile = File(...),
    topk: int = Form(5),
    use_ocr_pipeline: Optional[bool] = Form(True),
) -> JSONResponse:
    """
    Main entry: accepts an uploaded image, runs OCR pipeline (best-effort),
    extracts tests, calls predictor and returns structured response.
    """
    try:
        contents = await image.read()
        if not contents:
            return JSONResponse({"error": "Empty file upload"}, status_code=400)

        ocr_text = ""
        tests_list = None

        # 1) Try full OCR->extract pipeline if available
        if use_ocr_pipeline:
            try:
                from app.services.ocr_extract import ocr_and_extract  # type: ignore
                ocr_result = ocr_and_extract(contents, filename=(image.filename or "upload"))
                # ocr_and_extract returns a dict with 'raw_text_sample' (short) and possibly 'summary'
                ocr_text = ocr_result.get("raw_text_sample", "") or ocr_result.get("raw_text", "") or ""
                # If ocr_result included parsed tests already (some implementations might), preserve them
                if not tests_list:
                    # try to read 'summary' -> list of parsed items (compat)
                    parsed = ocr_result.get("parsed_tests") or ocr_result.get("lab_results") or None
                    if isinstance(parsed, list) and parsed:
                        tests_list = parsed
                # try structured extraction from raw_text
                if extract_tests_from_text and ocr_text:
                    try:
                        tests_list = extract_tests_from_text(ocr_text)
                    except Exception:
                        tests_list = tests_list or None
            except Exception:
                # fallback to other OCR methods below
                logger.exception("ocr_and_extract failed (fallback enabled)")
                ocr_text = ""
                tests_list = None

        # 2) Fallback: use image_bytes_to_ocr_text helper (if installed)
        if not ocr_text:
            if image_bytes_to_ocr_text is not None:
                try:
                    # allow preprocess; this helper typically handles CLAHE/upscale etc.
                    ocr_text = image_bytes_to_ocr_text(contents, preprocess=True)
                except Exception:
                    ocr_text, err = _local_pytesseract_ocr(contents)
                    if err:
                        logger.debug("local pytesseract error: %s", err)
            else:
                ocr_text, err = _local_pytesseract_ocr(contents)
                if err:
                    logger.debug("local pytesseract error: %s", err)

        # 3) If tests_list still None, try structured extractor on OCR text
        if tests_list is None and extract_tests_from_text and ocr_text:
            try:
                tests_list = extract_tests_from_text(ocr_text)
            except Exception:
                tests_list = None

        # 4) If still None, try heavier HF utils parse (if exists)
        if not tests_list:
            try:
                from app.ml.hf_utils import parse_tests_json as _parse_json  # type: ignore
                parsed = _parse_json(ocr_text)
                if isinstance(parsed, str):
                    try:
                        parsed_obj = json.loads(parsed)
                        tests_list = parsed_obj if isinstance(parsed_obj, list) else [parsed_obj]
                    except Exception:
                        tests_list = None
                elif isinstance(parsed, list):
                    tests_list = parsed
                elif isinstance(parsed, dict):
                    tests_list = [parsed]
            except Exception:
                tests_list = None

        # 5) Last-resort: return raw OCR as a single test entry
        if not tests_list:
            tests_list = [{"test_name": "ocr_raw", "value": ocr_text}]

        tests_json = json.dumps(tests_list)

        # 6) Load predictor and predict
        predictor_obj = await _lazy_load_predictor()
        preds = predictor_obj.predict_from_tests_json(tests_json, topk=topk)

        # --- DEBUG: log predictor output and tests_list for troubleshooting ---
        try:
            logger.info("DEBUG: predictor preds type=%s", type(preds))
            # be conservative with size in logs
            logger.info("DEBUG: predictor preds sample=%s", str(preds)[:200])
        except Exception:
            pass
        try:
            logger.info("DEBUG: tests_list sample=%s", str(tests_list)[:400])
        except Exception:
            pass
        try:
            logger.info("DEBUG: ocr_text snippet=%s", (ocr_text or "")[:400].replace("\n", " "))
        except Exception:
            pass

        # 7) Build lightweight ML explanations (best-effort)
        ml_explanations = []
        try:
            from app.ml.disease_analyzer import DISEASES  # type: ignore
            top_items = []
            if isinstance(preds, dict):
                if "top_10" in preds:
                    top_items = preds.get("top_10", [])[:topk]
                elif "probabilities" in preds and isinstance(preds["probabilities"], dict):
                    sorted_items = sorted(preds["probabilities"].items(), key=lambda x: x[1], reverse=True)[:topk]
                    top_items = [(k, v) for k, v in sorted_items]
                else:
                    try:
                        sorted_items = sorted([(k, v) for k, v in preds.items()], key=lambda x: x[1], reverse=True)[:topk]
                        top_items = [(k, v) for k, v in sorted_items]
                    except Exception:
                        top_items = []
            elif isinstance(preds, list):
                top_items = preds[:topk]

            # robust handling for various item formats
            for it in top_items:
                label = None
                score = None
                if isinstance(it, (list, tuple)) and len(it) >= 2:
                    label, score = it[0], float(it[1])
                elif isinstance(it, dict):
                    # dict can be {"label":..,"score":..} or {"anemia":0.99}
                    if "label" in it:
                        label = it.get("label"); score = float(it.get("score") or 0.0)
                    else:
                        items = list(it.items())
                        if items:
                            label, score = items[0][0], float(items[0][1])
                else:
                    # fallback: item may be string label
                    label = str(it)
                    score = None

                # safe-normalize & KB lookup
                norm = _normalize_label_for_kb(str(label))
                kb = DISEASES.get(norm) or DISEASES.get(str(label).lower()) or {}
                ml_explanations.append({
                    "label": label,
                    "score": score,
                    "what": kb.get("what"),
                    "why": kb.get("why"),
                    "how": kb.get("how"),
                })
        except Exception:
            logger.exception("ml_explanations building failed")
            ml_explanations = []

        # 8) Rule-based detections (best-effort) — include what/why/how from KB
        rule_detections = []
        try:
            from app.ml.disease_analyzer import detect_disease, DISEASES  # type: ignore

            # run rule engine
            rule_detections_raw = detect_disease(ocr_text, tests_list)

            # normalize / enrich each detection with KB fields (what, why, how)
            rule_detections = []
            for d in rule_detections_raw:
                disease_name = (d.get("disease") or "").strip()
                key = disease_name.lower().replace(" ", "_")
                kb = DISEASES.get(key, {}) if isinstance(DISEASES, dict) else {}

                rule_detections.append({
                    "disease": disease_name or d.get("disease"),
                    "status": d.get("status", ""),
                    "what": d.get("what") or kb.get("what"),
                    "why": d.get("why") or kb.get("why"),
                    "how": d.get("how") or kb.get("how"),
                })
        except Exception:
            logger.exception("rule-based detection failed")
            rule_detections = []

        response_body = {
            "predictions_raw": preds,
            "ml_explanations": ml_explanations,
            "rule_detections": rule_detections,
            "ocr_text": ocr_text,
            "tests_json": tests_json,
        }
        return JSONResponse(response_body)

    except Exception as e:
        logger.exception("predict_camera top-level error")
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run("app.api.camera_predict_api:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
