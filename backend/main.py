from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from PIL import Image
import torch
from typing import List, Dict, Any

from ProyectoModelo.model.model2 import load_model, build_transforms, get_device

app = FastAPI(
    title="Food Classification API",
    description="Clasificación de imágenes de alimentos usando un modelo preentrenado.",
    version="1.0",
)

# CORS (para frontend en otro contenedor / puerto)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, reemplazar por el dominio del frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = Path("./artifacts/food_classifier.pth")
CLASSES_PATH = Path("./artifacts/classes.json")

device = get_device()
try:
    model, classes = load_model(MODEL_PATH, CLASSES_PATH, device)
    _, eval_transform = build_transforms()
except Exception as e:
    raise RuntimeError(f"Error al cargar el modelo: {e}")


def _predict_pil_image(image: Image.Image) -> Dict[str, Any]:
    tensor = eval_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)

    predicted_class = classes[pred.item()]
    return {"predicted_class": predicted_class, "confidence": round(confidence.item(), 4)}


@app.get("/", summary="Verifica el estado de la API")
def root():
    return {"message": "API de clasificación de alimentos está activa."}


@app.post("/predict", summary="Realiza una predicción de la clase de alimento")
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo subido no es una imagen válida.")
    try:
        image = Image.open(file.file).convert("RGB")
        return _predict_pil_image(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar la imagen: {e}")


@app.post("/multipredict", summary="Realiza predicción de múltiples imágenes")
async def multipredict(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="Debes enviar al menos un archivo.")

    results = []
    errors = []

    for idx, file in enumerate(files):
        filename = file.filename or f"file_{idx}"

        if not file.content_type or not file.content_type.startswith("image/"):
            errors.append({"filename": filename, "error": "El archivo subido no es una imagen válida."})
            continue

        try:
            image = Image.open(file.file).convert("RGB")
            pred = _predict_pil_image(image)
            results.append({"filename": filename, **pred})
        except Exception as e:
            errors.append({"filename": filename, "error": f"Error al procesar la imagen: {e}"})

    return {
        "count": len(files),
        "success": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors,
    }