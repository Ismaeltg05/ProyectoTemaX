from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from PIL import Image
import torch
from typing import List, Dict, Any, Optional, Tuple
import json

# Torchvision solo se usa para transforms/model base (si no lo tienes, añádelo a requirements)
from torchvision import transforms, models


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


def get_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def build_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, eval_transform


def load_model(
    model_path: Path, classes_path: Path, device: torch.device
) -> Tuple[torch.nn.Module, List[str]]:
    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo en {model_path.resolve()}")
    if not classes_path.exists():
        raise FileNotFoundError(f"No se encontró classes.json en {classes_path.resolve()}")

    classes = json.loads(classes_path.read_text(encoding="utf-8"))
    if not isinstance(classes, list) or not classes:
        raise ValueError("classes.json debe ser una lista no vacía de nombres de clases")

    # Modelo base (ajústalo si tu .pth corresponde a otro backbone)
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))

    # Carga del state_dict (soporta pth con state_dict directo o dict con clave 'state_dict')
    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    # Limpia prefijos típicos ("module.") si viene de DataParallel
    cleaned = {}
    for k, v in state.items():
        cleaned[k.replace("module.", "")] = v

    model.load_state_dict(cleaned, strict=False)
    model.to(device)
    model.eval()
    return model, classes


device = get_device()
_, eval_transform = build_transforms()

model: Optional[torch.nn.Module] = None
classes: Optional[List[str]] = None
model_load_error: Optional[str] = None

try:
    model, classes = load_model(MODEL_PATH, CLASSES_PATH, device)
except Exception as e:
    # NO tumbes el servidor: deja el API arriba y reporta el error en /health
    model_load_error = str(e)


def _ensure_model_ready() -> None:
    if model is None or classes is None:
        detail = {
            "message": "Modelo no disponible en este despliegue.",
            "error": model_load_error,
            "expected_files": [str(MODEL_PATH), str(CLASSES_PATH)],
        }
        raise HTTPException(status_code=503, detail=detail)


def _predict_pil_image(image: Image.Image) -> Dict[str, Any]:
    _ensure_model_ready()

    assert model is not None
    assert classes is not None

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


@app.get("/health", summary="Estado del servicio y del modelo")
def health():
    return {
        "status": "ok",
        "device": str(device),
        "model_loaded": model is not None and classes is not None,
        "model_path": str(MODEL_PATH),
        "classes_path": str(CLASSES_PATH),
        "error": model_load_error,
    }


@app.post("/predict", summary="Realiza una predicción de la clase de alimento")
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo subido no es una imagen válida.")
    try:
        image = Image.open(file.file).convert("RGB")
        return _predict_pil_image(image)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar la imagen: {e}")


@app.post("/multipredict", summary="Realiza predicción de múltiples imágenes")
async def multipredict(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="Debes enviar al menos un archivo.")

    # Si no hay modelo, falla una sola vez con 503 (más claro que fallar por archivo)
    _ensure_model_ready()

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