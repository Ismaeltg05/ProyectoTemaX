# ProyectoTemaX

## Descripción general

ProyectoTemaX es una aplicación que utiliza un modelo de inteligencia artificial para clasificar imágenes de alimentos. El proyecto incluye:
- **Backend**: Implementado con FastAPI, sirve el modelo de clasificación.
- **Frontend**: Una interfaz web para cargar imágenes y mostrar resultados.
- **Despliegue**: Configuración para ejecutar los servicios en Kubernetes.

---

## Ejecución local

### 1) Backend

1. Instalar dependencias:

```bash
cd backend
pip install -r requirements.txt
```

2. Ejecutar el servidor:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

El backend estará disponible en `http://localhost:8000`.

### 2) Frontend

1. Instalar dependencias:

```bash
cd frontend
npm install
```

2. Ejecutar el servidor de desarrollo:

```bash
npm run dev
```

El frontend estará disponible en `http://localhost:5173`.

---

## Despliegue en Kubernetes

### 1) Crear el namespace

```bash
kubectl apply -f k8s/namespace.yaml
```

### 2) Desplegar el backend y el frontend

```bash
kubectl apply -f k8s/backend-deployment.yaml
kubectl apply -f k8s/backend-service.yaml
kubectl apply -f k8s/frontend-deployment.yaml
kubectl apply -f k8s/frontend-service.yaml
```

### 3) Verificar los pods

```bash
kubectl get pods -n proyectotemax
```

### 4) Acceder a los servicios

- **Backend**: Usar el puerto expuesto en el servicio de Kubernetes.
- **Frontend**: Acceder a través del servicio configurado (NodePort o LoadBalancer).

---

## Modelo IA: clasificación de alimentos

Se implementó un modelo en `ProyectoModelo/model/model.py` para clasificar imágenes por categorías de alimentos usando PyTorch y transferencia de aprendizaje (`ResNet18`).

### 1) Estructura esperada del dataset

```text
dataset/
	train/
		pizza/
		sushi/
		ensalada/
	val/
		pizza/
		sushi/
		ensalada/
	test/
		pizza/
		sushi/
		ensalada/
```

### 2) Instalar dependencias

```bash
pip install torch torchvision pillow kaggle
```

### 3) Descargar dataset de Kaggle y prepararlo automáticamente

Dataset recomendado: `kmader/food41`

1. Crea tu token en Kaggle: `Account` -> `Create New API Token`
2. Guarda `kaggle.json` en `%USERPROFILE%/.kaggle/kaggle.json`

Comando completo (descarga + extracción + split a train/val/test):

```bash
python ProyectoModelo/model/prepare_dataset.py --download-kaggle --kaggle-dataset kmader/food41 --raw-dir ./dataset_raw --output-dir ./dataset --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15
```

Alternativa sin token de Kaggle (descarga desde internet con torchvision):

```bash
python ProyectoModelo/model/prepare_dataset.py --download-food101 --raw-dir ./dataset_raw --output-dir ./dataset --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15
```

Si ya tienes imágenes locales organizadas por clase, usa:

```bash
python ProyectoModelo/model/prepare_dataset.py --source-images-root ./mis_imagenes_por_clase --output-dir ./dataset
```

### 4) Entrenar

```bash
python ProyectoModelo/model/model.py train --data-dir ./dataset --output-dir ./artifacts --epochs 10 --batch-size 32 --lr 0.001
```

Genera:
- `artifacts/food_classifier.pth`
- `artifacts/classes.json`

### 5) Predecir una imagen

```bash
python ProyectoModelo/model/model.py predict --model-path ./artifacts/food_classifier.pth --classes-path ./artifacts/classes.json --image-path ./mi_imagen.jpg
```