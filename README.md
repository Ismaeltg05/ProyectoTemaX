# Proyecto Tema X

## Descripción del Proyecto

Proyecto Tema X es una aplicación web diseñada para clasificar imágenes de alimentos utilizando un modelo de aprendizaje profundo. El sistema incluye un backend desarrollado en Python con FastAPI y un frontend construido con React y TypeScript. La infraestructura está preparada para ejecutarse tanto en contenedores Docker como en un clúster de Kubernetes, lo que facilita su despliegue y escalabilidad.

### Características Principales
- **Backend**: Proporciona una API para la clasificación de imágenes.
- **Frontend**: Interfaz de usuario para cargar imágenes y visualizar resultados.
- **Infraestructura**: Configuración para Docker y Kubernetes.

### Componentes Principales

1. **Backend**:
   - Desarrollado con FastAPI.
   - Proporciona una API para la clasificación de imágenes.
   - Utiliza un modelo de aprendizaje profundo para realizar predicciones.

2. **Frontend**:
   - Construido con React y TypeScript.
   - Proporciona una interfaz de usuario para cargar imágenes y visualizar los resultados de la clasificación.

3. **Infraestructura**:
   - Configuración de Docker Compose para desarrollo local.
   - Manifiestos de Kubernetes para despliegues en un clúster.

### Cómo Ejecutar el Proyecto

#### Usando Docker Compose
1. Construir las imágenes:
   ```bash
   docker-compose build
   ```
2. Levantar los servicios:
   ```bash
   docker-compose up
   ```
3. Acceder a la aplicación:
   - Backend: [http://localhost:8000](http://localhost:8000)
   - Frontend: [http://localhost:8080](http://localhost:8080)

#### Usando Kubernetes
1. Aplicar los manifiestos:
   ```bash
   kubectl apply -f k8s/
   ```
2. Verificar los pods y servicios:
   ```bash
   kubectl get pods -n proyectotemax
   kubectl get svc -n proyectotemax
   ```
3. Acceder a la aplicación:
   - Backend: Expuesto en el servicio `backend`.
   - Frontend: Expuesto en el servicio `frontend`.

### Estructura del Proyecto

- **backend/**: Contiene el código fuente del backend y el archivo `requirements.txt`.
- **frontend/**: Contiene el código fuente del frontend y los archivos de configuración de React.
- **k8s/**: Contiene los manifiestos de Kubernetes para el despliegue.
- **docker-compose.yaml**: Configuración para ejecutar los servicios con Docker Compose.

### Tecnologías Utilizadas
- **Backend**: Python, FastAPI, PyTorch.
- **Frontend**: React, TypeScript.
- **Infraestructura**: Docker, Kubernetes.

---

Este proyecto es un ejemplo de cómo integrar un modelo de aprendizaje profundo en una aplicación web moderna con soporte para contenedores y orquestación en Kubernetes.