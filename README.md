# Proyecto Tema X

## Descripción del Proyecto

Proyecto Tema X es una aplicación web diseñada para clasificar imágenes de alimentos utilizando un modelo de aprendizaje profundo. Este proyecto tiene como objetivo principal demostrar cómo integrar un modelo de inteligencia artificial en una aplicación web moderna, proporcionando una solución completa que incluye backend, frontend e infraestructura.

### Objetivos del Proyecto
- Facilitar la clasificación de imágenes de alimentos mediante un modelo de aprendizaje profundo.
- Proporcionar una interfaz de usuario intuitiva para interactuar con el sistema.
- Mostrar buenas prácticas en el desarrollo de aplicaciones web con soporte para contenedores y orquestación en Kubernetes.
- Servir como base para proyectos similares que requieran integración de IA en aplicaciones web.

### Casos de Uso
- **Educación**: Enseñar a estudiantes y desarrolladores cómo construir aplicaciones web con IA.
- **Prototipado**: Crear prototipos rápidos para soluciones basadas en clasificación de imágenes.
- **Demostraciones**: Mostrar capacidades de integración de IA en entornos empresariales o académicos.
- **Extensión**: Ampliar la funcionalidad para incluir más tipos de clasificación o análisis de imágenes.

### Características Principales
- **Backend**: Proporciona una API para la clasificación de imágenes.
- **Frontend**: Interfaz de usuario para cargar imágenes y visualizar resultados.
- **Infraestructura**: Configuración para Docker y Kubernetes.

---

## Prerrequisitos

Antes de comenzar, asegúrate de tener instalados los siguientes requisitos:

- **Docker**: Para construir y ejecutar contenedores.
- **Docker Compose**: Para orquestar los servicios localmente.
- **Kubernetes**: Para despliegues en un clúster.
- **kubectl**: Herramienta de línea de comandos para interactuar con Kubernetes.

---

## Instalación

### Clonar el Repositorio
```bash
git clone https://github.com/usuario/ProyectoTemaX.git
cd ProyectoTemaX
```

### Usando Docker Compose
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

### Usando Kubernetes

Puedes utilizar el archivo `makefile` para simplificar los comandos de Kubernetes. Los objetivos disponibles incluyen:

- **k8s-apply**: Aplica los manifiestos de Kubernetes.
- **k8s-wait**: Espera a que los despliegues estén listos.
- **pf-backend**: Reenvía el puerto del backend para pruebas locales.

Ejemplo de uso:
```bash
make run
```
Esto aplicará los manifiestos, esperará a que los despliegues estén listos y reenviará el puerto del backend.

Si prefieres usar los comandos manualmente:
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

---

## Estructura del Proyecto

- **backend/**: Contiene el código fuente del backend y el archivo `requirements.txt`.
- **frontend/**: Contiene el código fuente del frontend y los archivos de configuración de React.
- **k8s/**: Contiene los manifiestos de Kubernetes para el despliegue.
- **docker-compose.yaml**: Configuración para ejecutar los servicios con Docker Compose.

---

## Tecnologías Utilizadas

- **Backend**: Python, FastAPI, PyTorch.
- **Frontend**: React, TypeScript.
- **Infraestructura**: Docker, Kubernetes.

---