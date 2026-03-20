# Comandos mínimos para arrancar Kubernetes + desplegar + port-forward del backend

> Asume que ya tienes un clúster Kubernetes disponible (por ejemplo Docker Desktop con Kubernetes habilitado) y `kubectl` instalado.

## 0) Comprobar que Kubernetes está listo
```bash
kubectl cluster-info
kubectl get nodes
```

## 1) Desplegar namespace + backend + frontend
```bash
kubectl apply -f k8s/namespace.yaml

kubectl apply -f k8s/backend-deployment.yaml
kubectl apply -f k8s/backend-service.yaml

kubectl apply -f k8s/frontend-deployment.yaml
kubectl apply -f k8s/frontend-service.yaml 2>/dev/null || true
```

## 2) Esperar a que levanten (opcional, recomendado)
```bash
kubectl -n proyectotemax rollout status deploy/backend --timeout=180s
kubectl -n proyectotemax rollout status deploy/frontend --timeout=180s
```

## 3) Port-forward del backend (para probar la API)
Terminal A:
```bash
kubectl -n proyectotemax port-forward svc/backend 8000:8000
```

Terminal B (prueba rápida):
```bash
curl -i http://localhost:8000/
# si existe en tu imagen:
curl -i http://localhost:8000/health
```