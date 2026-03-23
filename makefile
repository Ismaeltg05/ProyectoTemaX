# Makefile para gestionar el despliegue y la ejecución del proyecto en Kubernetes

# Nombre del namespace utilizado en Kubernetes
NAMESPACE=proyectotemax

# Aplica los manifiestos de Kubernetes en el clúster
k8s-apply:
	kubectl apply -f k8s/

# Espera a que los despliegues de backend y frontend estén listos
k8s-wait:
	kubectl -n $(NAMESPACE) rollout status deploy/backend --timeout=180s
	kubectl -n $(NAMESPACE) rollout status deploy/frontend --timeout=180s

# Reenvía el puerto 8000 del servicio backend al puerto local 8000
pf-backend:
	kubectl -n $(NAMESPACE) port-forward svc/backend 8000:8000

# Ejecuta las tareas de aplicar manifiestos, esperar despliegues y reenviar puertos
run: k8s-apply k8s-wait pf-backend