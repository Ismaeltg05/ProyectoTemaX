NAMESPACE=proyectotemax

k8s-apply:
	kubectl apply -f k8s/

k8s-wait:
	kubectl -n $(NAMESPACE) rollout status deploy/backend --timeout=180s
	kubectl -n $(NAMESPACE) rollout status deploy/frontend --timeout=180s

pf-backend:
	kubectl -n $(NAMESPACE) port-forward svc/backend 8000:8000

run: k8s-apply k8s-wait pf-backend