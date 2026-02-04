# Federeted Learning with Tenforflow

## Set up images docker

```bash
docker build -t fl-client:latest ./client
docker build -t fl-server:latest ./server
```

## Set up docker images in cluster 

```bash
kubectl apply -f k8s/client.yaml
kubectl apply -f k8s/server.yaml
kubectl apply -f mqtt/broker.yaml
```

verify if images are running

```bash
kubectl get pods
```

## Verify logs

```bash
kubectl logs client/fl-client
kubectl logs server/fl-server
```

## If any changes are made in client or server

```bash
kubectl rollout restart deployment fl-client
kubectl rollout restart deployment fl-server
```

