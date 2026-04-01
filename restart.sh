#!/bin/bash
echo "Reiniciando todos os deployments..."

kubectl rollout restart deployment/fl-client deployment/fl-server deployment/mosquitto

echo "Aguardando pods subirem..."
kubectl rollout status deployment/mosquitto
kubectl rollout status deployment/fl-server
kubectl rollout status deployment/fl-client

echo "Tudo pronto!"