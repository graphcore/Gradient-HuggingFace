#!/bin/bash 
projectId=$1
deploymentName=$2
psToken=$3

pip install -U gradient > /dev/null
gradient apiKey $psToken
gradient secrets set project \
  --id $projectId\
  --name MyApiToken \
  --value $psToken
gradient models upload --name $deploymentName --modelType "custom" --projectId $projectId "./deployment"
modelId=$(gradient models list|grep $deploymentName|awk '{print $4}'|head -n 1|tr -d '\n')
echo "Deploying model ID: $modelId"

spec="enabled: true\n
image: gc0alexandrec/deployment-test:latest\n
containerRegistry: alexc-personal\n
port: 8100\n
env:\n
    - name: SSF_OPTIONS\n
      value: '--config gradient-model:$modelId|config.yml --deployment-api-key API_TOKEN clean build run'\n
    - name: API_TOKEN\n
      value: secret:MyApiToken\n
resources:\n
  replicas: 1\n
  instanceType: IPU-POD4\n

"
echo "$spec" > spec.yml
gradient deployments create --name $deploymentName --projectId $projectId --spec spec.yml --clusterId clehbtvty
deploymentId=$(gradient deployments list|grep $deploymentName|awk '{print $4}'|tr -d '\n')
project_url="https://console.paperspace.com/graphcorepaperspace/projects/$projectId/gradient-deployments/$deploymentId/overview"
gradient deployments get --id $deploymentId > deployment_info.json
endpoint=$(jq .deploymentSpecs[0].endpointUrl deployment_info.json | tr -d '"')
echo "Deployment overview URL: $project_url"
echo "Waiting server to be ready"
while true; do
  STATUS=$(curl -L -s -o /dev/null -w "%{http_code}" https://$endpoint/docs)
  if [ "$STATUS" -eq 200 ]
  then
    echo " READY."
    echo "API : https://$endpoint/docs"
    exit 0
  fi
  sleep 10
  echo -n "."
done