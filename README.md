# MLOps_course

**Docker**

Web:

docker build --tag web:latest ./web

docker run -it --rm -p 8000:80 web:latest


ML:

docker build --tag ml:latest ./ml

docker run -it --rm ml:latest


**Docker Hub**

docker login -u $DOCKER_HUB_USER -p $DOCKER_HUB_PASS


Web:

docker tag web:latest jpikovets/web:latest

docker push jpikovets/web:latest

ML:

docker tag ml:latest jpikovets/ml:latest

docker push jpikovets/ml:latest




**Kubernetes/Kind**

Create cluster

kind create cluster --name firstcluster

Create pod

kubectl create -f kub/pod_ml.yaml

kubectl create -f kub/pod_web.yaml

Create deployment

kubectl create -f kub/dep_web.yaml

Create job

kubectl create -f kub/job_ml.yaml

Check logs

kubectl logs jobs/job-ml

Run Kubernetes CLI
k9s -A


***WEEK 2***


**MINIO**

**Docker**

docker-compose -f ./minio/docker-compose.yml up -d

docker-compose ps

Set alias for minio

mc alias set ALIAS HOSTNAME ACCESS_KEY SECRET_KEY
mc alias set s3 http://0.0.0.0:9000 minio_user minio_password


**DVC**


Init DVC

dvc init --subdir


Create remote directory

mc mb s3/dataset

dvc remote add -d minio s3://dataset

dvc remote modify minio endpointurl http://0.0.0.0:9000

dvc remote modify minio access_key_id minio_user

dvc remote modify minio secret_access_key minio_password


Add folder to dvc

dvc add ./dataset/crema

git add dataset/.gitignore dataset/crema.dvc


Save code to git

git add .dvc/config
git commit -m "Configure remote storage"
git push



**Label studio**

Install locally with Docker
Run Label Studio in a Docker container and access it at http://localhost:8080.

docker pull heartexlabs/label-studio:latest

docker run -it -p 8080:8080 -v $(pwd)/mydata:/label-studio/data heartexlabs/label-studio:latest



***WEEK 3***

**WANDB**

export PYTHONPATH=$(pwd)

wandb login


Run hyperparameter tuning

wandb sweep ml/sweep.yaml

wandb agent SWEEP_ID




**Streamlit**
```
PYTHONPATH=`pwd` streamlit run app/streamlit/streamlit_ui.py

docker build . --file=app/streamlit/Dockerfile -t=streamlit:v1
docker tag streamlit:v1 jpikovets/streamlit:latest

docker push jpikovets/streamlit:latest

kind create cluster --name fast
kubectl create -f kub/app-streamlit.yaml
kubectl apply -f kub/app-streamlit.yaml

kubectl port-forward --address 0.0.0.0 svc/app-streamlit 8080:8080
```


**FastAPI**
```
docker build . --file=web/Dockerfile -t=web:v1
docker build --no-cache . --file=web/Dockerfile -t=web:v1
docker tag web:v1 jpikovets/web:v1

docker push jpikovets/web:v1

kind create cluster --name fast
kubectl create -f kub/app-fastapi.yaml
kubectl apply -f kub/app-fastapi.yaml
```
