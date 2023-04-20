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