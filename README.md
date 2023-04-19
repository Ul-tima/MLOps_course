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


