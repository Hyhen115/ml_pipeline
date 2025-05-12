# ML Pipeline System: A Distributed Machine Learning Pipeline on AWS EMR
## Overview
This project implements a complete ML pipeline system that allow users to upload datasets, select features, train models using different algorithms, and download the best model through a web interface.

## System Architecture
![cloudProj drawio](https://github.com/user-attachments/assets/4cc81494-97f4-4668-905d-45356ae5309a)
### Components
### Flow
- User upload data set via UI
- Frontend send csv to S3 using presigned URL
- Backend process data and train model using PySpark on EMR Clusters
- The Trained model is saved to S3
- User can download the trained model via UI

## Getting Started
### Prerequisites
- nodejs
- AWS account
- Docker
- Git for cloning the repo

### Deploy
#### Option 1: using our setup for the docker container, and your s3, emr cluster
- Create an EMR Cluster with spark and hadoop environment(3 worker node recommended)
- Set Up S3 Bucket on the same AWS location as the EMR Cluster (This Project did not support for cross location communication within AWS services)
> go to s3 bucket permissions -> Cross-origin resource sharing (CORS) and paste the code below
```
[
    {
        "AllowedHeaders": [
            "*"
        ],
        "AllowedMethods": [
            "POST",
            "PUT",
            "GET"
        ],
        "AllowedOrigins": [
            "*"
        ],
        "ExposeHeaders": []
    }
]
```
- Deploy Backend to EMR Master Node
  
  Connect to EMR master node:
  
  ```ssh -i your-key.pem hadoop@your-emr-master-dns```
  > make sure your EMR master node SSH port is opened for connection

  Install and configure Docker on EMR master node:

  ```
  sudo yum install -y docker
  sudo systemctl start docker
  ```

  Pull the backend container:
  
  ```docker pull hyhen/ml-pipeline-backend:emr```

  Run the backend container:
  ```
  docker run -d -p 5001:5001 \
    -e AWS_ACCESS_KEY_ID=your_access_key \
    -e AWS_SECRET_ACCESS_KEY=your_secret_key \
    -e AWS_SESSION_TOKEN=your_session_token \
    -e S3_BUCKET=your-ml-pipeline-bucket \
    -e AWS_REGION=your-s3-bucket-and-emr-cluster-aws-region \
    --name ml-pipeline-backend \
    hyhen/ml-pipeline-backend:emr
  ```
  > ```-e AWS_SESSION_TOKEN=your_session_token \``` is optional for temporary credentials
  > make sure your TCP port 5001 is opened for access for the Flask RESTful API

- Depoloy Frontend
  > the front end can be deployed on any web servers or hosting services
  
  Before deployment Please update the API endpoints in ```MLPipelinePage.jsx```

  change all ip address of the fetch ip to the Public IPv4 of the EMR master node
  ```const res = await fetch('http://your-emr-master-node-public-ipv4:5001/generate_upload_url' ```

#### Option 2: Deploy on Single EC2 Instance or Local (Testing)
you can run both components on Single EC2 instance or on your local machine

Local build and push your container

```
cd pipeline_backend

# Build and push for x86_64 (EC2/EMR compatibility)
docker buildx build --platform linux/amd64 -t your-repo/ml-pipeline-backend --push .
```

EC2 or local you want to run docker:

```
# download docker on EC2 ubuntu instance

sudo apt-get update

sudo apt-get install docker.io -y

sudo systemctl enable docker
```

```
# Run your custom image
docker run -d -p 5001:5001 \
  -e AWS_ACCESS_KEY_ID=your_access_key \
  -e AWS_SECRET_ACCESS_KEY=your_secret_key \
  -e AWS_SESSION_TOKEN=your_session_token \
  -e S3_BUCKET=your-ml-pipeline-bucket \
  -e AWS_REGION=your-s3-bucket-and-emr-cluster-aws-region \
  --name ml-pipeline-backend \
  your-repo/ml-pipeline-backend
```

### API Endpoints
| Endpoint                   | Method | Description                                    |
|----------------------------|--------|------------------------------------------------|
| ```/health```              | GET    | check server health                            |
| ```/generate_upload_url``` | POST   | generate presigned s3 url for user csv uploads |
| ```/start_pipeline```      | POST   | start the ml pipeline                          |
| ```/download_model_zip```  | GET    | download the trained model as zip              |

### Environment Variables
| Endpoint                    | Description                       | Default                           |
|-----------------------------|-----------------------------------|-----------------------------------|
| ```AWS_ACCESS_KEY_ID```     | aws access key id                 | Required                          |
| ```AWS_SECRET_ACCESS_KEY``` | the secret key for the access key | Required                          |
| ```AWS_SESSION_TOKEN```     | session token for temp sessions   | Optional                          |
| ```S3_BUCKET```             | s3 bucket name                    | ```ml-pipeline-data-test-hyhen``` |
| ```AWS_REGION```            | s3 and emr region name            | ```us-east-1```                   |


  
