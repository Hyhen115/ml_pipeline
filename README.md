# ML Pipeline System: A Distributed Machine Learning Pipeline on AWS EMR
## Overview
This project implements a complete ML pipeline system that allow users to upload datasets, select features, train models using different algorithms, and download the best model through a web interface.

[5-Page Submission Report](https://docs.google.com/document/d/14tjbfdkozPfu_rSfA9AxpVMGKPNDfFNmtr__Eu7IJFA/edit?usp=sharing)

[Demo Video](https://youtu.be/KZsfy7v6hPE?si=53Mx7zfd8nsoBAB-)

[Extra: Detailed Report](https://hkustconnect-my.sharepoint.com/:w:/g/personal/hqiad_connect_ust_hk/EUN68XsdRp5LsQofhqTju1kBr7E_GGJ29CLXhQFJxr_pVA?e=FnApHd)

## Table Of Content
- [System Architecture](https://github.com/HKUST-COMP4651-25S/course-project-comp4651_mlpipeline?tab=readme-ov-file#system-architecture)
    - [Flow](https://github.com/HKUST-COMP4651-25S/course-project-comp4651_mlpipeline?tab=readme-ov-file#flow)
- [Getting Started](https://github.com/HKUST-COMP4651-25S/course-project-comp4651_mlpipeline?tab=readme-ov-file#getting-started)
    - [Prerequisites](https://github.com/HKUST-COMP4651-25S/course-project-comp4651_mlpipeline?tab=readme-ov-file#prerequisites)
    - [Deploy](https://github.com/HKUST-COMP4651-25S/course-project-comp4651_mlpipeline?tab=readme-ov-file#deploy)
        - [1. Set up a General Purpose S3 Bucket](https://github.com/HKUST-COMP4651-25S/course-project-comp4651_mlpipeline?tab=readme-ov-file#1-set-up-a-general-purpose-s3-bucket)
        - [2. Deploy EMR Cluster](https://github.com/HKUST-COMP4651-25S/course-project-comp4651_mlpipeline?tab=readme-ov-file#2-deploy-emr-cluster)
        - [3. Docker Container Build & Deployment](https://github.com/HKUST-COMP4651-25S/course-project-comp4651_mlpipeline?tab=readme-ov-file#3-docker-container-build--deployment)
            - [Option 1: Use Docker Container we provided](https://github.com/HKUST-COMP4651-25S/course-project-comp4651_mlpipeline?tab=readme-ov-file#option-1-use-docker-container-we-provided)
            - [Option 2: Build your own Docker Container (Testing)](https://github.com/HKUST-COMP4651-25S/course-project-comp4651_mlpipeline?tab=readme-ov-file#option-2-build-your-own-docker-container-testing)
        - [4. Front-end Deployment](https://github.com/HKUST-COMP4651-25S/course-project-comp4651_mlpipeline?tab=readme-ov-file#4-frontend-deployment)
- [API Endpoints](https://github.com/HKUST-COMP4651-25S/course-project-comp4651_mlpipeline?tab=readme-ov-file#api-endpoints)
- [Environment Variables](https://github.com/HKUST-COMP4651-25S/course-project-comp4651_mlpipeline?tab=readme-ov-file#environment-variables)
- [Custom ML Algorithm Experiment (Weighted Average Regressor)](https://github.com/HKUST-COMP4651-25S/course-project-comp4651_mlpipeline/tree/main#custom-ml-algorithm-experiment-weighted-average-regressor)


## System Architecture
![cloudProj drawio](https://github.com/user-attachments/assets/643d3749-b864-454e-8c27-980f34d6b9fe)


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
> Make sure every AWS service related components are deployed on the same region
#### 1. Set up a General Purpose S3 Bucket
Create your S3 Bucket with AWS defualt settings:
- general purpose
- acls disabled
- block all public access
- bucket versioning disabled
- SSE-S3 encryption
- bucket key enabled

##### After creating the bucket make sure it enable RESTful api access
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

#### 2. Deploy EMR Cluster
we support EC2 local mode and EMR yarn cluster mode

[if you want to deploy on EC2 local mode press me!](https://github.com/HKUST-COMP4651-25S/course-project-comp4651_mlpipeline?tab=readme-ov-file#option-2-build-your-own-docker-container-testing)

##### Deploy EMR cluster type:

![Screen Shot 2025-05-15 at 21 09 59 PM](https://github.com/user-attachments/assets/89dd6c0e-187d-496f-bbd4-fb1f30a9955d)

##### Recommended Provisioning config (3 worker nodes):

![Screen Shot 2025-05-15 at 21 11 58 PM](https://github.com/user-attachments/assets/d4e57f86-07e4-4ec4-bafd-6c67b4ae814d)

##### Subnet config:
> please choose a subnet that support m5.xlarge or the instance type you chooses

##### Amazon EMR service role: EMR_DefualtRole

AmazonElasticMapReduceRole &#x2611;

##### EC2 instance profile for Amazon EMR: EMR_EC2_DefaultRole

AmazonElasticMapReduceforEC2Role &#x2611;

Remarks:
> After creating the cluster, make sure master node SSH port and port 5001 is opened for connecting and RESTful API request

#### 3. Docker Container Build & Deployment

##### Option 1: Use Docker Container we provided
just pull our container to your EMR cluster master node or EC2 instance

###### First Connect to EMR master node:

```ssh -i your-key.pem hadoop@your-emr-master-dns```
> make sure your EMR master node SSH port is opened for connection


[How to install Docker on EMR](https://docs.aws.amazon.com/emr/latest/ReleaseGuide/emr-spark-docker.html)

[How to install Docker on EC2](https://medium.com/@srijaanaparthy/step-by-step-guide-to-install-docker-on-ubuntu-in-aws-a39746e5a63d)

###### EMR support (Yarn Cluster Mode)

```docker pull hyhen/ml-pipeline-backend:emr```

###### EC2 support (Local mode)

```docker pull hyhen/ml-pipeline-backend:ec2```

###### Run the backend container:
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

#### Option 2: Build your own Docker Container (Testing)

##### how to decide local or yarn mode:

```
spark = (
        SparkSession.builder
        .appName("MLPipelineApp")
        .master("local[*]") # Comment me if running on a cluster
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.1")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", 
                "com.amazonaws.auth.EnvironmentVariableCredentialsProvider")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.connection.establish.timeout", "5000")
        .config("spark.hadoop.fs.s3a.connection.timeout", "10000")
        .config("spark.hadoop.fs.s3a.attempts.maximum", "20")
        .config("spark.hadoop.fs.s3a.endpoint", f"s3.{os.environ.get('AWS_REGION', 'us-east-1')}.amazonaws.com")
        .getOrCreate()
    )
```

##### Local build and push your container

```
cd pipeline_backend

# Build and push for x86_64 (EC2/EMR compatibility)
docker buildx build --platform linux/amd64 -t your-repo/ml-pipeline-backend --push .
```

[continue deployment on EMR or EC2](https://github.com/HKUST-COMP4651-25S/course-project-comp4651_mlpipeline?tab=readme-ov-file#3-docker-container-build--deployment)

#### EC2 or local you want to run docker:

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
#### 4. Frontend deployment
First check your EMR master node or EC2 instance ipv4 public ip and paste it on ```.env``` file

```VITE_API_URL=http://your-server-ipv4-public-ip-here:5001```

Run development Server:

```
cd mlpipeline_app

npm install

npm dev run
```

Build application:

```
cd mlpipeline_app

npm install

npm build
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

### Custom ML Algorithm Experiment (Weighted Average Regressor):
#### Algorithm Design
We combined correlation-based feature weighting with automatic feature interaction detection to balance linear relationships and non-linear patterns.

```test_custom_algorithm.py``` file to independently test the functionality of the custom algorithms. This script loads and invokes functions to debug and verify the outcome.

#### Key Features
- ***Dynamic Weight Calculation:*** Feature weights derived from squared Pearson correlation coefficients with the target variable, emphasizing strong linear relationships.
- ***Interaction Detection:*** Identifies significant pairwise interactions (e.g., feature_i × feature_j) between features with high individual correlations.
- ***Regularization:*** Applies L2 regularization to shrink weights and prevent overfitting.
- ***Bias Adjustment:*** Centers predictions using the target variable’s mean for balanced outputs.

#### Integration: 
Competed alongside standard models (e.g., Random Forest) in the pipeline, with hyperparameter tuning for regularization strength and interaction term inclusion.

#### Advantages: 
Offers interpretable feature importance scores while capturing non-linear patterns, bridging the gap between linear models and complex ensembles.


  

