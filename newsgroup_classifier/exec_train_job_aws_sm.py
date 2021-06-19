"""
This is a driver script to execute the training job and deploy the model on AWS Sagemaker
"""
from sagemaker.sklearn.estimator import SKLearn

# Role created in AWS IAM with 2 mandatory policies added : AmazonS3FullAccess, AmazonSageMakerFullAccess
role = 'SM_main'

# Create the SKLearn Object by directing it to the source_dir/train.py script.
aws_sklearn = SKLearn(entry_point='train.py',
                      source_dir='source_dir',
                      instance_type='ml.m4.xlarge',
                      role=role,
                      py_version='py3',
                      framework_version='0.20.0')

# Train the model using by passing the path to the S3 bucket with the training data
aws_sklearn.fit({'train': 's3://sklearn-deploy-dataset/train/train.csv'})

# Deploy model
aws_sklearn_predictor = aws_sklearn.deploy(instance_type='ml.m4.xlarge',
                                           initial_instance_count=1)

print(f"End point created : {aws_sklearn_predictor.endpoint_name}")
