"""
This code is for uploading the training data to AWS S3
"""
import boto3



bucket = 'sklearn-deploy-dataset'
region = 'us-east-1'
s3_session = boto3.Session().resource('s3')
s3_session.create_bucket(Bucket=bucket,
                         CreateBucketConfiguration=
                         {'LocationConstraint': region})
s3_session.Bucket(bucket).Object('train/train.csv').upload_file('data/train.csv')
