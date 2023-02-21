import json
import boto3
import base64
import sagemaker
from sagemaker.serializers import IdentitySerializer
from sagemaker.predictor import Predictor



### Lambda function for Function 1: data serialization
s3 = boto3.client('s3')

def lambda_handler(event, context):
    """A function to serialize target data from S3"""
    
    # Get the s3 address from the Step Function event input
    bucket = event["s3_bucket"]
    key = event["s3_key"]
    
    # Download the data from s3 to /tmp/image.png
    local_path = '/tmp/image.png'
    s3.download_file(bucket, key, local_path)
    
    # We read the data from a file
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read())

    # Pass the data back to the Step Function
    print("Event:", event.keys())
    return {
        'statusCode': 200,
        'body': {
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []
        }
    }


### Lambda function for Function 2: getting inferences

# Fill this in with the name of your deployed model
ENDPOINT = "image-classification01endpoint"

def lambda_handler(event, context):

    # Decode the image data
    image = event["image_data"]
    image = base64.b64decode(image)

    # Instantiate a Predictor
    predictor = Predictor(endpoint_name=ENDPOINT)

    # For this model the IdentitySerializer needs to be "image/png"
    predictor.serializer = IdentitySerializer("image/png")
    
    # Make a prediction:
    inferences = predictor.predict(image, initial_args={"ContentType": "image/png"})
    
    # We return the data back to the Step Function
    return {
        'statusCode': 200,
        'body': {
            "image_data": event["image_data"],
            "s3_bucket": event["s3_bucket"],
            "s3_key": event["s3_key"],
            "inferences": inferences.decode('utf-8')
        }
    }


### Lambda function for Function 3: Filter inferences

THRESHOLD = 0.93

def lambda_handler(event, context):
    
    # Grab the inferences from the event
    inferences = eval(event["inferences"])
    
    # Check if any values in our inferences are above THRESHOLD
    meets_threshold = False
    for x in inferences:
        if meets_threshold == False:
            if x > THRESHOLD:
                meets_threshold = True
    
    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    if meets_threshold:
        inference_result = "pass"
    else:
        inference_result = "fail"

    return {
        'statusCode': 200,
        'body': {
            "image_data": event["image_data"],
            "s3_bucket": event["s3_bucket"],
            "s3_key": event["s3_key"],
            "inferences": event["inferences"],
            "inference_result": inference_result
        }
    }


