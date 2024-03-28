import json
import boto3
from botocore.exceptions import ClientError

s3_client = boto3.client('s3')

def generate_presigned_url(bucket_name, object_name, expiration=3600):
    try:
        response = s3_client.generate_presigned_url('put_object',
                                                    Params={'Bucket': bucket_name,
                                                            'Key': object_name},
                                                    ExpiresIn=expiration)
    except ClientError as e:
        print(e)
        return None
    return response

def lambda_handler(event, context):
    # Default response structure
    response_structure = {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',  # Include this line to enable CORS
        },
        'body': '{}'
    }
    
    try:
        # Assuming the body is always present, but might not be JSON or might be empty
        body = json.loads(event.get('body', '{}'))  # Default to empty dict if body is None or not present
    except json.JSONDecodeError:
        # Return an error response if the body cannot be parsed
        response_structure['statusCode'] = 400  # Bad request
        response_structure['body'] = json.dumps({'error': 'Invalid JSON format in request body'})
        return response_structure

    # Extract the image name from the body
    image_name = body.get('imageName')
    
    # Check if image_name is present
    if not image_name:
        response_structure['statusCode'] = 400  # Bad request
        response_structure['body'] = json.dumps({'error': 'imageName not provided in request body'})
        return response_structure

    # Prepend the folder name to the object_name to save it in the "images/" folder
    object_name = f"images/{image_name}"

    # Proceed with generating the pre-signed URL
    bucket_name = 'asl-bucket2024'
    url = generate_presigned_url(bucket_name, object_name)
    
    if url is not None:
        response_structure['body'] = json.dumps({'url': url})
    else:
        response_structure['statusCode'] = 500
        response_structure['body'] = json.dumps({'error': 'Could not generate pre-signed URL'})
    
    return response_structure