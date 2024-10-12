
# PyTorch FastAPI Image Prediction

## Run the Application
To run the FastAPI application locally, use the following command:

```bash
make run
```

This will start the server at http://127.0.0.1:8000.

## Send an Image
To send an image for prediction, you can use the following curl command:

```bash
make send-image
```

This command will POST the image located at `test_images/dog.png` to the `/predict/` endpoint of the API. Make sure to replace the path with the image you want to test.

## Docker Commands
If you prefer to run the application in a Docker container, you can use the following commands:

### Build the Docker Image
To build the Docker image, run:

```bash
make docker-build
```

### Run the Docker Container
To run the Docker container, use:

```bash
make docker-run
```

This will start the container and expose the application on port 8000.

## API Endpoint
**POST /predict/**: Upload an image and receive the predicted class.

### Example Request
```bash
curl -X POST     'http://127.0.0.1:8000/predict/'     -H 'accept: application/json'     -H 'Content-Type: multipart/form-data'     -F 'file=@test_images/dog.png'
```

### Response
The response will include the predicted class in JSON format:

```json
{
    "predicted_class": "your_class_name"
}
```

