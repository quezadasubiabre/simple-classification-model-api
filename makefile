run:
	uvicorn main:app --host 0.0.0.0 --port 8000 --reload

send-image:
	curl -X POST \
		'http://127.0.0.1:8000/predict/' \
		-H 'accept: application/json' \
		-H 'Content-Type: multipart/form-data' \
		-F 'file=@test_images/dog.png'

docker-build:
	docker build -t pytorch-fastapi:latest .

docker-run:
	docker run -d --name pytorch-fastapi-container -p 8000:8000 pytorch-fastapi