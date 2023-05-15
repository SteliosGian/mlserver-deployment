# MLServer Deployment

[![LinkedIn][linkedin-shield]][linkedin-url]

Deployment of a Fake news detector using Seldon's MLServer open source package.


<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
      </ul>
    </li>
    <li><a href="#mlserver">MLServer</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

A simple fake news detector model that detects if a text is considered as ```fake``` or not. To accomplish this, we used the Bert pretrained model to make inference based on our new dataset.  
The goal is to create an inference server based on [Seldon's](https://www.seldon.io/) open source package [MLServer](https://mlserver.readthedocs.io/en/latest/index.html), and generate predictions based on a text input.

### Built With

* [Docker](https://www.docker.com/)
* [MLServer](https://mlserver.readthedocs.io/en/latest/index.html)
* [Pipenv](https://pipenv-fork.readthedocs.io/en/latest/)
* [PyTorch](https://pytorch.org/)


### Dataset

The dataset used in this project is the "REAL and FAKE news dataset" which is taken from [Kaggle](https://www.kaggle.com/datasets/nopdev/real-and-fake-news-dataset). The dataset contains 3 columns. The "title" column, the "text" column and the "label" column which is the target label.

## Getting Started

Before the inference server is created, a trained model needs to exist in the ```src/inference/model``` directory. The model needs to be named ```model.pt``` as indicated in the ```model-setttings.json``` file.

The ```src/fake_news_detector/train.py``` file can be used to train a new model and the Kaggle dataset needs to be placed inside the ```src/data``` directory.

The ```run-local.sh``` script takes one argument, depending on whether you want to do **training** or **inference**. The default argument is inference.  

To run the training phase, type:

```bash
bash run-local.sh training
```
After that, a PyTorch model names ```model.pt``` is placed inside the ```src/inference/temp_model``` directory. Copy that model in the ```src/inference/model``` directory and run the inference phase to start the server that will use that model.

Otherwise, just type:

```bash
bash run-local.sh
```

This script installs all the necessary packages from ```Pipfile.lock``` and creates the server using the ```MLServer``` package.

The inference can be accessed from:
```
http://<URL>/v2/models/<MODEL_NAME/infer
```
Which in our case, since we deploy it locally, is:
```
http://0.0.0.0:8080/v2/models/fake-news-detector/infer
```

An example payload is:
```json
{
  "inputs": [
    {
      "name": "text",
      "shape": [1],
      "datatype": "BYTES",
      "data": "This is a fake text"
    }
  ]
}
```

An example using curl:
```bash
curl -X 'POST' \
  'http://0.0.0.0:8080/v2/models/fake-news-detector/infer' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "inputs": [
    {
      "name": "text",
      "shape": [1],
      "datatype": "BYTES",
      "data": "This is a fake text"
    }
  ]
}'
```

## MLServer

MLServer is a way to serve ML models through a REST and gRPC interface.  
 As mentioned in the official [documentation](https://github.com/SeldonIO/MLServer#overview), MLServer supports:

 * Multi-model serving
 * Parallel inference
 * Adaptive batching
 * Scalability with Kubernetes native frameworks, such as [Seldon Core](https://docs.seldon.io/projects/seldon-core/en/latest/index.html) and [KServe](https://kserve.github.io/website/0.10/)
 * Standard [V2 Inference Protocol](https://docs.seldon.io/projects/seldon-core/en/latest/reference/apis/v2-protocol.html) on REST and gRPC


### Prerequisites

Docker must be installed in order to create the server.


[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-white.svg?
[linkedin-url]: https://linkedin.com/in/stelios-giannikis
