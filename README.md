# MAIRROR - Project Draft

Mairror is a project that uses Generative Adversarial Networks (GANs) to guess the age of a person based on a picture.

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [Dataset](#dataset)
- [Training](#training)
- [Applications](#applications)
- [Main Components](#main-components)
- [Languages and Libraries](#languages-and-libraries)
  - [Data, ML and DL](#data-ml-and-dl)
  - [Infrastructure](#infrastructure)
- [References](#references)
- [Credits](#credits)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Dataset

The dataset used is https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/

This dataset contains more than 250 GB of Wikipedia images of famous actors and cartoons.

## Training

The Age-cGAN consists of 4 networks - an encoder, the FaceNet, a generator network and a discriminator network.

The functions of each of the 4 networks:

- Encoder - Helps us learn the inverse mapping of input face images and the age condition with the latent vector z0.
- FaceNet - It is a face recognitiohttps://docs.aws.amazon.com/lambda/latest/dg/with-s3-example.htmln network which learns the difference between an input image x and a reconstructed image x~.
- Generator network - Takes a hidden (latent) representation consisting of a face image and a condition vector, and generates an image.
- Discriminator network - Discriminates between the real and fake images.

## Applications

Once the model is trained, the application will be accessible using a frontend that will query an API. The image will be uploaded and inferred using the trained model and the result will be displayed in the fronted.

Optionally, a face detection system will be used using OpenCV to detect faces in real time.

Also, a Telegram bot will be developed and it will have the same functionality as the frontend. To access the bot, a QR code will be generated (no needed, but we can use it to link the bot url) and the user will scan it to access the bot.

## Main Components

- GNU/Linux
- FastAPI API
- Streamlit Frontend
- MongoDB Database
  - Experiments data and results
  - Images URIs and results
- Tensorflow Model
- Generative Adversarial Networks (GANs)
- Kubeflow (optional)
- strimzi 
- Telegram bot
- AWS S3 Bucket
- AWS EKS Kubernetes Cluster with eksctl
- AWS Nuke
- Github
- Github Actions
- GNU Make
- Docker and docker-compose
- Terraform
- Helmfile and helm charts
- Pytest

## Languages and Libraries

### Data, ML and DL

The project will be developed using Python 3.9.

We will use Jupyter Lab for data cleaning and ETL.

The Deep Learning libraries that we will use are Tensorflow 2.7.0 and we may also use Pytorch. The ML experiments results will be upload to Tensorboard.dev.

Code will be tested using Pytest and GNU Make (locally)

### Infrastructure 

Code will be deployed automatically using Github workflows with Github Actions.

The infrastructure will be defined using Terraform, Helm and

The Kubernetes cluster will be created in AWS using eksctl.









## References

- https://analyticsindiamag.com/top-8-gan-based-projects-one-can-try-their-hands-on/
- https://iq.opengenus.org/face-aging-cgan-keras/
- https://medium.com/analytics-vidhya/introduction-to-facenet-a-unified-embedding-for-face-recognition-and-clustering-dbdac8e6f02
- https://dev.to/ash11sh/using-telegram-bot-for-image-classification-3afk
- https://youtu.be/7L6SCufzYT8
- https://github.com/ogurbych/Age-Gender-Recognition
- https://tensorboard.dev/
- https://discuss.streamlit.io/t/live-webcam-feed-into-the-web-app/397
## Credits

- https://github.com/thlorenz/doctoc


