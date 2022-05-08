# Pictionary
This is our final project for Computer Vision (EE 379V) at UT Austin. Our group members are Satya Boddu, Michael Chen, Udai Jain, Rishi Ponnekanti, and Ishan Shah.

This project seeks to use computer vision to recreate a computer-based game of Pictionary, in which a human player draws an image and the computer attempts to guess the drawing. We also plan on having the computer generate drawings for the player to guess using Generative Adversarial Networks (GANs).

![demo_gif](videos/pictionary.mov)

## Development
The development environment is containerized, so both the server and client can be run with a single Docker command. After installing [Docker](https://docs.docker.com/get-docker/), simply run the following command:

```bash
docker-compose up --build
```

The client (React app) will be accessible at `localhost:3000`, and the server (FastAPI app) will be accessible at `localhost:5000`.

To stop the container from running, simply type `Ctrl+C` and then the following command:

```bash
docker-compose down
```
