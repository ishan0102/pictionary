# Pictionary
This is our final project for Computer Vision (EE 379V) at UT Austin. Our group members are Satya Boddu, Michael Chen, Udai Jain, Rishi Ponnekanti, and Ishan Shah.

This project seeks to use computer vision to recreate a computer-based game of Pictionary, in which a human player draws an image and the computer attempts to guess the drawing. We also plan on having the computer generate drawings for the player to guess using Generative Adversarial Networks (GANs).

## Development
### Client
To run the client, you'll need `Node.js` and `npm`. You can install these using the [Node Version Manager](https://github.com/nvm-sh/nvm). After you've installed these modules, you can set up the client using the following commands. Make sure you're in the `client/` directory.

**Install all node packages:**
```sh
npm i
```

**Run the client:**
```sh
npm start
```

### Server
To run the server, you'll need `Python 3`. You can set up the server using the following commands. Make sure you're in the `server/` directory.

**Create and activate a virtual environment:**
```sh
python3 -m venv venv
source venv/bin/activate
```

**Install Python libraries to the virtual environment:**
```sh
pip install -r requirements.txt
```

**Run the server:**
```sh
uvicorn main:app --reload
```
