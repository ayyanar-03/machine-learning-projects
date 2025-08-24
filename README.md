# machine-learning-projects
Config files for my GitHub profile.


1.)   🌿 Medicinal Herbal Plant Identification using Deep Learning
📌 Overview

This project focuses on the automatic identification of medicinal herbal plants using deep learning techniques.
The goal is to build a classification model that can recognize and distinguish between different medicinal plants from images, aiding in research, healthcare, and conservation.

🚀 Features

Image classification of medicinal herbal plants.

Deep Learning models (Transfer Learning like VGG16, EfficientNet).

Trained and evaluated with accuracy, precision, recall, and F1-score.

Deployed in a Jupyter/Google Colab environment for easy usage.

📂 Dataset

Images of medicinal herbal plants collected from open-source datasets.

Preprocessed with resizing, normalization, and augmentation.

Split into training, validation, and testing sets.

🛠️ Tech Stack

Python

TensorFlow / Keras (for building the model)

NumPy, Pandas, Matplotlib (for data handling and visualization)

Google Colab (for training and execution)

⚙️ Model Architecture

Convolutional Neural Network (CNN) with multiple layers.

Transfer Learning (VGG16 / EfficientNet) for improved accuracy.

Optimizer: Adam

Loss Function: Categorical Crossentropy

📊 Results

Achieved high accuracy on the test dataset.

Model evaluation includes confusion matrix, precision, recall, and F1-score.

🔗 View Project

You can directly view and run the project on Google Colab here:
👉 Medicinal Herbal Plant Identification - Google Colab - https://colab.research.google.com/drive/1H3ope7VMOIjUi9INogG6juazNv3nnJO1?usp=sharing


2.)  🎭 Emotion Detection in Video
📌 Overview

This project focuses on detecting human emotions in real-time video using Deep Learning and Computer Vision.
It captures frames from a video, detects faces, and classifies emotions such as happy, sad, angry, surprised, neutral, fear, and disgust.

🚀 Features

Real-time face detection using OpenCV.

Emotion recognition using Deep Learning models.

Supports video input (uploaded video).

Visualizes results by drawing bounding boxes with emotion labels.

📂 Workflow

Upload or stream a video.

Extract frames using OpenCV.

Detect faces in each frame.

Recognize emotions with DeepFace / CNN.

Display results with labeled bounding boxes.

🛠️ Tech Stack

Python

OpenCV (for video and face detection)

DeepFace (for pre-trained emotion detection)

TensorFlow / Keras (for custom models if used)

Matplotlib / Seaborn (for visualization)

📊 Results

Emotions detected frame by frame.

Bounding boxes with labels (e.g., Happy 😀, Sad 😢, Angry 😡).

Can analyze emotional trends throughout the video.
to view a project as a video : https://drive.google.com/file/d/1fYN30psAeO0NwVd07ElLCMB-CfrmS8Oz/view?usp=drive_link

3.)

🤟 American Sign Language (ASL) Detection
📌 Overview

This project focuses on building a Sign Language Detection System that translates American Sign Language (ASL) gestures into text or speech using Deep Learning and Computer Vision.
It uses your webcam to capture hand gestures in real-time and translates them instantly.

🚀 Features

Real-time ASL detection via webcam.

Converts ASL gestures into text or speech.

Supports both static gestures (alphabets, numbers) and dynamic gestures (words, phrases).

Uses MediaPipe for hand landmark detection and LSTM networks for gesture recognition.

📂 Workflow

Start the program and turn on your webcam.

Show hand gestures in front of the camera.

Detects key landmarks (hand & fingers) using MediaPipe.

Predicts the corresponding ASL gesture using the trained LSTM model.

Displays the recognized gesture as text on screen (optionally converts to speech).

🛠️ Tech Stack

Python

TensorFlow / Keras (for training LSTM model)

MediaPipe (for hand tracking & landmark detection)

OpenCV (for webcam video capture)

📊 Results

Real-time ASL gesture detection using webcam.

High accuracy for both alphabets and simple words.

Optionally, converts recognized signs to speech.

📬 Future Improvements

Extend support for more phrases and sentences.

Enhance performance in low-light environments.

Deploy as a web app / mobile app.

to view a proj: https://colab.research.google.com/drive/1m-O4Vl5U7skDMaFQSiIGVys01tVV8Ueb
