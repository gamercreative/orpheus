# Orpheus – LSTM-Based Drawing Model

Orpheus is a neural network project that generates sketches and drawings using **sequence modeling**. Built with **PyTorch**, it leverages **LSTM (Long Short-Term Memory)** layers to learn stroke sequences and reproduce human-like drawing trajectories. The model supports **GPU acceleration** and variable-length sequences.

The biggest time investment for this project was decreasing the models capacity while retaining its ability to output correct strokes while still mainting the ability to identify correct stop tokens ( it outputs the stop token and only a failsafe is implemented for 500 steps) keeping learning to a maximum and just having memorized outputs be enough to a minimum.

The dataset has been recorded, curated and cleaned by me.

Although the project does not follow all the correct principles and best practices this is my first full implementation of a model from 0 to working.

---

## Table of content

- [Features](#Features)
- [Architecture](#Architecture)
- [Dataset](#Dataset)
- [Training](#Training)
- [Inference](#Inference)
- [Achievements](#Achievements)

---

## Features

- Handles variable length training data using padding
- Handles all the letters of the alphabet a-z lower case
- Gpu accelerated and optimized
- Visualization for the models stroke history

---

## Architecture

- **Input**: the input is based on a `[dx,dy,pen_state,timestamp]` approach.

- **lstm mode**: picked a lstm model as a base for my machine learning approach and retain memory along strokes. 

- **Output**: outputs the same pattern as the input including the timestamp to test for anomilies and to use for styling later on.

- **Loss Functions**: `MSE` for dx, dy and `CE` for the pen states.

---

## Dataset

### Capture
- Capture was done using python (pygames library) and saved as a json file (mostly chatgpt)

### Curation and cleaning
- Cleaning and curating the dataset was done in Julia(the programming language) as im most familiar with it and i find it best for data projects.
- After cleaning, structuring and handlling some anomilies in data we visualize the output and save it to a json file.

### Saving
- i choose a file system database as opposed to a relational one for simplicity

---

## Training

### Training method
- the training method mainly used is an autogressive approach

### training method varient and enhancement
- the autogresive approach was augmented with scheduled teacher foricing

### training targets
- correct dx dy output strokes to shape the letter well
- correct pen states
- correct start and end token sequences
- the more the model trains the more it can train on recovery due to shceduled teacher forcing

---

## Inference

### Inference method
- we use an autogressive approach to inference

---

## Achievements

### Things that went right
- the model was able to utilize embeddings where it could maximizing understanding
- the model learned the correct structure for each letter (when to raise pen, when to stop generation and output end token...)
- the model showed resemblance between similar letters for example (p,q | g,j) which might or might not be present in the epoch present in the script
- use its own embeddings for each letter to fully understand each one
- trained on its own output to train on recovery and stray handling in addition to adding variety

### Personal goals achieved
- Utilized julia for data and python for machine learning merging best of both worlds for me.
- Learned the dynamics of a model after architecture ( i focused on learning the model itself and forget everything around it ).
- Get a high level understanding of models in general during traning, inference, knowledge aquisition and way more which will help me write better model architectures in the future.

---
