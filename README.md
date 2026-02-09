# AI-based Robot Path Planning Game

An AI-based grid path planning application combining convolutional neural networks and classical D* Lite planning.

---

## Overview

This project presents an AI-based grid path planning game developed in Python.  
The system combines a **Convolutional Neural Network (CNN)** for local next-move prediction with the **D* Lite algorithm** for global path planning.

The application is implemented as an interactive game using **Pygame**, where an agent must navigate from a start position to a goal while avoiding obstacles.

---

## Methods

### CNN Next-Move Prediction
- **Input:** occupancy grid, start position, goal position, current agent position  
- **Output:** next movement direction (UP, DOWN, LEFT, RIGHT)

### D* Lite Algorithm
- Used as a global planner and fallback when the CNN fails

### Hybrid Planning Strategy
- CNN is used for fast local decisions
- D* Lite ensures path completion and robustness

---

## Training Details
- CNN trained on **120,000 samples**
- **8 training epochs**
- Final test accuracy: **~89%**
- Training performed in **Google Colab**

---

## Application Features
- Interactive grid-based game
- Player mode (manual control)
- Simulation mode (AI-controlled agent)
- Dynamic switching between CNN and D* Lite
- Character selection and graphical interface

---

## Technologies Used
- Python
- PyTorch
- Pygame
- NumPy
