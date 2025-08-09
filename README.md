# Pest detection with 2WD robot
*realtimepestdet.py*- Python script that performs real-time pest classification on a Raspberry Pi, powered by a trained EfficientNetV2-B0 model. The script captures a live video feed using Picamera2, annotates it with predictions, and automatically logs results and saves key frames for analysis.

*robotcontrol.py*: Provides direct keyboard control for a two-wheeled Raspberry Pi robot. This script utilizes the gpiozero library for simple motor control and the curses library to map arrow key inputs to corresponding movements like forward, reverse, and turning.

*final_robotics_train.ipynb*:A Jupyter Notebook that trains an EfficientNetV2B0 model with TensorFlow for pest classification. 

