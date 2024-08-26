import os

try:
    import tensorflow as tf
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2 as cv
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.applications.vgg16 import preprocess_input
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
    from tensorflow.keras.models import load_model
    from tensorflow.keras.applications.vgg16 import decode_predictions
    from tensorflow.keras.applications.vgg16 import preprocess_input
except ImportError:
    print("Error: Missing Required Libraries")
    print("Installing Required Libraries from Requirements.txt")
    os.system("pip install -r requirements.txt")
