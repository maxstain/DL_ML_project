import os
import shutil

try:
    import cv2 as cv
    import os
    import numpy as np
    from keras import Sequential
    from keras.src.applications.mobilenet_v2 import MobileNetV2
    from keras.src.layers import GlobalAveragePooling2D
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from keras.src.optimizers import Adam
    from keras.src.layers import Dense
except ImportError:
    print("Error: Missing Required Libraries")
    print("Installing Required Libraries from Requirements.txt")
    os.system("pip install -r requirements.txt")
