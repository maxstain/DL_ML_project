try:
    import cv2 as cv
    import os
    import shutil
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import openpyxl
    from keras import Sequential
    from keras.src.applications.mobilenet_v2 import MobileNetV2
    from keras.src.layers import GlobalAveragePooling2D
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from keras.src.optimizers import Adam
    from keras.src.layers import Dense
    import pandas as pd
    import scipy as sp
    import sklearn as sk
    import seaborn as sns
except ImportError:
    print("Error: Missing Required Libraries")
    print("Installing Required Libraries from Requirements.txt")
    os.system("pip install -r requirements.txt")
