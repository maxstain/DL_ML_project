# Imports/imports.py
try:
    import cv2 as cv
    import os
    import shutil
    import glob
    import xml.etree.ElementTree as ET
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import openpyxl
    import tensorflow as tf
    import pandas as pd
    import scipy as sp
    import sklearn as sk
    import seaborn as sns
    from keras import Sequential
    from keras.src.applications.mobilenet_v2 import MobileNetV2
    from keras.src.layers import GlobalAveragePooling2D, Dense
    from keras.src.legacy.preprocessing.image import ImageDataGenerator
    from keras.src.optimizers import Adam
except ImportError:
    print("Error: Missing Required Libraries")
    print("Installing Required Libraries from Requirements.txt")
    os.system("pip install -r requirements.txt")
