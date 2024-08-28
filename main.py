# This Script is made by Firas CHABCHOUB
# It's a Deep learning and Machine learning project that uses live feeds to detect and classify objects in real time.
# The project is made using Python and TensorFlow.
# The project is still in progress and will be updated regularly.

from functions.camera_functions import *


def main():
    try:
        model = load_model(MODEL_PATH, CONFIG_PATH)
        labels = load_labels(LABELS_PATH)
        open_camera(model, labels)
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    organize_dataset()
# main()
