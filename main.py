# This Script is made by Firas CHABCHOUB
# It's a Deep learning and Machine learning project that uses live feeds to detect and classify objects in real time.
# The project is made using Python and TensorFlow.
# The project is still in progress and will be updated regularly.

from functions.general_functions import *
from models.Person_recognition_model import PersonRecognitionModel


def main():
    # Initialize the recognition model
    recognition_model = PersonRecognitionModel(image_folder, output_folder)

    # Organize images into subfolders
    recognition_model.organize_images()

    # Create data generators
    recognition_model.create_data_generators()

    # Build the model
    recognition_model.build_model()

    # Train the model
    recognition_model.train_model(epochs=10)

    # Optionally fine-tune the model
    recognition_model.fine_tune_model(epochs=5)

    # Save the trained model
    recognition_model.save_model('person_recognition_model.h5')

    recognition_model.process_video()


if __name__ == '__main__':
    organize_dataset()
    main()
