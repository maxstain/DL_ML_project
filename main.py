# main.py
from functions.general_functions import *
from models.person_recognition_model import PersonRecognitionModel
from models.object_detection_model import ObjectDetectionModel


def main():
    organize_dataset()
    choice = input("Do you want face recognition or object recognition? (1/2): ")
    if choice == '1':
        # Initialize the recognition model
        recognition_model = PersonRecognitionModel(image_folder, output_folder)

        # Create data generators
        recognition_model.create_data_generators()

        # Build the model
        recognition_model.build_model()

        # Train the model
        recognition_model.train_model(epochs=10)

        # Save the trained model
        recognition_model.save_model('models/person_recognition_model.keras')

        # Process the video feed
        recognition_model.process_video()
    elif choice == '2':
        # Initialize the object detection model
        object_detection_model = ObjectDetectionModel(object_detection_labels_path)

        # Load the model
        object_detection_model.load_model()

        # Process the video feed
        object_detection_model.process_video()
    else:
        print("Invalid choice. Please enter 1 or 2.")


if __name__ == '__main__':
    main()
