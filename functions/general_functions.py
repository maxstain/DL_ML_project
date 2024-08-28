from Imports.imports import *

# Constants for file paths
CONFIDENCE_THRESHOLD = 0.5
image_folder = "C:/Projects/Python/DL_ML_project/images"
output_folder = "C:/Projects/Python/DL_ML_project/Organized_images"


def organize_dataset():
    for filename in os.listdir(image_folder):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            person_name = filename.split(".")[0]
            person_folder = os.path.join(output_folder, person_name)
            os.makedirs(person_folder, exist_ok=True)
            shutil.copy(os.path.join(image_folder, filename), os.path.join(person_folder, filename))
    print("Dataset organized")
