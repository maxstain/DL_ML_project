# functions/general_functions.py

from Imports.imports import *

# Constants for file paths
CONFIDENCE_THRESHOLD = 0.5
image_folder = "images"
output_folder = "Organized_images"


# Function to organize the dataset into an Excel file with the images in the correct folders for training the model.
def organize_dataset():
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create a list to store the image paths
    image_paths = []

    # Loop through the images folder and add the image paths to the list
    for root, dirs, files in os.walk(image_folder):
        for file in files:
            if file.endswith(".jpg"):
                image_paths.append(os.path.join(root, file))

    # Create a Pandas DataFrame to store the image paths
    df = pd.DataFrame(image_paths, columns=['image_path'])

    # Split the image paths into folders based on the file name
    df['folder'] = df['image_path'].apply(lambda x: x.split('\\')[-1].split('_')[0])

    # Create the output folders based on the unique folder names
    for folder in df['folder'].unique():
        os.makedirs(os.path.join(output_folder, folder), exist_ok=True)

    # Move the images to the correct output folders
    for index, row in df.iterrows():
        destination_folder = os.path.join(output_folder, row['folder'])
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
        shutil.copy(row['image_path'], os.path.join(destination_folder, row['image_path'].split('\\')[-1]))
