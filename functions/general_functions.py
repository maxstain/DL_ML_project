# functions/general_functions.py
from Imports.imports import *

# Constants for file paths
CONFIDENCE_THRESHOLD = 0.5
image_folder = "images"
output_folder = "Organized_images"
object_detection_model_path = "models/frozen_inference_graph.pb"
object_detection_labels_path = "models/labels.txt"


def organize_dataset():
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Image Labels"
    sheet["A1"] = "Image Name"
    sheet["B1"] = "Label"

    image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

    for image_file in image_files:
        label = image_file.split("_")[0]
        label_folder = os.path.join(output_folder, label)
        if not os.path.exists(label_folder):
            os.makedirs(label_folder)
        shutil.move(os.path.join(image_folder, image_file), os.path.join(label_folder, image_file))
        sheet.append([image_file, label])

    workbook.save(os.path.join(output_folder, "image_labels.xlsx"))


def detect_faces(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces


def create_model():
    model = Sequential()
    model.add(MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3)))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def create_train_generator(image_folder):
    datagen = ImageDataGenerator(validation_split=0.2)
    train_generator = datagen.flow_from_directory(
        image_folder,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        subset='training'
    )
    return train_generator


def create_validation_generator(image_folder):
    datagen = ImageDataGenerator(validation_split=0.2)
    validation_generator = datagen.flow_from_directory(
        image_folder,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        subset='validation'
    )
    return validation_generator


def load_labels(excel_path):
    workbook = openpyxl.load_workbook(excel_path)
    sheet = workbook.active
    labels = {}
    for row in sheet.iter_rows(min_row=2, values_only=True):
        image_name, label = row
        labels[image_name] = label
    return labels


def open_video():
    cap = cv.VideoCapture(0)
    return cap


def plot_confusion_matrix(y_true, y_pred):
    cm = sk.metrics.confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
