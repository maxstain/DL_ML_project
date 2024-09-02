from Imports.imports import *
from functions.general_functions import *


def detect_faces(frame):
    """Detect faces in the frame using Haar Cascades."""
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


class PersonRecognitionModel:
    def __init__(self, image_folder, output_folder):
        self.image_folder = image_folder
        self.output_folder = output_folder
        self.model = create_model()
        self.train_generator = None
        self.validation_generator = None

    def create_data_generators(self):
        datagen = ImageDataGenerator(validation_split=0.2)
        self.train_generator = datagen.flow_from_directory(
            self.image_folder,
            target_size=(150, 150),
            batch_size=32,
            class_mode='binary',
            subset='training'
        )
        self.validation_generator = datagen.flow_from_directory(
            self.image_folder,
            target_size=(150, 150),
            batch_size=32,
            class_mode='binary',
            subset='validation'
        )

        # Debugging information
        print(f"Train generator: {self.train_generator}")
        print(f"Validation generator: {self.validation_generator}")
        print(f"Number of training samples: {self.train_generator.samples}")
        print(f"Number of validation samples: {self.validation_generator.samples}")

    def build_model(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(150, 150, 3)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        self.model.compile(
            loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
            metrics=['accuracy']
        )

    def train_model(self, epochs):
        if self.train_generator is None or self.validation_generator is None:
            print("Error: Data generators are not created.")
            return

        if self.train_generator.samples == 0 or self.validation_generator.samples == 0:
            print("Error: No images found in the specified directory.")
            return

        self.model.fit(
            self.train_generator,
            steps_per_epoch=self.train_generator.samples // self.train_generator.batch_size,
            epochs=epochs,
            validation_data=self.validation_generator,
            validation_steps=self.validation_generator.samples // self.validation_generator.batch_size
        )

    def fine_tune_model(self, epochs):
        # Fine-tune the model
        pass

    def save_model(self, model_path):
        self.model.save(model_path)

    def process_video(self):
        # Process video frames (implementation not shown)
        pass
