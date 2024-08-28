from keras.src.layers import Dense

from Imports.imports import *


def detect_faces(frame):
    """Detect faces in the frame using Haar Cascades."""
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces


class PersonRecognitionModel:
    def __init__(self, image_folder, output_folder, input_shape=(224, 224, 3), batch_size=32, learning_rate=0.001):
        self.image_folder = image_folder
        self.output_folder = output_folder
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = None
        self.train_generator = None
        self.validation_generator = None

    def organize_images(self):
        """Organize images into subfolders based on the person name."""
        for filename in os.listdir(self.image_folder):
            if filename.endswith(".png"):
                person_name = filename.split(".")[0]
                person_folder = os.path.join(self.output_folder, person_name)
                os.makedirs(person_folder, exist_ok=True)
                shutil.copy(os.path.join(self.image_folder, filename), os.path.join(person_folder, filename))

    def create_data_generators(self):
        """Create training and validation data generators with augmentation."""
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=0.2  # 20% of data for validation
        )

        self.train_generator = train_datagen.flow_from_directory(
            self.output_folder,
            target_size=self.input_shape[:2],
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training'
        )

        self.validation_generator = train_datagen.flow_from_directory(
            self.output_folder,
            target_size=self.input_shape[:2],
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation'
        )

    def build_model(self):
        """Build the MobileNetV2 model with transfer learning."""
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=self.input_shape)

        self.model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(1024, activation='relu'),
            Dense(len(self.train_generator.class_indices), activation='softmax')
        ])

        base_model.trainable = False  # Freeze the base model

        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def train_model(self, epochs=10):
        """Train the model with the training and validation data."""
        self.model.fit(
            self.train_generator,
            validation_data=self.validation_generator,
            epochs=epochs
        )

    def fine_tune_model(self, fine_tune_learning_rate=1e-5, epochs=10):
        """Fine-tune the base model to improve accuracy."""
        self.model.layers[0].trainable = True  # Unfreeze the base model

        self.model.compile(optimizer=Adam(learning_rate=fine_tune_learning_rate), loss='categorical_crossentropy',
                           metrics=['accuracy'])

        self.model.fit(
            self.train_generator,
            validation_data=self.validation_generator,
            epochs=epochs
        )

    def save_model(self, model_path='person_recognition_model.h5'):
        """Save the trained model to a file."""
        self.model.save(model_path)

    def predict(self, img_array):
        """Predict the person in a given image array."""
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = self.model.predict(img_array)
        predicted_class = np.argmax(predictions[0])

        return list(self.train_generator.class_indices.keys())[predicted_class]

    def process_video(self, video_source=0):
        """Process video feed, detect faces, and predict the person in each frame."""
        cap = cv.VideoCapture(video_source)

        if not cap.isOpened():
            print("Error: Could not open video source.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            # Optionally use a face detector
            faces = detect_faces(frame)

            for (x, y, w, h) in faces:
                face_img = frame[y:y + h, x:x + w]
                face_img = cv.resize(face_img, self.input_shape[:2])
                prediction = self.predict(face_img)

                # Draw the bounding box and label
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv.putText(frame, prediction, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv.imshow('Video Feed', frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv.destroyAllWindows()
