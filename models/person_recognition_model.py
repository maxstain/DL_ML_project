# models/person_recognition_model.py
from Imports.imports import *
from functions.general_functions import *


class PersonRecognitionModel:
    def __init__(self, image_folder, output_folder):
        self.image_folder = image_folder
        self.output_folder = output_folder
        self.model = create_model()
        self.train_generator = create_train_generator(image_folder)
        self.validation_generator = create_validation_generator(image_folder)

    def create_data_generators(self):
        datagen = ImageDataGenerator(validation_split=0.2)
        self.train_generator = datagen.flow_from_directory(
            self.image_folder,
            target_size=(224, 224),
            batch_size=32,
            class_mode='binary',
            subset='training'
        )
        self.validation_generator = datagen.flow_from_directory(
            self.image_folder,
            target_size=(224, 224),
            batch_size=32,
            class_mode='binary',
            subset='validation'
        )

    def build_model(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(224, 224, 3)),
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
        self.evaluate_model()

    def evaluate_model(self):
        y_true = self.validation_generator.classes
        y_pred = self.model.predict(self.validation_generator)
        y_pred = np.where(y_pred > 0.5, 1, 0)

        print("Classification Report:")
        print(sk.metrics.classification_report(y_true, y_pred))

        print("Confusion Matrix:")
        plot_confusion_matrix(y_true, y_pred)

    def save_model(self, model_path):
        self.model.save(model_path)

    def process_video(self):
        labels = load_labels(os.path.join(self.output_folder, "image_labels.xlsx"))
        cap = open_video()
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            faces = detect_faces(frame)
            for (x, y, w, h) in faces:
                face = frame[y:y + h, x:x + w]
                face = cv.resize(face, (224, 224))
                face = np.expand_dims(face, axis=0)
                prediction = self.model.predict(face)
                label = "Unknown"
                if prediction > 0.5:
                    person_name = [person for person in os.listdir(self.output_folder) if person != '.jpg'][0]
                    label = labels.get(person_name, "Unknown")

                color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
                cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv.putText(frame, label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            cv.imshow('Face detector', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv.destroyAllWindows()
