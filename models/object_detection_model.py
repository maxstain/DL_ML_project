# models/object_detection_model.py
from functions.general_functions import *
from Imports.imports import *


class ObjectDetectionModel:
    def __init__(self, labels_path):
        self.model_path = create_obj_det_model()
        self.labels_path = labels_path
        self.model = None
        self.labels = None

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model path {self.model_path} does not exist.")
        if not os.path.exists(os.path.join(self.model_path, 'saved_model.pb')):
            raise FileNotFoundError(f"SavedModel file does not exist at {self.model_path}/saved_model.pb")
        if not os.path.exists(os.path.join(self.model_path, 'variables')):
            raise FileNotFoundError(f"Variables folder does not exist at {self.model_path}/variables")

        self.model = tf.saved_model.load(self.model_path)
        with open(self.labels_path, 'r') as f:
            self.labels = f.read().strip().split('\n')

    def detect_objects(self, frame):
        input_tensor = tf.convert_to_tensor(frame)
        input_tensor = input_tensor[tf.newaxis, ...]
        detections = self.model(input_tensor)
        return detections

    def draw_boxes(self, frame, detections):
        h, w = frame.shape[:2]
        for detection in detections['detection_boxes'][0]:
            confidence = detection[2]
            if confidence > CONFIDENCE_THRESHOLD:
                class_id = int(detection[1])
                label = self.labels[class_id]
                box = detection[0] * np.array([h, w, h, w])
                (y, x, y2, x2) = box.astype('int')
                cv.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
                cv.putText(frame, label, (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def process_video(self):
        cap = cv.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            detections = self.detect_objects(frame)
            self.draw_boxes(frame, detections)
            cv.imshow('Object Detection', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv.destroyAllWindows()
