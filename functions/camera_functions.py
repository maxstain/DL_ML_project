from Imports.imports import *

# Constants for file paths
MODEL_PATH = os.path.abspath("C:/Projects/Python/DL_ML_project/models/ssd_mobilenet_v2_coco/frozen_inference_graph.pb")
CONFIG_PATH = os.path.abspath(
    "C:/Projects/Python/DL_ML_project/models/ssd_mobilenet_v2_coco/ssd_mobilenet_v2_coco.pbtxt")
LABELS_PATH = os.path.abspath("C:/Projects/Python/DL_ML_project/models/ssd_mobilenet_v2_coco/coco_labels.txt")
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


def load_model(model_path, config_path):
    """Load the SSD MobileNet model from TensorFlow."""
    if not os.path.exists(model_path) or not os.path.exists(config_path):
        raise FileNotFoundError("Model or configuration file not found.")

    model = cv.dnn.readNetFromTensorflow(model_path, config_path)
    if model.empty():
        raise ValueError("Error: Could not load model.")

    return model


def load_labels(labels_path):
    """Load the labels for the COCO dataset."""
    if not os.path.exists(labels_path):
        raise FileNotFoundError("Labels file not found.")

    with open(labels_path) as file:
        labels = file.read().splitlines()

    return labels


def detect_objects(frame, model):
    """Detect objects in a frame using the loaded model."""
    blob = cv.dnn.blobFromImage(frame, scalefactor=0.007843, size=(300, 300), mean=127.5)
    model.setInput(blob)
    output = model.forward()
    height, width = frame.shape[:2]
    objects = []

    for detection in output[0, 0]:
        confidence = detection[2]
        if confidence > CONFIDENCE_THRESHOLD:
            class_id = int(detection[1])
            box = detection[3:7] * np.array([width, height, width, height])
            (startX, startY, endX, endY) = box.astype("int")
            objects.append({'class_id': class_id, 'box': (startX, startY, endX, endY)})

    return objects


def draw_boxes(frame, objects, labels):
    """Draw bounding boxes and labels on detected objects."""
    for found_object in objects:
        label = labels[found_object['class_id']]
        box = found_object['box']
        cv.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv.putText(frame, label, (box[0], box[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


def open_camera(model, labels):
    """Open the camera and perform real-time object detection."""
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Error: Could not open camera.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        objects = detect_objects(frame, model)
        draw_boxes(frame, objects, labels)
        cv.imshow("Camera", frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
