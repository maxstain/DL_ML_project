from Imports.imports import *


# Draw boxes around the detected objects
def draw_boxes(frame, objects, labels):
    for found_object in objects:
        label = labels[found_object['class_id']]
        box = found_object['box']
        cv.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv.putText(frame, label, (box[0], box[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


# Load the model
def load_model():
    model_path = os.path.abspath(
        "C:/Projects/Python/DL_ML_project/models/ssd_mobilenet_v2_coco/frozen_inference_graph.pb")
    config_path = os.path.abspath("models/ssd_mobilenet_v2_coco/ssd_mobilenet_v2_coco.pbtxt")

    # Debugging information
    print(f"Model path: {model_path}")
    print(f"Config path: {config_path}")
    print(f"Model exists: {os.path.exists(model_path)}")
    print(f"Config exists: {os.path.exists(config_path)}")

    model = cv.dnn.readNetFromTensorflow(model_path, config_path)
    if model.empty():
        print("Error: Could not load model.")
        return None

    return model


# Load the labels
def load_labels():
    labels_path = os.path.abspath("models/ssd_mobilenet_v2_coco/coco_labels.txt")
    print(f"Labels path: {labels_path}")
    print(f"Labels file exists: {os.path.exists(labels_path)}")

    with open(labels_path) as file:
        labels = file.read().splitlines()

    return labels


# Detect objects
def detect_objects(frame, model):
    blob = cv.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    model.setInput(blob)
    output = model.forward()
    height, width = frame.shape[:2]
    objects = []

    for detection in output[0, 0]:
        class_id = int(detection[1])
        confidence = detection[2]
        if confidence > 0.5:
            box = detection[3:7] * np.array([width, height, width, height])
            (startX, startY, endX, endY) = box.astype("int")
            objects.append({'class_id': class_id, 'box': (startX, startY, endX, endY)})

    return objects


# Open the camera and detect objects in real time
def open_camera():
    model = load_model()
    if model is None:
        return

    labels = load_labels()
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

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
