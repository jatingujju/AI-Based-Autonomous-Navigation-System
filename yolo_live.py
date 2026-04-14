import cv2
import numpy as np

# -----------------------------
# LOAD YOLO
# -----------------------------
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load class names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get output layers
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Colors for objects
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# -----------------------------
# START CAMERA
# -----------------------------
cap = cv2.VideoCapture(0)

print("🚀 Starting YOLO Object Detection...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # -----------------------------
    # PREPROCESS FRAME
    # -----------------------------
    blob = cv2.dnn.blobFromImage(
        frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False
    )

    net.setInput(blob)
    outs = net.forward(output_layers)

    # -----------------------------
    # DETECTION
    # -----------------------------
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # -----------------------------
    # REMOVE DUPLICATES (NMS)
    # -----------------------------
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # -----------------------------
    # DRAW BOXES + NAVIGATION
    # -----------------------------
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]

            # Draw box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # -----------------------------
            # 🚀 NAVIGATION LOGIC
            # -----------------------------
            center_x = x + w // 2

            if center_x < width // 3:
                direction = "MOVE RIGHT ➡️"
            elif center_x > 2 * width // 3:
                direction = "MOVE LEFT ⬅️"
            else:
                direction = "STOP ⛔"

            print("Decision:", direction)

            cv2.putText(frame, direction, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # -----------------------------
    # SHOW OUTPUT
    # -----------------------------
    cv2.imshow("YOLO Detection", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC key
        break

# -----------------------------
# CLEANUP
# -----------------------------
cap.release()
cv2.destroyAllWindows()