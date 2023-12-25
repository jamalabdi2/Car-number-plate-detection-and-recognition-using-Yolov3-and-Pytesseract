import cv2
import numpy as np

# Load Yolo
net = cv2.dnn.readNet("yolov3_training_final.weights", "yolov3_training (1).cfg")

# Name custom object
classes = ["number_plate"]

# Video filepath
video_path = "/Users/jamal/Documents/custom_yolov3/videos/IMG_2651.mp4"

# Open video capture
cap = cv2.VideoCapture(video_path)

# Get original video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Reduce height and width by half
new_width = int(width / 2)
new_height = int(height / 2)

# Define codec and create VideoWriter object (use 'avc1' for QuickTime compatibility)
fourcc = cv2.VideoWriter_fourcc(*'avc1')
output_video = cv2.VideoWriter('output_video_2.mp4', fourcc, fps, (new_width, new_height))

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = (255, 0, 0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame
    frame = cv2.resize(frame, (new_width, new_height))
    original_frame = frame.copy()

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing information on the screen
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                # Object detected
                center_x = int(detection[0] * new_width)
                center_y = int(detection[1] * new_height)
                w = int(detection[2] * new_width)
                h = int(detection[3] * new_height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y + 30), font, 3, color, 2)

    # Save the frame to the output video
    output_video.write(frame)

    # Display the resulting frame
    cv2.imshow("Video", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture, release VideoWriter, and close all windows
cap.release()
output_video.release()
cv2.destroyAllWindows()
