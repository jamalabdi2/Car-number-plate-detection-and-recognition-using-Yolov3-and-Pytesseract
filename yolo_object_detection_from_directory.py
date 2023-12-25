import cv2
import numpy as np
import glob
import os

# Load Yolo
net = cv2.dnn.readNet("yolov3_training_final.weights", "yolov3_training (1).cfg")

classes = ["number_plate"]

images_path = glob.glob("/Users/jamal/Documents/custom_yolov3/images/*.jpg") + glob.glob(
    "/Users/jamal/Documents/custom_yolov3/images/*.jpeg"
) + glob.glob("/Users/jamal/Documents/custom_yolov3/images/*.png") + glob.glob("/Users/jamal/Documents/custom_yolov3/images/*.JPG")
print("image_path", images_path)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = (0, 255, 0) 

# Create a folder to store cropped images
output_folder = "cropped_images"
os.makedirs(output_folder, exist_ok=True)

for img_path in images_path:
    # Loading image
    print("img_path", img_path)
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

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
                print(class_id)
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    print(indexes)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            color = colors 
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

            # Crop the detected region
            cropped_img = img[y : y + h, x : x + w]

            # Save the cropped image
            output_path = os.path.join(output_folder, f"cropped_{os.path.basename(img_path)}_{i}.jpg")
            cv2.imwrite(output_path, cropped_img)

    cv2.imshow("Image", img)
    key = cv2.waitKey(0)

cv2.destroyAllWindows()
