# Car Number Plate Detection and Recognition using YOLOv3 and Pytesseract

I conducted this research project for my bachelor's thesis, implementing an Automatic Number Plate Detection and Recognition system using YOLOv3 and Pytesseract.

The custom YOLOv3 model was trained specifically for car number plates and utilized as a detection model to identify the location of number plates on cars.

Once the number plate is detected, the image is cropped, and various image processing steps are performed using OpenCV.

The processed image is then passed through Pytesseract OCR to extract the text from the number plate.

## Overview of the project

#![whole_project_overview_](https://github.com/jamalabdi2/Car-number-plate-detection-and-recognition-using-Yolov3-and-Pytesseract/assets/113813239/6cf97cc1-efce-4a0e-9bf7-79c424b6b495)


## Technology used

| Development Tools  | Description                                               |
|---------------------|-----------------------------------------------------------|
| Yolov3              | Deep learning algorithm for real-time object detection.   |
| OpenCV              | Computer vision library for image and video processing.   |
| Python              | Programming language used for the development of algorithms. |
| Flask               | Web framework for creating web applications with Python.  |
| Pytesseract OCR    | Optical Character Recognition (OCR) tool for text.         |
| Visual Studio Code | An IDE for coding.                                         |
| Google Colab        | Cloud-based platform for collaborative coding in Python.  |


##  License Plate Detection

The images below showcase successful license plate detection on vehicles.

### Single Car
Figure 1: Single Car

<img width="312" alt="image" src="https://github.com/jamalabdi2/Car-number-plate-detection-and-recognition-using-Yolov3-and-Pytesseract/assets/113813239/4214bafc-f636-4b70-8fc8-4753dbcfd888">

### Multiple Cars

Figure 2: Multiple Cars

<img width="385" alt="image" src="https://github.com/jamalabdi2/Car-number-plate-detection-and-recognition-using-Yolov3-and-Pytesseract/assets/113813239/f40e5e11-4849-4ec5-9308-90396fb57d84">

These images demonstrate successful detection in scenarios with both single and multiple cars, showcasing the versatility of the detection system.

## ROI Extraction and Post-Processing


After license plate detection, a set of preprocessing steps is performed on the extracted license plate to enhance recognition accuracy. The key techniques include:

1. Grayscale Conversion: Simplifying subsequent processing steps and reducing computational complexity by converting the cropped RGB image to grayscale.

2. Gaussian Blur: Applying a 7x7 filter size for Gaussian blur to suppress noise and enhance license plate features.

3. Color Inversion: Enhancing the contrast between the license plate and background through color inversion using bitwise not operation.

4. Binarization: Deriving a binary image from the inverted grayscale image using thresholding (threshold value of 100), accentuating license plate features, and suppressing background noise.

5. Morphological Dilation: Utilizing a 3x3 rectangular kernel in the dilation process to enhance license plate boundaries. This improves contour identification, contributing to a more robust model in various scenarios, including varying angles, lighting conditions, and image noise.

The figure below illustrates the sequence of image processing steps.

<img width="378" alt="image" src="https://github.com/jamalabdi2/Car-number-plate-detection-and-recognition-using-Yolov3-and-Pytesseract/assets/113813239/5c57b3be-de3d-4482-8ed5-998de0b34c10">


## Detection results from various environmental challenges.

Model Performance in different environmental conditions.

<img width="377" alt="image" src="https://github.com/jamalabdi2/Car-number-plate-detection-and-recognition-using-Yolov3-and-Pytesseract/assets/113813239/4afc073d-27f4-49d5-8e75-7718ee3b54a9">

## License Plate Recognition using OCR.

In the OCR phase, Pytesseract is used on preprocessed images to extract alphanumeric characters from the cropped License Plates as shown in Figure below. This is achieved by using the ‘image_to_string()’ function, which retrieves the text. 

For Korean characters, the language parameter is set to ‘Hangul’ .
Post-OCR, the text undergoes post-processing to correct any mistakes, which includes spellchecking and validation against known license plate formats, as well as handling unique characters or symbols found in Korean plates. The final step is to output the recognized text for further use or storage. 
The effectiveness of Pytesseract in recognizing Korean license plates can be significantly influenced by the image's quality, the preprocessing methods, and the OCR engine's training. 

Figure: Text extraction using OCR

<img width="378" alt="image" src="https://github.com/jamalabdi2/Car-number-plate-detection-and-recognition-using-Yolov3-and-Pytesseract/assets/113813239/5277c253-096f-44cc-a946-4b3847841ffd">


## Real-time Licence Plate Detection From Video 

In this section, witness real-time license plate detection in action through the provided video demonstration. 
The YOLOv3 model dynamically identifies and highlights license plates in the video stream. 

https://github.com/jamalabdi2/Car-number-plate-detection-and-recognition-using-Yolov3-and-Pytesseract/assets/113813239/e4177189-e3c6-4ab4-a26a-661183b520b9

## Real-time Licence Plate Detection From Webcam

I've implemented a live license plate detection system using a webcam, and the real-time feed is streamed to a browser using Flask. During testing, I used car pictures on my phone, bringing the phone in front of the webcam, and the system successfully detected license plates. Below are the results of the detection from the webcam:

<img width="798" alt="Screenshot 2023-12-23 at 7 03 12 PM" src="https://github.com/jamalabdi2/Car-number-plate-detection-and-recognition-using-Yolov3-and-Pytesseract/assets/113813239/f5c08982-edac-4b99-b8dc-25c9013100c7">
<img width="698" alt="Screenshot 2023-12-23 at 7 05 39 PM" src="https://github.com/jamalabdi2/Car-number-plate-detection-and-recognition-using-Yolov3-and-Pytesseract/assets/113813239/81869ba0-3da7-408d-85ce-62fcdeb5e297">



## Web Interface Using Flask

I have crafted the interface using Python, HTML, CSS, and Flask, with a primary focus on achieving a clean, simple, and user-friendly design. The homepage interface starts with a brief project introduction, as illustrated in Figure 15 and Figure 16.

Users can upload an image, and YOLOv3, based on the uploaded image, performs license plate detection and subsequently crops the license plate from the image. We utilize OpenCV to execute various post-processing steps aimed at enhancing the visibility and accuracy of the detected license plate.

The figure below shows the web interface:

<img width="353" alt="image" src="https://github.com/jamalabdi2/Car-number-plate-detection-and-recognition-using-Yolov3-and-Pytesseract/assets/113813239/7472bbf6-5e28-4640-88e4-bcdec35fe123">

<img width="320" alt="image" src="https://github.com/jamalabdi2/Car-number-plate-detection-and-recognition-using-Yolov3-and-Pytesseract/assets/113813239/421ddea9-1438-4b66-b1d2-49b14fff8c50">

## Future Work

To enhance the system, future efforts will concentrate on:

1. Improving character recognition through the implementation of single-character detection using deep learning.
2. Employing advanced techniques like CRNN and LSTM to bolster the system's robustness and achieve more accurate character recognition.
3. Exploring the integration of the system into mobile devices such as smartphones and tablets to leverage mobile environments.


While initially designed for Korean car number plates, the project's robust architecture enables effective functionality with number plates from diverse regions and countries.


Feel free to reach out to me via email if you have any inquiries about this project or if you're interested in collaborating:

Email: abdijumale2@gmail.com

Looking forward to hearing from you!





