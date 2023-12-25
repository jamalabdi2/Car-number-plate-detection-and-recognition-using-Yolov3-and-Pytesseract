import pytesseract
from PIL import Image
import matplotlib.pyplot as plt



def extract_text(img_path):
    # Set the Tesseract executable path
    pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'


    image = Image.open(img_path)

    # Use Tesseract to extract text
    text = pytesseract.image_to_string(image, lang='kor')

    # Print the extracted text
    print("Extracted Text:", text)
