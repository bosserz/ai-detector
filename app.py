from flask import Flask, request, jsonify
from PIL import Image
import io
import openai
import numpy as np
import cv2

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    img = Image.open(file.stream)

    # Process the image and identify brands
    brands = identify_brands(img)

    return jsonify({'brands': brands})

def identify_brands(image):
    processed_image = preprocess_image(image)

    # Assuming you have some method to extract text or labels from the processed image
    extracted_texts = extract_text_from_image(processed_image)

    # Use GPT-4 to identify brands from extracted texts
    brands = []
    for text in extracted_texts:
        response = openai.Completion.create(
            model="gpt-4",
            prompt=f"Identify the brand from this label: {text}",
            max_tokens=10
        )
        brands.append(response.choices[0].text.strip())

    return brands

def preprocess_image(image):
    # Convert PIL image to OpenCV format
    image = np.array(image)
    # Convert RGB to BGR
    image = image[:, :, ::-1].copy()

    # Your image processing code here
    # e.g., object detection to find product labels

    return image

def extract_text_from_image(image):
    # Placeholder for text extraction logic
    # Use OCR or other techniques to extract text from image
    return ["Sample Text 1", "Sample Text 2"]

if __name__ == '__main__':
    app.run(debug=True)
