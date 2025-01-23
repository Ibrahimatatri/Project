
import argparse
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Helper function to process the image
def process_image(image_path, target_size):
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image) / 255.0
    return np.expand_dims(image, axis=0)

def load_class_names(json_path):
    with open(json_path, 'r') as f:
        class_names = json.load(f)
    return class_names

def predict(image_path, model_path, top_k, class_names_path):
    # Load the model
    model = load_model(model_path)

    # Process the image
    image = process_image(image_path, target_size=(224, 224))

    # Make predictions
    predictions = model.predict(image)[0]

    # Get the top K predictions
    top_indices = predictions.argsort()[-top_k:][::-1]
    top_probs = predictions[top_indices]

    # Map indices to class names if JSON is provided
    if class_names_path:
        class_names = load_class_names(class_names_path)
        top_classes = [class_names[str(index)] for index in top_indices]
    else:
        top_classes = [str(index) for index in top_indices]

    # Print results
    print("Top Predictions:")
    for i in range(top_k):
        print(f"Class: {top_classes[i]} - Probability: {top_probs[i]:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict the class of an image using a pre-trained Keras model.")
    
    # Required arguments
    parser.add_argument('image_path', type=str, help="Path to the image file.")
    parser.add_argument('model_path', type=str, help="Path to the saved Keras model.")

    # Optional arguments
    parser.add_argument('--top_k', type=int, default=1, help="Number of top predictions to display. Default is 1.")
    parser.add_argument('--class_names_path', type=str, help="Path to a JSON file mapping class indices to names.")

    args = parser.parse_args()

    # Run prediction
    predict(args.image_path, args.model_path, args.top_k, args.class_names_path)
