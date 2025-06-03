import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

def predict_completion(model_path, image_path):
    # Load the trained model
    model = load_model(model_path, compile=False)


    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize to [0,1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Run prediction: two outputs [storey prediction, percent prediction]
    storey_pred, percent_pred = model.predict(img_array)

    # storey_pred is an array of class probabilities (softmax output)
    pred_storey = np.argmax(storey_pred[0]) + 1  # class 0 -> 1 storey, class 1 -> 2 storey, class 2 -> 3 storey

    # percent_pred is a direct regression output
    pred_percent = percent_pred[0][0]

    return pred_storey, pred_percent

if __name__ == "__main__":
    # 1. Ask user for their selected storey
    selected = int(input("Select storey (1, 2, or 3): ").strip())

    # 2. Ask user for the path to their building image
    image_path = input("Enter the path to your building image: ").strip()

    # 3. Predict
    model_path = "construction_model.h5"
    pred_storey, pred_percent = predict_completion(model_path, image_path)

    # 4. Display results
    print(f"\n🏗️  Model predicts this is a {pred_storey}-storey building.")
    print(f"🔢 Predicted construction completion: {pred_percent:.2f}%")

    # 5. Compare with user selection
    if pred_storey != selected:
        print("⚠️  Warning: Your selected storey doesn’t match the model’s prediction.")
