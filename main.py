import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

def predict_completion(model_path, image_path):
    # Load the trained model (compile=False to avoid 'mse' error)
    model = load_model(model_path, compile=False)
    
    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # normalize to [0,1]
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

    # Run prediction: output is [predicted_storey, predicted_percent]
    pred = model.predict(img_array)[0]
    pred_storey = int(round(pred[0]))
    pred_percent = pred[1]
    return pred_storey, pred_percent

if __name__ == "__main__":
    selected = int(input("Select storey (1, 2, or 3): ").strip())
    image_path = input("Enter the path to your building image: ").strip()

    model_path = "construction_model.h5"
    pred_storey, pred_percent = predict_completion(model_path, image_path)

    print(f"\n🏗️  Model predicts this is a {pred_storey}-storey building.")
    print(f"🔢 Predicted construction completion: {pred_percent:.2f}%")

    if pred_storey != selected:
        print("⚠️  Warning: Your selected storey doesn’t match the model’s prediction.")
