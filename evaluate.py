import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanSquaredError
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import mean_squared_error, r2_score

# 1) Paths
MODEL_PATH = 'construction_model.h5'
TEST_ROOT  = 'test'     # e.g. test/1_storey/20/*.png etc.
TARGET_SIZE = (128, 128)

# 2) Load model (skip compile, register mse)
model = load_model(
    MODEL_PATH,
    custom_objects={'mse': MeanSquaredError()},
    compile=False
)

y_true = []
y_pred = []

# 3) Walk through storey → percent → files
for storey_folder in os.listdir(TEST_ROOT):
    storey_path = os.path.join(TEST_ROOT, storey_folder)
    if not os.path.isdir(storey_path):
        continue

    for percent_folder in os.listdir(storey_path):
        percent_path = os.path.join(storey_path, percent_folder)
        if not os.path.isdir(percent_path):
            continue

        # Ground truth percentage
        try:
            true_pct = float(percent_folder)
        except ValueError:
            continue

        for fname in os.listdir(percent_path):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            img_path = os.path.join(percent_path, fname)
            try:
                # load & resize
                img = load_img(img_path, target_size=TARGET_SIZE)
                arr = img_to_array(img) / 255.0
                arr = np.expand_dims(arr, axis=0)  # shape (1,128,128,3)

                # predict — returns a list [storey_preds, percent_preds] for multi-output
                outputs = model.predict(arr, verbose=0)

                if isinstance(outputs, list) and len(outputs) >= 2:
                    # second head is percent regression
                    percent_preds = outputs[1]           # shape (1,1)
                    pct_pred = float(percent_preds[0][0])
                else:
                    # single-output fallback
                    out = outputs if not isinstance(outputs, list) else outputs[0]
                    pct_pred = float(out[0][0])

                y_true.append(true_pct)
                y_pred.append(pct_pred)

                print(f"✓ {storey_folder}/{percent_folder}/{fname} → True {true_pct}, Pred {pct_pred:.1f}")

            except Exception as e:
                print(f"✗ Skipped {fname}: {e}")

# 4) Compute metrics
if not y_true:
    print("❌ No images processed. Check TEST_ROOT structure and filenames.")
else:
    mse = mean_squared_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)
    print(f"\n✅ Evaluated {len(y_true)} images")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R² Score:            {r2:.2f}")
