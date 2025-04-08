from ultralytics import YOLO
import numpy as np
model = YOLO('./models/runs/classify/train4/weights/best.pt')
results = model('./test/Aadhar/WhatsApp Image 2025-02-22 at 20.27.57_cb003cdf.jpg')
names_dict = results[0].names
probs = results[0].probs.data  # Extract raw tensor data

# Convert to NumPy array (if not already)
probs = probs.numpy() if hasattr(probs, "numpy") else np.array(probs)

# Get the index of the highest probability class
predicted_class_index = np.argmax(probs)
confidence_score = probs[predicted_class_index]

# Assuming `names_dict` maps class indices to class names
print("Predicted ID Proof:" + names_dict[predicted_class_index])
print(f"Confidence Score: {confidence_score:.4f}")
