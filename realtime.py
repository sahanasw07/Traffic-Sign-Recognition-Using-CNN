import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("model.h5")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (32, 32))
    img = img / 255.0
    img = img.reshape(1, 32, 32, 3)

    prediction = model.predict(img)
    class_id = np.argmax(prediction)

    cv2.putText(frame, f"Class: {class_id}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0,255,0), 2)

    cv2.imshow("Traffic Sign Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
