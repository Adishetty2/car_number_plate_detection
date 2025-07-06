
import cv2
import os

# Path to the cascade file (make sure the .xml file is really there)
CASCADE_PATH = "model/haarcascade_russian_plate_number.xml"

if not os.path.exists(CASCADE_PATH):
    raise FileNotFoundError(f"Could not find cascade at: {CASCADE_PATH}")

plate_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if plate_cascade.empty():
    raise IOError("Failed to load Haar cascade (file is corrupt or wrong path).")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("❌ Could not open webcam. Is it in use by another app?")

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

SAVE_DIR = "plates"
os.makedirs(SAVE_DIR, exist_ok=True)   # create it if missing

min_area = 500          # ignore tiny detections
img_counter = 0         # running index for saved crops

print("[INFO] Press  S  to save a detected plate,  Q  to quit.")

while True:
    ok, frame = cap.read()
    if not ok or frame is None:
        print("❌ Failed to grab frame. Exiting.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(gray,
                                            scaleFactor=1.1,
                                            minNeighbors=4)

    plate_roi = None  # will hold last ROI (if any)
    for (x, y, w, h) in plates:
        if w * h < min_area:
            continue                           # too small → skip
        cv2.rectangle(frame, (x, y), (x + w, y + h),
                      color=(0, 255, 0), thickness=2)
        cv2.putText(frame, "Number Plate", (x, y - 5),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                    (255, 0, 255), 2)

        plate_roi = frame[y:y + h, x:x + w]    # keep the last good ROI
        cv2.imshow("ROI", plate_roi)

    cv2.imshow("Result", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    if key == ord('s') and plate_roi is not None:
        fname = os.path.join(SAVE_DIR, f"scanned_{img_counter:03d}.jpg")
        cv2.imwrite(fname, plate_roi)
        img_counter += 1
        print(f"[INFO] Saved {fname}")
        # quick visual feedback
        cv2.rectangle(frame, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, "Plate Saved", (140, 265),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
        cv2.imshow("Result", frame)
        cv2.waitKey(400)

cap.release()
cv2.destroyAllWindows()

