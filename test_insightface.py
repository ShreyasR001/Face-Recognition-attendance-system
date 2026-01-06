# test_insightface.py
import cv2
from insightface.app import FaceAnalysis

def main():
    print("insightface version:", __import__("insightface").__version__)

    # buffalo_l works
    app = FaceAnalysis(name='buffalo_l')

    print("Preparing models… (first run may download files)")
    # IMPORTANT — insightface 0.7.3 does NOT support 'nms' argument
    app.prepare(ctx_id=-1)

    print("Model ready!")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera.")
        return

    ret, frame = cap.read()
    if not ret:
        print("Could not read frame.")
        return

    faces = app.get(frame)
    print(f"Detected {len(faces)} faces")

    for i, f in enumerate(faces):
        print(f"Face {i}: bbox={f.bbox}, has_embedding={hasattr(f,'embedding')}")

    for f in faces:
        x1,y1,x2,y2 = f.bbox.astype(int)
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    cv2.imshow("Test", frame)
    cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
