import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('best.pt')

# Open the video file
video_path = '1.gif'  # Replace with the actual path to your video file
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)
        #print(results)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
