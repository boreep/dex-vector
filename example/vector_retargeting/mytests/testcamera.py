import cv2

def test_camera(device_id):
    cap = cv2.VideoCapture(device_id)
    if not cap.isOpened():
        print(f"Device {device_id} cannot be opened.")
        return

    print(f"Testing device {device_id}... Press 'q' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Device {device_id} failed to capture frame.")
            break
        cv2.imshow(f"Device {device_id}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

for i in range(8):  # Test /dev/video0 to /dev/video7
    test_camera(i)
