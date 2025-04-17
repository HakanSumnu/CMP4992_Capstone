import cv2

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# Must set MJPG before resolution

# Attempt to set 800x448
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 448)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# Read one frame
ret, frame = cap.read()

# Check what resolution and format we really got
actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

print(f"Requested: 800x448 (MJPG)")
print(f"Actual resolution: {actual_width}x{actual_height}")
print(f"Actual FOURCC codec: {codec}")
print(f"Frame shape: {frame.shape if ret else 'No frame'}")

cap.release()