import cv2

video_path = 'output/output.mp4'
video = cv2.VideoCapture(video_path)

if not video.isOpened():
    print("Ошибка: не удается открыть видеофайл")
    exit()

while True:
    ret, frame = video.read()

    if not ret:
        print("Видео закончилось")
        break

    cv2.imshow('Video', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
