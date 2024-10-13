import cv2
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
from super_gradients.training import models


def main():
    """
    Основная функция для захвата видео, обнаружения объектов и их отслеживания с использованием модели YOLO и трекера DeepSort.

    Процесс включает:
    1. Захват видео из файла "input/crowd.mp4".
    2. Получение свойств видео: ширина, высота, частота кадров.
    3. Инициализация записи обработанного видео в файл "output/output.mp4".
    4. Инициализация трекера DeepSort для отслеживания объектов.
    5. Загрузка предобученной модели YOLO для детекции объектов (COCO dataset).
    6. Цикл обработки кадров:
       - Обнаружение объектов на текущем кадре.
       - Фильтрация объектов по классу (только "person") и уверенности.
       - Обновление трекера DeepSort для каждого кадра.
       - Отрисовка ограничивающих рамок и информации об объектах.
       - Запись обработанных кадров в выходной файл.
    7. Освобождение ресурсов (закрытие файлов видео, окон OpenCV).
    """
    video_cap = cv2.VideoCapture("input/crowd.mp4")
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))

    writer = cv2.VideoWriter(
        "output/output.mp4",
        cv2.VideoWriter_fourcc(*'MP4V'),
        fps,
        (frame_width, frame_height)
    )

    tracker = DeepSort(max_age=50)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model = models.get("yolo_nas_l", pretrained_weights="coco").to(device)

    while True:
        ret, frame = video_cap.read()
        if not ret:
            break

        detect = next(iter(model.predict(frame, iou=0.5, conf=0.6)))

        bboxes_xyxy = torch.from_numpy(detect.prediction.bboxes_xyxy).tolist()
        confidence = torch.from_numpy(detect.prediction.confidence).tolist()
        labels = torch.from_numpy(detect.prediction.labels).tolist()

        # Объединение предсказаний в один список
        concat = [sublist + [element] for sublist, element in zip(bboxes_xyxy, confidence)]
        final_prediction = [sublist + [element] for sublist, element in zip(concat, labels)]

        results = []
        for data in final_prediction:
            conf = data[4]
            if int(data[5]) != 0:
                continue

            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            class_id = int(data[5])

            results.append([[xmin, ymin, xmax - xmin, ymax - ymin], conf, class_id])

        tracks = tracker.update_tracks(results, frame=frame)

        for track in tracks:
            conf = track.get_det_conf()
            if not track.is_confirmed() or conf is None:
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])

            B, G, R = 0, 0, 255
            text = f"{track_id} - {str(round(conf, 2)) if conf is not None else 'None'}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
            cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(text) * 12, y1), (B, G, R), -1)
            cv2.putText(frame, text, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        writer.write(frame)

    video_cap.release()
    writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
