import cv2
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
from super_gradients.training import models


def main():
    """
    Основная функция для захвата видео, обнаружения объектов и отслеживания их с использованием модели YOLO и алгоритма DeepSort.

    Процесс включает в себя следующие этапы:
    1. Захват видео из файла "input/input.mp4".
    2. Получение свойств видео, таких как ширина, высота и частота кадров.
    3. Инициализация записи выходного видео в файл "output/output.mp4".
    4. Настройка трекера DeepSort для отслеживания объектов.
    5. Загрузка модели YOLO для обнаружения объектов с использованием предобученных весов на наборе данных COCO.
    6. Чтение и обработка кадров видео в цикле:
       - Обнаружение объектов на текущем кадре.
       - Формирование списка результатов с ограничивающими рамками, уверенностью и классами объектов.
       - Обновление трекеров с использованием полученных данных.
       - Отрисовка ограничивающих рамок и текстовой информации на кадре.
       - Запись обработанного кадра в выходное видео.
       - Отображение текущего кадра в окне.
    7. Завершение работы: освобождение ресурсов видео и закрытие всех окон.

    Нажмите 'q' для выхода из процесса обработки видео.
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
        concate = [sublist + [element] for sublist, element in zip(bboxes_xyxy, confidence)]
        final_prediction = [sublist + [element] for sublist, element in zip(concate, labels)]

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

        cv2.imshow("Frame", frame)
        writer.write(frame)

        if cv2.waitKey(1) == ord("q"):
            break

    video_cap.release()
    writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
