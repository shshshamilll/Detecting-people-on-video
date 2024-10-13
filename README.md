# Тестовое задание на позицию Junior Data Science  
## Описание проекта
Данный проект осуществляет детекцию людей и их отрисовку на видео.  
Трекинг объектов реализован с использованием YOLO и DeepSORT.  
В качестве детектора используется модель YOLO (You Only Look Once) под названием yolo_nas_l, т. к. она обеспечивает хороший баланс между производительностью и точностью.   
Алгоритмом трекинга выбран однин из самых популярных алгоритмов трекинга DeepSORT.
## Установка
```
git clone https://github.com/shshshamilll/Detecting-people-on-video.git
cd Detecting-people-on-video
pip install -r requirements.txt
mkdir output
```
## Запуск
```
python main.py
```
## Отображение результатов
```
python show_results.py
```
