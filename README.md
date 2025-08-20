# Object detection (amount, titles, positions of tables) in 1-page PDF file



### Загальний опис проекту:
Десктопний застосунок для виявлення та локалізації таблиць у 1-сторінковому PDF файлі за допомогою горизонтальних обмежувальних рамок.



### Функціонал проекту: 
- виявляє та локалізує таблиці за допомогою горизонтальних обмежувальних рамок (HBB);
- знаходить та помічає маркерами центри таблиць;
- підраховує кількість таблиць;
- для кожної з таблиць намагається сформувати чи згенерувати заголовок на основі даних;
- визначає (x, y) координати центру таблиці (вираховуються починаючи з лівого верхнього кута); 
- візуалізує результати роботи та повертає JSON об'єкт.



### Обмеження підходу реалізації проекту:
- детектування таблиць можливе лише для файлів з розширенням .pdf;
- PDF файл з таблицями повинен мати лише 1 сторінку;
- алгоритми детектування працюють дуже добре для 1 сторінкових PDF файлів великих/середніх розмірів (для великих/середніх таблиць) та гірше для малих 1 сторінкових PDF файлів (таблиць);
- алгоритми детектування налаштовані під структуровані таблиці з видимими лініями рядків/стовпців та щільними рядками/колонками; для слабко структурованих макетів можливі пропуски;
- алгоритми генерації заголовків працюють дуже добре для 1 сторінкових PDF файлів, де текст можливо виділити курсором (гірше - де дані таблиці представлені у вигляді зображення) та чутливі до якості рендеру (DPI), артефактів, шрифтів розмірів тексту тощо;
- через застосування великої кількості алгоритмів-обробників працює інколи повільно (але зазвичай швидко);



### Структура проекту:
```
Object detection (amount, titles, positions of tables)/
├─ Desktop_application/
|  ├─ README.md                                             # Опис проекту
│  ├─ dockerfile                                            # Автоматичне встановлення та налаштування проекту
│  ├─ docker-compose.yml                                    # Конфігурація докер-файлу
│  ├─ requirements.txt                                      # Бібліотеки/фреймворки для встановлення
│  ├─ table_detector_in_PDF_file_icon.png                   # Іконка застосунку
│  ├─ desktop_application.py                                # Точка входу в програму
│  ├─ controller.py                                         # Логіка взаємодії графічного інтерфейсу користувача з детектором таблиць
│  ├─ graphical_user_interface.py                           # Графічний інтерфейс користувача
│  └─ object_detector.py                                    # Детектор та генератор заголовків таблиць
├─ Files_with_tables/                                       # Приклади PDF/PNG файлів з таблицями
│  ├─ example_1.pdf 
│  ├─ example_2.pdf 
│  ├─ example_3.pdf 
│  ├─ example_4.pdf 
│  ├─ example_5.pdf 
│  └─ example_6.png
├─ Implemented_object_detection_solution.ipynb              # Розроблене рішення детектування таблиць
└─ Demonstration of table detection in 1-page PDF files.mkv # Демонстрація роботи
```



### Застосований інструментарій:
- пайплайн виявлення та локалізації таблиць поєднує класичні методи OpenCV (морфологія, проекційні профілі) та аналіз тексту сторінки через PyMuPDF;
- пайплайн формування та генерації заголовків складається з каскаду OCR рушіїв (PaddleOCR → docTR → Tesseract) та генератора короткого заголовку таблиці за допомогою моделі Flan-T5 (transformers).



### Бібліотеки/фреймворки/програми:
- Docker Desktop;
- X-server для Windows (наприклад, VcXsrv);
- Python 3.11.9
    - Tkinter
    - matplotlib
    - numpy
    - opencv-python
    - pymupdf
    - pytesseract
    - paddlepaddle
    - paddleocr
    - pillow
    - python-doctr
    - git+https://github.com/huggingface/transformers.git
    - torch==2.4.1
    - torchvision==0.19.1
    - tesseract-ocr



### Запуск на локальній системі (операційна система Windows):
- `cd "..\Object detection (amount, titles, positions of tables)\Desktop_application"`
- `py -3.11 -m venv .venv`
- `.venv\Scripts\activate.bat`
- `pip install -r requirements.txt`
- `python desktop_application.py`



### Запуск в Docker контейнері:
- `cd "..\Object detection (amount, titles, positions of tables)\Desktop_application"`
- `docker compose build --no-cache`
- `docker compose up`



### Демонстрація роботи проекту:
Відеозапис з такою назвою доступний в репозиторії: `Demonstration of table detection in 1-page PDF files.mkv`
