from flask import Flask, request, render_template, send_file, jsonify
from ultralytics import YOLO
import os
import json
from datetime import datetime
from fpdf import FPDF


app = Flask(__name__)
UPLOAD_FOLDER = 'app/static'
HISTORY_FILE = 'history.json'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Загружаем предобученную модель (автоматически скачает при первом запуске)
model = YOLO('yolov8l.pt')

def save_to_history(filename, num_dryers):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "filename": filename,
        "detected_hair_dryers": num_dryers
    }
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = []
    data.append(entry)
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def generate_pdf_report():
    from fpdf import FPDF
    import os

    pdf = FPDF()
    pdf.add_page()

    font_path = "/usr/share/fonts/TTF/DejaVuSans.ttf"
    if os.path.exists(font_path):
        # Убираем `uni=True` — он устарел, но шрифт всё равно поддерживает Unicode
        pdf.add_font("DejaVu", "", font_path)
        pdf.set_font("DejaVu", "", 12)
        use_unicode = True
    else:
        pdf.set_font("Helvetica", "", 12)
        use_unicode = False

    title = "Отчёт по контролю фенов в парикмахерской" if use_unicode else "Hair Dryer Control Report"
    pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT", align='C')
    pdf.ln(5)

    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            history = json.load(f)
        for i, entry in enumerate(history, 1):
            timestamp = entry['timestamp'][:19]
            filename = entry['filename']
            dryers = entry['detected_hair_dryers']
            if use_unicode:
                text = f"{i}. {timestamp} | Файл: {filename} | Найдено фенов: {dryers}"
            else:
                text = f"{i}. {timestamp} | File: {filename} | Hair dryers: {dryers}"
            pdf.cell(0, 10, text, new_x="LMARGIN", new_y="NEXT")
    else:
        msg = "Нет данных" if use_unicode else "No data"
        pdf.cell(0, 10, msg, new_x="LMARGIN", new_y="NEXT")

    # ✅ Сохраняем в UPLOAD_FOLDER (т.е. в static/)
    report_path = '/home/andrew/Study/study-practice/app/static/report.pdf'
    pdf.output(report_path)
    return report_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    if 'image' not in request.files:
        return "Нет файла", 400
    file = request.files['image']
    if file.filename == '':
        return "Файл не выбран", 400

    # Сохраняем
    input_path = os.path.join(UPLOAD_FOLDER, "input.jpg")
    output_path = os.path.join(UPLOAD_FOLDER, "output.jpg")
    file.save(input_path)

    # # Детекция
    results = model(input_path)
    # Фильтруем только "hair drier" (класс 7 in COCO)
    hair_dryer_class_id = 78
    detected_boxes = []

    for box in results[0].boxes:
        if int(box.cls.item()) == hair_dryer_class_id:
            detected_boxes.append(box)

    # results = model(input_path, conf=0.001)
    # num_total = len(results[0].boxes)
    # print(f"Обнаружено объектов всего: {num_total}")
    # for box in results[0].boxes:
    #     cls_id = int(box.cls.item())
    #     cls_name = model.names[cls_id]
    #     print(f"  → Класс {cls_id}: {cls_name}")

    num_dryers = len(detected_boxes)
    # Сохраняем результат с боксами
    results[0].save(filename=output_path)

    # Логируем
    save_to_history(file.filename, num_dryers)

    return render_template('result.html', 
                          input_img='input.jpg',
                          output_img='output.jpg',
                          num_dryers=num_dryers)

@app.route('/report')
def report():
    try:
        report_path = generate_pdf_report()
        return send_file(report_path, as_attachment=True)
    except Exception as e:
        print("❌ Ошибка генерации PDF:", str(e))
        return "Не удалось создать отчёт. Проверьте наличие шрифта DejaVu и права записи.", 500

if __name__ == '__main__':
    app.run(debug=True)