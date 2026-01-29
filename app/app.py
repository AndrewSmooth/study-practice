from flask import Flask, request, render_template, send_file, jsonify
from ultralytics import YOLO
import os
import json
from datetime import datetime
from fpdf import FPDF

app = Flask(__name__)
UPLOAD_FOLDER = 'static'
HISTORY_FILE = 'history.json'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Загружаем предобученную модель (автоматически скачает при первом запуске)
model = YOLO('yolov8n.pt')

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
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    
    pdf.cell(0, 10, "Отчёт по контролю фенов в парикмахерской", ln=True, align='C')
    pdf.ln(10)

    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            history = json.load(f)
        for i, entry in enumerate(history, 1):
            text = f"{i}. {entry['timestamp'][:19]} | Файл: {entry['filename']} | Найдено фенов: {entry['detected_hair_dryers']}"
            pdf.cell(0, 10, text, ln=True)
    else:
        pdf.cell(0, 10, "Нет данных", ln=True)

    report_path = "static/report.pdf"
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

    # Детекция
    results = model(input_path)
    # Фильтруем только "hair drier" (класс 7 in COCO)
    hair_dryer_class_id = 7
    detected_boxes = []

    for box in results[0].boxes:
        if int(box.cls.item()) == hair_dryer_class_id:
            detected_boxes.append(box)

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
    report_path = generate_pdf_report()
    return send_file(report_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)