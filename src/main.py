import os
import sys
import tkinter as tk
from tkinter import ttk

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, '..')))

from data.data_processor import PropertyProcessor
from data.label_encoder import LabelEncoder
from app_logic import load_trained_models

INPUT_CSV = os.path.join(BASE_DIR, '..', 'data', 'estates_raw_data.csv')
PROCESSED_CSV = os.path.join(BASE_DIR, '..', 'data', 'final_training_data.csv')

model_rf = None
model_knn = None
district_encoder = None
construction_encoder = None


def calculate_price():
    global model_rf, model_knn
    try:
        rooms = int(rooms_entry.get())
        area = float(area_entry.get())
        floor = int(floor_entry.get())
        total_floors = int(total_floors_entry.get())
        year = int(year_entry.get())

        dist_val = district_encoder.transform([district_var.get()])[0]
        const_val = construction_encoder.transform([construction_var.get()])[0]

        is_first = 1 if floor == 1 else 0
        is_last = 1 if (total_floors > 0 and floor == total_floors) else 0

        features_list = [
            rooms, area, floor, total_floors, year,
            is_first, is_last,
            1 if garage_var.get() else 0,
            1 if closed_complex_var.get() else 0,
            dist_val, const_val,
            1 if gas_var.get() else 0,
            1 if tep_var.get() else 0,
            1 if luxury_var.get() else 0,
            1 if act16_var.get() else 0
        ]

        res_rf = model_rf.predict([features_list])
        pred_rf = res_rf[0] if isinstance(res_rf, list) else res_rf

        res_knn = model_knn.predict([features_list])[0]

        final_price = 0.3 * res_knn + 0.7 * pred_rf

        result_label.config(text=f"{final_price:,.0f} €", fg="#27ae60")
        print(f"Prediction: RF={pred_rf:,.0f} | KNN={res_knn:,.0f} | Final={final_price:,.0f}")

    except Exception as e:
        print(f"Error in calculation: {e}")
        result_label.config(text="Грешка в данните", fg="#e74c3c")


def start_app():
    global model_rf, model_knn, district_encoder, construction_encoder
    global rooms_entry, area_entry, floor_entry, total_floors_entry, year_entry
    global district_var, construction_var, garage_var, closed_complex_var
    global gas_var, tep_var, luxury_var, act16_var, result_label

    processor = PropertyProcessor()
    if not os.path.exists(PROCESSED_CSV):
        print("Initial data processing...")
        processor.process_data(INPUT_CSV, PROCESSED_CSV)

    model_rf, model_knn = load_trained_models(PROCESSED_CSV)

    district_encoder = LabelEncoder().load(os.path.join(BASE_DIR, '..', 'data', 'District_encoder.json'))
    construction_encoder = LabelEncoder().load(os.path.join(BASE_DIR, '..', 'data', 'Construction_Type_encoder.json'))

    root = tk.Tk()
    root.title("Интелигентна Оценка на Имоти")
    root.geometry("550x850")
    root.configure(bg="#f0f2f5")

    style = ttk.Style()
    style.theme_use('clam')
    main_container = tk.Frame(root, bg="#f0f2f5", padx=30, pady=20)
    main_container.pack(expand=True, fill="both")
    tk.Label(main_container, text="Пазарна Оценка на Имот", font=("Helvetica", 18, "bold"),
             bg="#f0f2f5", fg="#1c1e21").pack(pady=(0, 20))

    basic_frame = tk.LabelFrame(main_container, text=" Основни Характеристики ", bg="white", padx=15, pady=15)
    basic_frame.pack(fill="x", pady=10)

    def create_labeled_entry(parent, label, row, col):
        tk.Label(parent, text=label, bg="white").grid(row=row, column=col, sticky="w", padx=5)
        entry = tk.Entry(parent, width=12)
        entry.grid(row=row + 1, column=col, padx=5, pady=5)
        return entry

    rooms_entry = create_labeled_entry(basic_frame, "Стаи", 0, 0)
    area_entry = create_labeled_entry(basic_frame, "Площ", 0, 1)
    year_entry = create_labeled_entry(basic_frame, "Година", 0, 2)
    floor_entry = create_labeled_entry(basic_frame, "Етаж", 2, 0)
    total_floors_entry = create_labeled_entry(basic_frame, "Общо етажи", 2, 1)

    loc_frame = tk.LabelFrame(main_container, text=" Локация и Тип ", bg="white", padx=15, pady=15)
    loc_frame.pack(fill="x", pady=10)
    districts = sorted(list(district_encoder.mapping.keys()))
    district_var = tk.StringVar(value=districts[0])
    ttk.Combobox(loc_frame, textvariable=district_var, values=districts, state="readonly").pack(fill="x", pady=5)
    const_types = sorted(list(construction_encoder.mapping.keys()))
    construction_var = tk.StringVar(value=const_types[0])
    ttk.Combobox(loc_frame, textvariable=construction_var, values=const_types, state="readonly").pack(fill="x", pady=5)

    extra_frame = tk.LabelFrame(main_container, text=" Екстри ", bg="white", padx=15, pady=15)
    extra_frame.pack(fill="x", pady=10)
    garage_var = tk.BooleanVar();
    tk.Checkbutton(extra_frame, text="Гараж", variable=garage_var, bg="white").grid(row=0, column=0, sticky="w")
    closed_complex_var = tk.BooleanVar();
    tk.Checkbutton(extra_frame, text="Затворен Комплекс", variable=closed_complex_var, bg="white").grid(row=0, column=1,
                                                                                                        sticky="w")
    gas_var = tk.BooleanVar();
    tk.Checkbutton(extra_frame, text="Газ", variable=gas_var, bg="white").grid(row=1, column=0, sticky="w")
    tep_var = tk.BooleanVar();
    tk.Checkbutton(extra_frame, text="ТЕЦ", variable=tep_var, bg="white").grid(row=1, column=1, sticky="w")
    luxury_var = tk.BooleanVar();
    tk.Checkbutton(extra_frame, text="Луксозно обзавеждане", variable=luxury_var, bg="white").grid(row=2, column=0,
                                                                                                   sticky="w")
    act16_var = tk.BooleanVar();
    tk.Checkbutton(extra_frame, text="Акт 16", variable=act16_var, bg="white").grid(row=2, column=1, sticky="w")

    tk.Button(main_container, text="ИЗЧИСЛИ ЦЕНА", command=calculate_price, bg="#1877f2", fg="black", font=("Arial", 12, "bold"), pady=10).pack(fill="x", pady=25)
    res_frame = tk.Frame(main_container, bg="#e7f3ff", pady=15)
    res_frame.pack(fill="x")
    result_label = tk.Label(res_frame, text="--- €", font=("Helvetica", 24, "bold"), bg="#e7f3ff", fg="#1877f2")
    result_label.pack()

    root.mainloop()

if __name__ == "__main__":
    start_app()