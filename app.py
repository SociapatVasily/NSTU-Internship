"""
Запуск:
    python app.py
Перед этим обязательно запусти train_model.py, чтобы создался
файл nba_model.joblib.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from joblib import load

# Описание каждого поля: (подпись в UI, значение по умолчанию)
FIELDS = {
    'FG_PCT_home':  ('% попаданий с игры',  0.47),
    'FT_PCT_home':  ('% штрафных',          0.78),
    'FG3_PCT_home': ('% трёхочковых',       0.36),
    'AST_home':     ('Передачи',            24),
    'REB_home':     ('Подборы',             44),
    'FG_PCT_away':  ('% попаданий с игры',  0.47),
    'FT_PCT_away':  ('% штрафных',          0.78),
    'FG3_PCT_away': ('% трёхочковых',       0.36),
    'AST_away':     ('Передачи',            24),
    'REB_away':     ('Подборы',             44),
}

HOME_FIELDS = [k for k in FIELDS if k.endswith('_home')]
AWAY_FIELDS = [k for k in FIELDS if k.endswith('_away')]


class NBAPredictApp:
    def __init__(self, root):
        self.root = root
        self.root.title("NBA -- предсказание исхода матча")
        self.root.geometry("620x620")
        self.root.configure(bg="#f4f4f4")

        # Загружаем модель
        try:
            bundle = load('nba_model.joblib')
            self.model = bundle['model']
            self.feature_names = bundle['features']
        except FileNotFoundError:
            messagebox.showerror(
                "Модель не найдена",
                "Файл nba_model.joblib не найден.\n"
                "Сначала запусти: python train_model.py"
            )
            root.destroy()
            return

        self.entries = {}  # сюда сложим поля ввода
        self._build_ui()

    def _build_ui(self):
        # Заголовок
        title = tk.Label(
            self.root,
            text="Предсказание исхода матча NBA",
            font=("Arial", 16, "bold"),
            bg="#f4f4f4", fg="#222",
        )
        title.pack(pady=(15, 5))

        subtitle = tk.Label(
            self.root,
            text="Введи статистику двух команд и нажми «Предсказать»",
            font=("Arial", 10),
            bg="#f4f4f4", fg="#666",
        )
        subtitle.pack(pady=(0, 15))

        # Два блока бок о бок: Хозяева | Гости
        columns = tk.Frame(self.root, bg="#f4f4f4")
        columns.pack(padx=20, pady=5, fill="x")

        self._build_team_block(columns, "🏠 Хозяева", HOME_FIELDS, col=0,
                               bg="#e3f2fd")
        self._build_team_block(columns, "✈ Гости",   AWAY_FIELDS, col=1,
                               bg="#fff3e0")

        # Кнопка предсказания
        predict_btn = tk.Button(
            self.root,
            text="Предсказать",
            font=("Arial", 12, "bold"),
            bg="#2e7d32", fg="white",
            activebackground="#1b5e20", activeforeground="white",
            padx=30, pady=8,
            relief="flat", cursor="hand2",
            command=self.predict,
        )
        predict_btn.pack(pady=15)

        # Кнопка сброса
        reset_btn = tk.Button(
            self.root,
            text="Сбросить значения",
            font=("Arial", 9),
            bg="#f4f4f4", fg="#666",
            relief="flat", cursor="hand2",
            command=self.reset_fields,
        )
        reset_btn.pack()

        # Область результата
        self.result_frame = tk.Frame(self.root, bg="#f4f4f4")
        self.result_frame.pack(pady=15, padx=20, fill="x")

        self.result_label = tk.Label(
            self.result_frame,
            text="Результат появится здесь",
            font=("Arial", 12),
            bg="#f4f4f4", fg="#999",
            pady=15,
        )
        self.result_label.pack(fill="x")

    def _build_team_block(self, parent, title, field_keys, col, bg):
        """Блок с полями одной команды (хозяева или гости)."""
        frame = tk.Frame(parent, bg=bg, padx=15, pady=12,
                         highlightbackground="#ccc", highlightthickness=1)
        frame.grid(row=0, column=col, padx=5, sticky="nsew")
        parent.grid_columnconfigure(col, weight=1)

        # Заголовок блока
        tk.Label(frame, text=title, font=("Arial", 12, "bold"),
                 bg=bg, fg="#222").grid(row=0, column=0, columnspan=2,
                                        pady=(0, 10), sticky="w")

        # Поля
        for i, key in enumerate(field_keys, start=1):
            label_text, default = FIELDS[key]
            tk.Label(frame, text=label_text, font=("Arial", 10),
                     bg=bg, fg="#333", anchor="w"
                     ).grid(row=i, column=0, sticky="w", pady=3)

            entry = ttk.Entry(frame, width=10, font=("Arial", 10))
            entry.insert(0, str(default))
            entry.grid(row=i, column=1, sticky="e", pady=3, padx=(10, 0))
            self.entries[key] = entry

    def reset_fields(self):
        """Вернуть значения по умолчанию."""
        for key, entry in self.entries.items():
            entry.delete(0, tk.END)
            entry.insert(0, str(FIELDS[key][1]))
        self.result_label.config(
            text="Результат появится здесь",
            bg="#f4f4f4", fg="#999",
        )
        self.result_frame.config(bg="#f4f4f4")

    def _read_values(self):
        """Читаем все поля и конвертируем в float. Возвращаем словарь или None."""
        values = {}
        for key, entry in self.entries.items():
            raw = entry.get().strip().replace(',', '.')
            try:
                values[key] = float(raw)
            except ValueError:
                messagebox.showerror(
                    "Ошибка ввода",
                    f"Поле '{FIELDS[key][0]}' ({key}) содержит не число:\n"
                    f"«{raw}»"
                )
                return None
        return values

    def predict(self):
        values = self._read_values()
        if values is None:
            return

        # Собираем DataFrame в том порядке, в котором обучалась модель
        row = [values[name] for name in self.feature_names]
        X_new = pd.DataFrame([row], columns=self.feature_names)

        proba = self.model.predict_proba(X_new)[0]
        p_away, p_home = proba[0], proba[1]

        # Выбираем победителя и уровень уверенности
        if p_home >= p_away:
            winner_text = "Победят ХОЗЯЕВА"
            confidence = p_home
        else:
            winner_text = "Победят ГОСТИ"
            confidence = p_away

        if confidence > 0.75:
            tag = "уверенно"
            color = "#2e7d32"   # зелёный
        elif confidence > 0.6:
            tag = "скорее всего"
            color = "#f57c00"   # оранжевый
        else:
            tag = "игра равная"
            color = "#c62828"   # красный

        result_text = (
            f"{winner_text} ({tag})\n\n"
            f"Вероятность победы хозяев: {p_home:.1%}\n"
            f"Вероятность победы гостей: {p_away:.1%}"
        )

        self.result_label.config(
            text=result_text,
            bg="white", fg=color,
            font=("Arial", 12, "bold"),
            relief="solid", bd=1,
        )
        self.result_frame.config(bg="#f4f4f4")


if __name__ == '__main__':
    root = tk.Tk()
    app = NBAPredictApp(root)
    root.mainloop()
