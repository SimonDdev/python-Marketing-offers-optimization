import sys
import logging
import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton,
    QComboBox, QSlider, QTextEdit, QHBoxLayout, QSpinBox,
    QMessageBox, QTabWidget, QTableWidget, QTableWidgetItem
)
from config import SLIDER_CONFIG
from data_loader import load_data
from model_trainer import ModelTrainer
from recommendation_engine import RecommendationSystem

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    clients, offers, history = load_data()
except Exception as e:
    logging.error("Грешка при зареждане на данните: %s", e)
    sys.exit(1)

trainer = ModelTrainer(clients, offers, history)
model, scaler = trainer.train_model()
recommender = RecommendationSystem(model, scaler, offers)

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Маркетингови оферти – Single & Campaign Optimization")
        self.setGeometry(100, 100, 800, 700)
        main_layout = QVBoxLayout()
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        self._build_single_tab()
        self._build_campaign_tab()
        self.setLayout(main_layout)

    def _build_single_tab(self):
        self.single_tab = QWidget()
        layout = QVBoxLayout()

        layout.addWidget(QLabel("Въведете вашите данни:"))
        # Пол
        h1 = QHBoxLayout()
        h1.addWidget(QLabel("Пол:"))
        self.gender_select = QComboBox()
        self.gender_select.addItem("Мъж", "M")
        self.gender_select.addItem("Жена", "F")
        h1.addWidget(self.gender_select)
        layout.addLayout(h1)

        # Възраст
        h2 = QHBoxLayout()
        h2.addWidget(QLabel("Възраст:"))
        self.age_spin = QSpinBox()
        self.age_spin.setRange(18, 80)
        self.age_spin.setValue(30)
        h2.addWidget(self.age_spin)
        layout.addLayout(h2)

        # Доход
        h3 = QHBoxLayout()
        h3.addWidget(QLabel("Годишен доход (лв):"))
        self.income_spin = QSpinBox()
        self.income_spin.setRange(20000, 120000)
        self.income_spin.setSingleStep(1000)
        self.income_spin.setValue(50000)
        h3.addWidget(self.income_spin)
        layout.addLayout(h3)

        # Предишни покупки
        h4 = QHBoxLayout()
        h4.addWidget(QLabel("Брой предишни покупки:"))
        self.purchases_spin = QSpinBox()
        self.purchases_spin.setRange(0, 100)
        self.purchases_spin.setValue(3)
        h4.addWidget(self.purchases_spin)
        layout.addLayout(h4)

        # Предпочитана категория
        h5 = QHBoxLayout()
        h5.addWidget(QLabel("Предпочитана категория:"))
        self.category_select = QComboBox()
        self.category_select.addItem("Без предпочитание", None)
        for cat in ["Electronics","Fashion","Home","Sports","Books","Beauty","Toys","Automotive","Grocery","Garden"]:
            self.category_select.addItem(cat, cat)
        h5.addWidget(self.category_select)
        layout.addLayout(h5)

        # Бюджет
        layout.addWidget(QLabel("Задай бюджет (лв):"))
        self.budget_slider = QSlider(Qt.Horizontal)
        self.budget_slider.setMinimum(SLIDER_CONFIG['min'])
        self.budget_slider.setMaximum(SLIDER_CONFIG['max'])
        self.budget_slider.setValue(SLIDER_CONFIG['default'])
        self.budget_slider.setTickInterval(SLIDER_CONFIG['interval'])
        self.budget_slider.setTickPosition(QSlider.TicksBelow)
        self.budget_slider.valueChanged.connect(self.on_budget_change)
        layout.addWidget(self.budget_slider)
        self.budget_label = QLabel(f"Текущ бюджет: {self.budget_slider.value()} лв")
        layout.addWidget(self.budget_label)

        self.single_button = QPushButton("Препоръчайте оферта")
        self.single_button.clicked.connect(self.on_click)
        layout.addWidget(self.single_button)

        self.single_results = QTextEdit()
        self.single_results.setReadOnly(True)
        layout.addWidget(QLabel("Препоръчана оферта:"))
        layout.addWidget(self.single_results)

        self.single_tab.setLayout(layout)
        self.tabs.addTab(self.single_tab, "Single Offer")

    def _build_campaign_tab(self):
        self.campaign_tab = QWidget()
        layout = QVBoxLayout()

        layout.addWidget(QLabel("Campaign budget (лв):"))
        self.campaign_budget_spin = QSpinBox()
        self.campaign_budget_spin.setRange(1000, 1_000_000)
        self.campaign_budget_spin.setSingleStep(1000)
        self.campaign_budget_spin.setValue(50_000)
        layout.addWidget(self.campaign_budget_spin)

        self.campaign_button = QPushButton("Optimize Campaign")
        self.campaign_button.clicked.connect(self.on_campaign_optimize)
        layout.addWidget(self.campaign_button)

        self.campaign_results = QTableWidget()
        layout.addWidget(self.campaign_results)

        self.campaign_tab.setLayout(layout)
        self.tabs.addTab(self.campaign_tab, "Campaign Optimization")

    def on_budget_change(self, value):
        self.budget_label.setText(f"Текущ бюджет: {value} лв")

    def on_click(self):
        client = {
            'gender': self.gender_select.currentData(),
            'age': self.age_spin.value(),
            'income': self.income_spin.value(),
            'previous_purchases': self.purchases_spin.value(),
            'preferred_category': self.category_select.currentData(),
            'budget': self.budget_slider.value()
        }
        rec = recommender.get_recommendations(client)
        if rec is None or rec.empty:
            self.single_results.setText("Няма подходяща оферта.")
            return
        row = rec.iloc[0]
        text = (
            f"Offer ID: {row.offer_id}\n"
            f"Name: {row.offer_name}\n"
            f"Price: {row.price} лв\n"
            f"Category: {row.category}\n"
            f"Brand: {row.brand}\n"
        )
        self.single_results.setText(text)

    def on_campaign_optimize(self):
        total_budget = self.campaign_budget_spin.value()
        assignments = recommender.optimize_campaign(clients, total_budget)
        if assignments is None or assignments.empty:
            QMessageBox.information(self, "Result", "No assignments found under this budget.")
            return

        df = assignments
        self.campaign_results.clear()
        self.campaign_results.setRowCount(len(df))
        self.campaign_results.setColumnCount(len(df.columns))
        self.campaign_results.setHorizontalHeaderLabels(df.columns.tolist())
        for i, row in df.iterrows():
            for j, col in enumerate(df.columns):
                item = QTableWidgetItem(str(row[col]))
                self.campaign_results.setItem(i, j, item)
        self.campaign_results.resizeColumnsToContents()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())











