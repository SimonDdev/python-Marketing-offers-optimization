# data_loader.py
import pandas as pd
from config import CSV_PATHS

def load_data():
    try:
        clients = pd.read_csv(CSV_PATHS['clients'])
        offers = pd.read_csv(CSV_PATHS['offers'])
        history = pd.read_csv(CSV_PATHS['history'])
    except Exception as e:
        raise Exception(f"Грешка при зареждане на CSV файловете: {e}")
    return clients, offers, history
