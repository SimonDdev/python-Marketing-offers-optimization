import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Задаване на seed за повторяемост
np.random.seed(42)
random.seed(42)


# Функция за случайна дата между две дати
def random_date(start, end):
    delta = end - start
    int_delta = delta.days
    random_day = random.randint(0, int_delta)
    return start + timedelta(days=random_day)


# ---------------------------
# 1. Генериране на данни за клиентите (clients.csv)
num_clients = 100
client_ids = list(range(1, num_clients + 1))
first_names = ["Ivan", "Petar", "Georgi", "Dimitar", "Stoian", "Hristo", "Todor", "Nikolay", "Miroslav", "Veselin"]
last_names = ["Ivanov", "Petrov", "Georgiev", "Dimitrov", "Stoianov", "Hristov", "Todorov", "Nikolov", "Miroslavov",
              "Veselinov"]
categories = ["Electronics", "Fashion", "Home", "Sports", "Books", "Beauty", "Toys", "Automotive", "Grocery", "Garden"]

clients = []
start_date = datetime(2022, 1, 1)
end_date = datetime.now()

for i in client_ids:
    gender = random.choice(["M", "F"])
    name = f"{random.choice(first_names)} {random.choice(last_names)}"
    age = np.random.randint(18, 70)
    # Разпределяне на доходите: половината с по-високи, половината с по-ниски стойности
    income = np.random.randint(60000, 120000) if random.random() < 0.5 else np.random.randint(20000, 60000)
    previous_purchases = np.random.randint(0, 21)
    # С вероятност 70% клиентът има поне една предпочитана категория (ако има повече от една, те са разделени със запетая)
    if random.random() < 0.7:
        num_pref = random.randint(1, 2)
        preferred_categories = random.sample(categories, num_pref)
        preferred_category = ", ".join(preferred_categories)
    else:
        preferred_category = None
    last_visit_date = random_date(start_date, end_date).strftime("%Y-%m-%d")

    clients.append({
        "client_id": i,
        "name": name,
        "gender": gender,
        "age": age,
        "income": income,
        "previous_purchases": previous_purchases,
        "preferred_category": preferred_category,
        "last_visit_date": last_visit_date
    })
clients_df = pd.DataFrame(clients)

# ---------------------------
# 2. Генериране на данни за офертите (offers.csv)
num_offers = 1000
offer_ids = list(range(1, num_offers + 1))
offer_categories = categories

brand_options = {
    "Electronics": ["Samsung", "Apple", "Sony", "LG", "Huawei"],
    "Fashion": ["Gucci", "Prada", "Zara", "H&M", "Uniqlo"],
    "Home": ["Ikea", "Home Depot", "Leroy Merlin"],
    "Sports": ["Nike", "Adidas", "Puma"],
    "Books": ["Penguin", "HarperCollins", "Random House"],
    "Beauty": ["L'Oreal", "Estée Lauder", "Maybelline"],
    "Toys": ["Lego", "Hasbro", "Mattel"],
    "Automotive": ["Toyota", "Ford", "BMW"],
    "Grocery": ["Whole Foods", "Kroger", "Aldi"],
    "Garden": ["Bunnings", "Lowe's", "Gardena"]
}

offers = []
for i in offer_ids:
    price = np.random.randint(50, 5001)  # Цена между 50 и 5000 лв
    category = random.choice(offer_categories)
    target_gender = random.choices(["All", "M", "F"], weights=[0.5, 0.25, 0.25])[0]
    min_age = np.random.randint(18, 41)
    max_age = np.random.randint(41, 71)
    if min_age >= max_age:
        min_age, max_age = max_age, min_age + 1
    margin_percentage = random.uniform(0.1, 0.5)
    estimated_profit = round(price * margin_percentage, 2)
    brand = random.choice(brand_options.get(category, ["Generic"]))

    min_income_required = np.random.randint(20000, 120001)  # доход между 20k и 120k
    min_previous_purchases_required = np.random.randint(0, 11)  # между 0 и 10

    offers.append({
        "offer_id": i,
        "offer_name": f"Product {i}",
        "price": price,
        "category": category,
        "target_gender": target_gender,
        "min_age": min_age,
        "max_age": max_age,
        "estimated_profit": estimated_profit,
        "brand": brand,
        "min_income_required": min_income_required,
        "min_previous_purchases_required": min_previous_purchases_required
    })

offers_df = pd.DataFrame(offers)

# ---------------------------
# 3. Генериране на данни за историята (history.csv)
num_history = 5000
history_entries = []
trans_start = datetime(2015, 1, 1)
trans_end = datetime(2023, 12, 31)
for _ in range(num_history):
    client_id = random.choice(client_ids)
    offer_id = random.choice(offer_ids)
    response = random.choices(["accepted", "rejected"], weights=[0.3, 0.7])[0]
    transaction_date = random_date(trans_start, trans_end).strftime("%Y-%m-%d")
    quantity = np.random.randint(1, 11)  # Между 1 и 10 единици
    cross_sell_count = np.random.randint(0, 6) if response == "accepted" else 0

    history_entries.append({
        "client_id": client_id,
        "offer_id": offer_id,
        "response": response,
        "transaction_date": transaction_date,
        "quantity": quantity,
        "cross_sell_count": cross_sell_count
    })
history_df = pd.DataFrame(history_entries)

# ---------------------------
# Записване във файлове (с UTF-8 с BOM за правилно показване на кирилица)
clients_df.to_csv("clients.csv", index=False, encoding="utf-8-sig")
offers_df.to_csv("offers.csv", index=False, encoding="utf-8-sig")
history_df.to_csv("history.csv", index=False, encoding="utf-8-sig")

print("Данните са успешно генерирани и записани в 'clients.csv', 'offers.csv' и 'history.csv'.")




