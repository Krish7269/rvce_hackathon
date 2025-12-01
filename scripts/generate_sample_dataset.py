import csv
import random
from datetime import datetime, timedelta


def main():
    start = datetime(2024, 1, 1)
    days = 90
    products = ["Alpha", "Beta", "Gamma"]
    rows = []
    for i in range(days):
        day = start + timedelta(days=i)
        for product in products:
            rows.append(
                {
                    "order_date": day.strftime("%Y-%m-%d"),
                    "product": product,
                    "revenue": random.randint(50, 500),
                }
            )

    with open("dataset.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["order_date", "product", "revenue"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated dataset.csv with {len(rows)} rows")


if __name__ == "__main__":
    main()


