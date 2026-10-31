
import sys
from pathlib import Path
import pandas as pd
from langchain_core.tools import tool

# Locate the clinic project
CLINIC_DIR = Path(__file__).resolve().parent.parent / "derma_clinic"
sys.path.insert(0, str(CLINIC_DIR))

from predict import predict_customer

# Load customer and visit records
customers = pd.read_csv(
    CLINIC_DIR / "customers.csv",
    dtype={"phone_number": str}
)

visits = pd.read_csv(CLINIC_DIR / "visits.csv")


@tool
def predict_customer_return_by_id(customer_id: str) -> dict:
    """Predict a clinic customer's return probability and evaluate discount options using their customer ID."""

    customer_id = customer_id.strip().upper()

    # Find customer
    customer_rows = customers[
        customers["customer_id"].str.upper() == customer_id
    ]

    if customer_rows.empty:
        return {"error": f"Customer {customer_id} was not found."}

    customer = customer_rows.iloc[0]

    # Get customer's historical visits
    customer_visits = visits[
        visits["customer_id"].str.upper() == customer_id
    ].copy()

    if customer_visits.empty:
        return {"error": "This customer has no visit history."}

    # Convert visit dates and keep visits up to prediction date
    customer_visits["visit_date"] = pd.to_datetime(
        customer_visits["visit_date"]
    )

    prediction_date = pd.Timestamp.today().normalize()

    customer_visits = customer_visits[
        customer_visits["visit_date"] <= prediction_date
    ].sort_values("visit_date")

    if customer_visits.empty:
        return {"error": "No historical visits are available."}

    # Calculate features
    visit_count = len(customer_visits)

    last_visit = customer_visits.iloc[-1]
    days_since_last_visit = (
        prediction_date - last_visit["visit_date"]
    ).days

    if visit_count > 1:
        average_days = (
            customer_visits["visit_date"]
            .diff()
            .dt.days
            .mean()
        )
    else:
        average_days = 0.0

    previous_visit_count_180d = len(
        customer_visits[
            customer_visits["visit_date"]
            >= prediction_date - pd.Timedelta(days=180)
        ]
    )

    total_spent = float(
        customer_visits["amount_paid_sar"].sum()
    )

    customer_data = {
        "visit_count": visit_count,
        "days_since_last_visit": days_since_last_visit,
        "avg_days_between_visits": float(average_days),
        "previous_visit_count_180d": previous_visit_count_180d,
        "service_type": last_visit["service_type"],
        "acquisition_channel": customer["acquisition_channel"],
        "previous_campaign_response": last_visit["campaign_response"],
        "last_procedure_cost_sar": float(
            last_visit["amount_paid_sar"]
        ),
        "total_spent_before_prediction_sar": total_spent,
        "discount_percent": 0,
    }

    # Run existing ML prediction
    result = predict_customer(customer_data)

    return {
        "customer_id": customer_id,
        "customer_name": customer["customer_name"],
        "prediction_date": str(prediction_date.date()),
        "features_used": customer_data,
        **result
    }
