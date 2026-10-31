
import pandas as pd
from pathlib import Path
from langchain_core.tools import tool

# Locate the clinic CSV files
CLINIC_DIR = Path(__file__).resolve().parent.parent / "derma_clinic"

customers = pd.read_csv(CLINIC_DIR / "customers.csv",dtype={"phone_number": str})
visits = pd.read_csv(CLINIC_DIR / "visits.csv")


@tool
def lookup_customer(customer_id: str) -> dict:
    """Find a clinic customer by customer ID and retrieve their profile and visit history."""

    customer = customers[
        customers["customer_id"].astype(str).str.upper()
        == customer_id.strip().upper()
    ]

    if customer.empty:
        return {"error": f"Customer {customer_id} was not found."}

    customer_info = customer.iloc[0].to_dict()

    customer_visits = visits[
        visits["customer_id"].astype(str).str.upper()
        == customer_id.strip().upper()
    ].copy()

    customer_visits = customer_visits.fillna("")

    return {
        "customer": customer_info,
        "total_visits": len(customer_visits),
        "visit_history": customer_visits.to_dict(orient="records")
    }
