
import pandas as pd
from pathlib import Path
from langchain_core.tools import tool

CLINIC_DIR = Path(__file__).resolve().parent.parent / "derma_clinic"

customers = pd.read_csv(CLINIC_DIR / "customers.csv")
visits = pd.read_csv(CLINIC_DIR / "visits.csv")


@tool
def clinic_analytics(year: int = 0) -> dict:
    """
    Retrieve clinic statistics.
    If a year is provided, calculate visit and revenue statistics for that year.
    Use year=0 for all available data.
    """

    data = visits.copy()
    data["visit_date"] = pd.to_datetime(data["visit_date"])
    data["amount_paid_sar"] = pd.to_numeric(data["amount_paid_sar"])

    if year != 0:
        data = data[data["visit_date"].dt.year == year]

    service_counts = (
        data["service_type"]
        .value_counts()
        .to_dict()
    )

    channel_counts = customers["acquisition_channel"].value_counts().to_dict()

    return {
        "total_customers": int(customers["customer_id"].nunique()),
        "total_visits": int(len(data)),
        "total_revenue_sar": float(data["amount_paid_sar"].sum()),
        "average_revenue_per_visit_sar": (
            float(data["amount_paid_sar"].mean())
            if not data.empty else 0
        ),
        "services_breakdown": service_counts,
        "acquisition_channels": channel_counts,
        "year_filter": year if year != 0 else "All years"
    }
