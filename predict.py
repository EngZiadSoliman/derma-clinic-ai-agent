
import joblib
import pandas as pd



from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent

# Load the trained models
preprocessor = joblib.load(MODEL_DIR / "clinic_preprocessor.joblib")
classifier = joblib.load(MODEL_DIR / "clinic_classifier.joblib")
discount_preprocessor = joblib.load(MODEL_DIR / "discount_preprocessor.joblib")
discount_classifier = joblib.load(MODEL_DIR / "discount_classifier.joblib")


def predict_customer(customer_data):

    # Calculate average spending per visit
    customer = customer_data.copy()

    customer["avg_spend_per_visit_sar"] = (
        customer["total_spent_before_prediction_sar"]
        / customer["visit_count"]
        if customer["visit_count"] > 0
        else 0.0
    )

    # 1. Baseline return prediction
    X = pd.DataFrame([customer])
    X_processed = preprocessor.transform(X)

    baseline_probability = classifier.predict_proba(
        X_processed
    )[0][1]

    # 2. Make the discount recommendation
    if baseline_probability > 0.80:

        return {
            "return_probability": float(baseline_probability),
            "recommended_discount": 0,
            "predicted_probability_with_discount": None,
            "message": (
                "No discount is needed. The customer already "
                "has a high likelihood of returning."
            )
        }

    else:
        discount_options = [0, 10, 15, 20]
        results = []

        for discount in discount_options:
            discount_customer = customer.copy()
            discount_customer["discount_percent"] = discount

            X_discount = pd.DataFrame([discount_customer])
            X_discount_processed = discount_preprocessor.transform(
                X_discount
            )

            probability = discount_classifier.predict_proba(
                X_discount_processed
            )[0][1]

            results.append((discount, probability))

        # Select the discount with the highest predicted probability
        best_discount, best_probability = max(
            results, key=lambda result: result[1]
        )


        message = (
            f"The baseline 90-day return probability is "
            f"{baseline_probability:.2%}. "
            f"The discount-response model estimates a "
            f"{best_probability:.2%} response probability "
            f"with a {best_discount}% discount. "
            f"This is the highest predicted response among "
            f"the tested discount options. "
            f"These are estimates from separate models, "
            f"not a measured causal effect of the discount."
        )

        return {
            "return_probability": float(baseline_probability),
            "recommended_discount": int(best_discount),
            "predicted_probability_with_discount": float(best_probability),
            "message": message
        }


# Allow the script to run independently for testing
if __name__ == "__main__":

    customer = {
        "visit_count": int(input("Total visit count: ")),
        "days_since_last_visit": int(input("Days since last visit: ")),
        "avg_days_between_visits": float(
            input("Average days between visits: ")
        ),
        "previous_visit_count_180d": int(
            input("Visits in last 180 days: ")
        ),
        "service_type": input("Service type: "),
        "acquisition_channel": input("Acquisition channel: "),
        "previous_campaign_response": input(
            "Previous campaign response (Yes/No/Never): "
        ),
        "last_procedure_cost_sar": float(
            input("Last procedure cost (SAR): ")
        ),
        "total_spent_before_prediction_sar": float(
            input("Total spent before prediction (SAR): ")
        ),
        "discount_percent": 0
    }

    result = predict_customer(customer)

    print("\n--- Prediction Results ---")
    print(f"Return probability: {result['return_probability']:.2%}")
    print(result["message"])
    print("\nNote: These are model estimates, not guaranteed outcomes.")
