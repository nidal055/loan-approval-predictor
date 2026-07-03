import pandas as pd
import joblib

from paths import MODEL_PATH, MODELS_DIR
MODEL_PATH = MODELS_DIR / "loan_model.pkl"
model = joblib.load(MODEL_PATH)

print("=" * 50)
print("        LOAN APPROVAL PREDICTOR")
print("=" * 50)
print("\nPlease enter the applicant's information.\n")

dependents = int(input("Number of dependents: "))
education = int(input("Education (1 = Graduate, 0 = Not Graduate): "))
self_employed = int(input("Self Employed (1 = Yes, 0 = No): "))
income = float(input("Annual Income: "))
loan_amount = float(input("Loan Amount: "))
loan_term = float(input("Loan Term (months): "))
cibil = float(input("CIBIL Score: "))
residential = float(input("Residential Assets Value: "))
commercial = float(input("Commercial Assets Value: "))
luxury = float(input("Luxury Assets Value: "))
bank = float(input("Bank Asset Value: "))

applicant = pd.DataFrame([{
    "no_of_dependents": dependents,
    "education": education,
    "self_employed": self_employed,
    "income_annum": income,
    "loan_amount": loan_amount,
    "loan_term": loan_term,
    "cibil_score": cibil,
    "residential_assets_value": residential,
    "commercial_assets_value": commercial,
    "luxury_assets_value": luxury,
    "bank_asset_value": bank
}])

prediction = model.predict(applicant)[0]

print("\n" + "=" * 50)

if prediction == 1:
    print("🎉 RESULT: LOAN APPROVED")
else:
    print("❌ RESULT: LOAN REJECTED")

print("=" * 50)