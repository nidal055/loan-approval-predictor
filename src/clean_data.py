import pandas as pd

def clean_dataset(input_path, output_path):
    # Load dataset
    df = pd.read_csv(input_path)

    # Clean column names
    # normalize column names
    # use regex=False for pandas>=1.4 compatibility
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)

    # Clean text columns
    for column in df.select_dtypes(include="object").columns:
        df[column] = df[column].str.strip().str.lower()

    # Drop loan_id because it is just an identifier
    if "loan_id" in df.columns:
        df = df.drop(columns=["loan_id"])

    # Convert target column
    df["loan_status"] = df["loan_status"].map({
        "approved": 1,
        "rejected": 0
    })

    # Convert categorical columns
    df["education"] = df["education"].map({
        "graduate": 1,
        "not graduate": 0
    })

    df["self_employed"] = df["self_employed"].map({
        "yes": 1,
        "no": 0
    })

    # Check missing values
    print("Missing values:")
    print(df.isnull().sum())

    # Remove rows with missing values
    df = df.dropna()

    # Save cleaned dataset
    df.to_csv(output_path, index=False)

    print("\nCleaned dataset saved successfully.")
    print("Shape:", df.shape)
    print(df.head())


clean_dataset(
    "data/loan_approval_dataset.csv",
    "data/cleaned_loan_data.csv"
)