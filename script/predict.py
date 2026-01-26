import joblib
import pandas as pd
from pathlib import Path

HOME_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = HOME_DIR / "models" / "house_price_model.pkl"

def pulire_input(input):
    df = pd.DataFrame([input])
    
    trasformazione_binaria = {
    'yes': 1,
    'no': 0,
    }
    # hanno un ordinamento furnished > semi-furnished > unfurnished
    trasformazione_fornitura = {
        'unfurnished': 0,
        'semi-furnished': 1,
        'furnished': 2,
    }

    colonne_yes_no = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']

    for col in colonne_yes_no:
        df[col] = df[col].map(trasformazione_binaria)
    df['furnishingstatus'] = df['furnishingstatus'].map(trasformazione_fornitura)
    return df
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"❌ File non trovato {MODEL_PATH}")
else:
    model = joblib.load(MODEL_PATH)

    # simulazione input
    user_input = {
        'area': 6000,
        'bedrooms': 3,
        'bathrooms': 2,
        'stories': 2,
        'mainroad': 'yes',
        'guestroom': 'no',
        'basement': 'yes',
        'hotwaterheating': 'no',
        'airconditioning': 'yes',
        'parking': 2,
        'prefarea': 'yes',
        'furnishingstatus': 'semi-furnished'
    }
    print("Dati di input: ")
    print(user_input)
    df_user_input = pulire_input(user_input)
    previsione = model.predict(df_user_input)[0] # ritorna una lista, prendo solo il primo elemento
    print("="*30)
    print(f"Prezzo stimato: {previsione:.2f}")
    print("="*30)