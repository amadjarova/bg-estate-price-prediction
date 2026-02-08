import pandas as pd
import numpy as np
from data.label_encoder import LabelEncoder


def impute_rooms_by_area(row):
    rooms = row['Rooms']
    area = row['Area']

    if pd.isna(rooms) or rooms <= 0:
        if area < 45:
            return 1
        elif area < 75:
            return 2
        elif area < 115:
            return 3
        elif area < 160:
            return 4
        else:
            return 5
    return rooms


class PropertyProcessor:
    def __init__(self):
        self.vocabulary = {
            'ext_has_gas': ['газ', 'газово', 'газови', 'газифициран'],
            'ext_has_tep': ['тец', 'парно', 'централно'],
            'ext_is_luxury': ['луксозен', 'луксозна', 'лукс', 'висок клас'],
            'ext_is_act16': ['акт 16', 'акт-16', 'разрешение за ползване']
        }
        self.negations = ['без', 'няма', 'не']
        self.label_encoders = {}

    def extract_from_text(self, text):
        if not isinstance(text, str): return pd.Series([0] * len(self.vocabulary))
        text = text.lower()
        results = {}
        for feature, variations in self.vocabulary.items():
            found = 0
            for var in variations:
                if var in text:
                    idx = text.find(var)
                    snippet = text[max(0, idx - 15):idx]
                    if not any(neg in snippet for neg in self.negations):
                        found = 1
                        break
            results[feature] = found
        return pd.Series(results)

    def process_data(self, input_path, output_path):
        df = pd.read_csv(input_path, sep=',', encoding='utf-8')

        print("Step 1: Extracting hidden features from text...")
        extracted_features = df['Description'].apply(self.extract_from_text)
        df = pd.concat([df, extracted_features], axis=1)

        print("Step 1.5: Imputing missing rooms based on area...")
        df['Rooms'] = df.apply(impute_rooms_by_area, axis=1)

        print("Step 2: Encoding categorical data...")
        categorical_cols = ['District', 'Construction_Type']
        for col in categorical_cols:
            encoder = LabelEncoder()
            encoder.fit(df[col].astype(str))
            df[f'{col}_Encoded'] = encoder.transform(df[col].astype(str))
            self.label_encoders[col] = encoder  # Save encoder for later use

        median_year = df[df['Construction_Year'] > 0]['Construction_Year'].median()
        df.loc[df['Construction_Year'] <= 0, 'Construction_Year'] = median_year

        features_to_keep = [
                               'Rooms', 'Area', 'Floor_Number', 'Total_Floors', 'Construction_Year',
                               'Is_First_Floor', 'Is_Last_Floor', 'Has_Garage', 'Is_Closed_Complex',
                               'District_Encoded', 'Construction_Type_Encoded'
                           ] + list(self.vocabulary.keys())

        X = df[features_to_keep]
        y = df['Price']

        processed_df = pd.concat([X, y], axis=1)
        processed_df = processed_df[processed_df['Price'] > 0]
        processed_df.to_csv(output_path, index=False)

        for col, encoder in self.label_encoders.items():
            encoder.save(f'{col}_encoder.json')

        print(f"Processing complete! Data saved to: {output_path}")
        return X, y


if __name__ == "__main__":
    processor = PropertyProcessor()
    input_file = 'estates_20260123_160142.csv'
    output_file = 'final_training_data.csv'

    print("--- Starting processing ---")

    try:
        X, y = processor.process_data(input_file, output_file)
        print("--- Success! ---")
        print(f"Processed {len(X)} records.")
        print(f"Prepared features: {list(X.columns)}")

    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found. Ensure it is in the same folder.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")