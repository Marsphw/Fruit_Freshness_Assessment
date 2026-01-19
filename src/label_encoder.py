import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# ===============================
# Load original training CSV
# ===============================
df = pd.read_csv("../dataset.csv")

labels = (df["fruit"] + "_" + df["condition"]).values

# ===============================
# Re-fit LabelEncoder
# ===============================
le = LabelEncoder()
le.fit(labels)

print("Recovered classes:")
for i, cls in enumerate(le.classes_):
    print(f"{i}: {cls}")

# ===============================
# Save encoder
# ===============================
joblib.dump(le, "./model/label_encoder.pkl")
print("\nlabel_encoder.pkl saved successfully!")
