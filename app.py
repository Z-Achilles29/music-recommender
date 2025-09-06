from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


clf = joblib.load("artifacts/histgb_model.pkl")
scaler = joblib.load("artifacts/scaler.pkl")
encoder = joblib.load("artifacts/label_encoder.pkl")


# Load metadata + features
meta_df = pd.read_csv("CPD_metadata.csv")
features_df = pd.read_csv("CPD_features.csv")

# Prepare scaled features
X = features_df.drop(columns=["track_genre"])
X_scaled = scaler.transform(X)

# Fit similarity search
knn = NearestNeighbors(metric="cosine", algorithm="brute")
knn.fit(X_scaled)

# --- Flask app ---
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []
    query = ""

    if request.method == "POST":
        query = request.form["song_name"]
        matches = meta_df[meta_df["track_name"].str.contains(query, case=False, na=False)]

        if not matches.empty:
            idx = matches.index[0]  # first match
            distances, indices = knn.kneighbors(X_scaled[idx:idx+1], n_neighbors=6)

            for i in indices[0][1:]:  # skip the first (same song)
                rec = {
                    "track": meta_df.iloc[i]["track_name"],
                    "artist": meta_df.iloc[i]["artists"],
                    "genre": meta_df.iloc[i]["track_genre"]
                }
                recommendations.append(rec)

    return render_template("index.html", query=query, recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
