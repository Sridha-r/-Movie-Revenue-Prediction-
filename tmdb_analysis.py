import pandas as pd
import ast
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -------------------------------------
# 1. Load & Merge Datasets
# -------------------------------------
movies_df = pd.read_csv("data/tmdb_5000_movies.csv")
credits_df = pd.read_csv("data/tmdb_5000_credits.csv")
credits_df.rename(columns={'movie_id': 'id'}, inplace=True)
df = movies_df.merge(credits_df, on='id')

# -------------------------------------
# 2. Clean Basic Fields
# -------------------------------------
df = df[(df['budget'] > 0) & (df['revenue'] > 0)]
df.drop_duplicates(inplace=True)
df['tagline'] = df['tagline'].fillna("No tagline")
df['homepage'] = df['homepage'].fillna("No homepage")
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

# -------------------------------------
# 3. Parse Genres
# -------------------------------------
df['genres'] = df['genres'].apply(ast.literal_eval)
df['genre_names'] = df['genres'].apply(lambda x: [d['name'] for d in x])

mlb_genres = MultiLabelBinarizer()
genre_df = pd.DataFrame(mlb_genres.fit_transform(df['genre_names']), columns=mlb_genres.classes_)

# -------------------------------------
# 4. Handle Production Companies (Top N)
# -------------------------------------
df['production_companies'] = df['production_companies'].apply(
    lambda x: [d['name'] for d in ast.literal_eval(x)] if pd.notnull(x) else [])

all_companies = [company for sublist in df['production_companies'] for company in sublist]
company_counts = Counter(all_companies)
top_companies = [name for name, count in company_counts.most_common(20)]

df['filtered_production_companies'] = df['production_companies'].apply(
    lambda comps: [c if c in top_companies else 'Other' for c in comps] or ['Other']
)

mlb_companies = MultiLabelBinarizer()
comp_df = pd.DataFrame(mlb_companies.fit_transform(df['filtered_production_companies']), columns=mlb_companies.classes_)

# -------------------------------------
# 5. Combine All Features
# -------------------------------------
df_model = pd.concat([df, genre_df, comp_df], axis=1)
df_model.drop(columns=['genres', 'genre_names', 'production_companies', 'filtered_production_companies'], inplace=True)

# Drop rows with missing revenue (if any)
df_model.dropna(subset=['revenue'], inplace=True)

C = df_model['vote_average'].mean()
m = 1000  # can adjust

df_model['vote_score'] = (
    (df_model['vote_count'] / (df_model['vote_count'] + m)) * df_model['vote_average'] +
    (m / (df_model['vote_count'] + m)) * C
)
df_model['release_year'] = df_model['release_date'].dt.year


# Cap revenue at 1 billion (₹1000 Cr)
df_model['revenue'] = df_model['revenue'].apply(lambda x: min(x, 1_000_000_000))

# -------------------------------------
# 6. Modeling Setup (with log-transformed revenue)
# -------------------------------------

# Define features
features = ['budget', 'popularity', 'runtime', 'original_language']
features += list(mlb_genres.classes_)          # genre one-hot columns
features += ['vote_score']
features += list(mlb_companies.classes_)       # production company one-hot columns
features += ['release_year']

df_model['revenue'] = df_model['revenue'].apply(lambda x: min(x, 1_000_000_000))
df_model['log_revenue'] = np.log1p(df_model['revenue'])



# Features and target
X = df_model[features]
y = df_model['log_revenue']

# Preprocessing
categorical = ['original_language']
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)
], remainder='passthrough')

# Pipeline: Preprocessing + Regressor
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Split data
X_train, X_test, y_train_log, y_test_log = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train_log)

# Predict (log scale), then inverse transform to original scale
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)          # predicted revenue
y_actual = np.expm1(y_test_log)        # actual revenue

# Evaluate
mae = mean_absolute_error(y_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_actual, y_pred))

print("\n Model Evaluation (log-transformed revenue):")
print(f"MAE (Mean Absolute Error): ₹{mae:,.2f}")
print(f"RMSE (Root Mean Squared Error): ₹{rmse:,.2f}")


plt.figure(figsize=(10, 6))
plt.scatter(y_actual, y_pred, alpha=0.5, color='royalblue', edgecolors='k')
plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--')  # Perfect prediction line
plt.xlabel("Actual Revenue")
plt.ylabel("Predicted Revenue")
plt.title("Predicted vs Actual Movie Revenue")
plt.grid(True)
plt.tight_layout()
plt.show()
