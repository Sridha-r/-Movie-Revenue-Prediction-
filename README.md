# -Movie-Revenue-Prediction-
Movie Revenue Prediction using Machine Learning

This project predicts box office revenue for movies based on TMDb metadata using a regression model.

Features Used
- Budget
- Popularity
- Runtime
- Genre (multi-hot encoded)
- Production company (top 20 + "Other")
- Vote score (weighted IMDB-style)
- Release year

 Model
- Random Forest Regressor
- Log-transformed revenue as target
- MAE: ₹58.3M, RMSE: ₹105.3M

 Tech Stack
- Python, Pandas, scikit-learn, Matplotlib
- Jupyter/VS Code
 Dataset
- [TMDb Movie Metadata](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

 Output
Includes:
- Scatter plot of predicted vs actual revenue
- Clean feature pipeline
- Log scale accuracy improvement


