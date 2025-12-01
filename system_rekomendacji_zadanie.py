import pandas as pd
import time
from surprise import SVD, KNNBasic, Dataset, Reader, accuracy
from surprise.model_selection import GridSearchCV, cross_validate, train_test_split

df_ratings = pd.read_csv('ratings.csv', usecols=['userId', 'movieId', 'rating'])
df_movies = pd.read_csv('movies.csv', usecols=['movieId', 'title'])

min_movie_ratings = 50
min_user_ratings = 50

# 1. Filtrowanie
movie_counts = df_ratings['movieId'].value_counts()
filter_movies = movie_counts[movie_counts > min_movie_ratings].index.tolist()

user_counts = df_ratings['userId'].value_counts()
filter_users = user_counts[user_counts > min_user_ratings].index.tolist()

df_filtered = df_ratings[
    (df_ratings['movieId'].isin(filter_movies)) &
    (df_ratings['userId'].isin(filter_users))
].copy()

# Ograniczenie próbki ocen żeby komputer nie wybuchł
df_sample = df_filtered.sample(n=25000, random_state=42)

# Konwersja do formatu Surprise
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(df_sample[['userId', 'movieId', 'rating']], reader)

# 2. SVD
param_grid_svd = {
    'n_epochs': [5, 10],
    'lr_all': [0.002, 0.005],
    'reg_all': [0.02, 0.1]
}

gs_svd = GridSearchCV(SVD, param_grid_svd, measures=['rmse', 'mae'], cv=3)

# Ocena wydajności
start_time_svd_gs = time.time()
gs_svd.fit(data)
end_time_svd_gs = time.time()

# Wybór najlepszego algorytmu SVD
best_svd = gs_svd.best_estimator['rmse']

# KNN - Grid search + walidacja
param_grid_knn = {
    'k': [20, 30],
    'sim_options': {
        'name': ['cosine', 'msd'],
        'user_based': [False]
    }
}

gs_knn = GridSearchCV(KNNBasic, param_grid_knn, measures=['rmse', 'mae'], cv=3)

# ocena wydajności parametrów
start_time_knn_gs = time.time()
gs_knn.fit(data)
end_time_knn_gs = time.time()

# Wybór najlepszego algorytmu kNN
best_knn = gs_knn.best_estimator['rmse']

#  Porównanie najlepszego SVD i najlepszego KNN

# SVD
start_time_svd_cv = time.time()
cv_svd = cross_validate(best_svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=False)
end_time_svd_cv = time.time()
time_svd = end_time_svd_cv - start_time_svd_cv

# kNN
start_time_knn_cv = time.time()
cv_knn = cross_validate(best_knn, data, measures=['RMSE', 'MAE'], cv=5, verbose=False)
end_time_knn_cv = time.time()
time_knn = end_time_knn_cv - start_time_knn_cv

avg_rmse_svd = cv_svd['test_rmse'].mean()
avg_mae_svd = cv_svd['test_mae'].mean()

avg_rmse_knn = cv_knn['test_rmse'].mean()
avg_mae_knn = cv_knn['test_mae'].mean()

print(f"SVD: avg_rmse_svd: {avg_rmse_svd} SVD avg_mae_svd: {avg_mae_svd} time_svd: {time_svd}")
print(f"kNN: avg_rmse_knn: {avg_rmse_knn} avg_mae_knn: {avg_mae_knn} time_knn: {time_knn}")

if avg_rmse_svd < avg_rmse_knn:
    print("Wnioski: Algorytm SVD osiągnął mniejszy błąd średniokwadratowy (RMSE).")
else:
    print("Wnioski: Algorytm kNN osiągnął mniejszy błąd średniokwadratowy (RMSE).")

if time_svd < time_knn:
    print("Wnioski: Algorytm SVD był szybszy w obliczeniach.")
else:
    print("Wnioski: Algorytm kNN był szybszy w obliczeniach.")

# Przykład rekomendacji
trainset = data.build_full_trainset()
best_svd.fit(trainset)

# Przykładowa predykcja dla Usera 1 i Filmu 50
pred = best_svd.predict(uid=1, iid=50)
print(f"Przykładowa predykcja SVD (User 1, Film 50): {pred.est}")