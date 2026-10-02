"""
Обучение модели для предсказания победы домашней команды в матче NBA.
Мы предсказываем исход по "качеству игры": проценты попаданий,
передачи, подборы.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump

# ----------------------------------------------------------------------
# 1. Загрузка и очистка данных
# ----------------------------------------------------------------------
df = pd.read_csv('games.csv')
print(f"Загружено строк: {len(df)}")

# Выбрасываем ненужные столбцы (id команд, даты, сезон и т.п.)
cols_to_drop = [
    'GAME_DATE_EST', 'GAME_ID', 'GAME_STATUS_TEXT',
    'HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'SEASON',
    'TEAM_ID_home', 'TEAM_ID_away',
    # !!! Выбрасываем PTS -- иначе будет утечка данных !!!
    'PTS_home', 'PTS_away',
]
df = df.drop(columns=cols_to_drop)

# Убираем строки с пропусками (их около 99 из 26651 -- не жалко)
df = df.dropna().reset_index(drop=True)
print(f"После очистки: {len(df)} строк")
print(f"Признаки: {list(df.columns.drop('HOME_TEAM_WINS'))}")

# ----------------------------------------------------------------------
# 2. Разделение на X и y, train/test
# ----------------------------------------------------------------------
X = df.drop('HOME_TEAM_WINS', axis=1)
y = df['HOME_TEAM_WINS']
feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------------------------------------------------
# 3. Обучаем Random Forest
# ----------------------------------------------------------------------
print("\n" + "=" * 60)
print("RANDOM FOREST")
print("=" * 60)

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred_rf,
                            target_names=['Гости', 'Хозяева']))
print("Confusion matrix (строки - факт, столбцы - предикт):")
print(confusion_matrix(y_test, y_pred_rf))

# Кросс-валидация для честной оценки
cv_scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy', n_jobs=-1)
print(f"\nCross-val accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ----------------------------------------------------------------------
# 4. Для сравнения -- логистическая регрессия
# ----------------------------------------------------------------------
print("\n" + "=" * 60)
print("LOGISTIC REGRESSION (для сравнения)")
print("=" * 60)

logreg = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000, random_state=42)),
])
logreg.fit(X_train, y_train)
y_pred_lr = logreg.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")

# ----------------------------------------------------------------------
# 5. Важность признаков (что больше всего влияет на победу)
# ----------------------------------------------------------------------
print("\n" + "=" * 60)
print("ВАЖНОСТЬ ПРИЗНАКОВ (Random Forest)")
print("=" * 60)

importances = pd.Series(rf.feature_importances_, index=feature_names)
importances = importances.sort_values(ascending=False)

# Красивая "текстовая гистограмма"
max_imp = importances.max()
for name, imp in importances.items():
    bar = '#' * int(imp / max_imp * 40)
    print(f"  {name:15s} {imp:.4f}  {bar}")

# Коэффициенты логрегрессии -- со знаком (что тянет в + / в -)
print("\nКоэффициенты логистической регрессии (знак = направление влияния):")
coefs = pd.Series(logreg.named_steps['clf'].coef_[0], index=feature_names)
coefs = coefs.reindex(coefs.abs().sort_values(ascending=False).index)
for name, c in coefs.items():
    direction = "→ ПОБЕДА хозяев" if c > 0 else "→ ПОБЕДА гостей"
    print(f"  {name:15s} {c:+.3f}  {direction}")

# ----------------------------------------------------------------------
# 6. Сохраняем модель и список фичей для predict_game.py
# ----------------------------------------------------------------------
dump({'model': rf, 'features': feature_names}, 'nba_model.joblib')
print("\nМодель сохранена в nba_model.joblib")
