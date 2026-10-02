"""
NBA Games Analytics — визуализация данных
Аналог интерактивного дашборда, построенный на matplotlib/seaborn.

Запуск:
    python nba_analysis.py

Файл games.csv должен быть в той же папке.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ──────────────────────────────────────────
# Настройка стиля
# ──────────────────────────────────────────
sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.facecolor": "#F8F9FA",
    "axes.facecolor":   "#FFFFFF",
    "axes.edgecolor":   "#CCCCCC",
    "axes.linewidth":   0.8,
    "grid.color":       "#E5E5E5",
    "grid.linewidth":   0.6,
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "axes.labelsize":   11,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
})

BLUE  = "#185FA5"
RED   = "#A32D2D"
LIGHT_BLUE = "#85B7EB"
LIGHT_RED  = "#F09595"

# ──────────────────────────────────────────
# 1. Загрузка и очистка данных
# ──────────────────────────────────────────
df = pd.read_csv("games.csv")
df = df.dropna().reset_index(drop=True)

FEATURES = [
    "FG_PCT_home", "FT_PCT_home", "FG3_PCT_home", "AST_home", "REB_home",
    "FG_PCT_away", "FT_PCT_away", "FG3_PCT_away", "AST_away", "REB_away",
]
FEATURE_LABELS = [
    "FG%_home", "FT%_home", "FG3%_home", "AST_home", "REB_home",
    "FG%_away", "FT%_away", "FG3%_away", "AST_away", "REB_away",
]

wins   = df[df["HOME_TEAM_WINS"] == 1]
losses = df[df["HOME_TEAM_WINS"] == 0]

print(f"Загружено матчей: {len(df)}")
print(f"Победы хозяев:   {len(wins)} ({len(wins)/len(df)*100:.1f}%)")
print(f"Поражения хозяев:{len(losses)} ({len(losses)/len(df)*100:.1f}%)")

# ──────────────────────────────────────────
# 2. ГРАФИК 1 — Корреляции признаков с победой
# ──────────────────────────────────────────
corr = df[FEATURES + ["HOME_TEAM_WINS"]].corr()["HOME_TEAM_WINS"].drop("HOME_TEAM_WINS")

fig1, ax1 = plt.subplots(figsize=(10, 5))

colors = [BLUE if v > 0 else RED for v in corr.values]
bars = ax1.barh(FEATURE_LABELS, corr.values, color=colors, height=0.6, zorder=3)

# Подписи значений
for bar, val in zip(bars, corr.values):
    ax1.text(
        val + (0.005 if val >= 0 else -0.005),
        bar.get_y() + bar.get_height() / 2,
        f"{val:+.3f}",
        va="center",
        ha="left" if val >= 0 else "right",
        fontsize=9,
        color="#333333",
    )

ax1.axvline(0, color="#555555", linewidth=0.9, zorder=4)
ax1.set_xlim(-0.52, 0.52)
ax1.set_xlabel("Корреляция Пирсона с HOME_TEAM_WINS")
ax1.set_title("График 1. Влияние признаков на победу домашней команды")

legend_handles = [
    mpatches.Patch(color=BLUE, label="Помогает хозяевам"),
    mpatches.Patch(color=RED,  label="Помогает гостям"),
]
ax1.legend(handles=legend_handles, loc="lower right", fontsize=9)
ax1.invert_yaxis()
fig1.tight_layout()
fig1.savefig("plot1_correlations.png", dpi=150, bbox_inches="tight")
print("Сохранён: plot1_correlations.png")

# ──────────────────────────────────────────
# 3. ГРАФИК 2 — Гистограмма FG_PCT_home: победы vs поражения
# ──────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(10, 5))

bins = np.arange(0.25, 0.72, 0.03)

ax2.hist(wins["FG_PCT_home"],   bins=bins, alpha=0.75, color=BLUE, label=f"Победа хозяев (ср. {wins['FG_PCT_home'].mean():.3f})",   zorder=3)
ax2.hist(losses["FG_PCT_home"], bins=bins, alpha=0.65, color=RED,  label=f"Поражение (ср. {losses['FG_PCT_home'].mean():.3f})", zorder=3)

ax2.axvline(wins["FG_PCT_home"].mean(),   color=BLUE, linestyle="--", linewidth=1.5, alpha=0.9)
ax2.axvline(losses["FG_PCT_home"].mean(), color=RED,  linestyle="--", linewidth=1.5, alpha=0.9)

ax2.set_xlabel("FG_PCT_home (% попаданий с игры — хозяева)")
ax2.set_ylabel("Количество матчей")
ax2.set_title("График 2. Распределение FG_PCT_home — победы vs поражения хозяев")
ax2.legend(fontsize=10)
fig2.tight_layout()
fig2.savefig("plot2_fg_distribution.png", dpi=150, bbox_inches="tight")
print("Сохранён: plot2_fg_distribution.png")

# ──────────────────────────────────────────
# 4. ГРАФИК 3 — Тепловая карта корреляций
# ──────────────────────────────────────────
corr_matrix = df[FEATURES].corr()
corr_matrix.index   = FEATURE_LABELS
corr_matrix.columns = FEATURE_LABELS

fig3, ax3 = plt.subplots(figsize=(10, 8))

cmap = sns.diverging_palette(220, 20, as_cmap=True)
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap=cmap,
    center=0,
    vmin=-1, vmax=1,
    linewidths=0.5,
    linecolor="#EEEEEE",
    ax=ax3,
    annot_kws={"size": 9},
    square=True,
)

ax3.set_title("График 3. Матрица корреляций игровых показателей")
ax3.tick_params(axis="x", rotation=30)
ax3.tick_params(axis="y", rotation=0)
fig3.tight_layout()
fig3.savefig("plot3_heatmap.png", dpi=150, bbox_inches="tight")
print("Сохранён: plot3_heatmap.png")

# ──────────────────────────────────────────
# 5. ГРАФИК 4 — Сравнение средних показателей: победы vs поражения
# ──────────────────────────────────────────
win_means  = wins[FEATURES].mean()
loss_means = losses[FEATURES].mean()

# Нормализуем каждый признак в диапазон [0,1] относительно min/max двух групп
mn = np.minimum(win_means.values, loss_means.values)
mx = np.maximum(win_means.values, loss_means.values)
rng = mx - mn
rng[rng == 0] = 1  # защита от деления на ноль

win_norm  = (win_means.values  - mn) / rng
loss_norm = (loss_means.values - mn) / rng

x = np.arange(len(FEATURE_LABELS))
width = 0.38

fig4, ax4 = plt.subplots(figsize=(12, 5))

bars_w = ax4.bar(x - width/2, win_norm,  width, color=BLUE, alpha=0.85, label="Хозяева выиграли", zorder=3)
bars_l = ax4.bar(x + width/2, loss_norm, width, color=RED,  alpha=0.85, label="Хозяева проиграли", zorder=3)

# Tooltip-аналог: подписи реальных значений сверху столбцов
def fmt_val(v, label):
    if "PCT" in label:
        return f"{v*100:.1f}%"
    return f"{v:.1f}"

for bar, val, label in zip(bars_w, win_means.values, FEATURES):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             fmt_val(val, label), ha="center", va="bottom", fontsize=7.5, color=BLUE)

for bar, val, label in zip(bars_l, loss_means.values, FEATURES):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             fmt_val(val, label), ha="center", va="bottom", fontsize=7.5, color=RED)

ax4.set_xticks(x)
ax4.set_xticklabels(FEATURE_LABELS, rotation=20, ha="right")
ax4.set_ylabel("Нормализованное значение [0–1]")
ax4.set_ylim(0, 1.25)
ax4.set_title("График 4. Средние показатели команд: хозяева выиграли vs проиграли")
ax4.legend(fontsize=10)
fig4.tight_layout()
fig4.savefig("plot4_comparison.png", dpi=150, bbox_inches="tight")
print("Сохранён: plot4_comparison.png")

# ──────────────────────────────────────────
# 6. Показываем все 4 графика разом
# ──────────────────────────────────────────
plt.show()
print("\nГотово! Все 4 графика сохранены как PNG-файлы.")
