from pathlib import Path
import sys

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "subjects" / "tiabd" / "1" / "submission"
SUBMISSION.mkdir(parents=True, exist_ok=True)


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def build_fires() -> None:
    cells = [
        md(
            """
# Практическая работа № 1
## Учебные блоки: анализ `FiresRu.csv`

**Цель:** пройти базовый цикл анализа табличных данных: настроить окружение, загрузить и диагностировать данные, выполнить предобработку и EDA, построить визуализации и проверить экспорт.

**Источник:** набор «Метеорологические условия пожаров на территории России (2021-2022)», предоставленный вместе с заданием. В таблице нет даты/времени; поэтому временной ряд по ней не строится. Единицы `relative_humidity` и `solar_radiation` в исходных метаданных явно не заданы, и ниже им не приписываются неподтверждённые единицы.
"""
        ),
        md(
            """
### 2.1. Рабочее окружение

Используется локальное окружение `.venv`. Версии фиксируются непосредственно при выполнении notebook, чтобы результат можно было воспроизвести.
"""
        ),
        code(
            """
from pathlib import Path
import io
import sys

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display

print("Python:", sys.version.split()[0])
print("pandas:", pd.__version__)
print("NumPy:", np.__version__)
print("Matplotlib:", matplotlib.__version__)

NOTEBOOK_DIR = Path.cwd()
DATA_PATH = (NOTEBOOK_DIR.parent / "FiresRu.csv").resolve()
OUT = NOTEBOOK_DIR / "output" / "fires"
OUT.mkdir(parents=True, exist_ok=True)
print("Данные:", DATA_PATH)
print("Результаты:", OUT.resolve())
"""
        ),
        md(
            """
**Интерпретация.** Все библиотеки импортированы без ошибок, а пути строятся относительно каталога запуска notebook. Это отделяет код от абсолютных путей конкретного компьютера.
"""
        ),
        md("### 3.1. Векторное преобразование в NumPy"),
        code(
            """
wind_ms = np.array([1.14, 0.80, 0.97])
wind_kmh = wind_ms * 3.6
print(np.round(wind_kmh, 2))
"""
        ),
        md(
            """
**Интерпретация.** Одно выражение применило формулу $v_{км/ч}=3.6 \\cdot v_{м/с}$ ко всему массиву без построчного цикла. Результат: 4.10, 2.88 и 3.49 км/ч.
"""
        ),
        md("### 3.2. `Series`, `DataFrame` и `groupby()`"),
        code(
            """
s = pd.Series([10, 20, 30], name="value")
frame = pd.DataFrame({"type": ["A", "B", "A"], "value": [10, 20, 30]})
print(type(s).__name__)
print(type(frame).__name__)
display(frame.groupby("type")["value"].sum())
"""
        ),
        md(
            """
**Интерпретация.** `Series` — одномерный именованный массив с индексом, а `DataFrame` — таблица согласованных столбцов. `groupby()` реализует схему «разбить по категории → агрегировать»: сумма для A равна 40, для B — 20.
"""
        ),
        md("### 4. Форматы хранения и загрузка CSV"),
        code(
            """
formats = pd.DataFrame({
    "Организация": ["строки и разделители", "объекты/массивы", "колоночный бинарный"],
    "Схема": ["встроенной строгой схемы нет", "типы выражены структурой JSON", "сохраняется"],
    "Сильная сторона": ["совместимость и читаемость", "вложенные структуры", "сжатие, типы, выборочное чтение"],
    "Ограничение": ["типы выводятся заново", "избыточен для таблиц", "нужен специальный reader"],
}, index=["CSV", "JSON", "Parquet"])
display(formats)

df = pd.read_csv(DATA_PATH, sep=",", encoding="utf-8")
print("Форма:", df.shape)
print("Столбцы:", df.columns.tolist())
display(df.head())
assert df.shape == (26681, 9)
"""
        ),
        md(
            """
**Интерпретация.** Файл читается с разделителем `,` и кодировкой UTF-8 как таблица из 26 681 строк и 9 столбцов. В перечне полей нет времени, поэтому индекс строк нельзя подменять временной осью.
"""
        ),
        md("### 5.1. Структура, типы и память"),
        code(
            """
buf = io.StringIO()
df.info(buf=buf, memory_usage="deep")
print(buf.getvalue())
"""
        ),
        md(
            """
**Интерпретация.** Все 9 полей заполнены. Семь столбцов распознаны как `float64`, `type_id` — как `int64`, а названия категорий — как строковый dtype `str` (в старых версиях pandas он отображался как `object`). Пять повторяющихся категорий являются кандидатом на `category`.
"""
        ),
        md("### 5.2. Пропуски, дубликаты и кардинальность"),
        code(
            """
quality = pd.DataFrame({
    "missing": df.isna().sum(),
    "unique": df.nunique(dropna=False),
})
display(quality)
print("Полных дубликатов:", int(df.duplicated().sum()))
"""
        ),
        md(
            """
**Интерпретация.** Пропусков и полных дубликатов нет, поэтому удаление строк не требуется. Это не доказывает семантическую корректность: например, пять значений `type_id` выглядят как числа, но обозначают категории.
"""
        ),
        md("### 5.3. Соответствие кода и названия категории"),
        code(
            """
type_map = (
    df[["type_id", "type_name"]]
      .drop_duplicates()
      .sort_values("type_id")
)
display(type_map)
assert type_map["type_id"].is_unique and type_map["type_name"].is_unique
"""
        ),
        md(
            """
**Интерпретация.** Пять кодов взаимно однозначно соответствуют пяти названиям. `type_id` можно хранить компактным целым типом, но среднее этого столбца содержательно бессмысленно.
"""
        ),
        md("### 5.4. Описательная статистика"),
        code(
            """
numeric_cols = [
    "lon", "lat", "temperature_c", "precipitation_mm",
    "relative_humidity", "wind_speed_ms", "solar_radiation",
]
display(df[numeric_cols].describe().round(2).T)
"""
        ),
        md(
            """
**Интерпретация.** Температура лежит в диапазоне от -34.88 до 31.94 °C при медиане 19.08 °C. Для осадков и скорости ветра максимум заметно выше верхнего квартиля, что указывает на правые хвосты и требует отдельной проверки необычных значений. Единицы влажности и солнечной радиации из файла не выводятся.
"""
        ),
        md("### 6.1. Предобработка и оптимизация типов"),
        code(
            """
before = df.memory_usage(deep=True).sum()
clean_df = df.copy()
clean_df["type_name"] = clean_df["type_name"].astype("category")
clean_df["type_id"] = clean_df["type_id"].astype("int8")
after = clean_df.memory_usage(deep=True).sum()

print(f"До: {before / 1024**2:.2f} MiB")
print(f"После: {after / 1024**2:.2f} MiB")
print(f"Снижение: {(1 - after / before) * 100:.1f}%")
assert len(clean_df) == len(df)
"""
        ),
        md(
            """
**Интерпретация.** Оптимизация меняет представление, а не смысл данных: все строки сохранены. `category` хранит повторяющиеся названия через компактные коды, а диапазон 1-5 безопасно помещается в `int8`. В текущем окружении pandas 3.0 память сократилась на 30.8%; отличие от значения в методичке связано с более компактным новым строковым dtype.
"""
        ),
        md("### 6.2. Вычисляемые признаки"),
        code(
            """
clean_df["wind_speed_kmh"] = (clean_df["wind_speed_ms"] * 3.6).round(2)
clean_df["is_forest_fire"] = clean_df["type_name"].eq("Forest fire")
display(clean_df[["type_name", "wind_speed_ms", "wind_speed_kmh", "is_forest_fire"]].head())
"""
        ),
        md(
            """
**Интерпретация.** `wind_speed_kmh = 3.6 * wind_speed_ms` лишь переводит единицу представления. `is_forest_fire` — логическое правило принадлежности категории и может использоваться как бинарный индикатор.
"""
        ),
        md("### 7.1. Частоты категорий"),
        code(
            """
type_stats = (
    clean_df["type_name"]
      .value_counts()
      .to_frame("n")
      .assign(share_pct=lambda x: x["n"] / len(clean_df) * 100)
)
display(type_stats.round(2))
"""
        ),
        md(
            """
**Интерпретация.** `Forest fire` составляет около 69.02% строк, а `Peat fire` — около 0.10% (27 наблюдений). Категории резко несбалансированы, поэтому средние и будущие модельные метрики следует оценивать с учётом размера групп.
"""
        ),
        md("### 7.2. Групповые статистики"),
        code(
            """
group_stats = (
    clean_df.groupby("type_name", observed=True)
      .agg(
          n=("type_name", "size"),
          temp_mean=("temperature_c", "mean"),
          precip_mean=("precipitation_mm", "mean"),
          humidity_mean=("relative_humidity", "mean"),
          wind_mean=("wind_speed_ms", "mean"),
          solar_mean=("solar_radiation", "mean"),
      )
      .sort_values("n", ascending=False)
)
display(group_stats.round(2))
"""
        ),
        md(
            """
**Интерпретация.** Сначала следует сравнивать `n`, затем средние. Например, статистика `Peat fire` основана только на 27 строках и не обладает той же устойчивостью, что статистика `Forest fire` из 18 415 строк. Различия средних не доказывают причинность.
"""
        ),
        md("### 7.3. Составной фильтр через `loc`"),
        code(
            """
subset = clean_df.loc[
    (clean_df["relative_humidity"] < 40) & (clean_df["wind_speed_ms"] > 2),
    ["type_name", "lon", "lat", "temperature_c", "relative_humidity", "wind_speed_ms"],
]
print("Строк в срезе:", len(subset))
display(subset.head())
"""
        ),
        md(
            """
**Интерпретация.** Двум учебным условиям одновременно соответствуют 337 строк. Пороги выбраны для демонстрации логического «И» и не являются нормативным критерием пожарной опасности.
"""
        ),
        md("### 8.1. Корреляционная матрица"),
        code(
            """
weather = [
    "temperature_c", "precipitation_mm", "relative_humidity",
    "wind_speed_ms", "solar_radiation",
]
corr = clean_df[weather].corr()
display(corr.round(3))
"""
        ),
        md(
            """
**Интерпретация.** Очень сильных линейных связей нет. Наиболее заметные по модулю коэффициенты находятся около 0.30; например, температура и скорость ветра связаны слабо отрицательно. Корреляция около нуля не исключает нелинейной связи и не устанавливает причинность.
"""
        ),
        md("### 8.2. Потенциальные выбросы скорости ветра по IQR"),
        code(
            """
q1 = clean_df["wind_speed_ms"].quantile(0.25)
q3 = clean_df["wind_speed_ms"].quantile(0.75)
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
outlier_mask = ~clean_df["wind_speed_ms"].between(lower, upper)
print(f"Q1={q1:.3f}; Q3={q3:.3f}; IQR={iqr:.3f}")
print(f"Границы: [{lower:.3f}; {upper:.3f}]")
print(f"Кандидатов: {outlier_mask.sum()} ({outlier_mask.mean() * 100:.2f}% строк)")
"""
        ),
        md(
            """
**Интерпретация.** Правило IQR помечает 3 652 наблюдения (13.69%). Такая доля больше похожа на выраженный правый хвост распределения, чем на массовую ошибку. Значения не удаляются без проверки источника и предметного обоснования.
"""
        ),
        md("### 9.1. Категориальное сравнение"),
        code(
            """
counts = clean_df["type_name"].value_counts().sort_values()
ax = counts.plot(kind="barh", figsize=(8, 4.5), color="#3b82a0")
ax.set_title("Количество наблюдений по типу события")
ax.set_xlabel("Число наблюдений")
ax.set_ylabel("Тип события")
plt.tight_layout()
plt.show()
"""
        ),
        md("**Интерпретация.** График наглядно подтверждает доминирование `Forest fire` и крайне малую представленность `Peat fire`."),
        md("### 9.2. Распределение температуры"),
        code(
            """
plt.figure(figsize=(8, 4.5))
plt.hist(clean_df["temperature_c"], bins=30, edgecolor="white", color="#d97706")
plt.title("Распределение температуры")
plt.xlabel("Температура, °C")
plt.ylabel("Частота")
plt.tight_layout()
plt.show()
"""
        ),
        md("**Интерпретация.** Основная масса значений положительна, но заметен длинный левый хвост до -34.88 °C. Сам график не даёт основания объявить эти наблюдения ошибочными."),
        md("### 9.3. Температура и относительная влажность"),
        code(
            """
sample = clean_df.sample(n=5000, random_state=42)
plt.figure(figsize=(8, 5))
plt.scatter(sample["temperature_c"], sample["relative_humidity"], s=10, alpha=0.35, color="#287271")
plt.title("Температура и относительная влажность (5000 строк)")
plt.xlabel("Температура, °C")
plt.ylabel("Относительная влажность (единица не указана)")
plt.tight_layout()
plt.show()
"""
        ),
        md("**Интерпретация.** Широкое облако соответствует слабой отрицательной линейной связи: при росте температуры влажность в среднем немного снижается, но разброс велик."),
        md("### 9.4. Пространственная структура"),
        code(
            """
plt.figure(figsize=(8, 5.5))
for name, part in clean_df.groupby("type_name", observed=True):
    plt.scatter(part["lon"], part["lat"], s=6, alpha=0.35, label=name)
plt.xlabel("Долгота")
plt.ylabel("Широта")
plt.title("Пространственное распределение наблюдений")
plt.legend(markerscale=2, fontsize=8)
plt.tight_layout()
plt.show()
"""
        ),
        md(
            """
**Интерпретация.** Наблюдения образуют несколько пространственных кластеров и занимают широкий диапазон координат. Это разведочный scatter plot на декартовых осях, а не карта: полноценная географическая трактовка требует проекции и границ регионов.
"""
        ),
        md("### 10. Экспорт и проверка round-trip"),
        code(
            """
fires_csv = OUT / "FiresRu_clean.csv"
fires_json = OUT / "FiresRu_clean.json"
fires_parquet = OUT / "FiresRu_clean.parquet"

clean_df.to_csv(fires_csv, index=False, encoding="utf-8")
clean_df.to_json(fires_json, orient="records", lines=True, force_ascii=False)
clean_df.to_parquet(fires_parquet, index=False, engine="pyarrow", compression="snappy")

size_table = pd.DataFrame({
    "format": ["CSV", "JSON Lines", "Parquet"],
    "size_MiB": [p.stat().st_size / 1024**2 for p in [fires_csv, fires_json, fires_parquet]],
}).set_index("format")
display(size_table.round(3))

csv_back = pd.read_csv(fires_csv)
json_back = pd.read_json(fires_json, lines=True)
parquet_back = pd.read_parquet(fires_parquet)
round_trip = pd.DataFrame({
    "shape": [csv_back.shape, json_back.shape, parquet_back.shape],
    "type_name_dtype": [str(csv_back["type_name"].dtype), str(json_back["type_name"].dtype), str(parquet_back["type_name"].dtype)],
    "type_id_dtype": [str(csv_back["type_id"].dtype), str(json_back["type_id"].dtype), str(parquet_back["type_id"].dtype)],
}, index=["CSV", "JSON Lines", "Parquet"])
display(round_trip)

assert csv_back.shape == clean_df.shape
assert json_back.shape == clean_df.shape
assert parquet_back.shape == clean_df.shape
assert str(parquet_back["type_name"].dtype) == "category"
assert str(parquet_back["type_id"].dtype) == "int8"
"""
        ),
        md(
            """
**Интерпретация.** Все три файла содержат одинаковое число строк и столбцов и успешно читаются обратно. CSV и JSON восстанавливают типы из текста, поэтому теряют `category`/`int8`; Parquet сохраняет схему и компактные аналитические типы. Размеры измерены в текущем окружении, а не переписаны из методички.
"""
        ),
        md(
            """
## 11. Итоговые выводы

1. Исходная таблица содержит 26 681 строку и 9 признаков; пропусков и полных дубликатов нет.
2. Пять кодов `type_id` взаимно однозначно соответствуют пяти категориям, поэтому код нельзя трактовать как непрерывную величину.
3. `Forest fire` формирует около 69.02% строк, а `Peat fire` — лишь 27 наблюдений; выборка сильно несбалансирована.
4. Приведение `type_name` к `category`, а `type_id` к `int8` уменьшает память на 30.8% без удаления данных; результат зависит от версии pandas.
5. Температура варьируется от -34.88 до 31.94 °C, а распределения осадков и скорости ветра имеют длинные правые хвосты.
6. Между погодными признаками нет очень сильных линейных корреляций; коэффициент около нуля не исключает иных форм зависимости.
7. IQR помечает 13.69% скоростей ветра, поэтому автоматическое удаление кандидатов было бы необоснованным.
8. Координаты показывают пространственные кластеры, но scatter plot без проекции не является картой.
9. Parquet сохраняет аналитические dtypes, тогда как CSV/JSON требуют повторного распознавания схемы.

**Ограничения:** (1) в файле нет времени, поэтому нельзя исследовать сезонность и динамику; (2) единицы `relative_humidity` и `solar_radiation` не подтверждены метаданными; (3) наблюдаемые связи описательны и не доказывают причинность; (4) выводы относятся к предоставленному набору, а не ко всем пожарам России.
"""
        ),
    ]

    nb = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python (TIABD PR1)",
                "language": "python",
                "name": "tiabd-pr1",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
    )
    nbf.write(nb, SUBMISSION / "01_FiresRu.ipynb")


def build_own() -> None:
    cells = [
        md(
            """
# Практическая работа № 1
## Самостоятельный анализ: UCI Bike Sharing Dataset

### 1. Постановка задачи

**Цель:** исследовать почасовой спрос на прокат велосипедов и его связь с сезоном, погодой, температурой и рабочими днями, а также проверить временную динамику и воспроизводимый экспорт.

**Источник:** Hadi Fanaee-T, *Bike Sharing Dataset*, UCI Machine Learning Repository, DOI: [10.24432/C5W894](https://doi.org/10.24432/C5W894). Лицензия на странице UCI — CC BY 4.0. Локальные сведения и оригинальный README сохранены в `data/SOURCE.md` и `data/UCI_Readme.txt`.

Данные агрегированы по часам за 2011-2012 годы для Capital Bikeshare (Вашингтон, США). Персональных записей нет: строки содержат суммарное число прокатов, календарные и погодные признаки.
"""
        ),
        md("### Окружение и проверка CSV"),
        code(
            """
from pathlib import Path
import csv
import io
import sys

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display

print("Python:", sys.version.split()[0])
print("pandas:", pd.__version__)
print("NumPy:", np.__version__)
print("Matplotlib:", matplotlib.__version__)

NOTEBOOK_DIR = Path.cwd()
DATA_PATH = NOTEBOOK_DIR / "data" / "bike_sharing_hour.csv"
OUT = NOTEBOOK_DIR / "output" / "own"
OUT.mkdir(parents=True, exist_ok=True)

raw_bytes = DATA_PATH.read_bytes()
encoding = "ascii" if all(byte < 128 for byte in raw_bytes[:100_000]) else "utf-8"
sample_text = raw_bytes[:10_000].decode(encoding)
delimiter = csv.Sniffer().sniff(sample_text).delimiter
print("Файл:", DATA_PATH.resolve())
print("Кодировка:", encoding)
print("Разделитель:", repr(delimiter))

raw_df = pd.read_csv(DATA_PATH, sep=delimiter, encoding=encoding)
print("Форма:", raw_df.shape)
display(raw_df.head())
assert raw_df.shape == (17379, 17)
"""
        ),
        md(
            """
**Интерпретация.** Файл однозначно распознан как ASCII/UTF-8 CSV с запятой. Он содержит 17 379 почасовых записей и 17 исходных полей, поэтому превышает минимумы задания.
"""
        ),
        md("## 2. Паспорт данных"),
        code(
            """
data_dictionary = pd.DataFrame([
    ("instant", "int", "Технический порядковый номер записи"),
    ("dteday", "date", "Календарная дата"),
    ("season", "category", "Сезон: 1 spring, 2 summer, 3 fall, 4 winter"),
    ("yr", "category", "Год: 0 = 2011, 1 = 2012"),
    ("mnth", "category", "Месяц 1-12"),
    ("hr", "category", "Час 0-23"),
    ("holiday", "binary", "Праздничный день"),
    ("weekday", "category", "День недели"),
    ("workingday", "binary", "Не выходной и не праздник"),
    ("weathersit", "category", "Категория погодной ситуации 1-4"),
    ("temp", "float", "Нормированная температура; исходная шкала делилась на 41"),
    ("atemp", "float", "Нормированная ощущаемая температура; шкала делилась на 50"),
    ("hum", "float", "Нормированная влажность; шкала делилась на 100"),
    ("windspeed", "float", "Нормированная скорость ветра; шкала делилась на 67"),
    ("casual", "int", "Число прокатов незарегистрированных пользователей"),
    ("registered", "int", "Число прокатов зарегистрированных пользователей"),
    ("cnt", "int", "Общее число прокатов: casual + registered"),
], columns=["Поле", "Смысловой тип", "Описание"])
display(data_dictionary)
print(raw_df.dtypes)
"""
        ),
        code(
            """
buf = io.StringIO()
raw_df.info(buf=buf, memory_usage="deep")
print(buf.getvalue())
display(raw_df.describe(include="all").T)
"""
        ),
        md(
            """
**Интерпретация.** При первичном чтении дата является строкой, а календарные коды — целыми числами. Это технически корректно, но не отражает их смысл и будет исправлено. `instant` уникален и служит только номером строки.
"""
        ),
        md("## 3. Качество и предобработка"),
        code(
            """
quality = pd.DataFrame({
    "missing": raw_df.isna().sum(),
    "unique": raw_df.nunique(dropna=False),
})
display(quality)
print("Полных дубликатов:", int(raw_df.duplicated().sum()))
print("instant уникален:", bool(raw_df["instant"].is_unique))
print("cnt = casual + registered:", bool(raw_df["cnt"].eq(raw_df["casual"] + raw_df["registered"]).all()))
print("temp, atemp, hum, windspeed находятся в [0, 1]:", bool(raw_df[["temp", "atemp", "hum", "windspeed"]].apply(lambda s: s.between(0, 1).all()).all()))
"""
        ),
        md(
            """
**Интерпретация.** Ячеечных пропусков и полных дублей нет, контрольная сумма `cnt = casual + registered` выполняется для всех строк, а нормированные признаки лежат в ожидаемом диапазоне. Удалять наблюдения нет оснований; вместо этого уточняются типы и проверяется полнота временной сетки.
"""
        ),
        code(
            """
before = raw_df.memory_usage(deep=True).sum()
typed_df = raw_df.copy()
typed_df["dteday"] = pd.to_datetime(typed_df["dteday"], format="%Y-%m-%d", errors="raise")
for col in ["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit"]:
    typed_df[col] = typed_df[col].astype("int8")
after = typed_df.memory_usage(deep=True).sum()

print(f"Память до: {before / 1024**2:.2f} MiB")
print(f"Память после типизации: {after / 1024**2:.2f} MiB")
print(f"Снижение: {(1 - after / before) * 100:.1f}%")

clean_df = typed_df.drop(columns="instant").copy()
season_labels = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}
weather_labels = {
    1: "Clear/partly cloudy",
    2: "Mist/cloudy",
    3: "Light rain/snow",
    4: "Heavy rain/snow",
}
clean_df["season_name"] = clean_df["season"].map(season_labels).astype("category")
clean_df["weather_name"] = clean_df["weathersit"].map(weather_labels).astype("category")
print(clean_df.dtypes)
"""
        ),
        md(
            """
**Обоснование.** Дата приведена к `datetime64[us]`, небольшие коды — к `int8`, а подписи сезона и погоды — к `category`. Это уменьшило память типизированных исходных полей на 45.2%. `instant` удалён из аналитической таблицы как технический уникальный номер, не несущий признака спроса.
"""
        ),
        md("### Вычисляемые признаки и временная полнота"),
        code(
            """
clean_df["timestamp"] = clean_df["dteday"] + pd.to_timedelta(clean_df["hr"], unit="h")
clean_df["temp_c"] = (clean_df["temp"] * 41).round(2)
clean_df["feels_like_c"] = (clean_df["atemp"] * 50).round(2)
clean_df["humidity_pct"] = (clean_df["hum"] * 100).round(1)
clean_df["wind_speed_scaled"] = (clean_df["windspeed"] * 67).round(2)
clean_df["registered_share_pct"] = (clean_df["registered"] / clean_df["cnt"] * 100).round(2)

full_hours = pd.date_range(clean_df["timestamp"].min(), clean_df["timestamp"].max(), freq="h")
missing_hours = full_hours.difference(clean_df["timestamp"])
print("Период:", clean_df["timestamp"].min(), "-", clean_df["timestamp"].max())
print("Дубликатов timestamp:", int(clean_df["timestamp"].duplicated().sum()))
print("Отсутствующих часов внутри диапазона:", len(missing_hours))
display(clean_df[["timestamp", "temp", "temp_c", "registered_share_pct"]].head())
"""
        ),
        md(
            """
**Интерпретация.** Формулы возвращают температуру в °C (`temp_c = 41 * temp`) и долю зарегистрированных прокатов (`100 * registered / cnt`). Временные метки уникальны, но внутри полного часового диапазона отсутствуют 165 часов: отсутствие пустых ячеек не означает непрерывность временного ряда.
"""
        ),
        md("### Составной фильтр через `loc`"),
        code(
            """
morning_workday = clean_df.loc[
    (clean_df["workingday"] == 1)
    & (clean_df["weather_name"] == "Clear/partly cloudy")
    & (clean_df["temp_c"] >= 20)
    & (clean_df["hr"].between(7, 9)),
    ["timestamp", "season_name", "temp_c", "weather_name", "cnt"],
]
print("Строк в срезе:", len(morning_workday))
print("Среднее число прокатов:", round(morning_workday["cnt"].mean(), 2))
print("Медиана:", morning_workday["cnt"].median())
display(morning_workday.head())
"""
        ),
        md(
            """
**Интерпретация.** Условиям рабочего утра 07:00-09:00, ясной/переменно облачной погоды и температуры не ниже 20 °C соответствуют 383 часа. Среднее число прокатов в них — около 408.47, медиана — 370; это описательный срез, а не доказательство влияния каждого условия.
"""
        ),
        md("## 4. EDA"),
        md("### Группировки и агрегации"),
        code(
            """
season_stats = (
    clean_df.groupby("season_name", observed=True)
    .agg(
        n_hours=("cnt", "size"),
        total_rentals=("cnt", "sum"),
        mean_per_hour=("cnt", "mean"),
        median_per_hour=("cnt", "median"),
        mean_temp_c=("temp_c", "mean"),
    )
    .sort_values("mean_per_hour", ascending=False)
)
display(season_stats.round(2))

weather_stats = (
    clean_df.groupby("weather_name", observed=True)
    .agg(n_hours=("cnt", "size"), total_rentals=("cnt", "sum"), mean_per_hour=("cnt", "mean"))
    .sort_values("mean_per_hour", ascending=False)
)
display(weather_stats.round(2))
"""
        ),
        md(
            """
**Интерпретация.** Наибольшее среднее число прокатов наблюдается осенью (`Fall`) — 236.02 в час, наименьшее весной (`Spring`) — 111.11. При ясной/переменно облачной погоде среднее равно 204.87, при лёгком дожде/снеге — 111.58. Категория тяжёлой погоды содержит только 3 часа, поэтому её среднее 74.33 статистически неустойчиво.
"""
        ),
        md("### Описательные статистики и корреляция"),
        code(
            """
eda_cols = ["cnt", "temp_c", "humidity_pct", "wind_speed_scaled", "registered_share_pct"]
display(clean_df[eda_cols].describe().round(2).T)
display(clean_df[["cnt", "temp_c", "humidity_pct", "windspeed"]].corr().round(3))
"""
        ),
        md(
            """
**Интерпретация.** Почасовой спрос асимметричен: среднее 189.46 выше медианы 142, максимум — 977. Корреляция спроса с температурой умеренно положительна (`r = 0.405`), с влажностью — умеренно отрицательна (`r = -0.323`), а со скоростью ветра близка к нулю (`r = 0.093`). Эти коэффициенты не доказывают причинность.
"""
        ),
        md("### Кандидаты на выбросы по IQR"),
        code(
            """
q1 = clean_df["cnt"].quantile(0.25)
q3 = clean_df["cnt"].quantile(0.75)
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
cnt_outliers = ~clean_df["cnt"].between(lower, upper)
print(f"Q1={q1:.1f}; Q3={q3:.1f}; IQR={iqr:.1f}")
print(f"Границы: [{lower:.1f}; {upper:.1f}]")
print(f"Кандидатов: {cnt_outliers.sum()} ({cnt_outliers.mean() * 100:.2f}%)")
"""
        ),
        md(
            """
**Интерпретация.** Верхняя граница IQR равна 642.5 проката, выше неё лежат 505 часов (2.91%). Высокий спрос реалистичен для часов пик, поэтому кандидаты не удаляются автоматически.
"""
        ),
        md("### Визуализация 1. Категориальное сравнение"),
        code(
            """
order = season_stats.sort_values("mean_per_hour").index
ax = season_stats.loc[order, "mean_per_hour"].plot(kind="barh", figsize=(8, 4.5), color="#3b82a0")
ax.set_title("Среднее число прокатов в час по сезону")
ax.set_xlabel("Прокатов в час")
ax.set_ylabel("Сезон")
plt.tight_layout()
plt.show()
"""
        ),
        md("**Интерпретация.** Средний почасовой спрос выше всего для `Fall` и ниже всего для `Spring`; сезонные различия могут одновременно отражать погоду, тренд между годами и структуру календаря."),
        md("### Визуализация 2. Распределение спроса"),
        code(
            """
plt.figure(figsize=(8, 4.5))
plt.hist(clean_df["cnt"], bins=40, edgecolor="white", color="#d97706")
plt.axvline(clean_df["cnt"].median(), color="black", linestyle="--", label="Медиана")
plt.title("Распределение почасового числа прокатов")
plt.xlabel("Прокатов за час")
plt.ylabel("Частота")
plt.legend()
plt.tight_layout()
plt.show()
"""
        ),
        md("**Интерпретация.** Распределение имеет длинный правый хвост: большинство часов значительно ниже максимума 977, а среднее превышает медиану."),
        md("### Визуализация 3. Температура и спрос"),
        code(
            """
sample = clean_df.sample(n=5000, random_state=42)
plt.figure(figsize=(8, 5))
plt.scatter(sample["temp_c"], sample["cnt"], s=10, alpha=0.3, color="#287271")
plt.title("Температура и почасовое число прокатов (5000 строк)")
plt.xlabel("Температура, °C")
plt.ylabel("Прокатов за час")
plt.tight_layout()
plt.show()
"""
        ),
        md("**Интерпретация.** При более высокой температуре верхняя граница спроса в среднем растёт, но большой вертикальный разброс показывает влияние часа, сезона, года и других факторов."),
        md("### Визуализация 4. Временной ряд"),
        code(
            """
monthly = clean_df.resample("MS", on="dteday")["cnt"].sum()
plt.figure(figsize=(10, 4.5))
plt.plot(monthly.index, monthly.values, marker="o", linewidth=1.8, color="#7c3aed")
plt.title("Суммарное число прокатов по месяцам")
plt.xlabel("Месяц")
plt.ylabel("Прокатов")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()
display(monthly.rename("rentals").to_frame().head())
"""
        ),
        md(
            """
**Интерпретация.** В 2012 году месячные суммы в целом выше, чем в 2011-м, с сезонными подъёмами в тёплые месяцы. Максимум приходится на сентябрь 2012 года (218 573 проката), минимум — на январь 2011 года (38 189). Суммы рассчитаны по имеющимся строкам; 165 часов отсутствуют во временной сетке.
"""
        ),
        md("## 5. Интерпретация"),
        md(
            """
Анализ показывает совместное действие календаря, погоды и общего временного тренда. Сезонные и погодные средние нельзя трактовать изолированно: например, рост популярности сервиса между 2011 и 2012 годами способен повышать средние поздних периодов. Поэтому результаты описывают наблюдения и формируют гипотезы, но не оценивают причинный эффект температуры или погоды.
"""
        ),
        md("## 6. Экспорт и повторное чтение"),
        code(
            """
own_csv = OUT / "bike_sharing_clean.csv"
own_json = OUT / "bike_sharing_clean.json"
own_parquet = OUT / "bike_sharing_clean.parquet"

clean_df.to_csv(own_csv, index=False, encoding="utf-8")
clean_df.to_json(own_json, orient="records", lines=True, force_ascii=False, date_format="iso")
clean_df.to_parquet(own_parquet, index=False, engine="pyarrow", compression="snappy")

sizes = pd.DataFrame({
    "format": ["CSV", "JSON Lines", "Parquet"],
    "size_MiB": [p.stat().st_size / 1024**2 for p in [own_csv, own_json, own_parquet]],
}).set_index("format")
display(sizes.round(3))

csv_back = pd.read_csv(own_csv)
json_back = pd.read_json(own_json, lines=True)
parquet_back = pd.read_parquet(own_parquet)
round_trip = pd.DataFrame({
    "shape": [csv_back.shape, json_back.shape, parquet_back.shape],
    "dteday_dtype": [str(csv_back["dteday"].dtype), str(json_back["dteday"].dtype), str(parquet_back["dteday"].dtype)],
    "season_name_dtype": [str(csv_back["season_name"].dtype), str(json_back["season_name"].dtype), str(parquet_back["season_name"].dtype)],
}, index=["CSV", "JSON Lines", "Parquet"])
display(round_trip)

assert csv_back.shape == clean_df.shape
assert json_back.shape == clean_df.shape
assert parquet_back.shape == clean_df.shape
assert pd.api.types.is_datetime64_any_dtype(parquet_back["dteday"])
assert isinstance(parquet_back["season_name"].dtype, pd.CategoricalDtype)
"""
        ),
        md(
            """
**Интерпретация.** Все три формата прошли round-trip и сохранили форму таблицы. CSV и JSON требуют повторного распознавания дат и категорий, а Parquet сохранил `datetime` и `category`, поэтому удобнее для следующего аналитического этапа.
"""
        ),
        md(
            """
## 7. Итог

### Выводы

1. Набор содержит 17 379 почасовых наблюдений, 17 исходных полей, не имеет пустых ячеек и полных дубликатов; `cnt` всегда равен `casual + registered`.
2. После приведения даты и малых кодов к подходящим типам память исходных полей сократилась на 45.2%.
3. Средний почасовой спрос равен 189.46, медиана — 142, максимум — 977; распределение имеет правый хвост.
4. Наибольший средний спрос наблюдается в сезоне `Fall` (236.02 проката/час), наименьший — в `Spring` (111.11).
5. При ясной/переменно облачной погоде средний спрос (204.87) выше, чем при лёгком дожде/снеге (111.58), но причинность из групповых средних не следует.
6. Спрос умеренно положительно коррелирует с температурой (`r = 0.405`) и отрицательно — с влажностью (`r = -0.323`).
7. IQR помечает 505 часов (2.91%) с высоким спросом; они сохранены как возможные реальные часы пик.
8. Месячный максимум — сентябрь 2012 года (218 573 проката), минимум — январь 2011 года (38 189); виден общий рост и сезонность.

### Ограничения

1. Данные относятся к одной системе и двум годам; результаты нельзя автоматически переносить на другие города и периоды.
2. Внутри часового диапазона отсутствуют 165 временных меток, хотя пропусков в самих строках нет; месячные суммы рассчитаны по доступным наблюдениям.
3. В данных нет станций, маршрутов и индивидуальной длительности поездок, поэтому пространственный и пользовательский анализ невозможен.
4. Исследование наблюдательное: корреляции и групповые различия не доказывают причинный эффект погоды или календаря.
5. Для категории тяжёлой погоды есть только 3 часа, поэтому её среднее ненадёжно.
"""
        ),
    ]

    nb = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python (TIABD PR1)",
                "language": "python",
                "name": "tiabd-pr1",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
    )
    nbf.write(nb, SUBMISSION / "02_Own_Dataset.ipynb")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    if target in {"fires", "all"}:
        build_fires()
    if target in {"own", "all"}:
        build_own()
