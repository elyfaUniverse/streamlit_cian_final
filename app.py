from __future__ import annotations

import html
import json
import math
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

try:
    import folium
    from streamlit_folium import st_folium

    HAS_MAP = True
except Exception:
    HAS_MAP = False


APP_DIR = Path(__file__).resolve().parent
DATA_PATHS = [
    APP_DIR / "eda_result.csv",
    APP_DIR / "eda.csv",
    APP_DIR / "cian_processed.csv",
]
MODEL_PATHS = [
    APP_DIR / "roman_xgboost_pipeline.pkl",
]
METADATA_PATHS = [
    APP_DIR / "roman_xgboost_pipeline_metadata.json",
]
METRICS_PATHS = [
    APP_DIR / "model_metrics.json",
]
SPLIT_DATA_PATHS = [
    APP_DIR / "split_data.pkl",
]

TARGET = "price"
CENTER_LAT = 59.939095
CENTER_LNG = 30.315868
CURRENT_YEAR = 2026
QUALITY_SAMPLE_SIZE = 8000
BATCH_MAX_ROWS = 1000

px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = ["#0f8b8d", "#b7791f", "#27714a", "#b54d3d", "#65717f"]

PREDICTION_PRESETS: dict[str, dict[str, Any]] = {
    "Кастомные значения": {},
    "Студия в новостройке, Приморский район": {
        "region": "Санкт-Петербург",
        "district": "Приморский",
        "nearest_metro": "Комендантский проспект",
        "metro_travel_type": "transport",
        "metro_travel_time": 18,
        "area_total": 28.0,
        "area_living": 16.0,
        "area_kitchen": 6.0,
        "rooms": "studio",
        "is_studio": True,
        "build_year": 2024,
        "total_floors": 25,
        "current_floor": 12,
        "status_home": "Сдан",
        "type_building": "Монолитный",
        "is_new_building": True,
        "lat": 60.0080,
        "lng": 30.2600,
    },
    "2-комнатная квартира у метро, Московский район": {
        "region": "Санкт-Петербург",
        "district": "Московский",
        "nearest_metro": "Московская",
        "metro_travel_type": "walk",
        "metro_travel_time": 8,
        "area_total": 55.0,
        "area_living": 32.0,
        "area_kitchen": 10.0,
        "rooms": "2",
        "is_studio": False,
        "build_year": 2012,
        "total_floors": 16,
        "current_floor": 7,
        "status_home": "Сдан",
        "type_building": "Монолитно-кирпичный",
        "is_new_building": False,
        "lat": 59.8525,
        "lng": 30.3211,
    },
    "Просторная квартира в центре": {
        "region": "Санкт-Петербург",
        "district": "Центральный",
        "nearest_metro": "Невский проспект",
        "metro_travel_type": "walk",
        "metro_travel_time": 5,
        "area_total": 110.0,
        "area_living": 70.0,
        "area_kitchen": 18.0,
        "rooms": "3",
        "is_studio": False,
        "build_year": 1910,
        "total_floors": 6,
        "current_floor": 3,
        "status_home": "Сдан",
        "type_building": "Кирпичный",
        "is_new_building": False,
        "lat": 59.9343,
        "lng": 30.3351,
    },
    "Семейная квартира в Ленинградской области": {
        "region": "Ленинградская область",
        "district": "Всеволожский",
        "nearest_metro": "Девяткино",
        "metro_travel_type": "transport",
        "metro_travel_time": 25,
        "area_total": 68.0,
        "area_living": 42.0,
        "area_kitchen": 12.0,
        "rooms": "3",
        "is_studio": False,
        "build_year": 2020,
        "total_floors": 18,
        "current_floor": 9,
        "status_home": "Сдан",
        "type_building": "Монолитный",
        "is_new_building": True,
        "lat": 60.0500,
        "lng": 30.4500,
    },
    "2-комнатная вторичка на первом этаже": {
        "region": "Санкт-Петербург",
        "district": "Невский",
        "nearest_metro": "Улица Дыбенко",
        "metro_travel_type": "walk",
        "metro_travel_time": 12,
        "area_total": 45.0,
        "area_living": 28.0,
        "area_kitchen": 8.0,
        "rooms": "2",
        "is_studio": False,
        "build_year": 1985,
        "total_floors": 9,
        "current_floor": 1,
        "status_home": "Сдан",
        "type_building": "Панельный",
        "is_new_building": False,
        "lat": 59.9070,
        "lng": 30.4830,
    },
}


st.set_page_config(
    page_title="Cian: прогноз цены",
    page_icon="🏙️",
    layout="wide",
)

def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    data_path = first_existing(DATA_PATHS)
    if data_path is None:
        st.error("Не найден датасет: положите рядом с app.py файл eda_result.csv, eda.csv или cian_processed.csv.")
        st.stop()

    df = pd.read_csv(data_path)
    if "price" not in df.columns and "cost" in df.columns:
        df = df.rename(columns={"cost": "price"})

    if "log_price" not in df.columns and "price" in df.columns:
        df["log_price"] = np.log(df["price"])
    if "log_area" not in df.columns and "area_total" in df.columns:
        df["log_area"] = np.log(df["area_total"].clip(lower=1))
    if "age" not in df.columns and "build_year" in df.columns:
        df["age"] = CURRENT_YEAR - df["build_year"]
    if "is_spb" not in df.columns and "region" in df.columns:
        df["is_spb"] = df["region"].eq("Санкт-Петербург")
    if "is_first_floor" not in df.columns and "current_floor" in df.columns:
        df["is_first_floor"] = df["current_floor"].eq(1)
    if {"current_floor", "total_floors"}.issubset(df.columns) and "is_last_floor" not in df.columns:
        df["is_last_floor"] = df["current_floor"].eq(df["total_floors"])

    for col in ["is_new_building", "is_studio", "is_completed", "is_spb", "is_first_floor", "is_last_floor"]:
        if col in df.columns:
            df[col] = df[col].astype(bool)

    return df


@st.cache_resource(show_spinner=False)
def load_model() -> Any:
    model_path = first_existing(MODEL_PATHS)
    if model_path is None:
        st.error("Не найден файл модели: положите рядом с app.py файл roman_xgboost_pipeline.pkl.")
        st.stop()
    return joblib.load(model_path)


@st.cache_data(show_spinner=False)
def load_metadata() -> dict[str, Any]:
    metadata_path = first_existing(METADATA_PATHS)
    if metadata_path is None:
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_model_metrics() -> dict[str, Any]:
    metrics_path = first_existing(METRICS_PATHS)
    if metrics_path is None:
        st.error("Не найден файл метрик моделей: положите рядом с app.py файл model_metrics.json.")
        st.stop()

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    required_tables = ["leaderboard", "baseline", "top_segments", "top_features"]
    missing = [table for table in required_tables if table not in payload]
    if missing:
        st.error(f"В model_metrics.json не хватает разделов: {', '.join(missing)}.")
        st.stop()

    return {
        "source": payload.get("source", ""),
        "updated_at": payload.get("updated_at", ""),
        "selection_metric": payload.get("selection_metric", "cv_rmse_log"),
        "leaderboard": pd.DataFrame(payload["leaderboard"]),
        "baseline": pd.DataFrame(payload["baseline"]),
        "top_segments": pd.DataFrame(payload["top_segments"]),
        "top_features": pd.DataFrame(payload["top_features"]),
    }


@st.cache_data(show_spinner=False)
def load_split_data() -> dict[str, Any]:
    split_path = first_existing(SPLIT_DATA_PATHS)
    if split_path is None:
        return {}
    try:
        payload = joblib.load(split_path)
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def get_model_features(model: Any) -> tuple[list[str], list[str], list[str]]:
    preprocessor = model.named_steps["preprocessor"]
    cat_features: list[str] = []
    rest_features: list[str] = []
    for name, _, cols in preprocessor.transformers_:
        if name == "cat":
            cat_features = list(cols)
        elif name == "num_bool":
            rest_features = list(cols)
    return cat_features, rest_features, cat_features + rest_features


def get_encoder_categories(model: Any, cat_features: list[str]) -> dict[str, list[str]]:
    preprocessor = model.named_steps["preprocessor"]
    encoder = preprocessor.named_transformers_["cat"]
    return {
        feature: [str(value) for value in values]
        for feature, values in zip(cat_features, encoder.categories_)
    }


def sorted_options(values: pd.Series, fallback: list[str] | None = None) -> list[str]:
    clean = values.dropna().astype(str)
    options = sorted(clean.unique().tolist())
    if options:
        return options
    return fallback or []


OPTION_LABELS = {
    "walk": "Пешком",
    "transport": "Транспортом",
    "studio": "Студия",
    "free_plan": "Свободная планировка",
    "none_type": "Не указан",
}


REVERSE_OPTION_LABELS = {label: value for value, label in OPTION_LABELS.items()}

DISPLAY_COLUMN_NAMES = {
    "price": "Цена",
    "price_pred": "Прогноз",
    "prediction_price_rub": "Прогноз",
    "prediction_price_sqm_rub": "Прогноз за м²",
    "abs_error_rub": "Ошибка, ₽",
    "ape_price": "Ошибка, %",
    "cost_sqm": "Цена за м²",
    "area_total": "Площадь",
    "area_living": "Жилая площадь",
    "area_kitchen": "Кухня",
    "rooms": "Комнаты",
    "region": "Регион",
    "district": "Район",
    "nearest_metro": "Метро",
    "metro_travel_type": "До метро",
    "metro_travel_time": "Мин. до метро",
    "is_new_building": "Тип рынка",
    "is_studio": "Студия",
    "is_spb": "Регион",
    "is_first_floor": "Первый этаж",
    "is_last_floor": "Последний этаж",
    "is_completed": "Дом сдан",
    "build_year": "Год постройки",
    "current_floor": "Этаж",
    "total_floors": "Этажей",
    "type_building": "Тип здания",
    "prediction_status": "Статус",
    "validation_notes": "Проверка",
    "input_similarity_score": "Похожесть",
    "input_similarity_label": "Зона модели",
    "objects": "Объектов",
    "median_price": "Медианная цена",
    "median_area": "Медианная площадь",
    "median_prediction": "Медианный прогноз",
}

PRICE_COLUMNS = {
    "price",
    "price_pred",
    "prediction_price_rub",
    "prediction_price_sqm_rub",
    "abs_error_rub",
    "cost_sqm",
    "median_price",
    "median_prediction",
}
AREA_COLUMNS = {"area_total", "area_living", "area_kitchen", "median_area"}
STATUS_LABELS = {
    "ok": "ОК",
    "warning": "Проверить",
    "error": "Ошибка",
    "not_processed": "Не обработано",
}


def option_label(value: Any) -> str:
    return OPTION_LABELS.get(str(value), str(value))


def room_label(value: Any) -> str:
    label = option_label(value)
    return f"{label} комн." if str(value).isdigit() else label


def normalize_option_label(value: Any) -> Any:
    if pd.isna(value):
        return value
    text = str(value).strip()
    return REVERSE_OPTION_LABELS.get(text, text)


def numeric_or_none(value: Any) -> float | None:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return None
    return float(number)


def display_value(column: str, value: Any, format_numbers: bool = True) -> str:
    if pd.isna(value):
        return ""

    if format_numbers and column in PRICE_COLUMNS:
        number = numeric_or_none(value)
        return format_rub(number) if number is not None else str(value)
    if format_numbers and column in AREA_COLUMNS:
        number = numeric_or_none(value)
        return f"{number:.1f} м²" if number is not None else str(value)
    if format_numbers and column == "ape_price":
        number = numeric_or_none(value)
        return f"{number:.1%}" if number is not None else str(value)
    if column in {"rooms", "metro_travel_type", "type_building"}:
        return option_label(value)
    if column == "is_new_building":
        return "Новостройка" if parse_bool(value) else "Вторичка"
    if column == "is_spb":
        return "СПб" if parse_bool(value) else "ЛО"
    if column == "is_first_floor":
        return "Да" if parse_bool(value) else "Нет"
    if column == "is_last_floor":
        return "Да" if parse_bool(value) else "Нет"
    if column in {"is_studio", "is_completed"}:
        return "Да" if parse_bool(value) else "Нет"
    if column == "prediction_status":
        return STATUS_LABELS.get(str(value), str(value))
    if isinstance(value, (bool, np.bool_)):
        return "Да" if bool(value) else "Нет"
    return str(value)


def format_display_dataframe(df: pd.DataFrame, rename_columns: bool = True, format_numbers: bool = True) -> pd.DataFrame:
    if df.empty:
        return df
    view = df.copy()
    for column in view.columns:
        if column in DISPLAY_COLUMN_NAMES or column in PRICE_COLUMNS or column in AREA_COLUMNS:
            view[column] = view[column].map(
                lambda value, col=column: display_value(col, value, format_numbers=format_numbers)
            )
    if rename_columns:
        view = view.rename(columns=DISPLAY_COLUMN_NAMES)
    return view


def option_index(options: list[str], value: Any, default: int = 0) -> int:
    value_str = str(value)
    return options.index(value_str) if value_str in options else default


def district_options_for_regions(df: pd.DataFrame, regions: list[str], fallback: list[str] | None = None) -> list[str]:
    if {"region", "district"}.issubset(df.columns) and regions:
        region_values = [str(region) for region in regions]
        region_frame = df[df["region"].astype(str).isin(region_values)]
        options = sorted_options(region_frame["district"])
        if options:
            return options
    if fallback:
        return fallback
    if "district" in df.columns:
        return sorted_options(df["district"])
    return []


def metro_options_for_location(
    df: pd.DataFrame,
    region: str | None,
    district: str | None,
    fallback: list[str] | None = None,
) -> list[str]:
    if "nearest_metro" not in df.columns:
        return fallback or []

    location_frame = df
    if region and "region" in location_frame.columns:
        location_frame = location_frame[location_frame["region"].astype(str).eq(str(region))]
    if district and "district" in location_frame.columns:
        location_frame = location_frame[location_frame["district"].astype(str).eq(str(district))]

    options = sorted_options(location_frame["nearest_metro"])
    if options:
        return options
    return fallback or sorted_options(df["nearest_metro"])


def median_number(df: pd.DataFrame, col: str, default: float) -> float:
    if col not in df.columns:
        return default
    value = pd.to_numeric(df[col], errors="coerce").median()
    if pd.isna(value):
        return default
    return float(value)


def prediction_defaults(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "lat": median_number(df, "lat", 59.9343),
        "lng": median_number(df, "lng", 30.3351),
        "area_total": median_number(df, "area_total", 52.0),
        "area_living": median_number(df, "area_living", 30.0),
        "area_kitchen": median_number(df, "area_kitchen", 10.0),
        "build_year": int(median_number(df, "build_year", 2010)),
        "current_floor": int(median_number(df, "current_floor", 5)),
        "total_floors": int(median_number(df, "total_floors", 12)),
        "metro_travel_time": int(median_number(df, "metro_travel_time", 10)),
        "region": "Санкт-Петербург",
        "district": "Центральный",
        "nearest_metro": "Невский проспект",
        "metro_travel_type": "walk",
        "rooms": "2",
        "is_studio": False,
        "status_home": "Сдан",
        "type_building": "Монолитный",
        "is_new_building": False,
    }


def parse_bool(value: Any, default: bool = False) -> bool:
    if pd.isna(value):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "да", "истина", "новостройка", "студия", "сдан", "спб"}:
        return True
    if text in {"0", "false", "no", "n", "нет", "ложь", "вторичка", "не сдан", "ло"}:
        return False
    return default


def interval_label(value: float, labels: list[str]) -> str:
    if not labels:
        return ""

    parsed: list[tuple[float, float, str]] = []
    for label in labels:
        match = re.match(r"^[([]([^,]+),\s*([^\]]+)\]$", label)
        if not match:
            continue
        left = float(match.group(1))
        right = float(match.group(2))
        parsed.append((left, right, label))

    for index, (left, right, label) in enumerate(parsed):
        if index == 0 and left <= value <= right:
            return label
        if left < value <= right:
            return label

    if parsed:
        nearest = min(parsed, key=lambda item: min(abs(value - item[0]), abs(value - item[1])))
        return nearest[2]
    return labels[0]


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def format_rub(value: float) -> str:
    return f"{value:,.0f} ₽".replace(",", " ")


def format_percent(value: float) -> str:
    return f"{value:.1%}"


def format_signed_percent(value: float) -> str:
    return f"{value:+.1%}"


def market_name(value: Any) -> str:
    return "Новостройка" if bool(value) else "Вторичка"


def raw_value_for_field(raw: dict[str, Any], field: str) -> Any:
    if field == "is_first_floor":
        return int(raw["current_floor"]) == 1
    return raw[field]


def filter_by_raw_segment(frame: pd.DataFrame, raw: dict[str, Any], fields: list[str]) -> pd.DataFrame:
    if frame.empty:
        return frame

    mask = pd.Series(True, index=frame.index)
    for field in fields:
        if field not in frame.columns:
            continue
        value = raw_value_for_field(raw, field)
        if pd.api.types.is_bool_dtype(frame[field]):
            mask &= frame[field].fillna(False).astype(bool).eq(bool(value))
        else:
            mask &= frame[field].astype(str).eq(str(value))
    return frame[mask].copy()


def percentile_rank(values: pd.Series, value: float) -> float | None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return None
    return float((clean <= value).mean() * 100)


def safe_price_per_sqm(frame: pd.DataFrame) -> pd.Series:
    if "cost_sqm" in frame.columns:
        return pd.to_numeric(frame["cost_sqm"], errors="coerce")
    if {"price", "area_total"}.issubset(frame.columns):
        price = pd.to_numeric(frame["price"], errors="coerce")
        area = pd.to_numeric(frame["area_total"], errors="coerce").replace(0, np.nan)
        return price / area
    return pd.Series(np.nan, index=frame.index)


def html_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p>Нет данных для таблицы.</p>"
    return df.astype(str).to_html(index=False, escape=True, border=0, classes="report-table")


def stringify_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    view = df.copy()
    for column in view.columns:
        view[column] = view[column].astype(str)
    return view


def price_col_for_analysis(frame: pd.DataFrame) -> str:
    return "cost_sqm" if "cost_sqm" in frame.columns else "price"


def group_median_delta(frame: pd.DataFrame, group_col: str, value_col: str) -> tuple[str, str, float] | None:
    if {group_col, value_col}.issubset(frame.columns) is False:
        return None
    grouped = (
        frame.dropna(subset=[group_col, value_col])
        .groupby(group_col)[value_col]
        .median()
        .sort_values(ascending=False)
    )
    if len(grouped) < 2:
        return None
    top_label = str(grouped.index[0])
    bottom_label = str(grouped.index[-1])
    bottom = float(grouped.iloc[-1])
    if bottom <= 0:
        return None
    return top_label, bottom_label, float(grouped.iloc[0] / bottom - 1)


def build_executive_takeaways(filtered: pd.DataFrame, model_metrics: dict[str, Any]) -> list[str]:
    takeaways: list[str] = []
    if filtered.empty or "price" not in filtered.columns:
        return takeaways

    value_col = price_col_for_analysis(filtered)
    value_label = "цене за м²" if value_col == "cost_sqm" else "полной цене"

    if "region" in filtered.columns:
        region_delta = group_median_delta(filtered, "region", value_col)
        if region_delta:
            top, bottom, delta = region_delta
            takeaways.append(f"По {value_label} самый дорогой регион в текущем срезе — {top}; разрыв с {bottom}: {format_signed_percent(delta)}.")

    if "is_new_building" in filtered.columns:
        market_summary = (
            filtered.assign(market=np.where(filtered["is_new_building"], "новостройки", "вторичка"))
            .groupby("market")[value_col]
            .median()
        )
        if {"новостройки", "вторичка"}.issubset(set(market_summary.index)):
            secondary = float(market_summary["вторичка"])
            if secondary > 0:
                delta = float(market_summary["новостройки"] / secondary - 1)
                takeaways.append(f"Новостройки отличаются от вторички по {value_label} на {format_signed_percent(delta)}.")

    if "district" in filtered.columns:
        district_delta = group_median_delta(filtered, "district", value_col)
        if district_delta:
            top, bottom, delta = district_delta
            takeaways.append(f"Районный разброс заметен: {top} выше {bottom} по {value_label} на {format_signed_percent(delta)}.")

    leaderboard = model_metrics.get("leaderboard")
    if isinstance(leaderboard, pd.DataFrame) and not leaderboard.empty and "test_mape_price" in leaderboard.columns:
        best = leaderboard.iloc[0]
        takeaways.append(
            f"Выбранная по текущему CV модель дает ориентир с test MAPE {float(best['test_mape_price']):.1%}; точечный прогноз надо читать вместе с сегментным интервалом."
        )

    return takeaways[:4]


def show_executive_summary(filtered: pd.DataFrame, model_metrics: dict[str, Any]) -> None:
    st.subheader("Ключевые выводы")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Объектов в срезе", f"{len(filtered):,}".replace(",", " "))
    col2.metric("Медианная цена", format_rub(float(filtered["price"].median())) if "price" in filtered else "н/д")
    col3.metric("Медиана за м²", format_rub(float(filtered["cost_sqm"].median())) if "cost_sqm" in filtered else "н/д")
    if "area_total" in filtered.columns:
        col4.metric("Медианная площадь", f"{float(filtered['area_total'].median()):.1f} м²")
    else:
        col4.metric("Медианная площадь", "н/д")

    takeaways = build_executive_takeaways(filtered, model_metrics)
    if takeaways:
        st.markdown("\n".join(f"- {line}" for line in takeaways))


def add_insight(message: str) -> None:
    st.info(message)


def render_app_header(
    df: pd.DataFrame,
    model_metrics: dict[str, Any],
    metadata: dict[str, Any],
    presentation_mode: bool,
) -> None:
    leaderboard = model_metrics.get("leaderboard")
    if isinstance(leaderboard, pd.DataFrame) and not leaderboard.empty:
        best = leaderboard.iloc[0]
        model_name = str(best.get("model", metadata.get("model_name", "модель")))
        test_mape = best.get("test_mape_price", metadata.get("metrics", {}).get("mape_price", np.nan))
    else:
        model_name = str(metadata.get("model_name", "модель"))
        test_mape = metadata.get("metrics", {}).get("mape_price", np.nan)

    test_mape_text = f"{float(test_mape):.1%}" if not pd.isna(test_mape) else "н/д"
    mode_text = "презентация" if presentation_mode else "полный режим"
    total_rows_text = f"{len(df):,}".replace(",", " ")

    st.markdown("### Прогноз цены квартиры")
    st.caption(
        f"СПб и Ленинградская область · {model_name} · Test MAPE {test_mape_text} · "
        f"{total_rows_text} объявлений · {mode_text}."
    )


def render_section_title(title: str, caption: str | None = None) -> None:
    st.markdown(f"#### {title}")
    if caption:
        st.caption(caption)


def canonical_feature_name(feature: Any) -> str:
    return re.sub(r"^(cat|num_bool|remainder)__", "", str(feature))


def clean_feature_name(feature: str) -> str:
    cleaned = canonical_feature_name(feature)
    exact_labels = {
        "area_total": "Общая площадь",
        "center_travel_km": "Расстояние до центра",
        "log_area": "Логарифм площади",
        "build_year": "Год постройки",
        "metro_travel_km": "Расстояние до метро",
        "metro_travel_time": "Время до метро",
        "area_kitchen": "Площадь кухни",
        "area_living": "Жилая площадь",
        "current_floor": "Этаж",
        "total_floors": "Этажность дома",
        "lat": "Широта",
        "lng": "Долгота",
        "is_new_building": "Новостройка",
        "is_first_floor": "Первый этаж",
        "is_last_floor": "Последний этаж",
        "is_spb": "Санкт-Петербург",
        "is_studio": "Студия",
        "is_completed": "Дом сдан",
    }
    if cleaned in exact_labels:
        return exact_labels[cleaned]

    prefix_labels = {
        "district": "Район",
        "nearest_metro": "Метро",
        "metro_travel_type": "Способ до метро",
        "status_home": "Статус дома",
        "rooms": "Комнаты",
        "region": "Регион",
        "type_building": "Тип здания",
        "lat_bin": "Широта, интервал",
        "lng_bin": "Долгота, интервал",
        "metro_bin": "Метро, интервал",
    }
    for prefix, label in prefix_labels.items():
        marker = f"{prefix}_"
        if cleaned.startswith(marker):
            value = cleaned[len(marker) :]
            return f"{label}: {option_label(value)}"

    return cleaned.replace("_", " ")


def prepare_shap_importance(top_features: pd.DataFrame) -> pd.DataFrame:
    if top_features.empty or not {"feature", "importance"}.issubset(top_features.columns):
        return pd.DataFrame()
    view = top_features[["feature", "importance"]].copy()
    view["feature_key"] = view["feature"].map(canonical_feature_name)
    view["factor"] = view["feature_key"].map(clean_feature_name)
    view["importance"] = pd.to_numeric(view["importance"], errors="coerce")
    view = view.dropna(subset=["importance"])
    total = float(view["importance"].sum())
    view["importance_norm"] = view["importance"] / total if total > 0 else 0
    return view.sort_values("importance", ascending=False)


def build_xgb_feature_importance(model: Any) -> pd.DataFrame:
    try:
        preprocessor = model.named_steps["preprocessor"]
        regressor = model.named_steps["model"]
        feature_names = [canonical_feature_name(feature) for feature in preprocessor.get_feature_names_out()]
        booster = regressor.get_booster()
    except Exception:
        return pd.DataFrame()

    gain_scores = booster.get_score(importance_type="gain")
    weight_scores = booster.get_score(importance_type="weight")
    rows: list[dict[str, Any]] = []

    if gain_scores:
        for feature_id, gain in gain_scores.items():
            if not feature_id.startswith("f"):
                continue
            index = int(feature_id[1:])
            if index >= len(feature_names):
                continue
            feature_key = feature_names[index]
            rows.append(
                {
                    "feature": feature_key,
                    "feature_key": feature_key,
                    "factor": clean_feature_name(feature_key),
                    "gain": float(gain),
                    "split_count": int(weight_scores.get(feature_id, 0)),
                }
            )
    elif hasattr(regressor, "feature_importances_"):
        importances = np.asarray(regressor.feature_importances_, dtype=float)
        for index, importance in enumerate(importances[: len(feature_names)]):
            if importance <= 0:
                continue
            feature_key = feature_names[index]
            rows.append(
                {
                    "feature": feature_key,
                    "feature_key": feature_key,
                    "factor": clean_feature_name(feature_key),
                    "gain": float(importance),
                    "split_count": np.nan,
                }
            )

    view = pd.DataFrame(rows)
    if view.empty:
        return view
    total = float(view["gain"].sum())
    view["gain_norm"] = view["gain"] / total if total > 0 else 0
    return view.sort_values("gain", ascending=False)


def build_feature_importance_comparison(shap_importance: pd.DataFrame, xgb_importance: pd.DataFrame) -> pd.DataFrame:
    if shap_importance.empty or xgb_importance.empty:
        return pd.DataFrame()
    comparison = shap_importance[["feature_key", "factor", "importance_norm"]].merge(
        xgb_importance[["feature_key", "gain_norm"]],
        on="feature_key",
        how="outer",
    )
    comparison["factor"] = comparison["factor"].fillna(comparison["feature_key"].map(clean_feature_name))
    comparison[["importance_norm", "gain_norm"]] = comparison[["importance_norm", "gain_norm"]].fillna(0)
    comparison["max_score"] = comparison[["importance_norm", "gain_norm"]].max(axis=1)
    return comparison.sort_values("max_score", ascending=False)


def explain_prediction(model: Any, input_df: pd.DataFrame, pred_price: float, top_n: int = 8) -> pd.DataFrame:
    try:
        import xgboost as xgb

        preprocessor = model.named_steps["preprocessor"]
        regressor = model.named_steps["model"]
        transformed = preprocessor.transform(input_df)
        contributions = regressor.get_booster().predict(xgb.DMatrix(transformed), pred_contribs=True)[0]
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        return pd.DataFrame()

    rows = pd.DataFrame(
        {
            "feature": feature_names,
            "contribution_log": contributions[:-1],
        }
    )
    rows["abs_contribution_log"] = rows["contribution_log"].abs()
    rows = rows.sort_values("abs_contribution_log", ascending=False).head(top_n).copy()
    rows["factor"] = rows["feature"].map(clean_feature_name)
    rows["direction"] = np.where(rows["contribution_log"] >= 0, "повышает", "снижает")
    rows["effect_price_approx"] = pred_price * (np.exp(rows["contribution_log"]) - 1)
    return rows[["factor", "direction", "contribution_log", "effect_price_approx"]]


def validate_prediction_inputs(raw: dict[str, Any]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    area_total = float(raw["area_total"])
    area_living = float(raw["area_living"])
    area_kitchen = float(raw["area_kitchen"])
    current_floor = int(raw["current_floor"])
    total_floors = int(raw["total_floors"])
    build_year = int(raw["build_year"])
    rooms = str(raw["rooms"])

    if area_living > area_total:
        errors.append("Жилая площадь не может быть больше общей площади.")
    if area_kitchen > area_total:
        errors.append("Площадь кухни не может быть больше общей площади.")
    if area_living + area_kitchen > area_total * 1.15:
        warnings.append("Сумма жилой площади и кухни выглядит завышенной относительно общей площади.")
    if current_floor > total_floors:
        errors.append("Этаж квартиры не может быть выше этажности дома.")
    if rooms == "studio" and not raw["is_studio"]:
        warnings.append("Выбраны комнаты `studio`, но переключатель студии выключен.")
    if rooms != "studio" and raw["is_studio"]:
        warnings.append("Переключатель студии включен, но в поле комнат выбрана не студия.")
    if raw["is_new_building"] and build_year < CURRENT_YEAR - 10:
        warnings.append("Для новостройки год постройки выглядит слишком старым.")
    if raw["status_home"] == "Не сдан" and not raw["is_new_building"]:
        warnings.append("Дом не сдан, но объект не отмечен как новостройка.")
    if raw["region"] == "Санкт-Петербург" and raw["district"] in {"Всеволожский", "Гатчинский", "Ломоносовский", "Тосненский"}:
        warnings.append("Район похож на Ленинградскую область, а регион выбран как Санкт-Петербург.")

    return errors, warnings


def make_reference(df: pd.DataFrame, categories: dict[str, list[str]]) -> dict[str, Any]:
    reference: dict[str, Any] = {}
    for col in ["district", "nearest_metro", "metro_travel_type", "status_home", "rooms", "region", "type_building"]:
        if col in categories:
            reference[col] = categories[col]
        elif col in df.columns:
            reference[col] = sorted_options(df[col])
        else:
            reference[col] = []

    for col in ["lat_bin", "lng_bin", "metro_bin"]:
        if col in categories:
            reference[col] = categories[col]
        elif col in df.columns:
            reference[col] = sorted_options(df[col])
        else:
            reference[col] = []

    metro_cols = ["nearest_metro", "metro_lat", "metro_lng"]
    if set(metro_cols).issubset(df.columns):
        metro_ref = (
            df[metro_cols]
            .dropna()
            .groupby("nearest_metro", as_index=True)
            .median(numeric_only=True)
        )
    else:
        metro_ref = pd.DataFrame(columns=["metro_lat", "metro_lng"])
    reference["metro_ref"] = metro_ref

    return reference


def prepare_input(
    raw: dict[str, Any],
    df: pd.DataFrame,
    all_features: list[str],
    cat_features: list[str],
    reference: dict[str, Any],
) -> pd.DataFrame:
    build_year = int(raw["build_year"])
    total_floors = max(int(raw["total_floors"]), 1)
    current_floor = min(max(int(raw["current_floor"]), 1), total_floors)
    area_total = max(float(raw["area_total"]), 1.0)
    area_living = max(float(raw["area_living"]), 0.0)
    area_kitchen = max(float(raw["area_kitchen"]), 0.0)
    rooms = str(raw["rooms"])
    lat = float(raw["lat"])
    lng = float(raw["lng"])
    metro_time = int(raw["metro_travel_time"])
    nearest_metro = str(raw["nearest_metro"])

    metro_ref = reference["metro_ref"]
    if nearest_metro in metro_ref.index:
        metro_lat = float(metro_ref.loc[nearest_metro, "metro_lat"])
        metro_lng = float(metro_ref.loc[nearest_metro, "metro_lng"])
    else:
        metro_lat = median_number(df, "metro_lat", CENTER_LAT)
        metro_lng = median_number(df, "metro_lng", CENTER_LNG)

    is_completed = str(raw["status_home"]) == "Сдан"
    is_new_building = bool(raw["is_new_building"])
    is_studio = rooms == "studio" or bool(raw["is_studio"])

    values = {
        "district": str(raw["district"]),
        "lat": lat,
        "lng": lng,
        "nearest_metro": nearest_metro,
        "metro_travel_time": metro_time,
        "metro_travel_type": str(raw["metro_travel_type"]),
        "status_home": str(raw["status_home"]),
        "is_new_building": is_new_building,
        "build_year": build_year,
        "area_total": area_total,
        "area_living": area_living,
        "area_kitchen": area_kitchen,
        "rooms": rooms,
        "region": str(raw["region"]),
        "current_floor": current_floor,
        "total_floors": total_floors,
        "type_building": str(raw["type_building"]),
        "is_studio": is_studio,
        "is_completed": is_completed,
        "is_spb": str(raw["region"]) == "Санкт-Петербург",
        "is_first_floor": current_floor == 1,
        "is_last_floor": current_floor == total_floors,
        "prop_floor": current_floor / total_floors,
        "age": CURRENT_YEAR - build_year,
        "metro_lat": metro_lat,
        "metro_lng": metro_lng,
        "metro_travel_km": haversine(lat, lng, metro_lat, metro_lng),
        "center_travel_km": haversine(lat, lng, CENTER_LAT, CENTER_LNG),
        "build_decade": (build_year // 10) * 10,
        "log_area": math.log(area_total),
        "lat_bin": interval_label(lat, reference.get("lat_bin", [])),
        "lng_bin": interval_label(lng, reference.get("lng_bin", [])),
        "metro_bin": interval_label(float(metro_time), reference.get("metro_bin", [])),
    }

    row = {feature: values.get(feature, np.nan) for feature in all_features}
    input_df = pd.DataFrame([row], columns=all_features)

    for feature in cat_features:
        input_df[feature] = input_df[feature].astype(str)
    for feature in input_df.columns.difference(cat_features):
        if feature.startswith("is_"):
            input_df[feature] = input_df[feature].astype(bool)
        else:
            input_df[feature] = pd.to_numeric(input_df[feature], errors="coerce").fillna(0)

    return input_df


def coerce_model_frame(frame: pd.DataFrame, all_features: list[str], cat_features: list[str]) -> pd.DataFrame:
    model_frame = frame.reindex(columns=all_features).copy()
    for feature in cat_features:
        model_frame[feature] = model_frame[feature].fillna("missing").astype(str)
    for feature in model_frame.columns.difference(cat_features):
        if feature.startswith("is_"):
            model_frame[feature] = model_frame[feature].fillna(False).astype(bool)
        else:
            numeric = pd.to_numeric(model_frame[feature], errors="coerce")
            model_frame[feature] = numeric.fillna(numeric.median() if not pd.isna(numeric.median()) else 0)
    return model_frame


@st.cache_data(show_spinner=False)
def build_quality_frame(_model: Any, df: pd.DataFrame, all_features: list[str], cat_features: list[str]) -> pd.DataFrame:
    split_data = load_split_data()
    if {"X_test", "y_test"}.issubset(split_data):
        x_test = split_data["X_test"].copy().reset_index(drop=True)
        y_test = split_data["y_test"].copy()
        available = [feature for feature in all_features if feature in x_test.columns]
        if len(available) == len(all_features) and len(x_test) == len(y_test):
            quality = x_test.copy()
            model_frame = coerce_model_frame(quality, all_features, cat_features)
            pred_log = _model.predict(model_frame)
            y_values = pd.to_numeric(pd.Series(y_test).reset_index(drop=True), errors="coerce")
            target_name = str(split_data.get("target", "log_price"))
            if target_name == "log_price" or y_values.median() < 100:
                true_log = y_values
                quality[TARGET] = np.exp(true_log)
            else:
                quality[TARGET] = y_values
                true_log = np.log(quality[TARGET].clip(lower=1))

            quality["price_pred"] = np.exp(pred_log)
            quality["residual_log"] = true_log.to_numpy() - pred_log
            quality["abs_error_log"] = quality["residual_log"].abs()
            quality["abs_error_rub"] = (quality[TARGET] - quality["price_pred"]).abs()
            quality["ape_price"] = quality["abs_error_rub"] / quality[TARGET].clip(lower=1)
            quality["residual_source"] = "holdout_test"
            return quality

    available = [feature for feature in all_features if feature in df.columns]
    if TARGET not in df.columns or len(available) != len(all_features):
        return pd.DataFrame()

    quality = df.dropna(subset=[TARGET]).copy()
    if len(quality) > QUALITY_SAMPLE_SIZE:
        quality = quality.sample(QUALITY_SAMPLE_SIZE, random_state=42).copy()

    model_frame = coerce_model_frame(quality, all_features, cat_features)
    pred_log = _model.predict(model_frame)
    true_log = np.log(quality[TARGET].clip(lower=1))

    quality["price_pred"] = np.exp(pred_log)
    quality["residual_log"] = true_log - pred_log
    quality["abs_error_log"] = quality["residual_log"].abs()
    quality["abs_error_rub"] = (quality[TARGET] - quality["price_pred"]).abs()
    quality["ape_price"] = quality["abs_error_rub"] / quality[TARGET].clip(lower=1)
    quality["residual_source"] = "available_data_resubstitution"
    return quality


def build_segment_interval(
    quality: pd.DataFrame,
    raw: dict[str, Any],
    pred_price: float,
    fallback_mape: float,
) -> dict[str, Any]:
    fallback_low = max(pred_price * (1 - fallback_mape), 0)
    fallback_high = pred_price * (1 + fallback_mape)
    residual_source = (
        str(quality["residual_source"].iloc[0])
        if not quality.empty and "residual_source" in quality.columns
        else "fallback"
    )

    if quality.empty or "residual_log" not in quality.columns:
        return {
            "label": "Общий MAPE модели",
            "fields": [],
            "n": 0,
            "low": fallback_low,
            "high": fallback_high,
            "mean_ape": fallback_mape,
            "median_ape": np.nan,
            "q80_ape": fallback_mape,
            "source": "fallback",
            "residual_source": residual_source,
        }

    segment_levels = [
        ("район + комнаты + рынок + первый этаж", ["region", "district", "rooms", "is_new_building", "is_first_floor"], 25),
        ("регион + комнаты + рынок + первый этаж", ["region", "rooms", "is_new_building", "is_first_floor"], 40),
        ("регион + комнаты + рынок", ["region", "rooms", "is_new_building"], 50),
        ("регион + рынок", ["region", "is_new_building"], 80),
    ]

    for label, fields, min_count in segment_levels:
        segment = filter_by_raw_segment(quality, raw, fields)
        residuals = pd.to_numeric(segment.get("residual_log", pd.Series(dtype=float)), errors="coerce").dropna()
        if len(residuals) < min_count:
            continue

        q10, q90 = residuals.quantile([0.10, 0.90])
        ape = pd.to_numeric(segment.get("ape_price", pd.Series(dtype=float)), errors="coerce").dropna()
        return {
            "label": label,
            "fields": fields,
            "n": int(len(residuals)),
            "low": max(float(pred_price * np.exp(q10)), 0),
            "high": float(pred_price * np.exp(q90)),
            "mean_ape": float(ape.mean()) if not ape.empty else fallback_mape,
            "median_ape": float(ape.median()) if not ape.empty else np.nan,
            "q80_ape": float(ape.quantile(0.80)) if not ape.empty else fallback_mape,
            "source": "segment",
            "residual_source": residual_source,
        }

    residuals = pd.to_numeric(quality["residual_log"], errors="coerce").dropna()
    if not residuals.empty:
        q10, q90 = residuals.quantile([0.10, 0.90])
        ape = pd.to_numeric(quality.get("ape_price", pd.Series(dtype=float)), errors="coerce").dropna()
        return {
            "label": "Все доступные диагностические наблюдения",
            "fields": [],
            "n": int(len(residuals)),
            "low": max(float(pred_price * np.exp(q10)), 0),
            "high": float(pred_price * np.exp(q90)),
            "mean_ape": float(ape.mean()) if not ape.empty else fallback_mape,
            "median_ape": float(ape.median()) if not ape.empty else np.nan,
            "q80_ape": float(ape.quantile(0.80)) if not ape.empty else fallback_mape,
            "source": "diagnostic",
            "residual_source": residual_source,
        }

    return {
        "label": "Общий MAPE модели",
        "fields": [],
        "n": 0,
        "low": fallback_low,
        "high": fallback_high,
        "mean_ape": fallback_mape,
        "median_ape": np.nan,
        "q80_ape": fallback_mape,
        "source": "fallback",
        "residual_source": residual_source,
    }


def find_similar_objects(df: pd.DataFrame, raw: dict[str, Any], pred_price: float, n: int = 10) -> pd.DataFrame:
    required = {"price", "area_total", "build_year", "lat", "lng"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    candidates = df.dropna(subset=list(required)).copy()
    filters = [
        ("region", raw["region"]),
        ("district", raw["district"]),
        ("rooms", str(raw["rooms"])),
    ]
    if "is_new_building" in candidates.columns:
        filters.append(("is_new_building", bool(raw["is_new_building"])))

    for col, value in filters:
        if col not in candidates.columns:
            continue
        filtered = candidates[candidates[col].astype(str).eq(str(value))]
        if len(filtered) >= n:
            candidates = filtered

    area_scale = max(float(df["area_total"].quantile(0.75) - df["area_total"].quantile(0.25)), 1.0)
    build_scale = 30.0
    metro_scale = 10.0
    if "total_floors" in df.columns:
        floors = pd.to_numeric(df["total_floors"], errors="coerce").dropna()
        floor_scale = max(float(floors.quantile(0.75) - floors.quantile(0.25)), 1.0) if not floors.empty else 10.0
    else:
        floor_scale = 10.0

    score = (
        (candidates["area_total"].astype(float) - float(raw["area_total"])).abs() / area_scale
        + (candidates["build_year"].astype(float) - float(raw["build_year"])).abs() / build_scale
    )
    if "metro_travel_time" in candidates.columns:
        score += (pd.to_numeric(candidates["metro_travel_time"], errors="coerce").fillna(0) - int(raw["metro_travel_time"])).abs() / metro_scale
    if "current_floor" in candidates.columns:
        score += (pd.to_numeric(candidates["current_floor"], errors="coerce").fillna(0) - int(raw["current_floor"])).abs() / floor_scale
    if "total_floors" in candidates.columns:
        score += (pd.to_numeric(candidates["total_floors"], errors="coerce").fillna(0) - int(raw["total_floors"])).abs() / floor_scale
    score += candidates.apply(lambda row: haversine(float(row["lat"]), float(row["lng"]), float(raw["lat"]), float(raw["lng"])) / 8, axis=1)

    result = candidates.assign(similarity_score=score).sort_values("similarity_score").head(n).copy()
    result["price_diff_to_prediction"] = result["price"] - pred_price
    result["ape_to_prediction"] = (result["price_diff_to_prediction"].abs() / max(pred_price, 1)) * 100
    visible_cols = [
        col
        for col in [
            "price",
            "price_diff_to_prediction",
            "ape_to_prediction",
            "area_total",
            "rooms",
            "region",
            "district",
            "lat",
            "lng",
            "nearest_metro",
            "metro_travel_time",
            "is_new_building",
            "build_year",
            "current_floor",
            "total_floors",
            "similarity_score",
        ]
        if col in result.columns
    ]
    return result[visible_cols]


def format_similar_objects(similar: pd.DataFrame) -> pd.DataFrame:
    if similar.empty:
        return similar
    view = similar.copy()
    for col in ["price", "price_diff_to_prediction"]:
        if col in view.columns:
            view[col] = view[col].map(format_rub)
    if "ape_to_prediction" in view.columns:
        view["ape_to_prediction"] = view["ape_to_prediction"].map(lambda x: f"{x:.1f}%")
    if "area_total" in view.columns:
        view["area_total"] = view["area_total"].map(lambda x: f"{x:.1f} м²")
    if "similarity_score" in view.columns:
        view["similarity_score"] = view["similarity_score"].map(lambda x: f"{x:.3f}")
    for col in ["rooms", "is_new_building", "type_building", "metro_travel_type"]:
        if col in view.columns:
            view[col] = view[col].map(lambda value, column=col: display_value(column, value))
    return view.rename(
        columns={
            "price": "Цена",
            "price_diff_to_prediction": "Разница с прогнозом",
            "ape_to_prediction": "Отклонение",
            "area_total": "Площадь",
            "rooms": "Комнаты",
            "region": "Регион",
            "district": "Район",
            "lat": "Широта",
            "lng": "Долгота",
            "nearest_metro": "Метро",
            "metro_travel_time": "Мин. до метро",
            "is_new_building": "Новостройка",
            "build_year": "Год",
            "current_floor": "Этаж",
            "total_floors": "Этажей",
            "similarity_score": "Сходство",
        }
    )


def build_market_comparison(df: pd.DataFrame, raw: dict[str, Any], pred_price: float, similar: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pred_sqm = pred_price / max(float(raw["area_total"]), 1)

    pools: list[tuple[str, pd.DataFrame]] = []
    district_pool = filter_by_raw_segment(df, raw, ["region", "district"])
    if len(district_pool) >= 5:
        pools.append(("Выбранный район", district_pool))

    segment_pool = filter_by_raw_segment(df, raw, ["region", "rooms", "is_new_building"])
    if len(segment_pool) >= 5:
        pools.append(("Регион + комнаты + рынок", segment_pool))

    floor_pool = filter_by_raw_segment(df, raw, ["region", "rooms", "is_new_building", "is_first_floor"])
    if len(floor_pool) >= 5:
        pools.append(("Регион + комнаты + рынок + этаж", floor_pool))

    if not similar.empty:
        pools.append(("Похожие объекты", similar))

    if len(df) >= 5:
        pools.append(("Весь датасет", df))

    used_labels: set[str] = set()
    for label, pool in pools:
        if label in used_labels or "price" not in pool.columns:
            continue
        used_labels.add(label)
        prices = pd.to_numeric(pool["price"], errors="coerce").dropna()
        price_sqm = safe_price_per_sqm(pool).dropna()
        if prices.empty:
            continue
        median_price = float(prices.median())
        median_sqm = float(price_sqm.median()) if not price_sqm.empty else np.nan
        rows.append(
            {
                "Срез рынка": label,
                "Объектов": int(len(prices)),
                "Медиана цены": median_price,
                "Прогноз к медиане": (pred_price / median_price - 1) if median_price else np.nan,
                "Перцентиль цены": percentile_rank(prices, pred_price),
                "Медиана за м²": median_sqm,
                "Цена за м² к медиане": (pred_sqm / median_sqm - 1) if median_sqm and not np.isnan(median_sqm) else np.nan,
                "Перцентиль за м²": percentile_rank(price_sqm, pred_sqm),
            }
        )

    return pd.DataFrame(rows)


def format_market_comparison(market: pd.DataFrame) -> pd.DataFrame:
    if market.empty:
        return market
    view = market.copy()
    for col in ["Медиана цены", "Медиана за м²"]:
        if col in view.columns:
            view[col] = view[col].map(lambda x: format_rub(float(x)) if not pd.isna(x) else "н/д")
    for col in ["Прогноз к медиане", "Цена за м² к медиане"]:
        if col in view.columns:
            view[col] = view[col].map(lambda x: f"{x:+.1%}" if not pd.isna(x) else "н/д")
    for col in ["Перцентиль цены", "Перцентиль за м²"]:
        if col in view.columns:
            view[col] = view[col].map(lambda x: f"{x:.0f}" if x is not None and not pd.isna(x) else "н/д")
    return view


def show_similar_objects_map(similar: pd.DataFrame, raw: dict[str, Any]) -> None:
    if similar.empty or not {"lat", "lng"}.issubset(similar.columns):
        return
    if not HAS_MAP:
        return

    map_obj = folium.Map(location=[float(raw["lat"]), float(raw["lng"])], zoom_start=12, tiles="OpenStreetMap")
    folium.Marker(
        [float(raw["lat"]), float(raw["lng"])],
        tooltip="Прогнозируемый объект",
        icon=folium.Icon(color="red", icon="home"),
    ).add_to(map_obj)

    for _, row in similar.dropna(subset=["lat", "lng"]).head(20).iterrows():
        popup = (
            f"{format_rub(float(row['price']))}<br>"
            f"{row.get('district', '')}, {row.get('area_total', '')} м²"
        )
        folium.CircleMarker(
            location=[float(row["lat"]), float(row["lng"])],
            radius=5,
            popup=popup,
            color="#2563eb",
            fill=True,
            fill_opacity=0.75,
        ).add_to(map_obj)

    st_folium(map_obj, height=360, use_container_width=True, returned_objects=[])


def assess_prediction_context(df: pd.DataFrame, raw: dict[str, Any], reference: dict[str, Any]) -> tuple[str, int, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    penalty = 0

    numeric_checks = {
        "area_total": "Общая площадь",
        "build_year": "Год постройки",
        "metro_travel_time": "Время до метро",
        "lat": "Широта",
        "lng": "Долгота",
    }
    for col, label in numeric_checks.items():
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue
        value = float(raw[col])
        q01, q05, q95, q99 = series.quantile([0.01, 0.05, 0.95, 0.99])
        if value < q01 or value > q99:
            status = "вне типичного диапазона"
            penalty += 2
        elif value < q05 or value > q95:
            status = "редкое значение"
            penalty += 1
        else:
            status = "типично"
        rows.append(
            {
                "Проверка": label,
                "Значение": value,
                "Диапазон 5-95%": f"{q05:.2f} - {q95:.2f}",
                "Статус": status,
            }
        )

    category_checks = {
        "district": "Район",
        "nearest_metro": "Метро",
        "region": "Регион",
        "type_building": "Тип здания",
        "rooms": "Комнаты",
    }
    for col, label in category_checks.items():
        options = set(reference.get(col, []))
        value = str(raw[col])
        if not options:
            continue
        status = "есть в обучении" if value in options else "новая категория"
        if status == "новая категория":
            penalty += 2
        rows.append({"Проверка": label, "Значение": value, "Диапазон 5-95%": "категория", "Статус": status})

    if penalty <= 2:
        label = "Типичный объект"
        score = 90 - penalty * 8
    elif penalty <= 5:
        label = "Редкий, но похож на данные"
        score = 68 - (penalty - 3) * 7
    else:
        label = "Вне комфортной зоны модели"
        score = max(25, 55 - penalty * 5)

    return label, int(score), pd.DataFrame(rows)


def build_scenario_comparison(
    raw: dict[str, Any],
    df: pd.DataFrame,
    model: Any,
    all_features: list[str],
    cat_features: list[str],
    reference: dict[str, Any],
) -> pd.DataFrame:
    scenarios: list[tuple[str, dict[str, Any]]] = []
    scenarios.append(("Текущий объект", raw.copy()))

    larger = raw.copy()
    larger["area_total"] = float(larger["area_total"]) + 10
    larger["area_living"] = min(float(larger["area_living"]) + 7, larger["area_total"])
    scenarios.append(("+10 м² общей площади", larger))

    faster_metro = raw.copy()
    faster_metro["metro_travel_time"] = max(1, int(faster_metro["metro_travel_time"]) - 5)
    scenarios.append(("-5 минут до метро", faster_metro))

    higher_floor = raw.copy()
    higher_floor["current_floor"] = min(int(higher_floor["total_floors"]), int(higher_floor["current_floor"]) + 3)
    scenarios.append(("Этаж выше", higher_floor))

    newer_building = raw.copy()
    newer_building["build_year"] = min(CURRENT_YEAR, int(newer_building["build_year"]) + 5)
    scenarios.append(("Дом на 5 лет новее", newer_building))

    larger_kitchen = raw.copy()
    larger_kitchen["area_kitchen"] = min(float(larger_kitchen["area_kitchen"]) + 3, float(larger_kitchen["area_total"]))
    scenarios.append(("+3 м² кухни", larger_kitchen))

    rows: list[dict[str, Any]] = []
    base_price = None
    for name, scenario_raw in scenarios:
        errors, _ = validate_prediction_inputs(scenario_raw)
        if errors:
            continue
        scenario_input = prepare_input(scenario_raw, df, all_features, cat_features, reference)
        price = float(np.exp(model.predict(scenario_input)[0]))
        if base_price is None:
            base_price = price
        rows.append(
            {
                "Сценарий": name,
                "Прогноз": price,
                "Изменение": price - base_price,
                "Изменение, %": (price / base_price - 1) * 100 if base_price else 0,
            }
        )
    return pd.DataFrame(rows)


def append_prediction_history(preset_name: str, raw: dict[str, Any], pred_price: float) -> None:
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []
    st.session_state.prediction_history.append(
        {
            "time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "preset": preset_name,
            "prediction_price": pred_price,
            "price_sqm": pred_price / max(float(raw["area_total"]), 1),
            "region": raw["region"],
            "district": raw["district"],
            "rooms": raw["rooms"],
            "area_total": raw["area_total"],
            "is_new_building": raw["is_new_building"],
        }
    )


def show_prediction_history() -> None:
    history = st.session_state.get("prediction_history", [])
    if not history:
        return

    history_df = pd.DataFrame(history)
    view = history_df.copy()
    view["prediction_price"] = view["prediction_price"].map(format_rub)
    view["price_sqm"] = view["price_sqm"].map(format_rub)
    view["area_total"] = view["area_total"].map(lambda x: f"{float(x):.1f} м²")
    st.subheader("История прогнозов в этой сессии")
    st.dataframe(
        view.rename(
            columns={
                "time": "Время",
                "preset": "Сценарий",
                "prediction_price": "Прогноз",
                "price_sqm": "Цена за м²",
                "region": "Регион",
                "district": "Район",
                "rooms": "Комнаты",
                "area_total": "Площадь",
                "is_new_building": "Новостройка",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Скачать историю CSV",
            data=history_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="prediction_history.csv",
            mime="text/csv",
            width="stretch",
        )
    with col2:
        if st.button("Очистить историю", width="stretch"):
            st.session_state.prediction_history = []
            st.rerun()


def build_prediction_report(
    raw: dict[str, Any],
    pred_price: float,
    low_price: float,
    high_price: float,
    interval_label: str,
    segment_median: float | None,
    market_comparison: pd.DataFrame,
    scenario_comparison: pd.DataFrame,
    explanation: pd.DataFrame,
    similar: pd.DataFrame,
) -> str:
    lines = [
        "# Отчет по прогнозу цены",
        "",
        f"- Прогноз цены: **{format_rub(pred_price)}**",
        f"- Интервал: **{format_rub(low_price)} - {format_rub(high_price)}**",
        f"- Основание интервала: **{interval_label}**",
    ]
    if segment_median is not None:
        lines.append(f"- Медиана похожего сегмента: **{format_rub(segment_median)}**")

    lines.extend(["", "## Входные параметры"])
    for key, value in raw.items():
        lines.append(f"- `{key}`: {value}")

    if not market_comparison.empty:
        lines.extend(["", "## Сравнение с рынком"])
        for _, row in market_comparison.iterrows():
            lines.append(
                f"- {row['Срез рынка']}: медиана {format_rub(float(row['Медиана цены']))}, "
                f"прогноз к медиане {row['Прогноз к медиане']:+.1%}"
            )

    if not scenario_comparison.empty:
        lines.extend(["", "## Модельная чувствительность"])
        for _, row in scenario_comparison.iterrows():
            lines.append(
                f"- {row['Сценарий']}: {format_rub(float(row['Прогноз']))} "
                f"({row['Изменение, %']:+.1f}%)"
            )

    if not explanation.empty:
        lines.extend(["", "## Основные факторы"])
        for _, row in explanation.iterrows():
            lines.append(
                f"- {row['factor']}: {row['direction']}, "
                f"вклад в log(price) {row['contribution_log']:+.4f}"
            )

    if not similar.empty:
        lines.extend(["", "## Похожие объекты"])
        similar_view = similar.head(10)
        for _, row in similar_view.iterrows():
            price = format_rub(float(row["price"])) if "price" in row else "н/д"
            district = row.get("district", "н/д")
            area = row.get("area_total", "н/д")
            rooms = row.get("rooms", "н/д")
            lines.append(f"- {district}, {rooms} комн., {area} м²: {price}")

    lines.extend(
        [
            "",
            "## Важно",
            "Прогноз является ML-ориентиром по данным объявлений и не заменяет оценочный отчет.",
        ]
    )
    return "\n".join(lines)


def build_prediction_report_html(
    raw: dict[str, Any],
    pred_price: float,
    low_price: float,
    high_price: float,
    interval: dict[str, Any],
    context_label: str,
    context_score: int,
    market_comparison: pd.DataFrame,
    scenario_comparison: pd.DataFrame,
    explanation: pd.DataFrame,
    similar: pd.DataFrame,
) -> str:
    params = pd.DataFrame(
        [{"Параметр": key, "Значение": value} for key, value in raw.items()]
    )
    market_view = format_market_comparison(market_comparison)

    scenario_view = scenario_comparison.copy()
    if not scenario_view.empty:
        scenario_view["Прогноз"] = scenario_view["Прогноз"].map(format_rub)
        scenario_view["Изменение"] = scenario_view["Изменение"].map(format_rub)
        scenario_view["Изменение, %"] = scenario_view["Изменение, %"].map(lambda x: f"{x:+.1f}%")

    explanation_view = explanation.copy()
    if not explanation_view.empty:
        explanation_view["contribution_log"] = explanation_view["contribution_log"].map(lambda x: f"{x:+.4f}")
        explanation_view["effect_price_approx"] = explanation_view["effect_price_approx"].map(format_rub)
        explanation_view = explanation_view.rename(
            columns={
                "factor": "Фактор",
                "direction": "Направление",
                "contribution_log": "Вклад в log(price)",
                "effect_price_approx": "Примерный вклад в ₽",
            }
        )

    similar_view = format_similar_objects(similar.head(15)) if not similar.empty else pd.DataFrame()
    generated_at = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    interval_source = html.escape(str(interval.get("label", "н/д")))
    interval_n = int(interval.get("n", 0))

    return f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <title>Отчет по прогнозу Cian</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 36px; color: #111827; }}
    h1, h2 {{ margin-bottom: 8px; }}
    .muted {{ color: #6b7280; }}
    .cards {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 18px 0 24px; }}
    .card {{ border: 1px solid #d8dee8; border-radius: 8px; padding: 14px 16px; background: #f8fafc; }}
    .label {{ color: #64748b; font-size: 13px; }}
    .value {{ font-size: 22px; font-weight: 700; margin-top: 4px; }}
    .report-table {{ width: 100%; border-collapse: collapse; margin: 10px 0 22px; font-size: 14px; }}
    .report-table th, .report-table td {{ border-bottom: 1px solid #e5e7eb; text-align: left; padding: 8px; }}
    .report-table th {{ background: #f3f4f6; }}
  </style>
</head>
<body>
  <h1>Отчет по прогнозу цены квартиры</h1>
  <p class="muted">Сформировано: {html.escape(generated_at)}. Прогноз является ML-ориентиром по данным объявлений.</p>
  <div class="cards">
    <div class="card"><div class="label">Прогноз цены</div><div class="value">{html.escape(format_rub(pred_price))}</div></div>
    <div class="card"><div class="label">Эмпирический интервал</div><div class="value">{html.escape(format_rub(low_price))} - {html.escape(format_rub(high_price))}</div></div>
    <div class="card"><div class="label">Близость к обучению</div><div class="value">{context_score}/100</div><div class="muted">{html.escape(context_label)}</div></div>
  </div>
  <p><b>Основание интервала:</b> {interval_source}; наблюдений в срезе: {interval_n}.</p>
  <h2>Входные параметры</h2>
  {html_table(params)}
  <h2>Сравнение с рынком</h2>
  {html_table(market_view)}
  <h2>Модельная чувствительность</h2>
  {html_table(scenario_view)}
  <h2>Факторы прогноза</h2>
  {html_table(explanation_view)}
  <h2>Похожие объекты</h2>
  {html_table(similar_view)}
  <h2>Ограничения</h2>
  <p>Модель обучена на исторических объявлениях и хуже переносится на редкие объекты, новые категории и объекты вне типичных диапазонов признаков.</p>
</body>
</html>"""


def build_single_prediction_payload(
    raw: dict[str, Any],
    df: pd.DataFrame,
    model: Any,
    input_df: pd.DataFrame,
    all_features: list[str],
    cat_features: list[str],
    reference: dict[str, Any],
    metadata: dict[str, Any],
    context_label: str,
    context_score: int,
) -> dict[str, Any]:
    pred_log = float(model.predict(input_df)[0])
    pred_price = float(np.exp(pred_log))
    mape = metadata.get("metrics", {}).get("mape_price", 0.1121)
    quality = build_quality_frame(model, df, all_features, cat_features)
    segment_interval = build_segment_interval(quality, raw, pred_price, mape)
    low_price = float(segment_interval["low"])
    high_price = float(segment_interval["high"])

    scenario_comparison = build_scenario_comparison(raw, df, model, all_features, cat_features, reference)

    segment = df.copy()
    if "district" in segment.columns:
        segment = segment[segment["district"].astype(str).eq(str(raw["district"]))]
    if "rooms" in segment.columns:
        segment = segment[segment["rooms"].astype(str).eq(str(raw["rooms"]))]
    if "is_new_building" in segment.columns:
        segment = segment[segment["is_new_building"].eq(bool(raw["is_new_building"]))]

    segment_median = float(segment["price"].median()) if len(segment) >= 10 else None
    similar = find_similar_objects(df, raw, pred_price)
    market_comparison = build_market_comparison(df, raw, pred_price, similar)
    explanation = explain_prediction(model, input_df, pred_price)

    report_text = build_prediction_report(
        raw=raw,
        pred_price=pred_price,
        low_price=low_price,
        high_price=high_price,
        interval_label=str(segment_interval["label"]),
        segment_median=segment_median,
        market_comparison=market_comparison,
        scenario_comparison=scenario_comparison,
        explanation=explanation,
        similar=similar,
    )
    report_html = build_prediction_report_html(
        raw=raw,
        pred_price=pred_price,
        low_price=low_price,
        high_price=high_price,
        interval=segment_interval,
        context_label=context_label,
        context_score=context_score,
        market_comparison=market_comparison,
        scenario_comparison=scenario_comparison,
        explanation=explanation,
        similar=similar,
    )

    return {
        "raw": raw.copy(),
        "pred_price": pred_price,
        "low_price": low_price,
        "high_price": high_price,
        "segment_interval": segment_interval,
        "segment_median": segment_median,
        "scenario_comparison": scenario_comparison,
        "similar": similar,
        "market_comparison": market_comparison,
        "explanation": explanation,
        "report_text": report_text,
        "report_html": report_html,
        "context_label": context_label,
        "context_score": context_score,
    }


def residual_source_label(source: Any) -> str:
    source_text = str(source)
    if source_text == "holdout_test":
        return "тестовая выборка модели"
    if source_text == "available_data_resubstitution":
        return "диагностика на доступном датасете приложения"
    if source_text == "fallback":
        return "общая ошибка модели"
    return source_text if source_text and source_text != "nan" else "н/д"


def preferred_market_row(market_comparison: pd.DataFrame) -> pd.Series | None:
    if market_comparison.empty:
        return None
    preferred = market_comparison[market_comparison["Срез рынка"].eq("Выбранный район")]
    if not preferred.empty:
        return preferred.iloc[0]
    return market_comparison.iloc[0]


def build_product_verdict(payload: dict[str, Any]) -> dict[str, Any]:
    pred_price = float(payload["pred_price"])
    low_price = float(payload["low_price"])
    high_price = float(payload["high_price"])
    segment_interval = payload["segment_interval"]
    market_comparison = payload["market_comparison"]
    similar = payload["similar"]
    context_score = int(payload.get("context_score", 0))

    interval_ratio = max((high_price - low_price) / max(pred_price, 1), 0)
    segment_n = int(segment_interval.get("n", 0) or 0)
    trust_score = context_score
    if interval_ratio > 0.65:
        trust_score -= 20
    elif interval_ratio > 0.45:
        trust_score -= 12
    elif interval_ratio > 0.30:
        trust_score -= 6
    if segment_n < 20:
        trust_score -= 12
    elif segment_n < 40:
        trust_score -= 6
    if similar.empty:
        trust_score -= 8
    trust_score = int(np.clip(trust_score, 0, 100))

    if trust_score >= 80:
        trust_label = "Высокое"
        trust_note = "прогноз можно использовать как рабочий ориентир"
    elif trust_score >= 60:
        trust_label = "Среднее"
        trust_note = "прогноз полезен для первичной оценки, но требует сверки"
    else:
        trust_label = "Низкое"
        trust_note = "нужна ручная проверка и дополнительные рыночные аналоги"

    row = preferred_market_row(market_comparison)
    if row is None or pd.isna(row.get("Прогноз к медиане", np.nan)):
        market_status = "Недостаточно рыночного среза"
        status_kind = "warning"
        market_sentence = "Для выбранных параметров не хватило устойчивого среза рынка."
        action = "Используйте прогноз как грубый ориентир и проверьте похожие объявления вручную."
    else:
        delta = float(row["Прогноз к медиане"])
        market_label = str(row["Срез рынка"])
        percentile = row.get("Перцентиль цены", np.nan)
        percentile_text = f"{float(percentile):.0f}-й перцентиль" if not pd.isna(percentile) else "перцентиль не рассчитан"
        if delta >= 0.12:
            market_status = "Дороже похожего рынка"
            status_kind = "warning"
            market_sentence = (
                f"Ориентир выше медианы среза “{market_label}” на {abs(delta):.1%}; "
                f"по цене объект попадает примерно в {percentile_text}."
            )
            action = "Проверьте, чем объясняется премия: локацией, площадью, домом или редкостью предложения."
        elif delta <= -0.12:
            market_status = "Дешевле похожего рынка"
            status_kind = "success"
            market_sentence = (
                f"Ориентир ниже медианы среза “{market_label}” на {abs(delta):.1%}; "
                f"по цене объект попадает примерно в {percentile_text}."
            )
            action = "Есть признак привлекательного уровня цены, но перед выводом нужно проверить состояние объекта и похожие объявления."
        else:
            market_status = "В рыночном диапазоне"
            status_kind = "info"
            market_sentence = (
                f"Ориентир близок к медиане среза “{market_label}” "
                f"({delta:+.1%}); по цене объект попадает примерно в {percentile_text}."
            )
            action = "Используйте прогноз как рабочий ориентир и сверяйте его с похожими объектами ниже."

    next_steps = [
        "Посмотреть похожие объекты и убедиться, что они действительно сопоставимы.",
        "Сравнить прогноз с медианами района и сегмента.",
        "Скачать отчет, если нужно передать оценку дальше.",
    ]
    if trust_score < 60:
        next_steps.insert(0, "Не принимать решение по точечной цене без ручной проверки.")

    return {
        "market_status": market_status,
        "status_kind": status_kind,
        "market_sentence": market_sentence,
        "action": action,
        "trust_score": trust_score,
        "trust_label": trust_label,
        "trust_note": trust_note,
        "interval_ratio": interval_ratio,
        "similar_count": int(len(similar)),
        "trust_reasons": [
            f"похожесть на данные: {context_score}/100",
            f"объектов в срезе ошибок: {segment_n}",
            f"похожих объявлений: {int(len(similar))}",
            f"ширина диапазона: {format_percent(float(interval_ratio))}",
        ],
        "next_steps": next_steps,
    }


def build_prediction_takeaways(payload: dict[str, Any]) -> list[str]:
    pred_price = float(payload["pred_price"])
    raw = payload["raw"]
    segment_interval = payload["segment_interval"]
    market_comparison = payload["market_comparison"]
    explanation = payload["explanation"]
    scenario_comparison = payload["scenario_comparison"]
    context_label = str(payload.get("context_label", "н/д"))
    context_score = int(payload.get("context_score", 0))

    takeaways = [
        f"Ориентир цены: {format_rub(pred_price)}; рабочий диапазон: {format_rub(float(payload['low_price']))} - {format_rub(float(payload['high_price']))}.",
        f"Похожесть объекта на данные модели: {context_score}/100 ({context_label}); диапазон построен по срезу “{segment_interval['label']}”.",
    ]

    if not market_comparison.empty:
        preferred = market_comparison[market_comparison["Срез рынка"].eq("Выбранный район")]
        row = preferred.iloc[0] if not preferred.empty else market_comparison.iloc[0]
        delta = row.get("Прогноз к медиане", np.nan)
        percentile = row.get("Перцентиль цены", np.nan)
        if not pd.isna(delta):
            relation = "выше" if float(delta) > 0 else "ниже"
            percentile_text = f"{float(percentile):.0f}" if not pd.isna(percentile) else "н/д"
            takeaways.append(
                f"Относительно среза “{row['Срез рынка']}” прогноз {relation} медианы на {abs(float(delta)):.1%}; "
                f"перцентиль по цене: {percentile_text}."
            )

    if not explanation.empty:
        top_factor = explanation.iloc[0]
        takeaways.append(
            f"Главный локальный фактор: {top_factor['factor']} ({top_factor['direction']} прогноз)."
        )

    if not scenario_comparison.empty and "Изменение" in scenario_comparison.columns:
        scenarios = scenario_comparison[scenario_comparison["Сценарий"].ne("Текущий объект")].copy()
        if not scenarios.empty:
            strongest = scenarios.iloc[scenarios["Изменение"].abs().argmax()]
            takeaways.append(
                f"Самая заметная модельная чувствительность в текущем наборе: {strongest['Сценарий']} "
                f"({format_rub(float(strongest['Изменение']))}, {float(strongest['Изменение, %']):+.1f}%)."
            )

    takeaways.append(
        "Важно: диапазон не является оценочным отчетом; это прикладной ориентир по ошибкам похожих объектов."
    )
    return takeaways


def status_css_class(status_kind: str) -> str:
    if status_kind == "success":
        return "status-success"
    if status_kind == "warning":
        return "status-warning"
    return "status-info"


def render_result_panel(payload: dict[str, Any], verdict: dict[str, Any]) -> None:
    raw = payload["raw"]
    pred_price = float(payload["pred_price"])
    low_price = float(payload["low_price"])
    high_price = float(payload["high_price"])
    area_total = max(float(raw["area_total"]), 1)
    price_sqm = pred_price / area_total
    trust_score = int(verdict["trust_score"])
    object_line = (
        f"{raw['district']}, {room_label(raw['rooms'])}, "
        f"{float(raw['area_total']):.1f} м², {int(raw['current_floor'])}/{int(raw['total_floors'])} эт."
    )
    st.subheader("Оценка объекта")
    st.caption(object_line)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Прогноз цены", format_rub(pred_price))
    col2.metric("Цена за м²", format_rub(price_sqm))
    col3.metric("Нижняя граница", format_rub(low_price))
    col4.metric("Верхняя граница", format_rub(high_price))

    st.metric("Доверие к прогнозу", f"{verdict['trust_label']} · {trust_score}/100")
    st.progress(trust_score / 100)
    st.caption(
        f"{verdict['market_status']}. Похожих объектов: {int(verdict['similar_count'])}; "
        f"ширина диапазона: {format_percent(float(verdict['interval_ratio']))}."
    )


def render_similar_cards(similar: pd.DataFrame, limit: int = 3) -> None:
    if similar.empty:
        return
    rows: list[dict[str, Any]] = []
    for _, row in similar.head(limit).iterrows():
        area = float(row.get("area_total", np.nan))
        area_text = f"{area:.1f} м²" if not pd.isna(area) else "н/д"
        price = format_rub(float(row["price"])) if "price" in row and not pd.isna(row["price"]) else "н/д"
        floor = row.get("current_floor", np.nan)
        floors = row.get("total_floors", np.nan)
        floor_text = f"{int(floor)}/{int(floors)}" if not pd.isna(floor) and not pd.isna(floors) else "н/д"
        delta = format_rub(float(row["price_diff_to_prediction"])) if "price_diff_to_prediction" in row and not pd.isna(row["price_diff_to_prediction"]) else "н/д"
        rows.append(
            {
                "Цена": price,
                "Район": row.get("district", "н/д"),
                "Комнат": room_label(row.get("rooms", "н/д")),
                "Площадь": area_text,
                "Этаж": floor_text,
                "Метро": row.get("nearest_metro", "н/д"),
                "К прогнозу": delta,
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def render_prediction_payload(payload: dict[str, Any]) -> None:
    raw = payload["raw"]
    pred_price = float(payload["pred_price"])
    low_price = float(payload["low_price"])
    high_price = float(payload["high_price"])
    segment_interval = payload["segment_interval"]
    segment_median = payload["segment_median"]
    scenario_comparison = payload["scenario_comparison"]
    similar = payload["similar"]
    market_comparison = payload["market_comparison"]
    explanation = payload["explanation"]

    st.caption("Показан последний рассчитанный прогноз.")
    verdict = build_product_verdict(payload)
    render_result_panel(payload, verdict)
    st.subheader("Итог для решения")
    verdict_message = f"{verdict['market_status']}. {verdict['market_sentence']} {verdict['action']}"
    if verdict["status_kind"] == "success":
        st.success(verdict_message)
    elif verdict["status_kind"] == "warning":
        st.warning(verdict_message)
    else:
        st.info(verdict_message)
    st.caption(
        f"Уровень доверия: {verdict['trust_note']}. "
        f"На него влияют: {', '.join(verdict['trust_reasons'])}."
    )

    st.markdown("**Следующие действия**")
    st.markdown("\n".join(f"- {step}" for step in verdict["next_steps"]))

    report_col1, report_col2 = st.columns(2)
    with report_col1:
        st.download_button(
            "Скачать HTML-отчёт",
            data=payload["report_html"].encode("utf-8"),
            file_name="cian_prediction_report.html",
            mime="text/html",
            width="stretch",
        )
    with report_col2:
        st.download_button(
            "Скачать markdown-отчёт",
            data=payload["report_text"].encode("utf-8"),
            file_name="cian_prediction_report.md",
            mime="text/markdown",
            width="stretch",
        )

    st.subheader("Почему такой вывод")
    takeaways = build_prediction_takeaways(payload)
    st.markdown("\n".join(f"- {line}" for line in takeaways))

    with st.expander("Как читать диапазон и доверие", expanded=False):
        interval_col1, interval_col2 = st.columns(2)
        interval_col1.metric("Наблюдений в срезе", int(segment_interval["n"]))
        interval_col2.metric("Средняя ошибка среза", format_percent(float(segment_interval["mean_ape"])))
        st.info(
            f"Диапазон посчитан по срезу: {segment_interval['label']}. "
            "Если похожий сегмент мал, приложение автоматически берет более широкий срез. "
            f"Источник проверки ошибок: {residual_source_label(segment_interval.get('residual_source', 'н/д'))}."
        )

    if not similar.empty:
        with st.expander("Похожие объекты для проверки", expanded=True):
            st.caption("Объекты отобраны по независимым признакам: локация, площадь, этажность, год, метро и сегмент рынка. Цена не участвует в расчете сходства.")
            render_similar_cards(similar)
            st.dataframe(format_similar_objects(similar), width="stretch", hide_index=True)
            show_similar_objects_map(similar, raw)
            st.download_button(
                "Скачать похожие объекты CSV",
                data=similar.to_csv(index=False).encode("utf-8-sig"),
                file_name="similar_objects.csv",
                mime="text/csv",
                width="stretch",
            )

    if segment_median is not None or not market_comparison.empty:
        with st.expander("Сравнение с рынком", expanded=False):
            if segment_median is not None:
                delta = (pred_price / segment_median - 1) * 100
                st.info(
                    f"Медиана похожего сегмента в данных: {format_rub(segment_median)} "
                    f"({delta:+.1f}% к прогнозу)."
                )

            if not market_comparison.empty:
                st.dataframe(format_market_comparison(market_comparison), width="stretch", hide_index=True)
                chart_market = market_comparison.dropna(subset=["Медиана цены"]).copy()
                if not chart_market.empty:
                    fig = px.bar(
                        chart_market.sort_values("Медиана цены"),
                        x="Медиана цены",
                        y="Срез рынка",
                        orientation="h",
                        title="Прогноз против медиан рыночных срезов",
                        labels={"Медиана цены": "Медианная цена, ₽", "Срез рынка": ""},
                    )
                    fig.add_vline(
                        x=pred_price,
                        line_dash="dash",
                        line_color="#ef4444",
                        annotation_text="прогноз",
                        annotation_position="top right",
                    )
                    fig.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10))
                    st.plotly_chart(fig, width="stretch")

    if not scenario_comparison.empty:
        with st.expander("Модельная чувствительность", expanded=False):
            st.caption("Это не причинный эффект, а пересчет прогноза при контролируемом изменении отдельных признаков. Нереалистичные смены района без координат и метро здесь не моделируются.")
            scenario_view = scenario_comparison.copy()
            scenario_view["Прогноз"] = scenario_view["Прогноз"].map(format_rub)
            scenario_view["Изменение"] = scenario_view["Изменение"].map(format_rub)
            scenario_view["Изменение, %"] = scenario_view["Изменение, %"].map(lambda x: f"{x:+.1f}%")
            st.dataframe(scenario_view, width="stretch", hide_index=True)
            fig = px.bar(
                scenario_comparison.sort_values("Изменение"),
                x="Изменение",
                y="Сценарий",
                orientation="h",
                title="Как меняется прогноз при изменении отдельных признаков",
                labels={"Изменение": "Изменение цены, ₽", "Сценарий": ""},
            )
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10))
            st.plotly_chart(fig, width="stretch")

    if not explanation.empty:
        with st.expander("Факторы прогноза", expanded=False):
            explanation_view = explanation.copy()
            explanation_view["contribution_log"] = explanation_view["contribution_log"].map(lambda x: f"{x:+.4f}")
            explanation_view["effect_price_approx"] = explanation_view["effect_price_approx"].map(format_rub)
            st.dataframe(
                explanation_view.rename(
                    columns={
                        "factor": "Фактор",
                        "direction": "Направление",
                        "contribution_log": "Вклад в log(price)",
                        "effect_price_approx": "Примерный вклад в ₽",
                    }
                ),
                width="stretch",
                hide_index=True,
            )
            fig = px.bar(
                explanation.sort_values("contribution_log"),
                x="contribution_log",
                y="factor",
                color="direction",
                orientation="h",
                title="Локальные вклады факторов",
                labels={"contribution_log": "Вклад в log(price)", "factor": ""},
            )
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10))
            st.plotly_chart(fig, width="stretch")


def row_to_raw(row: pd.Series, defaults: dict[str, Any]) -> dict[str, Any]:
    raw = defaults.copy()
    for key in raw:
        if key in row.index and not pd.isna(row[key]):
            raw[key] = row[key]

    if "is_new_building" in raw:
        raw["is_new_building"] = parse_bool(raw["is_new_building"], bool(defaults["is_new_building"]))
    if "is_studio" in raw:
        raw["is_studio"] = parse_bool(raw["is_studio"], bool(defaults["is_studio"]))

    for key in ["rooms", "metro_travel_type", "type_building"]:
        if key in raw:
            raw[key] = normalize_option_label(raw[key])

    raw["rooms"] = str(raw["rooms"])
    for key in ["metro_travel_time", "build_year", "current_floor", "total_floors"]:
        raw[key] = int(float(raw[key]))
    for key in ["area_total", "area_living", "area_kitchen", "lat", "lng"]:
        raw[key] = float(raw[key])
    return raw


def run_batch_predictions(
    uploaded_df: pd.DataFrame,
    df: pd.DataFrame,
    model: Any,
    all_features: list[str],
    cat_features: list[str],
    reference: dict[str, Any],
) -> tuple[pd.DataFrame, list[str]]:
    defaults = prediction_defaults(df)
    rows: list[pd.DataFrame] = []
    messages: list[str] = []

    limited_df = uploaded_df.head(BATCH_MAX_ROWS).copy()
    if len(uploaded_df) > BATCH_MAX_ROWS:
        messages.append(f"Загружено больше {BATCH_MAX_ROWS} строк; обработаны первые {BATCH_MAX_ROWS}.")

    prepared_rows: list[pd.DataFrame] = []
    valid_indexes: list[int] = []
    row_notes: dict[int, str] = {}
    row_statuses: dict[int, str] = {}
    row_similarity_scores: dict[int, int] = {}
    row_similarity_labels: dict[int, str] = {}
    for idx, row in limited_df.iterrows():
        try:
            raw = row_to_raw(row, defaults)
            context_label, context_score, _ = assess_prediction_context(df, raw, reference)
            row_similarity_scores[idx] = context_score
            row_similarity_labels[idx] = context_label
            errors, warnings = validate_prediction_inputs(raw)
            if errors:
                messages.append(f"Строка {idx}: {'; '.join(errors)}")
                row_statuses[idx] = "error"
                row_notes[idx] = "; ".join(errors)
                continue
            if warnings:
                messages.append(f"Строка {idx}: предупреждение - {'; '.join(warnings)}")
                row_statuses[idx] = "warning"
                row_notes[idx] = "; ".join(warnings)
            else:
                row_statuses[idx] = "ok"
                row_notes[idx] = ""
            prepared_rows.append(prepare_input(raw, df, all_features, cat_features, reference))
            valid_indexes.append(idx)
        except Exception as exc:
            messages.append(f"Строка {idx}: не удалось подготовить вход ({exc}).")
            row_statuses[idx] = "error"
            row_notes[idx] = str(exc)

    result = limited_df.copy()
    result["prediction_price_rub"] = np.nan
    result["prediction_price_sqm_rub"] = np.nan
    result["prediction_status"] = "not_processed"
    result["validation_notes"] = ""
    result["input_similarity_score"] = np.nan
    result["input_similarity_label"] = ""
    for idx, status in row_statuses.items():
        result.loc[idx, "prediction_status"] = status
        result.loc[idx, "validation_notes"] = row_notes.get(idx, "")
        result.loc[idx, "input_similarity_score"] = row_similarity_scores.get(idx, np.nan)
        result.loc[idx, "input_similarity_label"] = row_similarity_labels.get(idx, "")

    if prepared_rows:
        batch_input = pd.concat(prepared_rows, ignore_index=True)
        pred_price = np.exp(model.predict(batch_input))
        for source_idx, price in zip(valid_indexes, pred_price):
            result.loc[source_idx, "prediction_price_rub"] = price
            area = float(result.loc[source_idx, "area_total"]) if "area_total" in result.columns else np.nan
            result.loc[source_idx, "prediction_price_sqm_rub"] = price / area if area and not pd.isna(area) else np.nan

    return result, messages


def build_batch_report_html(result: pd.DataFrame, messages: list[str]) -> str:
    valid = result[result["prediction_price_rub"].notna()].copy()
    generated_at = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")

    summary_rows = [
        {"Метрика": "Строк в файле", "Значение": len(result)},
        {"Метрика": "Успешных прогнозов", "Значение": len(valid)},
        {"Метрика": "Строк с предупреждениями", "Значение": int(result["prediction_status"].eq("warning").sum()) if "prediction_status" in result else 0},
        {"Метрика": "Строк с ошибками", "Значение": int(result["prediction_status"].eq("error").sum()) if "prediction_status" in result else 0},
        {"Метрика": "Низкая близость к обучению", "Значение": int(result["input_similarity_score"].lt(55).sum()) if "input_similarity_score" in result else 0},
    ]
    if not valid.empty:
        summary_rows.extend(
            [
                {"Метрика": "Медианный прогноз", "Значение": format_rub(float(valid["prediction_price_rub"].median()))},
                {"Метрика": "Средний прогноз", "Значение": format_rub(float(valid["prediction_price_rub"].mean()))},
                {"Метрика": "Минимальный прогноз", "Значение": format_rub(float(valid["prediction_price_rub"].min()))},
                {"Метрика": "Максимальный прогноз", "Значение": format_rub(float(valid["prediction_price_rub"].max()))},
            ]
        )

    group_tables: list[str] = []
    for group_col, title in [("region", "По регионам"), ("district", "По районам")]:
        if group_col in valid.columns and not valid.empty:
            grouped = (
                valid.groupby(group_col, as_index=False)
                .agg(objects=("prediction_price_rub", "size"), median_prediction=("prediction_price_rub", "median"))
                .sort_values("median_prediction", ascending=False)
                .head(20)
            )
            grouped["median_prediction"] = grouped["median_prediction"].map(format_rub)
            group_tables.append(f"<h2>{html.escape(title)}</h2>{html_table(format_display_dataframe(grouped))}")

    top_cols = [
        col
        for col in [
            "prediction_price_rub",
            "prediction_price_sqm_rub",
            "region",
            "district",
            "rooms",
            "area_total",
            "prediction_status",
            "validation_notes",
            "input_similarity_score",
            "input_similarity_label",
        ]
        if col in valid.columns
    ]
    top_expensive = valid.sort_values("prediction_price_rub", ascending=False).head(10)[top_cols].copy() if not valid.empty else pd.DataFrame()
    top_cheap = valid.sort_values("prediction_price_rub", ascending=True).head(10)[top_cols].copy() if not valid.empty else pd.DataFrame()
    for frame in [top_expensive, top_cheap]:
        for col in ["prediction_price_rub", "prediction_price_sqm_rub"]:
            if col in frame:
                frame[col] = frame[col].map(lambda x: format_rub(float(x)) if not pd.isna(x) else "")
    top_expensive = format_display_dataframe(top_expensive) if not top_expensive.empty else top_expensive
    top_cheap = format_display_dataframe(top_cheap) if not top_cheap.empty else top_cheap

    status_problem = result.get("prediction_status", pd.Series(index=result.index, dtype=str)).isin(["warning", "error"])
    similarity_problem = result["input_similarity_score"].lt(55) if "input_similarity_score" in result.columns else pd.Series(False, index=result.index)
    problems = result[status_problem | similarity_problem].copy()
    problem_cols = [col for col in ["prediction_status", "validation_notes", "input_similarity_score", "input_similarity_label", "region", "district", "rooms", "area_total"] if col in problems.columns]
    problems = problems[problem_cols].head(30) if problem_cols else pd.DataFrame()
    problems = format_display_dataframe(problems) if not problems.empty else problems

    message_table = pd.DataFrame({"Сообщение": messages[:30]}) if messages else pd.DataFrame()

    return f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <title>Отчет по пакетной оценке Cian</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 36px; color: #111827; }}
    h1, h2 {{ margin-bottom: 8px; }}
    .muted {{ color: #6b7280; }}
    .report-table {{ width: 100%; border-collapse: collapse; margin: 10px 0 22px; font-size: 14px; }}
    .report-table th, .report-table td {{ border-bottom: 1px solid #e5e7eb; text-align: left; padding: 8px; }}
    .report-table th {{ background: #f3f4f6; }}
  </style>
</head>
<body>
  <h1>Отчет по пакетной оценке</h1>
  <p class="muted">Сформировано: {html.escape(generated_at)}.</p>
  <h2>Сводка</h2>
  {html_table(pd.DataFrame(summary_rows))}
  {''.join(group_tables)}
  <h2>Самые дорогие объекты</h2>
  {html_table(top_expensive)}
  <h2>Самые дешевые объекты</h2>
  {html_table(top_cheap)}
  <h2>Строки, требующие проверки</h2>
  {html_table(problems)}
  <h2>Сообщения валидации</h2>
  {html_table(message_table)}
</body>
</html>"""


def page_options_for_mode(presentation_mode: bool) -> list[str]:
    if presentation_mode:
        return ["Оценка квартиры", "Рынок", "Качество модели", "Паспорт модели", "Методология"]
    return ["Оценка квартиры", "Рынок", "Качество модели", "Пакетная оценка", "Паспорт модели", "Методология"]


NAV_LABELS = {
    "Оценка квартиры": "Оценка",
    "Рынок": "Рынок",
    "Качество модели": "Модели",
    "Пакетная оценка": "Пакет",
    "Паспорт модели": "Паспорт",
    "Методология": "Методология",
}


def sidebar_presentation_mode() -> bool:
    with st.sidebar:
        st.header("Настройки")
        st.session_state.presentation_mode = st.toggle("Режим презентации", value=st.session_state.get("presentation_mode", False))
        if st.session_state.presentation_mode:
            st.caption("Технические таблицы и раздел пакетной оценки скрыты для защиты.")

    return bool(st.session_state.get("presentation_mode", False))


def render_page_navigation(page_options: list[str]) -> str:
    current_page = st.session_state.get("current_page", page_options[0])
    if current_page not in page_options:
        current_page = page_options[0]

    if st.session_state.get("current_page_pills") not in page_options:
        st.session_state.current_page_pills = current_page

    page = st.segmented_control(
        "Раздел",
        page_options,
        key="current_page_pills",
        required=True,
        format_func=lambda value: NAV_LABELS.get(value, value),
        label_visibility="collapsed",
        width="stretch",
    )
    if page not in page_options:
        page = current_page

    st.session_state.current_page = page
    return page


def render_market_filters(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df.copy()

    with st.expander("Фильтры рынка", expanded=False):
        st.caption("Задают текущий рыночный срез для графиков и таблиц. На расчет цены квартиры не влияют.")

        if "region" in df.columns:
            regions = sorted_options(df["region"])
            selected_regions = st.multiselect("Регион", regions, default=regions)
            if selected_regions:
                filtered = filtered[filtered["region"].astype(str).isin(selected_regions)]
        else:
            selected_regions = []

        if "district" in df.columns:
            districts = district_options_for_regions(df, selected_regions, sorted_options(filtered["district"]))
            selected_districts = st.multiselect("Район", districts, default=[])
            if selected_districts:
                filtered = filtered[filtered["district"].astype(str).isin(selected_districts)]

        if "rooms" in df.columns:
            rooms = sorted_options(filtered["rooms"])
            selected_rooms = st.multiselect("Комнаты", rooms, default=[], format_func=option_label)
            if selected_rooms:
                filtered = filtered[filtered["rooms"].astype(str).isin(selected_rooms)]

        if "is_new_building" in df.columns:
            market = st.radio(
                "Тип рынка",
                ["Все", "Новостройки", "Вторичка"],
                index=0,
                horizontal=True,
            )
            if market == "Новостройки":
                filtered = filtered[filtered["is_new_building"]]
            elif market == "Вторичка":
                filtered = filtered[~filtered["is_new_building"]]

        if "price" in filtered.columns and len(filtered):
            low = int(filtered["price"].quantile(0.01))
            high = int(filtered["price"].quantile(0.99))
            low_mln = math.floor(low / 1_000_000 * 2) / 2
            high_mln = math.ceil(high / 1_000_000 * 2) / 2
            price_range = st.slider(
                "Цена, млн ₽",
                min_value=max(0.0, low_mln),
                max_value=high_mln,
                value=(max(0.0, low_mln), high_mln),
                step=0.5,
            )
            filtered = filtered[
                filtered["price"].between(price_range[0] * 1_000_000, price_range[1] * 1_000_000)
            ]

        if len(filtered):
            st.caption(f"В срезе: {len(filtered):,} объектов".replace(",", " "))
        else:
            st.warning("В выбранном рыночном срезе нет объектов.")

    return filtered


def show_overview(df: pd.DataFrame, presentation_mode: bool = False) -> None:
    render_section_title("Рынок и данные", "Обзор структуры объявлений, географии и ценовых распределений.")
    filtered = render_market_filters(df)
    if filtered.empty:
        st.warning("По выбранным фильтрам нет объектов.")
        return

    if presentation_mode:
        tab_summary, tab_geo, tab_price, tab_districts = st.tabs(["Обзор", "География", "Цена и площадь", "Районы"])
        tab_data = None
    else:
        tab_summary, tab_geo, tab_price, tab_districts, tab_data = st.tabs(
            ["Обзор", "География", "Цена и площадь", "Районы", "Данные"]
        )

    with tab_summary:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Объектов", f"{len(filtered):,}".replace(",", " "))
        col2.metric("Медианная цена", format_rub(filtered["price"].median()))
        col3.metric("Медиана за м²", format_rub(filtered["cost_sqm"].median()) if "cost_sqm" in filtered else "н/д")
        col4.metric("Медианная площадь", f"{filtered['area_total'].median():.1f} м²")
        summary_lines = build_executive_takeaways(filtered, {"leaderboard": pd.DataFrame()})
        if summary_lines:
            add_insight(" ".join(summary_lines[:2]))

        charts_left, charts_right = st.columns((1.1, 0.9))
        with charts_left:
            fig = px.histogram(
                filtered,
                x="price",
                nbins=60,
                title="Распределение цены",
                labels={"price": "Цена, ₽", "count": "Количество"},
            )
            fig.update_layout(showlegend=False, height=360, margin=dict(l=10, r=10, t=45, b=10))
            st.plotly_chart(fig, width="stretch")

        with charts_right:
            if "is_new_building" in filtered.columns:
                market_counts = (
                    filtered.assign(market=np.where(filtered["is_new_building"], "Новостройки", "Вторичка"))
                    .groupby("market", as_index=False)
                    .size()
                )
                fig = px.pie(
                    market_counts,
                    names="market",
                    values="size",
                    hole=0.45,
                    title="Структура рынка",
                )
                fig.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10))
                st.plotly_chart(fig, width="stretch")

        if "region" in filtered.columns:
            region_summary = (
                filtered.groupby("region", as_index=False)
                .agg(objects=("price", "size"), median_price=("price", "median"), median_area=("area_total", "median"))
                .sort_values("objects", ascending=False)
            )
            region_summary["median_price"] = region_summary["median_price"].map(format_rub)
            region_summary["median_area"] = region_summary["median_area"].map(lambda x: f"{x:.1f} м²")
            st.dataframe(format_display_dataframe(region_summary), width="stretch", hide_index=True)
            if len(region_summary) >= 2:
                add_insight("Региональный разрез нужен как базовый контроль: СПб и Ленинградская область формируют разные рыночные подгруппы, поэтому их нельзя смешивать без проверки долей.")

    with tab_geo:
        if {"lat", "lng", "price"}.issubset(filtered.columns):
            sample = filtered.sample(min(len(filtered), 6000), random_state=42) if len(filtered) > 6000 else filtered
            fig = px.scatter_map(
                sample,
                lat="lat",
                lon="lng",
                color="price",
                zoom=8.6,
                height=520,
                title="География объявлений",
                color_continuous_scale="Viridis",
                map_style="open-street-map",
                hover_data=[col for col in ["district", "rooms", "area_total", "price"] if col in sample.columns],
            )
            fig.update_layout(margin=dict(l=10, r=10, t=45, b=10))
            st.plotly_chart(fig, width="stretch")
            if {"district", "cost_sqm"}.issubset(filtered.columns):
                district_prices = filtered.groupby("district")["cost_sqm"].median().sort_values(ascending=False)
                if not district_prices.empty:
                    add_insight(
                        f"География подтверждает неоднородность рынка: верхний район по медиане за м² — {district_prices.index[0]}, "
                        f"нижний — {district_prices.index[-1]}. Это главный аргумент держать район и регион в модели."
                    )
            else:
                add_insight("Карта показывает географию объявлений, а не фактических сделок; плотность точек нельзя трактовать как объем продаж.")
        else:
            st.info("В датасете нет координат для карты.")

    with tab_price:
        sample = filtered.sample(min(len(filtered), 5000), random_state=42) if len(filtered) > 5000 else filtered
        row1, row2 = st.columns(2)
        with row1:
            if {"area_total", "price", "is_new_building"}.issubset(filtered.columns):
                fig = px.scatter(
                    sample,
                    x="area_total",
                    y="price",
                    color="is_new_building",
                    opacity=0.45,
                    title="Цена и площадь",
                    labels={"area_total": "Площадь, м²", "price": "Цена, ₽", "is_new_building": "Новостройка"},
                )
                fig.update_layout(height=420, margin=dict(l=10, r=10, t=45, b=10))
                st.plotly_chart(fig, width="stretch")
        with row2:
            if {"rooms", "price"}.issubset(filtered.columns):
                fig = px.box(
                    filtered,
                    x="rooms",
                    y="price",
                    points=False,
                    title="Цена по числу комнат",
                    labels={"rooms": "Комнаты", "price": "Цена, ₽"},
                )
                fig.update_layout(height=420, margin=dict(l=10, r=10, t=45, b=10))
                st.plotly_chart(fig, width="stretch")

        if {"cost_sqm", "area_total"}.issubset(filtered.columns):
            fig = px.scatter(
                sample,
                x="area_total",
                y="cost_sqm",
                color="region" if "region" in sample.columns else None,
                opacity=0.45,
                title="Цена за м² и площадь",
                labels={"area_total": "Площадь, м²", "cost_sqm": "Цена за м², ₽"},
            )
            fig.update_layout(margin=dict(l=10, r=10, t=45, b=10))
            st.plotly_chart(fig, width="stretch")
            corr = sample[["area_total", "cost_sqm"]].dropna().corr().iloc[0, 1]
            if not pd.isna(corr):
                direction = "снижается" if corr < 0 else "растет"
                add_insight(
                    f"Связь площади и цены за м² в текущем срезе: корреляция {corr:.2f}; цена за м² в среднем {direction} вместе с площадью. "
                    "Поэтому полная цена и цена за м² должны анализироваться отдельно."
                )

    with tab_districts:
        if {"district", "price"}.issubset(filtered.columns):
            district_summary = (
                filtered.groupby("district", as_index=False)
                .agg(
                    objects=("price", "size"),
                    median_price=("price", "median"),
                    median_cost_sqm=("cost_sqm", "median") if "cost_sqm" in filtered.columns else ("price", "median"),
                    median_area=("area_total", "median"),
                )
                .sort_values("median_price", ascending=False)
            )
            fig = px.bar(
                district_summary.head(15).sort_values("median_price", ascending=True),
                x="median_price",
                y="district",
                orientation="h",
                title="Топ районов по медианной цене",
                labels={"median_price": "Медианная цена, ₽", "district": ""},
            )
            fig.update_layout(height=480, margin=dict(l=10, r=10, t=45, b=10))
            st.plotly_chart(fig, width="stretch")

            table = district_summary.copy()
            table["median_price"] = table["median_price"].map(format_rub)
            table["median_cost_sqm"] = table["median_cost_sqm"].map(format_rub)
            table["median_area"] = table["median_area"].map(lambda x: f"{x:.1f} м²")
            st.dataframe(table, width="stretch", hide_index=True)
            if len(district_summary) >= 2:
                top_row = district_summary.iloc[0]
                bottom_row = district_summary.iloc[-1]
                add_insight(
                    f"Районы отличаются не только уровнем цены, но и составом предложения: {top_row['district']} лидирует по медианной цене, "
                    f"а {bottom_row['district']} находится внизу списка. При сравнении объектов районный контекст обязателен."
                )

    if tab_data is not None:
        with tab_data:
            st.caption("Первые строки и качество заполнения после фильтров. Данные отражают объявления, а не подтвержденные сделки.")
            visible_cols = [
                col
                for col in [
                    "price",
                    "area_total",
                    "rooms",
                    "region",
                    "district",
                    "nearest_metro",
                    "is_new_building",
                    "build_year",
                    "current_floor",
                    "total_floors",
                    "type_building",
                ]
                if col in filtered.columns
            ]
            st.dataframe(format_display_dataframe(filtered[visible_cols].head(1000)), width="stretch", hide_index=True)

            missing = (
                filtered[visible_cols]
                .isna()
                .sum()
                .rename("missing")
                .reset_index()
                .rename(columns={"index": "column"})
            )
            missing["column"] = missing["column"].map(lambda value: DISPLAY_COLUMN_NAMES.get(value, value))
            st.dataframe(missing.rename(columns={"column": "Поле", "missing": "Пропусков"}), width="stretch", hide_index=True)


def show_models(df: pd.DataFrame, model: Any, metadata: dict[str, Any], model_metrics: dict[str, Any], presentation_mode: bool = False) -> None:
    render_section_title("Качество модели", "Сравнение моделей, ошибки на тесте и факторы лучшей модели.")
    if model_metrics.get("source") or model_metrics.get("updated_at"):
        st.caption(
            f"Источник метрик: {model_metrics.get('source', 'model_metrics.json')}; "
            f"обновлено: {model_metrics.get('updated_at', 'н/д')}."
        )

    cat_features, _, all_features = get_model_features(model)
    quality = build_quality_frame(model, df, all_features, cat_features)
    leaderboard = model_metrics["leaderboard"]
    baseline = model_metrics["baseline"]
    top_segments = model_metrics["top_segments"]
    top_features = model_metrics["top_features"]
    shap_importance = prepare_shap_importance(top_features)
    xgb_importance = build_xgb_feature_importance(model)
    importance_comparison = build_feature_importance_comparison(shap_importance, xgb_importance)

    best = leaderboard.iloc[0]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Выбранная модель", best["model"])
    col2.metric("CV RMSE log", f"{best['cv_rmse_log']:.4f}")
    col3.metric("Test RMSE log", f"{best['test_rmse_log']:.4f}")
    col4.metric("Test MAPE", f"{best['test_mape_price']:.1%}")
    st.info(
        f"В прикладном чтении test MAPE {best['test_mape_price']:.1%} означает, что точечный прогноз не должен идти без интервала, "
        "проверки похожих объектов и сегментного контекста."
    )
    if len(leaderboard) >= 2 and {"cv_rmse_log", "model"}.issubset(leaderboard.columns):
        gap = float(leaderboard.iloc[1]["cv_rmse_log"] - leaderboard.iloc[0]["cv_rmse_log"])
        st.caption(
            f"Разрыв между первой и второй моделью по CV RMSE log: {gap:.4f}; поэтому выбор модели стоит читать как текущий лидер leaderboard, а не абсолютное доминирование."
        )
    if not top_segments.empty and {"segment", "mae_log"}.issubset(top_segments.columns):
        risky = top_segments.sort_values("mae_log", ascending=False).iloc[0]
        st.warning(
            f"Зона повышенного риска по leaderboard: {risky['segment']} "
            f"(MAE log {float(risky['mae_log']):.4f}). Такие объекты стоит проверять вручную по рынку."
        )

    if presentation_mode:
        tab_board, tab_features = st.tabs(["Leaderboard", "Факторы"])
        tab_residuals = None
        tab_segments = None
    else:
        tab_board, tab_residuals, tab_segments, tab_features = st.tabs(
            ["Leaderboard", "Остатки", "Сегменты и worst cases", "Факторы"]
        )

    with tab_board:
        display_cols = [
            "rank",
            "participant",
            "model",
            "cv_rmse_log",
            "test_rmse_log",
            "test_mae_log",
            "test_r2_log",
            "test_mape_price",
            "notes",
        ]
        leaderboard_display = leaderboard[display_cols].copy()
        leaderboard_display["test_mape_price"] = leaderboard_display["test_mape_price"].map(lambda x: f"{x:.1%}")
        st.dataframe(leaderboard_display, width="stretch", hide_index=True)

        fig = px.bar(
            leaderboard.sort_values("cv_rmse_log", ascending=False),
            x="cv_rmse_log",
            y="model",
            color="participant",
            orientation="h",
            title="Сравнение моделей по CV RMSE log",
            labels={"cv_rmse_log": "CV RMSE log", "model": ""},
        )
        fig.update_layout(height=330, margin=dict(l=10, r=10, t=45, b=10), showlegend=False)
        st.plotly_chart(fig, width="stretch")

        col1, col2 = st.columns(2)
        with col1:
            st.caption("Baseline")
            baseline_display = baseline.copy()
            baseline_display["test_mape_price"] = baseline_display["test_mape_price"].map(lambda x: f"{x:.1%}")
            st.dataframe(baseline_display, width="stretch", hide_index=True)

        with col2:
            metrics = metadata.get("metrics", {})
            if metrics:
                st.caption("Сохраненная локальная модель")
                model_info = pd.DataFrame(
                    [
                        {"metric": "MAE log", "value": f"{metrics.get('mae_log', np.nan):.4f}"},
                        {"metric": "RMSE log", "value": f"{metrics.get('rmse_log', np.nan):.4f}"},
                        {"metric": "R² log", "value": f"{metrics.get('r2_log', np.nan):.4f}"},
                        {"metric": "MAPE price", "value": f"{metrics.get('mape_price', np.nan):.1%}"},
                        {"metric": "Median APE price", "value": f"{metrics.get('median_ape_price', np.nan):.1%}"},
                    ]
                )
                st.dataframe(model_info, width="stretch", hide_index=True)

    if tab_residuals is not None:
        with tab_residuals:
            if quality.empty:
                st.info("Недостаточно признаков для диагностического пересчета ошибок по доступному датасету.")
            else:
                residual_source = quality["residual_source"].iloc[0] if "residual_source" in quality.columns else "н/д"
                if residual_source == "holdout_test":
                    st.caption("Остатки и сегментные ошибки ниже пересчитаны на holdout `X_test/y_test` из `split_data.pkl`.")
                else:
                    st.caption("Остатки ниже являются fallback-диагностикой на доступном датасете приложения; для финальной оценки используйте leaderboard/test-метрики.")
                q1, q2, q3, q4 = st.columns(4)
                q1.metric("MAE log", f"{quality['abs_error_log'].mean():.4f}")
                q2.metric("RMSE log", f"{np.sqrt(np.mean(quality['residual_log'] ** 2)):.4f}")
                q3.metric("MAPE", f"{quality['ape_price'].mean():.1%}")
                q4.metric("Медианная APE", f"{quality['ape_price'].median():.1%}")

                col1, col2 = st.columns(2)
                with col1:
                    fig = px.histogram(
                        quality,
                        x="residual_log",
                        nbins=70,
                        title="Распределение остатков log(price)",
                        labels={"residual_log": "Факт - прогноз, log(price)"},
                    )
                    fig.update_layout(height=390, margin=dict(l=10, r=10, t=45, b=10))
                    st.plotly_chart(fig, width="stretch")
                with col2:
                    fig = px.scatter(
                        quality,
                        x="price_pred",
                        y="residual_log",
                        opacity=0.35,
                        title="Остатки относительно прогноза",
                        labels={"price_pred": "Прогноз, ₽", "residual_log": "Остаток log(price)"},
                    )
                    fig.update_layout(height=390, margin=dict(l=10, r=10, t=45, b=10))
                    st.plotly_chart(fig, width="stretch")

    if tab_segments is not None:
        with tab_segments:
            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(
                    top_segments.sort_values("mae_log", ascending=True),
                    x="mae_log",
                    y="segment",
                    orientation="h",
                    title="Где модель ошибается сильнее",
                    labels={"mae_log": "MAE log", "segment": ""},
                )
                fig.update_layout(height=410, margin=dict(l=10, r=10, t=45, b=10))
                st.plotly_chart(fig, width="stretch")
                if not top_segments.empty:
                    worst_segment = top_segments.sort_values("mae_log", ascending=False).iloc[0]
                    add_insight(
                        f"Этот график отвечает не на вопрос “какая модель лучшая”, а на вопрос “где ей верить осторожнее”. "
                        f"Самый проблемный сегмент сейчас: {worst_segment['segment']}."
                    )

            with col2:
                if not quality.empty and {"is_new_building", "is_spb", "is_first_floor"}.issubset(quality.columns):
                    seg = (
                        quality.groupby(["is_new_building", "is_spb", "is_first_floor"], as_index=False)
                        .agg(mae_log=("abs_error_log", "mean"), objects=("price", "size"))
                        .sort_values("mae_log", ascending=False)
                    )
                    seg["segment"] = seg.apply(
                        lambda row: (
                            ("Новостройка" if row["is_new_building"] else "Вторичка")
                            + ", "
                            + ("СПб" if row["is_spb"] else "ЛО")
                            + ", "
                            + ("первый этаж" if row["is_first_floor"] else "не первый этаж")
                        ),
                        axis=1,
                    )
                    fig = px.bar(
                        seg.sort_values("mae_log", ascending=True),
                        x="mae_log",
                        y="segment",
                        orientation="h",
                        title="Диагностические ошибки по группам",
                        labels={"mae_log": "MAE log", "segment": ""},
                    )
                    fig.update_layout(height=410, margin=dict(l=10, r=10, t=45, b=10))
                    st.plotly_chart(fig, width="stretch")

            if not quality.empty:
                st.caption("Worst cases в диагностическом срезе")
                worst_cols = [
                    col
                    for col in [
                        "price",
                        "price_pred",
                        "abs_error_rub",
                        "ape_price",
                        "area_total",
                        "rooms",
                        "region",
                        "district",
                        "is_new_building",
                        "build_year",
                    ]
                    if col in quality.columns
                ]
                worst = quality.sort_values("abs_error_rub", ascending=False).head(15)[worst_cols].copy()
                for col in ["price", "price_pred", "abs_error_rub"]:
                    if col in worst.columns:
                        worst[col] = worst[col].map(format_rub)
                if "ape_price" in worst.columns:
                    worst["ape_price"] = worst["ape_price"].map(lambda x: f"{x:.1%}")
                st.dataframe(format_display_dataframe(worst), width="stretch", hide_index=True)

    with tab_features:
        shap_tab, builtin_tab, compare_tab = st.tabs(["SHAP", "Важность XGBoost", "Сравнение"])

        with shap_tab:
            if shap_importance.empty:
                st.info("SHAP-важности не найдены в model_metrics.json.")
            else:
                st.caption("Mean |SHAP| показывает средний абсолютный вклад фактора в прогноз. Это важность по влиянию на значение прогноза, а не причинный эффект.")
                shap_plot = shap_importance.head(15).sort_values("importance", ascending=True)
                fig = px.bar(
                    shap_plot,
                    x="importance",
                    y="factor",
                    orientation="h",
                    title="Top SHAP факторов лучшей модели",
                    labels={"importance": "Mean |SHAP|", "factor": ""},
                )
                fig.update_layout(height=430, margin=dict(l=10, r=10, t=45, b=10))
                st.plotly_chart(fig, width="stretch")
                shap_table = shap_importance.head(20)[["factor", "importance", "importance_norm"]].copy()
                shap_table["importance"] = shap_table["importance"].map(lambda x: f"{x:.4f}")
                shap_table["importance_norm"] = shap_table["importance_norm"].map(lambda x: f"{x:.1%}")
                st.dataframe(
                    shap_table.rename(
                        columns={
                            "factor": "Фактор",
                            "importance": "Mean |SHAP|",
                            "importance_norm": "Доля среди показанных факторов",
                        }
                    ),
                    width="stretch",
                    hide_index=True,
                )

        with builtin_tab:
            if xgb_importance.empty:
                st.info("Встроенная важность XGBoost недоступна для сохраненной модели.")
            else:
                st.caption("Feature importance по gain показывает, насколько разбиения по признаку в среднем улучшали функцию потерь в деревьях XGBoost.")
                xgb_plot = xgb_importance.head(15).sort_values("gain", ascending=True)
                fig = px.bar(
                    xgb_plot,
                    x="gain",
                    y="factor",
                    orientation="h",
                    title="Встроенная важность XGBoost: gain",
                    labels={"gain": "Gain", "factor": ""},
                )
                fig.update_layout(height=430, margin=dict(l=10, r=10, t=45, b=10))
                st.plotly_chart(fig, width="stretch")
                xgb_table = xgb_importance.head(20)[["factor", "gain", "gain_norm", "split_count"]].copy()
                xgb_table["gain"] = xgb_table["gain"].map(lambda x: f"{x:.4f}")
                xgb_table["gain_norm"] = xgb_table["gain_norm"].map(lambda x: f"{x:.1%}")
                st.dataframe(
                    xgb_table.rename(
                        columns={
                            "factor": "Фактор",
                            "gain": "Gain",
                            "gain_norm": "Доля gain",
                            "split_count": "Число split",
                        }
                    ),
                    width="stretch",
                    hide_index=True,
                )

        with compare_tab:
            if importance_comparison.empty:
                st.info("Недостаточно данных, чтобы сопоставить SHAP и встроенную важность.")
            else:
                st.caption("Сравнение нормировано внутри каждого метода. Совпадение вверху списка усиливает доверие к фактору; расхождения нормальны, потому что методы отвечают на разные вопросы.")
                compare_view = importance_comparison.head(15).copy()
                compare_long = compare_view.melt(
                    id_vars=["factor"],
                    value_vars=["importance_norm", "gain_norm"],
                    var_name="method",
                    value_name="score",
                )
                compare_long["method"] = compare_long["method"].map(
                    {"importance_norm": "SHAP", "gain_norm": "XGBoost gain"}
                )
                fig = px.bar(
                    compare_long,
                    x="score",
                    y="factor",
                    color="method",
                    orientation="h",
                    barmode="group",
                    title="SHAP vs встроенная важность",
                    labels={"score": "Нормированная важность", "factor": "", "method": ""},
                )
                fig.update_layout(height=520, margin=dict(l=10, r=10, t=45, b=10))
                st.plotly_chart(fig, width="stretch")
                table = compare_view[["factor", "importance_norm", "gain_norm"]].copy()
                table["importance_norm"] = table["importance_norm"].map(lambda x: f"{x:.1%}")
                table["gain_norm"] = table["gain_norm"].map(lambda x: f"{x:.1%}")
                st.dataframe(
                    table.rename(
                        columns={
                            "factor": "Фактор",
                            "importance_norm": "SHAP",
                            "gain_norm": "XGBoost gain",
                        }
                    ),
                    width="stretch",
                    hide_index=True,
                )


def show_prediction(df: pd.DataFrame, model: Any, reference: dict[str, Any], metadata: dict[str, Any], presentation_mode: bool = False) -> None:
    render_section_title(
        "Оценка квартиры",
        "Заполните параметры объекта. После расчета появятся цена, диапазон, рыночный вердикт и похожие объявления.",
    )

    cat_features, _, all_features = get_model_features(model)

    data_defaults = {
        "lat": median_number(df, "lat", 59.9343),
        "lng": median_number(df, "lng", 30.3351),
        "area_total": median_number(df, "area_total", 52.0),
        "area_living": median_number(df, "area_living", 30.0),
        "area_kitchen": median_number(df, "area_kitchen", 10.0),
        "build_year": int(median_number(df, "build_year", 2010)),
        "current_floor": int(median_number(df, "current_floor", 5)),
        "total_floors": int(median_number(df, "total_floors", 12)),
        "metro_travel_time": int(median_number(df, "metro_travel_time", 10)),
        "region": "Санкт-Петербург",
        "district": "Центральный",
        "nearest_metro": "Невский проспект",
        "metro_travel_type": "walk",
        "rooms": "2",
        "is_studio": False,
        "status_home": "Сдан",
        "type_building": "Монолитный",
        "is_new_building": False,
    }

    preset_name = st.selectbox("Быстрый профиль", list(PREDICTION_PRESETS), index=0)
    defaults = {**data_defaults, **PREDICTION_PRESETS[preset_name]}

    if "picked_lat" not in st.session_state:
        st.session_state.picked_lat = defaults["lat"]
    if "picked_lng" not in st.session_state:
        st.session_state.picked_lng = defaults["lng"]
    if st.session_state.get("active_preset") != preset_name:
        st.session_state.active_preset = preset_name
        st.session_state.picked_lat = defaults["lat"]
        st.session_state.picked_lng = defaults["lng"]

    st.markdown("**Основные параметры объекта**")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Локация**")
        region_options = reference["region"] or ["Санкт-Петербург", "Ленинградская область"]
        region = st.selectbox("Регион", region_options, index=option_index(region_options, defaults["region"]))
        district_fallback = reference["district"] or sorted_options(df.get("district", pd.Series(dtype=str)))
        district_options = district_options_for_regions(df, [region], district_fallback)
        district = st.selectbox("Район", district_options, index=option_index(district_options, defaults["district"]))
        nearest_fallback = reference["nearest_metro"] or sorted_options(df.get("nearest_metro", pd.Series(dtype=str)))
        nearest_options = metro_options_for_location(df, region, district, nearest_fallback)
        nearest_metro = st.selectbox(
            "Ближайшее метро",
            nearest_options,
            index=option_index(nearest_options, defaults["nearest_metro"]),
        )
        metro_travel_time = st.slider("Время до метро, мин", 1, 40, min(max(int(defaults["metro_travel_time"]), 1), 40))

        st.markdown("**Квартира**")
        area_total = st.number_input("Общая площадь, м²", min_value=10.0, max_value=1000.0, value=round(float(defaults["area_total"]), 1), step=1.0)
        rooms_options = reference["rooms"] or ["studio", "1", "2", "3", "4", "5", "free_plan"]
        rooms = st.selectbox(
            "Комнаты",
            rooms_options,
            index=option_index(rooms_options, defaults["rooms"]),
            format_func=option_label,
        )
        is_studio = st.toggle("Студия", value=bool(defaults["is_studio"]))

    with col2:
        st.markdown("**Дом и рынок**")
        build_year = st.number_input("Год постройки", min_value=1850, max_value=CURRENT_YEAR, value=int(defaults["build_year"]), step=1)
        total_floors = st.number_input("Этажей в доме", min_value=1, max_value=100, value=max(int(defaults["total_floors"]), 1), step=1)
        current_floor = st.number_input("Этаж квартиры", min_value=1, max_value=int(total_floors), value=min(max(int(defaults["current_floor"]), 1), int(total_floors)), step=1)
        is_new_building = st.toggle("Новостройка", value=bool(defaults["is_new_building"]))

    with st.expander("Дополнительные параметры", expanded=False):
        extra_col1, extra_col2 = st.columns(2)
        with extra_col1:
            metro_type_options = reference["metro_travel_type"] or ["walk", "transport"]
            metro_travel_type = st.selectbox(
                "Способ добраться до метро",
                metro_type_options,
                index=option_index(metro_type_options, defaults["metro_travel_type"]),
                format_func=option_label,
            )
            area_living = st.number_input(
                "Жилая площадь, м²",
                min_value=0.0,
                max_value=max(float(area_total), 1.0),
                value=min(round(float(defaults["area_living"]), 1), float(area_total)),
                step=1.0,
            )
            area_kitchen = st.number_input(
                "Кухня, м²",
                min_value=0.0,
                max_value=max(float(area_total), 1.0),
                value=min(round(float(defaults["area_kitchen"]), 1), float(area_total)),
                step=1.0,
            )
        with extra_col2:
            status_options = reference["status_home"] or ["Сдан", "Не сдан"]
            status_home = st.selectbox("Статус дома", status_options, index=option_index(status_options, defaults["status_home"]))
            type_options = reference["type_building"] or ["Монолитный", "Кирпичный", "Панельный", "none_type"]
            type_building = st.selectbox(
                "Тип здания",
                type_options,
                index=option_index(type_options, defaults["type_building"]),
                format_func=option_label,
            )

    lat = float(st.session_state.picked_lat)
    lng = float(st.session_state.picked_lng)
    with st.expander("Уточнить точку на карте", expanded=False):
        st.caption("Карту можно использовать для уточнения локации. Числовые координаты скрыты, потому что обычно их не нужно вводить вручную.")
        if HAS_MAP:
            map_obj = folium.Map(
                location=[st.session_state.picked_lat, st.session_state.picked_lng],
                zoom_start=11,
                tiles="OpenStreetMap",
            )
            folium.Marker(
                [st.session_state.picked_lat, st.session_state.picked_lng],
                tooltip="Выбранная точка",
            ).add_to(map_obj)
            map_data = st_folium(map_obj, height=360, use_container_width=True)
            if map_data and map_data.get("last_clicked"):
                st.session_state.picked_lat = float(map_data["last_clicked"]["lat"])
                st.session_state.picked_lng = float(map_data["last_clicked"]["lng"])
                st.rerun()

        if st.checkbox("Показать координаты вручную", value=False):
            coords_col1, coords_col2 = st.columns(2)
            with coords_col1:
                lat = st.number_input("Широта", min_value=58.0, max_value=61.5, value=float(st.session_state.picked_lat), step=0.001, format="%.6f")
            with coords_col2:
                lng = st.number_input("Долгота", min_value=28.0, max_value=32.5, value=float(st.session_state.picked_lng), step=0.001, format="%.6f")
            st.session_state.picked_lat = lat
            st.session_state.picked_lng = lng

    raw = {
        "region": region,
        "district": district,
        "nearest_metro": nearest_metro,
        "metro_travel_type": metro_travel_type,
        "metro_travel_time": metro_travel_time,
        "area_total": area_total,
        "area_living": area_living,
        "area_kitchen": area_kitchen,
        "rooms": rooms,
        "is_studio": is_studio,
        "build_year": build_year,
        "total_floors": total_floors,
        "current_floor": current_floor,
        "status_home": status_home,
        "type_building": type_building,
        "is_new_building": is_new_building,
        "lat": lat,
        "lng": lng,
    }
    current_prediction_signature = json.dumps(
        {"preset": preset_name, "raw": raw},
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )

    errors, warnings = validate_prediction_inputs(raw)
    for warning in warnings:
        st.warning(warning)
    for error in errors:
        st.error(error)

    input_df = prepare_input(raw, df, all_features, cat_features, reference)
    context_label, context_score, context_details = assess_prediction_context(df, raw, reference)

    col_quality, col_quality_detail = st.columns((0.35, 0.65))
    with col_quality:
        st.metric("Похожесть на данные модели", f"{context_score}/100", context_label)
    with col_quality_detail:
        if context_score >= 80:
            st.success("Объект выглядит типичным для данных модели.")
        elif context_score >= 55:
            st.warning("Объект похож на данные, но часть признаков редкая.")
        else:
            st.error("Объект находится вне комфортной зоны модели; прогноз стоит трактовать осторожно.")
        st.caption("Чем выше похожесть, тем меньше риск, что модель переносится на непривычный для нее объект.")
    if not presentation_mode:
        with st.expander("Почему такой уровень доверия", expanded=False):
            context_view = context_details.copy()
            for column in context_view.columns:
                context_view[column] = context_view[column].astype(str)
            st.dataframe(context_view, width="stretch", hide_index=True)

    col1, col2 = st.columns((0.8, 1.2))
    with col1:
        predict = st.button("Рассчитать цену", type="primary", width="stretch", disabled=bool(errors))
    with col2:
        st.caption("После расчета появятся цена, диапазон, рыночный вердикт и похожие объекты для проверки.")

    if predict and not errors:
        payload = build_single_prediction_payload(
            raw=raw,
            df=df,
            model=model,
            input_df=input_df,
            all_features=all_features,
            cat_features=cat_features,
            reference=reference,
            metadata=metadata,
            context_label=context_label,
            context_score=context_score,
        )
        st.session_state.last_prediction_payload = payload
        st.session_state.last_prediction_signature = current_prediction_signature
        append_prediction_history(preset_name, raw, float(payload["pred_price"]))

    if st.session_state.get("last_prediction_payload"):
        if st.session_state.get("last_prediction_signature") == current_prediction_signature:
            st.divider()
            render_prediction_payload(st.session_state.last_prediction_payload)
        else:
            st.warning("Параметры объекта изменились после последнего расчета. Нажмите `Рассчитать цену`, чтобы обновить прогноз.")

    show_prediction_history()

    if not presentation_mode:
        with st.expander("Технические признаки, отправленные в модель", expanded=False):
            debug_df = input_df.T.rename(columns={0: "value"})
            debug_df["value"] = debug_df["value"].astype(str)
            st.dataframe(debug_df, width="stretch")


def show_batch_prediction(df: pd.DataFrame, model: Any, reference: dict[str, Any]) -> None:
    render_section_title("Пакетная оценка объектов", "Загрузите CSV и получите прогнозы для нескольких объектов одним расчетом.")
    st.caption("Сценарий для портфеля: быстро оценить группу объявлений, найти крайние цены и выделить строки для ручной проверки.")

    cat_features, _, all_features = get_model_features(model)
    template = pd.DataFrame(
        [
            PREDICTION_PRESETS["2-комнатная квартира у метро, Московский район"],
            PREDICTION_PRESETS["Студия в новостройке, Приморский район"],
            PREDICTION_PRESETS["Семейная квартира в Ленинградской области"],
        ]
    )

    col1, col2 = st.columns((0.8, 1.2))
    with col1:
        template_download = format_display_dataframe(template, rename_columns=False, format_numbers=False)
        st.download_button(
            "Скачать шаблон CSV",
            data=template_download.to_csv(index=False).encode("utf-8-sig"),
            file_name="batch_prediction_template.csv",
            mime="text/csv",
            width="stretch",
        )
    with col2:
        uploaded = st.file_uploader("Загрузить CSV", type=["csv"])

    if uploaded is None:
        st.dataframe(format_display_dataframe(template), width="stretch", hide_index=True)
        return

    try:
        uploaded_df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Не удалось прочитать CSV: {exc}")
        return

    result, messages = run_batch_predictions(uploaded_df, df, model, all_features, cat_features, reference)
    valid_count = int(result["prediction_price_rub"].notna().sum())

    col1, col2, col3 = st.columns(3)
    col1.metric("Строк в файле", len(uploaded_df))
    col2.metric("Обработано", len(result))
    col3.metric("Успешных прогнозов", valid_count)

    for message in messages[:10]:
        st.warning(message)
    if len(messages) > 10:
        st.info(f"Показаны первые 10 сообщений из {len(messages)}.")

    view = result.copy()
    for col in ["prediction_price_rub", "prediction_price_sqm_rub"]:
        if col in view.columns:
            view[col] = view[col].map(lambda x: format_rub(x) if not pd.isna(x) else "")
    st.dataframe(format_display_dataframe(view.head(BATCH_MAX_ROWS)), width="stretch", hide_index=True)

    valid = result[result["prediction_price_rub"].notna()].copy()
    if not valid.empty:
        st.subheader("Сводка пакетной оценки")
        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
        summary_col1.metric("Медианный прогноз", format_rub(float(valid["prediction_price_rub"].median())))
        summary_col2.metric("Средний прогноз", format_rub(float(valid["prediction_price_rub"].mean())))
        summary_col3.metric("Медиана за м²", format_rub(float(valid["prediction_price_sqm_rub"].median())))
        low_similarity = int(valid["input_similarity_score"].lt(55).sum()) if "input_similarity_score" in valid.columns else 0
        summary_col4.metric("На ручную проверку", low_similarity)
        add_insight(
            "Пакетная оценка полезна как первичный скрининг: крайние прогнозы и строки с предупреждениями должны уходить на ручную проверку."
        )

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            fig = px.histogram(
                valid,
                x="prediction_price_rub",
                nbins=30,
                title="Распределение прогнозов по портфелю",
                labels={"prediction_price_rub": "Прогноз цены, ₽", "count": "Количество"},
            )
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10))
            st.plotly_chart(fig, width="stretch")

        with chart_col2:
            group_col = "district" if "district" in valid.columns else ("region" if "region" in valid.columns else None)
            if group_col is not None:
                grouped = (
                    valid.groupby(group_col, as_index=False)
                    .agg(objects=("prediction_price_rub", "size"), median_prediction=("prediction_price_rub", "median"))
                    .sort_values("median_prediction", ascending=False)
                    .head(15)
                )
                fig = px.bar(
                    grouped.sort_values("median_prediction"),
                    x="median_prediction",
                    y=group_col,
                    orientation="h",
                    title="Медианный прогноз по группам",
                    labels={"median_prediction": "Медианный прогноз, ₽", group_col: ""},
                    hover_data=["objects"],
                )
                fig.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10))
                st.plotly_chart(fig, width="stretch")

        top_cols = [
            col
            for col in [
                "prediction_price_rub",
                "prediction_price_sqm_rub",
                "region",
                "district",
                "rooms",
                "area_total",
                "prediction_status",
                "validation_notes",
                "input_similarity_score",
                "input_similarity_label",
            ]
            if col in valid.columns
        ]
        top_expensive = valid.sort_values("prediction_price_rub", ascending=False).head(10)[top_cols].copy()
        top_cheap = valid.sort_values("prediction_price_rub", ascending=True).head(10)[top_cols].copy()
        for frame in [top_expensive, top_cheap]:
            for col in ["prediction_price_rub", "prediction_price_sqm_rub"]:
                if col in frame.columns:
                    frame[col] = frame[col].map(lambda x: format_rub(float(x)) if not pd.isna(x) else "")

        top_col1, top_col2 = st.columns(2)
        with top_col1:
            st.caption("Самые дорогие объекты")
            st.dataframe(format_display_dataframe(top_expensive), width="stretch", hide_index=True)
        with top_col2:
            st.caption("Самые дешевые объекты")
            st.dataframe(format_display_dataframe(top_cheap), width="stretch", hide_index=True)

    if "prediction_status" in result.columns:
        similarity_problem = result["input_similarity_score"].lt(55) if "input_similarity_score" in result.columns else pd.Series(False, index=result.index)
        problems = result[result["prediction_status"].isin(["warning", "error"]) | similarity_problem].copy()
        if not problems.empty:
            st.subheader("Строки для ручной проверки")
            problem_cols = [
                col
                for col in ["prediction_status", "validation_notes", "input_similarity_score", "input_similarity_label", "region", "district", "rooms", "area_total"]
                if col in problems.columns
            ]
            st.dataframe(format_display_dataframe(problems[problem_cols].head(50)), width="stretch", hide_index=True)

    batch_report_html = build_batch_report_html(result, messages)
    st.download_button(
        "Скачать HTML-отчёт по пакетной оценке",
        data=batch_report_html.encode("utf-8"),
        file_name="portfolio_prediction_report.html",
        mime="text/html",
        width="stretch",
        disabled=valid_count == 0,
    )

    st.download_button(
        "Скачать CSV с прогнозами",
        data=result.to_csv(index=False).encode("utf-8-sig"),
        file_name="portfolio_predictions.csv",
        mime="text/csv",
        width="stretch",
        disabled=valid_count == 0,
    )


def show_methodology(df: pd.DataFrame, model_metrics: dict[str, Any], metadata: dict[str, Any]) -> None:
    render_section_title("О проекте и методология", "Что показывает приложение, какие модели сравнивались и как читать результат.")

    leaderboard = model_metrics["leaderboard"]
    baseline = model_metrics["baseline"]
    best = leaderboard.iloc[0]
    metrics = metadata.get("metrics", {})
    data_path = first_existing(DATA_PATHS)
    compared_models_count = len(leaderboard) + len(baseline)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Наблюдений в датасете", f"{len(df):,}".replace(",", " "))
    col2.metric("Моделей в сравнении", compared_models_count)
    col3.metric("Выбранная модель", str(best["model"]))
    col4.metric("Test MAPE", f"{best['test_mape_price']:.1%}")

    st.markdown(
        """
        **Задача.** Прогноз полной цены квартиры по характеристикам объявления:
        локация, район, метро, площадь, этаж, год постройки, тип здания, статус дома и признаки рынка.

        **Целевая переменная.** Модели обучались на `log(price)`, а в приложении прогноз переводится обратно в рубли через `exp`.

        **Разбиение данных.** Для этапа моделирования использовался сохраненный stratified split:
        train/test и CV-фолды сохраняют пропорции ключевых рыночных групп.
        Стратификация держит баланс по новостройкам/вторичке, Санкт-Петербургу/ЛО и первому этажу.

        **Метрики.** Основная метрика выбора модели — `cv_rmse_log`.
        Для интерпретации в рублях дополнительно показывается `MAPE` по цене.
        """
    )
    st.caption(
        "В сравнении учитываются 4 основные модели этапа 4, а также две модели Арины Щучкиной: "
        "наивная baseline-модель и регрессионная модель."
    )
    st.info(
        f"Источник данных приложения: `{data_path.name if data_path else 'не найден'}`. "
        "Это очищенная выгрузка объявлений, а не реестр фактических сделок, поэтому выводы описывают рынок предложения."
    )

    st.caption("Сводное сравнение моделей")
    display_cols = [
        "rank",
        "participant",
        "model",
        "role",
        "cv_rmse_log",
        "test_rmse_log",
        "test_mae_log",
        "test_r2_log",
        "test_mape_price",
    ]
    main_models = leaderboard.copy()
    main_models["role"] = "основная модель этапа 4"

    arina_models = baseline.copy()
    arina_models.insert(0, "rank", [f"B{i + 1}" for i in range(len(arina_models))])
    arina_models.insert(1, "participant", "Щучкина Арина")
    arina_models["role"] = arina_models["model"].map(
        {
            "MedianBaseline": "наивная baseline-модель",
            "LinearRegression": "регрессионная baseline-модель",
        }
    ).fillna("baseline-модель")

    methodology_models = pd.concat([main_models, arina_models], ignore_index=True, sort=False)
    available_cols = [col for col in display_cols if col in methodology_models.columns]
    methodology_models = methodology_models[available_cols].copy()
    if "rank" in methodology_models.columns:
        methodology_models["rank"] = methodology_models["rank"].astype(str)
    if "test_mape_price" in methodology_models.columns:
        methodology_models["test_mape_price"] = methodology_models["test_mape_price"].map(lambda x: f"{x:.1%}" if not pd.isna(x) else "н/д")
    for metric_col in ["cv_rmse_log", "test_rmse_log", "test_mae_log", "test_r2_log"]:
        if metric_col in methodology_models.columns:
            methodology_models[metric_col] = methodology_models[metric_col].map(lambda x: f"{float(x):.4f}" if not pd.isna(x) else "н/д")

    st.dataframe(methodology_models, width="stretch", hide_index=True)

    with st.expander("Ограничения модели", expanded=False):
        st.markdown(
            """
            - Модель обучена на исторических объявлениях Cian и наследует структуру этого датасета.
            - Прогноз хуже переносится на очень дорогие и редкие объекты, потому что таких наблюдений мало.
            - Приложение показывает ориентир, а не оценочный отчет: результат нужно проверять по рынку и сопоставимым объектам.
            - Геокодирование в приложении не используется как обязательный внешний сервис: координаты можно задать вручную.
            """
        )

    if metrics:
        st.caption("Метрики локально сохраненной модели")
        st.dataframe(
            pd.DataFrame(
                [
                    {"metric": "MAE log", "value": f"{metrics.get('mae_log', np.nan):.4f}"},
                    {"metric": "RMSE log", "value": f"{metrics.get('rmse_log', np.nan):.4f}"},
                    {"metric": "R² log", "value": f"{metrics.get('r2_log', np.nan):.4f}"},
                    {"metric": "MAPE price", "value": f"{metrics.get('mape_price', np.nan):.1%}"},
                    {"metric": "Median APE price", "value": f"{metrics.get('median_ape_price', np.nan):.1%}"},
                ]
            ),
            width="stretch",
            hide_index=True,
        )


def show_model_passport(df: pd.DataFrame, model: Any, metadata: dict[str, Any], model_metrics: dict[str, Any]) -> None:
    render_section_title("Паспорт модели", "Данные, признаки, диагностика и ограничения лучшей модели.")

    cat_features, rest_features, all_features = get_model_features(model)
    metrics = metadata.get("metrics", {})
    params = metadata.get("params", {})
    leaderboard = model_metrics["leaderboard"]
    best = leaderboard.iloc[0] if not leaderboard.empty else {}

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Модель", str(metadata.get("model_name", best.get("model", "XGBRegressor"))))
    col2.metric("Target", str(metadata.get("target", "log_price")))
    col3.metric("Признаков", len(all_features))
    col4.metric("Test MAPE", f"{metrics.get('mape_price', best.get('test_mape_price', np.nan)):.1%}")

    passport_tab1, passport_tab2, passport_tab3, passport_tab4 = st.tabs(
        ["Кратко", "Признаки", "Метрики", "Ограничения"]
    )

    with passport_tab1:
        train_shape = metadata.get("train_shape", ["н/д", "н/д"])
        test_shape = metadata.get("test_shape", ["н/д", "н/д"])
        summary = pd.DataFrame(
            [
                {"Поле": "Артефакт", "Значение": metadata.get("artifact", "roman_xgboost_pipeline.pkl")},
                {"Поле": "Источник", "Значение": metadata.get("source", "локальный artifact")},
                {"Поле": "Создан", "Значение": metadata.get("created_at", "н/д")},
                {"Поле": "Train shape", "Значение": f"{train_shape[0]} x {train_shape[1]}"},
                {"Поле": "Test shape", "Значение": f"{test_shape[0]} x {test_shape[1]}"},
                {"Поле": "Разбиение", "Значение": metadata.get("split_path", "split_data/split_data.pkl")},
                {"Поле": "Пайплайн", "Значение": "ColumnTransformer + OneHotEncoder + XGBRegressor"},
            ]
        )
        st.dataframe(stringify_dataframe(summary), width="stretch", hide_index=True)

        st.markdown(
            """
            Модель прогнозирует `log(price)`, после чего приложение переводит результат обратно в рубли.
            Категориальные признаки кодируются внутри сохраненного sklearn-пайплайна, поэтому форма приложения
            не должна вручную повторять one-hot-кодирование.
            """
        )

        if params:
            params_df = pd.DataFrame([{"Параметр": key, "Значение": value} for key, value in params.items()])
            st.caption("Основные параметры XGBoost")
            params_height = min(720, 38 + 35 * (len(params_df) + 1))
            st.dataframe(
                stringify_dataframe(params_df),
                width="stretch",
                height=params_height,
                hide_index=True,
            )

    with passport_tab2:
        feature_groups = pd.DataFrame(
            [
                {"Группа": "Категориальные", "Количество": len(cat_features), "Признаки": ", ".join(cat_features)},
                {"Группа": "Числовые и булевы", "Количество": len(rest_features), "Признаки": ", ".join(rest_features)},
            ]
        )
        st.dataframe(feature_groups, width="stretch", hide_index=True)

        feature_table = pd.DataFrame(
            {
                "feature": all_features,
                "type": ["categorical" if feature in cat_features else "numeric/bool" for feature in all_features],
            }
        )
        st.dataframe(feature_table, width="stretch", hide_index=True)

    with passport_tab3:
        metric_rows = [
            {
                "Метрика": "MAE log",
                "Значение": f"{metrics.get('mae_log', np.nan):.4f}",
                "Как читать": "Средняя абсолютная ошибка в логарифме цены.",
            },
            {
                "Метрика": "RMSE log",
                "Значение": f"{metrics.get('rmse_log', np.nan):.4f}",
                "Как читать": "Штрафует крупные ошибки сильнее MAE.",
            },
            {
                "Метрика": "R² log",
                "Значение": f"{metrics.get('r2_log', np.nan):.4f}",
                "Как читать": "Доля объясненной вариации в логарифме цены.",
            },
            {
                "Метрика": "MAPE price",
                "Значение": f"{metrics.get('mape_price', np.nan):.1%}",
                "Как читать": "Средняя относительная ошибка после перевода прогноза в рубли.",
            },
            {
                "Метрика": "Median APE price",
                "Значение": f"{metrics.get('median_ape_price', np.nan):.1%}",
                "Как читать": "Типичная относительная ошибка без сильного влияния выбросов.",
            },
        ]
        st.dataframe(pd.DataFrame(metric_rows), width="stretch", hide_index=True)

        display_cols = [
            "rank",
            "participant",
            "model",
            "cv_rmse_log",
            "test_rmse_log",
            "test_r2_log",
            "test_mape_price",
        ]
        available_cols = [col for col in display_cols if col in leaderboard.columns]
        leaderboard_display = leaderboard[available_cols].copy()
        if "test_mape_price" in leaderboard_display.columns:
            leaderboard_display["test_mape_price"] = leaderboard_display["test_mape_price"].map(lambda x: f"{x:.1%}")
        st.caption("Место модели в общем сравнении")
        st.dataframe(leaderboard_display, width="stretch", hide_index=True)

    with passport_tab4:
        st.markdown(
            """
            - Прогноз строится по объявлениям, а не по фактическим сделкам.
            - Самые редкие объекты требуют ручной проверки через блок близости к обучающим данным.
            - Сегментный интервал не является доверительным интервалом в строгом статистическом смысле; это эмпирический диапазон ошибок похожих объектов.
            - Новые станции метро, новые районы и признаки вне обучающего диапазона могут ухудшить переносимость модели.
            - Для защиты важно показывать не только точечный прогноз, но и похожие объекты, сегментный интервал и сравнение с рынком.
            """
        )

        if {"price", "area_total"}.issubset(df.columns):
            data_card = pd.DataFrame(
                [
                    {"Показатель": "Наблюдений в текущем датасете", "Значение": len(df)},
                    {"Показатель": "Медианная цена", "Значение": format_rub(float(df["price"].median()))},
                    {"Показатель": "Медианная площадь", "Значение": f"{float(df['area_total'].median()):.1f} м²"},
                ]
            )
            st.dataframe(stringify_dataframe(data_card), width="stretch", hide_index=True)


def main() -> None:
    df = load_data()
    model = load_model()
    metadata = load_metadata()
    model_metrics = load_model_metrics()
    cat_features, _, _ = get_model_features(model)
    categories = get_encoder_categories(model, cat_features)
    reference = make_reference(df, categories)

    presentation_mode = sidebar_presentation_mode()
    render_app_header(df, model_metrics, metadata, presentation_mode)
    page_options = page_options_for_mode(presentation_mode)
    page = render_page_navigation(page_options)

    if page == "Оценка квартиры":
        show_prediction(df, model, reference, metadata, presentation_mode=presentation_mode)
    elif page == "Рынок":
        show_overview(df, presentation_mode=presentation_mode)
    elif page == "Качество модели":
        show_models(df, model, metadata, model_metrics, presentation_mode=presentation_mode)
    elif page == "Пакетная оценка":
        show_batch_prediction(df, model, reference)
    elif page == "Паспорт модели":
        show_model_passport(df, model, metadata, model_metrics)
    elif page == "Методология":
        show_methodology(df, model_metrics, metadata)


if __name__ == "__main__":
    main()
