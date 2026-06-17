import streamlit as st
import joblib
import pandas as pd
import numpy as np
import math
import time
from streamlit_folium import st_folium
import folium
from geopy.geocoders import Nominatim


st.set_page_config(layout="centered", page_title="Недвижимость")

# -------------------- 1. Загрузка модели --------------------
@st.cache_resource
def load_model():
    try:
        pipeline = joblib.load('roman_xgboost_pipeline.pkl')
    except FileNotFoundError:
        st.error("Файл 'roman_xgboost_pipeline.pkl' не найден. Убедитесь, что он лежит в папке с приложением.")
        return None, None, None
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        return None, None, None

    # Извлекаем списки признаков из ColumnTransformer
    preprocessor = pipeline.named_steps['preprocessor']
    cat_features = []
    num_bool_features = []
    for name, trans, cols in preprocessor.transformers_:
        if name == 'cat':
            cat_features = list(cols)
        elif name == 'num_bool':
            num_bool_features = list(cols)
    all_features = cat_features + num_bool_features

    return pipeline, cat_features, all_features

pipeline, cat_features, all_features = load_model()
if pipeline is None:
    st.stop()

# -------------------- 2. Справочники районов и метро --------------------
districts_list = [
    'Василеостровский', 'Петроградский', 'Невский', 'Выборгский',
    'Адмиралтейский', 'Тосненский', 'Приморский', 'Фрунзенский', 'Всеволожский',
    'Центральный', 'Пушкинский', 'Московский', 'Красногвардейский',
    'Красносельский', 'Калининский', 'Ломоносовский', 'Колпинский',
    'Петродворцовый', 'Курортный', 'Кировский', 'Гатчинский'
]

okrug_to_district = {
    "Коломна": "Адмиралтейский",
    "Сенной округ": "Адмиралтейский",
    "Адмиралтейский округ": "Адмиралтейский",
    "Семеновский": "Адмиралтейский",
    "Измайловское": "Адмиралтейский",
    "Екатерингофский": "Адмиралтейский",
    "№ 7": "Василеостровский",
    "Васильевский": "Василеостровский",
    "Гавань": "Василеостровский",
    "Морской": "Василеостровский",
    "Остров Декабристов": "Василеостровский",
    "Сампсониевское": "Выборгский",
    "Светлановское": "Выборгский",
    "Сосновское": "Выборгский",
    "№ 15": "Выборгский",
    "Сергиевское": "Выборгский",
    "Шувалово-Озерки": "Выборгский",
    "Левашово": "Выборгский",
    "Парголово": "Выборгский",
    "Гражданка": "Калининский",
    "Академическое": "Калининский",
    "Финляндский округ": "Калининский",
    "№ 21": "Калининский",
    "Пискаревка": "Калининский",
    "Северный": "Калининский",
    "Прометей": "Калининский",
    "Княжево": "Кировский",
    "Ульянка": "Кировский",
    "Дачное": "Кировский",
    "Автово": "Кировский",
    "Нарвский округ": "Кировский",
    "Красненькая речка": "Кировский",
    "Морские ворота": "Кировский",
    "Полюстрово": "Красногвардейский",
    "Большая Охта": "Красногвардейский",
    "Малая Охта": "Красногвардейский",
    "Пороховые": "Красногвардейский",
    "Ржевка": "Красногвардейский",
    "Юго-Запад": "Красносельский",
    "Южно-Приморский": "Красносельский",
    "Сосновая Поляна": "Красносельский",
    "УРИЦК": "Красносельский",
    "Константиновское": "Красносельский",
    "Горелово": "Красносельский",
    "Красное Село": "Красносельский",
    "Колпино": "Колпинский",
    "Металлострой": "Колпинский",
    "Петро-Славянка": "Колпинский",
    "Понтонный": "Колпинский",
    "Усть-Ижора": "Колпинский",
    "Саперный": "Колпинский",
    "Кронштадт": "Кронштадтский",
    "Сестрорецк": "Курортный",
    "Зеленогорск": "Курортный",
    "Песочный": "Курортный",
    "Белоостров": "Курортный",
    "Комарово": "Курортный",
    "Молодёжное": "Курортный",
    "Репино": "Курортный",
    "Серово": "Курортный",
    "Смолячково": "Курортный",
    "Солнечное": "Курортный",
    "Ушково": "Курортный",
    "Московская застава": "Московский",
    "Гагаринское": "Московский",
    "Новоизмайловское": "Московский",
    "Пулковский меридиан": "Московский",
    "Звездное": "Московский",
    "Невская застава": "Невский",
    "Ивановский": "Невский",
    "Обуховский": "Невский",
    "Рыбацкое": "Невский",
    "Народный": "Невский",
    "№ 54": "Невский",
    "Невский округ": "Невский",
    "Оккервиль": "Невский",
    "Правобережный": "Невский",
    "Ломоносов": "Петродворцовый",
    "Петергоф": "Петродворцовый",
    "Стрельна": "Петродворцовый",
    "Пушкин": "Пушкинский",
    "Павловск": "Пушкинский",
    "Шушары": "Пушкинский",
    "Александровская": "Пушкинский",
    "Тярлево": "Пушкинский",
    "Введенский": "Петроградский",
    "Кронверкское": "Петроградский",
    "Посадский": "Петроградский",
    "Аптекарский остров": "Петроградский",
    "Петровский": "Петроградский",
    "Чкаловское": "Петроградский",
    "Лахта-Ольгино": "Приморский",
    "№ 65": "Приморский",
    "Ланское": "Приморский",
    "Комендантский аэродром": "Приморский",
    "Озеро Долгое": "Приморский",
    "Юнтолово": "Приморский",
    "Коломяги": "Приморский",
    "Лисий Нос (поселок)": "Приморский",
    "Волковское": "Фрунзенский",
    "№ 72": "Фрунзенский",
    "Купчино": "Фрунзенский",
    "Георгиевский": "Фрунзенский",
    "№ 75": "Фрунзенский",
    "Балканский": "Фрунзенский",
    "Дворцовый округ": "Центральный",
    "№ 78": "Центральный",
    "Литейный округ": "Центральный",
    "Смольнинское": "Центральный",
    "Лиговка-Ямская": "Центральный",
    "Владимирский округ": "Центральный",
    "Пушкинский район": "Пушкинский",
    "Шушары (поселок)": "Пушкинский",
    "Всеволожский район": "Всеволожский",
    "Всеволожский": "Всеволожский",
    "Гатчинский муниципальный округ": "Гатчинский",
    "Гатчинский район": "Гатчинский",
    "Гатчинский": "Гатчинский",
    "Ломоносовский район": "Ломоносовский",
    "Ломоносовский": "Ломоносовский",
    "Тосненский район": "Тосненский",
    "Тосненский": "Тосненский",
    "Кировский район": "Кировский",
    "Кировский": "Кировский"
}

districts_metro_map = {
    "Адмиралтейский": ["Адмиралтейская", "Садовая", "Балтийская"],
    "Василеостровский": ["Василеостровская", "Приморская", "Горный институт"],
    "Выборгский": ["Выборгская", "Лесная", "Удельная", "Озерки", "Проспект Просвещения", "Парнас"],
    "Петроградский": ["Горьковская", "Петроградская", "Чкаловская", "Спортивная", "Крестовский остров", "Зенит"],
    "Приморский": ["Пионерская", "Чёрная речка", "Старая Деревня", "Комендантский проспект", "Беговая"],
    "Калининский": ["Академическая", "Гражданский проспект", "Политехническая", "Площадь Мужества", "Площадь Ленина"],
    "Центральный": ["Владимирская", "Площадь Восстания", "Чернышевская", "Гостиный двор", "Адмиралтейская", "Лиговский проспект", "Площадь Александра Невского"],
    "Красногвардейский": ["Ладожская", "Новочеркасская"],
    "Красносельский": ["Юго-Западная"],
    "Кировский": ["Кировский завод", "Нарвская", "Автово", "Ленинский проспект", "Проспект Ветеранов"],
    "Московский": ["Московские ворота", "Парк Победы", "Электросила", "Фрунзенская", "Московская", "Звёздная"],
    "Невский": ["Ломоносовская", "Обухово", "Проспект Большевиков", "Елизаровская", "Рыбацкое", "Пролетарская", "Улица Дыбенко"],
    "Пушкинский": ["Шушары"],
    "Фрунзенский": ["Купчино", "Дунайская", "Проспект Славы", "Международная", "Бухарестская", "Волковская", "Обводный канал"],
    "Колпинский": [],
    "Петродворцовый": []
}

# -------------------- 3. Геокодер --------------------
@st.cache_resource
def get_geocoder():
    try:
        return Nominatim(user_agent="real_estate_app", timeout=5)
    except:
        return None

geolocator = get_geocoder()

def match_district_by_okrug(district_raw):
    """Определяет район по названию муниципального округа."""
    if not district_raw:
        return "Центральный"
    raw = district_raw.strip()
    raw = raw.replace("округ", "").replace("(поселок)", "").replace("(город)", "").strip()
    if raw in okrug_to_district:
        return okrug_to_district[raw]
    raw_lower = raw.lower()
    for okrug, district in okrug_to_district.items():
        okrug_lower = okrug.lower()
        if raw_lower == okrug_lower or okrug_lower in raw_lower or raw_lower in okrug_lower:
            return district
    return "Центральный"

match_district = match_district_by_okrug

def reverse_geocode(lat, lon):
    if geolocator is None:
        return {"region": "Санкт-Петербург", "district": "Центральный", "full_address": ""}
    try:
        time.sleep(1)
        location = geolocator.reverse(f"{lat}, {lon}", language="ru")
        if location and location.raw:
            addr = location.raw.get("address", {})
            region = "Санкт-Петербург" if "Санкт-Петербург" in addr.get('state', '') else addr.get('state', 'Санкт-Петербург')
            district_raw = (
                addr.get('city_district') or
                addr.get('county') or
                addr.get('suburb') or
                addr.get('village') or
                addr.get('town') or
                addr.get('municipality') or
                addr.get('hamlet') or
                'Центральный'
            )
            
            if district_raw == 'Центральный':
                full_addr = location.address
                for district_name in ["Всеволожский", "Гатчинский", "Ломоносовский", "Тосненский", "Кировский"]:
                    if district_name in full_addr:
                        district_raw = district_name
                        break
            district = match_district(district_raw)
            full = location.address
            return {"region": region, "district": district, "full_address": full}
    except:
        pass
    return {"region": "Санкт-Петербург", "district": "Центральный", "full_address": ""}

# -------------------- 4. Инициализация состояния --------------------
if "click_lat" not in st.session_state:
    st.session_state.click_lat = 59.9343
if "click_lng" not in st.session_state:
    st.session_state.click_lng = 30.3351
if "region" not in st.session_state:
    st.session_state.region = "Санкт-Петербург"
if "district" not in st.session_state:
    st.session_state.district = "Центральный"
if "full_address" not in st.session_state:
    st.session_state.full_address = ""

st.title("Предсказание цены недвижимости")

# -------------------- 5. Карта --------------------
with st.expander("Кликните на карте, чтобы выбрать точку", expanded=True):
    m = folium.Map(location=[st.session_state.click_lat, st.session_state.click_lng], zoom_start=12)
    folium.Marker(location=[st.session_state.click_lat, st.session_state.click_lng], popup="Выбранная точка", icon=folium.Icon(color="red")).add_to(m)
    map_data = st_folium(m, width=700, height=450)
    if map_data and map_data.get("last_clicked"):
        new_lat = map_data["last_clicked"]["lat"]
        new_lng = map_data["last_clicked"]["lng"]
        if new_lat != st.session_state.click_lat or new_lng != st.session_state.click_lng:
            st.session_state.click_lat = new_lat
            st.session_state.click_lng = new_lng
            with st.spinner("Определяем адрес..."):
                info = reverse_geocode(new_lat, new_lng)
            st.session_state.region = info["region"]
            st.session_state.district = info["district"]
            st.session_state.full_address = info["full_address"]
            st.rerun()

if st.session_state.full_address:
    st.info(f"📍 Определённый адрес: {st.session_state.full_address}")
    st.info(f"Определённый регион: {st.session_state.region}")
    st.info(f"Определённый район: {st.session_state.district}")
else:
    st.warning("Кликните на карту, чтобы определить адрес.")

col_region, col_district = st.columns(2)
with col_region:
    manual_region = st.selectbox("Регион", ["Санкт-Петербург", "Ленинградская область"], 
                                 index=0 if st.session_state.region == "Санкт-Петербург" else 1)
with col_district:
    manual_district = st.selectbox("Район", districts_list,
                                   index=districts_list.index(st.session_state.district) if st.session_state.district in districts_list else 0)

# -------------------- 6. Транспорт --------------------
st.subheader("Транспорт")
col5, col6 = st.columns(2)
with col5:
    metro_options = districts_metro_map.get(manual_district, [])
    if not metro_options:
        nearest_metro = st.text_input("Ближайшее метро", value="Невский проспект")
    else:
        nearest_metro = st.selectbox("Ближайшее метро", metro_options, index=0)
    metro_travel_time = st.number_input("Время до метро (мин)", min_value=0, step=1, value=10)
with col6:
    metro_travel_type = st.selectbox("Способ передвижения", ("walk", "transport"))
    center_travel_km = st.number_input("Расстояние до центра (км)", min_value=0, step=1, value=5)

# -------------------- 7. Характеристики объекта --------------------
st.subheader("Введите характеристики")
current_year = 2026

col1, col2 = st.columns(2)
with col1:
    area_total = st.number_input("Общая площадь м²", min_value=0, step=1, value=50)
    area_living = st.number_input("Жилая площадь м²", min_value=0, step=1, value=35)
    area_kitchen = st.number_input("Площадь кухни м²", min_value=0, step=1, value=8)
    rooms = st.number_input("Количество комнат (0 студия и 0 свободная планировка)", min_value=0, max_value=5, step=1, value=2)
with col2:
    build_year = st.number_input("Год постройки", min_value=1900, max_value=current_year, step=1, value=2000)
    total_floors = st.number_input("Всего этажей", min_value=1, step=1, value=9)
    current_floor = st.number_input("Этаж квартиры", min_value=1, step=1, value=3)

st.subheader("Дополнительно")
col3, col4 = st.columns(2)
with col3:
    is_new_building = st.selectbox("Новостройка?", ("Нет", "Да"))
    is_studio = st.selectbox("Студия?", ("Нет", "Да"))
with col4:
    type_building = st.selectbox("Тип здания", [
        "Панельный", "Кирпичный", "Монолитный", "Блочный", "Старый фонд", "Сталинский",
        "Монолитно-кирпичный", "Газобетонный блок", "none_type"
    ])
    is_completed = st.selectbox("Дом сдан?", ("Да", "Нет"))

# -------------------- 8. Вычисление автоматических признаков --------------------
lat = st.session_state.click_lat
lng = st.session_state.click_lng

age = current_year - build_year
log_area = math.log(area_total) if area_total > 0 else 0
prop_floor = current_floor / total_floors if total_floors > 0 else 0
area_per_room = area_total / (rooms + 1)
build_decade = (build_year // 10) * 10
is_first_floor = 1 if current_floor == 1 else 0
is_last_floor = 1 if current_floor == total_floors else 0

metro_lat = 59.93
metro_lng = 30.33

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))

metro_travel_km = haversine(lat, lng, metro_lat, metro_lng)

# -------------------- 9. Формирование словаря признаков --------------------
user_dict = {
    "district": manual_district,
    "lat": lat,
    "lng": lng,
    "nearest_metro": nearest_metro,
    "metro_travel_time": metro_travel_time,
    "metro_travel_type": metro_travel_type,
    "status_home": "Сдан" if is_completed == "Да" else "Не сдан",
    "is_new_building": 1 if is_new_building == "Да" else 0,
    "build_year": build_year,
    "area_total": area_total,
    "area_living": area_living,
    "area_kitchen": area_kitchen,
    "rooms": rooms,
    "region": manual_region,
    "current_floor": current_floor,
    "total_floors": total_floors,
    "type_building": type_building,
    "is_studio": 1 if is_studio == "Да" else 0,
    "is_completed": 1 if is_completed == "Да" else 0,
    "is_spb": 1 if manual_region == "Санкт-Петербург" else 0,
    "is_first_floor": is_first_floor,
    "is_last_floor": is_last_floor,
    "prop_floor": prop_floor,
    "age": age,
    "metro_lat": metro_lat,
    "metro_lng": metro_lng,
    "metro_travel_km": metro_travel_km,
    "center_travel_km": center_travel_km,
    "build_decade": build_decade,
    "log_area": log_area,
    "area_per_room": area_per_room,
}

# Оставляем только признаки, которые ожидает модель
full_input = {f: user_dict.get(f, 0) for f in all_features}

# -------------------- 10. Отладочная информация --------------------
with st.expander("Технические признаки", expanded=False):
    st.write("### Все признаки, которые будут отправлены в модель")
    debug_df = pd.DataFrame(list(full_input.items()), columns=["Признак", "Значение"])
    st.dataframe(debug_df, use_container_width=True)
    st.write("**Категориальные признаки (преобразуются в строки):**", cat_features)
    st.caption("Здесь показаны значения, которые будут переданы в модель.")

# -------------------- 11. Предсказание --------------------
if st.button("Предсказать цену"):
    df = pd.DataFrame([full_input])
    # Приводим категориальные признаки к строке, остальные – к числам
    for c in cat_features:
        df[c] = df[c].astype(str)
    for col in df.columns:
        if col not in cat_features:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    try:
        log_price = pipeline.predict(df)[0]
        price_rub = np.exp(log_price)
        st.success(f"Предсказанная цена: {price_rub:,.0f} руб.")
    except Exception as e:
        st.error(f"Ошибка предсказания: {e}")

# -------------------- 12. Анализ модели --------------------
