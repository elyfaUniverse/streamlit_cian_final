# Streamlit-приложение для прогноза цены квартиры

Эта папка самодостаточна: в ней уже лежат приложение, модель, данные, метрики и зависимости. Ее можно передать другому человеку или загрузить в отдельный репозиторий для деплоя.

## Состав папки

- `app.py` - Streamlit-приложение.
- `requirements.txt` - Python-зависимости.
- `check_bundle.py` - проверка, что данные, модель и зависимости доступны.
- `run_local.sh` - запуск на macOS/Linux одной командой.
- `run_local.bat` - запуск на Windows одной командой.
- `eda_result.csv` - подготовленный датасет объявлений.
- `roman_xgboost_pipeline.pkl` - сохраненная модель `XGBRegressor` внутри sklearn pipeline.
- `roman_xgboost_pipeline_metadata.json` - metadata модели, признаки, параметры и метрики.
- `model_metrics.json` - leaderboard, baseline, сегменты ошибок и SHAP-факторы.
- `split_data.pkl` - train/test split для holdout-диагностики и интервалов.

## Локальный запуск

Нужен Python 3.11+.

### Самый простой способ

На macOS/Linux:

```bash
cd real-estate-price-prediction-cian-streamlit-deploy
./run_local.sh
```

На Windows:

```bat
cd real-estate-price-prediction-cian-streamlit-deploy
run_local.bat
```

Скрипт сам создаст `.venv`, установит зависимости, проверит комплектность папки и запустит Streamlit.

### Ручной способ

1. Перейдите в папку:

```bash
cd real-estate-price-prediction-cian-streamlit-deploy
```

2. Создайте и активируйте виртуальное окружение:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

На Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Установите зависимости:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

4. Проверьте комплектность папки:

```bash
python check_bundle.py
```

Ожидаемый результат:

```text
Bundle check passed.
```

5. Запустите приложение:

```bash
python -m streamlit run app.py
```

После запуска Streamlit покажет локальный адрес вида:

```text
http://localhost:8501
```

## Быстрый деплой

Для Streamlit Community Cloud или похожего сервиса загрузите содержимое этой папки в репозиторий и укажите:

- main file: `app.py`
- dependency file: `requirements.txt`

Все файлы данных и модели должны лежать рядом с `app.py`, как в этой папке.

Приложение ищет файлы именно рядом с `app.py`, поэтому папку можно переносить целиком на другую машину без сохранения структуры исходного проекта.

## Проверка перед деплоем

Перед загрузкой можно выполнить:

```bash
python -m py_compile app.py
python check_bundle.py
python -m streamlit run app.py
```

Если приложение открылось локально и видны разделы `Оценка`, `Рынок`, `Модели`, `Пакет`, `Паспорт`, `Методология`, сборка готова к деплою.

## Важно

`roman_xgboost_pipeline.pkl` сохранен как pickle/sklearn pipeline, поэтому версии библиотек в `requirements.txt` зафиксированы. Не обновляйте `scikit-learn`, `xgboost`, `numpy` и `pandas` без повторной проверки загрузки модели.
