import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


@st.cache_data
def load_data():
    return pd.read_csv("cian_processed.csv")  


def run_eda():
    st.header("EDA — Анализ данных")

    df = load_data()

    #  1. Общая инфа 
    st.subheader("Общая информация")
    st.write(df.head())
    st.write(df.describe())

    #  2. Пропуски 
    st.subheader("Пропуски")
    nulls = df.isnull().sum().sort_values(ascending=False)
    st.dataframe(nulls)

    # 3. Фичи 
    st.subheader("Генерация признаков")

    if "build_year" in df.columns:
        df["age"] = 2026 - df["build_year"]

    if "area_total" in df.columns:
        df["log_area"] = np.log(df["area_total"] + 1)

    if "rooms" in df.columns:
        df["area_per_room"] = df["area_total"] / (df["rooms"] + 1)

    if "price" in df.columns:
        df["log_price"] = np.log(df["price"] + 1)

    st.write(df.head())

    #  4. Распределения 
    st.subheader("Распределения")

    if "price" in df.columns:
        fig1, ax1 = plt.subplots()
        sns.histplot(df["price"], bins=50, ax=ax1)
        ax1.set_title("Распределение цены")
        st.pyplot(fig1)

    if "log_price" in df.columns:
        fig2, ax2 = plt.subplots()
        sns.histplot(df["log_price"], bins=50, ax=ax2)
        ax2.set_title("Логарифм цены")
        st.pyplot(fig2)

    #  5. Корреляция 
    st.subheader("Корреляция")

    numeric_df = df.select_dtypes(include=[np.number])

    fig3, ax3 = plt.subplots(figsize=(12, 8))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", ax=ax3)
    st.pyplot(fig3)

    # 6. Зависимости 
    st.subheader("Зависимости")

    if "area_total" in df.columns and "price" in df.columns:
        fig4, ax4 = plt.subplots()
        sns.scatterplot(x=df["area_total"], y=df["price"], ax=ax4)
        ax4.set_title("Цена vs Площадь")
        st.pyplot(fig4)

    if "rooms" in df.columns and "price" in df.columns:
        fig5, ax5 = plt.subplots()
        sns.boxplot(x=df["rooms"], y=df["price"], ax=ax5)
        ax5.set_title("Цена по комнатам")
        st.pyplot(fig5)

    # 7. Районы 
    if "district" in df.columns and "price" in df.columns:
        st.subheader("Средняя цена по районам")

        district_price = (
            df.groupby("district")["price"]
            .mean()
            .sort_values(ascending=False)
        )

        st.dataframe(district_price)

    #  8. Выбросы 
    st.subheader("Выбросы")

    if "price" in df.columns:
        fig6, ax6 = plt.subplots()
        sns.boxplot(x=df["price"], ax=ax6)
        st.pyplot(fig6)

    # 9. Фильтры 
    st.subheader("Интерактивный анализ")

    if "district" in df.columns:
        selected_district = st.selectbox(
            "Выбери район", ["Все"] + list(df["district"].dropna().unique())
        )

        if selected_district != "Все":
            filtered_df = df[df["district"] == selected_district]
        else:
            filtered_df = df

        st.write("Количество объектов:", len(filtered_df))

        if "price" in filtered_df.columns:
            fig7, ax7 = plt.subplots()
            sns.histplot(filtered_df["price"], bins=50, ax=ax7)
            ax7.set_title("Цена (фильтр)")
            st.pyplot(fig7)