import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="Dashboard Prediksi Diabetes", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv('diabetes.csv', sep=';')
    df.columns = df.columns.str.strip()
    cols_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols_zero] = df[cols_zero].replace(0, np.nan)
    imputer = SimpleImputer(strategy='median')
    df[cols_zero] = imputer.fit_transform(df[cols_zero])
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df_clean = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
    return df_clean

df = load_data()
X = df.drop(columns='Outcome')
y = df['Outcome']
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

st.sidebar.title("Navigasi")
menu = st.sidebar.radio("Pilih Halaman", ["Overview", "K-Means Clustering", "Logistic Regression"])

if menu == "Overview":
    st.title("📊 Dashboard Analisis Prediksi Diabetes")
    st.markdown("Dataset: Pima Indians Diabetes Dataset")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Data", df.shape[0])
    col2.metric("Tidak Diabetes", int((df['Outcome']==0).sum()))
    col3.metric("Diabetes", int((df['Outcome']==1).sum()))

    st.subheader("Preview Data")
    st.dataframe(df, use_container_width=True)

    st.subheader("Statistik Deskriptif")
    st.dataframe(df.describe(), use_container_width=True)

    st.subheader("Distribusi Outcome")
    fig, ax = plt.subplots()
    sns.countplot(x='Outcome', data=df, palette='Set2', ax=ax)
    ax.set_xticklabels(['Tidak Diabetes', 'Diabetes'])
    st.pyplot(fig)

    st.subheader("Heatmap Korelasi")
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    st.pyplot(fig)

elif menu == "K-Means Clustering":
    st.title("🔵 K-Means Clustering")

    k = st.slider("Pilih Jumlah Cluster (k)", min_value=2, max_value=6, value=3)
    kmeans = KMeans(n_clusters=k, random_state=42)
    df_cluster = df.copy()
    df_cluster['Cluster'] = kmeans.fit_predict(X_scaled)

    st.subheader(f"Hasil Clustering dengan k={k}")
    st.dataframe(df_cluster['Cluster'].value_counts().rename("Jumlah Data"), use_container_width=True)

    st.subheader("Rata-rata Fitur per Cluster")
    st.dataframe(df_cluster.groupby('Cluster').mean().T, use_container_width=True)

    st.subheader("Visualisasi Cluster (Glucose vs BMI)")
    fig, ax = plt.subplots(figsize=(8,6))
    sns.scatterplot(x=X_scaled['Glucose'], y=X_scaled['BMI'],
                    hue=df_cluster['Cluster'], palette='Set1', ax=ax)
    ax.set_title('K-Means Clustering')
    st.pyplot(fig)

    st.subheader("Elbow Method")
    inertia = []
    for i in range(1, 11):
        km = KMeans(n_clusters=i, random_state=42)
        km.fit(X_scaled)
        inertia.append(km.inertia_)
    fig, ax = plt.subplots()
    ax.plot(range(1,11), inertia, 'bo-')
    ax.set_xlabel('Jumlah Cluster')
    ax.set_ylabel('Inertia')
    ax.set_title('Elbow Method')
    st.pyplot(fig)

elif menu == "Logistic Regression":
    st.title("📈 Logistic Regression")

    test_size = st.slider("Test Size (%)", min_value=10, max_value=40, value=20) / 100
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)

    logreg = LogisticRegression(random_state=42)
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
    col2.metric("Precision", f"{precision_score(y_test, y_pred):.2f}")
    col3.metric("Recall", f"{recall_score(y_test, y_pred):.2f}")
    col4.metric("F1-Score", f"{f1_score(y_test, y_pred):.2f}")

    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
                xticklabels=['Tidak Diabetes', 'Diabetes'],
                yticklabels=['Tidak Diabetes', 'Diabetes'], ax=ax)
    ax.set_ylabel('Aktual')
    ax.set_xlabel('Prediksi')
    st.pyplot(fig)

    st.subheader("Tabel Hasil Prediksi (Data Uji)")
    hasil_prediksi = X_test.copy()
    hasil_prediksi['Outcome Aktual'] = y_test.values
    hasil_prediksi['Outcome Prediksi'] = y_pred
    hasil_prediksi['Status'] = np.where(
        hasil_prediksi['Outcome Aktual'] == hasil_prediksi['Outcome Prediksi'],
        'Benar', 'Salah'
    )
    st.dataframe(hasil_prediksi, use_container_width=True)
    
    st.subheader("🔍 Cek Risiko Diabetes (Input Manual)")
    st.markdown("Masukkan data kesehatan pasien untuk memprediksi risiko diabetes:")

    col1, col2 = st.columns(2)
    with col1:
        in_preg = st.number_input("Pregnancies", min_value=0, max_value=20, value=2)
        in_glucose = st.number_input("Glucose", min_value=0, max_value=250, value=120)
        in_bp = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
        in_skin = st.number_input("Skin Thickness", min_value=0, max_value=100, value=25)
    with col2:
        in_insulin = st.number_input("Insulin", min_value=0, max_value=900, value=100)
        in_bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=28.0)
        in_dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.4)
        in_age = st.number_input("Age", min_value=1, max_value=100, value=30)

    if st.button("Prediksi"):
        input_data = pd.DataFrame([[in_preg, in_glucose, in_bp, in_skin, in_insulin, in_bmi, in_dpf, in_age]],
                                   columns=X.columns)
        input_scaled = scaler.transform(input_data)
        pred = logreg.predict(input_scaled)[0]
        prob = logreg.predict_proba(input_scaled)[0][1]

        if pred == 1:
            st.error(f"⚠️ Hasil: Berisiko Diabetes (probabilitas {prob:.2%})")
        else:
            st.success(f"✅ Hasil: Tidak Berisiko Diabetes (probabilitas {prob:.2%})")