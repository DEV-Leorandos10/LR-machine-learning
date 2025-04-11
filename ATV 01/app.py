import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

st.title("Projeto de Machine Learning com Geração Automática de Dados")

# Escolha de tarefa
task = st.radio("Escolha a tarefa:", ["Classificação", "Clusterização"])

# Geração de dados
if st.button("Gerar Dados Artificialmente"):
    if task == "Classificação":
        X, y = make_classification(n_samples=500, n_features=4, n_classes=2)
        df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
        df['target'] = y
    else:
        X, y = make_blobs(n_samples=300, centers=3, n_features=2)
        df = pd.DataFrame(X, columns=['x', 'y'])
        df['cluster'] = y
    st.write("Dados Gerados:")
    st.dataframe(df)

    if task == "Classificação":
        # Classificação com RandomForest
        X_train, X_test, y_train, y_test = train_test_split(df.drop('target', axis=1), df['target'], test_size=0.3)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        st.success(f"Acurácia do Modelo: {acc:.2f}")
    else:
        # Clusterização com KMeans
        model = KMeans(n_clusters=3)
        df['cluster_pred'] = model.fit_predict(df[['x', 'y']])
        st.write("Resultado da Clusterização")
        fig, ax = plt.subplots()
        for cluster in df['cluster_pred'].unique():
            cluster_data = df[df['cluster_pred'] == cluster]
            ax.scatter(cluster_data['x'], cluster_data['y'], label=f"Cluster {cluster}")
        ax.legend()
        st.pyplot(fig)