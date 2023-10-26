import streamlit as st
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt


# Library
from sklearn.metrics import make_scorer, accuracy_score,precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score ,precision_score,recall_score,f1_score
from sklearn.model_selection import KFold,train_test_split,cross_val_score

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
# import library yang di butuhkan
import requests
from bs4 import BeautifulSoup
import csv

# recruitments punctuation
# import string

# Stopwords 
import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import sent_tokenize, word_tokenize
# from nltk.corpus import stopwords
nltk.download('punkt')
# Download kamus stop words
nltk.download('stopwords')

# from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Library Ekstaksi Fitur
# from sklearn.preprocessing import MultiLabelBinarizer
# from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
# LDA
from sklearn.decomposition import LatentDirichletAllocation
import os

from sklearn import metrics
import pickle


st.title("Website Applikasi Pengolahan dan Penambangan Web")
st.write("""
# Web Apps - Crowling Dataset
Applikasi Berbasis Web untuk melakukan **Sentiment Analysis**,
Jadi pada web applikasi ini akan bisa membantu anda untuk melakukan sentiment analysis mulai dari 
mengambil data (crowling dataset) dari sebuah website dan melakukan preprocessing dari data yang 
sudah diambil kemudian dapat melakukan klasifikasi serta dapat melihat akurasi dari model yang diinginkan.
### Menu yang disediakan dapat di lihat di bawah ini :
""")

# inisialisasi data 
# data = pd.read_csv("ecoli.csv")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Crowling Data", "Preprocessing Data", "Modeling", "Reduksi Dimensi", "implmentasi"])

with tab1:

    st.subheader("Deskripsi")
    st.write("""Crowling Data adalah proses automatis untuk mengumpulkan dan mengindeks data dari 
    berbagai sumber seperti situs web, database, atau dokumen. Proses ini menggunakan software atau 
    aplikasi khusus yang disebut "crawler" untuk mengakses sumber data dan mengambil informasi yang 
    dibutuhkan. Data yang dikumpulkan melalui crawling kemudian dapat diproses dan digunakan untuk 
    berbagai tujuan, seperti analisis data, penelitian, atau pengembangan sistem informasi. Tujuanya
    untuk mengumpulkan data dari berbagai sumber dan mengindeksnya sehingga mudah untuk diakses 
    dan dianalisis.
    """)

    st.write("""
    ### Want to learn more?
    - Dataset (studi kasus) [pta.trunojoyo](https://pta.trunojoyo.ac.id/)
    - Github Account [github.com](https://github.com/AliGhufron-28/datamaining)
    """)

    st.subheader("Crowling Data")
    fakultas = 4
    page = 1
    url = st.text_input("Masukkan Link website")

    hasil = st.button("submit")

    if hasil:
        # Membuka file CSV untuk menulis hasil scraping
        with open('hasil_scraping.csv', 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Judul', 'Penulis', 'Dosen Pembimbing I', 'Dosen Pembimbing II', 'Abstrak']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Menulis header ke dalam file CSV
            writer.writeheader()

            while page <= 10:
                url = url.format(fakultas, page)
                req = requests.get(url)
                soup = BeautifulSoup(req.text, 'html.parser')
                items = soup.find_all('li', attrs={'data-id': 'id-1'})

                if not items:
                    break

                for it in items:
                    data = {}
                    title = it.find('a', class_='title').text
                    data['Judul'] = title
                    div_elements = it.find_all('div', style='padding:2px 2px 2px 2px;')
                    for div in div_elements:
                        span = div.find('span')
                        if span:
                            span_text = span.get_text()
                            key, value = span_text.split(':', 1)
                            data[key.strip()] = value.strip()

                    # Mengambil link abstrak dari elemen dengan kelas 'gray button'
                    abstrak_button = it.find('a', class_='gray button')
                    if abstrak_button:
                        abstrak_link = abstrak_button['href']
                        abstrak_req = requests.get(abstrak_link)
                        abstrak_soup = BeautifulSoup(abstrak_req.text, 'html.parser')
                        abstrak = abstrak_soup.find('p', align='justify')
                        if abstrak:
                            abstrak_text = abstrak.get_text(strip=True)
                            data['Abstrak'] = abstrak_text
                        else:
                            data['Abstrak'] = "Abstrak tidak ditemukan"

                    # Menulis data ke dalam file CSV
                    writer.writerow(data)

                page += 1

        print("Scraping selesai")

with tab2:
    st.subheader("Data yang berhasil di Crowling kemudian di kasih label")
    st.subheader("Data Asli")
    path = ("pta-infor-abstrak.csv")
    data = pd.read_csv(path, delimiter=';')
    st.write(data)

    proc = st.checkbox("Normalisasi")
    if proc:
        st.subheader("Hasil Data Abstrak dan menghilangkan data yang NaN")
        # hapus data NaN
        data.dropna(subset=['Abstrak'], inplace=True)
        # Cek kembali nilai NaN
        data['Abstrak'].isna().sum()
        st.write(data['Abstrak'])

        st.subheader("Data Abstrak Punctuation dan tanpa angka")
        dt_punctuation = pd.read_csv('punctuation.csv')
        st.write(dt_punctuation)


        st.subheader("Data Hasil Stopwords")
        st.write("Stopwords digunakan untuk menghilangkan kata umum yang sering muncul dalam teks seperti: di, dan, atau, dari, ke, saya.")
        dt_stopwords = pd.read_csv('stopwords_abstrak.csv')
        st.write(dt_stopwords)

        st.subheader("Data hasil Tokenisasi")
        st.write("Tokenizing yaitu proses memecah teks atau dokumen menjadi potongan-potongan yang lebih kecil, yang disebut token.")
        dt_tokenizing_abstrak = pd.read_csv('tokenizing_abstrak.csv')
        st.write(dt_tokenizing_abstrak)

        st.subheader("Data Hasil Stemming")
        st.write("Stemming merupakan proses normalisasi data teks menjadikan kata dasar")
        dt_stemming = pd.read_csv("stemming_abstrak.csv")
        st.write(dt_stemming)
        

        st.subheader("Ekstraksi Fitur OneHotEncoder")
        df_onehotencoding = pd.read_csv('one_hot_encoding.csv')
        st.write(df_onehotencoding)

        st.subheader("Ekstraksi Fitur Term Frequency")
        df_term_frequency = pd.read_csv('Term_Frequency.csv')
        st.write(df_term_frequency)

        st.subheader("Ekstraksi Fitur Log Frekuency")
        df_log_frequency = pd.read_csv('log_tf.csv')
        st.write(df_log_frequency)

        st.subheader("Ekstraksi Fitur TF IDF")
        tfidf_df = pd.read_csv('tfidf.csv')
        st.write(tfidf_df)

with tab3:
    st.subheader("LDA dan Modelling")
    st.write("""Latent Dirichlet Allocation (LDA) adalah model probabilistik generatif dari koleksi 
    data diskrit seperti korpus teks. Ide dasarnya adalah bahwa dokumen direpresentasikan sebagai 
    campuran acak atas topik laten (tidak terlihat).""")

    st.write("""LDA merupakan model Bayesian hirarki tiga tingkat, di mana setiap item koleksi dimodelkan 
    sebagai campuran terbatas atas serangkaian set topik. Setiap topik dimodelkan sebagai campuran tak 
    terbatas melalui set yang mendasari probabilitas topik. Dalam konteks pembuatan model teks, 
    probabilitas topik memberikan representasi eksplisit dari sebuah dokumen.""")

    # ambil label
    datalabel = data['Kelas']
    tfidf_df = pd.read_csv('tfidf.csv')

    st.subheader("Proporsi topik pada dokumen")
    k = 3
    alpha = 0.1
    beta = 0.2

 
    lda_model = LatentDirichletAllocation(n_components=k, doc_topic_prior=alpha, topic_word_prior=beta)
    # Proporsi topik pada dokumen
    proporsi_topik_dokumen = lda_model.fit_transform(tfidf_df)
    topic_names = [f"Topik {i+1}" for i in range(k)]
    proporsi_topik_dokumen_df = pd.DataFrame(proporsi_topik_dokumen, columns=topic_names)
    # proporsi_topik_dokumen_df.insert(0,'stemmed_tokens', abstrak)
    data_final = pd.read_csv("data_final.csv")
    dt_fin_jud = data_final['Judul']
    dt_top_judul = pd.concat([dt_fin_jud,proporsi_topik_dokumen_df],axis=1)
    st.write(dt_top_judul)

    # Proporsi kata pada topik
    st.subheader('Proporsi Kata pada Setiap Topik')
    # Menampilkan distribusi kata pada setiap topik
    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(tfidf_df)
    proporsi_kata_topik = lda_model.components_
    proporsi_kata_topik_df = pd.DataFrame(proporsi_kata_topik.T, columns=[f"Topik {i+1}" for i in range(k)],index=vectorizer.get_feature_names_out())
    dt_kata_top_judul = pd.concat([dt_fin_jud,proporsi_kata_topik_df],axis=1)
    st.write(dt_kata_top_judul)       

    # Data Topik dengan kelas
    st.subheader("Data Topik dengan Kelas")
    df_final = pd.concat([dt_fin_jud,proporsi_topik_dokumen_df,datalabel],axis=1)
    # menghapus data NaN
    df_final.isna()
    df_final.dropna(inplace=True)
    st.write(df_final)

    #Train and Test split
    X = df_final.iloc[:,1:k]
    y = df_final['Kelas']
    X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.3,random_state=0)

    st.subheader("Pilih Model")
    model1 = st.checkbox("Naive Bayes")
    model2 = st.checkbox("KNN")

    if model1:
        model = GaussianNB()
        filename = "naivebayes_model.pkl"

        model.fit(X_train,y_train)
        Y_pred = model.predict(X_test)

        score=metrics.accuracy_score(y_test,Y_pred)
        loaded_model = pickle.load(open(filename, 'rb'))
        st.write("Hasil Akurasi Algoritma Naive Bayes GaussianNB : ",score)
    
    if model2:
        model = KNeighborsClassifier(n_neighbors=3)
        filename = "knn_model.pkl"
        model.fit(X_train,y_train)
        Y_pred = model.predict(X_test)

        score=metrics.accuracy_score(y_test,Y_pred)
        loaded_model = pickle.load(open(filename, 'rb'))
        st.write("Hasil Akurasi Algoritma KNN : ",score)


with tab4:
    st.subheader("Reduksi Dimensi")
    tfidf_df = pd.read_csv('tfidf.csv')
    data_final = pd.read_csv("data_final.csv")
    st.subheader("Data TF IDF")

    dt_fin_jud = data_final['Judul']
    dt_label = data_final['Kelas']
    df_tfidf = pd.concat([dt_fin_jud,tfidf_df,dt_label],axis=1)
    st.write(df_tfidf)
    jml_tfidf = df_tfidf.shape
    st.write("Jumlah Baris dan Kolom : ",jml_tfidf)


    st.subheader("Data Proporsi Topik Dalam Dokument")
    st.subheader("Proporsi topik pada dokumen")
    k = 3
    alpha = 0.1
    beta = 0.2
    lda_model = LatentDirichletAllocation(n_components=k, doc_topic_prior=alpha, topic_word_prior=beta)
    # Proporsi topik pada dokumen
    proporsi_topik_dokumen = lda_model.fit_transform(tfidf_df)
    proporsi_topik_dokumen_df = pd.DataFrame(proporsi_topik_dokumen, columns=['Topik 1', 'Topik 2', 'Topik 3'])
    # proporsi_topik_dokumen_df.insert(0,'stemmed_tokens', abstrak)
    df_final_abs = pd.concat([dt_fin_jud, proporsi_topik_dokumen_df,datalabel],axis=1)
    # menghapus data NaN
    df_final_abs.isna()
    df_final_abs.dropna(subset=['Judul','Topik 1', 'Topik 2', 'Topik 3', 'Kelas'],inplace = True)
    st.write(df_final_abs)
    jml_top = df_final_abs.shape
    st.write("Jumlah Baris dan Kolom : ",jml_top)
    
with tab5:
    st.subheader("Implementasi")

    # ambil label
    datalabel = data['Kelas']
    tfidf_df = pd.read_csv('tfidf.csv')
    k = 3
    alpha = 0.1
    beta = 0.2

 
    lda_model = LatentDirichletAllocation(n_components=k, doc_topic_prior=alpha, topic_word_prior=beta)
    # Proporsi topik pada dokumen
    proporsi_topik_dokumen = lda_model.fit_transform(tfidf_df)
    topic_names = [f"Topik {i+1}" for i in range(k)]
    proporsi_topik_dokumen_df = pd.DataFrame(proporsi_topik_dokumen, columns=topic_names)
    # proporsi_topik_dokumen_df.insert(0,'stemmed_tokens', abstrak)
    data_final = pd.read_csv("data_final.csv")
    dt_fin_jud = data_final['Judul']
    dt_top_judul = pd.concat([dt_fin_jud,proporsi_topik_dokumen_df],axis=1)

    # Proporsi kata pada topik
    # Menampilkan distribusi kata pada setiap topik
    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(tfidf_df)
    proporsi_kata_topik = lda_model.components_
    proporsi_kata_topik_df = pd.DataFrame(proporsi_kata_topik.T, columns=[f"Topik {i+1}" for i in range(k)],index=vectorizer.get_feature_names_out())
    dt_kata_top_judul = pd.concat([dt_fin_jud,proporsi_kata_topik_df],axis=1)      

    df_final = pd.concat([dt_fin_jud,proporsi_topik_dokumen_df,datalabel],axis=1)
    # menghapus data NaN
    df_final.isna()
    df_final.dropna(inplace=True)

    st.subheader("Parameter Inputan")
    topik1 = st.number_input("Masukkan Nilai Topik 1 :")
    topik2 = st.number_input("Masukkan Nilai Topik 2 :")
    topik3 = st.number_input("Masukkan Nilai Topik 3 :")

    hasil = st.button("cek klasifikasi")

    #Train and Test split
    X = df_final.iloc[:,1:k]
    y = df_final['Kelas']
    X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.3,random_state=0)

    if hasil:
        model = GaussianNB()
        filename = "naivebayes_model.pkl"

        model.fit(X_train,y_train)
        Y_pred = model.predict(X_test)

        score=metrics.accuracy_score(y_test,Y_pred)
        loaded_model = pickle.load(open(filename, 'rb'))
        
        dataArray = [topik1, topik2, topik3]
        pred = loaded_model.predict([dataArray])

        st.success(f"Prediksi Hasil Klasifikasi : {pred[0]}")
        st.write(f"Algoritma yang digunakan adalah = Naive Bayes")
        st.success(f"Hasil Akurasi : {score}")