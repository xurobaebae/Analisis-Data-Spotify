<p align="center">
    <img src="images/spotify_logo.png" align="center" width="30%">
</p>
<p align="center"><h1 align="center">ANALISIS-DATA-SPOTIFY</h1></p>
<p align="center">
	<em><code>-------------</code></em>
</p>
<p align="center">
	<img src="https://img.shields.io/github/license/xurobaebae/Analisis-Data-Spotify?style=default&logo=opensourceinitiative&logoColor=white&color=0080ff" alt="license">
	<img src="https://img.shields.io/github/last-commit/xurobaebae/Analisis-Data-Spotify?style=default&logo=git&logoColor=white&color=0080ff" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/xurobaebae/Analisis-Data-Spotify?style=default&color=0080ff" alt="repo-top-language">
	<img src="https://img.shields.io/github/languages/count/xurobaebae/Analisis-Data-Spotify?style=default&color=0080ff" alt="repo-language-count">
</p>
<p align="center"><!-- default option, no dependency badges. -->
</p>
<p align="center">
	<!-- default option, no dependency badges. -->
</p>
<br>


# Analisis Data Spotify

# Struktur Project
```sh
└── Analisis-Data-Spotify/
    ├── DataAnalyze.ipynb
    ├── README.md
    ├── correlation_matrix.csv
    ├── dataset
    │   ├── Most Streamed Spotify Songs 2023 data.csv
    │   └── Most Streamed Spotify Songs 2024 data.csv
    ├── images
    │   └── spotify_logo.png 
    ├── p_values.csv
    └── spotify_analysis_report.txt
```

## Deskripsi Proyek
Proyek ini bertujuan untuk menganalisis data dari Spotify pada tahun 2023 dan 2024. Kami menggunakan berbagai teknik pemodelan dan analisis untuk memahami tren dan performa lagu di platform Spotify.

## Library yang Digunakan
Kami menggunakan beberapa library berikut dalam proyek ini:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec  # Import for more control over layout
import seaborn as sns
from scipy.stats import pearsonr
```

## Membaca Dataset
Dataset untuk tahun 2023 dan 2024 dibaca menggunakan `pandas`:

```python
spotify2024_df = pd.read_csv("dataset/Most Streamed Spotify Songs 2024 data.csv", encoding='latin1')
spotify2023_df = pd.read_csv("dataset/Most Streamed Spotify Songs 2023 data.csv", encoding='latin1')
```

## Menggabungkan Dataset
Kedua dataset digabungkan berdasarkan kolom yang relevan dengan menggunakan outer join:

```python
spotify_combined_df = pd.merge(
    left=spotify2024_df,
    right=spotify2023_df,
    left_on='Track',  # Kolom di DataFrame 2024
    right_on='track_name',  # Kolom di DataFrame 2023
    how='outer',  # Menggunakan outer join untuk memasukkan semua data
    suffixes=('_2024', '_2023')  # Menambahkan suffix untuk kolom yang sama
)
```

## Memilih Kolom untuk Analisis
Kolom yang ingin ditampilkan ditentukan sebagai berikut:

```python
comparison_columns = [
    'Track', 'Artist', 'Album Name', 'Release Date', 
    'Spotify Streams', 'Spotify Popularity', 
    'Spotify Playlist Count', 'Spotify Playlist Reach', 
    'track_name', 'artist(s)_name', 'streams', 
    'released_year', 'released_month', 'released_day'
]
```

## Menyaring DataFrame
DataFrame disaring untuk kolom yang relevan:

```python
spotify_comparison_df = spotify_combined_df[comparison_columns]
```

## Menampilkan Hasil Gabungan
Hasil gabungan ditampilkan untuk analisis awal:

```python
print(spotify_comparison_df.head())  # Tampilkan beberapa baris pertama dari DataFrame yang digabung
```

### Contoh Output
Berikut adalah contoh output dari DataFrame yang digabung:

| Track                           | Artist          | Album Name                      | Release Date | Spotify Streams | Spotify Popularity | Spotify Playlist Count | Spotify Playlist Reach | track_name                     | artist(s)_name          | streams | released_year | released_month | released_day |
|---------------------------------|-----------------|----------------------------------|--------------|-----------------|--------------------|-----------------------|-----------------------|--------------------------------|-------------------------|---------|----------------|-----------------|---------------|
| "Slut!" (Taylor's Version)      | Taylor Swift    | 1989 (Taylor's Version)         | 10/26/2023   | 265,932,119     | NaN                | 15,133                | 21,522,387            | "Slut!" (Taylor's Version)   | Taylor Swift           | NaN     | NaN            | NaN             | NaN           |
| "ýýýýýýý ýýýýýýýýýýý"          | Myriam Fares    | "ýýýýýý ýýýýýýýýýý "ýýýýýýýýýýý  | 3/10/2022    | 7,383,901       | 48.0               | 4,008                 | 539,281               | "ýýýýýýý ýýýýýýýýýýý"          | Myriam Fares           | NaN     | NaN            | NaN             | NaN           |
| #BrooklynBloodPop!             | SyKo            | #BrooklynBloodPop!              | 7/27/2020    | 289,085,486     | 62.0               | 76,485                | 8,876,090            | #BrooklynBloodPop!             | SyKo                   | NaN     | NaN            | NaN             | NaN           |
| 'Til You Can't                  | Cody Johnson     | 'Til You Can't / Longer Than S  | 6/11/2021    | 258,769,609     | 68.0               | 48,619                | 12,448,112           | 'Til You Can't                  | Cody Johnson           | NaN     | NaN            | NaN             | NaN           |
| 'Till I Collapse                | Eminem          | The Eminem Show                 | 5/26/2002    | 1,958,809,069   | 79.0               | 282,888               | 74,055,843           | 'Till I Collapse                | Eminem, Nate Dogg     | 1695712020 | 2002.0         | 5.0            | 26.0         |

## Memeriksa Nilai yang Hilang
Pemeriksaan nilai yang hilang dapat dilakukan dengan:

```python
missing_values = spotify_comparison_df.isnull().sum()
print(missing_values)
```

---

## Memeriksa Nilai yang Hilang
Pertama, kita memeriksa apakah terdapat nilai yang hilang dalam DataFrame yang telah digabungkan. Hal ini dilakukan dengan menggunakan fungsi `isnull()` dan `sum()`.

```python
# Memeriksa nilai yang hilang
missing_values = spotify_comparison_df.isnull().sum()

# Menampilkan hasil
print("Jumlah Missing Values per Kolom:")
print(missing_values[missing_values > 0])  # Hanya menampilkan kolom dengan missing values
```

### Hasil Jumlah Missing Values per Kolom
Output yang dihasilkan menunjukkan jumlah nilai yang hilang untuk setiap kolom:

```
Jumlah Missing Values per Kolom:
Track                     280
Artist                    285
Album Name                280
Release Date              280
Spotify Streams           393
Spotify Popularity      1088
Spotify Playlist Count    350
Spotify Playlist Reach    352
track_name              3852
artist(s)_name          3852
streams                 3852
released_year           3852
released_month          3852
released_day            3852
dtype: int64
```

## Memeriksa Duplikasi
Selanjutnya, kita memeriksa apakah ada duplikasi dalam DataFrame dengan menggunakan metode `duplicated()`.

```python
# Memeriksa duplikasi
duplicate_count = spotify_comparison_df.duplicated().sum()

# Menampilkan hasil
print("\nJumlah Duplicates:", duplicate_count)
```

### Hasil Jumlah Duplicates
Output yang dihasilkan menunjukkan jumlah baris duplikat dalam DataFrame:

```
Jumlah Duplicates: 2
```

## Pembersihan Data
Setelah memeriksa nilai yang hilang dan duplikasi, langkah selanjutnya adalah membersihkan data dengan menghapus nilai null dan NaN.

```python
# Pembersihan data: Menghapus nilai null dan NaN
spotify_cleaned_df = spotify_comparison_df.dropna()

# Menampilkan hasil setelah pembersihan nilai null dan NaN
print("Data setelah menghapus nilai null dan NaN:")
print(spotify_cleaned_df.head())  # Menampilkan beberapa baris pertama dari DataFrame yang telah dibersihkan
```

### Data Setelah Menghapus Nilai Null dan NaN
Output yang ditampilkan setelah pembersihan data memberikan contoh beberapa baris yang tersisa:

| Track                           | Artist          | Album Name                      | Release Date | Spotify Streams | Spotify Popularity | Spotify Playlist Count | Spotify Playlist Reach | track_name                     | artist(s)_name          | streams     | released_year | released_month | released_day |
|---------------------------------|-----------------|----------------------------------|--------------|-----------------|--------------------|-----------------------|-----------------------|--------------------------------|-------------------------|-------------|----------------|-----------------|---------------|
| 'Till I Collapse                | Eminem          | The Eminem Show                 | 5/26/2002    | 1,958,809,069   | 79.0               | 282,888               | 74,055,843           | 'Till I Collapse                | Eminem, Nate Dogg     | 1695712020  | 2002.0         | 5.0            | 26.0         |
| (It Goes Like) Nanana - Edit    | Peggy Gou       | (It Goes Like) Nanana [Edit]   | 6/15/2023    | 460,156,070     | 77.0               | 163,449               | 127,827,271          | (It Goes Like) Nanana - Edit    | Peggy Gou             | 57876440    | 2023.0         | 6.0            | 15.0         |
| 10 Things I Hate About You      | Leah Kate       | 10 Things I Hate About You      | 3/23/2022    | 238,502,829     | 4.0                | 38,595                | 7,419,495            | 10 Things I Hate About You      | Leah Kate             | 185550869   | 2022.0         | 3.0            | 23.0         |
| 10:35                           | Tiësto          | 10:35                           | 11/3/2022    | 539,802,784     | 74.0               | 82,093                | 99,413,749           | 10:35                           | Tiësto, Tate McRae    | 325592432   | 2022.0         | 11.0           | 1.0          |
| 2 Be Loved (Am I Ready)        | Lizzo           | Special                         | 7/15/2022    | 335,194,737     | 62.0               | 36,991                | 30,514,705           | 2 Be Loved (Am I Ready)        | Lizzo                 | 247689123   | 2022.0         | 7.0            | 14.0         |

---
Berikut adalah langkah-langkah untuk melakukan analisis yang sama pada DataFrame Spotify yang telah dibersihkan dengan fokus pada pengukuran tendensi pusat dan sebaran, serta visualisasi. Saya akan menguraikan kode Python yang perlu Anda jalankan. Pastikan untuk menggunakan pandas dan matplotlib dalam analisis Anda.

### 1. Memeriksa Nilai yang Hilang
```python
# Memeriksa nilai yang hilang setelah pembersihan
missing_values_after_cleaning = spotify_cleaned_df.isnull().sum()

# Menampilkan hasil
print("Jumlah Missing Values per Kolom setelah Pembersihan:")
print(missing_values_after_cleaning[missing_values_after_cleaning > 0])
```

### 2. Memeriksa Duplikasi
```python
# Memeriksa duplikasi setelah pembersihan
duplicate_count_after_cleaning = spotify_cleaned_df.duplicated().sum()

# Menampilkan hasil
print("\nJumlah Duplicates setelah Pembersihan:", duplicate_count_after_cleaning)
```

### 3. Menghitung dan Menampilkan Ukuran Tendensi Pusat
```python
# Mengambil kolom numerik dari DataFrame
numeric_columns = spotify_cleaned_df.select_dtypes(include='number').columns

# Menghitung mean, median, dan mode untuk setiap kolom numerik
mean_values = spotify_cleaned_df[numeric_columns].mean()
median_values = spotify_cleaned_df[numeric_columns].median()
mode_values = spotify_cleaned_df[numeric_columns].mode().iloc[0]  # Mengambil mode dari setiap kolom

# Menampilkan hasil
print("Mean (Rata-rata) dari Kolom Numerik:")
print(mean_values)

print("\nMedian (Nilai Tengah) dari Kolom Numerik:")
print(median_values)

print("\nMode (Nilai Paling Sering Muncul) dari Kolom Numerik:")
print(mode_values)
```

### 4. Menghitung dan Menampilkan Ukuran Sebaran
```python
# Menghitung ukuran sebaran
range_values = spotify_cleaned_df[numeric_columns].max() - spotify_cleaned_df[numeric_columns].min()
variance_values = spotify_cleaned_df[numeric_columns].var()
std_deviation_values = spotify_cleaned_df[numeric_columns].std()

# Menampilkan hasil
print("Rentang (Range) dari Kolom Numerik:")
print(range_values)

print("\nVarians (Variance) dari Kolom Numerik:")
print(variance_values)

print("\nDeviasi Standar (Standard Deviation) dari Kolom Numerik:")
print(std_deviation_values)
```

### 5. Menghitung Kuartil dan IQR
```python
# Menghitung Q1 dan Q3 untuk setiap kolom numerik
Q1 = spotify_cleaned_df[numeric_columns].quantile(0.25)
Q3 = spotify_cleaned_df[numeric_columns].quantile(0.75)

# Menghitung IQR
IQR = Q3 - Q1

# Menampilkan hasil
print("Kuartil Pertama (Q1) dari Kolom Numerik:")
print(Q1)

print("\nKuartil Ketiga (Q3) dari Kolom Numerik:")
print(Q3)

print("\nInterquartile Range (IQR) dari Kolom Numerik:")
print(IQR)
```

### 6. Visualisasi Ukuran Tendensi Pusat
```python
import matplotlib.pyplot as plt

# Menghitung nilai tendency dan dispersion
mean_values = spotify_cleaned_df[numeric_columns].mean()
median_values = spotify_cleaned_df[numeric_columns].median()
mode_values = spotify_cleaned_df[numeric_columns].mode().iloc[0]

# Atur ukuran figure
plt.figure(figsize=(15, 5))

# Visualisasi untuk Measure of Central Tendency
plt.subplot(1, 3, 1)
bar_height = 0.2  # Tinggi bar
y = range(len(numeric_columns))

# Plot Mean, Median, Mode sebagai bar horizontal terpisah
plt.barh(y, mean_values, height=bar_height, color='blue', alpha=0.7, label='Mean', edgecolor='black')
plt.barh([p + bar_height for p in y], median_values, height=bar_height, color='orange', alpha=0.7, label='Median', edgecolor='black')
plt.barh([p + bar_height * 2 for p in y], mode_values, height=bar_height, color='green', alpha=0.7, label='Mode', edgecolor='black')

plt.yticks([p + bar_height for p in y], numeric_columns)  # Sesuaikan posisi y
plt.title('Measure of Central Tendency')
plt.xlabel('Nilai')
plt.legend()

plt.show()
```

## Visualisasi Ukuran Sebaran

Untuk memahami distribusi ukuran sebaran dalam dataset, kita dapat menggunakan grafik batang horizontal untuk menunjukkan variance, standard deviation, interquartile range (IQR), dan range. 

### Mengatur Ukuran Figure
Pertama, kita atur ukuran figure untuk visualisasi.

```python
plt.figure(figsize=(15, 5))
```

### Membuat Subplot untuk Ukuran Sebaran
Kita kemudian membuat subplot untuk visualisasi ukuran sebaran:

```python
plt.subplot(1, 3, 2)
bar_height = 0.2  # Tinggi bar
y = range(len(numeric_columns))

# Plot Variance, Standard Deviation, IQR dan Range sebagai bar horizontal
plt.barh(y, variance_values, height=bar_height, color='green', alpha=0.7, label='Variance', edgecolor='black')
plt.barh([p + bar_height for p in y], std_dev_values, height=bar_height, color='red', alpha=0.7, label='Standard Deviation', edgecolor='black')
plt.barh([p + bar_height * 2 for p in y], IQR, height=bar_height, color='purple', alpha=0.7, label='IQR', edgecolor='black')
plt.barh([p + bar_height * 3 for p in y], Q3 - Q1, height=bar_height, color='blue', alpha=0.7, label='Range', edgecolor='black')

plt.yticks([p + 1.5 * bar_height for p in y], numeric_columns)  # Sesuaikan posisi y
plt.title('Measure of Dispersion')
plt.xlabel('Nilai')
plt.legend()
```

### Menambahkan Label untuk Setiap Bar
Kita juga menambahkan label untuk setiap bar agar informasi lebih jelas.

```python
for i in range(len(numeric_columns)):
    plt.text(variance_values[i] + 0.05, i, f'{variance_values[i]:.2f}', color='green', va='center')
    plt.text(std_dev_values[i] + 0.05, i + bar_height, f'{std_dev_values[i]:.2f}', color='red', va='center')
    plt.text(IQR[i] + 0.05, i + bar_height * 2, f'{IQR[i]:.2f}', color='purple', va='center')
    plt.text((Q3[i] - Q1[i]) + 0.05, i + bar_height * 3, f'{Q3[i] - Q1[i]:.2f}', color='blue', va='center')
```

### Menampilkan Grafik
Akhirnya, kita menampilkan grafik dengan layout yang baik.

```python
plt.tight_layout()
plt.show()
```

## Boxplot untuk IQR

Untuk memvisualisasikan sebaran data, kita juga membuat boxplot untuk beberapa kolom yang dipilih.

### Menentukan Kolom yang Ditampilkan
Kita mulai dengan menentukan kolom yang ingin ditampilkan.

```python
columns_to_plot = ['Spotify Popularity', 'released_year', 'released_month', 'released_day']
```

### Mengatur Ukuran Figure untuk Boxplot
Mengatur ukuran figure untuk boxplot.

```python
plt.figure(figsize=(15, 10))
```

### Loop untuk Membuat Boxplot
Kita kemudian membuat boxplot untuk setiap kolom yang ditentukan.

```python
for i, column in enumerate(columns_to_plot):
    plt.subplot(2, 2, i + 1)  # Membagi plot menjadi 2x2
    sns.boxplot(y=spotify_cleaned_df[column], palette='Set2', showmeans=True, linewidth=1.5)
    plt.title(f'Boxplot for {column}')
    plt.ylabel('Nilai')
    
    # Menonaktifkan anotasi default dari Seaborn
    sns.despine(top=True, right=True, left=True)
    
    # Menambahkan anotasi untuk IQR secara manual
    Q1_value = spotify_cleaned_df[column].quantile(0.25)
    Q3_value = spotify_cleaned_df[column].quantile(0.75)
    IQR_value = Q3_value - Q1_value
    plt.text(0.5, Q3_value + 0.1, f'IQR: {IQR_value:.2f}', color='purple', ha='center')

plt.tight_layout()
plt.show()
```

## Analisis Korelasi

### Menghitung Matriks Korelasi
Kita menghitung matriks korelasi untuk kolom numerik dalam dataset.

```python
numeric_columns = spotify_cleaned_df.select_dtypes(include='number').columns
correlation_matrix = spotify_cleaned_df[numeric_columns].corr()
```

### Visualisasi Matriks Korelasi
Kita dapat memvisualisasikan matriks korelasi menggunakan heatmap.

```python
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Matriks Korelasi')
plt.show()
```

### Uji Signifikansi
Kita melakukan uji Pearson untuk setiap pasangan kolom dan menyimpan p-values.

```python
p_values = pd.DataFrame(index=numeric_columns, columns=numeric_columns)

for col1 in numeric_columns:
    for col2 in numeric_columns:
        if col1 != col2:
            _, p_value = pearsonr(spotify_cleaned_df[col1].dropna(), spotify_cleaned_df[col2].dropna())
            p_values.loc[col1, col2] = p_value
        else:
            p_values.loc[col1, col2] = np.nan  # Mengisi diagonal dengan NaN

print("P-values dari Uji Signifikansi Korelasi:")
print(p_values)
```

## Evaluasi Model

### Persiapan Dataset
Kita mempersiapkan dataset dengan mengidentifikasi dan memisahkan fitur kategorikal dan numerikal.

```python
numeric_columns = spotify_cleaned_df.select_dtypes(include=['int64', 'float64']).columns
categorical_columns = spotify_cleaned_df.select_dtypes(include=['object']).columns

print("Numerik Kolom:", numeric_columns)
print("Kategorikal Kolom:", categorical_columns)
```

### Mengubah Kategorikal ke Numerikal
Menggunakan `OneHotEncoder` untuk mengubah kolom kategorikal menjadi numerikal.

```python
encoder = OneHotEncoder(sparse_output=False)  # Gunakan sparse_output
X_encoded = encoder.fit_transform(spotify_cleaned_df[categorical_columns])
encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_columns))
X = pd.concat([spotify_cleaned_df[numeric_columns].reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
y = spotify_cleaned_df['Spotify Popularity']
```

### Pembagian Dataset
Membagi dataset menjadi data latih dan data uji.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Normalisasi Fitur
Melakukan normalisasi pada fitur.

```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### Membuat Model Klasifikasi
Membuat dan melatih model klasifikasi menggunakan Random Forest.

```python
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Melakukan prediksi pada data uji
y_pred = model.predict(X_test)

# Menghitung metrik evaluasi
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2f}")
```

### Tuning Parameter
Menentukan parameter untuk tuning dan menggunakan GridSearchCV untuk mencari parameter terbaik.

```python
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best Parameters from Grid Search:")
print(grid_search.best_params_)

# Menggunakan model terbaik untuk prediksi
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

# Menghitung metrik evaluasi dengan model terbaik
print("\nConfusion Matrix for Best Model:")
print(confusion_matrix(y_test, y_pred_best))

print("\nClassification Report for Best Model:")
print(classification_report(y_test, y_pred_best))

best_accuracy = accuracy_score(y_test, y_pred_best)
print(f"\nBest Model Accuracy: {best_accuracy:.2f}")
```

### Tabel Classification Report
Berikut adalah tabel dari Classification Report untuk model:

| Class          | Precision | Recall | F1-Score | Support |
|----------------|-----------|--------|----------|---------|
| 4.0            | 0.00      | 0.00   | 0.00

     | 5       |
| 5.0            | 0.00      | 0.00   | 0.00     | 7       |
| 6.0            | 0.45      | 0.82   | 0.58     | 11      |
| 7.0            | 0.67      | 0.33   | 0.44     | 6       |
| 8.0            | 0.85      | 0.75   | 0.80     | 20      |
| 9.0            | 1.00      | 0.10   | 0.18     | 10      |

*Support: jumlah data untuk masing-masing kelas

### 1. Memeriksa DataFrame
Pertama, kita memeriksa nama kolom yang ada dalam DataFrame dan beberapa baris pertama untuk memahami struktur data yang kita miliki.

```python
# Memeriksa nama kolom yang ada dalam DataFrame
print(spotify_cleaned_df.columns)

# Memeriksa beberapa baris pertama dalam DataFrame
print(spotify_cleaned_df.head())
```
Output yang diharapkan:
```
Index(['Track', 'Artist', 'Album Name', 'Release Date', 'Spotify Streams', 
      'Spotify Popularity', 'Spotify Playlist Count', 'Spotify Playlist Reach', 
      'track_name', 'artist(s)_name', 'streams', 'released_year', 
      'released_month', 'released_day'], dtype='object')
```
### 2. Analisis Top 5 Artis dan Lagu
Kita akan menentukan 5 artis dengan jumlah stream tertinggi dan 5 lagu dengan popularitas tertinggi untuk tahun 2023 dan 2024.

#### Fungsi untuk Menghitung 5 Artis dengan Stream Tertinggi
```python
def top_artists_by_streams(df, years, top_n=5):
    filtered_df = df[df['released_year'].isin(years)]
    top_artists = (filtered_df.groupby('artist(s)_name')['Spotify Streams'].sum()
                   .reset_index()
                   .sort_values(by='Spotify Streams', ascending=False)
                   .head(top_n))
    return top_artists
```

#### Fungsi untuk Menghitung 5 Lagu Paling Populer
```python
def top_songs_by_popularity(df, years, top_n=5):
    filtered_df = df[df['released_year'].isin(years)]
    top_songs = (filtered_df.groupby('track_name')['Spotify Popularity'].mean()
                 .reset_index()
                 .sort_values(by='Spotify Popularity', ascending=False)
                 .head(top_n))
    return top_songs
```

### 3. Mendapatkan Data Top Artis dan Lagu
Tahun yang ingin dianalisis:
```python
years_to_analyze = [2023, 2024]

# Mendapatkan 5 artis dengan stream tertinggi
top_artists = top_artists_by_streams(spotify_cleaned_df, years_to_analyze)
print("Top 5 Artists by Streams (2023 & 2024):")
print(top_artists)

# Mendapatkan 5 lagu paling populer
top_songs = top_songs_by_popularity(spotify_cleaned_df, years_to_analyze)
print("\nTop 5 Songs by Popularity (2023 & 2024):")
print(top_songs)
```

### 4. Visualisasi Data
Kita akan memvisualisasikan hasil dengan membuat dua grafik batang.

#### Fungsi Visualisasi
```python
def plot_top_artists_songs(top_artists, top_songs):
    plt.figure(figsize=(12, 6))

    # Plot artis (grafik batang vertikal)
    plt.subplot(1, 2, 1)
    sns.barplot(x='artist(s)_name', y='Spotify Streams', data=top_artists, palette='Blues')
    plt.title('Top 5 Artists by Streams (2023 & 2024)')
    plt.ylabel('Total Streams')
    plt.xticks(rotation=45, ha='right')

    # Plot lagu (grafik batang vertikal)
    plt.subplot(1, 2, 2)
    sns.barplot(x='track_name', y='Spotify Popularity', data=top_songs, palette='Oranges')
    plt.title('Top 5 Songs by Popularity (2023 & 2024)')
    plt.ylabel('Average Popularity')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.show()

# Memanggil fungsi untuk plot artis dan lagu
plot_top_artists_songs(top_artists, top_songs)
```

### 5. Kesimpulan dan Rekomendasi
Setelah melakukan analisis, kita menarik kesimpulan dan memberikan rekomendasi berdasarkan hasil yang diperoleh.

#### Fungsi Kesimpulan
```python
def overall_conclusion(top_artists, top_songs):
    conclusion = (
        f"5 Artis dengan jumlah stream tertinggi antara tahun 2023 dan 2024 adalah:\n"
        f"{top_artists.to_string(index=False)}\n\n"
        f"5 Lagu paling populer antara tahun 2023 dan 2024 adalah:\n"
        f"{top_songs.to_string(index=False)}\n\n"
        "Dari analisis ini, dapat disimpulkan bahwa artis-artis tersebut memiliki dampak signifikan "
        "dalam industri musik dan lagu-lagu mereka mendapat respons yang sangat positif dari "
        "para pendengar pada tahun-tahun tersebut."
    )
    return conclusion

# Menampilkan kesimpulan keseluruhan
final_conclusion = overall_conclusion(top_artists, top_songs)
print(final_conclusion)
```

### 6. Laporan
Kita akan mendokumentasikan seluruh proses analisis dan hasilnya dalam sebuah laporan.

#### Fungsi untuk Membuat Laporan
```python
def generate_report(cleaning_steps, analysis_steps, conclusions, recommendations, comparison_results, output_file='spotify_analysis_report.txt'):
    with open(output_file, 'w') as file:
        file.write("""Spotify Data Analysis Report
Data Cleaning Steps
""")
        file.write('\n'.join(cleaning_steps))
        file.write("""

Analysis Steps
""")
        file.write('\n'.join(analysis_steps))
        file.write("""

Analysis Results
""")

        # Tambahkan hasil analisis untuk setiap tahun
        for year, result in comparison_results.items():
            file.write(f"\nYear: {year}\n")
            file.write("Top 5 Artists:\n")
            file.write(result['top_artists'].to_string(index=False))
            file.write("\n\nTop 5 Songs:\n")
            file.write(result['top_songs'].to_string(index=False))
            file.write("\n")

        file.write("""
4. Conclusions
""")
        file.write('\n'.join([f"- {conclusion}" for conclusion in conclusions]))
        file.write("""

Recommendations
""")
        file.write('\n'.join([f"- {recommendation}" for recommendation in recommendations]))

        file.write("""
6. Summary
Hasil analisis menunjukkan bahwa artis dan lagu tertentu memiliki dampak besar di platform Spotify. Rekomendasi lebih lanjut dapat digunakan untuk strategi pemasaran berbasis data.
""")

# Contoh input untuk dokumentasi
cleaning_steps = [
    "Menghapus data yang memiliki nilai null di kolom penting seperti 'artist(s)_name' dan 'Spotify Streams'.",
    "Menghapus duplikasi berdasarkan 'track_name' dan 'artist(s)_name'.",
    "Menyesuaikan format kolom tanggal dan mengekstraksi tahun dari kolom 'release_date'."
]

analysis_steps = [
    "Menghitung total streams untuk setiap artis berdasarkan tahun.",
    "Menghitung rata-rata popularitas untuk setiap lagu berdasarkan tahun.",
    "Membandingkan hasil analisis historis (2023 & 2024)."
]

conclusions = [
    "Artis dengan streams tertinggi tetap mendominasi pada tahun 2023 hingga 2024.",
    "Lagu dengan popularitas tinggi di masa lalu cenderung mempertahankan performa.",
    "Data menunjukkan adanya tren positif terhadap artis-artis baru pada tahun 2024."
]

recommendations = [
    "Fokus pada artis dengan popularitas stabil untuk promosi jangka panjang.",
    "Identifikasi tren lagu berdasarkan popularitas untuk mengoptimalkan strategi playlist.",
    "Gunakan hasil ini untuk memperluas pasar di wilayah dengan engagement tinggi."
]

# Panggil fungsi visualisasi dan laporan
final_visualization(comparison_results)
generate_report(cleaning_steps, analysis_steps, conclusions, recommendations, comparison_results)

print("Laporan telah dibuat: spotify_analysis_report.txt")
```

# Credit
Spotify 2024 dataset : [https://www.kaggle.com/datasets/nelgiriyewithana/most-streamed-spotify-songs-2024]

Spotify 2023 dataset : [https://www.kaggle.com/datasets/nelgiriyewithana/top-spotify-songs-2023]
