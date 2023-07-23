# Machine Learning 

## Domain Proyek
Dalam bisnis tentu diperlukan pemahaman terhadap pasar, terutama dalam bisnis asuransi. 
Menurut [Wikipedia](https://id.wikipedia.org/wiki/Asuransi), Asuransi adalah pertanggungan atau perjanjian antara dua belah pihak, 
dimana pihak satu berkewajiban membayar iuran/kontribusi/premi. ketika bisnis asuransi sudah bisa dijalankan, kita perlu memahami bagaimana cara 
menawarkan produk asuransi tersebut kepada pelanggan yang tepat, baik itu adalah pelanggan baru maupun pelanggan lama. Oleh karena itu, 
sangat penting bagi sebuah perusahaan untuk mengerti pasar yang mereka sedang kuasai. Di dalam projek ini, saya memilih untuk menggunakan 
data dari sebuah perusahaan asuransi. Data ini berisi tentang data pribadi pelanggan seperti jenis kelamin, usia, kepemilikan sim dan tempat tinggal. 
Dalam data ini juga terdapat apabila pelanggan sudah mempunyai asuransi kendaraan, umur dari kendaraan pelanggan, konfirmasi bahwa kendaraan tersebut 
sudah pernah rusak, pembayaran per tahun pelanggan, jumlah hari pelanggan terikat pada perusahaan tersebut,dan respon terhadap hasil tawaran asuransi. 
Projek ini dibuat dengan tujuan untuk memprediksi pelanggan yang tertarik kepada produk asuransi kendaraan. Untuk menjawab masalah ini, tentu perlu diketahui 
siapa pelanggan yang tepat untuk menerima respon tawaran ini. Membuat grafik secara manual hanya akan memakan waktu yang cukup lama. Oleh sebab itu, 
diperlukan solusi dengan cara analisa dan membuat machine learning model untuk menjawab masalah ini.

## Business Understanding

### Problem Statement
Membuat prediksi untuk mengetahui pelanggan yang tepat untuk diberi tawaran produk asuransi kendaraan. 
Ini merupakan masalah klasifikasi karena data yang ditargetkan merupakan respon dari para pelanggan.

### Goals
Meningkatkan pemasukan perusahaan asuransi dan mempersingkat waktu penawaran.

### Solution statements
Untuk menyelesaikan masalah ini, saya akan mengajukan 3 solusi machine learning model yang sederhana karena data ini merupakan data klasifikasi non-linear. 
Berikut adalah penjelasan model-model machine learning yang akan digunakan untuk masalah ini. :
- **Random Forest** : Random forest merupakan model ensemble atau model gabungan dari beberapa decision tree dimana ini tentu akan memakan waktu yang lebih lama karena model ini akan menggunakan decision tree dalam jumlah yang banyak dan dari hasil tersebut akan dilakukan voting atau dengan membuat rata-rata agar menghasilkan prediksi yang akurat.
- **XGBoosting** : XGBoosting merupakan salah satu model gradient boosting yang menggunakan beberapa model decision tree secara iteratif dan menghasilkan prediksi yang akurat. Model ini memiliki performa akurasi yang sangat baik dan waktu pelatihan yang cepat. Model ini banyak digunakan oleh ilmuwan data untuk menjawab masalah klasifikasi dan masalah regresi. Salah satu kelemahan yang dimiliki oleh XGBoost adalah kolom kategori harus diubah menjadi one-hot encoding sebelum dimasukan ke dalam model ini. Selain itu, nama kolom juga harus diperhatikan karena model ini tidak menerima nama kolom yang memiliki koma atau simbol lainnya.
- **Decision Tree** : Decision Tree merupakan representasi dari berbagai macam opsi. Sama halnya dengan membuat bagan opsi jikalau terjadi sesuatu maka opsi tersebut akan dipilih. Model ini juga dihitung dengan probabilitas dan menggunakan beban untuk menghitung kondisi dari bagan ini. Untuk penambahan, perhitungan probabilitas ini tergantung dari setiap fitur yang ada dalam data tersebut. Kelebihan dari model ini adalah kita tidak perlu menggunakan normalisasi ataupun standarisasi untuk menggunakan model ini. Tetapi, perhitungan dalam mencari kondisi ini dapat menjadi lama dan komplex karena semakin banyak data yang diproses atau semakin banyak kolom yang diberikan maka pelatihan model ini akan menjadi lebih kompleks dan lebih lama.

## Data Understanding
Dataset ini diambil dari sebuah platform untuk ilmuwan data yaitu [Kaggle](https://www.kaggle.com/). Dalam platform tersebut terdapat banyak dataset dari 
berbagai sumber dan perusahaan yang dapat membantu para pemula mengerti tentang dunia ilmuwan data. Untuk projek ini, 
saya mengambil data ini dari sumber [ini](https://www.kaggle.com/anmolkumar/health-insurance-cross-sell-prediction). 
Berikut adalah keterangan mengenai maksud dari variabel - variabel atau kolom tersebut : 
- id : merupakan ID dari pelanggan.
- Gender : merupakan jenis kelamin pelanggan.
- Age  : Umur pelanggan.
- Driving_License : Bukti pelanggan mempunyai SIM.
    - 1 : Pelanggan mempunyai SIM.
    - 0 : Pelanggan tidak mempunyai SIM.
- Region_Code : Kode unik tempat tinggal pelanggan.
- Previously_Insured : Bukti pelanggan sudah memiliki asuransi kendaraan.
    - 1 : Pelanggan sudah mempunyai asuransi kendaraan.
    - 0 : Pelanggan belum mempunyai asuransi kendaraan.
- Vehicle_Age : Umur dari kendaraan pelanggan.
- Vehicle_Damage : Bukti kendaraan pelanggan sudah pernah mengalami luka atau kecelakaan.
    - 1 : Kendaraan pelanggan sudah pernah engalami luka atau kecelakaan.
    - 0 : Kendaraan pelanggan belum pernah engalami luka atau kecelakaan.
- Annual_Premium : Pembayaran yang perlu dibayar pelanggan setiap tahun.
- PolicySalesChannel : Kode untuk menghubungi pelanggan.
- Vintage : Hari pelanggan sudah berasosiasi dengan perusahaan tersebut.
- Response : Respon pelanggan atau target yang ingin diprediksi 
    - 1 : Pelanggan setuju jika ditawarkan asuransi kendaraan.
    - 0 : Pelanggan tidak setuju jika ditawarkan asuransi kendaraan.

Dalam proses ini, tentu diperlukannya visualisasi data. Visualisasi data yang diberi dalam projek ini adalah countplot dari library seaborn yaitu 
sebuah grafik bar yang menandakan berapa banyak pelanggan yang setuju dan tidak setuju. 

![image](https://user-images.githubusercontent.com/82896196/135032389-d8f9aa11-1047-431a-b37f-37d6206d6ed7.png)
![image](https://user-images.githubusercontent.com/82896196/135032449-62e5768e-c29f-4cd6-b712-b298cf578060.png)
![image](https://user-images.githubusercontent.com/82896196/135032478-d424c07f-a916-4070-b766-6293c2c19520.png)


Dari visualisasi data ini, saya mendapatkan informasi bahwa banyak pelanggan yang tidak setuju untuk membeli asuransi kendaraan, baik pelanggan tersebut adalah pria maupun wanita. Ditambah lagi, sebagian besar dari penolakan tersebut adalah pelanggan yang sudah memiliki SIM. Selain itu, terdapat bahwa lebih banyak pria yang berasosiasi dalam perusahaan tersebut jika dibandingkan dengan wanita. 


Setelah menggunakan countplot untuk mengetahui informasi tersebut. saya ingin melihat umur dari pelanggan yang merespon tidak dan yang merespon iya terhadap penawaran asuransi kendaraan. Saya menggunakan violin plot yaitu sebuah grafik yang menunjukan data numerik yang dapat dipadatkan bila semakin banyak data numerik yang sama dalam data tersebut.

![image](https://user-images.githubusercontent.com/82896196/135032586-92901b45-4c0b-4be7-b8fe-23a3ab87d8c7.png)

Setelah membuat violin plot, saya menemukan bahwa pelanggan berumur 20-30 tahun lebih banyak merespon tidak sedangkan pelanggan berumur 40-50 tahun lebih menerima tawaran asuransi kendaraan. Tentu hal ini membuat suatu pertanyaan. Jikalau grafik ini memberi informasi tersebut, mengapa sebagian besar pelanggan berkata tidak untuk menerima tawaran asuransi kendaraan ? Hal tersebut membuat saya membuat satu teknik visualisasi data selanjutnya yaitu histogram.

Histogram merupakan grafik yang menandakan seberapa banyak fitur yang terdistribusi dalam suatu kolom. 

![image](https://user-images.githubusercontent.com/82896196/135032646-8f2f2a26-ceff-40f3-afbb-e9179cecd4d1.png)


Setelah dibuatnya grafik ini, terbukti bahwa penyebab banyak pelanggan yang menolak tawaran asuransi kendaraan tersebut adalah pelanggan yang berumur 20-30 tahun memiliki angka yang sangat besar jika dibandingkan dengan pelanggan berumur 40-50 tahun. Dari hasil ini, bisa disimpulkan bahwa jikalau ingin persentase pelanggan yang menerima tawaran naik maka perusahaan harus mendekati atau menawarkan produk asuransi kendaraan kepada pelanggan berumur 40-50 tahun.


## Data Preparation
Untuk data preparation, saya menggunakan one-hot encoding yaitu metode untuk membagi fitur dalam suatu kolom kategori menjadi jumlah kolom yang serupa dengan jumlah fitur dalam kolom kategori tersebut dan mengubah setiap fitur dalam setiap kolom yang baru menjadi bilangan binari seperti gambar dibawah ini.

![image](https://user-images.githubusercontent.com/82896196/135027208-4811b300-e162-4ad8-b138-90505a3d262b.png)

Saya menggunakan metode ini karena model akan lebih menerima angka dibanding dengan kata-kata dan terlebih lagi hal ini akan membuat model semakin baik dalam pelatihan maupun prediksi. 

Selain itu, saya juga menggunakan standarisasi kepada model ini. standarisasi adalah metode yang membuat mean suatu data menjadi 0 dan standar deviasi menjadi 1. Saya menggunakan ini karena standarisasi dapat membantu proses pelatihan model dan membuat model semakin baik dalam performanya. Tentu ini mempunyai kekurangan yaitu saat melakukan prediksi, data harus dilakukan standarisasi terlebih dahulu lalu diprediksi dengan model yang sudah dibuat lalu dikembalikan lagi dalam bentuk normal. Berikut adalah fungsi atau rumus matematika dari standarisasi.

![image](https://user-images.githubusercontent.com/82896196/135034438-0b7afa68-e8fc-4e7b-94d3-62b6a5d30e4f.png)

Dalam gambar ini, X merupakan fitur dari sebuah kolom , μ adalah mean dari data tersebut , σ merupakan standar deviasi.


## Modeling
Pada subab sebelumnya, telah disampaikan bahwa ada empat model machine learning yang akan digunakan sebagai solusi yaitu  ***Random Forest** , **XGBoosting*** , dan ***Decision Tree***. dalam tahap ini, saya tidak melakukan *hyperparameter tuning* melainkan saya ingin menguji bagaimana performa empat model ini tanpa adanya perlakuan *hyperparameter tuning*. Untuk menguji performa model saya membuat suatu fungsi dimana fungsi tersebut akan melatih model yang sudah dibuat, lalu melakukan prediksi, dan membandingkan hasil model tersebut dengan menghitung berapa lama model ini dilatih dan bagaimana hasil akurasi dari model tersebut. Hasil akurasi model tersebut juga dihitung melalui metrik yang akan dibahas pada subab selanjutnya.

Dari hasil pelatihan, terbukti bahwa ada satu solusi yang sangat baik dalam menjawab masalah ini. Kedua model tersebut adalah ***XGBoosting***. Hal ini dapat disimpulkan karena kedua model ini mencapai akurasi tertinggi dalam prediksi yaitu 88%. 

## Evaluation

Untuk bagian Evaluasi, karena ini adalah masalah klasifikasi, saya menguji performa model ini dengan *classification report* , *confussion matrix* , dan *accuracy score*. Menurut [sumber](https://medium.com/@MohammedS/performance-metrics-for-classification-problems-in-machine-learning-part-i-b085d432082b) yang saya temukan, ketiga metrik ini sangat cocok untuk mengukur performa model *machine learning*. Berikut adalah penjelasan dari setiap metrik :
- **Confussion Matrix** : Metrik ini merupakan metrik yang digunakan dalam *hypothesis testing* karena metrik ini menghasilkan empat variabel yaitu *true posisitve* , *true negative* , *false positive*, *false negative*. Metrik ini digunakan untuk membuktikan berapa hasil prediksi yang sesuai. Berikut penjelasan empat variabel ini :
    - ***True Posisitve(TP)*** : Terjadi apabila hasil prediksi bernilai positif (1) dan sesuai dengan data yang sebenarnya (1)
    - ***False Posisitve(FP)*** : Terjadi apabila hasil prediksi bernilai positif (1) tetapi tidak sesuai dengan data yang sebenarnya (0)
    - ***True Negative(TN)*** : Terjadi apabila hasil prediksi bernilai negatif (0) dan sesuai dengan data yang sebenarnya (0)
    - ***False Negative(FN)*** : Terjadi apabila hasil prediksi bernilai negatif (0) tetapi tidak sesuai dengan data yang sebenarnya (1)
   
   ![image](https://user-images.githubusercontent.com/82896196/135038617-1f5b7656-b5e7-4e04-b575-17184b078a1e.png)

   
   Untuk menerapkan dalam kode, import library *sklearn.metric* untuk bisa menggunakan metode ini. Kemudian, Anda hanya perlu mengetik kode dibawah ini :
   
    *confusion_matrix(actual,prediction)*
    
- **Classification Report** : Metrik yang berasal dari hasil *confussion matrix* dan dihitung dengan rumus matematika. Di dalam metrik ini, terdapat **akurasi, precision, recall, dan F1 score**. berikut adalah penjelasan dari setiap fitur *classification report* :
    - **Akurasi** : akurasi dari model yang dibuat. Cara menghitung ini adalah dengan menjumlah semua prediksi yang benar dan dibagi oleh total semua prediksi yang benar maupun yang salah.
   
   ![image](https://user-images.githubusercontent.com/82896196/135038656-83cb4af3-1caf-444c-b9e7-7235e6b140e3.png)

   
    - **Presisi** : Metrik ini menghitung berapa hasil positif yang benar dari *confussion matrix*. Hal ini perlu dilakukan agar mengerti apakah hasil yang diberikan mengalami bias atau tidak. Berikut adalah rumus dari presisi :
    
    ![image](https://user-images.githubusercontent.com/82896196/135046499-fcbcbdbf-f2b4-414f-8bd4-170fd280cc52.png)

    
    - **Recall** : Metrik ini digunakan untuk menghitung berapa data yang salah dalam melakukan prediksi. Berikut adalah rumus dari metrik *recall* :
    
    ![image](https://user-images.githubusercontent.com/82896196/135047658-cfd69959-6f22-4ddd-9eea-10b6b90ed746.png)

    -  **F1-score** : Metrik ini merupakan *harmonic mean* atau rata-rata harmonik dari presisi dan recall. nilai tertinggi adalah 1.0 dan nilai terendah 0.0. Metrik ini digunakan sebagai patokan apabila model yang sudah dilatih bisa memprediksi setiap fitur dengan benar. Berikut adalah rumus dari F1-score :
    
    
    ![image](https://user-images.githubusercontent.com/82896196/135048996-03d6f1af-15fd-46bc-b792-885a9ca98bb3.png)

  untuk menggimplementasikan kode classification report, Anda hanya perlu menggunakan hal yang sama dengan *confussion matrix*, yaitu import library *sklearn.metric* lalu gunakan kode dibawah ini :
  
  *classification_report(actual,prediction)*
  
- **Accuracy Score** : Akurasi disini sama perhitungannya dengan akurasi di *classification report* yaitu dengan menjumlahkan prediksi yang benar lalu dibagikan dengan prediksi yang benar ditambah dengan prediksi yang salah. Tetapi, dalam *classification report* akurasi tersebut dibulatkan menjadi 2 bilangan dibelakang 0. Hal ini tentu akan menyebabkan bias ketika kita sedang memastikan model yang tepat untuk menghasilkan prediksi yang akurat. 
  
  Selanjutnya adalah implementasi kode *accuracy score*. Anda hanya perlu import library *sklearn.metric* lalu gunakan kode dibawah ini :
  
  *accuracy_score(actual,prediction)*
  
 ## Penutup
 
 Dengan berakhirnya penjelasan metrik, berakhir juga laporan ini. Terima kasih karena telah membaca laporan ini. Saya harap apa yang saya sudah sampaikan dapat menjadi bermanfaat bagi yang membaca laporan ini.
 


**---Terima Kasih---**


