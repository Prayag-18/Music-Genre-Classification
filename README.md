<p align="center">
<img src="musicGenereClassification.png?raw=true" alt="MusicGenreClassification" width="250">
</p>

# Music-Genre-Classification
-----------------------------------
<h2>Problem Being Addressed</h2>
<p>Music genre classification is a fundamental task in music information retrieval that assists in
organizing vast amounts of music data, recommending systems, and music analysis. This project
utilizes the GTZAN dataset, which is popular for benchmarking classification models but also known
for its challenges like noise in genre labels and data integrity.</p>

<h2>Relevant Literature</h2>
<p>Several studies have used the GTZAN dataset for genre classification, employing various machine learning techniques:</p>
<ul>
  <li>Tzanetakis and Cook (2002) introduced the GTZAN dataset in their foundational paper, highlighting its use for feature extraction and genre classification.</li>
  <li>Recent works have incorporated deep learning methods, notably Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs), demonstrating significant improvements in classification accuracy over traditional methods like Support Vector Machines (SVMs) and K-Nearest Neighbors (KNN).</li>
</ul>

<h2>Methodology</h2>
<h3>Data Preparation</h3>
<p>
  Data Loading: Each audio file from the GTZAN dataset, which consists of tracks each 30
seconds long, is loaded into the system.
Feature Extraction: From each audio track, Mel-frequency cepstral coefficients (MFCCs) are
extracted. MFCCs are chosen because they effectively represent the power spectrum of a
sound, capturing timbral and textual aspects which are important for distinguishing music
genres.
The calculation of the Mel-cepstrum is described by equation:

</p>
