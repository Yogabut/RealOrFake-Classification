# 📰 Real or Fake News Classification using LSTM

This project aims to classify news articles as **Real** or **Fake** using a deep learning model based on **Long Short-Term Memory (LSTM)**, a type of recurrent neural network (RNN) well-suited for natural language processing tasks.

## 📌 Overview

Fake news has become a serious issue in the digital era, influencing public opinion and spreading misinformation rapidly. In this project, we develop an LSTM-based text classification model that can distinguish between real and fake news articles.

## 🚀 Project Objectives

* Preprocess news data (title/content) for machine learning.
* Build and train an LSTM model using TensorFlow/Keras.
* Evaluate model performance using metrics such as accuracy, precision, recall, and F1-score.
* Provide an interactive prediction function for user-input news.

## 🧠 Tech Stack

* **Programming Language**: Python
* **Libraries**:

  * `TensorFlow / Keras`
  * `NumPy`
  * `Pandas`
  * `Matplotlib / Seaborn`
  * `NLTK / SpaCy` (for preprocessing)
  * `Scikit-learn` (for evaluation)

## 📂 Dataset

* **Source**: [Fake and Real News Dataset – Kaggle](https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news/data)
* **Content**:

  * `title`: The title of the news article.
  * `text`: Full content of the news article.
  * `label`: `FAKE` or `REAL`

## ⚙️ Preprocessing Steps

* Lowercasing text
* Removing punctuation, stopwords, and special characters
* Tokenization and padding
* Word embedding using Keras `Tokenizer` and `Embedding` layer

## 🏗️ Model Architecture

```plaintext
Embedding Layer
↓
LSTM Layer
↓
Dropout Layer
↓
Dense Output Layer (Sigmoid Activation)
```

* **Loss Function**: Binary Crossentropy
* **Optimizer**: Adam
* **Metrics**: Accuracy, Precision, Recall

## 📈 Evaluation Metrics

* **Accuracy**
* **Precision**
* **Recall**
* **F1-Score**
* **Confusion Matrix**
* **ROC-AUC Curve**

## 📊 Example Results

| Metric    | Value |
| --------- | ----- |
| Accuracy  | 0.92  |
| Precision | 0.91  |
| Recall    | 0.93  |
| F1-Score  | 0.92  |


## 🔍 Future Improvements

* Incorporate attention mechanisms for better context understanding
* Use BERT or Transformer-based models for improved accuracy
* Deploy the model using Flask/Streamlit for real-time prediction

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.


