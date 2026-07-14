
# 👗 StyleSense AI

### AI-Powered Fashion Recommendation System

StyleSense AI is a Machine Learning-based fashion recommendation system that helps users discover similar fashion products and generate outfit suggestions.

The application uses **TF-IDF Vectorization** and **Cosine Similarity** to recommend products based on textual product information. It also provides an AI Outfit Stylist that suggests outfits according to the user's **occasion, budget, and preferred colour**.

---

## ✨ Features

- 🔍 Search products by name
- 🏷 Filter by Brand
- 📂 Filter by Category
- 💰 Budget-based filtering
- 🎨 Colour selection
- 👗 AI Product Recommendation
- 👚 AI Outfit Stylist
- ⭐ Product Ratings
- 🖼 Product Images
- 📱 Interactive Streamlit Web App

---

## 🧠 Machine Learning Techniques

- TF-IDF (Term Frequency–Inverse Document Frequency)
- Cosine Similarity
- Content-Based Recommendation System

---

## 🛠 Tech Stack

| Category | Technologies |
|----------|--------------|
| Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Web Framework | Streamlit |
| Model Storage | Pickle |

---

## 📂 Dataset

The project uses a fashion dataset containing **14,268 products**.

Each product contains:

- Product Name
- Brand
- Category
- Colour
- Price
- Rating
- Product Image URL
- Description

---

## ⚙️ Project Workflow

```
Fashion Dataset
       │
       ▼
Data Cleaning & Preprocessing
       │
       ▼
Feature Engineering
       │
       ▼
TF-IDF Vectorization
       │
       ▼
Cosine Similarity Matrix
       │
       ▼
Recommendation Engine
       │
       ▼
Streamlit Web Application
```

---

## 📁 Project Structure

```
StyleSense-AI
│
├── app
│   └── app.py
│
├── Data
│
├── models
│   ├── fashion_df.pkl
│   └── similarity.pkl
│
├── notebook
│
├── images
│
├── requirements.txt
├── README.md
└── .streamlit
```

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/Aditi4131/StyleSense-AI.git
```

Move into the project directory

```bash
cd StyleSense-AI
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app/app.py
```

---

## 📸 Application Preview

### 🏠 Home Page
![Home Page](../images/homepage.png)


---

### 🤖 AI Recommendations

![Recommendations](../images/Recommendations.png)


---

### 👗 Outfit Generator

![Outfit Generator](../images/Outfit%20Generator.png)

---

### 🎯 Filters

![Filters](../images/Filters.png)


---

## 📈 Future Enhancements

- User Authentication
- Wishlist Feature
- Deep Learning Recommendation Model
- Collaborative Filtering
- Fashion Trend Prediction
- Voice-Based Search
- Fashion Chatbot

---

## 👩‍💻 Developer

**Aditi Kumari**

B.Tech Computer Science & Engineering (Data Science)

Lovely Professional University

---

⭐ If you found this project useful, consider giving it a star on GitHub.