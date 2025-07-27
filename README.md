
# 🚦 Road Accident Severity Classification

This project predicts the **severity of road accidents** using machine learning techniques, helping authorities respond effectively and take preventive measures.

🔗 **Live Demo**: [View on Render](https://github.com/suma-sree/mini_project)

---

## 📌 Objective

To classify road accidents as *Minor*, *Serious*, or *Fatal* using structured accident data from India.

---

## 🧠 Workflow Overview

### 🔧 Preprocessing
- Handled missing values
- Encoded categorical variables using `LabelEncoder`
- Extracted time-based features (e.g., hour of accident)
- Handled class imbalance using `SMOTE`

### 🎯 Models Used
- ✅ Decision Tree (Best performer)
- ✅ Support Vector Machine (SVM)
- ✅ Logistic Regression

### 📊 Evaluation
- Accuracy Score
- Confusion Matrix
- Visualizations using Matplotlib & Seaborn

### 💾 Deployment
- Trained model saved using `pickle`
- Web app built with Flask and deployed on **Render**

---

## 📁 Dataset

- **Name**: RTA Dataset
- **Source**: [Kaggle - Road Accidents Severity Dataset](https://www.kaggle.com/datasets)
- **Path**: `/kaggle/input/road-accidents-severity-dataset/RTA Dataset.csv`
- **Entries**: 12,316 rows, 32 columns
- **Target Column**: `Accident_severity`

---

## 🛠️ Tech Stack

| Layer       | Tools Used                        |
|-------------|-----------------------------------|
| Language    | Python                            |
| Libraries   | pandas, numpy, sklearn, imblearn  |
| Visualization | matplotlib, seaborn             |
| Model Saving | pickle                           |
| Deployment  | Flask, HTML/CSS, Render           |

---

## 📂 Project Structure

```
Road_Accident_Prediction/
├── data/
│   └── RTA Dataset.csv
├── model/
│   └── severity_model.pkl
├── templates/
│   ├── home.html
│   ├── form.html
│   └── result.html
├── static/
│   └── main.css
├── app.py
├── README.md
└── requirements.txt
```

---

## ▶️ Run Locally

```bash
git clone https://github.com/suma-sree/mini_project.git
cd mini_project

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
python app.py
```

Visit `http://localhost:5000` in your browser.

---

## 🙋‍♀️ Developed By

- [Suma Sree](https://github.com/suma-sree)

---

## 🚀 Future Enhancements

- Integrate Geolocation & Weather APIs
- Add SMS alert system for fatal crashes
- Save reports to a cloud database (e.g., MongoDB)
