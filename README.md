# 🎓 Student Progress Hub

An interactive Machine Learning-powered web application that predicts student exam performance and enables real-time what-if analysis.

---

## 🚀 Live Demo

🔗 https://progress-shine-board.lovable.app

---

## ✨ Features

* 📊 Predict student exam outcomes instantly
* 🔍 Compare multiple ML models (Linear, Ridge, Random Forest, Gradient Boosting)
* ⚡ Real-time **What-if analysis**
* 📈 Feature contribution insights
* 🌙 Dark/Light mode support
* 📱 Fully responsive (mobile + desktop)

---

## 🛠 Tech Stack

* **Frontend:** React (via Lovable AI)
* **Machine Learning:** Scikit-learn
* **Data Analysis:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Platform:** Lovable

---

## 📸 Screenshots

<img width="3572" height="722" src="https://github.com/user-attachments/assets/08bbdac7-650f-4d3b-bd8a-dad8f53d213e" />
<img width="1468" height="590" src="https://github.com/user-attachments/assets/ac34f081-e84b-41a6-87ba-387c2e0b766a" />
<img width="1174" height="722" src="https://github.com/user-attachments/assets/bb0b642c-6cbb-4d70-953c-2d5fb08e7495" />
<img width="1471" height="1531" src="https://github.com/user-attachments/assets/f4236817-636d-4b8a-88e9-0cd2e3382208" />
<img width="2072" height="721" src="https://github.com/user-attachments/assets/346d1b67-f2a9-4744-9f1d-54424419de28" />
<img width="2222" height="1820" src="https://github.com/user-attachments/assets/4143ccbb-8074-4a53-a634-a18c7cf0d9c4" />
<img width="1386" height="1230" src="https://github.com/user-attachments/assets/98071213-cc15-4aea-bda1-66a14510b2cb" />
<img width="1044" height="602" src="https://github.com/user-attachments/assets/3fde0c7a-4276-4d6a-809c-d005a1e1d00f" />
<img width="1104" height="602" src="https://github.com/user-attachments/assets/9633d44c-5ac3-485a-b2e0-20d8421ee085" />

---

## 🎯 Project Overview

This project predicts student exam scores using academic, behavioral, and lifestyle factors.
It combines **Machine Learning + Interactive UI** to provide insights into student performance.

---

## 📁 Project Structure

```
student_performance/
├── main.py
├── requirements.txt
├── README.md
│
├── src/
│   ├── generate_data.py
│   ├── utils.py
│   ├── eda.py
│   └── model.py
│
├── data/
├── models/
└── plots/
```

---

## 🚀 How It Works

### 1. Train Models

```bash
pip install -r requirements.txt
python main.py train
```

### 2. Predict Performance

```bash
python main.py predict
```

---

## 🧪 Models & Performance

| Model                 | MAE   | R²         |
| --------------------- | ----- | ---------- |
| **Linear Regression** | ~3.20 | **~0.918** |
| Ridge Regression      | ~3.20 | ~0.918     |
| Gradient Boosting     | ~4.25 | ~0.861     |
| Random Forest         | ~4.79 | ~0.823     |

---

## 🔬 Key Features Used

* Study hours
* Sleep hours
* Attendance
* Previous scores
* Assignments completed
* Stress level
* Tutoring
* Parental education
* Gender

---

## 📊 Insights

* Study hours & attendance strongly impact performance
* Previous scores are a major predictor
* Tutoring improves scores significantly
* Stress has a negative correlation

---

## ⚠️ Note

This project is built using Lovable AI.
The source code for the web interface is not exportable in the free version.
👉 Please use the **Live Demo link** to experience the full application.

---

## 💡 Future Improvements

* Backend deployment with real-time ML API
* User authentication system
* Personalized dashboards
* Database integration

---

## 👩‍💻 Author

**Ayushi Saini**
