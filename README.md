# Maternal Health Risk Prediction API

A FastAPI-based backend for predicting maternal health risk levels using vital parameters such as blood pressure, glucose level, body temperature, and heart rate.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/maternal-health-api.git
cd maternal-health-api
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv-server
venv-server\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> If `requirements.txt` is not present, create it using:

```bash
pip freeze > requirements.txt
```

---

## 🧪 Running the Server

Start the FastAPI server with:

```bash
uvicorn main:app --reload
```

By default, the app will be available at:

- API Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Root Route: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

---

## 📡 API Usage

### `POST /`

#### Request Body (JSON)

```json
{
  "Age": 27,
  "SystolicBP": 125,
  "DiastolicBP": 82,
  "BS": 6.3,
  "BodyTemp": 98.6,
  "HeartRate": 88
}
```

#### Response (JSON)

```json
{
  "Predicted Risk Level": "low risk",
  "Prediction Probability": 0.87
}
```

---

## 🌐 CORS Config

CORS is configured to allow:

- `http://localhost:3000`
- `https://maternal-health-risk.vercel.app`

Update in `main.py` if you deploy from a different frontend origin.

---

## 📁 Project Structure

```
maternal-health-api/
│
├── main.py                  # FastAPI app
├── model.py                 # ML model class
├── Maternal Health Risk Data Set.csv
├── requirements.txt
└── README.md
```

---

## 🧠 Model Info

The ML model uses:

- Logistic Regression
- Random Forest
- Gradient Boosting
- SVM

With preprocessing like:

- Age and BP categorization
- Imputation and scaling
- Label encoding of `RiskLevel`

---

## 🛠 Tech Stack

- **FastAPI**
- **Scikit-learn**
- **Pandas**
- **Uvicorn**
- **Pydantic**

---

## 👨‍⚕️ Disclaimer

This app is intended for educational purposes and **not** for real medical diagnosis or use in clinical settings.

---

## 📬 Contact

If you have suggestions or issues, feel free to open an issue or contact [prayashpriyansu27@gmail.com](mailto:prayashpriyansu27@gmail.com).
