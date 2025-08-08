# Payday Loan Defaulter Prediction – Deploy with GUI

This project predicts whether a payday loan applicant is likely to **default** using a trained machine learning model.  
It includes a **Streamlit-based graphical interface** for easy interaction and deployment.

---

## 📌 Features
- **User-friendly GUI** built with [Streamlit](https://streamlit.io/)
- Takes applicant details as input (numerical sliders & dropdown menus)
- Loads a pre-trained **LightGBM model** for prediction
- Displays results instantly with probability scores
- Supports both **local testing** and **online deployment** (e.g., Streamlit Cloud)

---

## 🗂 Project Structure
```
Payday-Loan-Defaulter-Prediction/
│
├── app.py               # Main Streamlit app
├── mypipes.py           # Custom preprocessing pipeline
├── tuned_lgbm_model.pkl # Trained LightGBM model
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

---

## ⚙️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/nikhileshnarkhede/Payday-Loan-Defaulter-Prediction.git
cd Payday-Loan-Defaulter-Prediction
git checkout Depoly_with_GUI
```

2. **Create & activate a virtual environment** (optional but recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

---

## ▶️ Running the App

Run the following command in the terminal:
```bash
streamlit run app.py
```
This will start a local server and open the app in your default browser (usually at `http://localhost:8501`).

---

## 📊 Model Information
- **Algorithm:** LightGBM Classifier
- **Training:** Based on applicant loan & demographic data
- **Preprocessing:** Custom pipeline (`mypipes.py`) for scaling and encoding
- **Output:** Binary classification — `Defaulter` or `Non-Defaulter` with probability scores

---

## 🚀 Deployment
You can deploy this app to:
- **[Streamlit Cloud](https://streamlit.io/cloud)** (free & simple)
- **Heroku**, **Azure**, or **AWS** (requires extra config)

For **Streamlit Cloud**:
1. Push this branch (`Depoly_with_GUI`) to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io/)
3. Connect your GitHub repo and select `app.py` as the entry point
4. Done! Your app will be live with a public URL.

---

## 📜 License
This project is licensed under the MIT License — feel free to use and modify it.

---

## ✍️ Author
**Nikhilesh Narkhede**  
📧 Contact: [Your Email Here]  
🔗 GitHub: [nikhileshnarkhede](https://github.com/nikhileshnarkhede)
