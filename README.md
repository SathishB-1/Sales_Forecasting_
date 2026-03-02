# Sales & Demand Forecasting System

A machine learning–powered web application that forecasts future monthly retail sales using a pre-trained **SARIMA** time-series model. Designed to help businesses make informed decisions around **inventory management**, **staffing**, and **cash-flow planning** — without requiring any coding knowledge.

---

## Project Overview

This project follows a two-phase approach:

1. **Backend (Offline)** — Data is cleaned, aggregated, and used to train a SARIMA model inside a Jupyter Notebook. The trained model and processed data are saved as `.pkl` files.
2. **Frontend (Online)** — A Streamlit web app loads those saved files and provides an interactive dashboard where users can generate and explore sales forecasts.

---

## Project Structure

```
ML_TASK_1/
├── app.py                    # Streamlit web application (Frontend)
├── ML_Task_1.ipynb           # Model training & evaluation (Backend)
├── sarima_sales_model.pkl    # Pre-trained SARIMA model
├── monthly_sales.pkl         # Preprocessed monthly sales data
├── Sample - Superstore.csv   # Raw Kaggle Superstore dataset
├── requirements.txt          # Python dependencies
└── README.md
```

---

## Tools & Technologies

| Layer     | Tool / Library          | Purpose                                      |
|-----------|-------------------------|----------------------------------------------|
| Language  | Python 3.x              | Core programming language                    |
| Backend   | pandas, NumPy           | Data loading, cleaning, and aggregation      |
| Backend   | statsmodels (SARIMA)    | Time-series forecasting model                |
| Backend   | scikit-learn            | Model evaluation (MAE, RMSE, MAPE)           |
| Backend   | joblib                  | Saving and loading trained model & data      |
| Frontend  | Streamlit               | Interactive web dashboard                    |
| Frontend  | matplotlib              | Sales chart and forecast visualization       |
| Notebook  | Jupyter Notebook        | Training environment and experimentation     |

---

## Dataset

- **Source:** [Kaggle Superstore Dataset](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final)
- **Size:** 4,578 rows × 21 columns of US retail sales records
- **Date Range:** January 2014 – December 2017
- **Key column used:** `Sales`, aggregated monthly into `Total_Sales`

---

## Backend — Model Training (`ML_Task_1.ipynb`)

The Jupyter Notebook is the offline training phase of the project. It produces two `.pkl` files that are used by the web app.

### Workflow

#### 1. Data Loading & Cleaning
- Reads the Superstore CSV using `latin1` encoding.
- Removes duplicate rows, converts `Order Date` to datetime format, and filters out zero or negative sales values.

#### 2. Monthly Aggregation
- Groups individual transactions by **month-end** frequency using `pd.Grouper`.
- Sums the `Sales` column per month and renames it to `Total_Sales`.
- Result: a clean time-series with two columns — `Order Date` and `Total_Sales`.

#### 3. Feature Engineering
- Extracts `Month` and `Year` as separate columns.
- Creates a `Lag_1` feature (previous month's sales) for the Linear Regression baseline.
- Drops rows with `NaN` values created by the lag shift.

#### 4. Train / Test Split
- The **last 6 months** of data are used as the test set.
- All earlier data forms the training set.

#### 5. SARIMA Model
- Model configuration: `order=(1,1,1)` and `seasonal_order=(1,1,1,12)`.
- The `12`-month seasonal period captures **annual retail patterns** (e.g., holiday spikes).
- SARIMA was chosen because it handles both **trend** (long-term growth) and **seasonality** (recurring monthly patterns) — both of which are present in retail sales data.

#### 6. Model Evaluation

| Metric | Value     | Interpretation                                          |
|--------|-----------|---------------------------------------------------------|
| MAE    | ~12,784   | Average absolute difference between actual & predicted  |
| RMSE   | ~18,248   | Penalises large errors more than MAE                    |
| MAPE   | ~15.4%    | ~15% average error — acceptable for business forecasting|

#### 7. Saving Artifacts
- The trained model is saved to `sarima_sales_model.pkl` (~6 MB).
- The processed monthly sales DataFrame is saved to `monthly_sales.pkl`.
- These files are loaded at runtime by `app.py` — **no retraining needed in the app**.

---

## Frontend — Web Application (`app.py`)

The Streamlit app provides a clean, dark-themed dashboard that business users can interact with directly.

### Design
- **Dark theme** with a custom CSS overlay (`#0f1117` background, `#00d4ff` cyan accent).
- **Wide layout** to make full use of screen space.
- **Sidebar** for forecast controls (slider + generate button).
- **KPI metric cards** with gradient backgrounds for at-a-glance summaries.
- **Business Insight panel** with auto-generated text recommendations.

### App Sections

| Section | Description |
|---------|-------------|
| **Sidebar** | Slider to pick 3–12 months to forecast + "Generate Forecast" button |
| **KPI Cards** | Avg monthly forecast, % change vs historical avg, peak month, lowest month |
| **Sales Chart** | Historical (cyan, solid) vs Forecasted (orange, dashed) with shaded forecast zone, currency-formatted axes, and a clear divider line |
| **Forecast Table** | Month-by-month predicted sales formatted as `$XX,XXX.XX` |
| **Business Insight** | Auto-written recommendations for inventory, staffing, and cash-flow planning |

### Visualization Details
The chart is built with `matplotlib` and styled to match the dark theme:
- **Cyan line** (`#00d4ff`) — historical sales with a subtle fill
- **Orange dashed line** (`#ff9a3c`) — forecasted sales connected from the last historical point
- **Vertical dotted divider** — clearly marks where history ends and forecast begins
- **X-axis** — date labels every 3 months, rotated for readability
- **Y-axis** — currency-formatted values (e.g., `$25,000`)

---

## How to Run

### Prerequisites
Make sure Python 3.x is installed, then install all dependencies:

```bash
pip install -r requirements.txt
```

### Step 1 — (Optional) Retrain the Model
Open `ML_Task_1.ipynb` in Jupyter and run all cells to regenerate the `.pkl` files.
> The `.pkl` files are already included, so this step is only needed if you change the data or model configuration.

### Step 2 — Launch the Web App
```bash
streamlit run app.py
```

Open your browser at: **http://localhost:8501**

---

## End-to-End Data Flow

```
Raw CSV  →  Clean & Aggregate  →  SARIMA Training  →  Save .pkl Files
                                                              │
                                              Load in Streamlit App (app.py)
                                                              │
                                   KPI Cards + Chart + Table + Business Insight
```

---

## Why SARIMA?

SARIMA (Seasonal AutoRegressive Integrated Moving Average) was selected because:
- Retail sales data contains **both trend and seasonality**, which SARIMA is specifically designed to handle.
- The `seasonal_order=(1,1,1,12)` setting captures annual patterns (12-month cycle).
- It produces interpretable forecasts suitable for business communication.
- It achieves **~15% MAPE**, which is generally considered acceptable for sales forecasting.

---

## User Interface
![img alt](https://github.com/SathishB-1/Sales_Forecasting_/blob/3491532fb0867e957bef852ea37c6720cb3d212e/Screenshot%202026-03-02%20204754.png)
![img alt](https://github.com/SathishB-1/Sales_Forecasting_/blob/af7066e6f6b00d6f29fac148a34d5b0ed85d2b1b/Screenshot%202026-03-02%20204817.png)
![img alt](https://github.com/SathishB-1/Sales_Forecasting_/blob/1065452ab75c8102c4cac3a6a46854f20c30d3b6/Screenshot%202026-03-02%20204832.png)

## Future Enhancements

- Add **confidence intervals** (upper/lower bounds) to the forecast chart
- Support **product-wise** and **region-wise** forecasting breakdowns
- Improve accuracy using **Prophet** or **LSTM** (deep learning) models
- Allow users to **download forecast results** as a CSV file
- **Deploy** to Streamlit Cloud for wider business access

---

## Conclusion

This project shows how a machine learning time-series model can be packaged into a clean, user-friendly web application. The SARIMA model is trained offline, saved as an artifact, and served through a Streamlit dashboard — giving business stakeholders an interactive tool to explore future sales predictions with no coding required.
