# NHS Appointment No-Show Prediction

Predicting missed healthcare appointments (Did Not Attend / DNA events) using machine learning, with weather data integrated as a novel predictive feature.

## Overview

Missed NHS appointments cost the health service hundreds of millions of pounds annually and disrupt care for patients who need it most. This project builds an end-to-end machine learning pipeline to predict which appointments are at risk of being missed — giving clinics the ability to intervene before it happens (e.g. reminder calls, overbooking, rescheduling).

This was developed as my MSc Data Science dissertation at the University of Salford.

## Key Results

| Metric | Result |
|---|---|
| Records processed | 9,305,337 NHS appointment records (30 source files merged) |
| Weather data | 4 years of London weather records (2021–2024) merged on appointment date |
| Best F1 (Physical appointments) | 0.7599 (MLP) |
| Best F1 (Virtual appointments) | 0.8167 (Gradient Boosting and MLP, tied) |
| Models compared | Random Forest, Gradient Boosting, Decision Tree, Naive Bayes, MLPClassifier |

**Full model comparison (F1 score):**

| Model | Physical | Virtual |
|---|---|---|
| Gradient Boosting | 0.7336 | 0.8167 |
| Decision Tree | 0.6207 | 0.7839 |
| Naive Bayes | 0.5284 | 0.8149 |
| MLP | 0.7599 | 0.8167 |

*Note: high recall on several virtual-appointment models reflects class imbalance in the virtual appointment subset — a known limitation worth discussing honestly in interviews.*

## What Makes This Project Different

Most appointment no-show prediction models rely only on appointment metadata (time, day, specialty, history). This project integrates **historical weather data** (temperature, precipitation, humidity, cloud cover) as external features and tests whether environmental conditions affect attendance.

**Key finding:** weather-related features behaved differently across appointment modes — leading to separate modelling pipelines being built for physical and virtual appointments rather than a single combined model, since the underlying attendance drivers differ meaningfully between the two.

## Methodology

The project follows the **CRISP-DM** framework:

1. **Business Understanding** — define the cost of DNAs and the value of early prediction
2. **Data Understanding** — explore 9.3M NHS appointment records and merge with London weather data
3. **Data Preparation** — clean, merge, and engineer features:
   - Appointment mode (Face-to-Face, Home Visit, Telephone, etc.) and healthcare professional (HCP) type
   - Time between booking and appointment
   - Weather features: temperature, precipitation, humidity, cloud cover
4. **Modelling** — train and compare 5 classifiers, with separate pipelines for physical and virtual appointments
5. **Evaluation** — assess using accuracy, precision, recall, and F1 score
6. **Deployment Planning** — translate findings into operationally actionable recommendations

## Tech Stack

- **Python** — pandas, NumPy, scikit-learn
- **Visualisation** — Matplotlib, Seaborn
- **Modelling** — Random Forest, Gradient Boosting, Decision Tree, Naive Bayes, MLPClassifier

## Repository Structure

```
├── data/                  # Data loading and preprocessing notebooks
├── notebooks/
│   ├── 01_data_loading.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modelling.ipynb
│   └── 05_results.ipynb
├── README.md
└── requirements.txt
```

## Key Takeaways

- Built a full ML pipeline from raw data ingestion (30 separate CSV files, 9.3M+ rows) through to model comparison and evaluation
- Integrated an external data source (weather) not typically used in appointment prediction, and tested its effect across two appointment modes
- Compared 5 classification algorithms systematically using consistent evaluation metrics
- Identified and was transparent about a class imbalance limitation affecting recall on the virtual appointment subset

## Author

**Mohammed Adnan Englampurath**
MSc Data Science, University of Salford
[LinkedIn](https://linkedin.com/in/your-profile) · [GitHub](https://github.com/adnane777)
