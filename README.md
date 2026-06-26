# NHS Appointment No-Show Prediction

Predicting missed healthcare appointments (Did Not Attend / DNA events) using machine learning, with weather data integrated as a novel predictive feature.

## Overview

Missed NHS appointments cost the health service hundreds of millions of pounds annually and disrupt care for patients who need it most. This project builds an end-to-end machine learning pipeline to predict which appointments are at risk of being missed — giving clinics the ability to intervene before it happens (e.g. reminder calls, overbooking, rescheduling).

This was developed as my MSc Data Science dissertation at the University of Salford.

## Key Results

| Metric | Result |
|---|---|
| Records processed | 9.3M appointments (filtered to 1.2M for modelling) |
| Best model | Gradient Boosted Trees |
| F1 score (virtual appointments) | 0.82 |
| AUC | 0.70 |
| Models compared | Random Forest, Gradient Boosted Trees, Decision Tree, Naive Bayes, MLPClassifier |

## What Makes This Project Different

Most appointment no-show prediction models rely only on appointment metadata (time, day, specialty, history). This project integrates **historical weather data** as an external feature and tests whether environmental conditions affect attendance.

**Key finding:** weather meaningfully affects attendance for **physical** appointments but has almost no effect on **virtual** appointments — which led to building separate models for each appointment mode rather than a single combined model.

## Methodology

The project follows the **CRISP-DM** framework:

1. **Business Understanding** — define the cost of DNAs and the value of early prediction
2. **Data Understanding** — explore 9.3M NHS appointment records and merge with London weather data
3. **Data Preparation** — clean, filter, and engineer features:
   - Booking-to-appointment lead time
   - Day-of-week effects
   - Mode-time interaction terms
   - Rolling weather averages
4. **Modelling** — train and compare 5 classifiers, with separate pipelines for physical and virtual appointments
5. **Evaluation** — assess using F1, AUC, precision/recall, and feature importance
6. **Deployment Planning** — translate findings into operationally actionable recommendations (e.g. targeted reminder strategies for high-risk physical appointments on poor-weather days)

## Tech Stack

- **Python** — pandas, NumPy, scikit-learn
- **Visualisation** — Matplotlib, Seaborn
- **Modelling** — Random Forest, Gradient Boosted Trees, Decision Tree, Naive Bayes, MLPClassifier

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

- Demonstrated that external, non-obvious features (weather) can meaningfully improve prediction when applied to the right subset of data
- Built separate models for physical vs. virtual appointments after identifying that a single combined model masked an important interaction effect
- Translated technical model output into a business-ready recommendation rather than stopping at model accuracy

## Author

**Mohammed Adnan Englampurath**
MSc Data Science, University of Salford
[LinkedIn](https://linkedin.com/in/your-profile) · [GitHub](https://github.com/adnane777)
