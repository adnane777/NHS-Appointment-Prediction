# NHS Appointment No-Show Prediction

Can appointment details and the weather predict who misses an NHS GP appointment (a "DNA", Did Not Attend)? This was my MSc Data Science dissertation at the University of Salford.

## Key results

| | |
|---|---|
| Records processed | 9,305,337 NHS appointment records, merged from 30 source files |
| Weather data | 4 years of London daily weather (2021–2024): temperature, precipitation, humidity, cloud cover |
| Modelling data | London regions only: 1,771,609 rows after the weather join; 1,186,775 Attended/DNA appointments |
| Design | Separate pipelines for physical (face-to-face, home visit) and virtual (telephone, video) appointments |
| Models | Random Forest, Gradient Boosting, Decision Tree, Naive Bayes, MLP |

**The honest headline: no model clearly beat a "predict everyone attends" baseline.** Most appointments are attended (about 61% of physical and 69% of virtual ones here), and the best-scoring models learned to predict "Attended" every time. That is why their recall is 1.0 and their F1 matches the baseline.

### Full results (test set; the positive class is *Attended*)

| Model | Segment | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| *Baseline: predict all Attended* | Physical | 0.613 | 0.613 | 1.000 | 0.760 |
| Random Forest | Physical | 0.528 | 0.600 | 0.686 | 0.640 |
| Gradient Boosting | Physical | 0.610 | 0.631 | 0.875 | 0.734 |
| Decision Tree | Physical | 0.522 | 0.604 | 0.639 | 0.621 |
| Naive Bayes | Physical | 0.572 | 0.812 | 0.392 | 0.528 |
| MLP | Physical | 0.613 | 0.613 | 1.000 | 0.760 |
| *Baseline: predict all Attended* | Virtual | 0.690 | 0.690 | 1.000 | 0.817 |
| Random Forest | Virtual | 0.661 | 0.685 | 0.944 | 0.794 |
| Gradient Boosting | Virtual | 0.690 | 0.690 | 1.000 | 0.817 |
| Decision Tree | Virtual | 0.650 | 0.682 | 0.921 | 0.784 |
| Naive Bayes | Virtual | 0.689 | 0.691 | 0.993 | 0.815 |
| MLP | Virtual | 0.690 | 0.690 | 1.000 | 0.817 |

A grid-searched MLP (24 settings, 3-fold cross-validation) did not change the picture.

## What I learned

- **F1 on the majority class can hide a model that has learned nothing.** The useful question is how many missed appointments a model catches, which means measuring recall and precision on the DNA class.
- **The weather signal is weak next to class imbalance.** The weather features behaved differently for physical and virtual appointments, which justified the split design, but they weren't enough on their own to separate DNAs from attendances.
- **Next steps:** class weighting or resampling, DNA-class metrics and precision-recall curves, threshold tuning, and patient-level history features (previous DNAs are a strong predictor in published studies), which this aggregated dataset doesn't include.

## Method (CRISP-DM)

1. **Business understanding:** missed appointments cost the NHS money and clinic time, so early warning would let clinics send reminders or rebook.
2. **Data understanding:** explored the 9.3M national records, then focused on London to match the weather data.
3. **Data preparation:** merged and validated 30 CSV files, joined weather on appointment date, and engineered features: appointment mode, healthcare professional type, booking lead time, day of week, and a mode × time interaction.
4. **Modelling:** five classifiers in scikit-learn pipelines with one-hot encoding, run separately for physical and virtual appointments.
5. **Evaluation:** accuracy, precision, recall and F1, compared against the all-Attended baseline.

## Files

- `NHS_Appointment_Prediction.ipynb`: the full pipeline, from loading and merging through EDA and modelling to the results charts

The raw NHS and weather CSVs aren't included because of their size. The NHS data is the public *Appointments in General Practice* sub-ICB dataset from NHS Digital.

## Tech stack

Python · pandas · NumPy · scikit-learn · Matplotlib · Seaborn

## Author

**Mohammed Adnan Englampurath**, MSc Data Science, University of Salford
[LinkedIn](https://www.linkedin.com/in/madnanep) · [GitHub](https://github.com/adnane777)
