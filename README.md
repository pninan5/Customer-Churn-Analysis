# Customer Churn Analysis (Video Streaming Subscription)

Predict whether a customer will churn (cancel) from a video streaming subscription, using interpretable models (Logistic Regression and Decision Tree) and evaluation focused on **sensitivity/recall** for churners.

---

## Project highlights
- **Dataset:** 243,787 customer records (train), 21 columns, clean (no missing values / duplicates)
- **Class imbalance:** ~18% churners
- **Models:** Logistic Regression (scaled + one hot encoding) and CART Decision Tree
- **Key technique:** Oversampling to 50/50 churn vs non churn in training to improve churn recall
- **Best model:** Logistic Regression (Accuracy 0.68, Precision 0.32, Sensitivity 0.70, ROC AUC 0.84)

---

## Business problem
Customer churn (attrition) directly impacts subscription revenue. Predicting churn enables targeted retention actions and prioritizes outreach for customers most at risk.

---

## Data
**Source:** Kaggle — “Predictive Analytics for Customer Churn” (streaming service)

**Target**
- `Churn` (1/0): whether the customer churned

**Numeric features (examples)**
- AccountAge, MonthlyCharges, TotalCharges
- ViewingHoursPerWeek, AverageViewingDuration
- ContentDownloadsPerMonth, SupportTicketsPerMonth, UserRating, WatchlistSize

**Categorical features (examples)**
- SubscriptionType, PaymentMethod, PaperlessBilling, ContentType
- MultiDeviceAccess, DeviceRegistered, GenrePreference
- Gender, ParentalControl, SubtitlesEnabled

---

## Methodology
### 1) Exploratory Data Analysis
- Distribution checks for categorical and numeric features
- Correlation heatmap (notable correlation: AccountAge and TotalCharges)
- Layered churn vs non churn histograms indicated strong signals from engagement and account age

### 2) Modeling
**Why interpretable models?**
The end goal is not just prediction — it is to explain *why* customers churn so a business can act.

- Logistic Regression
  - Min max scaling for numeric features to improve coefficient interpretability
  - One hot encoding (drop one level to avoid dummy trap)
- Decision Tree
  - CART with gini impurity
  - Grid search and early stopping to reduce overfitting

### 3) Handling class imbalance
Initial models had low churn sensitivity due to imbalance. We oversampled the training set to a **50/50** churn vs non churn split to increase sensitivity for churn cases.

---

## Results
### Logistic Regression
- **Before oversampling:** Accuracy 0.83, Precision 0.57, Sensitivity 0.12
- **After oversampling (preferred):** Accuracy 0.68, Precision 0.32, Sensitivity 0.70

Top drivers from coefficients (directional):
- Lower AccountAge (newer accounts churn more)
- Lower engagement: AverageViewingDuration, ViewingHoursPerWeek, ContentDownloadsPerMonth
- Higher MonthlyCharges
- Higher SupportTicketsPerMonth

### Decision Tree
- **Before oversampling:** Accuracy 0.82, Precision 0.53, Sensitivity 0.07
- **After oversampling:** Accuracy 0.64, Precision 0.29, Sensitivity 0.69

The churn rule region identified by the tree aligns with the same story:
younger accounts + low engagement + higher monthly charges are most likely to churn.

---

## Recommendations
- Prioritize retention for **newer accounts** showing **low engagement** signals
- Strengthen early lifecycle onboarding and content discovery to increase viewing hours and session duration
- Consider targeted discounts or offers for high monthly charge customers early in tenure
- Use support ticket volume as a risk flag for proactive support

---

## Repository contents
Notebooks
- `churn_data_exploration.ipynb`
- `churn_Logistic_Regression.ipynb`
- `churn_decision_tree.ipynb`

Docs
- `AML Project Proposal - churn dataset.pdf`
- `AML Group 5 Final Project.pdf`
- `AIML PPT.pptx`

Data dictionary
- `data_descriptions.csv`

Suggested repo structure
```
notebooks/
  churn_data_exploration.ipynb
  churn_Logistic_Regression.ipynb
  churn_decision_tree.ipynb
docs/
  AML Project Proposal - churn dataset.pdf
  AML Group 5 Final Project.pdf
  AIML PPT.pptx
data/
  data_descriptions.csv
```

---

## How to run
1. Create a Python environment (3.9+ recommended)
2. Install dependencies:
   - pandas, numpy
   - scikit-learn
   - matplotlib
   - imbalanced-learn
   - jupyter
3. Run notebooks in order:
   1) churn_data_exploration.ipynb  
   2) churn_Logistic_Regression.ipynb  
   3) churn_decision_tree.ipynb  

---

## Team
Valerie Garcia, Patrick Gervadis Ninan, Albin Poulose, Srivani Kakumani, Ashiq Mohammed Al Ameen
