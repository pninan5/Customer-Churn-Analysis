# Customer Churn Analysis (Video Streaming Subscription)

## Overview
This project predicts whether a customer will churn from a video streaming subscription service. The focus is on building interpretable models that can both identify at risk customers and explain what drives churn, so that product and retention teams can take action.

## Problem statement
Customer churn is when a subscriber cancels their relationship with a service. Predicting churn enables proactive retention strategies, better customer experience, and more efficient resource allocation.

## Dataset
Source: Kaggle dataset titled Predictive Analytics for Customer Churn (streaming service).
Size and shape:
• Training set: 243,787 customer records with 21 columns, including the churn target  
• Test set: features only (no churn column), not used for modeling  
Data quality:
• No missing values and no duplicate records reported

Target:
• Churn (1 or 0), whether the customer churned

Feature groups:
• Numeric: AccountAge, MonthlyCharges, TotalCharges, ViewingHoursPerWeek, AverageViewingDuration, ContentDownloadsPerMonth, SupportTicketsPerMonth, UserRating, WatchlistSize  
• Categorical: SubscriptionType, PaymentMethod, PaperlessBilling, ContentType, MultiDeviceAccess, DeviceRegistered, GenrePreference, Gender, ParentalControl, SubtitlesEnabled

## Approach
1. Exploratory data analysis
• Compared churned vs retained customers across categorical and numeric variables  
• Built a correlation heatmap to understand relationships between features  
• Found engagement metrics and account age were most informative for churn risk signals

2. Modeling choices
I prioritized interpretability to connect model outputs to actionable recommendations:
• Logistic Regression  
• Decision Tree (CART with gini impurity)

3. Preprocessing and evaluation
• Train test split: 80 percent train, 20 percent test  
• One hot encoding for categorical variables  
• Min max scaling for numeric features in logistic regression to improve coefficient interpretability  
• Class imbalance handling: oversampling the training data to a 50 50 churn vs non churn ratio to improve sensitivity (recall) for churn cases

## Results
Baseline models on original data tended to have low sensitivity, which is expected with an imbalanced target.

Preferred model: oversampled Logistic Regression  
• Accuracy: 0.68  
• Precision: 0.32  
• Sensitivity (recall): 0.70  
• ROC AUC: 0.75

Oversampled Decision Tree  
• Accuracy: 0.64  
• Precision: 0.29  
• Sensitivity (recall): 0.69  
• ROC AUC: 0.72

## Key drivers of churn
Across models, the strongest churn indicators were consistent:
• Lower AccountAge (newer accounts churn more)  
• Lower engagement: AverageViewingDuration, ViewingHoursPerWeek, ContentDownloadsPerMonth  
• Higher MonthlyCharges  
• Higher SupportTicketsPerMonth

## Business recommendations
• Focus retention campaigns on newer subscribers with low engagement signals  
• Improve early lifecycle onboarding and content discovery to increase viewing time and downloads  
• Consider targeted discounts or trials for high monthly charge customers early in their lifecycle  
• Use support ticket frequency as a risk flag for proactive outreach

## Repository contents
Core notebooks:
• churn_data_exploration.ipynb  
• churn_Logistic_Regression.ipynb  
• churn_decision_tree.ipynb  

Supporting files:
• data_descriptions.csv  
• AML Project Proposal (pdf)  
• Final Project Report (pdf)  
• Presentation deck (pptx)

Suggested organization:
docs/
  AML Project Proposal - churn dataset.pdf
  AML Group 5 Final Project.pdf
  AIML PPT.pptx
notebooks/
  churn_data_exploration.ipynb
  churn_Logistic_Regression.ipynb
  churn_decision_tree.ipynb
data/
  data_descriptions.csv

## How to run
1. Create a Python environment (3.9+ recommended)
2. Install dependencies
3. Run notebooks in this order:
   1) churn_data_exploration.ipynb
   2) churn_Logistic_Regression.ipynb
   3) churn_decision_tree.ipynb

Typical dependencies:
• pandas, numpy
• scikit learn
• matplotlib
• imbalanced learn (for oversampling)
• jupyter

## Credits
Group 5: Valerie Garcia, Patrick Gervadis Ninan, Albin Poulose, Srivani Kakumani, Ashiq Mohammed Al Ameen
