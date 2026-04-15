# Using Machine Learning Models to Predict Genitourinary Involvement Among Gastrointestinal Stromal Tumour Patients

- Decision: Reject
- Scores: 1, 1, 1, 1

## Abstract
Gastrointestinal stromal tumors (GISTs) can lead to involvement of other organs, including the genitourinary (GU) system. Machine learning may be a valuable tool in predicting GU involvement in GIST patients, and thus improving prognosis. This study aims to evaluate the use of machine learning algorithms to predict GU involvement among GIST patients in a specialist research center in Saudi Arabia. We analyzed data from all patients with histopathologically confirmed GIST at our facility from 2003 to 2020. Patient files were reviewed for the presence of renal cell carcinoma, adrenal tumors, or other genitourinary cancers. Three supervised machine learning algorithms were used: Logistic Regression, XGBoost Regressor, and Random Forests. A set of variables, including independent attributes, was entered into the models. A total of 170 patients were included in the study, with 58.8% (n=100) being male. The median age was 57 (range 9-91) years. The majority of GISTs were gastric (60%, n=102) with a spindle cell histology. The most common stage at diagnosis was T2 (27.6%, n=47) and N0 (20%, n=34). Six patients (3.5%) had GU involvement. The Random Forest model achieved the highest accuracy with 97.1%. Our study suggests that the Random Forest model is an effective tool for predicting GU involvement in GIST patients. Larger multicenter studies, utilizing more powerful algorithms such as deep learning and other artificial intelligence subsets, are necessary to further refine and improve these predictions.

## Human Reviews

## Human Reviewer 1

### Rating
1: strong reject

### Rating Number
1

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper evaluates the use of machine learning algorithms to predict GU involvement among GIST patients in a specialist research center in Saudi Arabia. A total of 170 patients were included in the study, and traditional machine-learning models were applied (Logistic regression, XGBoost, and Random Forests). The Random Forest model achieved the highest accuracy with 97.1%.

### Strengths
1) Readability. The paper is simple to read. The problem is clearly explained. 

2) Novelty of the application. Few papers applied machine-learning models to this type of problem.

### Weaknesses
The paper has several weaknesses and limitations related to the application of machine learning models, from data description to the application of the models. I could not understand the positive and negative class. In terms of model application, I could not find the hyperparameters used in this paper (either a search of them). There is also a lack of results, considering that not all of them are shown in the paper. Also, you have to justify your data separation. Considering that is a small dataset, it is better to use k-fold cross-validation. 

In general, the paper must be largely improved by describing all the processes that allow me to observe that machine learning models were applied correctly.

### Questions
Not applicable.

### Soundness
1 poor

### Presentation
1 poor

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
1: strong reject

### Rating Number
1

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper does not adhere to the ICLR format and is not aligned with the area of interest of ICLR. This work should have been desk-rejected.

### Strengths
This paper does not adhere to the ICLR format and is not aligned with the area of interest of ICLR.

### Weaknesses
This paper does not adhere to the ICLR format and is not aligned with the area of interest of ICLR.

### Questions
None.

### Soundness
1 poor

### Presentation
1 poor

### Contribution
1 poor

---

## Human Reviewer 3

### Rating
1: strong reject

### Rating Number
1

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper uses logistic regression, XGBoost and random forest to predict GIST.

### Strengths
The paper works on a real-world dataset.

### Weaknesses
The paper is at a very immature stage. Here are some detailed comments:
   -- Lack of validation. The test set only contains test samples
   -- No contribution. The paper only presents several supervised learning algorithms on a dataset.

### Questions
Why there are only 2 cancer patients in the test set according to Figure 1?

### Soundness
1 poor

### Presentation
1 poor

### Contribution
1 poor

---

## Human Reviewer 4

### Rating
1: strong reject

### Rating Number
1

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This work uses machine learning models to predict genitourinary system involvement in patients with gastrointestinal stromal tumors. Employing various supervised learning methods including logistic regression, XGBoost, and Random Forest across a dataset of 170 patients from a specific hospital, the researchers achieved impressive accuracy rates around 97%, showcasing the potential of these models in medical classification problems.

### Strengths
This paper identifies an important understudied problem in the medical literature. It provides a good motivation/explanation of the underlying issues related to gastrointestinal stromal tumors and the need for ML in this context.

### Weaknesses
- *Insufficient Data*: The work uses 170 example and a highly imbalanced distribution of the positive class, consisting of only 6 positive samples. This limitation significantly constrains the ability to train and validate the machine learning models effectively.
- *Class Imbalance and Evaluation Metric*: The severe class imbalance in the dataset makes accuracy a poor choice for an evaluation metric, as it can provide a misleadingly high performance measure.
- *Limited Technical Contribution and Originality*: The application of common machine learning methods without any notable innovation or significant technical advancement limits the paper's contribution to the broader machine learning community.

### Questions
N/A

### Soundness
2 fair

### Presentation
1 poor

### Contribution
1 poor
