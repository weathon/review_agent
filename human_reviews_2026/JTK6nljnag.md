# Scent of Health (S-O-H): Olfactory Multivariate Time-Series Dataset for Non-Invasive Disease Screening

- Decision: Reject
- Scores: 6, 4, 6, 4

## Abstract
Exhaled breath analysis offers a promising, non-invasive alternative to traditional medical diagnostics. Electronic nose (eNose) sensors enable low-cost screening but progress is limited by small, site-specific datasets and sensor-specific temporal artifacts like baseline drift. We introduce Scent of Health (S-O-H), a large clinical eNose dataset with 1,027 patients across eight diagnostic groups, and reframe breath diagnosis as a realistic multivariate time series task. Our contribution includes curated temporal splits that control for sensor drift and mimic real-world deployment. We provide a reproducible benchmark with classical feature-based models, convolutional neural networks, and specialized time series classifiers. Our results demonstrate the dataset's utility, with methods achieving promising performance (e.g., ROC AUC up to 0.75 for lung cancer and 0.70 for hepatitis) while revealing significant gaps in robustness under drift and limited data. By releasing the dataset, splits, and code, we provide a foundational resource to advance research into robust, generalizable machine learning for clinical breathomics.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Scent of Health (S-O-H), an openly available olfactory multivariate time-series dataset for disease screening using electronic-nose (eNose) sensing.
It comprises 1,027 patient breath samples across eight diagnostic categories (healthy and seven disease conditions such as lung cancer, diabetes, and hepatic disorders), collected with a 17-sensor ZnO-based printed metal-oxide array. Each sensor produces 15-minute temporal recordings under controlled environmental conditions (temperature, humidity, CO₂, pressure).

The authors frame breath analysis as a multivariate time-series classification problem and benchmark:

CNN-based odor maps, treating sensor readings as spatiotemporal images, and

Feature-based ensemble models (e.g., CatBoost) using kinetic and statistical descriptors.

Temporal train/test splits emulate sensor drift and deployment chronology. CNNs achieve ROC-AUC ≈ 0.71 for some diseases, while feature-based models perform better for metabolic disorders.
All preprocessing scripts, splits, and metadata are planned for public release.

### Strengths
*High clinical and societal relevance:* Demonstrates non-invasive disease screening through breath sensing — an emerging, patient-friendly diagnostic approach.

*Open, reproducible resource:* One of the few publicly accessible, sensor-level olfactory datasets with detailed documentation and ethical clearance.

*Temporal realism:* Time-ordered splits realistically capture drift and device aging, a key challenge in olfaction research.

*Comprehensive methodology:* Clear sensor fabrication details, sampling setup, and preprocessing pipeline.

*Balanced baselines:* Includes both classical ML and deep architectures for fair benchmarking.

*Community orientation:* Code and data release encourage standardized evaluation in breathomics.

### Weaknesses
Moderate algorithmic novelty: CNN and CatBoost baselines are standard; no architecture tailored for drift-robust time-series olfaction is proposed.

*Limited diagnostic performance:* Best ROC-AUC ≈ 0.71 — below thresholds for practical screening.

*Single-center collection:* All samples originate from one institution, limiting generalization.

*Dataset imbalance:* Certain disease classes are underrepresented or temporally clustered.

*Drift unmodeled:* Drift is acknowledged but not explicitly quantified or corrected algorithmically.

*Missing broader dataset context:* The paper does not situate S-O-H relative to other open olfactory resources such as Olfaction-Vision-Language, SmellNet, OlfactionBase, M2OR, or Multi-Labelled SMILES Odors. A short comparison or feature-coverage table would clarify its unique positioning.

### Questions
This work signals a quiet but significant evolution — the maturation of machine olfaction from isolated prototypes toward shared, structured data ecosystems.
S-O-H transforms something intangible — the scent of human metabolism — into analyzable, time-bound data, bridging clinical sensing and AI reproducibility.

Its deeper value lies not in accuracy metrics but in infrastructure: it defines how the field can speak a common language. By quantifying sensor drift, linking breath profiles to diseases, and releasing data transparently, the authors lay groundwork for a research culture where olfactory AI becomes testable, comparable, and cumulative.

Yet, the study also exposes a limitation fundamental to the domain: olfactory signals are memory-laden. They encode environmental and temporal context as much as disease.
Recognizing that entanglement — and designing models resilient to it — will determine the future of digital smell analytics.

While algorithmic novelty is limited, the dataset’s openness, design realism, and reproducibility make it a meaningful benchmark contribution. To strengthen impact, the authors should:

1. Discuss overlap with existing open olfactory datasets.
2. Add metrics or models addressing sensor drift quantitatively.
3. Outline dataset maintenance and update strategy for long-term usability

I expect the authors to defend or rebut the points in the weakness section during the rebuttal phase. However, without that also this paper can be accepted as is as a Data Track / Benchmark Paper

**Update**:  After some internal discussion, I recognize that dataset-focused contributions should also be evaluated for strong methodological novelty. Accordingly, I have adjusted my scoring.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces Scent of Health (S-O-H) — a new multivariate time-series dataset for breath-based disease screening using electronic-nose (eNose) sensors. It represents one of the first large-scale efforts (1,027 participants across eight disease groups) to systematize olfactory sensing data and frame odor diagnosis as a time-series learning task. The authors provide baseline benchmarks using classical feature-based methods and CNNs, highlighting challenges such as sensor drift and temporal data splits.

### Strengths
1. New dataset contribution, the paper proposes olfactory or breathomics datasets limited exist for ML research.
2. The dataset covers over 1,000 participants with multiple diagnostic categories, enabling broader modeling and validation.
3. Provides baseline results that expose important challenges like drift, temporal bias, and limited labels.
4. Detailed sensor and protocol description.

### Weaknesses
1. Incomplete benchmarking scope: Evaluations use relatively simple CNN and CatBoost models but omit established or state-of-the-art time-series methods such as Transformers, InceptionTime, TS2Vec, Autoformer, Mamba, or masked-autoencoder approaches.

2. Lack of generalization studies: No exploration of cross-cohort, cross-time, or cross-sensor generalization, which is essential for deployment realism.

3. Limited statistical validation: The paper lacks multiple runs, variance reporting, or ablations to confirm robustness.

4. Clinical claims ahead of evidence: While non-invasive screening is promising, reported ROC-AUCs (~0.7) and limited sampling per subject make clinical readiness uncertain. More clinical valid evaluation metrics can be also reported.

### Questions
see weakness

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Scent of Health (S-O-H) fills a notable gap in olfactory/breathomics ML with a sizable, carefully described, drift-aware collection and executable baselines; code/data availability (even if contingent) and a realistic temporal-split protocol raise its community value. However, the authors should should strengthen the quantitative evaluation (CIs, PR-AUC, calibration), analyze demographic/environmental confounds, report the missing self-supervised baselines, and document feature/channel selection strictly within training folds.

### Strengths
The strengths of the paper can be summarised as follows,

- Substantial, clinician-relevant dataset + clear problem framing. 1,027 participants across eight diagnostic groups (healthy + seven ICD-10 disease cohorts) measured on a printed metal-oxide eNose array with auxiliary environmental sensors. Single sample per participant helps avoid subject-level leakage.
- Drift-aware, temporally stratified evaluation. Collection spans 11 weeks; the paper proposes per-disease week-wise train/test splits to reduce leakage from sensor drift and temporally concentrated sampling, which is an important and realistic protocol choice for deployment settings.
- Hardware/system description supports reproducibility. Sensor fabrication, electrode layout, thermal control, sampling rate, storage schema and bagging protocol are explained with enough specificity to replicate data acquisition.
- Baseline breadth across modeling styles. Two families of baselines one with CNN treating the series as images after smoothing/normalization/feature aggregation; while the second is tabular CatBoost with kinetic-curve fits and basic statistics.

### Weaknesses
The weaknesses of the paper can be summarised as follows,

- Demographic confounding risk not analyzed. Disease cohorts differ markedly in age/sex (e.g., lung cancer and COPD groups older and more male than healthy). Without demographic balancing/adjustment or reporting of models with/without these covariates, results may reflect confounds rather than VOC signature.
- Potential internal inconsistencies / incomplete baselines. The text references contrastive/self-supervised baselines, but the main results table shows only CNN and CatBoost; claims such as "convolutional contrastive learning shows promising results" are not substantiated numerically in the table. Clarify or add the missing results.
- Choice of features/channels may induce selection bias. CatBoost uses only eight "stable" channels; criteria for selecting them (and whether selection used only training data per split) are not detailed, risking information peeking.
- Limited clinical utility metrics. Beyond ROC-AUCs, there’s no reporting of sensitivity at clinically relevant specificity (or vice-versa), decision curves, or subgroup performance (e.g., smokers vs non-smokers if logged), which are essential for a screening narrative. 
- No cross-site validation; single-center collection. The dataset appears to be collected at one site/device configuration; claims of generalization under drift would be stronger with cross-site tests or at least leave-block-of-weeks-out CV with CIs.

### Questions
The questions for the authors as are posted below,

- Confound analysis. How do results change after adjusting for age/sex (e.g., propensity matching, adding them as nuisance covariates, or stratified evaluation)? Please report AUC/PR-AUC with and without demographic adjustment.
- Missing baselines. The text mentions contrastive/self-supervised methods; can you include their numeric results in the main table (with seeds/CIs) and specify training details?
- Temporal split robustness. Beyond the recommended three validation weeks, can you report leave-one-week-out CV (or multiple week-folds) with mean 95% CI to quantify variability under drift?
- Channel selection protocol. The CatBoost models use eight "stable" channels. How were they chosen (per-split on train only vs. globally)? Please document the selection rule and add a sensitivity analysis over channel subsets.
- Pre-processing clarity. Is the "weightlet" transform a typo for wavelet? I assume so but want to double check...
- Bagged-air protocol variability. Do you record breath volume/flow, or bag fill level? If not, could variable dilution explain some across-week variability? Any normalization by peak flow or a proxy?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces the Scent of Health dataset, a large-scale clinical breathomics collection designed to advance machine learning for eNose based disease diagnostics. The dataset comprises 1,027 patients across eight groups (one control and seven disease) and captures breath samples as multivariate time series (~15-minute) from a custom 17-sensor metal oxide microsensors along with an auxiliary sensor (that captures environmental parameters like temperature, pressure, humidity, and CO2 levels). The authors highlight two main challenges in the field: the scarcity of large, public clinical datasets and the difficulty of modeling eNose data, which is affected by sensor drift and temporal artifacts. 

To address these, they provide

i) the dataset with recommended temporally stratified, week-wise train/test splits that account for sensor drift and propose a benchmark evaluating both feature-based (CatBoost) and deep learning (CNN) methods that reinterpret time series as images. 

ii) The results demonstrate the potential of eNose technology for non-invasive screening while revealing significant challenges in robustness and generalization, paving the way for future work in domain adaptation and specialized time-series modeling. The paper provides code and temporary data links for reproducibility.

### Strengths
The authors have touched each dimension of originality, quality, clarity, and significance.

Originality: The primary contribution is the creation and public release of the S-O-H dataset. It is positioned as the largest and most diverse of its kind for this specific eNose technology, filling a critical gap in the availability of large, clinically annotated olfactory time-series data. The paper thoughtfully frames the problem around realistic deployment challenges, specifically temporal sensor drift and week-wise data distribution. The creation of temporal splits to prevent data leakage and better simulate real-world performance is a nuanced and highly relevant contribution beyond standard random splits.

Quality: The data collection protocol is described in detail, covering participant preparation, sensor chip design and fabrication (including material synthesis), and data logging procedures. This thoroughness inspires confidence in the dataset's integrity and supports reproducibility. The validation scheme is carefully motivated to reduce leakage due to temporally concentrated sampling. Also, for reproducibility, open-sourcing the dataset and code is a significant strength that will catalyze research in the community.

Clarity: The paper is well-written and logically structured. The motivation is clear, the technical gaps are well-articulated, architecture choices, validation strategies, and the contributions are succinctly stated. Figures like the sample distribution over weeks (Fig. 2) effectively communicate the core challenge of temporal concentration.

Significance: Given the increasing interest in non-invasive diagnostics, the dataset and this work provide a critical foundation for advancing AI in breathomics. The open release and reproducible baselines maximize potential impact across both medical and machine learning communities

### Weaknesses
i) At line 77, the authors mention using a unique 17-sensor array of printed ZnO-based metal-oxide microsensors on a temperature-controlled chip. However, the paper does not include a schematic of the sensor layout. Providing even a simplified diagram would greatly enhance clarity and help readers understand the sensing configuration and spatial arrangement of the sensors.

ii) The paper evaluates only two models, omitting other competitive baselines such as Random Forest, HistGradientBoost, XGBoost, or deep architectures like ResNet that can process time-series data as images. Including or at least discussing these alternatives would provide a more comprehensive and fair performance comparison.

iii) The choice of a generic CNN that treats the data as an image may not be the most effective or natural way to model the underlying temporal dynamics and cross-sensor correlations. 

iv) Why were more established architectures like LSTM, InceptionTime, HIVE-COTE, or transformers not included as baselines? Was the relatively simple CNN structure chosen primarily for its speed and suitability for embedded deployment, and if so, could this be stated more explicitly?

v) While the temporal split is a major strength, the paper does not provide a detailed analysis of the drift. Quantifying the drift (e.g., using dimensionality reduction to visualize feature distribution shift across weeks) or directly evaluating baseline drift-correction methods would have strengthened the practical analysis of deployment challenges contribution.

vi) The authors correctly note the inability to perform multi-fold validation due to the temporal concentration of data. However, this limits the statistical robustness of the reported AUC scores. The results should be interpreted as initial findings rather than definitive performance metrics.

vii) The work motivates cross-site issues but presents data from a single device setting; even a small leave-session or leave-batch analysis or synthetic domain shift (temperature or humidity perturbations) would add evidence for external validity.

viii) The code is not visible at the given anonymous link. It shows the message "The requested file is not found".

ix) Are there plans for additional data collection to better balance the disease splits and enable more powerful multi-fold validation?

x) How does the sensor drift observed here compare quantitatively to other sensor deployments or to data in existing olfactory benchmarks?

xi) Was any analysis performed to ensure the models are learning from the breath signal and not from spurious correlations with, for example, the average age difference between healthy and disease groups?

### Questions
I would request authors to answer all points that are raised in the Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2
