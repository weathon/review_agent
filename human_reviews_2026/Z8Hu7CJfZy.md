# Time Conditioned Foreseeing: Temporal Generative Pretraining for EHR foundation models

- Decision: Reject
- Scores: 2, 6, 6, 4, 2

## Abstract
Electronic Health Records (EHRs) possess unique characteristics that differ significantly from natural language. However, existing models have overlooked these properties and largely relied on Natural Language Processing (NLP) approaches, resulting in suboptimal performance. To address these limitations, we propose a pretraining method designed to effectively capture the distinctive features of EHRs. First, EHRs contain both clinically critical and less informative numerical ranges. To reflect this, we introduce a Pathology-Focused Binning strategy that emphasizes values with clinical significance. Second, both absolute timestamps and relative time intervals are important in EHRs. To incorporate these temporal aspects, we propose a Dual-Calendar Rotary Positional Embedding (RoPE) that jointly encodes complementary temporal signals. Third, many medical applications require modeling long-term patient interactions. Accordingly, we extend conventional next-token prediction with a Time-Conditioned Foreseeing (TCF) objective, enabling the model to forecast long-range clinical events across multiple temporal horizons. Our approach establishes the first genuine temporal generative EHR model, advancing long-range clinical forecasting. It outperforms existing EHR foundation models on seven diverse downstream tasks and enables realistic and temporally consistent EHR generation. All code and models will be made publicly available in the final version of the manuscript.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper improves EHR foundation models by better featurizing time and numeric values and incorporating time into the pertraining objective.

### Strengths
- MIMIC-III is a good starting point for evaluation as it is reproducible (but as noted in weaknesses, I don't think it's sufficient by itself).
- The set of evaluation tasks is good.
- Calendar time features are generally a good idea (except for using the year as a feature). In theory, these features are unnecessary (since they can be learned from ages with neural network). Still, the transformation from ages to calendar time is non-trivial, and giving the model a shortcut might be an excellent idea.

### Weaknesses
Experimental Setup / Soundness Weaknesses:
- This model appears to use the year of the visit as a feature, but it is not evaluated correctly. In order to evaluate the effectiveness of absolute time features, you must perform a time split (with held-out test years) to assess whether your trained use of absolute time generalizes to the future. Otherwise, you face severe overfitting issues.
- I would argue that MIMIC-III is too small a dataset for modern foundation model work. Especially when datasets like MIMIC-IV are available.
- The lack of a gradient tree baseline (xgboost or lightgbm) is very concerning. Those baselines are essential for ensuring more sophisticated models are set up correctly, as they perform very well and require minimal tuning. 
- Qualitative evaluation of temporal generative modeling is very flawed. Following prior work (see https://www.nature.com/articles/s41746-024-01235-0), the proper procedure is to instead evaluate temporal generation by comparing the output timelines to your predictive models. Out of the patients who died, ETHOS's generated timelines predicted death x% of the time while our model's timelines predicted death y% of the time. If you want to measure the timeliness of the timelines, measure how well the models output death tokens within a certain time window.
- The lack of public code limits my ability to more properly review this paper. 
- The summarized test set is insufficient for evaluating ablations in table 4. Test loss is probablemtaic because it is very heavily driven by outliers. Measuring AUROC and AUPRC would be much more reliable.
- How were baselines tuned for hyperparameters? This is especially important with the lack of a gradient tree based baseline (which still requires tuning, but is much less sensitive).

Related Work / Contribution Weaknesses
- The Table 1 (literature review) table has many flaws. EHRMamba and EHRSHOT all use numeric values with quantile binning. A lot of the models also use relative intervals. I know Foresight, EHRMamba and EHRSHOT at minimum use them.
- The use of the word "uniform" numeric binning is not accurate for describing the prior work. The prior work being compared to uses quantile binning, which is very non-uniform. 
- I don't understand the non-concurrency column in table 1 and in your contributions. Isn't non-concurrency implicit with any relative time featurization? How can a model have relative time input but be missing non-concurrency?
- EventStreamGPT should really be mentioned in related work, as it focuses on time conditioned pretraining with EHR data.
- The marks for temporal generation appears to be wrong for many of the entries in the literature review. For example, Foresight supports temporal generation.
- The novelty of the time conditioned forcasting objective is unclear. Especially compared to related work like EventStreamGPT. I would argue ETHOS also supports it to a similar extent to your model.
- The time-conditioned pretraining for this model appears to require rounding the time to various calendar time bins? And then cross entropy loss is taken? How is this different from ETHOS's time tokenization? AKA it's not a real continuous time prediction model like EventStreamGPT?

### Questions
See weaknesses section.

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a new temporal generative pretraining framework for Electronic Health Record foundation models, called Time-Conditioned Foreseeing. The model introduces three main innovations: a Pathology-Focused Binning method for emphasizing clinically significant numeric ranges, a Dual-Calendar Rotary Positional Embedding for encoding both absolute and relative time, and a Time-Conditioned Foreseeing objective for forecasting future clinical events at multiple temporal horizons. The model is trained and evaluated on the MIMIC-III dataset, showing improved performance across seven downstream clinical prediction tasks, with up to 48% higher AUPRC than prior EHR foundation models. Experiments further show that the model generates realistic and temporally consistent synthetic EHRs that capture physiological time patterns such as circadian variations. The authors conclude that their approach establishes the first genuine temporal generative EHR model and identify a lack of established metrics for evaluating temporal EHR generation as a key limitation.

### Strengths
Below are the strengths of the paper in my opinion:

1. The paper identifies a clear problem in how prior EHR foundation models rely on NLP-style pretraining and motivates its approach by emphasizing the temporal and numerical nature of EHR data, making the contribution well justified.
2. It introduces three specific techniques (Pathology-Focused Binning, Dual-Calendar RoPE, and Time-Conditioned Foreseeing) with mathematical definitions and clear intuition, showing conceptual clarity and novelty.
3. The model consistently outperforms all baselines across seven downstream tasks on MIMIC-III and demonstrates up to 48% higher AUPRC, supporting the strength of its results.
4. An ablation study verifies that removing or replacing any of the proposed components degrades performance, which strengthens the paper’s claims.
5. The use of the public MIMIC-III dataset helps with transparency and reproducibility although no code is provided as supplementary material yet.
6. The visualizations of the paper are great. Visualizations such as the circadian rhythm plot provide qualitative evidence that the model generates realistic and temporally consistent EHR sequences.

### Weaknesses
Below are the weaknesses of the paper in my opinion:

1. The paper lacks quantitative metrics for evaluating generative realism and acknowledges that no established evaluation exists for temporal EHR generation.
2. All experiments are limited to the MIMIC-III dataset, leaving generalization to other settings untested. Additionally, I do not see why the authors have not used much richer MIMIC-IV dataset. Extending the methods to MIMIC-IV should not be too hard and I think is a clear missing point in the current state of the work.
3. The Time-Conditioned Foreseeing mechanism is densely written and could be difficult to follow, despite including equations and diagrams. Please add more descriptive details to the current appendix section for this part.
4. The paper omits any discussion of computational cost or efficiency implications of its multi-component architecture.
5. There is no analysis of fairness, subgroup behavior, or bias, even though the domain involves clinical data.
6. The limitations of the paper is not discussed well. Failure cases and uncertainty sources are not discussed, leaving the reader without insight into model limitations.

### Questions
Most my questions are embedded in the weaknesses section. My main question to improve is use of other diverse EHRs and seeing how the modeling pipeline transfers to datasets other than MIMIC-III. In addition, some extra clarifying details would be helpful for the following minor questions:

1. How does the proposed TCF objective differ in practice from standard next-token prediction when generating long-term event sequences?
2. Can the model handle irregular or missing time intervals in EHR sequences, and if so, how is this addressed in preprocessing or training?
3. What specific criteria were used to select the seven downstream tasks, and how do they relate to the generative capabilities claimed by the paper?
4. Does the Dual-Calendar RoPE generalize to time zones or varying calendar systems, or is it tailored specifically to the MIMIC-III dataset’s timestamp format? This is especially important because across datasets the deanonymization on time could vary and this can contaminate the pipeline.
5. How sensitive is the Pathology-Focused Binning approach to the bandwidth parameter in the Gaussian kernel density estimation, and was any hyperparameter tuning performed?
6. How do the authors ensure that the qualitative evaluations meaningfully reflect real-world clinical plausibility beyond what is currently done? I feel like the interpretability aspect here can be improved.
7. Could the proposed model inadvertently memorize training patients or sequences during generative pretraining, and were any checks for memorization performed?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper focuses on EHR generation, addressing the limitation of over-relying on natural language processing to model numeric intensive and time sensitive EHRs. This paper proposes 1) Pathology-Focused Binning strategy to emphasize clinically significant numerical values with density-based tokenizing, 2) Dual-Calendar Rotary Positional Embedding to consider calendrical context and relative time intervals at the same time, 3) Time-Conditioned Foreseeing learning objective to inform forecasting with explicit temporal horizons. Extensive experiments demonstrate the effectiveness of the proposed method, producing substantially highest results over seven downstream tasks. Human evaluation and LLM judgements both suggest high generation fidelity. Case study shows that the proposed method can correctly separate calendrical relevant and irrelevant values, which further validates the Dual-Calendar RoPE design.

### Strengths
1. The paper is well-motivated and is well-positioned within literature. The overall structure is clear, and the writing is easy to follow. Discussion into related work and contributions are thorough.
2. The Pathology-Focused Binning strategy, Dual-Calendar Rotary Positional Embedding and Time-Conditioned Foreseeing learning objective are innovative, each addressing a critical and specific problem of existing EHR foundation model. 
3. The effectiveness of the proposed model is evaluated through multiple angles, including test loss, downstream metrics, human evaluation, LLM evaluation and case study. Results demonstrate that the proposed method consistently outperforms baselines.
4. The implementation descriptions contain sufficient details for understanding, and source codes are claimed to be released, so reproducibility is satisfying.

### Weaknesses
1. The calendrical period embedding seems to be sparse with exponentially expanding discrete window, which may result in inefficiency.
2. The two-stage manner and multi-scale forecasting in time-conditioned foreseeing consume extra computes.
3. The ablation study is inadequate, presenting only test loss results instead of downstream task-specific evaluation metrics which is of higher clinical significance.
4. Minor formatting issue: the equations are not numbered.

### Questions
1. How is value-sharing implemented? Why do methods with shared value perform worse than methods without value-sharing?
2. Is the density-based binning strategy robust to OOD values? Will there be any potential distribution shift problem in the experimental or real clinical scenario, and how does this method handle it?
3. Is the performance sensitive to model size? I have noted that some baseline models (e.g., MOTOR, ETHOS) are originally designed to contain more trainable parameters than in this paper, and some results in the original papers show higher performance (e.g. ETHOS obtains an AUC of 0.921 in IHM task.)

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a pretraining approach tailored to the special structure of Electronic Health Records (EHRs). The authors propose three main contributions: (1) Pathology-Focused Binning — a density-weighted percentile binning for numerical values that preserves resolution in clinically important (low-density) ranges; (2) Dual-Calendar RoPE — a split rotary positional embedding that jointly encodes relative token order and multi-scale calendrical time (minute, hour, day, etc.); and (3) Time-Conditioned Foreseeing (TCF) — a new dual-objective pretraining head that (i) predicts the next event time in a multi-scale calendrical decomposition and (ii) conditions generation on arbitrary future timestamps to “foresee” events at those times, enabling genuine temporal generative modeling. The method is trained and evaluated on MIMIC-III; it outperforms a broad set of baselines on seven downstream tasks and shows qualitatively and (to some extent) quantitatively superior temporally consistent EHR generation.

### Strengths
- The design of the model architecture are well-motivated, including pathology-based binning and temporal positional encoding.

- Strong empirical gains on standard clinical benchmarks. The model consistently outperforms multiple baselines on 8 downstream classification tasks.

- The analysis on generative results are illustrative. The paper includes human and LLM-based evaluation and qualitatively compares the generated results of the model.

### Weaknesses
- All experiments use MIMIC-III only. While MIMIC-III is standard, it's quite small and unique. Claims about generality of calendrical modelling and generative realism would be stronger if validated on at least one additional dataset (e.g., eICU, other hospital systems). This limits conclusions about generalization to different EHR systems.

- It is unclear how test loss is defined. Not sure the comparison is fair as the objectives of baselines are different.

- The generative prediction have limited real-world application. The generated timestamps and vital features are random, which can not help clinical decision making. A feasible improvement can be generating prediction for all features at constant temporal gaps, and conduct quantitative analysis comparing to ground truth.

- Figure 6 does not have scale of y-axis.

- The discussion on the failure is insufficient. It would be better to show some failure cases of the model.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses the construction of an EHR foundation model. The authors propose Pathology-Focused binning, which enables density-adjusted binning of numerical observation values, Dual-Calendar RoPE, which incorporates calendrical time in addition to ordinary positional information to capture calendrical periodicity and event concurrency, and Time Conditioned Foresee Objective, which learns when an event will occur. The proposed method was evaluated on tasks commonly used in EHR model evaluation and consistently outperformed baselines.

### Strengths
-- Addressing density-adjusted binning, incorporation of calendrical time, and learning when an event will occur is quite a significant problem, and its importance is well-justified in the paper.

-- Experimental results on multiple tasks demonstrated the effectiveness of the proposed method.

### Weaknesses
-- The proposed method may have novel points and have some practical impact, but the clarity issues are too severe to understand the method:
* In l.134, l.147, and l.183, E, F, and T are not defined.
* The paragraph "Pathology-Focused Binning" is hard to follow, where a lot of variables and concepts are not defined, such as X, x, h, sigma, and raw count. rho() is used for both functions of x and v, which is also confusing.
* The paragraph "Dual-Calendar Rotary Position Embedding" also contains confusing notations, such as the variable x may be used in a different meaning from the paragraph "Pathology-Focused Binning". What kind of operation is "||"? s and t should be defined. If RoPE() is used as a function, it should be defined.
* l.272 first mentions "the transformer backbone", which looks important, but it does not appear and is not defined before.
* The paragraphs from l.280 are hard to follow since there are many undefined concepts and variables, and equations are not written formally.
* Appendix may contain additional details, but there is an inconsistency of notations from the main text, which makes it difficult to get a clue to understand the proposed method. Also, the main text should be self-contained.
* In Figure 5, what do the bars mean?
* It is better to put the tables and figures on top.

### Questions
Nothing.

### Soundness
3

### Presentation
1

### Contribution
3
