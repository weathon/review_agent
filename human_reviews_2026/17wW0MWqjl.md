# LLM-OAP: LLM-based Data Augmentation Framework for Enhancing Order Acceptance Prediction in Mobility-on-Demand Systems

- Decision: Reject
- Scores: 4, 6, 6, 2

## Abstract
In Mobility-on-Demand (MoD) systems, drivers’ order acceptance behaviour directly influences matching, pricing, and thus overall system efficiency. Traditional discrete choice models rely on pre-specified utility functions and error structures. This introduces specification risk and limits their ability to capture complex nonlinear interactions, and correlated choices, reducing effectiveness for modelling driver decisions. Meanwhile, behavioural data often come from stated-preference (SP) surveys; these datasets are typically small-scale and based on hypothetical responses, which can be subjective and limit external validity, reducing predictive performance and generalisability. This paper proposes LLM-OAP, a novel framework  that integrates large language model (LLM)-based data augmentation with machine learning (ML) to improve the estimation of drivers' order acceptance behaviour. Our method leverages an LLM to generate synthetic samples based on the real SP data and employs a curation scheme to mitigate implausibility, reduce bias, and maintain diversity. The augmented dataset is used to train ML models beyond fixed utility specifications. Evaluations on two types of SP datasets (covering full- and limited-information settings) show that our framework significantly enhances the performance of state-of-the-art ML models in order acceptance behaviour estimation, while maintaining good generalizability and explainability.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents LLM-OAP, a data-augmentation framework that leverages Large Language Models (LLM) to produce high-quality synthetic samples for small-scale behavioural datasets, with the goal of improving order-acceptance prediction in Mobility-on-Demand (MoD) systems. Drivers are first clustered into personas via feature-aware grouping; synthetic orders and corresponding behavioural labels are then generated, after which a confidence-based and uncertainty-based filter retains only the most reliable samples. Evaluations on stated-preference (SP) survey data, under both full-information and limited-information regimes, show consistent gains over strong ML baselines, GAN / diffusion augmentations and recent LLM-centric methods. Extensive ablations and empirical analyses are offered to support interpretability and generalisability claims.

### Strengths
• Tackles a commercially relevant, technically hard problem: scarce data, distribution gaps and non-linear driver behaviour in MoD platforms.  
• Proposes a novel, fully structured LLM pipeline that mitigates data scarcity, strong subjectivity and the limited external validity typical of SP surveys.  
• Combines persona-based grouping, explicit behavioural summarisation, synthetic-order generation and label simulation, capturing heterogeneous driver preferences instead of forcing a single model on all users. Figure 1 convincingly illustrates the design.  
• Confidence / uncertainty filtering demonstrably reduces noisy or biased samples; Figure 3 quantifies the denoising effect.  
• Experiments are meticulous: Tables 1 & 3 report accuracy and calibration metrics (ACC, AUC, AUCPR, ECE, BS, NLL) under both information regimes, beating GAN, diffusion and LLM baselines.  
• Tests multiple LLM back-ends (DeepSeek-V3, R1, Qwen3-plus); performance is largely invariant to the choice, indicating robustness.  
• Comprehensive baselines (GAN, diffusion, other LLM schemes) and ablations (Tables 2 & 4) together with filtering-ratio analysis (Figure 2) isolate the contribution of each module.  
• Figures are polished and informative; prompt designs are disclosed in Figure 4, enhancing reproducibility.  
• Writing is clear; related work is well positioned.

### Weaknesses
1. External validity: all experiments use SP data; no validation on large-scale real order logs. The framework’s practical impact on revealed-preference (RP) environments remains unverified.  
2. Potential SP over-fitting: the XGBoost filter is trained only on the original SP set, risking preservation of SP-specific artefacts rather than true behavioural realism.  
3. Diversity evaluation is missing: no quantitative metric or visualisation of behavioural diversity post-augmentation; aggressive filtering may inadvertently over-represent dominant patterns.  
4. Feature grouping lacks justification: random selection from the top-10 features (§3.2) is ad-hoc; no sensitivity analysis on group size or feature choice.  
5. Black-box LLM decisions: no theoretical or empirical evidence that synthetic labels possess behavioural fidelity; consistency checks against human drivers or RP data are absent.  
6. LLM hallucination and adversarial robustness are not examined. And no explicit stress tests or human audits of harmful or contradictory samples.  
7. Reproducibility gaps: hyper-parameters of the filtering stage, pre- filtering statistics, post-filtering statistics and exact prompt variants are only partially reported.  
8. Small data scale: the total augmented volume is tiny compared with industrial MoD data; many features show negligible permutation importance, suggesting under-utilisation of the feature space.  
9. Unequal post-generation filtering: GAN and diffusion baselines are not subjected to the same confidence-based pruning, confounding the comparison.  
10. Limited interpretability: despite claims, no model-level explanations (e.g., SHAP, attention heat-maps) are supplied for either the final predictor or the generation process.  
11. Figures could be richer: Fig. 2 omits diversity curves; Fig. 3 shows pre- filter but not post-filter distributions.  
12. Cost concerns: large-scale LLM API calls for data generation and label simulation may be prohibitively expensive for production deployment.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This manuscript introduces a LLM-based data augmentation process in the context of riding hailing for improving the prediction performance of driver accepting an order or not. It presents the complete steps from real data processing, based on the processed real data applying LLM to generate simulated data with quality examination, to making use of the augmented data to train prediction models with necessary evaluations.

### Strengths
The strengths of the work:
1. Exploring data augmentation by applying LLMs is still a quite new, conceptual area.
2. The framework considers and takes into account feature engineering, persona grouping, and preference analysis, which well aligns with the characteristics of the target context.
3. The experiments include five baselines of data augmentation and six (four major focused) prediction models, which makes the performance results sound with reliable evidence to support. Table 2 indicates each major component in the framework serves a necessary role.
4. The manuscript are easy-to-follow with efficient expressions.

### Weaknesses
The weaknesses of the work:
1. The generalization of the work's propose framework is less discussed and proved. In addition to the SP data, what other common data in the riding hailing context can be also processed and simulated using the propose framework with good performance? Are the data processing approaches very specific to the used SP data?
2. Is the diversity of the real-data set (~3000) used representative enough to a regional riding hailing market (both spatial and temporal dimensions)? As the LLM generated data heavily rely on the real data, the real-world, practical impact and performance of prediction seem majorly associated with the real-data quality. The contribution of the work is not that significant.
3. Figure 1 typo: 20% of real data as a "test" set.

### Questions
1. In addition to the SP data, what other common data in the riding hailing context can be also processed and simulated using the propose framework with good performance? 
2. Are the data processing approaches very specific to the used SP data?
3. As this work more belongs to application-perspective rather than theoretical insight, could the authors justify the contribution of the work regarding real-world application and impact?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
In this paper, authors propose LLM-OAP, that integrates LLM-based data augmentation with machine learning to improve the driver's order acceptance behavior.

### Strengths
1. A comprehensive discussion about group-based data augmentation.
2. This paper consider most of the baselines for this task. 
3. The ablation study is also relatively comprehensive.

### Weaknesses
1. Right now, are there only limited data for driver order acceptance data? Do we really need data augmentation in this task?
2. A comparison between QWen and other LLMs can be helpful (can be included in tha main section). 
3. It would be useful if there are some experimental results to show the importance of Consistent Check.
4. For ML models selections, authors can discuss some selection principles while dealing with different dataset/data size......

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes LLM-OAP, a framework that uses Large Language Models to generate synthetic data for improving machine learning models that predict ride-hailing driver order acceptance behavior. The method employs feature-aware persona grouping, LLM-based data generation, and confidence-based curation. Experiments on stated-preference survey data show 2-3% accuracy improvements over baselines. While the paper addresses a relevant problem and demonstrates consistent improvements, it suffers from critical methodological limitations that prevent acceptance at a top-tier venue. The work lacks novelty in its core approach, provides no mechanistic understanding of why augmentation helps, cannot be reproduced due to missing details, and is evaluated only on a single small dataset.

### Strengths
1. The paper compares against recent, relevant methods rather than weak strawmen. Including CLLM, Pred-LLM, and GPAIS as baselines shows engagement with recent LLM-based augmentation work. The comparison with traditional generative methods (CTAB-GAN) and diffusion models (TabDDPM) is also appropriate and fair.

### Weaknesses
1. Limited Novelty. The paper essentially combines existing techniques without fundamental innovation. LLM-based tabular data generation has been well explored. What distinguishes this work is primarily its application to ride-hailing behavior, but no evidence suggests this domain requires fundamentally different treatment.
2. Weak performance improvement. The marginal improvement over existing methods is minimal. Compared to Pred-LLM, LLM-OAP gains only 1-1.5%. Compared to the simpler, fully reproducible TabDDPM, the gain is merely 1%. This raises a critical question: does the added complexity and loss of reproducibility justify such marginal improvements?
3. Non-reproducibility. 
3.1 The paper has severe reproducibility problems that are never acknowledged. Critical LLM parameters—temperature, top-p, top-k, random seeds—are completely unspecified. Without these, replication is impossible. 
3.2 When the authors mention using "three random seeds," this appears to apply only to downstream ML training, not LLM generation itself. The core contribution—synthetic data generation—is completely uncontrolled. No variance across generation runs is reported, no confidence intervals, no statistical significance tests.
4. No Understanding of Why It Works. The paper's most fundamental gap is the complete absence of mechanistic understanding. It shows that adding synthetic data improves performance but never explains why. What information do synthetic samples add? Do they fill gaps in feature space? Balance classes? Introduce new behavioral patterns? Or simply provide regularization through noise? These critical questions remain unanswered. Without mechanistic understanding, we cannot rule out that improvements stem from basic oversampling effects rather than LLM-specific capabilities.
5. Methodological Concerns. The filtering uses circular validation—an XGBoost model trained on original data curates synthetic samples, biasing retention toward samples resembling the training set. This could reduce diversity rather than enhance it, creating an echo chamber where synthetic data reinforces existing patterns. Many choices appear arbitrary: why exactly four features for grouping? Why three randomly selected from the top ten? Why 32 groups for 3,000 samples? Why generate exactly 3,000 synthetic samples? These likely affect performance but go unexplored.

### Questions
1. In Figure 1, 20% training was used for evaluation, do you mean test set here?

### Soundness
2

### Presentation
2

### Contribution
2
