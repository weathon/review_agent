# DBGL: Decay-aware Bipartite Graph Learning for Irregular Medical Time Series

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 4, 4

## Abstract
Irregular Medical Time Series (IMTS) are of great importance in the healthcare domain to better understand the patient's condition. However, the inherent temporal irregularity, arising from heterogeneous sampling rates, asynchronous observations, and variable gaps, poses significant challenges for reliable modeling. Existing methods distort the **temporal sampling irregularity** and missing pattern, while failing to capture **variable decay irregularity** in the clinical domain, leading to suboptimal representation. To address these limitations, we introduce DBGL: Decay-Aware Bipartite Graph Learning for Irregular Medical Time Series. DBGL first introduces a patient–variable bipartite graph that simultaneously captures irregular sampling patterns without artificial alignment and adaptively models variable relationships for temporal sampling irregularity modeling, enhancing representation learning. To model variable decay irregularity, DBGL designs a novel node-specific temporal decay encoding mechanism that enables each variable to decay at different rates based on sampling interval, yielding a more accurate and faithful representation of irregular temporal dynamics. We evaluate the performance of DBGL on four publicly available datasets: P19, Physionet, MIMIC-III, and P12. Results show that DBGL outperforms all baselines, and our code is also available in the supplementary material.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces DBGL to solve “temporal sampling irregularity” using a bipartite graph and “variable decay irregularity” using node-specific temporal decay encoding. Experiments demonstrate competitive performance over state-of-the-art models.

### Strengths
1.	Code is provided in supplementary materials and is runnable.

2.	Baselines included in experiments are up to date.

### Weaknesses
1.	The paper’s motivation is clearly problematic. First of all, “distort temporal sampling irregularity” at line 15 is already solved by the bipartite graph representation proposed in GraFITi[1] and adopted by TimeCHEAT[2] (included in baselines). DBGL merely changes the existing time-variable bipartite graph to a variable-patient one. Secondly, “variable decay irregularity” at line 16 is a problem specific to RNN-based models and is not a characteristic of irregular time series. The concept of “decay” is related to the forgetting mechanism in GRU-D, so the claim “most methods impose uniform or overly simplistic decay assumptions” at line 65 does not apply to methods like graph-based (TimeCHEAT[2], Raindrop[5]) or attention-based ones (SeFT[4]).

2.	Some claims in the paper are biased or wrong. (1) The claim “two fundamental limitations persist in existing approaches. One is the distortion of inherent irregularity, where resampling eliminates informative missingness…” at line 62 assumes resampling is a problem cannot be solved by existing methods, while TimeCHEAT[2], GRU-D[3], SeFT[4], Raindrop[5] did not use resampling at all. (2) The claim “variable-variable modeling neglects asynchronous interactions between patients and variables…” at line 63 is confusing. Since “patients” are essentially equal to “samples” in medical irregular time series datasets, this phrase is equal to “variable-variable modeling neglects sample-variable modeling”. They are two independent types of modelling perspectives and are not related at all. Therefore, the logic of this claim is chaotic.

3.	DBGL is composed of two main components: a patient-variable bipartite graph and a modified version of RNN. Patient-variable bipartite graph seems to be responsible for variable dependency learning, while RNN is responsible for temporal one. However, since the time-variable bipartite graph in GraFITi[1] and TimeCHEAT[2] can already model temporal and variable dependencies simultaneously without any problems in “temporal sampling irregularity” or “variable decay irregularity” (see explanations in weakness 1), DBGL’s more complicated design seems to be inferior.

4.	Patient-variable bipartite graph in Figure 1 does not provide any useful information in illustrating how it works. Graphs from t1 to tn look exactly the same. If the plotted IMTS input on the top left is p1, then why are p1 and v2 connected in the bipartite graph at t2? The green v2 does not have observation at t2.

5.	DBGL runs extremely slow using the codes provided in supplementary materials, similar to the speed of ODE-based models, which raises doubts about whether the design mechanisms are worth the trade-off.

6.	Implementation codes in supplementary materials seem to include lines written by LLMs, while the disclaim in Appendix G does not mentioned its usage.

[1] V. K. Yalavarthi et al.; “GraFITi: Graphs for Forecasting Irregularly Sampled Time Series”; AAAI 2024

[2] J. Liu, M. Cao, and S. Chen; “TimeCHEAT: A Channel Harmony Strategy for Irregularly Sampled Multivariate Time Series Analysis”; AAAI 2025

[3] Z. Che, S. Purushotham, K. Cho, D. Sontag, and Y. Liu; “Recurrent Neural Networks for Multivariate Time Series with Missing Values”; Sci Rep 2018

[4] M. Horn, M. Moor, C. Bock, B. Rieck, and K. Borgwardt; “Set Functions for Time Series”; ICML 2020 

[5] X. Zhang, M. Zeman, T. Tsiligkaridis, and M. Zitnik; “Graph-Guided Network for Irregularly Sampled Multivariate Time Series”; ICLR 2022

### Questions
1.	What’s the point of using RNN in DBGL? As mentioned in weakness 3, RNN appears redundant compared to the time-variable bipartite graph design in GraFITi and TimeCHEAT. The bipartite graph natively handles irregularities, while RNNs require carefully designed decay functions to compensate for them. 

2.	To use patient-variable bipartite graph for learning variable dependencies across samples, a key challenge is to ensure that messages passed from other samples can promote the learning in current sample, which require samples to be similar enough or belong to the same category. Does DBGL have any related design?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the problem of modeling irregularly sampled clinical time series. The authors propose constructing a Patient–Variable Bipartite Graph (PVG) at each time step, where edges correspond to actual observations, and using an autoregressive update to maintain variable-specific hidden states over time. The model further introduces a Temporal Decay Encoding (TDE) mechanism to capture variable-specific temporal dynamics and a learnable codebook to regularize patient representations.

### Strengths
1. The idea of jointly modeling patients and variables via a bipartite graph is reasonable. It provides a structured way to represent irregular observations and aligns well with clinical intuition.

2. The writing and organization are clear, with a well-structured.

3. The study offers extensive experiments.

### Weaknesses
1. The model builds on key hypotheses, e.g., that patient–variable relations better capture irregularity and that each variable should have its own decay rate, but these remain speculative. Providing more concrete evidence will make these arguments more convincing.

 2. The TDE employs an exponential decay function with a learned Softplus(MLP) rate. However, there is no discussion or comparison with alternative continuous-time kernels or analysis of identifiability and numerical stability under large time intervals.

3. The evaluation primarily targets mortality prediction on small-scale clinical datasets (<40 variables). This restricts generalizability. Testing on more diverse or large-scale tasks (e.g., phenotype classification, readmission) would better demonstrate practical applicability.

4. Constructing a patient–variable bipartite graph at each timestep and running EdgeSAGE message passing is computationally expensive. The experiments use relatively few variables, and the paper does not report graph construction overhead.

5. The paper suggests that most Transformer-based methods rely on resampling or imputation, which is misleading. Models such as SeFT, StraTS, and Warpformer also handle irregular sampling directly. The related work section should be more comprehensive and precise.

### Questions
1. During inference, are the patient–variable graphs pre-built or dynamically generated? If generated online, what is the computational overhead and runtime complexity compared to sequence-based baselines?

2. The matched code vector appears to contribute marginally based on ablation results. Is the codebook essential to model performance or mainly a design embellishment? Could the authors provide additional analysis (e.g., utilization rate, collapse behavior) to support its utility?

3. Could the authors provide quantitative evidence isolating the contributions of Time Embedding versus Temporal Decay Encoding? Their roles appear conceptually overlapping.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes DBGL, a novel decay-aware bipartite graph learning framework for irregular medical time series. DBGL models time series as patient-variable bipartite graphs to preserve irregular sampling patterns and introduces a node-specific temporal decay mechanism to handle variable-specific forgetting rates. Extensive experiments on four clinical datasets demonstrate that DBGL outperforms existing methods in prediction tasks and shows superior robustness under high missingness conditions.

### Strengths
1. Novel Bipartite Graph Formulation: It introduces an innovative patient-variable bipartite graph that directly encodes irregular sampling patterns without artificial alignment, effectively preserving crucial information about observation dependencies and missingness.

2. Adaptive, Variable-Specific Decay Mechanism: The model features a novel node-specific temporal decay encoding, allowing different clinical variables to "forget" information at unique, data-driven rates, which more accurately reflects real-world physiological processes compared to uniform decay assumptions.

3. Comprehensive Empirical Validation: DBGL demonstrates consistent and superior performance across four public clinical datasets, significantly outperforming a wide range of strong baselines and showing remarkable robustness in challenging "leave-variables-out" scenarios with high missingness.

### Weaknesses
1. Limited Scope of "Variable Decay" Evaluation: The paper's central claim is modeling "variable decay irregularity," where each clinical variable decays at a different rate. However, the evidence for this is primarily indirect, demonstrated through overall performance gains and ablation studies. A key missing analysis is a direct examination of the learned decay rates (λ_{p,n}^t). The work would be significantly strengthened by visualizing or statistically analyzing these rates across different variable types (e.g., vital signs vs. lab tests) to validate if they align with clinical intuition (e.g., heart rate changes minute-to-minute, while a creatinine level changes over hours or days). Without this, it remains unclear if the model is learning clinically meaningful, distinct decay patterns or simply using the mechanism as a flexible, yet uninterpretable, fitting tool.

2. Scalability and Complexity Concerns: The proposed method constructs a bipartite graph at each time step, which is a key to its success but also its primary computational burden. For long sequences with many patients and variables, this could lead to significant memory and time complexity, potentially limiting its application to real-time clinical settings or very large datasets. The paper would benefit from a more formal complexity analysis (e.g., O(|V| |E|) per time step) and a discussion of this trade-off. An actionable suggestion for future work would be to explore strategies for dynamic graph sparsification or sampling to improve scalability without a significant loss in performance.

3. Evaluation: A key weakness of the evaluation is its confinement to binary classification tasks, which fails to adequately demonstrate the generalizability and temporal modeling prowess of the proposed DBGL framework.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces an bipartite graph approach, establishing cross-patient and cross-series connections, for irregular medical time-series modeling.

### Strengths
- An interesting approach: a bipartite graph representation combined with irregular decay modeling of clinical variables
- Significant performance improvements (especially in AUPRC) in experiments

### Weaknesses
The graph representation seems to work well for existing patients, but how about new patients? I think comparing graph and non-graph methods in this setup is also critical to give a comprehensive evaluation of different modeling designs.

### Questions
- How could the proposed method handle unknown patients, which do not occur in the training data but has clinical signals in your test period? The graph-based approaches seem to in general outperform non-graph methods, I wonder whether this is related to the train-test division. Please explain clearly about your setup.
- How to align different patients whose signals may include in different time periods? Their clinical signals may have some connections but not recorded in the same global timestep. In practice, how you convert global timestamps recorded in the database to your time step $t$ that unifies multiple patients?
- How does using different groups of patients to construct the graph affect the modeling performance?
- How large the graph could your approach support given the current implementation? What is the potential bottleneck for scaling?

### Soundness
3

### Presentation
3

### Contribution
3
