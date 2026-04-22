# Detecting Scarce and Sparse Anomalous: Solving Dual Imbalance in Multi-Instance Learning

- Avg Score: 3.33
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4

## Abstract
In real-world applications, it is highly challenging to detect anomalous samples with extremely sparse anomalies, as they are highly similar to and thus easily confused with normal samples. Moreover, the number of anomalous samples is inherently scarce. This results in a dual imbalance Multi-Instance Learning (MIL) problem, manifesting at both the macro and micro levels. To address this "needle-in-a-haystack problem", we find that MIL problem can be reformulated as a fine-grained PU learning problem. This allows us to address the imbalance issue in an unbiased manner using micro-level balancing mechanisms. To this end, we propose a novel framework, Balanced Fine-Grained Positive-Unlabeled (BFGPU)-based on rigorous theoretical foundations. Extensive experiments on both synthetic and real-world datasets demonstrate the effectiveness of BFGPU.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the detection of scarce, sparse anomalies that cause dual imbalance (macro- and micro-level) in Multi-Instance Learning. It reformulates MIL as a fine-grained Positive-Unlabeled (PU) learning task, derives a balanced PU loss via rigorous theoretical analysis, and proposes the BFGPU framework that incorporates macro-level information to assign high-confidence pseudo-labels and dynamically adjust thresholds. Extensive experiments on synthetic datasets, real IDC (medical) and CSQI (customer service) datasets, plus comparisons with supervised/AD/MIL/PU methods and LLMs, validate BFGPU’s effectiveness with theoretical error bound analysis.

### Strengths
1. It clearly formalizes the "dual-imbalanced MIL problem" (macro-level scarce anomalous samples, micro-level sparse anomalous information), addressing a gap in traditional MIL that only focuses on single-level imbalance.

2. This paper reformulates MIL as a fine-grained PU learning task, integrating macro information into micro learning to realize micro-to-macro performance optimization.

3. The authors conduct systematic experiments (synthetic + real datasets like IDC/CSQI) and compare with multi-type baselines (supervised, MIL, PU, LLMs), plus ablation/parameter sensitivity analyses, effectively verifying method validity.

### Weaknesses
1. The paper reformulates MIL as fine-grained PU learning but does not distinguish it from existing MIL-PU hybrid methods explicitly. For example, PUMA, proposed by Perini et al. (2023), already explored "learning from positive and unlabeled multi-instance bags in anomaly detection," and PUMA also assigns instance-level labels. The paper does not clarify how its MIL-to-PU reformulation differs in essence from prior work, thereby weakening the uniqueness of its theoretical motivation. Additionally, PUMA is not compared in the experiment.

2. Though Line 83 claims to "dynamically adjust the confidence threshold to maintain balanced prediction", the critical threshold-related parameter π is fixed—it depends on pre-set σ_micro = l -1 and σ_macro (π=σ_micro/(σ_micro+1) and π=1/((σ_micro+1)(σ_macro+1)), contradicting the stated dynamic adjustment. 

3. Additionally, while the paper aims for equal positive/negative pseudo-labels by selecting "most anomaly-inclined" and "most normal-inclined" instances from anomalous bags, it provides no solution for scenarios where \(\hat{g}\)’s probabilities are all extremely large/small (e.g., negligible differences between "most inclined" instances or uniform class inclination).

4. The paper provides no details on whether comparative methods underwent consistent hyperparameter optimization, which is critical for fair performance evaluation. It specifies tuning for BFGPU (e.g., λ_bfgpu, λ_pse) but does not confirm whether the baselines were optimized within the same search space. 

5. The paper’s two θ updates per epoch (raw micro data + pseudo-labels) inherently increase computational overhead, yet it provides no efficiency comparisons with baselines. For real-time applications (e.g., online CSQI), metrics such as training time per epoch or inference latency are critical.

### Questions
Same to Weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper bridging ideas from MIL and PU learning. It proposes BFGPU to address the imbalance problem at bag-level (macro) and instance-level (micro). The method combines a balanced PU loss, pseudo-label refinement, and an adaptive decision threshold to handle extreme class imbalance and uncertain pseudo-positives. The authors provide theoretical generalization bounds. Experiments show that BFGPU outperforms baseline PU and MIL methods in F1 and average accuracy, particularly under imbalanced regimes.

### Strengths
- Motivation is strong.
- Baselines are comprehensive.
- Evaluations on vision and language tasks are comprehensive and have high practical values.

### Weaknesses
- My main concern is about the novelty. MIL is already studied for anomaly detection in connection to learning with unlabeled data [1], using a similar pseudo-label loss that trains models with the most confident instances [2]. The proposed algorithm BFGPU suggests a new PU paradigm, but it is effectively a weighted PU loss + pseudo-labeling + adaptive threshold, which has been studied separately in prior work [3,4,5].
- Writing needs improvements. For examples, no explanation of acronyms "PU" in abstract; "sparse anomalies" is easily confused with few anomalous samples, hence requires more explicit definition; "existing MIL methods often heuristically assign bag labels to all instances within the bag" is not accurate (assigning bag labels directly to instances corresponds to standard supervised fine-tuning, not MIL); "negaftive" with wrong spelling.
- The paper appears hastily prepared, with limited refinement. Figures are not rendered as vector graphics, making the text difficult to read. Moreover, Figure 2 is poorly integrated with the main discussion. For instance, the term “debias” appears in the figure but is never mentioned or explained in the text.

[1] Unbiased Multiple Instance Learning for Weakly Supervised Video Anomaly Detection, CVPR 2023

[2] Real-world Anomaly Detection in Surveillance Videos, CVPR 2018

[3] Positive-Unlabeled Learning with Non-Negative Risk Estimator, NeurIPS 2017

[4] Robust Positive-Unlabeled Learning via Noise Negative Sample Self-correction, KDD 2023

[5] Self-PU: Self Boosted and Calibrated Positive-Unlabeled Training, ICML 2020

### Questions
- Even with the imbalance setting of CSQI, getting F1 score of 0 across multiple reasonable baselines looks very suspicious. Could you explain this?

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper tackles the challenging task of detecting anomalies that are both scarce and sparse. The authors view it from a dual imbalance multi-instance learning perspective. They propose Balanced Fine-Grained Positive-Unlabeled learning (BFGPU), which reformulates MIL as a fine-grained positive-unlabeled learning problem. The method derives a theoretically grounded balanced PU loss, integrates pseudo-labeling using macro-level information, and dynamically adjusts thresholds for decision balancing. Theoretical analysis demonstrates tighter generalization bounds than classic MIL, and extensive experiments on image and textual datasets show consistent improvements over MIL, PU and anomaly-detection baselines.

### Strengths
1. The identification of dual imbalance (macro and micro) in MIL is insightful and well-motivated by real-world cases.
2. The paper derives unbiased and balanced PU learning objectives and clear generalization error bounds. The comparison of bounds for CGPN, MIL and BFGPU convincingly supports the proposed formulation.
3. The evaluation spans text and image domains and uses both real and synthetic datasets to test imbalance sensitivity. Results show large and consistent gains against baselines.

### Weaknesses
1. The notations are sometimes confusing. In lines 138-139, for the distribution p(x,y), "x is the input and y is the output". What do input and output mean in a distribution p? Same for lines 201-202. The use of mil and MIL, bfgpu and BFGPU, is mixed in text and equations, which is inconsistent.
2. I am confused about some definitions. In lines 209-211, why does the statement "if there exists at least one anomalous component, the macro label is anomalous" equal the statement "if the micro component most inclined towards the anomalous class is anomalous, then the macro label is anomalous"? In line 148, you use $\pi$ to represent the class prior of the positive data. In Equation(3) and the equations after it, should $\pi$ also be an estimator? Or maybe an additional assumption should be made?
3. More discussions of each theorem in Section 6 could be added to help readers understand the theoretical analysis.
4. The “LLM vs. BFGPU” experiment is interesting but simplistic. Prompt-based sentiment classification may not fairly represent modern LLM capabilities or tuning procedures. In addition, the specific versions of the LLMs used in the experiments are not clarified in the paper.

### Questions
1. See my questions in Weaknesses. My biggest concern lies in the writing and soundness of the methodology. 
2. What $\mathcal{L}$ do you use in the implementation?
2. The table size, figure size, font size inside the table and figures, and their arrangement could be improved so that readers can better compare the results.

### Soundness
3

### Presentation
2

### Contribution
2
