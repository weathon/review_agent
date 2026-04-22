# Fortifying Hallucination Detection to Out-of-Domain Data

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 6

## Abstract
Hallucinations remain one of the major barriers to the reliable deployment of Large Language Models (LLMs).
Recent works have explored both supervised classification based approaches and unsupervised metric based approaches, with the latter remaining popular since they do not require labeled data. However, unsupervised methods lag behind supervised ones for in-domain data, despite having slightly better performance out of domain, as we show across 11 datasets and 10 models. This underscores the importance of supervised approaches, but also highlights their weakness in generalizing to unseen domains. To narrow this generalization gap, we introduce a simple approach to make supervised hallucination detectors more generalizable by relying on a curated, multi-domain training mix, which can complement subsequent addition of task-specific data. In our experiments on hallucination detection on 691K QA samples from 11 open source QA datasets, we show that incorporating this general training allows supervised methods to surpass unsupervised metric based methods by an average of +7.25% on out of domain data, without addition of any task-specific data. We also analyze scaling behaviors and estimate how much task-specific data is required to achieve reliable performance, finding that models augmented with general data require up to 40.3% less task-specific data to achieve close to optimal performance. 
Together, our findings highlight both the brittleness of existing supervised hallucination detectors and a simple path toward fortifying them detection against domain shift.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper investigates the domain shift problem in supervised hallucination detection for Large Language Models (LLMs), where supervised methods demonstrate strong in-domain performance but degrade significantly under distribution shift. The papers propose a data-centric solution involving training supervised detectors on a diverse "general training mix" to improve out-of-domain robustness. They evaluate their approach across 697K QA samples from 12 datasets and 10 different LLMs, reporting an average improvement of +7.25% in out-of-domain performance and demonstrating that their approach can reduce task-specific data requirements by up to 40.3%.

### Strengths
1. Comprehensive empirical evaluation: The study spans a substantial experimental scope with 697K samples across 12 diverse QA datasets and 10 different LLMs, providing extensive coverage of domain shifts and model architectures.
2. Clear problem identification: The work systematically documents and quantifies the generalization gap between in-domain and out-of-domain performance for supervised hallucination detection, which is a legitimate practical concern.
3. Practical applicability: The proposed solution directly addresses a real-world deployment challenge where hallucination detection systems must operate across diverse domains.
4. Thorough benchmarking: The comparison against state-of-the-art unsupervised methods (Semantic Entropy, SIndex, GNLL, PTrue) across multiple evaluation metrics (AUROC, F1-score) provides a comprehensive baseline.

### Weaknesses
1. Limited novelty in methodology: The core contribution—using diverse training data for domain generalization—is a well-established technique in machine learning, particularly in domain adaptation literature. The application to hallucination detection, while practically motivated, does not introduce novel methodological innovations or theoretical insights.
2. Insufficient principled analysis: The paper lacks deep theoretical analysis of why diverse training improves out-of-domain performance in the specific context of uncertainty-based hallucination detection. The underlying mechanisms remain under-explored.
3. Methodological superficiality: The approach relies primarily on empirical observations without providing theoretical foundations or formal analysis of the generalization properties of the proposed method.

### Questions
1. What theoretical insights can you provide about why diverse training improves hallucination detection under domain shift? How do these insights extend beyond simple domain adaptation principles?
2. How do you reconcile the apparent contradiction between the reported benefits and the observation of "negative transfer" when relying on heterogeneous general data?
3. What specific components of the general training mix are most responsible for the observed improvements? Have you conducted systematic ablation studies to identify these?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a data-driven approach to enhance the generalization of supervised hallucination detection methods. Through extensive experiments on 11 open-source QA datasets, the authors demonstrate that incorporating general training data improves the performance of supervised methods on out-of-domain datasets.

### Strengths
1. **Comprehensive Experiments**: Experiments span 11 open-source QA datasets and 10 different LLMs. Compare supervised and unsupervised methods in both the in-domain and out-of-domain settings.

2. Introduces a data-driven method to improve the cross-domain generalization of supervised hallucination detectors.

### Weaknesses
1. **Limited Scope**: This paper only conduct experiments on QA dataset for white(grey)-box hallucination detection method.

2. **Hallucination Labeling**: The labeling fully relies on LLM judges, and potential biases or errors from the judge model could affect the results.

3. **Limited Insight**: The findings are not particularly insightful; the observation that general data can improve out-of-domain performance and enhance data efficiency is largely expected.

4. **Undefined Terms**: PT (Line 321) and FT (Line) 367 is introduced without prior explanation and not defined later.

5. **Missing Appendix Links**: The main text does not include clickable links to the appendices (e.g., Appendix C and D), making it hard to check important details.

### Questions
## Questions
I am confused about the exact size of the GE dataset and the mixing ratios of the different datasets used to construct it. I expect the training dataset to be large, but the experiments in Section 4.3 suggest that only a few thousand examples are used.

## Suggestions
1. The paragraph titled “Evaluation Dataset” is not entirely accurate, as these datasets are later split into training and test sets for the leave-one-out experiments.
2. Line 736 references a table that is missing.

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies domain adaptation for LLM-based QA models. Many empirical findings are stated, including 1) supervised > unsupervised in in-domain settings, 2) training with general (GE) data improves OOD, and 3) GE + task-specific (TS) data leads to lower annotation costs. 

There are extensive experiments and results to justify the claim. The evaluation uses the LLM-as-judge pipeline as the evaluator for accuracy, and includes F1 and AUROC as metrics. The reliability of the judge itself is also shown. Additionally, the findings align with much existing work (different tasks), making the conclusion credible. 

Overall, I believe this is a solid piece of research, and its conclusion is convincing in the given setting: LLM only, without RAG or external knowledge.

### Strengths
1. Easy to follow and clearly stated. The results are presented in clear, concise language without overemphasis. 

2. The reliability of the LLM-as-judge is evaluated, and its limitation is discussed for transparency. 

3. The white/grey-box pipeline experiments show the correlations between the model's internal states and errors (hallucinations), and I think the design is novel.

### Weaknesses
1. Despite this paper's findings in the QA task, similar conclusions have been stated in other tasks [1,2]. The impression of "contributing new knowledge"  (ICLR reviewing guideline) is not very significant, but this is a subjective view. 

2. The results are presented as bar charts, but they are too dense with information. Using tables while highlighting the difference between average unsupervised methods vs. supervised methods should be better. 


References

[1] Zihan Liu, Yan Xu, Tiezheng Yu, Wenliang Dai, Ziwei Ji, Samuel Cahyawijaya, Andrea Madotto, Pascale Fung. CrossNER: Evaluating Cross‑Domain Named Entity Recognition. AAAI 2021
[2] Jindong Wang, Cuiling Lan, Chang Liu, Yidong Ouyang, Tao Qin, Wang Lu, Yiqiang Chen, Wenjun Zeng, Philip S. Yu. Generalizing to Unseen Domains: A Survey on Domain Generalization. IEEE TKDE 2022

### Questions
What does task-specific domain data do? Suppose such data does not cover the knowledge for a test case. How could it help reduce the hallucination problem of LLM? Does it help its reasoning pattern or lower confidence for uncertain questions?

### Soundness
3

### Presentation
3

### Contribution
3
