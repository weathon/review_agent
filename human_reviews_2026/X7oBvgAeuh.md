# FairGRPO: Fair Reinforcement Learning for Equitable Clinical Reasoning

- Decision: Reject
- Scores: 4, 6, 4, 6

## Abstract
Medical artificial intelligence systems have achieved remarkable diagnostic capabilities, yet they consistently exhibit performance disparities across demographic groups, causing real-world harm to underrepresented populations. While recent multimodal reasoning foundation models have advanced clinical diagnosis through integrated analysis of diverse medical data, reasoning trainings via reinforcement learning inherit and often amplify biases present in training datasets dominated by majority populations. We introduce Fairness-aware Group Relative Policy Optimization (FairGRPO), a hierarchical reinforcement learning approach that promotes equitable learning across heterogeneous clinical populations. FairGRPO employs adaptive importance weighting of advantages based on representation, task difficulty, and data source. To address the common issue of missing demographic labels in the clinical domain, we further employ unsupervised clustering, which automatically discovers latent demographic groups when labels are unavailable. Through comprehensive experiments across 7 clinical diagnostic datasets spanning 5 clinical modalities across X-ray, CT scan, dermoscropy, mammography and ultrasound, we demonstrate that FairGRPO reduces predictive parity by 27.2% against all vanilla and bias mitigated RL baselines, while improving F1 score by 12.49%. Furthermore, training dynamics analysis reveals that FairGRPO progressively improves fairness throughout optimization, while baseline RL methods exhibit deteriorating fairness as training progresses. Based on FairGRPO, we release FairMedGemma-4B, a fairness-aware clinical VLLM that achieves state-of-the-art performance while demonstrating significantly reduced disparities across demographic groups. Our code, models, and fairness evaluation framework are publicly available at this anonymous link: https://anonymous.4open.science/r/fairness_submission-D923/.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
In this manuscript, the authors focus on an important but under-explored task, fairness in medical artificial intelligence systems. The authors aim to improve the fairness and effectiveness during the reinforcement learning phase of vision language models. And they propose to modify the advantages to address the limitations. Through experiments on 7 clinical diagnostic datasets with 2 representative VLMs, the authors demonstrate the effectiveness of their proposed method.

### Strengths
### **Strengths**
1. This manuscript focuses on a under-explored but important task, mitigating unfairness in the medical reasoning.

2. The experimental datasets are comprehensive. The authors collect 7 clinical diagnostic dataset spanning 5 clinical modalities, which is can adequately verify whether the method is effective.

3. The experimental section considers a comprehensive set of dimensions, including effectiveness, fairness, efficiency, case study, and others.

### Weaknesses
### **Weaknesses**

1. My primary concern regarding this paper is the lack of sufficient novelty in the proposed approach. The modification of advantages is easy and lack of theoretical analysis. And the clustering is a common approach in fairness machine learning without demographic information.

2. The authors conduct experiments on Qwen-2.5-VL-7B and MedGemma-4B. Although these two models are representative, it is better to conduct more experiments on more VLMs.

### Questions
### **Questions**

1. Could the authors provide more theoretical or empirical analysis of the proposed method/motivation?

2. How about the performance of the proposed FairGRPO on the other mainstream VLMs?

3. Could the authors provide more case studies to prove the importance of fairness in medical AI systems?

### Soundness
3

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
Modern medical AI systems may show performance disparities wrt demographic groups (e.g., by race, gender, age) because training data is heavily skewed toward majority groups. This paper addresses fairness in reinforcement learning for clinical reasoning. The paper proposes a method that weighs samples according to the representation of the demographic group (underrepresented groups get higher weight),  task difficulty, and data source. Whenever demographic labels are not available, clustering is used to create implicit demographic groups. The technique is used to train a multi-modal clinical model, FairMedGemma-4B and its performance is analyzed for different clinical tasks. The performance across demographic groups tends to be more homogeneous. In addition, the fairness continually improves during training.

### Strengths
One of the first works to embed group fairness considerations into the RL for clinical reasoning models

### Weaknesses
Not clear how the group fairness influences individual performance, which is what is most important.

### Questions
Do you have an intuition on how the group fairness policies affect individual metrics of fairness?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes FairGRPO, a reinforcement learning algorithm that mitigates demographic bias in clinical vision–language models. FairGRPO adaptively re-weights learning signals based on representation, task difficulty and data source, and uses unsupervised clustering to infer latent demographic groups when labels are missing. Evaluated on seven multimodal clinical datasets with Qwen-2.5-VL-7B and MedGemma-4B, the method improves both fairness and accuracy over RL and fairness mitigation baselines.

### Strengths
1. This paper addresses fairness in reinforcement learning for multimodal clinical models, a critical but underexplored area. By focusing on fairness in critic-free RL (e.g., GRPO-style optimization), the work bridges the gap of fair ML for healthcare communities.
2. The proposed method of scaling advantages inversely by group representation and task difficulty is simple to compute and adds negligible runtime overhead.
3. Experiments span seven clinical datasets covering diverse imaging types, which demonstrates the generalization ability of FairGRPO.
4. The paper presents fairness trajectories during optimization, showing that FairGRPO progressively improves fairness instead of degrading it, as seen in baseline RL methods. The qualitative case studies vividly demonstrate how fairness-aware training improves the model’s reasoning trace and diagnostic correctness.

### Weaknesses
1. The theoretical grounding for the proposed fairness optimization is limited. It is unclear whether the scaling guarantees convergence or prevents overcompensation.
2. The paper employs K-means clustering on reward vectors to infer latent demographic groups, but does not analyze the robustness of this clustering step. The number of clusters, initialization, and metric choice could influence group assignments, potentially introducing instability or even amplifying unintended biases in underrepresented groups. 
3. The design of group discovery lacks interpretability. The relation between the learned task-specific difficulty patterns and demographic groups is unclear. Adding some case studies may help validate the design of the group discovery.
4. In Table 2, the improvement in fairness and task performance is limited and inconsistent. On MedGemma-4B, for fairness-unaware baselines, there is little or no fairness improvement compared with vanilla GRPO in EOD and FPR diff. For fairness-aware baselines, there is little or no task improvement compared with GRPO+DRO in Acc and F1. Some discussion of such observations may be added.

### Questions
1. See some questions in Weaknesses.
2. This paper focuses on fairness in medical AI systems, but it gives limited attention to how FairGRPO’s reasoning outputs could integrate with real-world clinical workflows. Including a discussion of interpretability would make the contribution more actionable for medical AI practitioners.
3. I did not find the released FairMedGemma-4B. Maybe I'm missing something?
4. Typo in line 370 "faieness" -> "fairness"

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces Fairness-aware Group Relative Policy Optimization (FairGRPO), a reinforcement learning approach designed to train multimodal reasoning foundation models with reduced performance disparities across demographic groups. The method aims to mitigate biases and promote equitable clinical decision-making.

FairGRPO extends GRPO by introducing an adaptive importance-weighting mechanism that normalizes rewards using inverse temperature scaling, where the scaling factor depends on each group’s representation size and performance. Groups can be either explicitly defined (via demographic labels) or implicitly discovered. In the latter case, the authors apply K-means clustering on reward-based representations, selecting the number of clusters using the elbow method. The training objective follows GRPO’s policy gradient formulation with clipped importance sampling.

The authors also release FairMedGemma-4B, a fairness-aware clinical vision–language model trained with FairGRPO. It achieves state-of-the-art diagnostic performance while significantly reducing disparities across demographic groups, as demonstrated on seven clinical datasets spanning five imaging modalities.

### Strengths
•	The paper addresses fairness in reinforcement learning for multimodal foundation models, a highly relevant problem in clinical reasoning where demographic disparities can have critical implications.

•	The presentation is clear and easy to follow, with well-explained formulations and experimental design.

•	FairGRPO extends Group Relative Policy Optimization with an adaptive importance-weighting mechanism that normalizes rewards via inverse-temperature scaling based on group size and performance. The approach is conceptually sound and well motivated by recent fairness literature.

•	The authors also consider cases where explicit demographic labels are unavailable, proposing a clustering strategy on reward-based representations to infer latent groups.

•	Experiments on seven clinical datasets spanning five imaging modalities show consistent fairness gains and competitive or superior accuracy. The release of FairMedGemma-4B, a fairness-aware clinical vision–language model, further strengthens the paper’s practical contribution.

### Weaknesses
1. The clustering-based grouping used when demographic labels are unavailable is intuitive but not further analyzed. It remains unclear what the clusters capture in practice, or under which conditions they would align with meaningful demographic or clinical subgroups. 

2. The proposed FairGRPO objective is reasonable and empirically effective, but a theoretical analysis or discussion of the training objective and convergence behavior would provide a deeper understanding of its effect, rather than relying solely on intuition and empirical evidence.

3. Potential overfitting or generalization issues when up-weighting minority groups are not discussed. It would be helpful to analyze whether the scaling mechanism might amplify noise in small or underrepresented populations.

4. Table 2 does not include standard deviations. Also, a breakdown of performance for each dataset (similar to Table 2 but disaggregated) would improve transparency.

5. The information in Tables 4–17, which report dataset- and group-level metrics, is difficult to parse in its current format, even though these results seem to be relevant to understand the performance at a dataset/group level for each method. Could these be summarized?

### Questions
Please, see weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
