# GFMate: Empowering Graph Foundation Models with Pre-training-agnostic Test-time Prompt Tuning

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Graph prompt tuning has shown great potential in graph learning by introducing trainable prompts to enhance the model performance in conventional single-domain scenarios. Recent research has extended graph prompt methods to Graph Foundation Models (GFMs), aiming to improve their cross-domain generalisability from source domains to an unseen target domain by tuning auxiliary prompts using few-shot samples. Despite their progress, most existing GFM prompt methods embed domain-specific information from source domains into prompts, which serve either as input to GFMs or encoded during the GFM pre-training process. This entanglement of prompts with specific source domains and particular GFM pre-training strategy restricts their generalisability to target domains and different GFMs. Furthermore, existing methods merely rely on few-shot data for prompt tuning, neglecting the rich information in unlabelled target domain test data. Motivated by these insights, this paper aims to empower GFMs with a pre-training-agnostic test-time graph prompt tuning framework, named GFMate. GFMate introduces a centroid prompt and a layer prompt applied after pre-training on target domains, avoiding entanglement with the source domains and model pre-training. In addition, a test-time complementary learning objective is devised to exploit both labelled and unlabelled target domain data for effective test-time prompt tuning. Extensive experiments on 12 benchmark datasets across diverse domains demonstrate the superior performance and efficiency of GFMate, achieving improvements of up to 30.63\%. Code will be released upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes GFMate, a framework for pre-training-agnostic test-time prompt tuning of GFMs. The key idea is to decouple prompt optimization from pre-training, introducing two lightweight prompts that are tuned only at test time. Additionally, a Test-time Graph Complementary Learning objective leverages both few-shot labeled and unlabeled samples through complementary labeling to improve target-domain generalization. Experiments on twelve benchmark datasets claim up to 30% improvement over existing GFM prompt methods with large gains in efficiency.

### Strengths
1. The paper tackles an important problem: cross-domain generalization of GFMs without re-pretraining, and positions itself within an emerging research trend of test-time adaptation for graphs.

2. The method is conceptually simple and computationally light, requiring only prompt updates without retraining the GFM backbone.

3. The empirical section is extensive, covering multiple datasets, backbones, and pre-training objectives.

4. The writing is clear and the figures provide an intuitive illustration of the pre-training-entanglement problem and the proposed test-time workflow.

### Weaknesses
1. Conceptual novelty is overstated. The claimed “pre-training-agnostic” feature is somewhat misleading: most existing few-shot or prompt-based GFM methods (e.g., SAMGPT, MDGPT) also freeze the backbone and do not require coupling with pre-training. The difference between few-shot fine-tuning and test-time tuning is incremental.

2. The paper repeatedly asserts that prior prompts are “pre-training-entangled” but does not empirically demonstrate that this entanglement actually harms transferability.

3. The theoretical contribution (the “Excess Risk Bound”) offers no genuine insight into the TGCL mechanism. It restates a generic Rademacher complexity bound without linking assumptions to the graph setting or validating them empirically.

4. The role of complementary labeling is weakly justified. The entropy-based pivot-layer strategy is heuristic, and there is no analysis comparing it with pseudo-labeling or entropy minimization baselines.

5. Some comparisons seem selective: BRIDGE (ICML 2025), a closely related multi-domain GFM with generalization guarantees, is discussed in related work but not included in experiments, which undermines the completeness of evaluation.

6. The datasets used are small-scale academic benchmarks (Cora, Citeseer, Texas, etc.), which makes it unclear whether the approach scales to true “foundation” settings.

### Questions
1. What concrete evidence shows that “pre-training entanglement” limits cross-domain generalization? Could you provide an ablation where pre-training and prompt learning are decoupled within SAMGPT or MDGPT for comparison?

2. How sensitive is the TGCL loss to noisy complementary labels? Have you compared it with simple entropy minimization or self-training?

3. Can the proposed method handle real large-scale GFMs (e.g., GraphMAE, OpenGraph) or text-attributed graphs, or is it limited to small GNN-based backbones?

4. Since GFMate tunes prompts on unlabeled test data, how do you avoid potential label leakage or overfitting to test distribution shifts?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a pre-training-agnostic test-time graph prompt tuning framework, named GFMate. It consists of centroid prompt and a layer prompt to align the distribution shift between the source graph and target graph and exploit the rich neighborhood information from unlabelled node, respectively.

### Strengths
1. The extensive experiments demonstrate the effectiveness and efficiency of the proposed method.
2. The presentation of the proposed method is good and the paper is easy to follow.
3. This paper provides the generalization bound of test-time learning loss.

### Weaknesses
1. I am not fully convinced by the design of test-time graph complementary learning. The authors mention that the test-time learning loss encourages centroids to be distant from testing samples being predicted to the most dissimilar class, which might be useful when the number of classes is not large, such as 2-class or 3-class. Table 2 and Table 17 show the performance comparison in the node classification and graph classification tasks, respectively. These results in 1-shot setting show that the improvement of the proposed method over other baseline methods in binary graph classification task (e.g., around 4%) is less significant than that in the node classification task (5-8 classes classification). Intuitively, if the design of test-time graph complementary learning can remove one wrong answer (by pushing the centroid away from the most dissimilar class), the performance improvement in the binary class task can be greatly improved compared with the 5-class classification task.
2. The experimental setup is not quite clear. See question below.
3. In the introduction, the authors claim that existing GFM prompt designs are generally entangled with pre-training on source domains and cannot easily generalize to unseen target domains. This statement seems overstated, as several existing methods have incorporated mechanisms to adapt to target domains. For instance, MDGPT [1] introduces dual prompts, consisting of a unifying prompt and a mixing prompt, to enable adaptation to target domains by leveraging both unified multi-domain knowledge and tailored mixtures of domain-specific knowledge. Then, the authors assert that the domain-specific information learned in the prompts cannot be effectively transferred from source to target domains due to substantial differences in structural and feature patterns. If figure 3 is used to validate the claim, the authors should visualize both the source graph and the target graphs by GFM prompts methods **explicitly**. However, current version of figure 3 fail to convey the misalignment between test distribution and the training distribution. This makes the first challenge less important.
4. Although the paper presents a generalization bound, the authors do not provide any substantive discussion or interpretation of the theoretical analysis. It remains unclear how the derived gap contributes to understanding the model’s generalization capability, particularly under domain shift or prompt adaptation scenarios. The absence of theoretical insights, such as which factors (e.g., number of classes?) most influence the gap, makes this analysis appear superficial. Without further explanation or empirical validation linking the theoretical findings to observed performance, the inclusion of the generalization gap offers limited theoretical or practical value to the paper’s overall contribution.
5. The code is not provided.

[1] Xingtong Yu, Chang Zhou, Yuan Fang, and Xinming Zhang. Text-free multi-domain graph pretraining: Toward graph foundation models. arXiv preprint arXiv:2405.13934, 2024c.

### Questions
1. The experimental setup is not quite clear. In Figure 3, which graph is the GFM pretrained on? Cora? Which GFM method is used for visualization? In table 1, which graphs are used for model pretraining?
2. I am not fully convinced by the design of test-time graph complementary learning. The authors mention that the test-time learning loss encourages centroids to be distant from testing samples being predicted to the most dissimilar class, which might be useful when the number of classes is not large, such as 2-class or 3-class. Table 2 and Table 17 show the performance comparison in the node classification and graph classification tasks, respectively. These results in 1-shot setting show that the improvement of the proposed method over other baseline methods in binary graph classification task (e.g., around 4%) is less significant than that in the node classification task (5-8 classes classification). Intuitively, if the design of test-time graph complementary learning can remove one wrong answer (by pushing the centroid away from the most dissimilar class), the performance improvement in the binary class task can be greatly improved compared with the 5-class classification task.

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
The paper proposes GFMate, empowering GFMs with a pre-training-agnostic test-time graph prompt tuning framework.

### Strengths
1. It's important to explore GFM.
2. The paper is well motivated. It's reasonable to leverage abundant unlabeled data.
3. Centroid shifting and layer re-weighting are simple, efficient, and intuitive.

### Weaknesses
1. Pivot-layer selection in TGCL may be unstable under data noise or heterophily; sensitivity analysis could be stronger.
2. Assumes full transductive access to the test graph; performance under inductive or streaming settings is unclear.

### Questions
See weaknesses.

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
3

### Summary
This paper tackles two major limitations of adapting Graph Foundation Models (GFMs) to unseen target domains: (1) existing prompt tuning methods are tightly coupled with pre-training, limiting generality, and (2) few-shot supervision alone cannot capture target-domain distribution, leaving unlabeled test nodes under-utilized. To address these issues, the authors propose (1) **pre-training-agnostic test-time prompt tuning**, which decouples prompt learning from pre-training, enabling flexible adaptation, and (2) **test-time graph complementary learning**, which leverages unlabeled target nodes to mitigate distribution shift. Together, these components significantly improve GFM adaptation under scarce labels.

### Strengths
1. The proposed approach achieves **substantial performance gains** while maintaining superior efficiency compared to existing cross-domain GFM methods.

2. The **experimental evaluation is thorough**, covering 12 benchmark datasets with extensive comparisons, ablation studies, efficiency analyses, and compatibility tests, providing strong empirical evidence for the method’s effectiveness.

3. The paper **is clearly written and logically structured**, making the methodology and insights easy to follow.

### Weaknesses
1. The paper's **motivation is not sufficiently sound**. Specifically, the claim that the "pre-training–entangled" nature of existing GFM prompt designs is a significant disadvantage is not thoroughly substantiated. Intuitively, given that graph data distributions and node behaviors can vary significantly across domains, distinguishing between them during training (e.g., via domain tokens or vectors) appears to be a reasonable approach. The theoretical support for the idea that this "entanglement" is harmful is currently insufficient, which makes the core premise confusing, even in light of the strong experimental results.
2. The authors attempt to elaborate on the problem in the PRELIMINARY section, but the explanation remains unclear. It is axiomatic that node behaviors differ across domains, which necessitates the injection of domain-related information. The paper would be strengthened if the authors could emphasize **the distinction between injecting domain information at *pre-training time* versus *test-time*.** A thorough discussion of the advantages and disadvantages of each approach is needed.

3. The experimental section **lacks crucial implementation details**, making reproducibility difficult. The following key information appears to be missing:  (1) For the *Single-domain Training and Testing baselines*: The specific training tasks used and the methodology for the training, validation, and testing splits. (2) For all *GFM* baselines: The specific pre-training tasks and datasets (i.e., the source graphs) that were used.

### Questions
1. Why would ignoring domain information during pre-training be beneficial?
2. During pre-training, given the different data distributions of source domains, would omitting domain information lead to training instability?
3. The datasets vary significantly in size. Is it possible that a single large source domain dominates the pre-training process?
4. What is the specific meaning of "different-hop neighbourhood aggregation accuracy" in Figure 2?

### Soundness
2

### Presentation
3

### Contribution
2
