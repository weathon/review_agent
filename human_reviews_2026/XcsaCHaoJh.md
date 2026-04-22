# Asymmetric Synthetic Data Update for Domain Incremental Dataset Distillation

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 4

## Abstract
Dataset distillation (DD) attempts to construct a compact synthetic dataset that serves as a proxy for a large real dataset under a fixed storage budget, thereby reducing the storage burden and training costs.
Prior works assume the full dataset is available upfront which is distilled at once, although real datasets are collected incrementally over time in practice.
To alleviate this gap, we introduce a new problem setting, *Domain Incremental Dataset Distillation*, that continually distills datasets from different domains into a single synthetic dataset.
The conventional DD sequentially processes arriving datasets in order, overwriting the old knowledge with new one, causing catastrophic forgetting problem.
To overcome this drawback, we propose *Asymmetric Synthetic Data Update* strategy that adjusts the per-sample update rates for synthetic dataset while balancing the stability-plasticity trade-off. Specifically, we design a bi-level optimization method based on meta-learning framework to estimate the optimal update rates, which allows each sample to focus on either stability or plasticity, thereby striking a balance between them.
Experimental results demonstrate that our approach effectively mitigates the catastrophic forgetting and achieves superior performance of DD across continually incoming datasets compared with existing methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Domain Incremental Dataset Distillation (DIDD) — a new problem setting where synthetic data must be updated continuously as new domains arrive, under a fixed memory budget.
Unlike conventional dataset distillation, which assumes access to all data at once, DIDD requires the synthetic set to maintain past knowledge while integrating new information, analogous to continual learning but at the *data level* rather than *parameter level*.

To tackle this, the authors propose Asymmetric Synthetic Data Update, a bi-level optimization framework where each synthetic sample learns its own *stability–plasticity* trade-off coefficients. These coefficients adaptively balance gradients from old and new domain objectives, allowing the synthetic dataset to evolve without catastrophic forgetting.

Experiments on R-MNIST, Seq-CORe50, and PACS demonstrate significant gains and reduced forgetting over standard distillation and continual learning baselines. Ablation analyses show the importance of asymmetric updates and bi-level learning for maintaining cross-domain knowledge.

### Strengths
* **Novel problem definition:** The paper bridges the gap between dataset distillation and continual learning, providing a fresh research direction.
* **Conceptually elegant solution:** The asymmetric update mechanism offers an intuitive and interpretable way to balance stability and plasticity at the data level. The per-sample α/β weighting is original and theoretically motivated.
* **Strong empirical gains on small-scale benchmarks:** The proposed method outperforms both conventional dataset distillation methods (e.g., MTT, DSA) and continual learning baselines (e.g., EWC, LwF, MAS), validating the effectiveness of the asymmetric strategy.

### Weaknesses
* **Scalability concerns:** The method relies on *bi-level optimization for each sample*, which is computationally expensive. The current experiments use small networks (3-layer ConvNet) and small datasets (R-MNIST, PACS). It is unclear whether the approach can scale to modern architectures (ResNet, ViT) or larger datasets (CIFAR-100, ImageNet).
* **Lack of large-scale validation:** The paper would be significantly stronger if it included experiments or runtime analysis on medium-scale settings to demonstrate practical feasibility.
* **Limited diversity of tasks:** The benchmarks are all domain-incremental vision tasks; it would be interesting to see if this framework can handle class-incremental or multimodal distillation scenarios.
* **Ablation on efficiency missing:** Although the method is theoretically well-motivated, there is no quantification of time or memory overhead compared to one-shot distillation.

### Questions
1. How does training time scale with dataset size or number of domains? Can the authors provide runtime or complexity analysis?
2. Could the asymmetric update be approximated with parameter-efficient strategies (e.g., low-rank updates, meta-network sharing) to improve scalability?
3. How would the approach behave on *class-incremental* or *multi-modal* tasks rather than domain-incremental ones?
4. Is it possible to replace full bi-level optimization with a single-level approximation or gradient truncation without significant performance loss?
5. Could this framework generalize to larger backbones (e.g., ResNet, ViT) or to non-vision modalities (text or graph distillation)?

### Soundness
3

### Presentation
3

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
The authors identified that the existing dataset distillation assumes the availability of the entire dataset, while in reality, real datasets are collected incrementally over time. To solve this issue, the authors propose Domain Incremental Dataset Distillation to continuously distill datasets from multiple sources into a single synthetic dataset. Additionally, traditional DD methods sequentially process arriving datasets and cause catastrophic forgetting. The authors propose an Asymmetric Synthetic Data Update strategy to balance the stability-plasticity trade-off:  a bi-level optimization approach based on the meta learning framework to estimate the optimal update rates. The authors perform evaluations to demonstrate the mitigation of the catastrophic forgetting problem and the efficacy of DD in their proposed approach.

### Strengths
+ The idea is simple but effective, as demonstrated by the evaluation results that the proposed work achieves the state-of-the-art level performance. 
+ The idea is clearly and effectively formulated in mathematical terms, and the writing is very straightforward and easy to follow.

### Weaknesses
- While there's beauty in simplicity, the idea is \textit{too} simple. For example, Eq 12 is just a standard gradient-based optimization with the defined losses.
- It seems to me (please correct me if I'm wrong) that $\bar{\alpha}_i$ and $\bar{\beta}_i$ are separately optimized yet they are used jointly (Eq 9). Why aren't they updated jointly instead?
- The theoretical evidence on why this method is working is missing. The existing mathematical formulation is simply descriptive of the approach, not highlighting any insights on why the approach works.
- In a paper that is 9 pages, only 2 pages are the methodology. This balance is a little bit off as I was expecting a longer methodology section.

### Questions
Please see "Weaknesses." Mainly I'm concerned why aren't $\bar{\alpha}_i$ and $\bar{\beta}_i$ updated jointly since they are used jointly?

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
4

### Summary
The paper proposed a new problem in dataset distillation: Domain Incremental Dataset Distillation (DIDD). The problem assumed that a sequence of domain-shifted datasets (classification datasets that shared the same label space). The difference between this problem and the Continual Learning is the data storage budget is fixed in DIDD. The authors introduce a stability loss to preserve prior knowledge and an asymmetric per-sample update learned via bi-level meta-optimization. The authors conducted experiments to verify their method's effectiveness.

### Strengths
1. The authors propose a new problem setting of dataset distilaltion.

2. The proposed method seems address the proposed problem well.

### Weaknesses
1. The paper defines a new hybrid setting DIDD by combining dataset distillation and continual learning. However, this formulation appears somewhat artificial and tailored to the proposed method, rather than motivated by a clearly established real-world need or widely recognized benchmark.

2. The paper lacks the baseline of dataset distillation, which condense the accumulated datasets directly. Additionally, the performance comparision is not so fair as the continue learning baselines are not designed for the new DIDD setting. 

3. Even in R-MNIST dataset, there is a huge performance loss compared to the whole dataset.

### Questions
1. What is the definition of Domain Incremental Dataset Distillation (DIDD)? It was defined as a problem in the abstract, but it was also defined as a framework in contributions. 

2. What is the main contribution of this paper? The proposed new problem DIDD? or the new proposed method to address the DIDD problem? 

3. Why are there no baselines listed under dataset distillation? Any dataset distillation (DD) method that condenses an accumulated dataset can serve as a valid baseline.

4. How is the computational cost ? There should be an experiments to dicuss the computational cost.

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
This paper investigates Domain Incremental Dataset Distillation, where data arrives sequentially. The authors propose Asymmetric Synthetic Data Update, which introduces a stable loss that constrains representation during updates. Additionally, they introduce a meta-learning method and regularization term to balance the weights of the two losses and prevent them from growing excessively.

### Strengths
1. The paper is well-written and easy to follow
2. The setting is novel

### Weaknesses
1. Meta-learning makes optimization more difficult.
2. Constraining the label space to be the same makes the setting less general.
3. DD is already challenging to train. Adding meta-learning raises concerns about training instability and tuning difficulty.

### Questions
1. Does the domain sequence order affect the results?
2. Could you provide non-DD fine-tuning results to help us understand how the gap changes between DD and full incremental domain training?
3. What’s the training cost comparing to baselines?

### Soundness
3

### Presentation
3

### Contribution
2
