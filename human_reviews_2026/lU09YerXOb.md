# Arboreal Neural Network

- Decision: Reject
- Scores: 6, 4, 8, 2

## Abstract
Connectionist models and symbolic models have long embodied two divergent paradigms: the former excel at differentiable representation learning yet struggle with transparency, while the latter deliver explicit rule-based reasoning but resist gradient-based optimization. We introduce Arboreal Neural Networks (ArbNN), a neural–symbolic framework that unifies these paradigms both computationally and conceptually.
At the design level, ArbNN departs fundamentally from prior neuralized-tree models through a depth-aware routing mechanism and a topology-informed softmax aggregation, which together enable one-shot multi-path gradient propagation and consequently achieving rapid and well-conditioned optimization dynamics and high parallel inference efficiency.
At the conceptual level, ArbNN reveals that decision-tree branching and self-attention routing are two realizations of the same conditional computation primitive. We prove a structural isomorphism between a decision tree and a single-query attention head, enabling a differentiable architecture that faithfully preserves symbolic decision logic.
A key property of ArbNN is Bidirectional Fidelity, ensuring that the neural module can be compiled from—and losslessly decompiled back into—a symbolic tree, yielding both ordering consistency in ranking behavior and explicit, auditable interpretability via reconstructed if–else rules. ArbNN further supports GBDT-based initialization, allowing it to inherit strong inductive biases and integrate seamlessly with existing production workflows.
Empirically, ArbNN achieves state-of-the-art performance on various public tabular benchmarks and delivers consistent gains under temporal distribution shift in large-scale industrial credit-risk systems. To support realistic evaluation, we additionally contribute TabCredit, a feature-rich, temporally partitioned dataset built from millions of real-world loan applications. Together, these results demonstrate that ArbNN forms a unified, reversible, and practically deployable bridge between symbolic reasoning and neural computation for high-stakes tabular domains.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper introduces Arboreal Neural Networks (ArbNN), a differentiable architecture that bridges gradient-boosted decision trees and neural networks. The key idea is to encode a pretrained XGBoost model into a neural form by translating its structure—feature splits, thresholds, and leaf values—into matrix operations that can be optimized end-to-end. This allows the model to retain the interpretability and inductive bias of trees while gaining the flexibility of gradient-based learning. Experiments on eight public tabular datasets and one large industrial credit dataset (TabCredit) show that ArbNN consistently matches or outperforms strong baselines.

### Strengths
The paper proposes a novel and well-structured idea that combines the structural bias of decision trees with the flexibility of neural networks. The concept is intuitive yet original, and the formulation is clearly presented.

The writing is clean and logically organized, making the technical details easy to follow. The experiments are thorough within the chosen scope and demonstrate consistent improvements over strong baselines such as XGBoost.

### Weaknesses
- **Limited Benchmark Coverage**

The evaluation includes only eight public datasets, which is considerably below the current standard in the tabular learning community. This narrow benchmark scope limits the credibility of the claimed generalization. Given the model’s conceptual promise, it would be valuable to test ArbNN on a broader set of heterogeneous tabular tasks.

- **Unclear Motivation and Overemphasis on Industrial Data**

The paper’s motivation is not fully convincing. Although the central idea—learning the structural bias of trees—is conceptually interesting, the claimed interpretability advantage remains unsubstantiated, as XGBoost provides only limited transparency. It appears that the work may be driven by a specific industrial objective, possibly related to the proprietary dataset used. If so, this motivation can be stated explicitly and the framing adjusted accordingly. Clarifying how the industrial requirements connect to the model’s broader scientific contribution and analyze why previous dl models perform worse, would significantly strengthen the paper’s coherence and impact.

### Questions
See Above.

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
The paper proposes Arboreal Neural Networks (ArbNNs), a framework that converts trained decision trees into differentiable neural operators called ArborCells. Each ArborCell encodes a tree’s split features, thresholds, structure, and leaf values into four matrices/vectors, enabling end-to-end optimization while preserving the original tree semantics.

### Strengths
1. A differentiable “tree-as-layer” formulation (ArborCell) with explicit feature–node selection matrix $W$, split-threshold vector $f$, tree-structure / routing matrix $P$, and leaf-value vector $v$ that avoids path-probability products via one-shot matrix aggregation
2. An algorithm to parse trees into ArborCells and the ability to decompile trained ArborCells back to refined trees, maintaining symbolic interpretability
3. Competitive performance on public tabular tasks and consistent vintage-curve improvements over XGBoost on TabCredit under temporal drift
4. Introduction of TabCredit, an industrial credit-risk dataset with temporal splits to benchmark robustness and interpretability in realistic settings

### Weaknesses
1. The experimental section does not include comparisons with strong, modern baselines, especially tabular foundation models.

2. Limited gains over XGBoost in Table 2 relative to method complexity. On the reported datasets, the improvement over a well-tuned XGBoost baseline is small.

3. The paper evaluates on a relatively small set of benchmarks

4. Dependence on pretrained tree models for initialization. The core recipe assumes the availability of a strong GBDT (XGBoost/LightGBM) to parse into ArborCells. This limits applicability in settings where (i) trees are hard to train well, or (ii) one would like to learn the structure jointly with the downstream objective. The paper does not show a convincing “from-scratch ArbNN” alternative.

### Questions
1. Can the authors add comparisons with recent tabular foundation models (e.g., TabPFNv2 [1], TabICL [2])?

2. Can the authors clarify the necessity of GBDT-based initialization? The current version treats “compiling from a strong GBDT” as a given prerequisite, but there is no experiment demonstrating whether ArbNN can still achieve comparable performance.

3. Can the authors provide more detail on scalability and serving? Since each ArborCell does a one-shot aggregation over all leaves, how does inference time and memory compare to the original XGBoost model. A brief complexity analysis or inference time comparison would make the method more practical.

[1] Hollmann, Noah, et al. "Accurate predictions on small data with a tabular foundation model." Nature 637.8045 (2025): 319-326.  
[2] Qu, Jingang, et al. "Tabicl: A tabular foundation model for in-context learning on large data." ICML 2025

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
1

### Summary
This paper addresses the lack of tree-structured inductive bias in deep neural networks for tabular data. To this end, the authors propose ArbNN, a novel architecture that reformulates decision trees into differentiable neural modules, enabling end-to-end gradient optimization while preserving interpretability. Extensive experimental results on multiple public benchmarks and a large-scale industrial credit risk dataset demonstrate that ArbNN consistently outperforms both traditional tree-based models and neural baselines, achieving superior accuracy and interpretability in tabular learning tasks.

### Strengths
* This paper proposes the ArborCell structure to introduce the inductive bias of decision trees, and I am happy to see that the authors also provide visual comparisons to demonstrate the interpretability of the proposed method.
* The authors discuss the related literature in considerable detail.
* The paper is well written and easy to follow.

### Weaknesses
1. I am not an expert in tabular data, but I am curious about the convergence behavior of the proposed ArbNN. Could the authors provide training curves and compare them with other networks to illustrate convergence stability?
2. How does the training cost of the proposed method compare to other baselines? In addition, please evaluate the computational efficiency during inference, e.g., in terms of FLOPs, memory usage, and inference time.
3. The figures contain text that is too small to read clearly. It is recommended to increase the font size, use vector graphics for better clarity, and include a complete schematic diagram of the model architecture.
5. The authors do not provide code for reproducibility checks.

### Questions
My questions are in Weakness Section.

### Soundness
3

### Presentation
2

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
The paper proposes a new architecture for tabular data that is based on the idea of converting decision trees to a particular variation of two matrix multiplications with a non-linearity. It proposes to initialize such trees with XGBoost and then finetune the thresholds and values. The paper also proposes a new credit scoring dataset TabCredit. The method is tested on the new dataset and a simple benchmark constructed from pytorch-frame, claiming state-of-the-art performance.

### Strengths
I think that looking into tree-structured models and combining their inner workings with DL models is an interesting pursuit. I had a great time digging through related work on the topic and think that there is something in this line of work that could lead to strong and interpretable models and this direction is currently underexplored.

The dataset contribution also seems very timely and important as there are not a lot of realistic testbeds for tabular machine learning methods readily available in academia. When done right this is a major contribution, so I encourage authors to go through with it regardless of this review period decision.

### Weaknesses
At times the writing is very hard to make sense of. In the related work section, for example, I still can't make sense of how challenging instances in datasets are related to the pre-tuned default hyperparameter configurations (lines 91-93). The overall algorithm for constructing an "ArborCell" may also be improved I believe (see the next point for examples).

I believe the paper does not fully cover the relevant related work. It packages an idea of decision tree inference in matrix form into an "ArborCell", but this idea seemed not novel, and there are very similar existing approaches indeed:
- https://arxiv.org/abs/1604.07143 - Neural Random Forests. Which seems to do exactly what authors propose here
- https://blog.dailydoseofds.com/p/transform-decision-tree-into-matrix a blog post example, which does a better job in explaining the same procedure which is used in the paper

Finally, I do not believe the results are solid as there are some indications of poorly tuned baselines. Like TabM (recent SoTA model) performing on par with or sometimes worse than an MLP, or some large performance gains over XGBoost just from tuning the thresholds and leaf values (may indicate poorly tuned XGBoost in the first place). I also had trouble understanding some of the results like which datasets exactly were used (e.g. what dataset is CH?, why JA - Jannis is seemingly binclass and not multiclass as it is in the pytorch-frame benchmark). Without code being available this is impossible to check further. I suggest the authors compare to an established and well tuned set of baselines, you can take TabArena benchmark which publishes reference model scores in a csv on github:

```python
import pandas as pd
pd.read_parquet("https://tabarena.s3.us-west-2.amazonaws.com/results/df_results_leaderboard.parquet")
```

Comparing the method to a correct set of baselines would increase reliability of the results very much.

### Questions
See suggestions in weaknesses.

Regarding the newly introduced dataset. Does it have a dedicated train/val/test split which is time-based? Or is it different? Can you provide more details regarding the evaluation and tuning setup on the new dataset?

### Soundness
1

### Presentation
2

### Contribution
2
