# Glance for Context: Learning When to Leverage LLMs for Node-Aware GNN-LLM Fusion

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 8, 4, 4, 6

## Abstract
Learning on text-attributed graphs has motivated the use of Large Language Models (LLMs) for graph learning. However, most fusion strategies are applied uniformly across all nodes and attain only small overall performance gains. We argue this result stems from aggregate metrics that obscure when LLMs provide benefit, inhibiting actionable signals for new strategies. In this work, we reframe LLM–GNN fusion around nodes where GNNs typically falter. We first show that performance can significantly differ between GNNs and LLMs, with each excelling on distinct structural patterns, such as local homophily. To leverage this finding, we propose **GLANCE** (**G**NN with **L**LM **A**ssistance for **N**eighbor- and **C**ontext-aware **E**mbeddings), a framework that invokes an LLM to refine a GNN's prediction. GLANCE employs a lightweight router that, given inexpensive per-node signals, decides whether to query the LLM. Since the LLM calls are non-differentiable, the router is trained with an advantage-based objective that compares the utility of querying the LLM against relying solely on the GNN. Across multiple benchmarks, GLANCE achieves the best performance balance across node subgroups, achieving significant gains on heterophilous nodes (up to +13\%) while simultaneously achieving top overall performance. Our findings highlight the value of adaptive, node-aware GNN-LLM architectures, where selectively invoking the LLM enables scalable deployment on large graphs without incurring high computational costs.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper studies leveraging LLMs to assist with the node classification task and introduces a novel approach: selectively invoking LLMs for specific nodes. Through an empirical study, the authors identify homophily as a key factor in determining whether predictions should rely on LLMs or GNNs. Building on this insight, the paper proposes the GLANCE framework, which integrates a router component for selecting nodes to be processed by LLMs, a fusion component to combine embeddings from GNNs and LLMs, and a reward-based strategy to optimize both components stably. Extensive experiments on citation networks demonstrate GLANCE's improvements for low-homophily nodes and its scalability to larger e-commerce networks. 

This work introduces a promising direction for graph learning, i.e., **selectively invoking LLMs only when necessary** to balance performance and efficiency.

### Strengths
1. **This paper is well-motivated and explores a novel research direction**. It is intuitive that GNNs and LLMs are complementary: GNNs excel in high-homophily scenarios, while LLMs can significantly improve predictions for heterophilic graphs (as supported by prior works like [Ref A]). However, no prior work has investigated the selective usage of LLMs to improve performance for heterophilic nodes while maintaining computational efficiency.

2. **The preliminary experiments are comprehensive and provide meaningful insights**. To determine which nodes should be processed by LLMs instead of GNNs, the authors first analyze structural features like degree and clustering density, then identify homophily as the decisive factor. Finally, they propose a label-free metric, estimated local homophily, to guide the routing. This multi-step analysis provides a clear and logical progression for deriving the proposed factor.

3. **The GLANCE framework is carefully designed to align with the problem's motivation and is grounded in rational principles**. During inference, GLANCE achieves efficiency by using only two feature forward passes: one for the router to decide whether to invoke LLMs and another for combining complementary embeddings. During training, the authors incorporate both task-level prediction loss and a policy-gradient-inspired loss to optimize the routing policy, ensuring robust performance.

4. **The experimental evaluation is extensive and promising**, demonstrating GLANCE's improvement in predictive performance, its controllable computational cost, and its scalability to large-scale graphs. The results convincingly show that GLANCE is effective, particularly for low-homophily nodes.

--- 

[Ref A] Wu, Xixi et al. *When Do LLMs Help With Node Classification? A Comprehensive Analysis*. In ICML, 2025

### Weaknesses
1. While the paper is generally well-written and easy to follow, some minor points require clarification. For example:
     * In Figure 1, the notations, e.g., GCN (Std.), GCN (Enh), LLM, are unclear. What do they represent?
     * In Section 5, the authors state that GNNs and LLMs are pre-trained, but the training pipeline is ambiguous. My understanding is that GNNs and LLMs are first trained separately on each dataset using classification loss, and then these two encoders are frozen during the GLANCE framework's joint optimization of the router and feature refiner. If this understanding is incorrect, clarifying the exact training process would be helpful.

2. While the authors provide a general efficiency study to showcase GLANCE’s computational cost, a more detailed analysis would enhance the paper, e.g., what is the computational complexity of GLANCE during inference, and could the authors provide a breakdown of costs for both training and inference?

3. Although the experiments cover datasets of varying scales, they are still limited in diversity. All four datasets exhibit relatively high homophily, meaning heterophilic nodes form only a small portion. As a result, the overall performance improvements from GLANCE are marginal. Extending the experiments to more heterophilic datasets, e.g., the original arXiv dataset, would better validate GLANCE’s effectiveness for heterophilic scenarios.

### Questions
1. For Weakness 1, is my understanding of the training pipeline correct? Additionally, could the authors define the notations used in Figure 1?

2. Have the authors considered removing the router loss entirely and jointly optimizing the router and refiner solely based on task prediction loss? If so, how does this alternative approach compare in terms of performance and stability?

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
3

### Summary
The paper proposes GLANCE, a selective GNN–LLM fusion for text-attributed graphs. A router decides which nodes “deserve” LLM embeddings; others use the base GNN only. The routing signal mixes estimated homophily, uncertainty, degree, and representation cues. A refiner fuses GNN and LLM features when routed. Training uses a cost-aware counterfactual reward that prefers routes which reduce supervised loss under a budget. Experiments on several TAG benchmarks report stronger accuracy on low-homophily regions while keeping LLM calls sparse. Ablations suggest homophily features drive much of the routing benefit, and sensitivity studies show larger budgets help heterophilous nodes most.

### Strengths
1) New method.
A simple, modular selective GNN–LLM fusion (router + refiner on frozen backbones) that learns when to invoke LLM embeddings under a budget. Clear cost–accuracy tradeoff and easy to plug into existing TAG pipelines.

2) Extensive experiments.
Broad evaluations across multiple text-attributed graph benchmarks, with routing rates, ablations, and sensitivity to budgets. Results consistently show gains where graphs are hard (low homophily) while keeping LLM calls sparse.

3) Interesting task.
Addresses a timely, practical problem: cost-aware reasoning on text-attributed graphs. Selectively augmenting nodes with LLM features makes TAG learning more applicable to real deployments with tight compute limits.

### Weaknesses
**Weakness**

1) **Router vs. eval is entangled.**  
You use homophily to route and also to judge gains. That can create circularity. Show results when the router does **not** use homophily (or uses only degree/uncertainty). Also slice by other structures (roles, motifs, k-core, betweenness). The ablation drop for \(h_v<0.5\) strengthens this concern.

2) **Homophily estimator lacks detail.**  
You build \(\hat h_v\) with an auxiliary classifier \(Q\). Specify \(Q\)’s architecture and size. Report loss, early stopping, and regularization. Clarify train/val label usage and leakage risk. Add calibration metrics (ECE/Brier).

3) **Missing bounds and fuller ablations.**  
Add an **oracle router** (LLM correct, GNN wrong) as an upper bound. Add a **shuffled router** with the same budget as a lower bound. This will contextualize NCS and show headroom. Also run **only-X** and **pairwise** feature ablations for redundancy/synergy. The big drop when removing homophily for \(h_v<0.5\) is not the whole story.


**Suggestions**

1) **Counterfactual reward may be biased.**  
It uses labels and batch top-k, and only for routed items. That can skew learning. Consider off-policy estimates, stochastic routing, or periodic global calibration.

2) **Test LLM and prompt diversity.**  
You rely on one LLM/embedding and one multi-level prompt. Try other sizes/models and one-level vs. multi-level prompts. This separates routing effects from encoder/prompt artifacts.

3) **Disentangle routing vs. feature gains.**  
Add a control that routes “hard” nodes but **does not** call the LLM. Reweight the GNN head only. If gains persist, they may come from attention, not LLM content.

### Questions
1) **Batching for top-k?**  
How do you form batches—random nodes or topology-aware sampling?

2) **Neighborhood sampling details?**  
For ego/1-hop/2-hop text: what sampler, sizes, randomness, seeds, and truncation/token caps?

3) **Refiner capacity and overfitting?**  
Since GNN/LLM are frozen, what is the refiner’s depth/width/parameter count vs. the GNN head? Any overfitting on small graphs (e.g., Cora)?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the inefficiency and uneven performance of uniform GNN-LLM fusion for text-attributed graphs and proposes GLANCE, a node-aware fusion framework. It identifies that GNNs excel at high-local-homophily/high-degree nodes while LLMs outperform on heterophilous/low-degree nodes, with local homophily as a signal for LLM benefit. The proposed approach GLANCE freezes pre-trained GNN and LLM, trains a lightweight router to invoke LLMs on hard nodes selectively, and uses a refiner MLP to fuse embeddings.

### Strengths
- The paper proposes a node-aware GNN-LLM fusion mechanism, targeting the inefficiency of uniform LLM invocation and filling a gap in TAG learning.
- The proposed GLANCE takes local homophily as an LLM-benefit signal, combined with a homophily estimation scheme, balancing practical deployment applicability.
- GLANCE achieves good performance across different graph datasets, while freezing pre-trained models ensures scalability for large-scale TAGs like OGB-Products.

### Weaknesses
- The label-free homophily estimate relies on a trained MLP to predict node labels. This is rather counterintuitive: since this is essentially a node classification task, why is an MLP capable of fulfilling this role instead of a GNN? Furthermore, additional justification is required regarding how this MLP achieves robust performance. For instance, on low-resource data (with few training labels) or high-noise data (with short/noisy node text). If the MLP’s label predictions are inaccurate, will the estimated homophily become a misleading routing signal?
- Given the complex nature of graphs, uncertainty, local homophily, and degree are not necessarily consistent in their implications for label prediction, as also observed in Appendix E.3. Nodes with high homophily/degree may not exhibit low uncertainty (as well as high prediction success); conversely, nodes with low homophily/degree might show high GNN certainty. The manner in which the router interprets these conflicting signals remains an issue of interpretability.
- To control prompt length, GLANCE samples only 5 neighbors per hop for LLM input. This may result in the discard of critical structural information (e.g., a key neighbor of a heterophilous node that explains its label being excluded from sampling).

### Questions
In addition to the questions raised above, I have two more questions to be answered:

- Does the paper employ random sampling rather than structure-aware sampling (such as prioritizing neighbors with high degrees or edge weights)? Could this simplification restrict the LLM’s capacity to capture fine-grained structural correlations?
- Is an adaptive $\beta$ that adjusts based on real-time computational constraints better than a fixed one? Would this affect the tradeoff between heterophilous node accuracy and inference speed?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper argues that current GNN-LLM hybrids waste LLM calls by applying a single fusion rule to all nodes, which hides where LLMs actually help. By stratifying nodes with respect to local homophily and relative degree, the authors show that GNNs win in high-homophily/high-degree regions, while LLMs win in heterophilous/low-degree pockets. They then introduce GLANCE learns a lightweight per-node router to route top-k nodes to the LLM and fuses GNN+LLM via a small refiner head. On four datasets, GLANCE shows the improvement and it does so with a small query budget.

### Strengths
1. The studied problem of deciding when to invoke an LLM for graph node prediction is timely; the observation that LLM benefits are uneven across nodes is clearly argued and well grounded in the analysis.
2. The proposed GLANCE framework is coherent with routing, context construction, and fusion stages reinforcing one another; learning to gate LLM usage from cheap structure aware signals is a sensible and practical design.
3. The reported gains on heterophilous and low degree subsets, together with consistent improvements in overall accuracy under a limited query budget, indicate that selective LLM supervision actually strengthens node level recognition.

### Weaknesses
1. Since GLANCE routes only a subset of nodes to the LLM for cost reasons, it would be useful to report the result which sends every node through the LLM GNN fusion path. Even if this is expensive, such a result could act as an empirical upper bound for the proposed router and show how close the learned routing actually gets to the best case. It would also be helpful to understand the trade-off between budget cost and model performance.
2. Why did the authors include heterophilous GNNs when the datasets in Table 2 are all homophilous, where these models are unlikely to be advantageous? It would be more informative to include genuinely heterophilous or non citation graphs, since the current experiments are mostly on citation networks; this would make the empirical evaluation of the proposed method more convincing.
3. To better understand the quality of the routing decision, it would help to break down accuracy by routed vs non routed nodes. Among the nodes that the router decided to send to the LLM, how many became correct due to LLM input, and how many stayed wrong? Among the nodes that were not routed, how many were already correct under GNN only? This breakdown would show whether the router is actually focusing LLM budget on the hard regions of the graph.

### Questions
1. In Section 5.1.1 it is not fully specified how the GNN backbone F is obtained before it is used inside GLANCE. Could the authors explain how it is pre-trained?

I would be happy to raise the score if the above concerns can be addressed.

### Soundness
4

### Presentation
4

### Contribution
3
