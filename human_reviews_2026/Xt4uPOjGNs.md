# Adaptive Dual Prompting: Hierarchical Debiasing for Fairness-aware Graph Neural Networks

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 2, 6

## Abstract
In recent years, pre-training Graph Neural Networks (GNNs) through self-supervised learning on unlabeled graph data has emerged as a widely adopted paradigm in graph learning. Although the paradigm is effective for pre-training powerful GNN models, the objective gap often exists between pre-training and downstream tasks. To bridge this gap, graph prompting adapts pre-trained GNN models to specific downstream tasks with extra learnable prompts while keeping the pre-trained GNN models frozen. As recent graph prompting methods largely focus on enhancing model utility on downstream tasks, they often overlook fairness concerns when designing prompts for adaptation. In fact, pre-trained GNN models will produces discriminative node representations across demographic subgroups because the downstream graph data itself contains inherent biases in both node attributes and graph structures. To address this issue, we propose an Adaptive Dual Prompting (ADPrompt) framework that enhances fairness for adapting pre-trained GNN models to downstream tasks. To mitigate attribute bias in graph prompting, we design an Adaptive Feature Rectification module that learns customized attribute prompts to suppress sensitive information via a self-gating mechanism at the input layer, thereby reducing biased inputs at the source. Afterward, we propose an Adaptive Message Calibration module that generates structure prompts at each layer, which adjust the messages from neighboring nodes to enable dynamic and soft calibration of the information flow. In the end, ADPrompt optimizes these two prompting modules using a joint optimization objective for adapting the pre-trained GNN model while enhancing model fairness. We conduct extensive experiments on four datasets with four pre-training strategies to evaluate the performance of ADPrompt. The results demonstrate that our proposed ADPrompt outperforms seven baseline methods on node classification tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces ADPrompt, a fairness-aware prompting framework for adapting pre-trained GNNs to downstream node classification while improving group fairness. It comprises two modules: Adaptive Feature Rectification (AFR), which gates feature dimensions via learnable attribute prompts to suppress sensitive information, and Adaptive Message Calibration (AMC), which injects edge-specific structure prompts at each layer to calibrate message passing. Experiments on four datasets and four pre-training strategies show higher accuracy with lower $\Delta\mathrm{EO}/\Delta\mathrm{SP}$ than seven baselines.

### Strengths
+ The modular method is compatible with frozen backbones. AFR and AMC are lightweight prompts on features and messages, easy to add to existing GNNs.
+ The theoretical results are tied to design. The $\Delta\mathrm{GSP}$ upper bound links AFR to reduced initial bias and AMC to damped propagation amplification.
+ Experiments are comprehensive. Four datasets $\times$ four pre-training schemes with seven baselines demonstrate the method's effectiveness.

### Weaknesses
- The work is restricted to binary $y$ and a single binary $s$. How about multi-class or multi-attribute evaluation?
- AMC learns an edge-specific prompt $e^{(l-1)}_{ij}$ at each layer, implying $\mathcal{O}(|E|\cdot d \cdot L)$ memory/compute overhead. The paper does not report runtime/memory comparisons with baselines.
- The fairness bound relies on Lipschitz assumptions and multiplicative factors $\tilde{\gamma}^{(l)}, \tilde{\epsilon}^{(l)}$, but the paper provides no estimators or empirical diagnostics for these constants. It is unclear how the training losses control the bound.
- Individual fairness is not assessed, though structural edits may alter local similarities.
- Potential error amplification from mislabeled $y/s$ is not analyzed.

### Questions
Please refer to the above weaknesses.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes Adaptive Dual Prompting (ADPrompt), a fairness-aware graph prompting framework for adapting pre-trained GNN backbones to downstream node classification while improving group fairness. It introduces two complementary prompt modules: (i) Adaptive Feature Rectification (AFR), a self-gated attribute prompt that suppresses sensitive information at the input; and (ii) Adaptive Message Calibration (AMC), edge- and layer-specific structure prompts to softly calibrate message passing. A min–max objective combines supervised training with an adversary predicting sensitive attributes from the prompted representations. Theoretical analysis bounds group disparity across layers, and experiments on four datasets demonstrate the effectiveness of the proposed method compared to existing baselines.

### Strengths
1. The idea is straightforward and easy to follow.

2. The theoretical analysis is provided.

3. The experimental results across four datasets demonstrate the empirical effectiveness of the proposed method.

### Weaknesses
1. The writing is unclear and can be largely improved. Specifically,  (i) in Section 1 (Introduction), the authors fail to claim why we need this proposed method instead of existing fairness graph prompt methods, such as [1]. The challenges mentioned in this section are merely some well-known fairness issues of GNN, making the reasons to design the proposed method unclear; (ii) The contributions mentioned in Section 1 should also be largely rewritten. The first two points are literally the same thing.

2. In Section 4 (Methodology), while some limitations of existing methods are mentioned, these points are not convincing. For example, the author claims that FPrompt [1] may disrupt critical topological information. However, it is unclear how and why it will disrupt critical topological information and lead to a performance drop. I can only find the underlying reason when I read Section 4.2. The authors are suggested to rewrite this section to make it clearer to readers. In addition, if possible, please add some preliminary empirical results to validate the claims. And I think the above clarification should also be mentioned in Section 1.

3. While theoretical analysis is provided, the statement in Section 5.2 is not clear enough to conclude Theorem 1. I understand that this subsection aims to give proofs and propose Theorem 1 to validate the effectiveness of the proposed method. However, a detailed proof for Theorem 1 should be included. Otherwise, it is unclear how Theorem 1 comes from and why Eq. (16) can lead to Theorem 1.

4. Computational overhead. AMC learns edge- and layer-specific structure prompts, which may impose significant memory and time costs on dense graphs or deep GNN backbones. The paper omits complexity analysis and runtime/memory profiling relative to baselines. 

5. It seems that the framework presumes a binary sensitive attribute is known for all nodes. The AFR and adversary rely on this supervision. Real-world graphs often have missing or multi-valued sensitive labels. More discussion and analysis on this scenario can benefit this paper.

6. Current ablation studies focus on removing AFR or AMC. However, it is also important to investigate the sensitivity of the hyperparameter,s such as $\lambda$, and the transferability of the proposed method. And the impact of layer numbers of GNNs is also important to investigate the effectiveness of the layer-specific techniques.

7. Too many critical analyses are put into Appendix. The authors are suggested to reorganize this paper such that some important analysis can be in the main text.

[1] Fairness-aware prompt tuning for graph neural networks. WWW 2025.

### Questions
1. Scalability analysis. Can the authors provide runtime/memory comparisons between ADPrompt and GPF/FPrompt on large graphs? Are there ways to sparsify AMC prompts (e.g., top‑k neighbors or low-rank factorization)?

2. Transferability of prompts. If prompts are learned on one dataset or under one pre‑training method, can they be transferred to another domain or backbone without retraining? Preliminary results would be interesting.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper presents ADPrompt, a fairness-aware prompting framework for adapting pre-trained GNNs to downstream tasks. The core idea involves two adaptive prompting modules: an Adaptive Feature Rectification (AFR) module that purifies node attributes at the input layer to suppress sensitive information, and an Adaptive Message Calibration (AMC) module that dynamically adjusts the message-passing between nodes at each GNN layer to mitigate structural bias. By jointly optimizing these lightweight prompts with a combination of supervised and adversarial losses, the method aims to enhance fairness while maintaining task utility, without updating the frozen pre-trained GNN parameters. Extensive experiments on four datasets under various pre-training strategies demonstrate its effectiveness.

### Strengths
1. This paper is well-written and easy to understand

2. This paper grounds its proposed method, ADPrompt, in a robust theoretical framework.

### Weaknesses
1. The paper's primary motivation—using graph prompting for fairness—rests on the assumption that pre-trained GNNs are a valuable and widely adopted resource that should be efficiently adapted. However, this premise is not thoroughly debated. In contrast to large language models or vision transformers, GNNs are often task-specific and can be trained from scratch relatively quickly and efficiently. The claimed benefit of prompting—parameter efficiency by freezing the backbone—is less compelling when the backbone itself (a GNN) is not an exceptionally large or general-purpose model. The paper would be stronger if it provided a more convincing justification for why prompting is the right paradigm for this problem, compared to simply building a fairness-aware objective into an end-to-end GNN training process, which is common in graph fairness literature.

2. The baselines are mostly graph prompting methods, lacking dedicated, state-of-the-art graph debiasing methods that do not rely on pre-training or prompting (e.g., Edits [1], FairVGNN [2]).

[1] EDITS: Modeling and Mitigating Data Bias for Graph Neural Networks

[2] Improving Fairness in Graph Neural Networks via Mitigating Sensitive Attribute Leakage

### Questions
Please see weaknesses

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies a fairness-aware graph prompting method called ADPrompt that integrates Adaptive Feature Rectification and Adaptive Message Calibration to mitigate biases in both node attributes and graph structure. This method reduces group prejudice in GNNs and also improves adaptability for downstream tasks. The authors also give empirical results on multiple datasets that ADPrompt outperforms some baselines.

### Strengths
Theorem 1 gives a fairness guarantee of ADPrompt, which shows that ADPrompt reduces initial feature bias and suppresses
bias propagation, providing a tighter upper bound on Δ_GSP than a standard GNN. The authors also provide empirical result to justify their theoretical findings. This results are very interesting.

### Weaknesses
Theorem 1 only shows a relationship of less or equal than, but does not really tell how much higher the inequality is.

### Questions
1. Can you explain how tight Eq. 12 and 16 are? What is the best possible inequality (i.e., the limit) of Eq. 17?

2. Where are the complete proofs of the results in Section 5.2? I cannot check the whole details in the current version. 

3. The empirical results show that their proposed method achieves the best or highly competitive performance across various pre-training strategies (Table 1). My question is, does a method that can better suppress the bias always show better performance? Can we analyze it quantitatively?

4. Small issue: after Theorem 1, the explanation "This formally shows that ADPrompt reduces initial feature bias and suppresses bias propagation, providing a tighter upper bound on ΔGSP than a standard GNN." should be not part of the theorem.

### Soundness
3

### Presentation
3

### Contribution
3
