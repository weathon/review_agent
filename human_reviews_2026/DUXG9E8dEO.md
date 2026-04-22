# Theoretical Analysis of Contrastive Learning under Imbalanced Data: From Training Dynamics to a Pruning Solution

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Contrastive learning has emerged as a powerful framework for learning generalizable representations, yet its theoretical understanding remains limited, particularly under imbalanced data distributions that are prevalent in real-world applications. Such an imbalance can degrade representation quality and induce biased model behavior, yet a rigorous characterization of these effects is lacking. In this work, we develop a theoretical framework to analyze the training dynamics of contrastive learning with Transformer-based encoders under imbalanced data. Our results reveal that neuron weights evolve through three distinct stages of training, with different dynamics for majority features, minority features, and noise. We further show that minority features reduce representational capacity, increase the need for more complex architectures, and hinder the separation of ground-truth features from noise. Inspired by these neuron-level behaviors, we show that pruning restores performance degraded by imbalance and enhances feature separation, offering both conceptual insights and practical guidance. Major theoretical findings are validated through numerical experiments.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper develops a theoretical framework to analyze how contrastive learning behaves under imbalanced data distributions, with a specific focus on Transformer-based encoders. It identifies how feature frequency imbalance (i.e. majority vs. minority features) affects neuron-level learning dynamics and the quality of representation learning.

The paper shows that the training progresses through three distinct stages—initial feature growth, specialization of “lucky” neurons, and final convergence. It also shows minority features are learned more weakly, leading to fewer neurons specializing in them and resulting in degraded representations. It further shows introducing magnitude-based pruning during training mitigates these effects by amplifying updates for neurons aligned with minority features, thus improving representation balance.

Theoretical results are corroborated by empirical experiments on various dataset (CIFAR10-LT, CIFAR100-LT, and ImageNet-LT), demonstrating consistent performance gains and reduced accuracy gaps between majority class and minority classes.
In addition, the paper provides theoretical proofs for convergence, establishes feature alignment properties, and rigorously derives the effects of pruning on the learning dynamics.

### Strengths
**Originality** The paper seems quite novel. Given my knowledge I am not aware of many existing works exploring the statistical generalization theory of contrastive learning and data imbalance. While prior works have addressed imbalance heuristically or empirically, this paper seems the first formal theoretical treatment of the phenomenon. 

**Quality**
The math of the paper seems rigorous, supported by clearly stated lemmas and theorems. Empirical experimental results seem to align with statistical theory. The synthetic data experiments (Appendix A.2) further validate theoretical predictions in controlled settings.

**Clarity**
The clarity of the paper is good. Although this paper is theoretically heavy and thus not easy to read, its clear presentation of results has made the task easier. The three training stages are clearly delineated, and key insights are summarized upfront (Section 3.1). The figures and tables are easy to interpret, and a table of notation is provided. The proof sketch is helpful too.

**Significance** 
This work has strong implications for self-supervised learning on real-world, imbalanced datasets. It offers theoretical understanding in the study of contrastive learning, showing how imbalance alters neuron specialization and model complexity requirements.

### Weaknesses
There is no major technical flaw detected. However, I do have the following concern.

Assumptions might be a bit strong.

(1) The entire analysis seems to focus on the sparse coding model with orthogonal features and independent Gaussian noise. I understand the technical challenge in analyzing a more general setting. But the current setting feels a bit simple, because in reality features are usually correlated. This orthogonality assumption prevents the theory from describing how contrastive learning handles overlapping or dependent semantic features. 

(2) The paper’s assumption that self-attention remains identity-fixed fundamentally limits the scope of its theory. In other words, $W_K, W_Q, W_V$​ are not learnable.

### Questions
(1) In Transformers, neurons often encode superposed features that are only linearly separable after training. The analysis assumes near-orthogonal features and pure specialization at convergence. Can the authors comment on whether their theory predicts or forbids feature superposition (neurons simultaneously encoding multiple correlated features)?

(2) Could the authors maybe clarify how the temperature parameter ττ influences imbalance sensitivity in their theory?
Specifically, does a smaller ττ (sharper similarity weighting) amplify majority-feature dominance by concentrating gradients, while a larger ττ mitigates it by smoothing updates?

(3) If the attention mechanism were trainable, pruning neurons downstream of attention would alter gradient flow through $W_K, W_Q, W_V$​. Could the authors speculate on whether such coupling might reinforce or dampen the minority-feature amplification effect?

(4) Could the authors comment on how neuron specialization (Theorem 3.2) is connected to the generalization performance?
In particular, does specialization provably enhance linear separability or any other preferred properties of the learned representations?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper provides a robust theoretical analysis of the training dynamics of Transformer-MLP models in learning feature representations through contrastive learning in imbalanced data scenarios. Specifically, the analysis of neuron weight evolution reveals how a minority of features undermines overall model performance. Building upon this, they revisit amplitude-based pruning methods, theoretically demonstrating that pruning yields more robust and balanced representations.

### Strengths
This paper explores how imbalanced data degrades representation quality in contrastive learning from a novel perspective of neural weight evolution. Through extensive theoretical analysis, the authors demonstrate that a minority of features weakens representational power while increasing the demand for complex architectures. Building upon this, they further prove that pruning techniques enhance gradient updates along these dominant features, thereby mitigating performance degradation caused by imbalance. This forward-looking approach holds significant promise for advancing the field.

### Weaknesses
1：This paper uses numerous notations, some of which lack clear definitions upon their first appearance. Additionally, maintaining consistent notation throughout the text would improve readability.

2：Could the authors please clarify the meaning of "feature frequency"? Specifically, how are the majority and minority features identified within the unsupervised learning framework?

3: The study is confined to the Transformer-MLP model. Could the authors discuss the generalizability of their approach to other architectures or tasks, such as long-tailed visual recognition?

4: While the authors provide substantial theoretical proofs, the effectiveness of the pruning strategy for imbalanced scenarios remains unverified by strong experimental evidence. Additionally, could its potential as a plug-and-play module for existing methods be discussed?

5:The authors are advised to provide precise citations to the appendix for their key conclusions, and to thoroughly review the manuscript to correct various notational errors and inconsistencies.

### Questions
As described in “Weaknesses”.

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper analyzes why contrastive learning deteriorates under imbalanced data. Using a sparse latent-feature model and a Transformer-MLP encoder with InfoNCE, it shows that rare (minority) features grow more slowly, fewer neurons specialize in them, and neurons are forced to represent multiple features, effectively increasing the capacity needed to cover all features. To mitigate this, the authors propose a magnitude-based, forward-masked but backward-unmasked pruning scheme: small-magnitude parameters are masked out only in the forward pass, but all parameters are still updated. This selectively amplifies gradients along minority-feature directions and restores more balanced representations. Experiments on CIFAR-LT and ImageNet-LT confirm better linear-probe accuracy and smaller head–tail gaps.

### Strengths
- Provides a rare, neuron-level theoretical analysis of contrastive learning under data imbalance, clearly explaining how minority features are under-learned.
- Connects the analysis to a simple, practical fix (magnitude-based forward-masked, backward-unmasked pruning), making the work actionable.
- Writing and structure are generally clear, making a dense theoretical contribution reasonably accessible.

### Weaknesses
- Experiments mainly compare “with vs. without pruning” and lack baselines from other long-tailed methods.
- Sensitivity to pruning ratio/schedule is not deeply analyzed.
- Paper could more explicitly discuss limitations and when the proposed analysis may not apply.

### Questions
- The paper sometimes reasons at the neuron level but prunes at the parameter level; please clarify the exact relationship between “parameter-level masking” and the claimed neuron-level effects.
- Which parts of the analysis rely essentially on (i) identity/self attention, (ii) sparse independent features, or (iii) single-layer MLP, and which parts you believe could be relaxed or extended? A short discussion of “what breaks / what survives” under more realistic assumptions would help.
- Can you provide empirical evidence of “lucky” vs. mixed-feature neurons (e.g., alignment plots)?
- In the experiments, how does the method behave when the imbalance ratio = 1 (i.e., no explicit long tail)? Since the proposed pruning is fairly general, do you observe gains in “implicitly imbalanced” settings (e.g., feature-frequency skew induced by augmentations or views) even without a constructed LT distribution?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper deals with contrastive learning, a self-supervised framework in which input data are paired into positive or negative pairs based on their similarity in semantic meaning. In particular, data are generated according to the Sparse Coding Model: they are linear combinations of feature vectors plus noise, feature $j$ has an activation probability controlled by a parameter $\epsilon_j$, and features are in general imbalanced (majority features have $\epsilon_j = \epsilon_{\rm max}$, minority ones $\epsilon_j = \epsilon_{\rm min}$). Positive pairs are formed with inputs sharing the same token-aggregate features. The model is a Transformer-MLP built with a single head attention layer followed by an MLP. Training is performed by minimizing the InfoNCE loss, suitable for contrastive learning. A pruning mask filtering temporarily small magnitude neurons is applied when computing the gradients at each training step, and released before updating the neurons. 

The paper provides formal results on the dynamics of training of this model. In particular, Lemmas 3.1, 3.2 and Theorem 3.1 show the existence of 3 temporal regimes during training with no pruning: in the first, neuron weights grow in feature directions but are suppressed in non-feature directions, with the growth rate in a feature direction depending on its frequency; in the second, few (lucky) neurons align significantly with single features, while ordinary neurons align in composite directions; in the last (at convergence), the training error is small and neurons are strongly aligned with a subset of features, weakly aligned with the remaining features, and small in the non-feature directions (still, only a limited number of neurons specialize in learning a single feature). Theorem 3.2 shows that pruning amplifies the learning of minority features.

### Strengths
Both the architecture and the training protocol are relevant. Theoretical predictions for the training dynamics of Transformer-based encoders are of utmost interest. Narratives on the assumptions behind the data model and on the formal results are provided. Numerical results on real data support the theoretical claims on the advantage of pruning.

### Weaknesses
The formal results are hard to read, as the main text is not really self-contained (see Questions below), despite the commendable effort of Table 1. Numerical illustrations in the vanilla setting, even with synthetic data, could help explaining the practical relevance of the bounds provided (for example, by tracking the inner products of lucky/non-lucky neurons with features during training in the 3 regimes, and comparing with theoretical bounds). Considering that reviewing proofs in Appendix is out of question due to time limitations, numerical checks should help strengthening the claims provided by the main text.

### Questions
- Can the authors make the main text more self-contained? For example:
    - I could find the hypothesis on the batch size $K$ (such that empirical gradients approximate population ones) only in Lemma B.1;
    - I am assuming $C_m$, $C_z$ to be positive constants, is that the case?
    - The scaling of $T_1$, $T_2$ is not given in the main text;
    - $\Xi_2$ is not defined in the main text, etc.
- In appendix A.2, the authors provide numerical experiments with synthetic data. Can they report in the main text numerical evidence for the 3 training regimes they are able to identify theoretically, for better illustration and check?
- How are negative pairs chosen for the case of real data of section 4?

Minor:

Sometimes * is used instead of $\star$ to denote lucky neurons.

### Soundness
3

### Presentation
2

### Contribution
3
