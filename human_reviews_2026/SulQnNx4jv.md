# Disentangled Causal Transformer: Counterfactual Prediction under Time-Varying Treatments

- Avg Score: 4.67
- Decision: Reject
- Scores: 6, 6, 2

## Abstract
Estimating longitudinal counterfactual outcomes from observational data is pivotal to personalized medicine and other domains. However, prevailing approaches for mitigating time-varying confounding bias typically balance all covariates indiscriminately, conflating confounders with instrumental variables and thus unnecessarily discarding valuable outcome-relevant information. While causal disentangled representation learning has proven effective in static settings, extending it to the longitudinal setting—where representation disentanglement and time-series modeling must be performed jointly over time—remains a key challenge. To address this, we introduce the $\textbf{Disentangled Causal Transformer (DCT)}$, a Transformer-based architecture designed to integrate causal representation disentanglement seamlessly within the sequence modeling process for robust longitudinal causal inference. DCT features a novel $\textbf{disentangled multi-head attention}$ mechanism that decomposes a patient’s history into instrumental, outcome, and confounder components. This design enables unbiased causal estimates while preserving the full predictive signal, thus mitigating the traditional trade-off between factual and counterfactual prediction accuracy. Extensive experiments on fully synthetic and semi-synthetic datasets derived from real electronic health records show that DCT consistently outperforms state-of-the-art baselines by a large margin in counterfactual outcome prediction. To the best of our knowledge, DCT pioneers the integration of causal representation disentanglement within a Transformer-based model for robust longitudinal causal inference.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper considers the task of longitudinal counterfactual estimation from observation data, which requires mitigating time-varying confounding bias. To tackle the probem, the authors consider adaptively disentangling covariates to enhance precision and causal effect estimation capability. Specificly, the authors hypothesize that the observation history can be encoded into decoupled latent representations, and design Disentangled Multi-Head Attention(DMHA) with carefully-crafted regularizers to ensure disentanglement. With the DMHA mechanism, the author proposed Disentangled Causal Transformer, an encoder-decoder transformer network employing DMHA, supervised with prediction losses and casual regularization losses to ensure disentanglement and address confounding bias. The model achieve SOTA on fully-synthetic and semi-synthetic datasets, demonstrating effectiveness of the design. Detailed ablation study has been carried out to further study the contribution of each component, including ablation on regularization terms within DMHA layers.

### Strengths
- The problem that the authors investigate is fairly significant, and the decomposition of covariates is relatively underexplored in longitudinal counterfactual estimation.
- The proposed solution is clear and easy to understand, with each regularization term consisting of a simple and direct approach to discourage feature entanglement and to address confounding bias.
- The evaluations thoroughly examined the architecture's effectiveness on counterfactual estimation tasks, with detailed ablations on diversity regularizations.

### Weaknesses
- While the overall writing quality is good and structure is clear, some parts seems lack of purpose. For example, the proposition of subspace orthogonality and attention disagreement regularizations seems lacking direction, as they are not employed in the final design of DMHA.

### Questions
- In the standard transformer blocks in the tranformer decoder, do the cross attention sub-layers also employ the design in  eq.(10), i.e. parallel branch for different factors? Will the cross attention entwine the features in the decoding process?

### Soundness
3

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
This paper makes a significant contribution to the field of longitudinal causal inference by proposing a novel Transformer-based framework that effectively addresses the challenge of time-varying confounding. The novel disentangled multi-head attention mechanism decompose patient history into distinct causal components (instrumental variables, outcome predictors, and confounders), which extends the prior idea of the static setting. The authors provide thorough experimentation on both fully synthetic and semi-synthetic datasets, demonstrating substantial improvements over state-of-the-art baselines in counterfactual outcome prediction.

### Strengths
1. The proposed Disentangled Causal Transformer (DCT) is a innovative architecture. The DMHA jointly addresses representation disentanglement and time-series modeling, a key challenge in longitudinal causal inference.

2. The authors have conducted extensive experiments on both fully and semi-synthetic datasets. The consistent and large-margin superiority over state-of-the-art baselines provides compelling evidence for the efficacy and robustness of the proposed framework.

3. By improving the accuracy of counterfactual outcome prediction, this work has direct and meaningful implications for high-stakes domains like personalized medicine.

### Weaknesses
1. It is widely acknowledged that Transformer suffers from heavy computation complexity. The authors introduce more complex design into the architecture. I suggest the authors to conduct analysis on the time cost of the method.

2.I suggest the authors to conduct ablation study on the effect of hyperparameters balancing the loss terms.

### Questions
Besides the questions listed in Weakness. I have following additional questions.

1. In equation (7), the heads of $f_a$ and $f_b$ are distinct by design, which in my opinion has been able to guarantee the dissimilarity of $Z^{f_a}$ and $Z^{f_b}$. Why does the authors introduce the loss of $L_{sep}$.

2. Why does the term $z^f$ appears twice in Equation (10).

### Soundness
2

### Presentation
3

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
The paper proposes a Disentangled Causal Transformer (DCT) for counterfactual prediction under time-varying treatments. Instead of enforcing treatment-invariance on a single latent space, DCT decomposes representations into three causal roles—Instrumental (I), Outcome-only (O), and Confounder (C)—via a specialized multi-head attention block, role-specific cross-attention, and task routing. It couples this architecture with independence/balancing losses (e.g., MMD for (O \perp A), distribution balancing for (C)) so that only the confounding component is balanced while preserving outcome-relevant signal. Across synthetic and semi-synthetic benchmarks, DCT yields lower counterfactual error, especially at longer horizons, and ablations show the disentangling/balancing components are the main source of gains.

### Strengths
By disentangling representations into Instrumental/Outcome/Confounder roles (via DMHA and role-specific cross-attention) and selectively balancing only the confounder pathway to preserve outcome signal, the method avoids over-adjustment and delivers consistent long-horizon counterfactual gains, as verified by thorough ablations.

### Weaknesses
- DCT hinges on a strong assumption that representations can be cleanly split into I/O/C. In real data, purely instrumental or outcome-only factors are rare and signals often interact; under such partial overlap, the enforced separation can become arbitrary, suppress useful cross-path interactions, and ultimately degrade both performance and interpretability.
- The paper lacks a systematic hyperparameter sensitivity study (e.g., MMD kernel/scale, balancing weights, head grouping, loss coefficients), so performance and disentanglement quality could vary substantially across settings and datasets.

### Questions
- Verify that the learned representations behave as intended: (i) \(z_{O}\) alone cannot predict treatment \(A\), and (ii) \(z_{I}\) alone cannot predict outcome \(Y\). This directly tests whether the I/O/C split carries the claimed semantic roles.
- On synthetic and semi-synthetic data, systematically increase the \emph{I\(\leftrightarrow\)O} interaction strength and the confounding intensity, and report counterfactual performance alongside the probing scores to identify regimes where disentanglement remains beneficial.

### Soundness
3

### Presentation
3

### Contribution
2
