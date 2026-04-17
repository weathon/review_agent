# Structural Inference: Interpreting Small Language Models with Susceptibilities

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 8

## Abstract
We develop a linear response framework for interpretability that treats a neural network as a Bayesian statistical mechanical system. A small perturbation of the data distribution, for example shifting the Pile toward GitHub or legal text, induces a first-order change in the posterior expectation of an observable localized on a chosen component of the network. The resulting susceptibility can be estimated efficiently with local SGLD samples and factorizes into signed, per-token contributions that serve as attribution scores. We combine these susceptibilities into a response matrix whose low-rank structure separates functional modules such as multigram and induction heads in a 3M-parameter transformer.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces susceptibilities, a new interpretability framework grounded in Bayesian statistical mechanics that measures how specific neural network components respond to controlled changes in the data distribution. By estimating these susceptibilities locally using SGLD, the authors define a notion of expression vs. suppression for model components and construct response matrices that reveal internal structure. Applied to a 3M-parameter transformer trained on the Pile, the method automatically separates known functional modules such as the induction circuit and identifies linguistic patterns like word segmentation and bracket matching.

### Strengths
1. This work introduces a principled and novel framework for understanding the internal computations of deep neural networks, drawing inspiration from Bayesian statistical mechanics and singular learning theory.
2. The proposed methodology offers an unsupervised pipeline for discovering functional circuits within neural networks.
3. It not only provides empirical results demonstrating the effectiveness of the framework, but also includes sanity checks to ensure its practical value in understanding transformer-based language models.
4. The approach establishes strong connections to physics, which may encourage researchers in the field to contribute to the study of the interpretability of neural networks.

### Weaknesses
1. Although the theoretical framing is elegant, its practical usefulness remains uncertain. The empirical evaluation focuses only on a small toy model with two attention layers, which is not representative of typical language models.
   - Training on only a subset of the Pile dataset may be too simplistic, potentially explaining why the first principal component accounts for  95–99% of the representational variance.
2. The interpretation of principal components remains largely manual and speculative, which may lead to inconsistent or subjective interpretations of the empirical findings.
3. The proposed susceptibility framework captures correlations in the local loss landscape, in contrast to many mechanistic interpretability approaches that emphasize causal relationships.
4. Finally, the paper requires expertise in Bayesian statistics, which may limit accessibility and reduce its potential audience.

### Questions
1. The empirical results in Section 4.2 show that the first principal component (PC1), interpreted as “word segmentation,” accounts for an overwhelming portion of the variance (95.3% in Layer 0 and 99.1% in Layer 1). Could the authors clarify how the proposed method can reveal more complex circuits when the learned representations appear to be dominated by such a simple feature?
2. Since susceptibility is fundamentally correlational, did you conduct any causal validation, such as steering or ablating the susceptibility-identified heads, to assess their influence on the model’s outputs?
3. Appendix C.5 presents a cost model that scales linearly with the number of components. While this may be feasible when analyzing only attention heads, it seems impractical once MLP neurons are included. Do you plan to extend the framework to investigate MLP neurons as well?

### Soundness
2

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
2

### Summary
The authors introduce a new interpretability framework grounded in statistical physics and Bayesian learning theory. The key idea is to view a neural network as a Bayesian statistical mechanical system, where the model’s parameters and interactions can be explored through small perturbations of the data distribution. Small shifts in the input data (like moving from natural text to programming code) produce first-order linear responses in specific components of the model. These responses (called susceptibilities) reveal how strongly and in what direction each component reacts to data changes, providing a principled way to quantify its sensitivity and functional role within the network.

### Strengths
* The framework is theoretically rigorous, grounded in statistical physics and Bayesian learning theory, while also being empirically validated through concrete experiments.

* The proposed approach is novel:  Susceptibility analysis connects the functional behavior of model components to shifts in the training distribution and shows that heads with similar response patterns cluster into interpretable groups.

### Weaknesses
* The analysis (Sec 4) focuses on a very simple set of patterns (Word Start, Word Part, Word End, Induction Pattern, Right Delimiter). In addition, the framework is demonstrated only on a small toy language model (3M, 2 attention layers, without MLP). Although the authors note that they do not anticipate major obstacles in scaling the method to larger models, applying it to a larger model could have strengthened the work by demonstrating the ability to capture more complex behaviors.

* This is only a suggestion, but I think the presentation could be made more accessible to readers who are less familiar with the theoretical background.

### Questions
* as I mentioned in the weaknesses section, I think the paper would be stronger if it demonstrated how the proposed method could be applied to analyze more complex patterns or behaviors (such as bias or knowledge acquisition).

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The authors provide a framework for approximating the effect of a small change in the training data on model components, called susceptibilities. They show that this is equivalent to the covariance between a component’s loss (in this case, attention heads) and the change in total loss under the perturbed data distribution. They then estimate, using locally sampled posteriors with SGLD, susceptibilities for loss on each pair of tokens in the vocabulary, for all attention heads in a 3M-parameter transformer, trained on subsets of Pile. The top principal components of this combined covariance matrix are sensitive to certain classes of interpretable token patterns, such as the induction.

### Strengths
1. The assumptions are clearly stated, and the theory seems well grounded. 
2. Applying susceptibilities to attention heads yields interpretable patterns, previously found in small attention-only transformers via mechanistic interpretability.

### Weaknesses
1. Although novel, this attribution method seems very hard to scale to larger models. 
2. The heads that express or suppress the reported patterns in Figure 2 are not yet mechanistically explained. The correlation with Direct Logit Attribution (Figure 32) shows no sign that they are actually responsible for the behaviour. 
3. Relaxing sampling to the local posterior makes sense for small changes, but it is unclear if this holds when 10% of the data is replaced. It would be great if this could be clarified.

### Questions
1. Could the authors explain why $\delta h = 0.1$ is justified? And additionally, compute $\chi$ for range of values of $\delta h$ to check how different the susceptibilities are?
2.  How to interpret the functional roles of these heads with respect to susceptibilities, given that the DLA plot doesn't show any correlation? One could conduct causal ablations on a few samples as a sanity check, but it is unclear if it would be correlated, given that DLA shows no signs of life. Does this method actually find a causal, induction circuit?

It would be great if the authors could address these two points in the paper (or point me to it, if it already exists). 

Some writing suggestions:
1. A notation table (and maybe an algorithm table) would be extremely helpful as the paper is quite mathematically dense. 
2. Add some more description about the key findings. Eg: suppression and expression in the introduction

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors introduce a novel interpretability framework for transformer networks through susceptibilities. The authors investigate how much do variations in the data distribution influence specific components (i.e. attention heads) of the network. The authors highlight two interesting patterns of behavior identified by a standardized notion of susceptibility, resulting from a change in the parameters, namely expression (loss decreases, probability of next token increases) and suppression (loss decreases, probability of next token decreases). Through experiments on a two-layer attention-only Transformer trained on a subset of the Pile, these patterns are related to existing findings from mechanistic interpretability, rediscovering the induction head, as well as identifying how simple linguistic capabilities are conducted within the network. 

Overall, the paper presents a strong mathematical foundation and a novel approach for uncovering mechanistic patterns within transformer networks. The paper is well written, albeit the math is quite dense. My main concerns with the paper are the method scaling to larger models, which the authors proactively address in Appendix C.5, but I believe remains a limitation of their work. Finally, I am slightly concerned how the method should be applied in cases where there is no clear next-token point where susceptibility should be estimated (open-ended generation tasks), as the studied patterns are relatively simple to identify from data.

### Strengths
- Strong mathematical foundation
- Provides an empirically validated novel method for discovering mechanistic patterns within transformer networks

### Weaknesses
- Computational costs when scaling (mentioned as a weakness)
- Applicability beyond simple tasks

### Questions
See above

### Soundness
3

### Presentation
3

### Contribution
3
