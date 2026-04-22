# Recursive Structure Discovery as an Inductive Bias for Symbolic Regression

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
Symbolic regression (SR) can recover analytic laws from data, but its search space is enormous. Many scientific targets are structurally simple, for example additively or multiplicatively separable, yet most SR pipelines do not exploit this. We introduce a recursive structure discovery step that tests for separability using accurate derivatives from a small neural model trained with second-order updates. The method decomposes $y=f(\mathbf{x})$ into a hierarchy of simpler subfunctions, which we feed to SR as a structure prior. This plug-in reduces search complexity, improves interpretability, and can attach to any SR backend; here we pair it with a deep RL generator. This substantially reduces search complexity, improves interpretability, and remains robust to noise, maintaining reliable separability detection under challenging conditions. On SRBench (Feynman, 120 equations), the structure-aware pipeline achieves state-of-the-art exact recovery, outperforming separability-only, pure RL, and prior hybrid baselines.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a two-stage framework: it first leverages derivative information to uncover the variable structure within expressions, then performs variable separation to decompose the overall problem into subproblems, and finally applies a reinforcement learning–based symbolic regression algorithm. The method achieves a high recovery rate on SRBench.

### Strengths
The ideas in the paper are sound, the language has no obvious issues, and the overall structure is complete.

### Weaknesses
The literature review is insufficient. Derivative-based variable separation was first introduced into symbolic regression by AI Feynman, and AI Feynman 2.0 already handles variable separation for most separable cases, including multiplicative separability.

Udrescu, S. M., Tan, A., Feng, J., Neto, O., Wu, T., & Tegmark, M. (2020). AI Feynman 2.0: Pareto-optimal symbolic regression exploiting graph modularity. Advances in Neural Information Processing Systems, 33, 4860-4871.

The paper lacks novelty. The proposed method merely stitches together AI Feynman and DSR, and this combination has already been explored by other researchers, such as the UDSR mentioned in the paper.

The experimental baselines omit recent strong methods. The experiments lack reproducible and convincing empirical support; the reported performance of some baselines differs markedly from results in other papers and from those obtained using the corresponding open-source implementations.

### Questions
None.

### Soundness
3

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
This paper proposes a recursive structure discovery framework that improves symbolic regression (SR) by uncovering how variables in scientific data combine through simple, interpretable structures. Instead of directly fitting equations, the method first trains a compact NestyNet with accurate estimated derivatives. These derivatives are used to detect additive, multiplicative, and nested nonlinear separabilities, building a hierarchical tree of sub-functions. This structure acts as a probabilistic prior guiding a deep reinforcement learning-based SR model to generate mathematically consistent expressions. The proposed method is demonstrate to outperform many existing SR methods.

### Strengths
1. This paper proposes a novel approach to first use powerful neural networks to fit the data and then detect key structures and inform symbolic regression. 
2. This paper proposes the key issue of compactness of the learned expression and the benefit of doing so, which may inspire later work.
3. The empirical performance is convincing.

### Weaknesses
1. The paper is a bit hard to read from time to time, possibly because there are many components in the method (NestyNet, finding structures, and incorporating it into symbolic regression). 
2. How the tree is built is not clear -- it might be helpful to elaborate on this. I understand that $f_i$ is a NestyNet, but it is unclear how the mapping $\Phi$ is determined. The paragraph "Composite model" on page 5 is difficult to read.
3. It's unclear how the threshold values are chosen in Table 1 (and there is a typo `treshold' above it). 
4. It is unclear why the method is paired with the RNN method.

### Questions
1. Can you explain how the composite model on page 5 is trained?
2. Why would you use a tree structure to combine the NestyNet layers? It is also common to use tree to represent expressions in symbolic regression, but it's unclear to me why using a tree makes sense here.
3. How did you choose the threshold values in Table 1?
4. How does the method compare with uDSR in Figure 5? Does uDSR use a similar RNN approach? The main question is that with the current presentation of the results, it is unclear which component leads to the margin of the proposed method.

### Soundness
2

### Presentation
2

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
This paper proposes a two-stage pipeline for symbolic regression (SR). First, it recursively discovers functional structure—additive/multiplicative separability and simple unary compositions—using accurate first/second-order derivatives from a compact neural “NestyNet” trained with Levenberg–Marquardt (LM). The resulting hierarchy is then used as a structural prior for an RL-based SR generator. On SRBench (Feynman, 120 equations), the method reports 72% exact recovery, outperforming separability-only, pure RL, and prior hybrid baselines.

### Strengths
Clear problem framing & contributions. The paper argues convincingly that many scientific targets are modular and that exposing separability can shrink SR search while improving interpretability. 

Methodological neatness. The two-stage design is simple and broadly compatible with SR backends: (a) recursive separability discovery, (b) structure-guided RL generation. 

Technical soundness. NestyNet has closed-form Jacobian/Hessian w.r.t. inputs, which enables reliable separability tests and LM optimization; formulas are explicit. 

Concrete, testable criteria. Additive and multiplicative separability tests are stated precisely with thresholds (ε_add=1e-4, ε_mul=1e-12, ε_βmad=1e-3). 

Empirical performance. Strong results on SRBench with consistent hyperparameters; operators and evaluation limits are specified. 
Interpretability angle. The structural prior biases the generator toward symbolically faithful forms rather than merely accurate fits. 

Interpretability angle. The structural prior biases the generator toward symbolically faithful forms rather than merely accurate fits.

### Weaknesses
Novelty boundary vs. prior separability work. The main leap is handling multiplicative separability and nested unary transforms and integrating them as soft priors, whereas prior AIF/uDSR emphasize additive separability. The paper should sharpen how much gain derives from each element (e.g., multiplicative test vs. nested transforms vs. RL prior shaping). 

Reliance on a bespoke emulator. NestyNet is referenced as to be “fully described” elsewhere, which weakens reproducibility claims and makes it hard to benchmark against standard MLPs beyond one figure. 

Sensitivity/robustness of the detectors. The separability decisions hinge on derivative quality and fixed thresholds; there is limited analysis of threshold sensitivity, failure modes (false positives/negatives), or uncertainty quantification during the recursive split. 

Noise robustness not yet validated. The Discussion acknowledges that formal noise testing is future work; given SRBench variations and real data, this is a notable gap. 

Compute profile. Pre-processing can average ~4 hours per case (GV100), and constant fitting is CPU-bound (~1 hour on 10-core Xeon). A deeper analysis of throughput and scaling to higher-dimensional inputs would help.

### Questions
See weakness

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
5

### Summary
This paper introduces Recursive Structure Discovery (RSD) to mitigate the combinatorial explosion inherent in symbolic regression (SR).
The method aims to automatically detect additive and multiplicative separability of target functions and leverage such structure as a prior for symbolic regression.

The approach consists of two stages:

1. Using a lightweight neural network called NestyNet to estimate high-precision derivatives and hierarchically detect separability;
2. Using the detected structure as a structural prior in a reinforcement-learning-based SR framework. On the Feynman SR benchmark (SRBench, 120 formulas), the method achieves a 72 % exact-recovery rate, outperforming existing systems such as AI Feynman 2.0, uDSR, and PySR.

The technique can handle non-linear nested functional structures, showing that structural inductive bias can significantly enhance SR performance.

### Strengths
- The paper presents a novel and coherent formulation of structural inductive bias for symbolic regression.
- The use of NestyNet for derivative-based structure detection provides an efficient reduction of the search space, compared with prior unstructured approaches.
- The method can recursively detect additive and multiplicative separability, enabling the discovery of complex, nested physical relations.
- Integration of the detected structure into the RL-based expression generator is conceptually natural and theoretically consistent.

### Weaknesses
- The main concern lies in the lack of hyper-parameter sensitivity analysis, particularly regarding the thresholds for separability detection (ϵadd, ϵmul, ϵβmad).

  These thresholds are empirically fixed, yet their effect on detection accuracy and mis-segmentation rates is not quantified.
  A visualization of how structural decomposition changes with threshold variation would substantially strengthen the paper’s reliability.

- Beyond these thresholds, other hyper-parameters—such as the LM damping factor, the hidden width h of NestyNet, and the entropy-regularization weight in the RL stage—are all kept fixed without discussion of robustness.

  The method’s stability across parameter variations remains unclear.

- No evaluation is provided on noisy or perturbed datasets. While the method is theoretically stable, empirical validation of noise robustness is missing.
- Statistical uncertainty is not reported: Fig. 5 lacks error bars or significance testing, so the reliability of the performance gaps is uncertain.

### Questions
1. Have you examined how sensitive the results are to the separability-detection thresholds (ϵadd, ϵmul, ϵβmad)?
2. Would it be feasible to determine these thresholds adaptively—for instance, via BIC/MDL-based model selection?
3. Could you provide stability analysis results for key hyper-parameters in both NestyNet and the RL stage?
4. How does the structure-detection accuracy behave when moderate noise (e.g., 10 % Gaussian perturbation) is added to the data?
5. Can you report statistical significance or variance estimates for the comparisons shown in Fig. 5?

### Soundness
3

### Presentation
2

### Contribution
3
