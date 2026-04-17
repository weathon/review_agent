# ProCoSA: Probabilistic Concept Learning with Spatial Alignment

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 4

## Abstract
Concepts are human-interpretable semantic units that enable intervenable intermediate representations in vision models. However, acquiring concept annotations is expensive and typically incomplete, limiting scalable interpretability. We propose \textbf{ProCoSA}, a probabilistic framework that treats missing concepts as latent variables and jointly infers concept posteriors and task predictions under partial supervision. To enhance spatial coherence and reduce pseudo-label bias, \textbf{ProCoSA} introduces a spatial alignment prior that encourages concept activations to align with salient image regions, yielding more calibrated concept probabilities for downstream reasoning. The framework integrates seamlessly into existing concept-to-task pipelines without relying on any specific bottleneck architecture. Experiments on four benchmark datasets under low concept supervision show that \textbf{ProCoSA} consistently matches or surpasses state-of-the-art methods on both concept and task performance under identical evaluation protocols. The code will be released upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses the challenge of learning interpretable concept representations under incomplete or missing concept supervision. To this end, the authors propose ProCoSA, a probabilistic framework that models missing concepts as latent variables and jointly infers concept posteriors and task predictions through an EM procedure. To ensure semantic grounding and reduce pseudo-label bias, ProCoSA introduces a spatial alignment prior that guides inferred concepts to align with salient image regions, supported by lightweight alignment and spatial-consistency regularization. Extensive experiments demonstrate the effectiveness of the proposed ProCoSA.

### Strengths
* Compared with previous heuristic pseudo-label propagation approaches (e.g., SSCBM), ProCoSA models concept uncertainty in a Bayesian inference framework, enabling more robust and principled learning of the concept space.
* The authors discuss the related literature in considerable detail.

### Weaknesses
1. Interpretability: Although the proposed method shows accuracy improvements over existing approaches, the authors lack sufficient evaluation of the method’s interpretability. Only a few qualitative visualizations are provided. The authors should conduct both qualitative and quantitative analyses to verify that the learned concepts consistently and accurately correspond to the intended semantic regions.
2. Method: The paper uses only ResNet as the feature extraction backbone. The authors should include additional architectures such as ViTs to further demonstrate the generality of the proposed method.
3. Written: This paper is poorly written, with incorrect citation formatting and an unclear presentation of Figure 1.
4. Code: The authors do not provide code for reproducibility check.

### Questions
My questions and concerns are in Weaknesses Section.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes ProCoSA, a method for CBMsthat addresses sparse concept annotations. The core idea is to treat missing concepts as latent variables and infer them using an EM framework. This process is guided by a spatial alignment prior, derived from the cosine similarity between spatial features and concept embeddings, and is supported by two simple regularizers. The model is trained end-to-end, using the posterior mean from the E-step to update the model in the M-step. On four standard CBM datasets, ProCoSA demonstrates improved performance over existing methods in low-label regimes and shows a clear, monotonic improvement in concept intervention curves.

### Strengths
1、The paper's use of EM to handle missing labels is a clean, probabilistic alternative to prior heuristic methods like pseudo-labeling.

2、he gating mechanism for the spatial prior is a smart design choice, effectively mitigating noise by applying constraints only when confidence is high.

3、The method shows clear performance gains in low-data regimes and achieves the monotonic intervention curves expected of a well-formed CBM.

### Weaknesses
1、Limited Novelty. The paper's core contribution is swapping the k-NN pseudo-labeling from SSCBM with an EM framework. This feels like an incremental methodological refinement rather than a significant conceptual leap.Self-Referential Prior. 

2、The spatial alignment prior is not independent, as it shares the same feature backbone with the concept head. This creates a circular problem where a feature is used to generate a prior that in turn constrains itself.

3、The central claim of providing "more reliable posterior uncertainty" is unsubstantiated：1）The paper is missing key experiments on confidence calibration (e.g., ECE, Brier score) to actually prove this；2）the motivation relies on aligning concepts to visual evidence, but the paper lacks any quantitative localization metrics to show this is happening effectively beyond a few qualitative examples.

3）the evaluation is restricted to standard benchmarks,to properly test the method's stability, the analysis should include robustness tests (e.g., against input noise/occlusions) or a small cross-domain experiment.

### Questions
1、Key Comparison Details Buried in Appendix. The paper claims a direct comparison with SSCBM under a consistent protocol, but crucial experimental details are relegated to the appendix, making this claim difficult to verify from the main text alone.

2、Central Claim of "Better Uncertainty" is Unproven. The core selling point is that the method produces more reliable uncertainty, yet the paper provides no quantitative evidence. Key metrics like Expected Calibration Error (ECE) or a selective risk analysis are completely missing.

3、Self-Referential Prior. The alignment prior is derived from the same backbone features it is meant to guide. This is a significant methodological flaw, as the prior provides no external information and risks simply amplifying the model's own biases.

4、Localization Claim Lacks Quantitative Support. The assertion that the spatial prior improves concept localization is backed only by qualitative heatmaps. A strong claim like this requires quantitative validation (e.g., pointing accuracy or other localization metrics).

5、Limited Robustness Evaluation. The experiments are confined to standard CBM benchmarks. The paper is missing any analysis of the model's robustness to domain shifts or input perturbations (e.g., noise, occlusions), making its real-world stability unclear.

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
2

### Summary
This paper proposes ProCoSA, a probabilistic framework for concept-based interpretable learning under partial concept supervision. The motivation is to treat missing concept labels as latent variables and jointly infer concept probabilities and task predictions using an Expectation-Maximization (EM) approach. To improve spatial consistency and reduce pseudo-label bias, ProCoSA introduces a spatial alignment prior that encourages concept embeddings to align with salient image regions. The method is evaluated on 4 datasets (CUB, AwA2, WBCatt, Derm7pt) under low concept supervision ratios (5%–20%), showing consistent improvements in both concept and task accuracy over existing methods like CBM, CEM, and SSCBM.

### Strengths
1. ProCoSA treats missing concepts as latent variables and uses variational inference within an EM framework, providing a principled alternative to heuristic pseudo-labeling. Besides, it explicitly models concept uncertainty, which is often overlooked in existing concept bottleneck models (CBMs)

2. ProCoSA provides concept-level saliency maps that align well with human intuition. Besides, it also supports test-time intervention by allowing concept correction, demonstrating causal alignment between concepts and task predictions.

### Weaknesses
1. Missing analysis on the computational overhead: the EM loop with multiple fixed-point iterations per E-step may increase training time compared to simpler pseudo-labeling methods, though this is not quantified.

2. While ablations show the impact of alignment and spatial losses, more analysis on the sensitivity to hyperparameters would be useful;

3. ProCoSA lacks a theoretical justification for the convergence of the truncated EM algorithm or the quality of the variational approximations

### Questions
Please see the weakness.

### Soundness
2

### Presentation
2

### Contribution
2
