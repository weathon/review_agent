# Hallucination Detection and Mitigation with Diffusion in Multi-Variate Time-Series Foundation Models

- Decision: Reject
- Scores: 4, 6, 0

## Abstract
Foundation models for natural language processing have many coherent definitions of hallucination and methods for its detection and mitigation. However, analogous definitions and methods do not exist for multi-variate time-series (MVTS) foundation models. We propose new definitions for MVTS hallucination, along with new detection and mitigation methods using a diffusion model to estimate hallucination levels. We derive relational datasets from popular time-series datasets to benchmark these relational hallucination levels. Using these definitions and models, we find that open-source pre-trained MVTS imputation foundation models relationally hallucinate on average up to 59.5\% as much as a weak baseline. The proposed mitigation method reduces this by up to 47.7\% for these models. The definition and methods may improve adoption and safe usage of MVTS foundation models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper investigates the hallucination problem in MVTS foundation models when applied to generative tasks. The authors distinguish between two types of hallucinations: distributional hallucinations and relational hallucinations. They propose a diffusion-based hallucination detection metric, termed composite error, to quantify the degree of hallucination, and design a sampling-based mitigation approach, which is validated across multiple MVTS datasets.

### Strengths
1. This work is the first to systematically explore the hallucination problem in MVTS models from the perspective of generative modeling. The research question is novel and holds strong potential for future studies in the time-series domain.
2. The proposed CE leverages diffusion–reverse diffusion discrepancies to measure latent distributional consistency, offering a degree of causal interpretability.
3. The combination of detection and mitigation modules forms a closed-loop framework, demonstrating both engineering practicality and methodological feasibility.

### Weaknesses
1. The paper lacks a theoretical explanation of the mathematical interpretability of the CE metric, as it does not derive its relationship with hallucination intensity from diffusion probability theory or an energy-based perspective.
2. No significance testing is provided, leaving the statistical reliability of the reported results unclear.
3. The adopted MLP-DDPM architecture lacks the capacity to model temporal dependencies.
4. The proposed method relies heavily on the integrity of the training data; if the dataset contains noise or erroneous relationships, the CE metric may fail.
5. The paper only compares hallucination levels across different models but does not evaluate against alternative hallucination detection methods, making it difficult to demonstrate the superiority of the CE metric.

### Questions
1. Can the CE metric be formalized as a variant of an energy function or likelihood estimation?
2. Is the choice of detection threshold statistically robust? Would it lead to false detections under different data distributions?
3. How sensitive are the detection results to the number of diffusion steps T?
4. Can relational hallucinations be validated through causal interventions, such as fixing variable dependency structures?
5. Has the study considered comparisons with traditional out-of-distribution detection or uncertainty estimation methods?

### Soundness
2

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
2

### Summary
This paper draws an analogy from NLP to define "relational hallucination" for Multi-Variate Time-Series (MVTS) Foundation Models. The core contribution is a detection method using an external diffusion model as a "verifier." This verifier calculates a "Combined Error" (CE) metric (RMSE between the FM's output and the verifier's correction) to quantify and mitigate hallucinations in existing MVTS FMs.

### Strengths
( I have little experience with Time-Series Foundation Models, so my confidence score is low. I kindly ask the AC to assign a lower weight to my review. )

The paper provides a valuable conceptual contribution by defining "hallucination" in the MVTS context, which is critical for model reliability in scientific applications.

The creation of "relational datasets" with known ground-truth functions allows for a robust, quantitative validation of the proposed CE metric against the true relational error, supporting the claims.

### Weaknesses
( I have little experience with Time-Series Foundation Models, so my confidence score is low. I kindly ask the AC to assign a lower weight to my review. )

The method's main drawback is its reliance on a dataset-specific diffusion model. This verifier must be trained for each target dataset, which increases computational overhead and undermines the zero-shot/few-shot promise of FMs.

The proposed mitigation strategy (sampling N=20 times and filtering) is a costly, brute-force approach.

### Questions
Please refer to the previous section

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
3

### Summary
The paper “Hallucination Detection and Mitigation with Diffusion in Multi-variate Time-Series Foundation Models” addresses the absence of formal definitions and detection methods for hallucinations in multivariate time-series (MVTS) foundation models (FMs). While hallucination in NLP FMs is well studied, its equivalent in time-series contexts has not been explored. The authors introduce two key definitions—distributional hallucination, where a model’s prompt–response pair is out-of-distribution (OOD) relative to training data, and relational hallucination, where variable relationships are inconsistent with ground-truth relational functions. The paper focuses on the latter, proposing a Combined Error (CE) metric derived from diffusion models to estimate relational hallucination levels. A diffusion-based conditioning mechanism (RePaint) allows the model to “re-impute” its own outputs, comparing predictions across denoising steps to measure deviation. The authors design relational versions of common MVTS datasets (rECL, rWTH, rTraffic, rIllness, rETT) by adding synthetic relational variables (sums, differences, products, nonlinear functions). Experiments show that state-of-the-art MVTS foundation models (MOMENT, TIMER) hallucinate relationally up to 59.5% as much as a weak baseline, while the diffusion-based mitigation reduces this by up to 47.7%. Results across three task types (over-constrained, under-constrained, forecast) confirm the method’s robustness, with low overlap coefficients between low- and high-hallucination distributions (≤1% in most cases).

### Strengths
This paper provides the first formalization of hallucination in the time-series domain, making an important conceptual and methodological contribution. The definitions of distributional and relational hallucination are clearly motivated by NLP analogies yet appropriately adapted for MVTS. The proposed Combined Error (CE) metric is elegant and computationally efficient, as it reuses a diffusion model’s denoising dynamics to quantify internal consistency without requiring external labels or supervision. The authors demonstrate ingenuity by creating relational benchmark datasets (rECL, rWTH, etc.) to enable quantitative evaluation, a valuable contribution in itself. The experiments are thorough and well-documented, showing clear and interpretable metrics (Tables 1–2). The CE-based quartile thresholding method for hallucination detection is simple yet empirically effective, yielding well-separated distributions across datasets. The mitigation strategy—sampling multiple responses and selecting the one with the lowest CE—provides an effective and generalizable filtering mechanism that reduces relational error substantially across models. Methodological transparency (Section 6, Reproducibility Statement) is commendable, and the inclusion of RePaint conditioning and diffusion background ensures accessibility for non-specialist readers.

### Weaknesses
Despite its novelty, the paper has several weaknesses limiting its theoretical depth and empirical generalizability. The diffusion-based CE metric is heuristically motivated and lacks theoretical grounding linking it to true relational error; no formal proof connects CE to hallucination likelihood beyond empirical correlation. Furthermore, the evaluation setting is limited—all relational datasets are derived from existing benchmarks by appending simple transformations, so the results may not generalize to complex domains like finance, climate, or medical forecasting. The experimental comparison is restricted to only two open-source FMs (MOMENT and TIMER), both of which are structurally different and much larger than the diffusion baseline, complicating fairness in interpretation.

### Questions
The paper would benefit from a more rigorous theoretical connection between the CE metric and the true relational error—perhaps via a proof of monotonicity or boundedness.

### Soundness
2

### Presentation
2

### Contribution
2
