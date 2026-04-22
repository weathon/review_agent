# Self-Supervised Dynamical System Representations for Physiological Time-Series

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 4, 2

## Abstract
The effectiveness of self-supervised learning (SSL) for physiological time series depends on the ability of a pretraining objective to preserve information about the underlying physiological state while filtering out unrelated noise. However, existing strategies are limited due to reliance on heuristic principles or poorly constrained generative tasks. To address this limitation, we propose a pretraining framework that exploits the information structure of a dynamical systems generative model across multiple time-series. This framework reveals our key insight that  class identity can be efficiently captured by extracting information about the generative variables related to the system parameters shared across similar time series samples, while noise unique to individual samples should be discarded. Building on this insight, we propose PULSE, a cross-reconstruction-based pretraining objective for physiological time series datasets that explicitly extracts system information while discarding non-transferrable sample-specific ones. We establish theory that provides sufficient conditions for the system information to be recovered, and empirically validate it using a synthetic dynamical systems experiment. Furthermore, we apply our method to diverse real-world datasets, demonstrating that PULSE learns representations that can broadly distinguish semantic classes, increase label efficiency, and improve transfer learning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
PULSE introduces a self-supervised pretraining method grounded in a dynamical-systems generative model.

### Strengths
1. Strong conceptual bridge between SSL and dynamical-systems theory.
2. Cross-reconstruction objective is original and well-motivated.
3, Empirical results show consistent gains across modalities.
4. Theoretical analysis clarifies what information is retained.

### Weaknesses
1. Mathematical proofs are brief, formal guarantees could be strengthened.
2. Requires grouping of “similar” time series, definition and implementation vague.
3. Limited comparison with modern contrastive MAE hybrids.
4. Does not explore scalability to high-dimensional multichannel data.

### Questions
1. Please provide a more detailed explanation of the method used to compute similarity between time-series samples during pretraining. Specifically, how is the similarity measure defined (e.g., Euclidean distance, cosine similarity, dynamic time warping)?
2. Compare with recent foundation-time-series models (e.g., Chronos, SleepTransformer). These models have demonstrated strong performance in various time-series tasks, and a comparison could help clarify the unique contributions and advantages of your model. Please highlight key differences in architecture, training strategies, and the specific domain(s) your model excels in.
3. To enhance the transparency of the model’s functionality, an ablation study comparing reconstruction and cross-reconstruction would be beneficial.
4. It would strengthen the paper to include a discussion on how your model performs across a variety of datasets with different characteristics.
5. Please provide further details on the training and optimization process, including hyperparameter tuning, loss functions, and any regularization techniques employed.

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
The paper addresses SSL for physiological time series, claiming that the standard methods: CL, MAE, and standard SVAEs, fail to isolate shared "system" dynamics from sample specific noise. It proposes PULSE, a framework based on a hierarchical dynamical systems view. The core idea is a pretraining objective that reconstructs a window from a randomly sampled start time within the same window, aiming to learn representations invariant to initial conditions. Experiments across synthetic and four real datasets (HAR, ECG, PPG, EEG) demonstrate PULSE often outperforms standard baselines.

### Strengths
1. The motivation is sound. Generic SSL heuristics are generally not suited for physiological signals where temporal structure is key. Framing this through dynamical systems is appropriate.
2. The paper provides a thorough comparison to other baselines across different algorithmic families.
3. The paper shows consistent performance improvements across diverse tasks especially on complex signals like EEG/ECG.

### Weaknesses
1. There appears to be a disconnect between the theoretical justification and the practical implementation. Theorem 1 states that full-sample masking (cross-reconstruction between independent samples) is required to isolate system parameters. However, PULSE uses partial window reconstruction from the same sample. The proof in appendix A indicates that partial masking fails to isolate and retains sample-specific information. The paper frames this as an approximation, but it seems to violate the conditions of the theory used to justify it.
2. The paper positions PULSE as a principled alternative to heuristic methods. However, it relies on its own set of design choices, such as the specific uniform sampling of t_0. It is unclear how this is fundamentally different from applying a random temporal cropping augmentation to a standard SVAE, a type of heuristic the paper initially critiques in the introduction.
3. The framework aims to separate "system" information from "initial conditions." However, since both encoders observe the same sample Y_i, and the system representation includes time-varying components $\tilde{\theta}_{i,t_k}$ derived from that same input, there is a risk of information leakage. The encoder could potentially copy local signal values directly into these time-varying parameters, allowing the decoder to reconstruct the signal without learning true underlying dynamics. 
4. The hierarchical model assumes a clean separation between shared "system parameters" and unique "initial conditions," which relies on the signal being stationary within the window W. In many physiological signals, this assumption may not hold. If W is small, there is a risk the "system" encoder might overfit to local, non-stationary patterns rather than true underlying dynamics. So selecting the right W is also somewhat of a heuristic like point 3 above.

### Questions
Please address the weaknesses section above.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a self-supervised framework for learning embeddings of physiological time series based on dynamical system modeling. By considering dynamical system modeling, the authors propose a new pre-training method to filter out irrelevant noise while creating embeddings that account for the physiological state. Namely, the embedding intends to represent the underlying dynamical systems  by cross-estimation rather than the observed trajectories. This pre-training framework intends to overcome limitations of current methods based on contrastive learning or variational encoding. Incrementally to previous work, the authors established a theorem ensuring the recovery of dynamical systems. In the experimental section, the authors explore the validity of the established theorem in degraded situations and evaluate their representations' performance in various ML tasks, including classification and transfer learning.

### Strengths
- The paper is incrementally written, making it easy to follow from start to end. The cross-reconstruction and its relaxation (PULSE) are well motivated and described. Theoretical estimation guarantees further back it.
- The paper benefits from an extensive experimental study, including analyzing the theorem validity on synthetic data, performance evaluation on real-world data for various ML tasks, and an ablation study.

### Weaknesses
- The estimation of dynamical systems from data is also known as dynamical mode decomposition; the authors should also position their work with regard to this literature.
- The “Time-series Dataset Generative Model” (section 3.1) is not easy to follow while central to the paper. I invite the authors to revise this paragraph; they can, for instance, make better use of Figure 2.
- The experiments seem to be run on a single train-validation-test split and a single network initialization. To ensure the robustness of the experimental results, I invite the authors to address this issue.
- As the classification experiment ends with a probe linear estimator, a 2D-visual representation of the embedding spaces would be interesting to add in the appendix to see if clusters of dynamical systems can represent the different classes.
- Format issue: In assumption 1, line 286, the abbreviation DAG is not defined.

### Questions
- Regarding practical settings, a point that I found unclear is the passage from latent space to observable space ($g_y$). How is it implemented in practice? Are its parameters considered in the DS embedding ($\Theta$)?
- Can the authors address my concern regarding the robustness of experimental results described in the weaknesses?
- An important baseline to compare to in the experiment would be the pretaining that considers the cross-reconstruction strategy with randomly selected pairs of subsequences. Can this baseline be added to the classification or ablation study?

### Soundness
2

### Presentation
2

### Contribution
2
