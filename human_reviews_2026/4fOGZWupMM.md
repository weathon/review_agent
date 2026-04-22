# Retain and Adapt: Auto-Balanced Model Editing for Open-Vocabulary Object Detection under Domain Shifts

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 4, 6, 8, 6

## Abstract
Recent advances in Open Vocabulary Object Detection (OVOD) have shown strong performance on standard benchmarks, but performance drops sharply under out-of-distribution (OOD) shifts. Continual learning offers a potential remedy by sequentially integrating new tasks, yet existing methods often struggle to balance retaining the pre-trained model capabilities with adapting to new tasks, and usually require retraining under specific task orders. To address these limitations, we observe that model editing naturally lends itself to this setting, as it enables efficient knowledge injection while retaining prior capabilities. Building on this insight, we introduce $\textbf{A}$utomatically $\textbf{B}$alanced $\textbf{M}$odel $\textbf{E}$diting ($\textbf{ABME}$), which injects new task knowledge into the powerful OVOD models while preserving the model’s original abilities. We first stores compact key–value representations with storage cost independent of task volume. Then we leverage the stored KV matrices to automatically balance the new and old knowledge for varying learning scenarios, 
supporting order-agnostic task insertion or removal without additional retraining. Experiments show that ABME consistently achieves a better trade-off between maintaining pre-trained performance and adapting to diverse OOD tasks compared to existing continual learning approaches for open-vocabulary object detection, and generalizes seamlessly across different models and task scales.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes ABME, which extends OVOD models to new domains without retraining. ABME injects new knowledge into FFNs through compact KV representations. Experiments on CDFSOD and ODinW-13 show that ABME maintains over 95% of base performance while achieving up to 97% adaptation gains on new domains, outperforming baselines like EWC, Adam-NSCL, and SD-LoRA.

### Strengths
1. ABME introduces model editing to OVOD for the first time.
2. ABME are validated across 19 datasets and multiple models (Grounding DINO, GLIP) with consistent performance gains.
3. The paper is well-written. The charts and formulas look clear and well-designed.

### Weaknesses
1. The auto-balancing mechanism lacks deeper theoretical analysis beyond empirical validation.
2. The approach assumes that FFN layers store the majority of knowledge, which might not hold for all architectures.
3. Detailed memory–accuracy trade-offs of KV storage are not analyzed.
4. The paper lacks broader open-world or long-term continual settings

### Questions
1. What are the limits of automatic balancing, and could it fail when domain shifts are extremely large or heterogeneous?
2. How would ABME perform in fully online continual learning with hundreds of sequential domains?
3. Can the KV-based editing be extended to multimodal or temporal models like video-language detection?

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
An innovative Auto-Balanced Model Editing (ABME) framework is proposed and applied to the domain of open-vocabulary object detection. By efficiently fine-tuning FFN layers, it achieves performance comparable to full model fine-tuning while effectively mitigating the issue of catastrophic forgetting. This research provides a novel reference solution for domain adaptation problems, demonstrating certain academic value and application potential.

### Strengths
1.The idea that transitioning the model editing technology from LLM to domain adaptation tasks in OVOD is motivating.

2.The proposed use of a data-driven regularization matrix to replace manual hyperparameter tuning is good.

3.The paper demonstrates clear logic, rigorous theoretical derivation.

### Weaknesses
1.There is no ablation on the choice of edited regions, like which layers of the FFN. To my knowledge, model editing at different layers in LLM has a significant impact.

2.The author did not provide the code for the paper, which to some extent reduces the reproducibility and practical impact of the study.

3.The baseline set is limited. The baselines in this paper , like Adam-NSCL(2021),EWC(2017) are too old and outdated.

### Questions
1.\begin{equation}
\min_{W} \|KW - V\|_F^2 + \|\Gamma(W - W_0)\|_F^2,
\end{equation}


\begin{equation}
\Gamma = \text{diag}\left(s_1^{1/4}, s_2^{1/4}, \ldots, s_d^{1/4}\right), \quad s_i = \sum_t k_{ti}^2.
\end{equation}
What is the rationale behind the specific definition of $\Gamma$? Why is $s_i^2$ not the  $s_i$? Is it derived from the least squares method? Have you considered the Gaussian kernel?


2.Can authors provide a comparison with more recent competitors.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents a method for open-vocabulary object detection that reframes adaptation as a model-editing problem. Instead of standard fine-tuning or rehearsal-based continual learning, the proposed approach leverages compact key–value statistics extracted from feed-forward layers to encode new task knowledge. The method automatically balances adaptation and retention through a data-driven regularization mechanism, eliminating the need for manually tuned hyperparameters. By maintaining only aggregated statistics, the system supports scalable and order-agnostic task integration without storing original data. Empirical results on multiple few-shot detection benchmarks show that the approach achieves strong performance in new domains while effectively preserving prior knowledge. Conceptually, the work bridges ideas from knowledge editing in large language models to vision-based detectors, offering a lightweight and interpretable alternative for cross-domain adaptation.

### Strengths
1. Framing few-shot detection as a model-editing problem is conceptually elegant and provides a refreshing perspective that clearly distinguishes this work from conventional fine-tuning and continual learning paradigms. 
2. The proposed method delivers a precise and effective solution to the central challenge of balancing adaptation and retention, offering a well-grounded formulation that is both simple and practical. 
3. Extensive experiments on ODinW-13 and CDFSOD demonstrate consistent gains in adaptation performance while maintaining strong retention of prior knowledge, reinforcing the robustness and practical value of the proposed approach.

### Weaknesses
While the paper presents an elegant and well-motivated formulation, it inherits the core assumption from LLM-based editing that knowledge primarily resides in FFN layers. The direct application of this assumption to vision detectors may not be fully validated, as the distribution of transferable representations could differ across modalities. The paper would be strengthened by a more detailed analysis or empirical study investigating whether FFN weights indeed serve as the main locus of visual knowledge.

### Questions
In section 4.2, the paper replaces the pre-defined scalar regularization weight $\lambda$ with a data-adaptive diagonal matrix $\Gamma$, which scales each feature dimension according to the key-vector energy. While this design is intuitively appealing and empirically effective, the underlying rationale is only briefly mentioned, and the referred Appendix B.2 seems missing. Could the authors elaborate on the specific motivation for this formulation.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper tackles the challenge of maintaining OOD robustness in Open Vocabulary Object Detection (OVOD). The authors propose Automatically Balanced Model Editing (ABME) to efficiently integrate new task knowledge while preserving pre-trained capabilities. ABME stores compact key–value representations and automatically balances old and new knowledge without retraining. It supports order-agnostic task insertion and removal, overcoming common continual learning limitations. Experiments show that ABME achieves a better trade-off between adaptation and retention than existing methods.

### Strengths
1. This paper is well-structured and easy to follow.

2. The idea of model editing through compact key–value representations and automatically balancing old and new knowledge is interesting and innovative.

### Weaknesses
1. In Eq. (5), what is the explicit advantage of the design of $\Gamma$? How does it achieve data-adaptivity? Could the authors provide experimental evidence or theoretical insights to support this design choice?

2. From Tables 1–3 and Table 6, why do RR and AGR decrease as the number of samples (shots) increases? This trend appears to contradict the common expectation that model performance should improve with larger data sample sizes. Could the authors clarify the reason behind this behaviour?

3.  Please provide a comparison of the computational cost (e.g., memory usage, training time) of the proposed method versus existing baselines.

4. Does the proposed ABME method generalize to other models and tasks, such as image classification with CLIP? I believe that including additional experiments on such tasks could strengthen the validity and generality of the proposed approach.

5. A simple approach to balance old and new knowledge is to use residual connections of weights. Could the authors demonstrate the superiority of ABME compared to this baseline, either through empirical results or theoretical justification?

### Questions
See Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3
