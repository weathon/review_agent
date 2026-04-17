# CaliDist: Calibrating Large Language Models via Behavioral Robustness to Distraction

- Decision: Reject
- Scores: 4, 6, 4

## Abstract
For Large Language Models (LLMs) to be trusted in high-stakes applications, it is paramount that their confidence scores are well-calibrated. However, existing calibration methods often overlook a critical dimension of trustworthiness: a model's behavioral robustness to irrelevant or misleading information. In this paper, we argue that a model's true confidence should reflect its stability under cognitive pressure. We introduce \textsc{CaliDist}, a novel, post-hoc calibration framework that directly measures and penalizes a model's susceptibility to distraction. \textsc{CaliDist} quantifies how an LLM's predictions and certainty change when its input prompt is perturbed with semantic \textit{distractors}. This instability signal is then used to adaptively scale the model's initial confidence score. 
Our extensive experiments on seven Natural Language Understanding (NLU) classification benchmarks using six distinct LLMs show that \textsc{CaliDist} consistently achieves lower Expected Calibration Error (ECE) than several baselines. Remarkably, our method reduces the ECE from 19\% to 11\% on average—a relative improvement of 47\%—demonstrating that behavioral stability is a powerful and practical signal for calibration.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces an inference-time technique to calibrate the outputs of LLMs, aiming to make their responses more robust and reliable. The method first inserts distractors into the input prompt, obtaining an output derived from the "perturbed" input. By comparing this output with the original output from the "clean" input, the method computes two scores representing the model's robustness to these distractions. These scores are then combined into a coefficient used to adjust the model's predictions.

### Strengths
1.  The method is compatible with both white-box and black-box models.
2.  Although it requires multiple forward passes, the method requires fewer forward passes than the baselines.
3.  The study includes multiple designs of distractors.

### Weaknesses
1.  Lines 38-39: The paper lacks discussion on the Bayesian family of calibration approaches, such as [1,2,3]. In addition, regarding Lines 71-73 ("where low competence on a task is paired with an inflated overestimation of ability..."), isn't this concept equivalent to an inaccurate estimate of **epistemic uncertainty**? If this concept is related to the conventional view of epistemic and aleatoric uncertainty, the authors should discuss this relationship to avoid reinventing concepts and to help readers connect the work to previous lines of calibration/uncertainty quantification research.
2.  Lines 239-253: Although the design of $\lambda$ ensures that "This score is designed to be high when both $\mu$ and $\delta$ are low", it may have different sensitivities to changes in $\mu$ and $\delta$. Specifically, the typical values and change of $\mu$ depend on the number of distractors $k$, while $\delta$ may also have a different range of "typical values". Therefore, the design of Eq. (3) might implicitly favor one of $\mu$ or $\delta$. Furthermore, in Eq. (4), the sigmoid function has saturation regions, which could affect the sensitivity of the resulting calibration coefficients $\sigma(\lambda, \alpha, \beta)$ to $\lambda$. The authors need to more **quantitatively** justify the choices for Eq. (3) and Eq. (4). Otherwise, the methodology appears highly heuristic.
3.  The methodology relies on the insertion of distractors. While this is straightforward for tasks like NLI and classification, designing and inserting effective distractors for open-ended generation and complex logical reasoning is substantially more challenging. This potentially limits the methodology's practical application scenarios.
4.  Related to Point 3: From an application perspective, if users want to use this approach to get a more reliable answer from an LLM, they must input their prompt *and* devise a distractor text and determine how to insert it. This workflow is much harder to scale compared to methods like Temperature Scaling, especially in multi-round chat contexts.
5.  Lines 312-313: Table 1 is too wide and exceeds `\textwidth`.


[1] Yang, Adam X., et al. "Bayesian Low-rank Adaptation for Large Language Models." ICLR 2024.

[2] Wang, Yibin, et al. "Blob: Bayesian low-rank adaptation by backpropagation for large language models." NeurIPS 2024.

[3] Li, Yawei, et al. "Calibrating LLMs with Information-Theoretic Evidential Deep Learning." ICLR 2025.

### Questions
Please see the Weaknesses

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
4

### Summary
This paper introduces CALIDIST, a novel framework for calibrating the confidence scores of Large Language Models (LLMs). The method leverages a measure of behavioral robustness—specifically, the stability of a model's predictions and confidence when its input is perturbed by semantic distractors. Extensive experiments demonstrate that CALIDIST consistently improves calibration, outperforming existing methods. The framework also serves as an effective behavioral proxy for temperature scaling in black-box settings.

### Strengths
1. The calibration of verbalized confidence in LLMs is a critical and highly relevant research topic. This study offers significant practical implications.

2. The paper provides a novel psychological grounding for the behavior of LLM confidence under distraction attacks, which is an original contribution.

3. The experiments are thorough, covering multiple perspectives, and the ablation studies are comprehensive.

4. The paper is well-written and clear.

### Weaknesses
1. The core intuition (robustness under perturbation) appears very similar to that of [1]. The authors should explicitly clarify the distinctions and similarities between their work and [1].

2. The lack of experimental results on state-of-the-art (SOTA) large-scale models (e.g., GPT-5 or Gemini 2.5 Pro) limits the generalizability of the findings. The black-box models used, GPT-4o-Mini (OpenAI, 2024) and Gemini 2.0 Flash (Google, 2024), are relatively smaller-scale, and their performance gap with open-source models may not be substantial.

3. The evaluation datasets do not seem to include benchmarks focused on mathematical or logical reasoning, such as GSM8K.

4. Regarding the "DISTRACTOR STYLES," it is unclear if the proposed categorization is exhaustive or mutually exclusive. The authors should justify this taxonomy and explain why it is a suitable classification.

[1] Zhou Z, Jin T, Shi J, et al. SteerConf: Steering LLMs for Confidence Elicitation[J]. NeurIPS 2025.

### Questions
Please refer to the weaknesses section. All points within that section reflect my opinion.

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
4

### Summary
This paper introduces CALIDIST, a post-hoc calibration framework that adjusts LLM confidence by evaluating behavioral robustness to semantic distractions. The method perturbs model inputs using designed “distractors” (assertion, probe, and corruption styles) and measures the stability of predictions and confidence scores under these perturbations. These instability measures are then combined into a reliability score and used to scale the model’s confidence via a parameterized sigmoid function. The framework is applied both to open-box models (using logits) and black-box APIs (using verbalized confidence). Extensive experiments across seven NLU datasets and six LLMs show improved calibration (lower ECE and Brier scores) compared to Temperature Scaling, Self-consistency, and other baselines.

### Strengths
1. The authors explore three semantically distinct types of input perturbations, offering a structured way to probe model robustness rather than random paraphrasing.
2. Experiments cover both open-source and proprietary LLMs, showing consistent gains across multiple datasets.
3. The method requires only a few additional forward passes and no model retraining, making it lightweight and deployable.

### Weaknesses
1. The core idea, estimating reliability from the stability of model outputs under input perturbations, has been explored in recent works such as Gao et al. 2024 and Khanmohammadi et al. 2025. Both measure predictive or representational stability under perturbed inputs to derive uncertainty or calibration signals. CALIDIST shares the same underlying mechanism (perturb → measure stability → rescale confidence) and thus may appear conceptually incremental. The main novelty lies in the behavioral framing and the distractor taxonomy, rather than in a fundamentally new algorithmic principle.
2. The paper does not explicitly position itself against these recent perturbation-based methods. Without a clear comparison (empirical or conceptual) it is difficult to assess whether the proposed “behavioral robustness” metric provides distinct predictive power.
3. The paper only reports ECE and Brier Score, which can be sensitive to binning choices and sample imbalance.
It would be more convincing to include a ranking-based metric such as AUROC (Area Under ROC Curve), which directly measures whether the confidence scores discriminate correctly between correct and incorrect predictions, and is more robust under varying dataset distributions.
4. Minor point. The sigmoid rescaling function and its parameters (α, β) are tuned via grid search on each task, but no theoretical motivation or generalization analysis is provided. This makes the approach appear heuristic rather than principled.

Reference:
1. Gao et al. 2024. SPUQ: Perturbation-Based Uncertainty Quantification for LLMs
2. Khanmohammadi et al. 2025. CCPS: Calibrating LLM Confidence by Probing Perturbed Representation Stability

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
