# Uncertainty Distillation: Teaching Language Models to Express Semantic Confidence

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4

## Abstract
As large language models (LLMs) are increasingly used for factual question-answering, it becomes more important for LLMs to have the capability to communicate the likelihood that their answer is correct. For these verbalized expressions of uncertainty to be meaningful, they should reflect the error rates at the expressed level of confidence. However, when prompted to express confidence, the error rates of current LLMs are inconsistent with their communicated confidences, highlighting the need for uncertainty quantification methods. Many prior methods calculate lexical uncertainty, estimating a model's confidence in the specific string it generated. In some cases, however, it may be more useful to estimate semantic uncertainty, or the model's confidence in the answer regardless of how it is verbalized. We propose a simple procedure, uncertainty distillation, to teach an LLM to verbalize calibrated semantic confidences. Using held-out data to map initial uncertainty estimates to meaningful probabilities, we create examples annotated with verbalized probabilities for supervised fine-tuning. We find that our method yields verbalized confidences that correlate well with observed error rates, even when compared to  strong baselines, some of which are more than twenty times slower at inference time.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes an uncertainty distillation method, which teaches the model to express its uncertainty to the answer through sft using a dataset with confidence labels collected through Monte Carlo sampling. Compared to sampling-based semantic UQ methods the proposed method has better inference-time efficiency. Experiments show that uncertainty distillation generally achieves higher AUROC scores and works well on OOD scenarios.

### Strengths
- Paper is clear and easy to follow

- The proposed method makes sense in theory and shows good performance under the predefined setup.

- The experiment is comprehensive an convincing.

### Weaknesses
- Practical Issue: The proposed method requires a significant amount of Monte Carlo samples of the training data responses, while the performance under more significant distribution drift from training to test is not studied. This may limit the practicability of the proposed method due to the significant computation overhead. The post-hoc calibration in Section 3.2 post an additional constraint to the answer format, i.e., it is only applicable to closed-form solutions such multi-choice QA, which further reduce the use cases of this method.

- Baselines: I'm not very familiar with verbalized uncertainty but it looks like there are some more similar baselines such as [1]. Some more efficient methods mentioned in the "Lexical uncertainty quantification" paragraph in Related Works can also be used as baselines, since the authors are testing on open source models.

- Minor Correctness Issue: The authors' claim in footnote 11 about not adopting ECE is, IMO, not precise. First of all, ECE does not require continuous probability. In addition, it has only one hyperparameter---the number of bins. But the authors present calibration plot in Figure 2 anyways so I do not find this issue significant.

- Reproducibility: no code or data available.


[1] Liu, Shudong, et al. "Can llms learn uncertainty on their own? expressing uncertainty effectively in a self-training manner." Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing. 2024.

### Questions
See above

### Soundness
3

### Presentation
3

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
This paper proposes uncertainty distillation, a fine-tuning approach to teach LLMs to verbalize calibrated semantic confidence in their answers. The method uses Monte Carlo sampling with semantic normalization to estimate answer probabilities, applies post-hoc calibration via isotonic regression, and fine-tunes models to output verbalized confidence levels alongside predictions. Evaluated on MMLU and SocialIQA with in-domain and out-of-domain transfer scenarios, the approach achieves competitive or better AUROC compared to stronger baselines, particularly semantic clustering methods, with negligible inference-time overhead.

### Strengths
The paper addresses a problem: models are often confidently incorrect, and enabling them to express calibrated uncertainty has significant practical value. The three-step approach is straightforward and doesn't require architectural changes or external models at inference time, making it immediately applicable to existing models.

A major practical advantage is efficiency. Unlike semantic clustering methods requiring 20 inference passes, this approach generates a single response with verbalized confidence, maintaining minimal inference-time overhead. The method is rigorous in its experimental design with proper in-domain and out-of-domain evaluation, comprehensive ablations on including incorrect examples, and careful dataset splits preventing data contamination. Results are evaluated on models ranging from small (FLAN-T5, T5-base) to medium (Llama-3B, Ministral-8B).

The paper reveals interesting findings: there's a calibration-accuracy trade-off where smaller models suffer more when trained on incorrect examples, and the approach shows better robustness to domain shift than supervised baselines. The comparison between Instruct-T5 (unseen during pre-training) versus FLAN-T5 (potentially contaminated) effectively illuminates which performance gains are genuine. The presentation is clear with effective figures showing calibration quality across confidence bins.

### Weaknesses
The evaluation is restricted to multiple-choice QA with only two datasets (MMLU and SocialIQA). There's no evaluation on generation tasks like summarization, machine translation, or code generation, nor on open-ended tasks where semantic normalization becomes challenging. Model evaluation is limited to parameters ≤8B, leaving unclear how the approach scales to larger models.

Semantic normalization is a fundamental limitation. The current approach (isolating answer letters, removing punctuation) works for multiple-choice QA but is inherently task-specific. The appendix acknowledges this is "trivial" for short-form QA but "challenging" for complex generation tasks, yet provides no concrete solution. This severely restricts generalizability beyond standardized QA formats.

A critical practical issue is data contamination. FLAN-T5 results show substantial performance degradation when the calibration set appears in pre-training data, raising questions about real-world applicability. The paper doesn't address how to identify unseen calibration data for closed-source models, making deployment in production environments uncertain.

The theoretical understanding is limited. Why does this particular combination of techniques improve calibration? The connection between semantic probability estimates and actual accuracy remains unclear. Design choices like the five-bin verbalization scheme and specific confidence labels lack principled justification, and the paper doesn't discuss when the method might fail or produce misleading outputs.

Several methodological concerns undermine confidence in the results. The choice of 100 Monte Carlo samples appears arbitrary with no principled justification.Additionally, the offline requirement for collecting and annotating calibration data is a practical burden. There's a potential overfitting issue: post-hoc calibration is applied on the calibration set and evaluated on the same set. Table 1 shows accuracy drops in some cases (Llama-3B MMLU AUROC improves but accuracy decreases), but when this trade-off is acceptable isn't clearly discussed. High-confidence predictions are often sparse (34.6% for Ministral on MMLU), limiting practical utility.

### Questions
1. Beyond QA, how would you apply semantic normalization for summarization or translation tasks?

2. For FLAN-T5, why does post-hoc calibration actually hurt performance (Table 6)?

3. What's the optimal number of Monte Carlo samples as a function of model size?

4. Can you provide deeper analysis of the accuracy-calibration trade-off?

5. How does the method perform with more fine-grained confidence levels (7-10 categories vs. 5)?

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
SFT is used to make models quantify their uncertainty directly using generated tokens after giving an answer to factual questions. Using a Monte Carlo sampling approach, semantically equivalent answers all get corresponding values. There is much less compute overhead during inference than you would get for a fully blackbox method, at the cost of being able to retrain the model.

### Strengths
Regarding the semantic/lexical/verbalized uncertainty quantification, the definitions make sense and are a good summary of existing black-box approaches. The test for domain shift is an important question for such trained methods, and strengthens the empiric section of this work.

### Weaknesses
More than strongly beating all baselines, this paper seems to propose a tradeoff. Better semantic confidence, in some cases, which comes at the cost of some model performance. Extra tokens (very few) have to be generated, but not as many as in sampling based methods. Fine tuning on the other hand requires some sampling, but not access to activation at inference time (I would here note that unlike what is stated line 092, fine tuning does access model weights - even if pragmatics of that paragraph imply authors mean direct access during inference, I would advise rephrasing). Overall the tradeoff in compute in terms of memory, and time could be more systematic.

Regarding the semantic/lexical/verbalized uncertainty quantification, I do not see metrics of performance on input variation. Is this something you expect the monte Carlo sampling to take care of? I do not see an empirical verification of it's success. Perhaps something along the lines of the Kuhn et al 2023, or a model generated approach Mahaut et al 2024 could be relevant to make explicit the success of your method, beyond performance. This seems to be a major advantage of the method.

Typo and minor:
There seems to be a missing space in the footnote line 214
I find the use of the word DISTILLATION confusing. Distillation is already applied in specific setting of model compression and teacher/student model - applying it here leads to extra confusion in my opinion.

### Questions
See Weaknesses. Additionally, more of a personal curiosity question : why the ISOTONIC regression specifically here?

### Soundness
3

### Presentation
3

### Contribution
2
