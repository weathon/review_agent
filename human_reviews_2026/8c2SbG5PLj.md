# Turning the Spell Around: Lightweight Alignment Amplification via Rank-One Safety Injection

- Avg Score: 5.00
- Decision: Reject
- Scores: 2, 6, 6, 6

## Abstract
Safety alignment in Large Language Models (LLMs) often involves mediating internal representations to refuse harmful requests. Recent research has demonstrated that these safety mechanisms can be bypassed by ablating or removing specific representational directions within the model. In this paper, we propose the opposite approach: ***Rank-One Safety Injection (ROSI)***, a white-box method that amplifies a model's safety alignment by permanently steering its activations toward the refusal-mediating subspace. **ROSI** operates as a simple, fine-tuning-free rank-one weight modification applied to all residual stream write matrices. The required safety direction can be computed from a small set of harmful and harmless instruction pairs. We show that **ROSI** consistently increases safety refusal rates - as evaluated by Llama Guard 3 - while preserving the utility of the model on standard benchmarks such as MMLU, HellaSwag, and Arc. Furthermore, we show that **ROSI** can also re-align 'uncensored' models by amplifying their own latent safety directions, demonstrating its utility as an effective last-mile safety procedure. Our results suggest that targeted, interpretable weight steering is a cheap and potent mechanism to improve LLM safety, complementing more resource-intensive fine-tuning paradigms.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
ROSI enhances model security by guiding model activations to a rejection-related subspace, eliminating the need to fine-tune the original model and modifying weights with the help of residual flow. Experiments on llama guard3 and general tasks demonstrate that the proposed method improves the rejection rate of harmful requests while maintaining practicality.

### Strengths
ROSI uses rank-one updates instead of multi-vector linear combinations, making it more lightweight. It tested various models and security evaluation datasets, and the results showed that it can improve the rejection rate. The paper is well-structured and the results are intuitive, making it a medium-quality empirical paper.

### Weaknesses
**1. Lack of in-depth follow-up and reference to cutting-edge work such as AlphaEdit/AlphaSteer:**

(1) The core idea of ​​ROSI is consistent with that of AlphaEdit [1], the best ICLR paper. However, there is no relevant citation and only a brief mention of "our method is inspired by interpretability-based steering". Both rely on extracting a direction vector in the model activation space and achieving good behavior control through linear intervention. In comparison, this paper has limited room for innovation and lacks in-depth discussion of the theoretical mechanism, assumptions and differences of activation guidance. 


(2) The discussion of Beyond Steering is very interesting. It focuses on research related to finetuning and red teaming outside of editing. It is recommended to add more supplements to highlight the focus of the work. 


**2. Lack of theoretical depth and mechanism analysis:**

There is no explanation of why rank-one injection can effectively capture or amplify the rejection subspace signal, nor is its theoretical advantage over multi-direction steering explained. There is no analysis of the stability of ROSI at different layers and different model sizes. The current results are only empirical observations and lack theoretical support.



Ref:

[1] Fang J, Jiang H, Wang K, et al. Alphaedit: Null-space constrained knowledge editing for language models. ICLR'25

### Questions
1. Is there any inter-layer interference or redundancy when ROSI performs multi-layer intervention?

2. Why haven't the experiments been replicated on the same security benchmark (such as the harmful-pairs dataset used by AlphaEdit) for direct comparison?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a fine-tuning-free safety enhancement method, ROSI (Rank-One Safety Injection), which permanently strengthens LLM safety by injecting a single “safety direction” as a rank-one modification into the residual stream weights.
The method derives the safety direction from a small set of harmful/harmless instruction pairs and significantly improves refusal rates and robustness against jailbreak attacks, without impairing model capability.

### Strengths
1. The proposed method is conceptually simple yet effective, introducing only a lightweight rank-one modification that achieves substantial safety improvements across diverse models and benchmarks.

2. The paper is well structured and clearly written, making the motivation, methodology, and experimental design easy to understand and follow even for readers outside the safety alignment community.

3. The rank-one update is easy to implement, requires no retraining, and facilitates reproducibility and deployment.

### Weaknesses
1. Could the learning of the safety direction be extended to a multi-dimensional subspace rather than a single vector?

This assumption may oversimplify the underlying representation of safety-related behaviors, which could be inherently multi-dimensional.

2. The stability of the safety vector is not analyzed — is it highly sensitive to the specific set of harmful and harmless prompts used?

3. Safety evaluation mainly relies on LLAMA GUARD 3; have the authors tested with multiple evaluators or different safety assessment models?

### Questions
See Weakness Section

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
3

### Summary
This paper proposes Rank-One Safety Injection (ROSI), a lightweight method for enhancing safety alignment. The approach computes a safety vector in the representation space of LLMs and injects a rank-one matrix update along this direction into the model’s weights, achieving safety alignment enhancement without fine-tuning. The method is simple, interpretable, and demonstrates consistent effectiveness across multiple models and benchmark evaluations.

### Strengths
The ROSI method is simple and interpretable, with a clear mathematical formulation. It requires only a single rank-one weight modification, needs no additional training, and offers transparent and controllable operation.
The ROSI method enhances model safety while exerting minimal impact on general performance. Moreover, it introduces no additional runtime overhead, demonstrating strong practical value.

### Weaknesses
The injection strength hyperparameter $\alpha$ and the number of injection layers in ROSI may affect model stability. It is recommended to include additional ablation studies to clarify their potential impact on model safety and general performance.
The paper lacks adversarial evaluations against several classic jailbreak attacks methods, such as GCG [1], PAIR [2], RandomSearch [3], etc. Adding such attack–defense experiments would help further validate the method's effectiveness in enhancing model safety.
The paper lacks a direct comparison with other defense methods, such as SmoothLLM [4], Safe LoRA [5], Jailbreak Antidote [6], etc.
[1] Zou, Andy, et al. "Universal and transferable adversarial attacks on aligned language models." _arXiv preprint arXiv:2307.15043_ (2023).

[2] Chao, Patrick, et al. "Jailbreaking black box large language models in twenty queries." _2025 IEEE Conference on Secure and Trustworthy Machine Learning (SaTML)_. IEEE, 2025.

[3] Andriushchenko, Maksym, Francesco Croce, and Nicolas Flammarion. "Jailbreaking leading safety-aligned llms with simple adaptive attacks." _arXiv preprint arXiv:2404.02151_ (2024).

[4] Robey, Alexander, et al. "Smoothllm: Defending large language models against jailbreaking attacks." _arXiv preprint arXiv:2310.03684_ (2023).

[5] Hsu, Chia-Yi, et al. "Safe lora: The silver lining of reducing safety risks when finetuning large language models." _Advances in Neural Information Processing Systems_ 37 (2024): 65072-65094.

[6] Shen, Guobin, et al. "Jailbreak antidote: Runtime safety-utility balance via sparse representation adjustment in large language models." _arXiv preprint arXiv:2410.02298_ (2024).

### Questions
- Besides using the mean to extract safety vectors, could other mathematical approaches, such as principal component analysis (PCA), be employed? How would safety vectors extracted using different methods affect the performance of the approach?  
- Could ROSI be extended to other value dimensions, such as honesty, to help mitigate hallucinations in large models?

### Soundness
3

### Presentation
3

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
This paper introduces ROSI (Rank-One Safety Injection), a white-box method for enhancing safety alignment in LLMs through permanent rank-one weight modifications. The approach extracts a "safety direction" from harmful/harmless instruction pairs using difference-in-means, then injects this direction into residual stream write matrices via the update rule W'_out ← W_out + α·ŝ·w̄^T. Experiments across aligned models (LLAMA, QWEN, GEMMA, YI) and uncensored models (DOLPHIN series) demonstrate improved harm refusal rates and jailbreak robustness with minimal utility degradation on standard benchmarks.

### Strengths
1. ROSI provides a lightweight alternative to expensive fine-tuning, requiring only 50 instruction pairs and simple weight modifications

2. The paper tests across 13 models, multiple safety benchmarks (CATQA, HARMBENCH, WILDJAILBREAK), utility benchmarks (MMLU, HELLASWAG, ARC, etc.), and attack scenarios

3. Demonstrating effectiveness on both aligned and uncensored models broadens the method's utility

4. Tables 3 and 6 show remarkably stable performance across capability benchmarks (typically <0.5% average change)

5. The method maintains transparency about what is being modified and why, unlike black-box fine-tuning approaches

### Weaknesses
1. Why is w̄·\hat{s}^T the right rank-one update? The paper doesn't justify this choice over alternatives like random projections or learned directions. An ablation comparing different rank-one formulations would strengthen the claims.

2. The paper states l* is "selected based on a validation set" but provides no details about this validation procedure, what metrics were optimized, or how many layers were tested.

3. Only 50 harmful/harmless pairs seems quite small. What's the variance across different samples?

4. The safety system prompt approach (Figure 2, Appendix A) seems somewhat circular—you're using a prompt to elicit safety behavior, then trying to make that permanent. How robust is this to variations in the prompt? The ❢ ablations suggest this is fragile for smaller models.

### Questions
1. Which layers benefit most from ROSI? Did you try layer-specific \alpha values or applying ROSI to only a subset of layers?
2. Does a safety direction extracted from one model transfer to architecturally similar models? This could have interesting implications for safety.

### Soundness
3

### Presentation
3

### Contribution
2
