# Distilling Safe LLM Systems via Soft Prompts

- Decision: Reject
- Scores: 2, 4, 2, 2

## Abstract
Large Language Models (LLMs) have enabled machine learning to be integrated in complex tasks across various domains. This is a cause for concern since LLMs may respond to carefully crafted prompts with unsafe content, thus necessitating concrete safety mechanisms. Current solutions involve dual-model systems combining LLMs with guard models. However, the substantial memory and computational demands of guard models pose significant challenges for deployment. This paper proposes an efficient method for approximating the behavior of dual-model systems using learned embeddings, also known as soft prompts.We introduce a novel distillation framework which optimizes the total variation distance between the outputs of an LLM paired with a guard and the same LLM equipped with our soft prompts. At test time, the learned soft prompts are prepended to user prompts, providing safety at a fraction of the memory and compute costs incurred by a guard model. Our evaluations on various benchmarks demonstrate improved safety of the LLM, offering an efficient alternative to guard models for memory- and computation-constrained settings such as hardware applications.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes TV-DiSP, a framework for distilling dual-model safety system, composed of a base LLM and a guard model, into a single model augmented with soft prompts. Instead of running a full guard model at inference time, the authors train soft prompt embeddings to minimize the total variation distance between the response distribution of the two-model system and that of the distilled single model. At test time, the learned soft prompts are prepended to user prompts, enabling the LLM to emulate the “safe” behavior of the guarded system with dramatically lower memory and compute overhead.

### Strengths
* Timely motivation: tackles a real deployment bottleneck—guard models are expensive for on-device inference.
* Simplicity: soft-prompt distillation is lightweight, easy to reproduce, and compatible with quantized models.
* Comprehensive experiments: includes multiple datasets, model sizes, and PEFT baselines.
* Balanced evaluation: assesses both safety (SGS) and usefulness (MMLU).
* Generalization results: shows modest gains even on out-of-distribution datasets like Detect-Jailbreak.

### Weaknesses
* Limited novelty. TV-based loss and prompt distillation are modest extensions over prior PEFT safety methods (GuardFormer, SafeLoRA, Safety-Prefix, etc.).

* Guard dependency. The method is entirely dependent on the guard model’s decisions, so any bias or blind spot in the guard is replicated in the distilled system.

* No ablations on number or location of soft prompts. Appendix B.2 provides limited analysis (10–200 prompts) but no clear convergence curves.

* Questionable evaluation metric. The “Safety Guard Score (SGS)” is derived from a different guard model (LlamaGuard 8B) and may not reflect true human-judged safety i.e llm as a safety judge might be incorrect/biased etc.

* No latency measurements (token/sec) or deployment case studies despite mention of edge-device motivation.

### Questions
1. How does TV-DiSP compare against supervised fine-tuning on guard-labeled data, without distillation?
2. Did you evaluate safety degradation over time (catastrophic forgetting) when mixing safety and general downstream data?
3. How sensitive is the method to the choice of guard model (e.g., LlamaGuard, Nvidia-Nemo Guard vs Granite Guardian)?
4. Can this approach compound safety errors if the guard model mislabels benign responses?
5. Would adding adversarial training or data augmentation improve robustness beyond imitation of the guard?

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
4

### Summary
This paper proposes TV-DiSP, a method that distills a dual-model safe LLM system (base model + guard) into a single model using soft prompts. The approach trains soft prompts by minimizing the total variation distance between the outputs of the safe system and the base model with soft prompts. This aims to retain safety while avoiding the compute and memory cost of a separate guard model. Experiments on several small instruction-tuned models show improved safety scores with limited overhead. The paper compares against alternative parameter-efficient tuning methods and loss objectives, and includes ablations on soft prompt size and a brief discussion of adversarial robustness.

### Strengths
**Strength 1**: The paper proposes soft prompts to approximate the behavior of a dual-model safe LLM pipeline, improving safety while requiring only minimal additional compute and memory, making it suitable for edge deployment.

**Strength 2**: The use of total variation as the distillation objective provides a clear performance deviation bound, contributing a sound theoretical foundation.

**Strength 3**: Experiments cover multiple base models, different training distributions, in- and out-of-distribution safety benchmarks, and comparisons with several PEFT and loss-based baselines.

### Weaknesses
**Weakness 1**: Both training supervision and evaluation rely on LlamaGuard models. The method may overfit guard model preferences instead of reducing real harmful behavior.

**Weakness 2**: The only measure of usefulness reported is 5-shot MMLU accuracy, which is a weak proxy for real-world utility in safety-critical contexts. Safety mechanisms may increase refusal rates or interfere with instruction following, but the authors do not report metrics that quantify harmful refusal or degraded responsiveness. As a result, the safety–utility trade-off remains unclear.

**Weakness 3**: Safety alignment requires high stability and low risk. Training only for a single epoch on a relatively small dataset does not provide evidence that the learned safety behavior is converged or robust. The reported improvements may not reflect a stable solution but rather transient behavior due to insufficient training.

### Questions
Q1: Could the authors provide justification for limiting training to a single epoch? Since safety alignment requires reliable and stable behavior, additional evidence such as performance curves across training steps would help verify that the reported safety gains reflect a converged solution rather than transient behavior.

Q2: Given the stated motivation of supporting efficient deployment, could the authors clarify why no Qwen3-series models were included? 

Q3: Could the authors provide explicit numerical results for compute and memory efficiency in tabular format? The current figures give helpful intuition, but concrete numbers would allow clearer comparisons across models and methods, especially for deployment-oriented readers.

Q4: The discussion in Appendix Section C acknowledges the vulnerability of TV-DiSP to adversarial/jailbreak attacks but remains largely qualitative. Could the authors provide empirical results against recent prompt-based or embedding-level attack methods specifically targeting soft prompts? Even a small-scale evaluation would help quantify how much robustness is preserved compared to the dual-model system.

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
This paper proposes TV-DiSP, a safety distillation framework that replaces a dual-model safe LLM system (base model + guard model) with a single LLM augmented using learned soft prompts. The method aims to approximate the behavior of the safe system by minimizing a total variation distance (TVD) objective between the two output distributions. The distilled model produces refusal responses for unsafe prompts without requiring the guard model at inference, reducing both memory and compute costs. Experiments demonstrate improved Safety Guard Score (SGS) with significantly reduced inference overhead.

### Strengths
- The proposed framework is conceptually simple and offers improvements in computational efficiency compared to dual-model systems while maintaining competitive safety performance with regards to Safety Guard Score (SGS) only.
- Although TVD has been previously used in previous distillation works (e.g., [1]), this work presents empirical evidence that optimizing TV distance yields a better safety-utilization balance than KL divergence and REINFORCE on the benchmarks.

[1] Wen et al., f-Divergence Minimization for Sequence-Level Knowledge Distillation, ACL 2023

### Weaknesses
- The rationale for TVD is explained only briefly (lines 153-154) and remains general. Theorem 3.1 concerns generic distributional closeness guarantees, but the paper does not sufficiently demonstrate why TVD is inherently more appropriate for safety alignment than other divergences. If there is no clear reason, it seems just applying different distance metric in distillation (also covered by previous work [1]) instead of introducing the new method for safety tuning.
- Insufficient evaluation of model usefulness. Usefulness degradation is a critical part of safety alignment, yet the main safety-cost plots (Fig 2, 3, 4, 6) omit any utility measure. Only Fig 5 includes MMLU results, and it clearly reveals substantial performance drops compared to both the Base LLM and the Safe LLM system. There is also no deeper discussion of over-refusal and precision/recall or F1 over safety measures.
- Distillation versus supervised training is not justified. If safety labels are available (either human or synthetic), a supervised fine-tuning baseline would be natural comparison. It should be explained why distillation of the guard model’s imperfect predictions is preferable and what types of errors the proposed approach is expected to inherit.
- Comparison to alternative cost-reduction strategies is missing. If resource usage is the primary motivation, another reasonable baseline would be to distill large guard into a small guard and maintain a two-stage system [2]. This work does not discuss why shifting safety into soft prompts is preferable beyond efficiency, nor whether dual-model safety is fundamentally more robust.
- Notation clarity. Section 3 uses a single symbol $p$ for both LLM generation and guard scoring, without clarifying model parameters, which complicates understanding.

[1] Wen et al., f-Divergence Minimization for Sequence-Level Knowledge Distillation, ACL 2023

[2] Lee et al., HarmAug: Effective Data Augmentation for Knowledge Distillation of Safety Guard Models, ICLR 2025

### Questions
- In Sec. 3, do the base LLM and guard model share any parameters? If not, please distinguish them with separate parameter sets (e.g., $\theta$ for the LLM and $\phi$ for the guard).
- How is the distribution $p(r|x)$ represented in practice?
- Have you evaluated the model under metrics that separately measure precision vs recall of safe refusals?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents an approach to distil an LLM-guard as additional prompts in the prompt embedding space to boost the LLM safety. Their approach works under the paradigm of a dual-model system, where after every generation, an LLM guard assesses the model's output to be harmful given the prompt and overrides the response with a fixed refusal response if it is harmful. Such systems can often be non-scalable due to prohibitive inference cost. Thus, they propose training additional input parameters in the prompt input embedding space to directly train the model to generate the refusal response based on the probability of the harm from the guard model. Experimental results show a tradeoff between improved safety of the generated responses and reduced compute, as assessed by the LLM-guard scores of the generations.

### Strengths
- The paper is mostly well-written with minimal typos and is easy to follow. 
- It is well-motivated to address the inference cost of such classifier-based guardrails.
- Experimental results show that the inference computation remains similar to the original model while enhancing its safety.
- Both in and out-of distribution datasets are compared for comprehensiveness.
- Use of the guard model enables on-policy training and is shown by the use of PPO but is not fully highlighted.

### Weaknesses
- Performance gains are very sensitive to the guard model used, as Table 2 shows minimal gains as compared to Figure 1. Other metrics should be included that leverage the ground-truth labels of these. 
- Performance should be split between harm-inducing and safe prompts separately to see if it is not overrefusing. 
- Figure 5 shows little usefulness of the tuned model, which means the training is overrefusing and not able to clearly separate safe and harmful prompts.
- Since a simple safety finetuning can also have the same gains in reducing the inference cost and increasing the safety, the motivation behind the architectural design is not clear. Low performance of LoRA is not clear and should be further investigated in Figure 4. 
- It is not clear why figures 4 and 5 show a trend with respect to the learning rate since only the best performance should be compared with the two trade-off metrics instead. 
- Experimented models are limited in size (upto 3B models, quantized to 4 bit), leaving open questions about generalization to larger or unquantized models. 
- Figure 6 should be extended to Toxigen and other LLMs. 
- More discussion is needed regarding the use of many samples but only a single epoch for training. 
- KL is shown to outperform the TV-DiSP method but it is not clear why the authors have still decided to go with the TV-DiSP loss. 
- While it is highlighted that number of prompts can be tuned to hit a balance between the inference compute and safety, it should be clearly presented in the main figure by showing TV-DiSP with different number of prompts. 
- In addition to the Llamaguard > 0.5 score, the exact average llamaguard scores and LLM judge based rejection evaluations of the generations should also be included, following existing works in the literature. 
- Examples of generated responses are not included. 
- It is not clear why existing benchmarks (JailbreakBench, StrongReject) are not used (with their jailbreaks). Furthermore, the details of the self-curated benchmark, DetectJailbreak, are not provided. 
  - Chao, Patrick, et al. "Jailbreakbench: An open robustness benchmark for jailbreaking large language models." Advances in Neural Information Processing Systems 37 (2024): 55005-55029.
  - Souly, Alexandra, et al. "A strongreject for empty jailbreaks." Advances in Neural Information Processing Systems 37 (2024): 125416-125440.
- Missing related works: The architecture design is similar to Zheng et al., 2024 but is omitted from the discussion. Circuit breakers can also be seen as a fast fine-tuning strategy. 
  - Zheng, Chujie, et al. "Prompt-driven llm safeguarding via directed representation optimization." ICML 2024. 
  - Zou, Andy, et al. "Improving alignment and robustness with circuit breakers." Advances in Neural Information Processing Systems 37 (2024): 83345-83373.
- Minor:
  - Update the colors in Figure 1 right for better understanding. 
  - Line 362: across all

### Questions
- What is the effect of changing the loss function to have binary labels of safe and unsafe as opposed to the probability? Safe can be detected as the guard model's probability or ground-truth in the dataset. 
- see above weaknesses

### Soundness
2

### Presentation
2

### Contribution
2
