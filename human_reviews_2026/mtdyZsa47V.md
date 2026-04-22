# Ultra-Fast Language Generation via Discrete Diffusion Divergence Instruct

- Avg Score: 6.67
- Decision: Accept (Poster)
- Scores: 6, 6, 8

## Abstract
Fast and high-quality language generation is the holy grail that people pursue in the age of AI. In this work, we introduce **Di**screte **Di**ffusion Divergence **Instruct** (**DiDi-Instruct**), a training-based method that initializes from a pre-trained diffusion large language model (dLLM) and distills a few-step student for fast generation. The model distilled with DiDi-Instruct matches or surpasses its dLLM teacher and the GPT-2 baseline while providing up to **64$\times$ acceleration**. The theoretical foundation of DiDi-Instruct is a novel framework based on integral KL-divergence minimization, which leads to a practical training algorithm. We further introduce *grouped reward normalization, intermediate-state matching, and the reward-guided ancestral sampler* to improve *training stability, model coverage, and inference quality*. On the OpenWebText benchmark, DiDi-Instruct achieves perplexity ranging from 62.2 (8 NFEs) to 18.4 (128 NFEs), outperforming prior accelerated dLLMs and the GPT-2 baseline. These gains incur a negligible entropy loss (around $1$%) and reduce additional training wall-clock time by **more than $20\times$** compared to competing dLLM distillation methods. We further validate the robustness and effectiveness of DiDi-Instruct through extensive ablation studies, model scaling, downstream task evaluations, and unconditional protein sequence generation. In conclusion, DiDi-Instruct enables efficient and effective distillation for language generation in the blink of an eye.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes Discrete Diffusion Divergence Instruct (DiDi-Instruct), a distillation framework that trains a few-step discrete diffusion language model (student) to match a pre-trained masked diffusion language model (teacher) by minimizing an integral KL divergence (IKL) across noise levels. The core gradient is rewritten as a score-function estimator weighted by a log density-ratio reward between student and teacher; the reward is estimated via an auxiliary discriminator. Training is stabilized by grouped reward normalization and a score decomposition that exposes the student to intermediate corruption states. At inference, a Reward-Guided Ancestral Sampler (RGAS) uses discriminator signals for gradient tilting in early steps and re-ranking in later steps. On OpenWebText, the model achieves perplexity 62.2- 18.4 at 8 - 128 NFEs, with small entropy loss and lower distillation cost, and it scales to larger models and a protein-sequence task.

### Strengths
1. The paper adapts IKL-based distribution matching to masked discrete diffusion and derives a policy-gradient form that bypasses non-differentiable paths, with a discriminator estimating the log density ratio used as reward. This method is different from time-matching distillations and provides a route to few-step discrete generation.
2. The derivation gives the IKL objective and yields a score-function identity in which the reward is the marginal log-density ratio between student and teacher; the proof sketch and assumptions are stated, and the link to a tractable discriminator estimator is explicit.
3. On OpenWebText the method reports consistently lower PPL across 8–128 steps, with faster distillation and small entropy loss. The paper includes cumulative and leave-one-out ablations indicating gains from the proposed components, and shows transfer to protein sequence generation.
4. The paper is clearly written and well structured.

### Weaknesses
1. The comparison to the baselines are is too abbreviated with only one figure of perplexity. It would be better to have a table presenting more metrics besides the ppl.
2. DiMO is discussed in the related work in appendix but is not include as an experiment baseline.

### Questions
1. Can you report tokens/sec (or sequences/sec) and latency per 1k tokens at matched perplexity versus a similarly sized AR model with and without speculative decoding, on the same hardware and kv-cache settings? This would clarify the practical speed–quality trade-off beyond NFEs.
2. Can you include downstream tasks? A single perplexity metric is not enough to establish practical advantage of the new method.

### Soundness
3

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
4

### Summary
This paper introduces a training-based distillation framework DiDi-Instruct to accelerate generation for discrete diffusion language models (dLLMs). The core idea of DiDi-Instruct is to distill a pre-trained teacher dLLM into a student model that can generate high-quality text in very few steps. The method is centered on minimizing the Integral KL (IKL) divergence between the teacher and student models, which forces the student to match the teacher's data distribution. The author proposes comprehensive solutions to make the training on discrete data tractable, concluding: 1) inspired by policy gradient, decomposing the gradient of the objective as a score function weighted by a log-density ratio to avoid non-differentiable issue. 2) using an auxiliary discriminator to approximate the log-density ratio because it is intractable to compute directly. 3) leveraging grouped reward normalization to stabilize reward. The training is end-to-end and jointly for discriminator and student model. During inference, the author uses the trained discriminator to guide the sampling process and improve output quality. Experiments show the effectiveness of the proposed method that DiDi-Instruct significantly outperforms existing accelerated models and the GPT-2 baseline in perplexity, and the distilled model surpasses the 1024-step teacher's quality with only 16 steps.

### Strengths
1. The paper presents a mathematically rigorous solution to the difficult problem of distillation in a discrete state space. Instead of relying on heuristics, it adapts the IKL divergence objective from continuous models and  reformulates it as a tractable policy gradient problem.
2. The method consistently outperforms other distillation methods and the GPT-2 baseline in perplexity across all sampling budgets.
3. The proposed method demonstrates high efficiency. The training only needs one GPU, compared with 8 GPU for training teacher model. For generation, it is able to surpass the text quality of its 1024-step teacher model with only 16 inference steps, representing a significant acceleration.
4. The proposed method avoids the mode collapse problem, which is a common problem in distillation. Experimental results on text generation demonstrate negligible entropy loss.

### Weaknesses
1. The author performs distillation on a self-trained teacher model. This raises questions about the method's transferability and effectiveness when applied to other pre-trained, open-source dLLMs, such as LLaDA [1] or Dream [2].
2. In the text generation experiments, the evaluation is limited to perplexity and entropy. These metrics alone cannot accurately reflect the true text generation capabilities. The authors should conduct a more comprehensive evaluation on standard text generation benchmarks (e.g., MMLU, GSM8K, Humaneval) to demonstrate that the student model can indeed achieve performance comparable to the teacher model while using fewer steps.
3. In the protein sequence experiments, the pLDDT score increases significantly after distillation. However, excessively high pLDDT scores might indicate a potential risk of mode collapse (e.g., collapsing to a few sequences with the highest pLDDT, thereby failing to sample other valid sequences), which will lead to a reduction in generation diversity. The authors need to clarify the generation results from the perspective of protein sequence diversity (e.g., by using MMseqs clustering) to address this concern.

[1] Dream 7B: Diffusion Large Language Models

[2] Large Language Diffusion Models

### Questions
See above weaknesses.

### Soundness
2

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
3

### Summary
This paper introduces Discrete Diffusion Divergence Instruct (DiDi-Instruct), a training-based distillation method that initializes from a pre-trained dLLM to create a fast, few-step student generator. The framework is founded on integral KL-divergence minimization, which is reformulated from a policy gradient perspective to derive a tractable update rule for the discrete state space of language.

### Strengths
- The paper proposes a principled training method by reformulating the IKL distillation objective from a policy gradient perspective, which provides a tractable update rule for discrete spaces.   
- DiDi-Instruct achieves new state-of-the-art results on the OpenWebText benchmark, consistently delivering lower PPL across 8-128 NFEs with over 20x faster distillation time than competing methods.

### Weaknesses
- The proposed Reward-Guided Ancestral Sampler (RGAS) introduces new hyperparameters ($h$, $M$), but there is no sensitivity analysis.
- It is unclear how the baseline results like DUO or SDTT are obtained.

### Questions
- Could you clarify the experimental setup for the SDTT and DUO baselines?   
- Could you provide a sensitivity analysis for the RGAS hyperparameters ($h$, $M$) and elaborate on the stability of the adversarial training?

### Soundness
4

### Presentation
3

### Contribution
4
