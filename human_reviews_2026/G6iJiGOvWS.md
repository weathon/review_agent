# Fine-Tuning Masked Diffusion for Provable Self-Correction

- Decision: Reject
- Scores: 6, 4, 4, 6

## Abstract
A natural desideratum for generative models is \emph{self-correction}--detecting and revising low-quality tokens at inference. While Masked Diffusion Models (MDMs) have emerged as a promising approach for generative modeling in discrete spaces, their capacity for self-correction remains poorly understood. Prior attempts to incorporate self-correction into MDMs either require overhauling MDM architectures/training or rely on imprecise proxies for token quality, limiting their applicability. Motivated by this, we introduce PRISM--**P**lug-in **R**emasking for **I**nference-time **S**elf-correction of **M**asked Diffusions--a lightweight, model-agnostic approach that applies to any pretrained MDM. Theoretically, PRISM defines a self-correction loss that provably learns per-token quality scores, without RL or a verifier. These quality scores are computed in the same forward pass with MDM and used to detect low-quality tokens. Empirically, PRISM advances MDM inference across domains and scales: Sudoku; unconditional text (170M); and code with LLaDA (8B).

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces PRISM, a lightweight fine-tuning framework that enables any pretrained masked diffusion model to perform self-correction during inference. The framework involves a self-correction loss that provably learns per-token quality scores without requiring reinforcement learning or external verifiers. PRISM attaches a lightweight adapter to pretrained MDMs and uses these quality scores to detect and remask low-quality tokens during generation. PRISM is shown to be effective on several benchmarks, such as Sudoku puzzles, unconditional text generation, and code generation with LLaDA.

### Strengths
- The paper provides a rigorous theoretical framework. The marginalization perspective connecting MDM training to the PRISM objective is well-motivated.

- The model-agnostic design requires only a lightweight adapter while preserving the original MDM's capabilities, which also enhances applicability. The shared backbone architecture that computes both unmasking posteriors and per-token quality in a single forward pass ensures computational efficiency.

- The experiments span diverse domains and scales, which illustrates the capability of various tasks.

### Weaknesses
- As is acknowledged by the authors, the reliance on per-position posterior marginals inherently limits its ability to detect global errors or complex reasoning mistakes. The issue can be particularly problematic for tasks requiring multi-step reasoning or long-range dependencies, where local token quality may be high while global coherence is poor.

- While statistically significant, the performance gains are often modest. For example, on MBPP with LLaDA-8B, PRISM achieves only 2.7% improvement over baselines at 256 steps, and the gains diminish at higher sampling steps. The improvements on text generation, while consistent, are similarly incremental. Given the additional fine-tuning overhead, the cost-benefit ratio may be questionable.
The paper also lacks an analysis of the failure cases. 

- The ablation studies reveal that large values of $k$ lead to performance degradation due to train-test distribution mismatch, which should be further explained.
Some experimental choices could be better justified, such as the specific baseline comparisons and the choice of evaluation metrics for different domains.

### Questions
Please see the weakness section

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
The paper addresses the problem of self-correction in discrete generative models. In particular, the authors proposed PRISM, a method to detect and fix low-quality tokens during inference. The technique can be applied to any pretrained MDM. The work is claimed to introduce a theoretically grounded self-correction loss that learns per-token quality scores directly within a single forward pass. Empirical results show consistent improvements in MDM inference across diverse tasks, including Sudoku solving, unconditional text generation (170M models), and code generation with LLaDA (8B).

### Strengths
1. The paper is generally well-written and easy to follow. 
2. The discrete diffusion is a promising topic, and the introduction of new self-correction will be helpful to improve the sample quality. 
3. The empirical study shows that the proposed method could significantly improve the performance of discrete diffusion models without introducing additional sampling costs.

### Weaknesses
1. I am not very sure why $g_{\phi^*}$ is independent of the selection of the pretrained model.  In particular, I wonder if the authors could explain in more detail mathematically why the ground truth remains unchanged when $\mathbf{y}$, the sample from the model, is involved in the definition. 

2.  In addition, if $g_{\phi^*}$ is indeed independent of the pretrained model, why we still need the forward pass of the original model? Can we obtain $\mathbf{y}$ in a model-free way? In addition, if what the authors claimed is true, it would be very interesting to see if the same correctors can be applied to different pretrained models. 

I am willing to adjust the score if my concerns are well addressed.

### Questions
Please refer to the weaknesses.

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
This paper proposes a lightweight fine-tuning framework that equips Masked Diffusion Models (MDMs) with inference-time self-correction. During fine-tuning, the model learns to predict per-token quality scores via a dedicated self-correction loss. In inference, tokens with low predicted quality are re-masked and re-sampled, allowing the model to fix its own mistakes iteratively. Experiments across multiple model scales demonstrate consistent improvements over the corresponding backbones.

### Strengths
- **Clarity and readability**. The paper is clearly written and easy to follow. The overall method is simple, modular, and implementation-friendly.

- **Principled quality score.** The authors give a precise definition of the token-level “quality” signal and motivate the loss design with a clean theoretical argument, which makes the learning target well-specified rather than heuristic.

- **Practicality and compatibility.** The approach leaves the base MDM architecture intact and is compatible with parameter-efficient adapters (e.g., LoRA), making it feasible to deploy on larger models without prohibitive costs.

### Weaknesses
- **Limited gains on large models.** Improvements become modest for the largest models, and in some cases training-free self-correction baselines can match or exceed the proposed method. The paper would benefit from a careful analysis of when/why fine-tuning helps beyond training-free strategies.

- **Figure 4 is hard to parse.** The semantics of the entries and, in particular, the black dashed lines are unclear from the text. The caption and main body should explicitly explain what each element represents and how to read the plot.

- **Inconsistent experimental setup descriptions.** Some details (e.g., masking schedule, selection policy for low-quality tokens, sequence lengths, decoding settings) are scattered or vary across sections. A consolidated, uniform description would improve reproducibility.

- **Quality vs. ambiguity.** Low quality can reflect either genuine model errors or inherent ambiguity (e.g., in “I eat an ___”, many nouns are plausible). In such cases, valid tokens might still receive low scores; the paper should discuss how the method distinguishes true mistakes from legitimate alternatives.

- **Positioning vs. prior self-correction work.** Given existing self-correction/verification methods, the incremental contribution can feel narrow unless the empirical and conceptual differences are emphasized with stronger head-to-head comparisons.

### Questions
- **Sequence length in LLaDA experiments.** On LLaDA with MBPP, zero-shot pass rates are already competitive at relatively short contexts. Could you share a bit more about why a 4096-token length was chosen? It would also be helpful to see results at shorter lengths (e.g., 128 and 256 tokens) to better understand performance in more practical settings.
- **Comparison to training-free quality estimation.** Some prior works [1,2] indicate that token-level “uncertainty/quality” signals can be obtained without fine-tuning. Would it be possible to include controlled, apples-to-apples comparisons (using the same backbone and inference budget) against such training-free approaches?
- **About error tokens.** I’m curious whether “error” tokens in your experiments are indeed assigned lower quality scores. Among low-quality tokens, some might simply reflect rare but valid choices, while others correspond to genuine errors. Do you have any analyses (e.g., error-type breakdowns, verifier-based labeling, or oracle comparisons) that demonstrate the model distinguishes these cases effectively?
- **Mask selection details during fine-tuning.** If the masking ratio for latent is too high, the model might produce uncertain predictions, potentially affecting training stability. Could you describe how the masking schedule and ratio were chosen, such as the range, any curriculum or annealing strategies, and how sensitive the results are to these settings?
- **Calibration of predicted quality.** It would be informative to compare the learned per-token quality scores with empirically measured correctness probabilities, for instance, by explicitly masking a token and evaluating its posterior probability.

---
# Reference
[1] Wide-In, Narrow-Out: Revokable Decoding for Efficient and Effective DLLMs

[2] Adaptive Classifier-Free Guidance via Dynamic Low-Confidence Masking

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
This paper introduces PRISM (Plug-in Remasking for Inference-time Self-correction), a lightweight fine-tuning framework that equips any pretrained Masked Diffusion Model (MDM) with the ability to self-correct at inference. Standard MDMs cannot revise tokens once unmasked, while prior self-correction attempts rely on ad-hoc proxies or architectural changes. PRISM defines a novel self-correction loss that provably learns per-token quality scores corresponding to the true posterior probability of a token being correct. These scores are computed in the same forward pass as the unmasking posterior, enabling efficient detection and remasking of low-quality tokens. Experiments on Sudoku, text, and code generation show consistent improvements over strong remasking baselines with minimal fine-tuning cost.

### Strengths
1. Data and Parameter Efficient: The proposed approach can improve the performance of MDM on MBPP by 3% with only 250M parameters and 0.1M paired training data, in comparison to state-of-the-art method ReMDM. 
2. Clarity and Thoroughness: The paper is well-written and structured. It provides the necessary background on MDMs, clearly formulates the problem of self-correction, and then introduces PRISM in a logical progression. 
3. Theoretical guaranteed improvement: Proposition 1 in the paper shows that the PRISM loss is minimized when the model’s learned score for a token equals the posterior probability that the token is correct (matching the ground truth) under the current context.

### Weaknesses
1. Reducing diversity: The fine-tuning loss funciton (quation 3) can reduce the diversity of generated samples because the loss enforce $\mathbf{z}$ map to single correct $\mathbf{x}$, which degrades the diversity of the generated text and ignore the multiple correct $\mathbf{x}$ sequence. Take figure 2 as an example, the training loss will force $\mathbf{z}$ (I [MASK] [MASK] water [MASK]) map to single correct sequence $\mathbf{x}$ (I have a water bottle) and reduce the likelihood of other sequence ([I drink some water.] or [I buy a water bottle]).
2. The correctness of unmasking posterior for more complex grammar correctness is unknown: Even if we have multiple correct sequence $(\mathbf{x}, i)$ corresponds to single masked sequence $\mathbf{z}$, crafting a fine-tuning dataset $(\mathbf{z}, (\mathbf{x}, i))$. The paper haven't shown any emprical or theorectical results if we train a MDM with multiple correct sequence. For example, a single $\mathbf{z}$ (I [MASK] [MASK] water [MASK]) map to 2 correct sequence $\mathbf{x}$ (I have a water bottle) and (I have two water bottles).
2. Lack of empirical result between scaling the training dataset and the diversity reduction: For above mentioned drawbacks, the paper doesn't conduct empirical evaluation on it, it should demonstrate how diversity of the generated text are affected by the fine-tuning dataset grows.

### Questions
1. Add an experiment on scaling the fine-tuning dataset and diversity reduction: As mentioned in the weakness, I suggest the authers to add an additional experiment in the main paper to discuss how diversity of the generated text are affected by the fine-tuning dataset grows.
2. Add more reasoning benchmarks: I would also suggest adding more advanced benchmark like GSM8K, MMLU, HumanEval, and TruthfulQA to evaluate the effect of the self-correctness and the diversity reduction.

### Soundness
3

### Presentation
3

### Contribution
2
