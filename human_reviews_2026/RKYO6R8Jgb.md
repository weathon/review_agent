# Thinking-Free Policy Initialization Makes Distilled Reasoning Models More Effective and Efficient Reasoners

- Decision: Accept (Poster)
- Scores: 6, 2, 8, 8

## Abstract
Reinforcement Learning with Verifiable Reward (RLVR) effectively solves complex tasks but demands extremely long context lengths during training, leading to substantial computational costs. While multi-stage training can partially mitigate this, starting with overly short contexts often causes irreversible performance degradation, ultimately failing to reduce overall training compute significantly. In this paper, we introduce **T**hinking-**F**ree **P**olicy **I**nitialization (**TFPI**), a simple yet effective adaptation to RLVR that bridges long Chain-of-Thought (CoT) distillation and standard RLVR. TFPI employs a simple *ThinkFree* operation, explicitly discarding the thinking content via a direct *&lt;/think&gt;* append, to reduce token usage during inference. Training with *ThinkFree*-adapted inputs improves performance and lowers token consumption, even in the original slow-thinking mode. Extensive experiments across various benchmarks have shown that {\method} accelerates RL convergence, achieves a higher performance ceiling, and yields more token-efficient reasoning models without specialized rewards or complex training designs. With TFPI only, we train a 4B model to reach 89.0% accuracy on AIME24 and 65.5% on LiveCodeBench using less than 4K H20 hours.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Thinking-Free Policy Initialization (TFPI), a lightweight pre-training stage for reasoning LLMs before RL with verifiable rewards. TFPI removes explicit <think> content, training models to output only final solutions under short contexts. Experiments on DS-1.5B, DS-7B, and Qwen3-4B show that TFPI improves reasoning accuracy while reducing compute. When followed by normal RL, TFPI further enhances performance and speeds up convergence.

### Strengths
- Efficiency: TFPI significantly reduces rollout tokens and compute cost while maintaining or improving reasoning accuracy.
- Generality: The method integrates seamlessly with existing RLVR frameworks (e.g., GRPO, DAPO) and scales consistently across 1.5B–7B models.
- Insightfulness: Analysis reveals that TFPI fosters implicit verification-style reasoning and provides better initialization for subsequent long-context RL.

### Weaknesses
- The paper adopts a multi-stage TFPI schedule (e.g., 2K→4K→8K→16K) but does not justify how these stage boundaries are chosen. It is unclear whether they are based on empirical tuning, stability constraints, or model-size scaling rules.
- The paper only evaluates TFPI on models up to 7 B parameters. It remains unclear whether the approach scales effectively to larger reasoning models (e.g., 32 B or 14B)

### Questions
- what happens if we remove the think segment entirely at inference after training—does accuracy remain similar?
- How are the stage lengths (2K→4K→8K) determined in practice? Are they tuned for stability or fixed heuristically?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces Thinking-Free Policy Initialization (TFPI), a training method for large reasoning models designed to improve performance and efficiency while reducing computational costs. The core idea here is to use a "Thinking-Free" operation during a multistage Reinforcement Learning process. This is achieved by modifying the input template to explicitly omit the model's chain-of-thought tokens (`</think>` append). The training follows a curriculum (CL) where the context length is progressively increased. The authors conduct experiments on various model sizes and benchmarks, demonstrating that TFPI can achieve strong reasoning performance with significantly less training compute compared to direct long context (includes CoT tokens) RL training.

### Strengths
- The paper tackles the significant and practical challenge of the high computational costs associated with training large reasoning models (LRMs) using long-context Reinforcement Learning with Verifiable Reward (RLVR).
- The authors provide an extensive evaluation across multiple model sizes (1.5B, 4B, 7B) and a diverse set of reasoning benchmarks, including mathematics (AIME), multi-task reasoning (GPQA), code generation (LiveCodeBench), and instruction following (IFEval). The inclusion of both in-domain and out-of-domain tasks is noteworthy.
- The results presented are strong and demonstrate high efficiency. For example, the paper reports that a 4B model trained with TFPI achieves 89.0% accuracy on AIME24 using less than 4,000 H20 GPU hours, which is good.
- The paper includes both behavioral (verification analysis) and parameter-level (PCA) analyses to provide insight into why TFPI is effective. This adds value to the empirical results and helps to understand the underlying mechanics of the proposed method.

### Weaknesses
The primary weaknesses of this paper relate to the experimental design and the isolation of the core contribution, which makes it difficult to definitively attribute the observed gains to the proposed method.

- The most significant weakness is the absence of a crucial baseline. The paper compares TFPI (multi-stage RL with thinking-free templates) against "Direct RL" (single-stage RL with a long context). However, it fails to compare against a model trained with the **same multi-stage schedule ($2K \rightarrow 4K \rightarrow 8K$) but using the standard "thinking" template**. Without this ablation, it is impossible to disentangle the performance gains and determine whether they stem from the proposed "Thinking-Free" template modification or simply from the known benefits of a multi-stage, curriculum-based training schedule. The improvements could be an artifact of the training curriculum rather than the novelty of TFPI. Clearly, ablations studies are needed.

- The paper acknowledges that multi-stage RLVR (starting with short contexts and gradually increasing them) is a "common mitigation strategy". The core technical contribution thus appears to be the addition of a simple template modification to this existing paradigm. While combining existing ideas can be effective, the lack of the aforementioned ablation makes it difficult to assess the true novelty and impact of the "Thinking-Free" operation itself.

- The paper claims that TFPI exhibits generalizability across domains, even when trained exclusively on mathematics. However, the results show inconsistencies that weaken this claim. For example:

    - In Table 1, the performance of the DS-7B model on GPQA drops from 49.0% after Stage 1 to 46.8% after Stage 3.
    - For the DS-1.5B model, performance on LiveCodeBench fluctuates significantly across stages (18.4% $\rightarrow$ 20.5% $\rightarrow$ 19.9%).
        These fluctuations suggest that the out-of-domain improvements are not consistently monotonic and may be sensitive to the training stage.

- The primary comparison is between TFPI's multi-stage strategy and a single-stage "Direct RL" approach. Even with matched total compute, these are fundamentally different training strategies, making a direct comparison challenging to interpret. Furthermore, evaluation settings are inconsistent, with different max sequence lengths (32K vs 48K) and different decoding parameters for "thinking" vs. "thinking-free" modes, which complicates a fair assessment.

### Questions
1. The paper's central claim hinges on the efficiency of the 'Thinking-Free' template. Could the authors conduct the critical ablation experiment: train a model using the exact same multi-stage curriculum ($2K \rightarrow 4K \rightarrow 8K$) but with the standard 'thinking' template enabled throughout? Please provide a direct comparison of both the final performance and the computational resources (e.g., GPU hours and memory usage) consumed against the TFPI-trained models. This is essential to determine if the gains come from the template modification or simply the curriculum.

2. How do the authors explain the performance fluctuations and, in some cases, degradations observed in out-of-domain tasks (e.g., GPQA for DS-7B) in the later stages of TFPI? 
3. Could the authors elaborate on the rationale for comparing the multi-stage TFPI against a single-stage "Direct RL" baseline, rather than a multi-stage "thinking" baseline, which would serve as a more direct control for the curriculum aspect of the training?
4. Your evaluation uses inconsistent decoding parameters (such as lower `top-p` for thinking-free) and max sequence lengths (32K vs. 48K) across different models and modes. This introduces confounding variables. How can you demonstrate that the observed efficiency gains are a direct result of the TFPI training, and not an artifact of these different evaluation settings?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes a fast, low-cost initialization phase for long-CoT RL called Thinking-Free Policy Initialization (TFPI). By eliminating the thinking process during generation, it significantly reduces the training cost of Long CoT RL. The implementation is also relatively straightforward: by simply replacing the Rollout in the existing DAPO method with inference that omits the thinking process, it simultaneously lowers RL computational costs and improves performance.

### Strengths
1）By replacing the rollout in the existing DAPO method with thinking-free inference, this work achieves simultaneous reduction of RL computational cost and improvement in performance, which is of great significance to RL training.
2）The experimental results are comprehensive, demonstrating that TFPI not only lowers computational cost but also enhances slow-thinking reasoning ability, with validations conducted across domains and settings. In addition, the experimental section discusses, from both behavioral and parameter perspectives, why TFPI is able to improve slow-thinking reasoning performance.

### Weaknesses
1）The description of the proposed method in the main text appears rather brief, making it difficult to grasp all the method details clearly.
2）In the experimental section, no details are provided for the choice of hyper-parameters such as temperature, topp, topk, or maximum output length.

### Questions
Why are the hyper-parameter settings inconsistent across comparative experiments? For instance, the hyper-parameters of thinking Mode and thinking-free mode in Table 4 are different.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes Thinking-Free Policy Initialization (TFPI), a short, low-cost RLVR stage that trains an SFT-distilled long-CoT model using `thinking-free' prompts (explicitly closing a <think> section with </think>). The hypothesis is that (i) thinking-free rollouts dramatically cut sequence length and compute; (ii) training the policy on this setting nonetheless improves slow-thinking performance when evaluated with normal CoT; and (iii) TFPI provides a stronger starting point for subsequent long-context RLVR, raising final performance and further improving token efficiency. Experiments across across math datasets, GPQA, code (LiveCodeBench) and instruction following (IFEval) support these claims.

### Strengths
1. The paper proposes a simple and practical method that shows promising results both in terms of training efficiency and performance.
2. Under similar compute budgets, TFPI improves slow-thinking performance while yielding shorter outputs in thinking-free inference.
3. The paper shows several useful ablations and analyses, and compares to several baselines.

### Weaknesses
**I do not find any major weaknesses in the paper.** Nonetheless some aspects of the paper could be improved:


1. Table 5 in Appendix shows that the full RL method is trained only for 20 steps, in order to match the compute with the baselines. However, it isn't clear whether the same would hold if the compute for all methods increased? Therefore it remains unclear, whether TFPI only works in low train compute settings, or even if the goal is to squeeze last bit of performance from RL training?
2. The paper uses Polaris-53K dataset, however given they had an extensive study on optimal temperature for RL showcasing the optimal temperature to be 1.4 for Qwen3-4B, why did authors choose to use 1.0? This might also mean, that performance of direct RL could be further improved by just changing this hyperparameter (though it might benefit TFPI as well).
3. PCA Analysis: Distances in low-dimensional projections can be misleading in high-dimensional weight spaces. The cosine similarity plot (alignment with the direct RL direction) is suggestive, but adding baselines (e.g., alignment to a separate training run, or to the early direct-RL update) would make the analysis more robust. Because if both TFPI and direct RL share the same reward and data, some alignment is expected.


Some Writing Suggestions:

1. The matched compute is using wall-clock, but that alone doesn't give full picture, since full utilization of gpu may not be happeining. So providing other variables such as total processed tokens, rl steps (already provided in appendix), etc would further add value.
2. `Generalizes across domains even when trained solely on mathematics.` Given modest but inconsistent gains (without provided statistical significance) outside math, I would call it `some transfer` rather than `generalizes across domains.`.
3. Please bold the numbers in all relevant tables (such as Table 1, 2)
4. I think parts of the paper could be simplified, equations for PPO and GRPO might be unecessary, or could be moved to appendix.

### Questions
1. Stage schedule sensitivity: Only (2K->4K->8K) and (4K->8K->16K) are tested. Do different schedules (e.g., more stages, skipping a stage, starting at 8K) matter? Is TFPI effective if started at 8K or 16K (eg, for hard datasets even thinkfree could lead to more than 8K tokens with Qwen3 models)?

### Soundness
3

### Presentation
3

### Contribution
3
