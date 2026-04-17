# QeRL: Beyond Efficiency - Quantization-enhanced Reinforcement Learning for LLMs

- Decision: Accept (Poster)
- Scores: 6, 4, 8, 6

## Abstract
We propose QeRL, a Quantization-enhanced Reinforcement Learning framework for large language models (LLMs). While RL is essential for LLMs' reasoning capabilities, it is resource-intensive, requiring substantial GPU memory and long rollout duration. QeRL addresses these issues by combining NVFP4 quantization with Low-Rank Adaptation (LoRA), accelerating rollout phase of RL while reducing memory overhead. Beyond efficiency, our findings show that quantization noise increases policy entropy, enhancing exploration in LoRA-based RL, and enabling the discovery of better strategies during RL. To further optimize exploration, QeRL introduces an Adaptive Quantization Noise (AQN) mechanism, which dynamically adjusts noise throughout training. Experiments demonstrate that QeRL delivers over 1.5× speedup in the rollout phase compared to QLoRA, and around 1.3× speedup compared to BF16 LoRA in 7B model. Moreover, this is the first framework to enable RL training of a 32B LLM on a single H100 80GB GPU, while delivering overall speedups for RL training. It also achieves faster reward growth and higher final accuracy than 16-bit LoRA and QLoRA, while matching the performance of full-parameter fine-tuning on mathematical benchmarks such as GSM8K (90.8%) and MATH 500 (77.4%) in the 7B model. These results establish QeRL as an efficient and effective framework for RL training in LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces QeRL, a quantization-enhanced reinforcement learning framework for large language models (LLMs). QeRL integrates NVFP4 quantization with Low-Rank Adaptation (LoRA) to address the resource-intensive nature of RL training for LLMs, specifically targeting substantial GPU memory and long rollout durations.

One of the key findings is that quantization noise, when controlled precisely, can actually benefit RL by increasing policy entropy and enhancing exploration, leading to the discovery of better strategies. To further optimize this, QeRL incorporates an Adaptive Quantization Noise (AQN) mechanism that dynamically adjusts noise during training.

### Strengths
- Improved Efficiency and Resource Utilization: QeRL directly tackles the substantial resource demands of RL for LLMs by combining NVFP4 quantization with LoRA. 

- Enhanced Exploration through Quantization Noise: Contrary to conventional understanding that quantization noise degrades training, this paper demonstrates that controlled quantization noise can actively benefit RL. By increasing policy entropy, it fosters better exploration and the discovery of more robust strategies, especially with the adaptive quantization noise (AQN) mechanism.

- Superior Performance and Robustness: QeRL not only accelerates training but also achieves faster reward growth and higher final accuracy compared to existing methods like 16-bit LoRA and QLoRA.

### Weaknesses
- Though effective in specific tasks, whether LoRA based RL post training can achieve comparable performance in a wide range of tasks, is still under discussion in both academia and industry.  I am wondering whether QeRL can also be workable in full-parameter fine tuning?

- The ablation of AQN, i.e., Figure 8 indicates that the curves with and without AQN are close to each other, will the authors give more experiments to demonstrate the effectiveness of AQN?

### Questions
please refer to the weakness

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
4

### Summary
This paper proposes QeRL, a quantization-aware RL framework for LLM reasoning. The method aims to improve RL efficiency by combining NVFP4 quantization with LoRA and an auxiliary “Adaptive Quantization Noise (AQN)” mechanism to encourage exploration. Experiments report faster rollouts, reduced memory usage, and higher exploration entropy compared to QLoRA-based baselines. The topic is relevant, and the goal of improving RLVR efficiency is valuable for the community.

However, several aspects of the work require clarification, especially around the efficiency claims, the validity and generality of AQN, and the justification for the hypothesis that quantization inherently encourages exploration. These issues seem addressable, but resolving them is important for the credibility of the contribution.

### Strengths
The paper addresses an important challenge in RL for reasoning models, where rollout latency and memory are major bottlenecks. The attempt to unify quantization with RL training and exploration is timely. The empirical results show promising improvements under quantized LoRA settings, and the paper is generally easy to follow, with well-organized experiments and clear motivation for reducing RL compute.

### Weaknesses
While the paper targets an important problem and presents promising results under quantized LoRA settings, several aspects of the work require further clarification to validate the core claims. In particular, the empirical support for the efficiency gains, the generality of AQN, and the justification for “quantization encourages exploration” are not yet fully convincing. The experimental design could be more comprehensive to isolate the effect of quantization from LoRA, and additional baselines would strengthen the contribution.

### Questions
1. **Baseline consistency in efficiency claims**
    The paper reports 1.5–2× speedups and significant memory savings, but these are measured against different baselines (speed vs QLoRA/NF4, memory vs BF16 Full). For example, in the **Conclusion (Line 485)**, the paper states: *“QeRL achieves around a 1.5× speedup in end-to-end RL training while drastically reducing memory usage.”* This phrasing may overstate the benefit of QeRL because it mixes different baselines into a single claim.
   To ensure clarity and fairness, efficiency results should be reported against **BF16 Full** and **BF16 LoRA**, and each speedup claim in the abstract, Section 4.3, and the conclusion should explicitly name the corresponding baseline rather than using an unqualified number.

2. **AQN ablation to validate its general effectiveness**
   AQN is only evaluated under the NVFP4 + LoRA setting. To demonstrate that AQN is a generally useful exploration mechanism rather than a component that compensates for limitations specific to quantized LoRA, its evaluation should also include **Full-parameter RL** and cover **multiple quantization formats**. This would more convincingly establish the generality and independent contribution of AQN.

3. **Justification for “Quantization Encourages Exploration”**
    The evidence currently shows quantized+LoRA > BF16+LoRA (Section 3.2), which may reflect LoRA limitations rather than quantization benefits. Could you isolate the effect of quantization by comparing **Quantized Full RL vs BF16 Full RL**, and potentially provide analysis on *why* quantization increases sampling entropy?

4. **Reporting time cost in main results and clarifying baselines for efficiency claims**
    Table 1 and Table 2 currently report only mathematical performance metrics, without the corresponding time cost. Since efficiency is a core claim of the paper, these tables should also include **Rollout Time** or the **End-to-End Training Time** for each configuration, so that performance and efficiency can be evaluated together rather than in isolation.

   In addition, the paper includes efficiency statements that do not explicitly state the baseline, which may lead to misinterpretation. For example:

   - **Section Abstract (Line 20–21):** “over 1.5× speedup in the rollout phase”
   - **Section Conclusion (Line 484–485):** “around a 1.5× speedup in end-to-end RL training while drastically reducing memory usage”

   Both statements implicitly compare against **QLoRA**, but this is not mentioned. These speedup claims should explicitly state the baseline to avoid mixing different reference points and to ensure transparent and fair interpretation.

5. **End-to-end speed vs rollout-only speed**
   Since RLVR training involves rollout, reward, logprob, update actor, and additional overheads, rollout speed alone does not capture end-to-end efficiency. Reporting full training wall-clock time would provide a more complete picture of the practical gains.

6. **Clarifying the ~2× speedup for 14B–32B models**
   For the larger models, it would be helpful to clarify whether the baseline was affected by KV cache exhaustion. If the 14B/32B BF16 baseline was already near the memory boundary or encountering KV cache overflow, the slowdown may be caused by resource bottlenecks rather than reflecting the intrinsic advantage of QeRL. In such cases, restricting the comparison to a single GPU may not represent a realistic or fair setting, since practitioners would not normally train 32B-scale RL models on a single GPU. A comparison under a configuration where the baseline does not suffer from KV cache exhaustion would provide a more accurate and meaningful assessment of the true speedup.

7. **Typos / Minor Issues**

   L235–236: quotation formatting should be corrected (''flatter'' → ``flatter''). The same issue also appears on **L238–239** for ''optimal''.

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
This paper introduces QeRL, a novel framework that integrates NVFP4 quantization with LoRA to enhance the efficiency and effectiveness of RL for LLMs. The work is motivated by the significant computational bottlenecks—high memory usage and slow rollout speeds—inherent in RL training for LLMs. While existing approaches like QLoRA address memory concerns, they often introduce slowdowns and are not designed to leverage the unique dynamics of RL. QeRL's primary innovation lies in its dual approach: it achieves computational efficiency while also demonstrating, for the first time, that the quantization error (or "noise") can be a powerful, implicit mechanism for enhancing exploration in RL.

To overcome the limitation of static quantization noise, the paper further proposes an Adaptive Quantization Noise (AQN) mechanism. AQN dynamically injects and schedules noise, integrated in a zero-overhead manner into LayerNorm layers, to balance exploration and exploitation throughout training.

The experimental validation is extensive and convincing. The authors demonstrate that QeRL achieves over 1.5x speedup in the rollout phase and reduces memory usage by 50-60% compared to 16-bit LoRA, enabling the training of a 32B parameter model on a single H100 GPU. Crucially, QeRL does not sacrifice performance for efficiency; it consistently matches or surpasses the performance of 16-bit LoRA and often reaches the performance ceiling of full-parameter fine-tuning on challenging mathematical reasoning benchmarks like GSM8K and MATH.

### Strengths
1. Novel and Counter-Intuitive Core Insight: The most significant strength of this paper is its conceptual leap: reframing quantization error from a detrimental artifact (as it is often viewed in SFT) into a beneficial exploratory force in RL. The careful analysis in Figures 3 and 5, showing the entropy increase and its correlation with improved reward growth, provides solid evidence for this claim.

2. Holistic and Practical Framework (QeRL + AQN): The paper builds a complete, end-to-end framework. The combination of high-performance NVFP4 quantization with LoRA is a pragmatic solution to the memory and speed problem. The introduction of the Adaptive Quantization Noise (AQN) mechanism is a clever innovation that addresses the inherent limitation of static noise. Significantly, the noise vector can be elegantly merged into LayerNorm to avoid extra computational cost. This demonstrates a deep understanding of both algorithmic needs and system-level constraints.

3. Extensive and Compelling Empirical Evaluation: The authors conduct a thorough ablation study across multiple model sizes, two  RL algorithms (GRPO and DAPO), and several challenging mathematical benchmarks (GSM8K, BigMath). On the efficiency front, QeRL achieves substantial memory reduction, shrinking model sizes to 25-30% of their 16-bit counterparts, while also providing a 1.2x to 1.5x end-to-end training speedup and a greater than 2x speedup in the rollout phase for larger models. Crucially, these efficiency gains do not come at the cost of performance; QeRL consistently achieves superior or comparable final accuracy and faster reward convergence compared to 16-bit LoRA and QLoRA, often matching the high-performance benchmark set by full-parameter fine-tuning.

### Weaknesses
1. Limited Task and Domain Generalization: The empirical validation, while extensive, is heavily focused on mathematical reasoning tasks (GSM8K, MATH, AIME, AMC). While these are excellent proxies for complex, multi-step reasoning, it remains an open question whether the benefits of QeRL generalize equally well to other domains such as coding.

2. Dependency on Hardware and Kernels: The significant speedups of QeRL are contingent on the efficient NVFP4 format and the supporting Marlin kernel, which are tied to recent NVIDIA architectures (Hopper, Blackwell). This might limit immediate adoption for researchers without access to the latest hardware. A brief discussion on the performance or feasibility on more widely available hardware (e.g., Ampere) would be helpful for context.

### Questions
1. The exponential scheduler for AQN was chosen after comparing several options. Could you elaborate on the sensitivity of the performance to the key hyperparameters of AQN, such as $\sigma_{start}$ and $\sigma_{end}$? Was the choice of 10 evenly spaced intervals for noise adjustment based on empirical tuning, or is there a heuristic for setting this based on the total number of training steps?

2. While the paper provides a strong empirical and intuitive explanation for how quantization noise increases entropy and aids exploration, could you elaborate on the theoretical connection between the specific properties of NVFP4 quantization noise (e.g., its distribution and magnitude) and established RL exploration theory? A deeper formal analysis could further solidify the foundational contribution of this work.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces QeRL, a framework that combines NVFP4 quantization with Low-Rank Adaptation (LoRA) for efficient training of large language models in reinforcement learning. The key insight is that quantization noise can enhance exploration during RL (unlike in supervised fine-tuning, where it's detrimental), leading to both improved efficiency and better performance. The authors propose an Adaptive Quantization Noise (AQN) mechanism that dynamically adjusts noise throughout training. Experiments on mathematical reasoning benchmarks (GSM8K, MATH500, AIME, AMC) demonstrate a 1.5× speedup in the rollout phase, the ability to train 32B models on a single H100 GPU, and performance matching or exceeding that of full-parameter fine-tuning on 7B models.

### Strengths
Practical impact: Enables RL training of large models (up to 32 B) on a single GPU with significant speedups.
 Exploration insight: Demonstrates that adaptive quantization noise can raise policy entropy and aid reward learning. 
Strong empirical results: Matches or exceeds 16-bit LoRA / QLoRA on math reasoning tasks. 
Good ablation coverage: Studies noise schedules and LoRA ranks; efficient integration using Marlin kernels.

### Weaknesses
Positioning and comparison to recent quantized RL and post-training quantization work: While the paper surveys relevant literature, it does not cite or compare with several recent approaches critical to advances in quantized RL or LLM quantization. For instance, Quantized Reinforcement Learning (QuaRL) [Krishnan et al., 2020, 2022] and newer post-training quantization frameworks (RPTQ, VPTQ, LRQuant, PREFIXQUANT, QuIP$). This makes it challenging to position the core empirical claims and efficiency gains in relation to the current state-of-the-art.

Limited statistical rigor: No multi-seed averages or error bars; robustness to randomness is unknown.

Scope restricted to math reasoning: No results for code, dialogue, or safety-critical RL tasks.

Missing FlashRL baseline: Discussed in related work but not compared directly.

Hyperparameter fairness: Learning-rate differences may partly explain improved stability; further tuning of BF16 LoRA could close the gap.

Hardware specificity: To the best of my knowledge, NVFP4 / Marlin optimizations are NVIDIA-centric; portability to other accelerators is unclear.

Potential overstatement of claims: In the introduction and conclusion, there are hints of broad generalization (e.g., “our technique can be seamlessly adapted to richer…datasets”, p. 21) without empirical demonstration. While acknowledging the focus on math reasoning, a slightly more tempered framing would match the current evidence base.

### Questions
Could you provide variance/error bars for multiple runs of GSM8K/MATH500?

Can you include a direct comparison with FlashRL to quantify gains beyond precision-mismatch correction?

How sensitive is AQN to the start/end noise levels and decay schedule?

Did you try stabilizing BF16 LoRA with smaller ranks or gradient clipping so it can use 1e-5 LR?

Please release a concise 32B single-GPU recipe (context length, microbatching, KV cache type, LoRA rank, and optimizer).

Any early results on code or safety-oriented tasks to test AQN’s generalization? Can the authors provide more quantitative discussion about the failure modes of AQN or when exploration becomes detrimental, e.g., in very large models or with more complex tasks?

### Soundness
3

### Presentation
3

### Contribution
4
