# Efficient Reasoning via Thought Compression for Language-Guided Segmentation

- Decision: Reject
- Scores: 2, 6, 4, 4

## Abstract
Chain-of-thought (CoT) reasoning has significantly improved the performance of large multimodal models in language-guided segmentation, yet its prohibitive computational cost, stemming from generating verbose rationales, limits real-world applicability. We introduce WISE (Wisdom from Internal Self-Exploration), a novel paradigm for efficient reasoning guided by the principle of \textit{thinking twice---once for learning, once for speed}. WISE trains a model to generate a structured sequence: a concise rationale, the final answer, and then a detailed explanation. By placing the concise rationale first, our method leverages autoregressive conditioning to enforce that the concise rationale acts as a sufficient summary for generating the detailed explanation. This structure is reinforced by a self-distillation objective that jointly rewards semantic fidelity and conciseness, compelling the model to internalize its detailed reasoning into a compact form. At inference, the detailed explanation is omitted. To address the resulting conditional distribution shift, our inference strategy, WISE-S, employs a simple prompting technique that injects a brevity-focused instruction into the user's query. This final adjustment facilitates the robust activation of the learned concise policy, unlocking the full benefits of our framework. Extensive experiments show that WISE-S achieves state-of-the-art zero-shot performance on the ReasonSeg benchmark with 58.3 cIoU, while reducing the average reasoning length by over \textbf{5$\times$}---from 112 to just 23 tokens.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
WISE trains a model to generate a structured sequence: a concise rationale, the final answer, and then a detailed explanation. By placing the concise rationale first, the method leverages autoregressive conditioning to enforce that the concise rationale acts as a sufficient summary for generating the detailed explanation. This structure is reinforced by a self-distillation objective that jointly rewards semantic fidelity and conciseness, compelling the model to internalize its detailed reasoning into a compact form. At inference, the detailed explanation is omitted.

### Strengths
The unique training structure is reinforced by a self-distillation objective that explicitly rewards the semantic fidelity between the concise rationale and the detailed explanation, while penalizing the verbosity of the former. This process encourages the model to internalize its elaborate reasoning capabilities into a compact, efficient policy.

To ensure this learned policy is robustly activated at inference—where the detailed explanation is entirely omitted to maximize speed—the WISE framework culminates in WISE-S, a simple, zero-overhead prompting strategy.  This final adjustment injects a brevity-focused instruction into the user’s query, mitigating the conditional distribution shift between training and inference and ensuring the model consistently defaults to its more efficient reasoning mode.

### Weaknesses
The novelty of this work may be limited. The method mainly uses GRPO framework. It generally rewards the model to generate a short concise reasoning step with reinforcement learning. This idea has been explored in various works such as [R1,R2]. During the training, it includes both long and short thinking tokens, which is similar to [R2] to train the model on paired longform and short-form responses for each query, ensuring it can generate both styles. During inference, to save tokens, it simply use promopts to ask the model not output the long thinking tokens. The technical contribution may be limited.

In experiments, it only experiments with 7B VLM. It is better to experiment with more models from different families with different sizes to demonstrate the general performance. 

In experiments, compared with the original Seg-Zero without any efficiency, it can save reasoning tokens with significant efficiency improvements. However, compared with other baseline methods optimized for brevity via reward shaping such as L1-Exact and L1-Max. The improvements seem to be marginal. For example, from Table 3, L1-Max needs 11 tokens to achieve 78.9 cIoU, while the proposed method achieves 79.1 with 24 tokens for RefCOCOtestA. It seems that the baselines are also effective and the proposed method leads to marginal improvements. 

In table 2, the reported Seg-Zero-7B results from the original paper is actually wrong. From the original paper, these results are not 7B, but  3B. It is not fair to compare the 3B results with 7B. And I am not sure whether the re-evaluated Seg-Zero resutls are from 3B or 7B models. 


It is built on Qwen2.5-VL-7B and SAM2. It is better to report their original results without finetuning.

 
[R1] Walk Before You Run! Concise LLM Reasoning via Reinforcement Learning

[R2] Thinkless: LLM Learns When to Think

### Questions
see the weakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces WISE (Wisdom from Internal Self-Exploration), a framework for efficient reasoning segmentation in large multimodal models (LMMs). The core innovation lies in compressing verbose Chain-of-Thought (CoT) reasoning into concise rationales while maintaining performance. Key contributions include:  

- A structured generation sequence (concise rationale → answer → detailed explanation) that enforces reasoning compression via autoregressive conditioning.  
- A self-distillation objective that rewards semantic fidelity between concise and detailed rationales while penalizing verbosity.  
- WISE-S, an inference-time prompting strategy that reduces reasoning length by 5× (from 112 to 23 tokens) while achieving state-of-the-art zero-shot performance on ReasonSeg (58.3 cIoU).

### Strengths
**Originality**: The idea of *self*-distillation (training a single model to compress its own reasoning) is novel in the CoT segmentation domain. Unlike prior work (e.g., Seg-Zero’s RL-based reasoning), WISE explicitly decouples training-time reasoning depth from inference-time efficiency through its structured action space.  
**Clarity**: The methodology is well-structured, particularly the hierarchical reward formulation (Eq. 3-6) and the distinction between WISE/WISE-S inference modes.  
**Significance**: Addresses a critical barrier (computational cost of CoT) for real-time applications like robotics. The efficiency-performance trade-off breakthrough (Table 3) is practically impactful.

### Weaknesses
**Limited Generalization Evidence**: While ReasonSeg and RefCOCOg are used, there is no validation on other reasoning-heavy benchmarks (e.g., VCR or GQA). The claim of "out-of-domain reasoning" in the abstract lacks supporting experiments beyond ReasonSeg.  
**Shallow Analysis of Prompt Engineering**: The WISE-S prompting strategy (e.g., “one sentence”) is under-explored. The paper does not test alternative brevity prompts or quantify their sensitivity, raising concerns about robustness.  
**Incomplete Cost-Benefit Analysis**: While token counts are reduced, actual latency/energy savings are not measured. For real-world deployment, hardware-level efficiency metrics (e.g., FPS on edge devices) would strengthen the claims.

### Questions
**Q1**: How does WISE scale with model size? The experiments use a 7B model—would the compression mechanism remain effective for smaller (1B) or larger (70B) models?  

**Q2**: The self-distillation reward relies on a pretrained SentenceTransformer for semantic similarity. Could this introduce bias? Was the similarity model domain-adapted for segmentation tasks?  

**Q3**: The brevity prompt in WISE-S (“one sentence”) appears heuristic. Have the authors explored learned prompt tuning or gradient-based optimization for brevity?  

**Q4**: The computational cost-saving of WISE is unclear. How does it compare to Seg-Zero in inference time beyond token counts?  

**Rebuttal Potential**: Addressing Q1/Q3 could demonstrate broader applicability, while resolving Q2/Q4 would strengthen the technical rigor. A deeper exploration of alternative efficiency metrics (Q4) might elevate the significance from incremental to transformative.

### Soundness
3

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
5

### Summary
This work trains the model to generate a structured, three-part response: a concise rationale ($\tau_c$), the final answer (A), and a detailed explanation ($\tau_d$). It achieves Thought Compression for the Language-Guided Segmentation task by designing a distillation reward during RL training to enforce that $\tau_c$ acts as a sufficient summary for generating $\tau_d$. Furthermore, a brevity-oriented instruction is incorporated when inference to further compress reasoning.

### Strengths
1. The core idea is sound: achieving thought compression by leveraging the $(\tau_c, a, \tau_d)$ structure and designing a distillation reward in RL training to compel $\tau_c$ to be a sufficient statistic for $\tau_d$'s generation.

2. Good performance: achieving good performance in benchmarks

### Weaknesses
1. WISE was trained based on Qwen2.5-VL-7B, but Qwen2.5-VL-7B was not used as a baseline in subsequent experiments, which seems insufficiently rigorous and cannot rule out the possibility that the fundamental performance of the Qwen2.5-VL-7B model is already strong enough.

2. While the concept of "thought compression" is central to the paper, the terminology could benefit from further clarification. For example, the distinction between "concise rationale" and "detailed explanation" might not be immediately clear, especially within  the field of tasks chosen by the author, it is not as clear as math reasoning tasks. A more explicit definition or example earlier in the paper would help.

3.  Task suitability: The necessity of thought compression for the Language-Guided Segmentation task is debatable. Compression on a $\sim$100-token baseline, regardless of whether a long CoT is needed, makes little sense. The method seems task-agnostic; it should be attempted on tasks requiring much longer CoT (e.g., several thousand tokens), like mathematical reasoning.

4. Unsubstantiated explanation for WISE-S: The explanation that omitting $\tau_d$ generation at inference time causes a distribution shift is insufficient. The relevance of the brevity-focused prompt (injected by the WISE-S strategy) to the distribution shift is also doubtful.  Supplementary experiments are required to verify the unique effect of the short prompt on the WISE model, e.g., testing if Seg-Zero-7B + brevity-focused prompt shows a similar phenomenon.

5. Limited model scales and series: The experiments only use Qwen2.5-VL-7B; more model sizes and series should be included.

### Questions
1. How is the omission of $\tau_d$ generation realized? Is it only via prompt or by decoding intervention?

2. In Table 6, why do the results for WISE-F ($\text{+$\tau_c$ + $\tau_d$}$) and WISE ($\text{+$\tau_c$ − $\tau_d$}$) differ?

3. Could the author provide several cases that include $\tau_d$ to verify whether $\tau_c$ acts as a sufficient summary, and to demonstrate the impact of the proposed method on the quality of the generated $\tau_d$?

4. In Table 2, the performance of WISE-7B-S is generally better than WISE-7B, while in Table 1 it cannot surpass WISE-7B. Have the authors considered why such results occur? Is it related to using a subset of RefCOCO as the training set?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes WISE,  a training based CoT approach build on top of Seg-Zero that condenses the long reasoning to speed up inference. This is achieved by generating a "concise rationale" before the detailed explanation as form of self distillation. During inference (WISE-S) the explanation is discarded for efficiency.
The key contributions of the paper are:
- WISE method for reasoning segmentation that doesn't need verbose reasoning at inference.
- Conditional self distillation approach using the rationale - answer - explanation.
- Strong experimental results and reduction of output tokens while retaining performance.

### Strengths
- 4.9x - 7.0x reduction in token overhead vs Seg-Zero with minimal performance loss.
- Interesting idea to perform self distillation using the auto-regressive nature of the model and the summary before the detailed explanation.
- Good experimental results with performance being consistently good across both reasoning and referring segmentation tasks.
- Paper ablations are very good and clearly establish the motivation for the method and hyperparameters.

### Weaknesses
- The model is the heavy reliance on the Seg-Zero paper, including method, scope and experiments which reduce the contribution over a general method.
- Implementation is only on the Qwen2.5 model, showing this method works across models and training methods would be a big benefit. (see questions).
- The self-distillation reward is only applied when IoU > 0.5 which means the compression learns only from successful paths. This could indicate a better performance on "easier" tasks and less of an improvement on harder datasets.

### Questions
- To distinguish the paper, the authors should show performance on other model architectures (beyond Qwen) or pipelines.
- The paper should include a direct measurement of inference speedup, not just token reduction, to gauge the practical impact more accurately.
- More qualitative demos to showcase the method under very different scenarios.
- An in depth comparison to identify which specific examples the model improves over Seg-Zero and what are the failure cases would be helpful to understand the scope and usefulness of the approach.
- How does this method compare to naive token length reduction?

### Soundness
4

### Presentation
3

### Contribution
2
