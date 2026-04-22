# LoRA is All You Need for Safety Alignment of Reasoning LLMs

- Avg Score: 3.33
- Decision: Reject
- Scores: 4, 4, 2

## Abstract
Reasoning LLMs have demonstrated remarkable breakthroughs in solving complex problems that were previously out of reach.  To ensure LLMs do not assist with harmful requests, safety alignment fine-tuning is necessary in the post-training phase. However, safety alignment fine-tuning has recently been shown to significantly degrade reasoning abilities, a phenomenon known as the ``Safety Tax”. In this work, we show that using LoRA for SFT on refusal datasets effectively aligns the model for safety without harming its reasoning capabilities. This is because restricting the safety weight updates to a low-rank space minimizes the interference with the reasoning weights. Our extensive experiments across four benchmarks covering math, science, and coding show that this approach produces highly safe LLMs—with safety levels comparable to full-model fine-tuning—without compromising their reasoning abilities. Our ablation studies further identify three key factors in LoRA: (1) rank-$1$ updates are sufficient to achieve the best reasoning and safety performance, (2) the up projection layers are the most critical modules, with LoRA applied to them alone achieving even better results, and (3) middle layers are more effective than early or late layers. Together, these findings show that strong safety and reasoning can be achieved at minimal computational cost when updates are applied in the right places. Additionally, we observe that LoRA induces weight updates with smaller overlap with the initial weights compared to full-model fine-tuning. Finally, while our attempts to further reduce this overlap yield only modest improvements on some tasks, they highlight the potential of developing methods that more reliably optimize the reasoning–safety tradeoff.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates whether the Safety Tax can be mitigated using only LoRA fine-tuning. Through extensive ablations on rank, module, and layer configurations, the authors show that low-rank LoRA updates can preserve reasoning performance while achieving strong safety alignment. They further analyze the geometric structure of LoRA updates, suggesting that their low alignment with reasoning weights explains the reduced interference and improved reasoning–safety balance.

### Strengths
- Demonstrates that LoRA-only fine-tuning can mitigate the Safety Tax while preserving reasoning ability — an interesting and practical finding.
- Well-designed ablations reveal key factors (rank = 1, MLP up-projection, middle layers) that contribute most to the reasoning–safety trade-off.
- Provides a geometric perspective on why LoRA interferes less with reasoning, through alignment and subspace analyses.

### Weaknesses
- The paper makes a strong claim that “LoRA is all you need” to address the safety–reasoning trade-off. While the presented results are intriguing, the current experimental scope is insufficient to substantiate this claim. A wider range of backbones and model sizes should be evaluated to demonstrate consistency across architectures and scales (R1-1.5 ~ R1-32B, s1, Qwen3, etc).

- Moreover, prior analyses (Jain et al., 2024; Wei et al., 2024) about low-rank safety directions generalize to general LLMs, not just reasoning-centric LRMs. Therefore, the authors’ argument should extend beyond LRMs to general LLMs. Since the Safety Tax is not unique to reasoning models but also arises in other capabilities, the proposed hypothesis should be validated across diverse tasks beyond reasoning.

- In Figure 7, the alignment difference between full fine-tuning and LoRA appears marginal, which weakens the causal claim.

- The trade-off plots contain multiple points per configuration, yet their meanings are under-explained (e.g., epoch checkpoints, random seeds, or hyperparameter variations). The large variance across these points also raises concerns about training stability and reproducibility.

- Finally, the evaluation would be more convincing if it included adaptive or adversarial safety tests—such as obfuscation, paraphrased jailbreaks, or multi-turn exploit prompts—that better reflect real-world attack surfaces and robustness challenges.

[1] Jain, Samyak, et al. "What makes and breaks safety fine-tuning? a mechanistic study." Advances in Neural Information Processing Systems 37 (2024): 93406-93478.

[2] Wei, Boyi, et al. "Assessing the brittleness of safety alignment via pruning and low-rank modifications." arXiv preprint arXiv:2402.05162 (2024).

### Questions
See Weaknesses

### Soundness
2

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
This  paper explores how Lora performs on safety alignment task and find that it can effective improve safety score without droping reasoning ability.

### Strengths
(1) This empirical study is careful and fairly comprehensive. Results on multiple benchmarks are reported. 

(2) The findings that Lora can successfully avoid the trade-off between reasoning ability and model safety is interesting and useful.

(3) This paper is well-structured and easy to follow.

### Weaknesses
(1) The benchmarked models lack diversity — all reasoning models are derived from DeepSeek. It remains unclear whether the findings generalize to other reasoning models or architectures, such as GPT-OSS-20B or GPT-OSS-120B.

(2) This paper lacks theoretical analysis explaining why the LoRA technique can mitigate the “safety tax” issue. A deeper investigation or theoretical justification would strengthen the claims.

(3) The safety evaluation pipeline may have limitations. Safety is automatically assessed using Llama-Guard-3-8B, which could introduce bias. Incorporating multi-metric safety evaluations (e.g., jailbreak or red-teaming tests) would provide more comprehensive and convincing evidence.

### Questions
(1) Can the main findings generalize to other reasoning models such as GPT-OSS-20B?

(2) In Figure 3(b), why does r = 64 yield the lowest model safety performance? Is this a consistent phenomenon observed across different models? The paper explains that the “middle-rank” setting might be harder to optimize, but is there any empirical or theoretical evidence supporting this explanation?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper targets "Safety Tax", which is preserving the model's reasoning capacity while aligning for safety. The paper utilizes LoRA on a refusal dataset to effectively align the model for safety while preserving the reasoning capacity. The authors show that using r=1 rank achieves the best reasoning-safety tradeoff.

### Strengths
- The paper shows simple yet effective usage of LoRA in finetuning for safety alignment. 
- The results show that the LoRA-trained model is both safe and has high reasoning performance in different benchmarks.

### Weaknesses
- The paper does not exhibit weight-level selectivity; instead, it adopts a more coarse-grained perspective, assuming that all parameters contribute collectively to the model’s reasoning capability. The selectivity applied is primarily at the layer or module level.

- The paper lacks of theoretical ground for its claims and has an experimental approach.

- The proposed post-hoc method doesn't improve the reasoning performance, but the authors also claim that the method needs more development.

- To me, the paper needs a more professional tone. The narration should support the claims with references or experimental data, which are given in lines 184-187.

- There is only one dataset used for the safety evaluation.

- The paper focuses on only one family of models.

### Questions
- Are the findings true for the different architectures?
- Why is only one family of models selected?
- Can authors compare their approach with other fine-tuning safety-alignment methods?

### Soundness
2

### Presentation
3

### Contribution
2
