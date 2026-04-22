# Entropy-Aware Speculative Decoding Toward Improved LLM Reasoning

- Avg Score: 2.50
- Decision: Reject
- Scores: 2, 2, 2, 4

## Abstract
Speculative decoding (SD) accelerates large language model (LLM) reasoning by using a small draft model to generate candidate tokens, which the target LLM either accepts directly or regenerates upon rejection. However, excessive alignment between the draft and target models constrains SD to the performance of the target LLM. To address this limitation, we propose Entropy-Aware Speculative Decoding (EASD), a training-free enhancement. Building on standard SD, EASD incorporates a dynamic entropy-based penalty. At each decoding step, we employ the entropy of the sampling distribution to quantify model uncertainty. When both models exhibit high entropy with substantial overlap among their top-N predictions, the corresponding token is rejected and re-sampled by the target LLM. This penalty prevents low-confidence errors from propagating. By incorporating draft-model verification, EASD enables the possibility of surpassing the target model's inherent performance. Experiments across multiple reasoning benchmarks demonstrate that EASD consistently outperforms existing SD methods and, in most cases, surpasses the target LLM itself. We further prove that the efficiency of EASD is comparable to that of SD. The code can be found in the Supplementary Materials.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes Entropy-Aware Speculative Decoding, a training-free method designed to enhance standard speculative decoding by introducing an entropy-based penalty mechanism. At each decoding step, the method measures both the draft and target model entropies to decide whether to accept or reject a token. The aim is to prevent low-confidence tokens from propagating and potentially improve accuracy. Experimental results on multiple reasoning benchmarks indicate that EASD achieves higher accuracy than standard SD and even outperforms the target model alone in some cases.

### Strengths
Simple and training-free: The proposed method does not require additional training, which increases its potential applicability in practice.

### Weaknesses
1. Lack of actual speed measurement: Despite being a speculative decoding paper, there are no end-to-end latency or throughput experiments. The “Computation Analysis” section only reports average generated token counts as a proxy for computational cost, which is insufficient. Speculative decoding methods are typically evaluated by real speedups (e.g., wall-clock latency or tokens/s) on standard hardware configurations. This omission significantly undermines the main claim regarding efficiency.

2. Weak performance on the accuracy results: The vanilla SD methods are lossless, which means that the gap between SD and Single Model in Table 1 should be considered as the experiment error. The gap between EASD and Single Model are not significantly higher than this experimental error, which weakens the significance of the experiments in Table 1.

3. Missing ablation on critical hyperparameters: The paper mentions determining the entropy threshold using the LIMO dataset but does not provide analysis of sensitivity to this choice. It is unclear how robust the method is to different thresholds or entropy percentile values.

### Questions
1. Why is the wallclock speedup not included in Computation Analysis part? 

2. Can you provide more detailed parameter sensitivity analysis?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes EASD, a training-free algorithm for speculative decoding in LLM reasoning. The two key components incorporate entropy-based uncertainty and distributional overlap (top-n) between the draft and target models to dynamically regulate token acceptance. When both models exhibit high entropy and their top-N token predictions substantially overlap, EASD penalizes the target model’s probability for the draft token and forces resampling, preventing low-confidence tokens from propagating errors. Experiments are conducted on math reasoning datasets and GPQA, showing that EASD outperforms speculative decoding baselines. Further analyses include an ablation study, computational analysis, and case studies, which help better understand the method.

### Strengths
- The idea of EASD is simple and effective: it is conceptually clear and a training-free extension to speculative decoding that leverages entropy and distributional overlap to improve reasoning quality, showing improvement in performance without much increasing complexity or computational cost.
- EASD consistently outperforms both standard and reward-guided speculative decoding (RSD) across diverse reasoning benchmarks.
- The paper is easy to read and understand, with analyses help interpret the performance and design of the method.

### Weaknesses
1. For methodology design, while leveraging entropy and top-n is a signal, the choice of the threshold is pretty heuristic. For entropy, the threshold has to be pre-defined through computation on a validation set, while for top-n, it is pre-defined by a heuristic. This reliance on fixed thresholds raises concerns about the method’s robustness and generalizability across different model pairs or domains. Since entropy distributions and token overlap patterns can vary significantly between architectures or tasks, the optimal thresholds for triggering the dynamic penalty may need to be re-tuned for each new setting.
2. I am concerned with the comprehensiveness of the main experiments and additional analyses. For example, the experiments are only done using Qwen-2.5-Series, and the experiments are mostly centered around math reasoning. To further claim general LLM reasoning, I would suggest the authors add experiments on code reasoning and more general ones, such as HumanEval and MMLU.
3. For speculative decoding, IMO, algorithms are designed for better efficiency claims. Instead of mainly claiming performance improvement, additional analyses on efficiency, in addition to Section 4.4 would be appreciated.
4. The performance improvement is not significant, while EASD will incur additional computation overhead. The authors should show strong evidence that the tradeoff between effectiveness and efficiency is worthwhile.

### Questions
1. For entropy and top-n threshold selection, how to set an appropriate value for different datasets and methods? Providing guidance would be better.
2. Could you provide a more comprehensive analysis for efficiency? E.g., a time breakdown to draft/verification, tokens per second, etc.
3. IMO, Figure 1 is not easy to understand. Removing baselines and leaving only EASD, with clearer illustrations on tokens would be better.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes Entropy-Aware Speculative Decoding (EASD), a training-free enhancement to standard Speculative Decoding (SD) designed to improve both the accuracy and reasoning quality of Large Language Models (LLMs) without sacrificing inference efficiency.

### Strengths
The paper offers a fresh and elegant extension of speculative decoding by incorporating entropy as a dynamic control signal. Its training free formulation and focus on uncertainty driven collaboration between large and small models distinguish it from prior reward or alignment based methods.

### Weaknesses
The paper attributes Reward Guided Speculative Decoding (RSD) to Li et al., 2025a, which actually refers to Reward Shifted Speculative Sampling (SSS). The correct citation should be Liao et al., 2025 (arXiv:2501.19324). This misattribution may mislead readers about the baseline implementation and the conceptual lineage of RSD. The authors should revise all mentions, tables, and references accordingly and clarify whether their RSD baseline follows Liao et al.’s procedure or the SSS variant.

While benchmarks are diverse, the evaluation is limited to Qwen-family models. It remains unclear whether EASD generalizes to other architectures (e.g., gpt-oss, mistral, llama). Including one additional model family would strengthen claims of universality. Moreover, entropy thresholds are tuned on the LIMO dataset; showing sensitivity analysis or per-task robustness would make the method more reproducible and credible.

The ablations (Table 2) are informative but stop short of explaining why certain components dominate performance. The authors could include entropy–accuracy correlation plots or visualize token rejection frequencies to give deeper insight into the mechanism’s behavior.

The claim that EASD exceeds the inherent target performance is intriguing but underexplained. It would be helpful to provide qualitative or quantitative evidence, e.g., error category breakdowns, human evaluation of reasoning correctness, to show that the improvement is not due to sampling variance.

### Questions
- The experiments use only Qwen models. Could you test EASD on a different model family  (e.g., gpt-oss, mistral, llama) to show that the entropy-based penalty is architecture-agnostic? This would substantiate the claim that the method generalizes without retraining.
- How sensitive are results to these thresholds? Could you provide a sweep or heuristic guideline for setting them when validation data are unavailable?
- Ablations show what matters but not why. Can you visualize how often the entropy penalty triggers, which tokens are rejected, and whether rejection frequency correlates with performance gains?
- The idea that EASD can outperform the base LLM is striking. Could you include statistical significance tests or qualitative error analyses showing that this improvement is consistent and not due to sampling variance?
- You state that EASD maintains SD-level efficiency, but Tables 3 suggest small deviations in token counts. Can you report relative FLOPs or wall-time speedups to confirm computational parity?
- Are entropies computed over the full vocabulary or only the top-n tokens? Clarifying this would help others replicate your setup precisely.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper focuses on efficient generation with speculative decoding (SD). The vanilla SD requires a strict distribution alignment between the draft and target LLMs, and constraints its performance to the target LLM. The authors proposes a new SD variant, entropy-aware speculative decoding (EASD), and aims to preserve SD's efficiency, while potentially outperforming the target LLM.

Compared to the vanilla SD, where a token generated by the draft LLM will be rejected if it's distribution is not aligned with the target LLM's, EASD applies two new rejection conditions for the generated token from vanilla SD: (1) The entropy from both draft and target LLM is too high; (2) The top-n predicted tokens from the draft and target tokens are not overlapped well. these two new conditions implies that the LLM itself is less confident to the generated token. After rejection,  the token will be regenerated by the target model by excluding the accepted token from vanilla SD.

Through extensive experiments, the results show that EASD outperforms SD and other baselines and has a similar efficiency as SD. With ablation studies, the design choices of EASD is justified.

### Strengths
1. The paper is well-written, with a clear motivation and contribution.
2. The proposed method is simple to apply.
3. The experiments are thorough, with clear results to show the performance and efficiency benefit from EASD.

### Weaknesses
1. The proposed method lacks of theoretical support. Both baselines, SD and RSD, are well supported by the theoretical justification. 
2. The efficiency comparison seems unfair. Only the number of generated tokens are shown. It's suggested to show how much tokens are generated by the draft and target models, separately. Compared to SD, EASD has two more conditions. I believe more tokens should be rejected and regenerated by the target model.

### Questions
1. For the two new conditions, high-entropy and top-n overlap, do you apply them to the accpeted tokens from the draft model? I.e. Do you have three conditions for token acceptance, including Equation (1, 4, 5), or only (4, 5)?


### Typos
1. L316 and Table 1: It seems the citation is wrong for RSD.

### Soundness
3

### Presentation
3

### Contribution
3
