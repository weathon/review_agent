# DiffuCoder: Understanding and Improving Masked Diffusion Models for Code Generation

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 8, 4

## Abstract
Diffusion large language models (dLLMs) are compelling alternatives to autoregressive (AR) models because their denoising models operate over the entire sequence. The global planning and iterative refinement features of dLLMs are particularly useful for code generation. However, current training and inference mechanisms for dLLMs in coding are still under-explored. To demystify the decoding behavior of dLLMs and unlock their potential for coding, we systematically investigate their denoising processes and reinforcement learning (RL) methods. We train a 7B dLLM, DiffuCoder, on 130B tokens of code. Using this model as a testbed, we analyze its decoding behavior, revealing how it differs from that of AR models: (1) dLLMs can decide how causal their generation should be without relying on semi-AR decoding, and (2) increasing the sampling temperature diversifies not only token choices but also their generation order. This diversity creates a rich search space for RL rollouts. For RL training, to reduce the variance of token log-likelihood estimates and maintain training efficiency, we propose coupled-GRPO, a novel sampling scheme that constructs complementary mask noise for completions used in training. In our experiments, coupled-GRPO significantly improves DiffuCoder's performance on code generation benchmarks (+4.4\% on EvalPlus) and reduces reliance on AR bias during decoding. Our work provides deeper insight into the machinery of dLLM generation and offers an effective, diffusion-native RL training framework.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors trained a 7B diffusion large language model (dLLM), DiffuCoder, on 130B tokens of code and analyzed its decoding behavior on the “AR-ness” of the its sampling process. The authors’ contributions are two-fold: 1. proposed a global and a local AR-ness metric for dLLMs and used the metrics to analyze the sampling process; 2. proposed coupled-GRPO, a RL technique that adapts GRPO of autoregressive LLMs to dLLMs.

### Strengths
1. The proposed AR-ness metrics are intriguing and provide insightful analysis of how and when discrete diffusion models exhibit autoregressive-like behavior. While some obervations may seem intuitive, concluding them in a formal way is a meaningful contribution.
2. The introduction of Coupled-GRPO is a simple yet effective technique for adapting GRPO to discrete diffusion models. Table 1 presents a comprehensive benchmark across various LLMs and dLLMs, showing that DiffuCoder clearly outperforms other diffusion-based counterparts, and that DiffuCoder + Coupled-GRPO further improves upon the base model. I also agree that approaches which make dLLMs overly autoregressive undermine their purpose, so it is encouraging to see these results.

### Weaknesses
The two main contributions of the paper are valuable in their own right, but I find it difficult to fully understand how they connect as part of a unified method. While the AR-ness metrics are conceptually interesting and insightful, it remains unclear how they relate to or motivate the proposed Coupled-GRPO. Specifically, how does Coupled-GRPO address or influence the behavior measured by the AR-ness metrics? There is only a brief discussion around line 468 and a few related observations in Appendix D.4, but it would be helpful if the authors could provide a clearer explanation or reference that explicitly connects how Coupled-GRPO contributes to improving non-AR behavior in diffusion models.

For Coupled-GRPO, the method pairs timesteps whose indices sum to T, the total number of diffusion steps. This design implicitly assumes a symmetric noising schedule, where the masking ratios at t and T−t are approximately complementary. It would be helpful for the authors to clarify whether this pairing choice is arbitrary or theoretically justified, and whether they have explored or evaluated alternative coupling strategies under non-symmetric schedules.

### Questions
See Weaknesses for major concerns.

Miscellaneous:
The figure legends overlap with the plots, which makes them difficult to read. Please consider reformatting the figures to improve clarity and presentation.

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
3

### Summary
This paper introduces DiffuCoder, a 7B-parameter diffusion LLM (dLLM) for code, and uses it to investigate the decoding behavior of dLLMs. The authors propose "autoregressive-ness" (AR-ness) metrics to quantify how left-to-right dLLM generation is and analyze the effect of sampling temperature on diversity. Based on these insights, they introduce Coupled-GRPO, a reinforcement learning method that uses complementary mask pairs to reduce variance in policy gradient estimation. Experiments show DiffuCoder is competitive with AR baselines and that Coupled-GRPO provides modest performance gains (+4.4% on EvalPlus) and improved decoding parallelism.

### Strengths
1- Provides the **first systematic measurement of AR-ness** in diffusion LLMs, with clear metrics and visualizations.

2- **Coupled-GRPO** is theoretically motivated (antithetic variates) and supported by formal variance-reduction proof.

3- Offers a **complete open-source training recipe** for a diffusion-based code LLM, potentially useful for the community.

4- Connects decoding order, sampling temperature, and parallelism, yielding practical insights for efficient generation.

5- Writing and figures are generally clear and reproducible; appendices are thorough.

### Weaknesses
### **Incremental Analytical Novelty** 
The key observations from the AR-ness analysis (e.g., the effect of temperature, the difference between adapted and from-scratch models) confirm properties that are largely expected from the principles of masked diffusion. The contribution is in the measurement, not the discovery of new phenomena. I would suggest that the authors tone the it down in the contribution sections.

### **Modest and Inconsistent Algorithmic Gains**
The improvements from Coupled-GRPO are relatively small and not uniform across all evaluated benchmarks (Table 2), raising questions about its overall impact.

### **Insufficient Ablations**
The paper lacks critical ablations for its core method, including:
- A sweep of the coupling parameter λ to test if the benefits of "coupled" sampling scale.
- A direct comparison against a "2x decoupled" baseline to isolate the effect of complementary sampling from simply doubling the number of forward passes.
- A clear analysis of the computational cost/overhead of Coupled-GRPO.

### **Missing Baseline**
I understand that **Dream-Coder-7B** is relatively a new baseline, but the absence of a comparison with that model gives an unjust favor to DiffuCoder over other open-sourced dLLMs. I would love to see that comparison added to the tables in the rebuttal period, as the model was released almost two months prior to the submission deadline.

### **Overstated Claims** 
Several passages (e.g., on temperature-driven diversity) could be more accurately framed as measurements of known model properties rather than novel discoveries.

### **Missing Citations**
The paper does not fully contextualize its work within the broader literature. Specifically:
- When discussing early masked diffusion formulations (e.g., around line 52), the paper should cite "Your Absorbing Discrete Diffusion Secretly Models the Conditional Distributions of Clean Data" (Ou et al., 2024)[1] as a concurrent and relevant work in this space.
- The discussion of semi-AR decoding and Block Diffusion (Arriola et al., 2025)[2] would be strengthened by also citing the concurrent work "Unifying Autoregressive and Diffusion-Based Sequence Generation" (Fathi et al., 2025)[3], which explores similar hybrid modeling ideas. Properly crediting these concurrent works is important for a complete scholarly narrative.

### **Practical Utility of AR-ness** 
Beyond interpretability, what are the actionable insights or design recommendations from the AR-ness analysis? For example, could a model's AR-ness score be used to dynamically adjust the number of diffusion steps during inference for speedup without a major performance drop?

### Minor Typographical or formatting issues
1- Page 4, Section 3: "We observed that training with 700B tokens in Stage 1 led to worse performance than using only 65B tokens" -> This is a key observation. The reasoning is plausible, but could be slightly expanded.

2-Page 5, Table 1 Footnote: "Scores are bolded when our model outperforms LLMs specialized for code (excluding LLaDA or Dream)." -> This is an unusual and potentially confusing bolding criterion. It would be clearer to bold the best score in each column or simply state the comparison in the text.

3- Page 8, Section 5: "w. full mask completion" -> Should be "w/ full mask completion" or "with full mask completion" for formal writing.

### **References**
[1]  Ou et al., "Your Absorbing Discrete Diffusion Secretly Models the Conditional Distributions of Clean Data"

[2] Arriola et al., "Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models"

[3] Fathi et al., "Unifying Autoregressive and Diffusion-Based Sequence Generation"

### Questions
Please refer to **Weaknesses**.

### Soundness
3

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
5

### Summary
This paper presents *DiffuCoder*, a large-scale diffusion language model (DLM) designed for code generation. Unlike traditional DLMs that struggle with fixed decoding order and limited self-correction capability, DiffuCoder introduces a complete four-stage training framework including pretraining, mid-tuning, instruction fine-tuning, and reinforcement learning. A major contribution lies in the proposed coupled-GRPO, a variance-reduced reinforcement learning algorithm that enables effective self-reflective sampling under high-temperature decoding. The authors also conduct detailed behavioral analysis, demonstrating DiffuCoder's flexible, non-autoregressive decoding strategy and strong performance across major code and math benchmarks.

### Strengths
* Proposes the first full pipeline for large-scale masked diffusion models in code generation, demonstrating competitive performance with AR models.
* Introduces coupled-GRPO, a principled, theoretically justified improvement over prior RL-based DLM training.
* Deep behavioral analysis offers valuable insight into how diffusion decoding diverges from autoregressive patterns.
* Experiments are comprehensive, including comparisons with prior DLMs, autoregressive baselines, and multiple ablations.
* The paper is clearly written and well-structured, making technical innovations accessible.

### Weaknesses
This work is strong across the board — in terms of methodological innovation, clarity of explanation, and thorough ablation studies. I did not find any substantive weaknesses, and I believe the paper easily meets the bar for poster acceptance at ICLR. However, given that the core method could likely generalize to domains beyond code (e.g., open-ended language modeling, multimodal reasoning), it would have been valuable to see some discussion of this potential in the main paper. Including such analysis or preliminary results could elevate the work to oral-level significance.

### Questions
1. Have you considered applying coupled-GRPO to existing diffusion backbones like LLaDA or Dream to assess its modularity and transferability?
2. Given that the proposed method enhances performance with only two diffusion blocks, have you observed any performance degradation or instability with longer generation lengths or more complex prompts?
3. Do you expect the entropy-sink decoding behavior to generalize to non-code domains, or is it specific to code and math where token dependencies are more structured?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper is a systematic study and methodological work on applying diffusion language models (dLLMs) to code generation; it introduces AR-ness metrics to quantify decoding order and a diffusion-native RL approach (coupled-GRPO), delivers consistent gains across multiple code benchmarks, and reveals the key phenomenon that temperature affects both token selection and generation order.

### Strengths
1.	Proposes local/global AR-ness@k to compare dLLMs vs. AR LLMs across stages and modalities. 
2.	Higher temperature makes decoding less AR-like and raises pass@k; real sample trajectories are visualized. 
3.	Coupled-GRPO lowers variance via complementary masks and avoids semi-AR bias, yielding stable post-training gains. 
4.	From adaptation → mid-training → SFT → RL, with compute setups that aid reproducibility. 
5.	When halving steps (2× speed), performance drops less after GRPO.

### Weaknesses
1.	Generalization to multi-language, multi-file, or agentic tasks is unclear. 
2.	Multi-stage large-scale training plus RL rollouts; dLLM GRPO takes ~2× AR’s wall time. 
3.	Even with variance reduction, the training still approximates token log-likelihoods.
4.	Stage-1 at ~700B tokens hurts downstream performance, implying sensitivity to data quality and early stopping. 
5.	No λ>1 (multi-pair coupling) or alternative t-distributions; limited analysis of reward weighting and verifiers.

### Questions
1.	Please evaluate λ>1 (multi-pair coupling) and alternative t-samplings (e.g., Beta/log-uniform), and report the variance–accuracy–compute trade-offs.
2.	Please add a compute-parity comparison against semi-autoregressive RL (block diffusion / re-masking) with matched rollouts and wall-clock time to quantify the benefit of avoiding AR bias.
3.	Please demonstrate robustness by sweeping syntax/format/test weights, adding static analysis and richer tests, and reporting mean ± σ over 3 seeds.
4.	Please report practicality: end-to-end throughput (tokens/s) and training/inference cost, and comment on whether your method stacks with existing dLLM accelerations.

### Soundness
3

### Presentation
3

### Contribution
3
