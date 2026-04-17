# From Narrow to Panoramic Vision: Attention-Guided Cold-Start Reshapes Multimodal Reasoning

- Decision: Accept (Poster)
- Scores: 4, 8, 4, 6

## Abstract
The cold-start initialization stage plays a pivotal role in training Multimodal Large Reasoning Models (MLRMs), yet its mechanisms remain insufficiently understood. To analyze this stage, we introduce the Visual Attention Score (VAS), an attention-based metric that quantifies how much a model attends to visual tokens. We find that reasoning performance is strongly correlated with VAS (r=0.9616): models with higher VAS achieve substantially stronger multimodal reasoning. Surprisingly, multimodal cold-start fails to raise VAS, leaving distributions close to the base model, whereas text-only cold-start induces a clear increase. We term this counter-intuitive phenomenon Lazy Attention Localization. To validate its causal role, we design training-free interventions that directly manipulate attention allocation at inference time, yielding consistent 1--2% gains without retraining. Building on these insights, we propose Attention-Guided Visual Anchoring and Reflection (AVAR), a comprehensive cold-start framework that integrates visual-anchored data synthesis, attention-guided objectives, and visual-anchored reward shaping. Applied to Qwen2.5-VL-7B, AVAR delivers an average gain of 7.0% across 7 multimodal reasoning benchmarks. Ablation studies further confirm that each component of AVAR contributes step-wise to the overall gains. The code, data, and models are available at https://github.com/lrlbbzl/Qwen-AVAR.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper focuses on the mechanisms and effects of multimodal large reasoning models (MLRM) during the cold-start stage, proposes the Visual Attention Score (VAS) to quantify “visual token attention,” and empirically shows a strong correlation between average VAS and multimodal reasoning performance (the paper reports Pearson r=0.9616). The authors further observe that adopting a multimodal cold start does not improve VAS, whereas a pure text cold start significantly increases VAS; this counterintuitive result is termed Lazy Attention Localization (LAL).
Based on this diagnosis, the authors first conduct inference-time (training-free) attention redistribution experiments. By amplifying attention to visual tokens and suppressing attention to “system” tokens, they achieve a consistent gain of about 1–2% across multiple benchmarks and propose the existence of a “System Token Redundancy Zone” that can be reallocated to visual attention.
The authors then propose a three-part cold-start framework, AVAR: (1) Visual-Anchored Reflection Data synthesis (VARD), which embeds explicit anchors such as “look back at the image/check the figure” in the chain of thought; (2) Attention-Guided Training Objective (AGTO), which uses \mathcal{L}{\text{enhance-img}} and \mathcal{L}{\text{suppress-sys}} to encourage attention to vision and suppress attention to system tokens, respectively; (3) Visual-Anchored Reward Shaping (VARS), which uses an attention ratio as an auxiliary reward during RL and combines it with accuracy and formatting rewards to form the total return.
Using Qwen2.5-VL-7B as the base, the authors report an average improvement of +7.0% on seven multimodal reasoning benchmarks. Ablations indicate “stepwise” contributions from the three components, and VAS increases progressively with training stages (7.5→10.1→13.8→18.9).

### Strengths
1. Treats “visual attention allocation” as the core explanatory variable for cold-start effectiveness, proposes VAS and systematically links it to reasoning performance, introduces and names the phenomenon Lazy Attention Localization, and provides a coherent diagnosis–intervention–training framework (VAS→inference-time intervention→training-time objective/reward), with a complete rationale and a clear trajectory.

2. Reports results on multiple public benchmarks (MathVista/MathVision/MathVerse-VO, MMMU/Pro, MMStar, HallusionBench), compares against various 7B-scale multimodal reasoning models, presents overall and per-benchmark improvements, and further details ablations and stage-wise VAS changes, forming a relatively solid chain of evidence.

3. Provides a formal definition of VAS (taking the ratio of attention from user query tokens to visual/system keys as the unit, then averaging across layers, heads, and tokens) and clear mathematical expressions of the losses/rewards, facilitating implementation and reproducibility.

4. If the observed LAL is widespread, attention reshaping for cold starts could become a general recipe for multimodal reasoning models, with potential value for both academic and industrial training paradigms.

### Weaknesses
1. Although a strong correlation of r=0.9616 is reported, there is a lack of stricter causal testing for VAS→performance and control of confounders (e.g., differences in visual encoders across models, differences in system prompt templates, decoding length/temperature, etc.). At present, “inference-time attention amplification” yields only a 1–2% gain, which makes it difficult to explain the entire 7% average improvement during training. Suggested additions: controlled experiments that remove/replace the visual encoder, fix the system prompt template, and test sensitivity to decoding length/temperature.

2. Using “attention to system tokens” as the denominator of VAS requires justification: the number and positional distribution of system tokens can vary widely across data/templates, potentially introducing bias. There is a lack of comparisons against alternative metrics that use user text tokens or the number of image patches as references and of consistency results. In addition, no sensitivity analyses at the layer/head/position levels are presented in the main text to support the robustness of the main conclusions (Appendix D is mentioned, but key statistics are not shown in the main body).

3. Although reproducibility is claimed, the VARD stage uses closed-source models such as Gemini 2.5-Pro to generate high-fidelity descriptions, which may make it difficult for the community to replicate the same data quality. Meanwhile, the prompts, filtering, and quality-control details of the three-model collaborative pipeline are said to be in Appendix G, but the main text does not provide enough “actionable recipe” details (e.g., sampling temperature, rejection strategies, QC thresholds). On the implementation side, only coarse-grained parameters such as 30.6K / 20 epochs are given, with no disclosure of random seeds and variance.

4. The main improvements are concentrated in visual mathematics scenarios such as MathVision/MathVerse-VO. Although MMStar/MMMU are included, there is a lack of dimensions such as real-world high-resolution VQA, video temporality, and multi-image multi-turn reasoning, making it difficult to judge the robustness and cost (inference latency/memory) of the method across a broader family of tasks.

5. There is no systematic comparison against classic training-time approaches such as attention regularization/pruning/head reweighting or visual token resampling/token-dropping. The inference-time intervention is also not compared in parallel, under equal compute budget and equal parameter changes, with existing “adaptive attention calibration/head-level intervention” methods.

6. Tables 1/2/4 mostly report point estimates without confidence intervals/significance tests, and they do not report mean±variance over multiple random seeds, making it difficult to assess the robustness of the +1–2% and +3–7% gains.

7. Suppressing attention to system tokens may weaken instruction following and safety (many safety/formatting control signals depend on system prompts), yet the paper focuses only on performance improvements and does not provide comparative evaluations or failure analyses for instruction compliance and safe refusal.

### Questions
The questions are those listed in the Weaknesses section.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper investigates the role of cold-start initialization in training Multimodal Large Reasoning Models (MLRMs). This work introduces Visual Attention Score (VAS) to measure how much attention is allocated to visual tokens, and explores the phenomenon of Lazy Attention Localization. These insights lead to the development of the Attention-Guided Visual Anchoring and Reflection (AVAR) framework, a comprehensive cold-start framework with three components: visual-anchored data synthesis, attention-guided training objectives, and visual-anchored reward shaping. This work demonstrates that AVAR improves multimodal reasoning performance across multiple benchmarks, providing a causal explanation for the importance of visual attention in MLRMs and offering an effective solution to improve reasoning capabilities.

### Strengths
1. Innovative and Logical Metric. The authors introduce VAS, a novel metric that quantifies visual attention and its relationship with reasoning performance. The strong correlation between VAS and model performance provides a fresh perspective on multimodal reasoning (Sec. 3.1-3.2).

2. Sound Motivation. Based on VAS, the paper offers a clear analysis revealing the challenges in MLRM training (Sec. 3.3). The authors discover Lazy Attention Localization, an unexpected phenomenon where multimodal signals are not effectively utilized during cold-start training, providing an insightful explanation for the bottlenecks in current training paradigms.

3. Effective AVAR Framework. The proposed framework improves multimodal reasoning by addressing three components: data synthesis, training objectives, and reward reshaping, contributing to better reasoning performance (Sec. 5).

4. Comprehensive Experiments. Extensive experiments across 7 benchmarks validate the effectiveness of AVAR (Sec. 6.2). Ablation study and analysis confirm the validity of the proposed data synthesis and training methods (Sec. 6.3-6.4).

5. The paper is well-structured and easy to understand.

### Weaknesses
1. In L184–186, "models initialized with unimodal reasoning data, such as OVR-CS and Revisual-R1-CS, maintain 15–20% higher attention to visual features compared to those trained with multimodal reasoning data such as R1-OneVision and ThinkLite-VL." appears unfair, since these methods differ in multiple factors such as dataset composition, model architecture, and training strategy, not merely in whether multimodal reasoning data are used.

2. For Eq. (4) and Eq. (8), the hyperparameters $\alpha$, $\beta$, $\lambda_v$, and $\lambda_f$ lack sufficient explanation. In L375–377, their settings are presented without detailed justification or sensitivity analysis, leaving unclear how these parameters were chosen and how they influence model performance.

### Questions
1. In Table 3, why does performance on MMStar and MMMU-val decrease across all frameworks after training with VAR data?

2. In L375–377, why are the hyperparameters $\alpha$, $\beta$ (from Eq. (4)) and $\lambda_v$, $\lambda_f$ (from Eq. (8)) set to those specific values? Are these empirically determined or theoretically derived configurations?

3. Is the performance improvement of AVAR related to the reasoning difficulty of the benchmarks? For example, does AVAR yield larger gains on tasks requiring complex reasoning, while showing smaller improvements on tasks less dependent on visual reasoning? Are there any observable patterns?

### Soundness
3

### Presentation
3

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
This paper investigates the cold-start initialization stage of Multimodal Large Reasoning Models (MLRMs). It introduces a new metric, the Visual Attention Score (VAS), to quantify a model's attention to visual tokens relative to system tokens. The authors identify a "lazy attention localization" phenomenon, where standard multimodal cold-starts fail to increase VAS, while text-only cold-starts counter-intuitively do. Based on this finding, the authors propose AVAR, a three-part framework designed to explicitly increase VAS. This includes: (1) visual-anchored reflection data synthesis (VARD), (2) attention-guided training objectives (AGTO), and (3) visual-anchored reward shaping (VARS). The authors demonstrate that each component of AVAR progressively increases the model's VAS and that the final model, AVAR-Thinker, achieves strong performance, particularly on mathematical reasoning benchmarks.

### Strengths
1. The finding that text-only cold-start initialization can be more effective at increasing VAS than multimodal cold-starts is a novel and impressive observation.

2. The training-free intervention (Section 4) provides a compelling causal link between attention allocation and performance, even without full retraining.

3. The final model, AVAR-Thinker, demonstrates a significant performance improvement over the Qwen-2.5-VL-7B baseline, especially on challenging reasoning tasks.

4. The curated Visual-Anchored Reflection Data (VARD) could be a valuable contribution to the community, as multimodal reasoning data is relatively scarce.

### Weaknesses
1. **The validity of VAS as a primary metric for visual grounding is questionable.** The paper defines VAS as a ratio of attention to visual tokens over system tokens (Eq. 1). A high VAS score could be achieved simply by aggressively reducing attention to system tokens, even if the absolute attention to visual tokens remains low or unchanged. The paper does not provide analysis to disentangle these two effects. To truly support the claim that AVAR "attends to visual tokens more," the authors should present the attention scores for visual tokens solely (or comparing text tokens), not just their ratio against system tokens, similar to analyses in other visual grounding works [1, 2].
2. **The core mechanism of "attention reshaping" is not properly disentangled.** The paper's central hypothesis is that performance improves by reshaping attention. This reshaping is consistently presented as a two-part mechanism: (1) enhancing visual attention and (2) suppressing system token attention. This duality appears in the training-free intervention (Sec 4), the AGTO loss (Eq 4), and the VARS reward (Sec 5.3). However, the contributions of these two sub-components are never ablated. It is unclear if the performance gain comes from the model genuinely learning to ground itself in visual tokens (as the "Panoramic Vision" title implies) or from simply learning to ignore the system prompt.

---
References:
1. Mitigating Visual Forgetting via Take-along Visual Conditioning for Multi-modal Long CoT Reasoning, Sun et al., ACL 2025
2. v1: Learning to Point Visual Tokens for Multimodal Grounded Reasoning, Chung et al., arXiv 2025

### Questions
1. Could the authors provide an analysis of the attention scores on visual tokens for the baseline vs. AVAR (and for each component addition, as in Table 4) in terms of absolute scores or as a ratio compared to user instruction tokens, rather than just the VAS ratio relative to system tokens? This would confirm that the model is genuinely enhancing visual attention and not just suppressing system prompts.

2. Could the authors provide an ablation study to disentangle the two core mechanisms of the attention-reshaping framework? Specifically, what is the performance of the AVAR-Thinker model and the training-free approach if its components (AGTO and VARS) are modified to only enhance visual attention (i.e., removing the system suppression parts) or only suppress system attention (i.e., removing the visual enhancement parts)?

### Soundness
2

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
3

### Summary
This paper identifies the imbalance distribution of attention weights toward system and visual tokens during multi-modal reasoning, and proposes a training method to tackle the problem. It first develops a step-by-step data synthesis paradigm to general problems with strong focus on visual grounding, and then incorporates training objectives that encourage higher attention weights on visual tokens. Experimental results show that the attention distribution (i.e., the visual attention score) is strongly correlated with model performance, and imposing higher weights on visual tokens (with post-hoc manipulation or training objectives/rewards) leads to considerable improvements.

### Strengths
(1) It is an interesting observation that reasoning performance has an almost linear correlation with attention toward visual tokens.

(2) The paper shows that, even with training-free re-weighting, models can achieve better performance when steering their focuses to visual tokens.

(3) A new data synthesis paradigm is developed, which could potentially benefit training subsequent models.

(4) The proposed method shows promise in improving the performance of the generic baselines on multiple benchmarks.

(5) The paper provides extensive ablation studies, which facilitates understanding the contribution of different components.

### Weaknesses
(1) The variance in attention distribution could also be affected by the design of system prompts used in different models. How does different choices of system prompts affect the visual attention score? Would tuning the prompts, e.g., imposing stronger focus on visual content, help boost the reasoning performance?

(2) The paper only experiments with a single baseline (i.e., Qwen2.5-VL-7B, which is not a reasoning-specific model), and it is unclear whether it will generalize. It would be reasonable to incorporate the method with state-of-the-art reasoning models, e.g., ThinkLite-VL.

(3) Different questions may desire diverse attention to visual content. For instance, some math problems would require more focus on textual tokens. How would the proposed method deal with the diversity of questions?

### Questions
(1) How would using different system prompts affect the observation made on attention distribution?

(2) Would a system prompt emphasizing visual content lead to better reasoning performance?

(3) How does the proposed method work with reasoning models?

(4) The proposed method enforces stronger visual focus among all types of questions, can it be extended to adaptive attention steering?

### Soundness
3

### Presentation
3

### Contribution
3
