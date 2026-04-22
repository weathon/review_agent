# Long-Tailed Distribution-Aware Router For Mixture-of-Experts in Large Vision-Language Model

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 4, 4, 6

## Abstract
The mixture-of-experts (MoE) architecture, which replaces dense architectures with sparse ones, has garnered attention in large vision-language models (LVLMs) for achieving comparable performance with fewer activated parameters. Existing MoE architectures for LVLMs primarily focus on token-to-expert routing (TER), encouraging different experts to specialize in processing specific tokens. However, these architectures typically rely on the load balancing mechanism, neglecting the inherent distributional differences between vision and language modalities. To address this, we propose the Long-Tailed Distribution-aware Router (LTDR) for vision-language TER, which tackles two key challenges: (1) Modality-specific distribution-aware routing. We observe that language TER follows a relatively uniform distribution, whereas vision TER exhibits a long-tailed distribution. This modality discrepancy necessitates specific routing strategies for each modality. (2) Vision-specific expert activation. Recognizing the importance of high-information vision tail tokens, we introduce an oversampling-like strategy by increasing the number of activated experts to sufficiently learn vision tail token representations. Experiments on extensive vision-language and vision benchmarks validate the effectiveness of our approach.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors introduce the Long-Tailed Distribution-aware Router (LTDR), a MoE routing strategy specifically designed for LVLMs. The core premise is that the standard MoE load balancing loss conflicts with the inherent long-tailed distribution of visual tokens, hindering expert specialization on salient, high-information (tailed) tokens. LTDR addresses this via two strategies: Modality-specific Distribution-aware Router (MsDaR), which removes the load balancing loss for visual tokens to increase their Routing Probability Variance (RPV), and Vision-specific Dynamic Expert Activation (VsDEA), which increases the active expert count ($k$) for tokens identified as high-RPV vision tail tokens.

### Strengths
1.  **Novel Modality-Specific Motivation:** The paper provides a well-motivated critique of standard MoE routing in LVLMs by highlighting the distributional mismatch between visual (long-tailed) and language (more uniform) tokens. This modality-aware design is a novel and important direction for MoE in multimodal settings.

2.  **Intuitive Two-Part Mechanism:** The proposed solution is architecturally clean, addressing both the specialization aspect (MsDaR via RPV maximization) and the learning sufficiency aspect (VsDEA via dynamic $k$) for critical tail tokens.

3.  **Clarity of Presentation:** The methodology is clearly presented with corresponding graphs and mathematical formulations.

### Weaknesses
1.  **Risk of Expert Collapse and Learning Stability of MsDaR:**
The removal of the explicit load balancing loss for visual tokens, aimed at increasing RPV, inherently introduces the risk of expert collapse, where a few experts become severely over-specialized and overloaded, undermining the fundamental purpose of MoE (distributed computation and diverse learning).

2.  **The Effectiveness of the Core Motivation:**
The empirical gains shown in Table 1 appear marginal, especially in smaller settings (e.g., Q-1.8B), leading to my concern of its effectiveness. Empirically, with sufficient training, standard MoE experts might naturally learn to differentiate between high-information foreground tokens (tail) and low-information background tokens (head), spontaneously increasing the RPV for critical tokens.

3.  **Experimental Scale and Generalization:**
The entire evaluation is limited to a single configuration using the LLaVA + ShareGPT dataset. MoE architectures typically realize their full benefits and stability when validated on llarge-scale pretraining or instruction-tuning datasets.

4.  **Suggestion for Future Work:**
While understandable for baseline comparison, the use of older encoders and LLMs limits the paper's immediate relevance and impact. For future work, it is strongly suggested that the method be migrated and validated using state-of-the-art vision encoders and LLMs (e.g., Siglip2 and Qwen2.5 series) to showcase the method's potential in competitive settings.

### Questions
1. The authors should provide a discussion and empirical evidence (e.g., expert load statistics or variance analysis over training) demonstrating the robustness of the MoE structure (vision parts) without the visual balancing loss. How is the expert load distribution maintained, and what implicit constraints prevent catastrophic expert dependency?

2. The authors should provide a direct comparison of the RPV distribution for visual tokens between a standard MoE baseline and LTDR at different training stages to demonstrate that the baseline fails to achieve the desired specialization, or that LTDR significantly accelerates this process. Marginal performance gains fail to justify the mechanical effectiveness.

3. The authors should demonstrate the scalability and generalization ability of LTDR by evaluating its performance and stability across larger-scale pretraining or SFT datasets to ensure the method holds value as a general MoE strategy.

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
This paper proposes LTDR, which (1) eliminates load balancing for vision tokens and (2) assigns more experts to tail vision tokens. Experimental results demonstrate consistent performance improvements on both vision and vision-language tasks.

### Strengths
1. The motivation is clear, and the paper is easy to follow.
2. Numerical results show performance gain compare to baselines.

### Weaknesses
1. **Efficiency and All-to-All Communication Implementation**:  
   The paper claims an efficient implementation of all-to-all communication. However, since different tokens may activate a varying number of experts under LTDR, standard efficient MoE frameworks (e.g., Deepspeed and Tutel)—which typically assume uniform expert activation—may not apply directly. Could the authors clarify how they achieve efficient all-to-all communication in this dynamic setting? Additionally, why is LTDR more computationally efficient than MoE-LLaVA despite activating more experts for tail vision tokens?

2. **Technical Novelty**:  
   The core modifications—removing the load-balancing loss and increasing the number of experts for tail vision tokens—appear relatively straightforward. Could the authors elaborate on what they consider the key technical innovations of LTDR, and how it meaningfully advances beyond existing MoE or routing strategies in multimodal settings?

3. **Attribution of Performance Gains**:  
   It is unclear whether the observed performance improvements stem primarily from increased model capacity (i.e., more activated parameters due to higher expert activation) rather than improved routing or representation learning. To better understand this, could the authors report the average top-K value per token and the total number of activated parameters during inference for both LTDR and baseline models?

4. **Scalability to High Vision-Token Scenarios**:  
   In tasks such as video understanding, the number of vision tokens can vastly exceed that of text tokens. How does LTDR scale in such high-token regimes? Does the method incur significant computational or memory overhead, and have the authors evaluated its performance or efficiency on long-sequence or dense vision tasks?

### Questions
See above

### Soundness
2

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
3

### Summary
The paper proposes LTDR, a long-tailed distribution-aware router for MoE-based LVLMs. It observes that language tokens exhibit near-uniform token-expert routing (TER), whereas vision tokens are long-tailed; standard load balancing thus misaligns with vision TER and can hinder expert specialization. LTDR consists of:
(i) MsDaR — retain load balancing for language but remove it for vision, increasing routing probability variance (RPV) and specialization.
(ii) VsDEA — identify high-RPV vision “tail” tokens and activate more experts (k→a) for them. Experiments show consistent gains across LVLMs and vision-only tasks with no significant increase in inference time.

### Strengths
1.	Simple, compatible design: MsDaR only alters the balancing objective by modality; VsDEA uses a mean-RPV threshold and k→a activation. Easy to integrate into existing MoE stacks.
2.	Broad, consistent improvements: Steady gains across MoE-LLaVA and Molmo; vision-only GMoE also benefits (e.g., ~+1.0% Avg).
3.	Latency and efficiency: Inference time remains essentially unchanged; the all-to-all bottleneck explanation is plausible and supported by expert-load analysis.

### Weaknesses
1. The paper describes visual tokens as "long-tailed" based on RPV histograms (Figure 1(b)), but lacks statistical tests (e.g., tail index, power-law fitting) to rigorously support this claim.
RPV is influenced by router parameters (gate temperature, regularization), so its "tail" may not reflect semantic or information-based long-tailedness.

2. The choice of base models is narrow, limiting generalizability. Descriptions like "Qwen-1.8B" are too vague; specific model details are needed.

3. No comparison with simpler methods (e.g., temperature/entropy scaling) to sharpen visual token routing, which could also increase RPV without disabling load balancing.

4. The mean-RPV threshold (Equation 11) lacks ablation studies (e.g., generalizability across datasets/models). Only fixed proportion thresholds (10/15/20%) are tested (Appendix B.5), with no deeper sensitivity analysis.

5. MsDaR removes load balancing for visual tokens (Equation 10), but risks like expert overloading or training instability under large-scale K (e.g., 64) or extreme data distributions are not quantified.

6. Main results show small improvements (+0.4 to +1.2, Table 1), but significance tests are only in Appendix C.2 (Table 18).
Confidence intervals and variance should be included in the main text.

7. The impact of reducing (not removing) load balancing or adding constraints (e.g., temperature/entropy) is not studied.
How do methods like temperature annealing or entropy regularization compare to VsDEA?

8. The link between RPV and token informativeness relies on visualizations/ablations; stronger statistical or interpretability evidence is needed.

9.  VsDEA’s choice of activated experts (a) is empirical; systematic tuning or adaptive policies would improve credibility.

10. Benefits under near-uniform token distributions are unclear; synthetic studies could help define boundary conditions.

11. Batch-wise mean-RPV thresholds may misclassify non-critical patches as "tails," diverting experts from important regions. A learnable joint mechanism is suggested.

### Questions
1.	RPV ↔ informativeness: Can you quantify correlation between high-RPV vision tokens and downstream performance or gradient-based relevance (e.g., leave-one-token-out, attribution metrics)?
2.	Adaptive a: Could a be chosen via RPV quantiles or a compute-budgeted policy? Any results across different K/k?
3.	Training stability: Does removing vision load balancing cause early expert skew or collapse? Is entropy regularization or warm-up needed?

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
3

### Summary
The paper revolves around the contradiction in MoE routing (TER) within LVLMs between “load balancing preferring uniform distribution” and “inconsistency in modality distributions.” Through multiple experiments and visualizations, the authors observe that the TER probability variance distribution for language tokens is closer to uniform, whereas for visual tokens it exhibits a clear long-tail pattern. Under these conditions, enforcing load balancing on visual tokens disrupts the specialization of the “tail” visual tokens that are rare yet critical. Based on this, the authors propose LTDR, which includes MsDaR, applying different balancing constraints to language and vision, and VsDEA, which uses routing probability variance (RPV) to identify “tail” visual tokens online and activate more experts for them.

The core idea is to retain load balancing only on the language side and remove it for the visual side to increase RPV; a dynamic threshold is used to route tail tokens to Top-$a$ experts instead of Top-$k$. The authors validate this on MoE-LLaVA, Molmo, and GMoE, achieving average gains of about +1.2%, +2.0%, and +1.0%, respectively, and provide ablations, routing comparisons, runtime and memory statistics, and visualization analyses. It should be noted that the conclusion “language nearly uniform, vision long-tailed” is primarily supported by the examples and visualizations in the paper, and its applicability still depends on the model and router setup; it cannot yet be regarded as a cross-architecture law.

### Strengths
The problem setup is practice-oriented: when the MoE balancing term favors “uniformity” but visual tokens are “long-tailed,” rare yet key information can indeed be diluted. LTDR’s two modifications are both minimal: MsDaR merely limits the balancing term to language sequences, and VsDEA activates more experts for a small number of high-RPV visual tokens without altering sequence length—effectively an “oversampling-like” opportunity expansion. The authors provide extensive comparisons, ablations, and cost analyses, including runtime, memory, expert load distribution, and random-seed intervals, making the method reproducible and practically valuable.

### Weaknesses
The “language near-uniform, vision long-tailed” characterization currently depends on the chosen models and visual tokenizer (Appendix D.2 notes 576 CLIP tokens per image) as well as specific router/Top-k settings. Although extensive experiments are conducted on MoE-LLaVA, Molmo, and GMoE, it remains unclear whether this distribution pattern holds when changing router types (e.g., dynamic expert counts, segmented softmax, top-1 vs top-2), expert scales and layers, or visual tokenizers. A systematic validation is lacking. I suggest adding cross-router/backbone RPV distribution comparisons and variance analyses to avoid over-generalizing empirical observations.

Moreover, the implementation of VsDEA Top-a is somewhat heuristic: Eq. (12) does not specify whether weights are renormalized or if temperature/thresholding is used to suppress noise from low-probability experts; inference-time latency and all-to-all waiting for a=12 versus k=8 on Molmo are not quantified (only MoE-LLaVA reports timing). In addition, slight degradation appears on the POPE Adversarial subset, suggesting that “activating more experts for tail tokens” may introduce bias toward overconfidence on adversarial negatives—this deserves further analysis and mitigation.

### Questions
I would especially like the authors to perform a “strategy-swap” ablation: remove load balancing on the language side while keeping it on the vision side, and examine the corresponding MsDaR and VsDEA variants. Observing whether de-balancing language routing harms its originally uniform distribution or introduces expert bias in instruction-alignment tasks would directly support (or challenge) the claim that “modality-specific strategies are necessary.” Conversely, if performance improves after swapping, it would significantly refine the boundary conditions of the conclusion.

Additionally, I hope to see two implementation and measurement clarifications:

Details on normalization, clipping, and temperature settings in Top-a, and their impact on gradients and communication costs.

Broader RPV distribution and performance comparisons across routers/backbones, especially including dynamic routing and top-1 gating.

Finally, regarding POPE Adversarial degradation, could VsDEA incorporate confidence- or consistency-based gating, or be combined with methods like STGC (gradient conflict mitigation) to suppress overconfidence when seeing partial visual evidence? Addressing these points could further raise my overall evaluation.

### Soundness
3

### Presentation
3

### Contribution
3
