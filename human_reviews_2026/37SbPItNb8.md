# Adaptive Residual-Update Steering for Low-Overhead Hallucination Mitigation in Large Vision-Language Models

- Decision: Reject
- Scores: 2, 6, 4, 4

## Abstract
Large Vision-Language Models (LVLMs) often suffer from object hallucination, generating text inconsistent with visual inputs, which can critically undermine their reliability. Existing inference-time interventions to mitigate this issue present a challenging trade-off: while methods that steer internal states or adjust output logits can be effective, they often incur substantial computational overhead, typically requiring extra forward passes. This efficiency bottleneck can limit their practicality for real-world, latency-sensitive deployments.
In this work, we aim to address this trade-off with **Residual-Update Directed DEcoding Regulation (RUDDER)**, a low-overhead framework that steers LVLMs towards visually-grounded generation. RUDDER is built on two key innovations: (1) Contextual Activation Residual Direction (CARD) vector, a per-sample visual evidence vector extracted from the residual update of a self-attention layer during a *single, standard forward pass*. (2) A Bayesian-inspired adaptive gate that performs token-wise injection, applying a corrective signal whose strength is conditioned on the model's deviation from the visual context.
Extensive experiments on key hallucination benchmarks, including POPE and CHAIR, indicate that RUDDER achieves performance comparable to state-of-the-art methods while introducing negligible computational latency, validating RUDDER as a pragmatic and effective approach for improving LVLMs' reliability without a significant compromise on efficiency.
Code is available at https://anonymous.4open.science/r/RrUuDdDdER-1C13/.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces RUDDER, an inference-time intervention technique designed to mitigate object hallucinations in large vision-language models (LVLMs) with minimal computational overhead. The method features two core components: (1) the Contextual Activation Residual Direction (CARD) vector, a per-sample visual evidence representation derived from residual updates in a self-attention layer during a single forward pass; and (2) a Beta Gate, a Bayesian-inspired adaptive gating mechanism that dynamically steers generation toward stronger visual grounding. Evaluations on hallucination benchmarks (CHAIR and POPE) and general multimodal assessments (MME) across three LVLM architectures demonstrate that RUDDER achieves  hallucination reduction at lower inference costs compared to existing interventions.

### Strengths
1). The CARD vector is efficiently extracted from the standard computation pipeline, while the Beta Gate relies on lightweight vector operations, enabling seamless deployment in latency-constrained real-world applications.

2). The ablation studies and illustrative examples are thorough, providing clear insights into the method's mechanics.

### Weaknesses
1). The approach shares conceptual similarities with cross-layer methods like DeCo, which integrates early-layer logits into later layers to address visual forgetting, and RUDDER similarly incorporates attention from early generated tokens into later ones. A direct comparison with such methods would strengthen the novelty claims.

2). The method introduces multiple hyperparameters (e.g., $L$, $k$, and $\alpha_{\max}$), raising concerns about its stability and generalizability across diverse VLMs.

3). Performance gains on the test sets are modest, and the method's efficacy diminishes as the underlying VLM's capabilities improve. The baselines are somewhat dated, omitting recent mainstream VLMs like Qwen-VL and InternVL, which questions its applicability to current open-source models. Additionally, comparisons with recent training-free hallucination mitigation techniques, such as DeGF and AGLA, are absent, limiting the benchmarking comprehensiveness.

4). The writing could be refined for clarity and conciseness; for instance, the introduction devotes excessive space to reiterating the need for an "effective and lightweight" solution and listing RUDDER's components, without adequately highlighting key insights, motivations for each module, or defining terms. The final three paragraphs overlap significantly, resulting in low information density.

### Questions
1). Could you include experimental results on contemporary VLMs such as Qwen-VL and InternVL? Additionally, please add comparisons with current SOTA methods like DeGF and AGLA.

2). To facilitate broader adoption, could you provide a concrete recipe for automated hyperparameter tuning? For example, suggest strategies like grid search, Bayesian optimization, or other efficient approaches for optimizing $L$, $k$, and $\alpha_{\max}$ in new deployment scenarios?

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
4

### Summary
This paper proposes RUDDER, a single‑pass inference‑time steering method for LVLMs that (1) extracts a per‑sample direction (CARD) from self‑attention residual updates during prefill and (2) applies a per‑token Beta‑gate to adapt the steering strength during decoding.  On CHAIR and POPE, across three LVLMs and three decoding strategies, RUDDER typically matches or outperforms strong ITI baselines, while keeping ~baseline latency/throughput. General ability on MME is largely preserved

### Strengths
1. Low‑overhead (no extra forwards) with clear efficiency gains versus prior steering methods; quantitative latency/throughput reported. 

2. Minimal code hooks; works across three distinct LVLM architectures; integrates with standard decoding loops. 

3. Token‑wise gate improves precision vs fixed‑strength steering; ablations show why adaptive > fixed for open‑ended captioning. 

4. Solid experiments, including cross‑model, cross‑decoding, efficiency measurements, and layer/parameter sweeps.

### Weaknesses
1. The “Bayesian‑inspired” gate is heuristic, there is no formal guarantee that the steering always improves negative log likelihood, though ablations suggest it works well.

2. Layer choice is model‑specific (e.g., late layers for LLaVA/Idefics2; early for InstructBLIP with Q‑former), and final configs differ substantially across backbones. Ablations confirm a sensitive trade-off between CHAIR scores and recall, implying non-trivial parameter search per model/task.

3. The evaluation scope is modest, evaluation on more capabilities like MM-Vet would be beneficial.

4. No diagnostic attribution of why corrections happen. The paper shows outcome metrics and some internal geometry analyses but lacks faithfulness diagnostics that could verify that the method truly reduces language-prior reliance rather than suppressing certain token types.

### Questions
1. Exactly which tensors are pooled to form CARD (per‑layer, per‑head residual updates after self‑attention, before MLP)? What pooling (mean, median, head‑weighted)?

2. How were g_min, g_max, and softplus temperature chosen?

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
4

### Summary
The paper introduces RUDDER, a lightweight inference-time framework to reduce hallucinations in LVLMs with (almost) no extra computational cost. RUDDER extracts a Contextual Activation Residual Direction (CARD) vector from residual updates during a single forward pass to capture visual evidence, and applies an adaptive Beta Gate to modulate correction strength per token based on visual alignment. Experiments on benchmarks like CHAIR and POPE show that RUDDER matches or surpasses SoTA hallucination mitigation methods while maintaining nearly identical inference speed and general multimodal performance, making it a practical solution for real-world LVLM deployment.

### Strengths
- **Good Writing.** The writing of the paper is clear and easy to follow (although more high-level intuition and motivation can be expressed in a better way). 
- **Extremely Low Computational Overhead.** It's a smart idea to utilize the intermediate results (embeddings, attention heads, etc.) of the pre-filling phase for later steering of the LVLMs. This indeed avoids typical repetitive computation in the contrasting-based methods. The empirical results on the efficiency analysis perfectly supports this. 
- **Extensive Experimental Results.** Experiments are conducted in various evaluation benchmarks on multiple LVLM backbones, supporting the main claims of the paper.

### Weaknesses
+ **Lack of (Sometimes Contradictory) Intuitive Explanation for the Proposed Method.** Despite its practical values in terms of performance and efficiency, I find it hard to understand the rationale behind the proposed method: 
  + What is the meaning of the main body of the steering vector $v\_{\text{CARD}}$? 
  + Why pooling the token-wise attention output $\Delta$ is a good idea, not causing too much information loss?
  + If the similarity score $g$ between the current token's hidden state $h$ and $v\_{\text{CARD}}$ is high, then this hidden state already contains lots of visual information. Why would we want to do stronger steering: $v\_{\text{steer}}=g \cdot v\_{\text{CARD}}$ in this case? Shouldn't we put more steering on the ones that loses lots of visual information? 
+ **Over-claims about "Bayesian".** It's a bit hard to persuade me to believe the gating mechanism is "Bayesian". This is a general training-free strategy, no parameters are updated based on new observations. To me this gating mechanism is at most "adaptive". 
+ **Sensitive hyperparameters setting.**
  + This method introduced many hyperparameters, and they are all adjustable: $L$, $\alpha\_{\text{max}}$, $k$, etc. 
  + The hyperparameters are all set differently for different models and different benchmarks, showcasing the sensitiveness of them.
  + It's not clear how the author found the optimal setting of the hyperparameters. Is it based on an extra validation dataset?

### Questions
See above (Weaknesses).

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
This paper proposes RUDDER (Residual-Update Directed DEcoding Regulation), a method to mitigate object hallucination in Large Vision-Language Models (LVLMs). The method introduces two key components: (1) the CARD vector, a per-sample visual steering signal extracted at negligible computational cost during the prefill stage, and (2) the Beta Gate, an adaptive token-wise mechanism that dynamically adjusts intervention strength. While the experimental results appear promising, several aspects require clarification and further validation.

### Strengths
1. The paper clearly identifies the trade-off in existing methods—existing approaches incur high computational overhead and require multiple forward passes, which limits their practical deployment.

2.  The concept of dynamically adjusting intervention strength based on the model's deviation from visual context is well-motivated. 

3. The paper is generally well-structured and clearly written, making it accessible to readers.

### Weaknesses
1. The Beta Gate design appears fundamentally counter-intuitive. According to Equation (3):

- When hl,t has high similarity with vCARD (i.e., cos⁡(hl,t, vCARD)≈1), the intervention strength g_t becomes large.
- When hl,t derivates from vCARD (i.e., cos⁡(hl,t, vCARD) is negative), the intervention strength g_t becomes small.
- This design contradicts common intuition: one would expect stronger intervention when the model deviates from visual grounding, not weaker. The paper does not adequately justify this seemingly backward design choice.


2. While the paper claims Beta Gate is "Bayesian-inspired," it lacks rigorous Bayesian derivation. The connection between the Beta distribution framework and the specific formulation in Equation (3) is unclear.

3. The paper omits several state-of-the-art hallucination mitigation methods. Missing references and performance comparisons (including effectiveness and efficiency): OPERA (Huang et al., 2023), HALC (Chen et al., 2024), ADHH (Yang et al., 2025). 

OPERA: Alleviating Hallucination in Multi-Modal Large Language Models via Over-Trust Penalty and Retrospection-Allocation. CVPR 2023

HALC: Object Hallucination Reduction via Adaptive Focal-Contrast Decoding. ICML 2024.

Understanding and Mitigating Hallucinations in Large Vision-Language Models via Modular Attribution and Intervention. ICLR 2025.

### Questions
1. How are k (sensitivity) and c (concentration) determined? Why they are necessary?

2. Do hyperparameters need adjustment for different models (e.g., 7B vs. 13B)?

### Soundness
2

### Presentation
3

### Contribution
2
