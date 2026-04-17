# Mamba-3: Improved Sequence Modeling using State Space Principles

- Decision: Accept (Oral)
- Scores: 6, 6, 8, 8

## Abstract
Scaling inference-time compute has emerged as an important driver of LLM performance, making inference efficiency a central focus of model design alongside model quality. While current Transformer models deliver strong quality, their quadratic compute and linear memory requirements make inference expensive. This has spurred the development of sub-quadratic models with reduced compute and constant memory requirements.
However, many recent linear models trade off model quality and capability for algorithmic efficiency, failing on tasks such as state tracking. Moreover, their theoretically linear inference remains hardware-inefficient in practice.
Guided by an inference-first perspective, we introduce three core methodological improvements inspired by the state space model (SSM) viewpoint of linear models. We combine: (1) a more expressive recurrence derived from SSM discretization, (2) a complex-valued state update rule enabling richer state tracking, and (3) a multi-input, multi-output (MIMO) formulation that improves model performance without increasing decode latency.
Together with architectural refinements, Mamba-3 achieves significant gains across retrieval, state-tracking, and downstream language modeling tasks. At the 1.5B scale, Mamba-3 improves average downstream accuracy by 0.6 percentage points compared to the next best model (Gated DeltaNet), with the MIMO variant further improving accuracy by an additional 1.2 points, for a total gain of 1.8 points.
Across state-size experiments, Mamba-3 achieves comparable perplexity to Mamba-2 despite using half the state size. These results demonstrate that Mamba-3 advances the performance–efficiency frontier.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes Mamba-3, an extension of Mamba-2 with a more inference-focused architecture, which introduces three core improvements to its state-space model (SSM) formulation. First, it replaces Euler discretization with a trapezoidal rule, which can offer offers a lower sequence approximation error.  Second, Mamba-3 empowers a complex-valued SSM via a RoPE-like parameterization. The authors actually show the equivalence of complex-valued SSMs to real SSMs with data-dependent rotations on B and C (analogous to keys and values). Third, Mamba-3 introduces a MIMO (multiple-input multiple-output) SSMs that replace single-vector outer products with matrix-matrix updates in order to increase the arithmetic intensity during decoding. The architecture itself includes other modifications, such as QK-style normalization after B and C projections, and removes the short convolution. The paper evaluates 180M-1.5B models trained on 100B FineWeb-Edu tokens (2K context) across language modeling, retrieval (NIAH), formal languages (parity, modular arithmetic), and inference latency. Mamba-3 generally matches or outperforms Mamba-2 and Gated DeltaNet, with notable gains on state-tracking tasks and inference efficiency, particularly with MIMO.

### Strengths
- The three modifications are grounded in classical SSM theory and integrate cleanly into Mamba-2's framework. The generalized trapezoid recurrence (Eq. 3–4) elegantly recovers Mamba-2 as a special case ($λ_t=1$).

- Propositions 2-4 clearly derive how complex SSMs reduce to real SSMs with RoPE-style rotations. This is a clean way to say that "we can have complex dynamics at only RoPE's computational cost."

- The arithmetic intensity analysis (SISO =  $\theta(1)$, MIMO = $\theta(r)$) convincingly motivates MIMO for IO-bound decoding, and the matrix-matrix update is of course implementation-friendly.

- The ablation study in Table 4 is quite valuable. It shows that convolutions become optional is. In addition, tthe formal-language ablation where disabling RoPE makes the model underperform directly validates the complex/RoPE mechanism. This is insightful and may open new future research ideas into incorporating other forms of positional encoding into SSMs.

- Training four model sizes (180M–1.5B) on identical 100B token budgets enables fair cross-model comparisons.

- Near-perfect accuracy on parity (~100%) and modular arithmetic (~98%), plus improvements on bracketed tasks, demonstrate that Mamba-3 successfully addresses state-tracking limitations of Mamba-2.

### Weaknesses
- All pretraining uses 2K context, and long-sequence tests mostly stay at 2K (NIAH) or only reach 4K (2x extrapolation). For a model claiming better sequence approximation with the trapezoidal rule, evaluation should stress-test extrapolation at longer scenarios (e.g., 32K), especially since the paper notes performance drops where Gated DeltaNet sometimes wins.

- Table 3 reports single-point latencies without variance. Critically, the authors use "custom kernels" for Mamba-3 but "standard reference" implementations for baselines—making results difficult to verify. Optimized kernels for competitors could reverse the rankings. The paper must specify kernel implementations, fusion levels, and hardware details. Without this, is very hard to access the real efficiency contributions of Mamba-3.

- Figure 3's missing Mamba-3 MIMO point makes the work seem a bit sloppy. Of course, I think the model will perform well $d_{state}=32$, but still find this unconfortable.

- While the main LM results use SISO, the inference analysis argues for MIMO. In the formal language tables there's only a mention to "Mamba-3.", so it is unclear which method was used here. Since MIMO affects FLOPs, parameters, and stability, every benchmark should explicitly state "SISO" or "MIMO (rank=r)."

- For an "inference-first" paper, comparisons should include production-grade transformer inference (FlashDecoding, paged KV cache, vLLM chunked prefill, etc) at realistic batch sizes and sequence lengths. Moreover, only decoding latency is reported, despite prefill being an extremelly important part of inference in LLMs. This omission hides important costs.

- Proposition 1 claims $O(\Delta_t^2)$ global error with data-dependent $\lambda_t$ but doesn't state required regularity conditions (e.g., Lipschitz continuity).

- I found the paper to be very condensed. It seems like the distance between letters was reduced to make more things fit into the paper. This make things hard to read.

### Questions
- What regularity assumptions on $\lambda_t(x_t)$ guarantee quadratic global error? 

- Does the rotation-absorption argument (Eq. 18–19) hold when $A_t$ is diagonal rather than scalar? 

- If $\lambda_t = \frac{1}{2}$, the scheme is 2nd order; if $\lambda_t$ is data-dependent but Lipschitz and satisfies $\lambda_t = \frac{1}{2} + O(\Delta_t)$, it is still 2nd order; otherwise the order collapses to 1---Euler's rule. Is this collapse ensured in Mamba-3?

- Inference implementation details: For Table 3, specify: (a) kernel type (Triton/CUDA) and fusion level for each baseline, (b) exact configuration (batch size, sequence length, heads, dstate), (c) whether all models use comparable optimization levels.

- Did you try evaluating the 1.5B model (trained at 2K) on 8K–32K retrieval tasks, not just LM perplexity extrapolation (Fig. 5)? The LM curve is nice, but the claim of better sequence approximation would be more convincing on long retrieval where the state is actually stressed.

- Since your goal is inference improvements, could you report prefill (1K-4K) and decode (1 token) wallclock-times for a strong transformer baseline using FlashDecoding (a comparable open kernel on the same hardware is also valuable)? Even if Mamba-3 still wins on decode, showing the transformer prefill margin would make the paper easier for readers.

- Hammering on this again, but why SISO for main LM runs? If MIMO is superior at equal state size (per the inference section), why use SISO for 820M/1.5B language modeling?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
Mamba-3 is a state-space model (SSM) designed from an inference-first perspective that addresses the limitations of existing sub-quadratic sequence models (e.g., Mamba-2, Gated DeltaNet) in quality, state-tracking capability, and hardware efficiency. It achieves SOTA performance across language modeling, state-tracking, and retrieval tasks while maintaining linear compute and constant memory, setting a new Pareto frontier for performance-efficiency tradeoffs.

### Strengths
1. The three core improvements, trapezoidal discretization, complex-valued SSMs, and MIMO formulation, are not isolated tweaks but creative combinations of classical SSM theory and modern LLM needs. Trapezoidal discretization generalizes Euler’s rule (used in Mamba-2) to a second-order accurate recurrence, while the complex-valued SSM recovers rotational dynamics absent in real-valued counterparts.

2. Rigorous Theory & Comprehensive Empirical Validation. The paper includes rigorous proofs for key propositions (e.g., complex-to-real SSM equivalence, trapezoidal discretization error bounds) and formalizes the mathematical underpinnings of each innovation. This theoretical rigor ensures the model’s behavior is well-understood, not just empirically effective.

3. The paper reports detailed experimental settings (training data, tokenization, hyperparameters) and provides clear metrics (perplexity, accuracy, TPS, arithmetic intensity) that enable follow-up work.
And structured presentation & accessible exposition increses the avaliability. 

4. Addressing Critical LLM Deployment Pain Points. By pushing the Pareto frontier of performance vs. inference efficiency, Mamba-3 directly addresses a key bottleneck in LLM deployment—scaling inference without prohibitive memory/compute costs. The MIMO variant, in particular, improves hardware utilization without increasing state size, making it viable for real-world applications.

### Weaknesses
1. Inherent limitations of retrieval capabilities and insufficient comparison. Mamba-3 significantly lags behind Transformer in information extraction tasks from semi-structured/unstructured data, and the root causes and potential solutions are not explored in depth.

Table 2 of the paper shows that Mamba-3 with 1.5B parameters has poor accuracy in real-world retrieval tasks such as SWDE and FDA; even in the synthetic "needle-in-a-haystack" task, when the context length exceeds 2048, the accuracy of Mamba-3 is still far below the theoretical upper limit of Transformer, and it is not compared with linear models specifically optimized for retrieval (such as Arora et al., 2025b's "Just Read Twice").

2. Limited experimental scale of the key variant (MIMO) and incomplete hyperparameter ablation. The MIMO variant was only trained on a 440M parameter model (paper 4.2.3). Because the "computational constraint" was not extended to a 1.5B scale, its performance-efficiency tradeoff under larger models could not be verified. Furthermore, the MIMO rank r was fixed at 4, and the impact of different values ​​such as r=2 and 8 on arithmetic strength and inference speed was not tested (Figure 2 only shows the theoretical trend of r, without experimental data).

3. The paper's baseline only includes Mamba-2, Gated DeltaNet, and the standard Transformer, omitting similar linear models such as Rwkv-7, DeltaProduct, and Block-Biased Mamba. These models all innovated on the expressiveness or efficiency of linear models, and some performed exceptionally well on state tracking tasks.

### Questions
1. Complex-Valued SSM & RoPE Equivalence: The paper argues that complex-valued SSMs are equivalent to data-dependent RoPE embeddings and compute-efficient. Could you provide a direct comparison between complex-valued SSMs and *explicitly applying RoPE* (e.g., on B/C projections of Mamba-2) in terms of: (1) training/inference speed, (2) memory usage, and (3) state-tracking performance? 

2. MIMO Rank Selection: The MIMO variant uses rank \( r=4 \) for 440M models, but there is no analysis of how \( r \) impacts performance-efficiency tradeoffs. What is the marginal gain of increasing \( r \) (e.g., \( r=8, 16 \))? At what point does \( r \) become compute-bound (negating efficiency gains)? Additionally, why was \( r=4 \) chosen as the default, was it optimized via grid search or heuristic?




3. Ablations. The BC bias synergizes with trapezoidal discretization, but you do not specify its initialization strategy (e.g., zero, Xavier, data-dependent) or whether it is updated during training. Does the bias’s initialization significantly affect performance?

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
Current State-Space Models have some draw-backs compared to Transformer based models, such as lack certain capabilities, low generalization quality, not hardware-efficient, etc. This paper proposes the Mamba-3 architecture, from inference-first perspective. Specifically, the trapezoidal discretization is used to improve the expressive dynamics, complex-valued state spaces is introduced for state-tracking, and the multi-input multi-output (MIMO) mechanism is proposed for hardware efficiency. Experiments of show that Mamba-3 outperforms Mamba-2 or even transformer based models across 180M to 1.5B parameters.

### Strengths
1. The proposed Generalized Trapezoidal Discretization is more accurate than the Euler's rule used in Mamba2, by using the second-order  approximation of the integral. 
2. The complex SSM is novel, which is equivalent to a real SSM with data-dependent rotary embeddings (RoPE). Detailed theoretical analyses are provided.
3. The MIMO method efficiently solves the I/O problem, which is a key bottleneck for Mamba-2.

### Weaknesses
More complex tasks such as reasoning can be explored to fully demonstrate the capability of Mamba-3.

### Questions
Can the proposed Mamba-3 be generalized to vision tasks? 

Does Mamba-3 still has advantages when compared to sparse attention based transformers?

### Soundness
3

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
This paper introduces Mamba-3, a followup sequence model architecture that builds upon the Mamba family of SSMs. The work is motivated by an "inference-first" perspective, aiming to create a model that excels across three axes: quality, capability, and inference efficiency, addressing limitations in prior sub-quadratic models like Mamba-2.  Basically, this paper propose three core, major improvements derived from classical SSM theory and also validate Mamba-3 through extensive experiments, showing that it outperforms strong baselines in language modeling, possesses new state-tracking capabilities on synthetic tasks, and establishes a new Pareto-frontier for performance versus inference cost.   It presents a clear, significant, and well-executed advancement in the design of efficient sequence models. The paper is strong for three main reasons:

1.  **Principled Improvements over existing SSMs/Mamba family:** The core contributions are not arbitrary architectural tweaks but are grounded in control theory and SSM principles. The connection drawn between complex SSMs and data-dependent RoPE is very elegant IMO and provides a clear theoretical justification for the observed gains in capability.

2.  **Holistic Improvement:** The work addresses the multi-faceted challenge of building a practical LLM. It simultaneously pushes forward model quality, unlocks new fundamental capabilities (state-tracking issue which has been rooted in RNNs), and improves hardware-level efficiency(MIMO formulation). This is a rare and impactful combination. (Though it will be more impactful if all artifacts will be open-sourced. 

3.  **Compelling and Rigorous Empirical Evidence:** The experimental validation is thorough and persuasive. The head-to-head comparisons on language modeling, the stark success/failure results on formal language tasks, and the Pareto-frontier analysis collectively provide a powerful and convincing case for the superiority of the Mamba-3 design.

### Strengths
- **Theoretical Elegance and Insight:** The paper's standout strength is its use of classical SSM theory to motivate architectural changes. The derivation of a data-dependent RoPE from a complex state-space is a beautiful and insightful result that bridges two important lines of work.

- **Targeted Problem Solving:** Each of the three methodological changes directly targets a known weakness in prior models. Trapezoidal discretization for expressivity, complexification for state-tracking, and MIMO for hardware utilization. This focused approach makes the paper's narrative clear and its contributions easy to appreciate.

- **Excellent Empirical Validation:** The experiments are top-notch. The Pareto-frontier plot (Figure 3) is a highly effective way to visualize the trade-off between performance and inference cost, clearly showing Mamba-3's dominance. The ablation on state-tracking capabilities (Table 4b) is a decisive demonstration of the complex SSM's impact.

*   **Practicality and Inference-First Focus:** The introduction of the MIMO variant shows a deep understanding of the practical bottlenecks in LLM deployment. By focusing on arithmetic intensity, the authors address a subtle but critical aspect of real-world performance that goes beyond theoretical FLOPs.

-  **High-Quality Presentation:** The paper is extremely well-written, with complex ideas explained clearly and supported by well-designed figures and tables.

### Weaknesses
My major concern is that retrieval Capabilities Still Lag Transformers: While the paper is honest about this, and Mamba-3 shows improvement over Mamba-2, the results in Table 2 confirm that a fundamental gap in retrieval performance versus Transformer models remains. This is an inherent challenge for fixed-state recurrent models and represents a key limitation. Maybe authors can add some potential exploration about hybrid models? I think we are at the age that Hybrid models are becoming more important and can help alleviate the shortages of RNNs which we should acknowledge and make good use of.

### Questions
- The results on retrieval (Table 2) are improved but still trail the Transformer baseline significantly. Do you view this as a fundamental ceiling for fixed-state models, or do you see pathways for future SSM-based architectures to further close this gap? Maybe authors can add some potential exploration about hybrid models? I think we are at the age that Hybrid models are becoming more important and can help alleviate the shortages of RNNs which we should acknowledge and make good use of.

- The "RoPE trick" is a very efficient way to implement the complex SSM. Could you briefly comment on how the cumulative product of rotation matrices `(Π R_i)` is handled efficiently during both the parallel training scan and the sequential inference steps?

Additional questions:

- The MIMO formulation is a compelling idea for improving hardware efficiency. Could you elaborate on the selection of the MIMO rank `r=4`? Have you explored the sensitivity of model performance and inference latency to different values of `r`? Is there a risk of diminishing returns or even performance degradation if `r` becomes too large?

- The ablation in Table 4a convincingly shows that the trapezoidal discretization and BC bias make the short convolution redundant. Can you provide more intuition on *why* this is the case? Does the learned size-two convolution from the trapezoidal rule effectively capture the same local information as a dedicated convolution layer?

### Soundness
3

### Presentation
4

### Contribution
3
