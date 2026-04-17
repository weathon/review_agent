# Closed-Loop Activation Density Control for Sparse Distributed Memory

- Decision: Reject
- Scores: 6, 6, 4, 4

## Abstract
Associative memory models often suffer from sensitive parameters that hinder stable operation in high-dimensional settings. In particular, Pentti Kanerva's Sparse Distributed Memory (SDM) requires setting a Hamming distance threshold $T$ to determine which memory locations are activated on a query, a choice that critically affects performance.

We address this parameter sensitivity: directly tuning $T$ is ill-conditioned around the nominal operating point (half the dimension), where a minute change in $T$ produces a large swing in the number of activated locations $k$. Instead, we control the activation density $p = k/L$, which is well-posed, and adjust $T$ indirectly.

Our controller combines inverse-CDF actuation with slope-normalized integral feedback to cancel the large plant gain near $n/2$. The result is a closed-loop SDM that adapts $T$ on the fly to track a desired sparsity level $p^*$ across queries.

Empirically, the loop achieves near-perfect target tracking ($R^2 \approx 1.00$) and improves query efficiency, reducing activation error by $\approx 5.2\times$ compared to naive threshold control at equal query budgets, while standard bisection attains similar raw error but requires $\approx 1.9\times$ more queries.

The method generalizes across dimensions $n\in{512,1024,2048}$ and target activation counts $k^*\in{3,6,12}$, and remains stable under mild departures from binomial assumptions when using empirical slope estimates.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
This paper solves a critical parameter sensitivity problem in Sparse Distributed Memory (SDM). Instead of tuning the ill-conditioned Hamming threshold $T$, the authors propose controlling the activation density $p = k/L$ directly using a closed-loop feedback controller. This method combines an inverse-CDF mapping with slope-normalized integral feedback to dynamically adjust T, achieving precise target tracking with high efficiency.

### Strengths
1. The paper's main strength is its novel and rigorous application of control theory to a known problem in associative memories. It correctly identifies the high plant gain as the source of instability and designs a principled controller to neutralize it.

2. The paper is extremely well-written, using clear explanations and insightful figures (e.g., Figure 1) to make a complex technical problem immediately understandable.

3. The experiments convincingly support the claims. The method is shown to be not only highly accurate ( $R^2 \approx 1.0$ ) but also more cost-efficient than relevant baselines like bisection search, which requires $\sim 1.9 \mathrm{x}$ more queries for similar performance.

### Weaknesses
1. Limited Scope Beyond Classical SDM: The experiments are confined to the classical SDM model. While the authors suggest broader relevance to modern architectures like MoE, the paper lacks even a small-scale experiment to demonstrate this transferability.
2. Idealized Data Assumptions: The controller design assumes the data is uniformly random, leading to a binomial distribution of Hamming distances. The paper doesn't address how performance would be affected by structured or correlated data that violates this assumption.

### Questions
1. Could you elaborate on the practical challenges of integrating this closed-loop controller into a modern MoE layer to replace a standard top-k gating function?

2. How robust is the controller if the true distribution of Hamming distances deviates significantly from the assumed binomial model due to structured data?

### Soundness
2

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
This paper tackles the critical parameter sensitivity of the activation threshold `T` in Sparse Distributed Memory (SDM), a long-standing ill-conditioned problem. The authors propose a principled closed-loop control system that directly regulates the activation density `p=k/L`. The core contribution is a slope-normalized feedback controller which effectively cancels the system's extreme sensitivity. Experiments show this method dramatically improves stability and accuracy, significantly outperforming naive control and bisection search in terms of both final error and query efficiency.

### Strengths
*   The paper's key strength is its novel formulation of the SDM threshold tuning issue as a formal control problem. The proposed solution, a slope-normalized feedback controller, is principled, theoretically sound, and a creative application of control theory.
*  The work is exceptionally clear, with rigorous experiments against strong baselines (e.g., bisection search) that convincingly demonstrate its superiority. Its significance extends beyond SDM, offering a promising paradigm for dynamic sparsity control in modern architectures like Mixture-of-Experts (MoE).

### Weaknesses
*    The method's performance depends on a pre-specified target activation count `k*`. The framework does not currently address how this target could be adapted dynamically, which might be necessary in scenarios where the optimal sparsity level changes with memory load or task demands.
*    While the proposed application to modern architectures like MoE is exciting, this claim is not yet substantiated. The paper lacks a discussion of the technical challenges in adapting the method from the discrete, binomial world of SDM to the continuous, data-dependent distributions found in attention mechanism.

### Questions
1.  How does the controller perform during transient periods if the target `k*` is changed dynamically during operation?
2.  Could you elaborate on the main technical challenges in adapting this control method to an MoE layer, particularly regarding the estimation of the activation function and its slope for normalization, given the complex, non-binomial distributions involved?

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
3

### Summary
The paper looks at a long-standing problem in Kanerva’s Sparse Distributed Memory: picking the distance threshold that decides how many memory locations light up during recall. Around the usual operating point, tiny tweaks to this threshold can flip thousands of activations, which makes the system brittle. The authors’ fix is to control the fraction of locations that activate directly, rather than tuning the threshold by hand. They first choose a threshold that should yield a target activation fraction based on a simple probabilistic model, then wrap this with a feedback controller that keeps the actual activations locked to that target despite randomness. They also argue the idea should transfer to modern architectures—like attention and Mixture-of-Experts—where keeping sparsity stable matters for speed, stability, and accuracy.

### Strengths
Importance. I like that you go after a core stability knob in SDM and connect it to attention/MoE sparsity control, it seems broadly relevant beyond classical associative memory. 

Novelty. It seems the combination of inverse-CDF actuation with slope-normalized integral feedback to regulate p is new. 

Clarity/organization. The paper has a reasonably clear presentation.

### Weaknesses
Your claim that you’re more cost-efficient than bisection hinges on the assumed “query cost” model; can you report results under alternative costings (e.g., wall-clock, oracle calls, or amortized per-step overhead) and include confidence intervals/seed-wise tests to show the ~1.9× advantage holds statistically? 

Budget-capped bisection shows 30% failures—can you analyze why (e.g., initial bracket miss, stochasticity) and add a variant with smarter bracketing to ensure the comparison isn’t penalizing a fixable implementation choice?

Missing citations. The controller is framed via slope normalization and feedback linearization; could you broaden the control-theory context (e.g., stochastic integral control / adaptive gain scheduling) and also cite more recent sparsity-control mechanisms in attention/MoE beyond the classics you already reference?

One related work but not cited: Fuzzy Tiling Activations: A Simple Approach to Learning Sparse Representations Online by Pan et al. It introduces a differentiable alternative to hard bin/threshold schemes that stabilizes sparse representations and is robust under shift; your work tackles a closely related brittleness (many locations flipping around a threshold) but solves it via closed-loop activation-density control.

The linearized analysis yields a practical envelope but leaves measurement noise and discretization effects to experiments; can you bound steady-state error and give a robustness margin (e.g., input noise variance → tracking error) so readers know when guarantees degrade? 

Your theoretical results rely on i.i.d. random addresses/queries (binomial geometry); how does stability and tracking behave with structured addresses or correlated query streams (e.g., clustered memories), and can you extend the theory (or add stress tests) for non-binomial regimes?

Since you motivate applicability to attention/MoE, can you add a small demo (e.g., regulating top-k keys or experts on a toy transformer/MoE) to show that the controller slots in cleanly and preserves accuracy under a fixed activation budget? 

You report tail error (post warm-up) and cost-weighted error; can you justify the warm-up choice, show sensitivity to the window, and test robustness to changing the query stream (not just fixing it per trial) so conclusions don’t hinge on one trajectory?

### Questions
see above

### Soundness
3

### Presentation
3

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
The paper presents AHSE, a dual-branch architecture that combines serial feature evolution (SEED) with parallel error-correcting output codes (PATH). It develops a SPOT fusion circuit to integrate these branches and provides theoretical results on ECOC robustness for Broad Learning Systems under weight noise. The model achieves strong results across multiple datasets and includes detailed ablations validating each design choice.

### Strengths
• The paper provides a principled control-theoretic formulation of a long-standing heuristic problem in associative memory, replacing brittle threshold tuning with activation-density feedback.
• The inverse-CDF actuation with slope normalization is a mathematically elegant and computationally efficient solution to the binomial sensitivity issue.
• Empirical results across multiple settings (see Figures 2–5, pages 6–8) demonstrate remarkably consistent tracking and stability, with clear quantitative improvements in cost-efficiency over baselines.
• The work establishes theoretical stability bounds (Eq. 6) and validates them experimentally, offering both analytical and practical insights rarely found in SDM literature.
• The discussion (page 9) thoughtfully extends the concept of activation-density control to attention mechanisms and sparse expert models, positioning the work as a conceptual bridge between symbolic memory and modern deep learning.

### Weaknesses
• The experiments are primarily synthetic, focusing only on canonical SDM configurations. No demonstrations on downstream learning or associative recall tasks are provided to illustrate the real-world impact of improved control.
• The controller is currently fixed-target—it maintains a single desired density P; adaptive or context-dependent sparsity targets could make the approach more versatile.
• The linearized analysis (Eq. 5–6) is elegant but approximate; stability predictions deviate from measured limits (see Figure 5), and the paper could discuss these discrepancies more deeply.
• The study would benefit from additional comparisons with modern sparse-gating or attention mechanisms to substantiate claims of broader applicability.
• Finally, while the system generalizes across dimensions, it is unclear how it scales when embedded in neural or hybrid architectures, where feedback latency and stochasticity differ from SDM assumptions.

### Questions
Could the authors extend their analysis to demonstrate recall accuracy or capacity improvements in SDM tasks when activation-density control is used, beyond query efficiency?

Have you experimented with adaptive or learned target densities P* that adjust dynamically based on load or retrieval error? If so, how does the control behave?

The slope-normalized controller relies on binomial statistics; how robust is it if the address distribution deviates from uniformity (e.g., correlated memory locations)?

### Soundness
3

### Presentation
3

### Contribution
3
