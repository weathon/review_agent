=== CALIBRATION EXAMPLE 47 ===

# Final Consolidated Review
Now I have read the full paper. Let me produce the consolidated final review.

---

## Summary

This paper proposes that grokking's failure to occur without regularization is primarily caused by floating-point absorption errors in the Softmax function — termed *Softmax Collapse* (SC) — and that SC is itself driven by a training dynamic called *Naïve Loss Minimization* (NLM), in which gradients after 100% training accuracy become dominated by a component that scales logits without changing predictions, inevitably triggering SC. The authors validate these claims through two interventions: **StableMax**, a piecewise-linear replacement for Softmax that avoids SC and enables grokking without regularization; and **⊥Grad**, a projected-gradient optimizer that removes the NLM component and eliminates the generalization delay altogether. The framework also offers a unified explanation for why existing grokking-inducing methods (weight decay, MSE loss, slingshots) work.

---

## Strengths

- **Concrete, testable mechanistic explanation for grokking failure.** The identification of SC via floating-point absorption errors is not a vague qualitative story but a formally defined, precisely measurable phenomenon (Definitions 1–3, Eq. 2–4). The float16/32/64 precision ablation in Fig. 2 directly confirms that precision limits determine when learning stops, providing unusually clean causal evidence.

- **Strong causal validation via targeted interventions.** Rather than stopping at correlation, the authors provide two independent interventions that directly manipulate the hypothesized causes. StableMax targets the numerical barrier (SC), causing models that completely overfit to grok without any regularization. ⊥Grad targets the upstream cause (NLM), removing the generalization delay altogether. The fact that these mechanistically distinct interventions both produce predicted outcomes substantially strengthens the causal narrative. App. B.1 further provides an explicit intervention confirming that zeroing correct-class gradients reproduces the SC failure mode.

- **Unified explanation for known grokking methods.** The paper synthesizes previously disparate findings — why weight decay induces grokking (it counteracts NLM by pulling weights back along the scaling direction), why MSE loss avoids the problem (logit scaling can *increase* MSE loss, capping NLM), and why slingshots can sometimes induce grokking — into a single mechanistic framework. This explanatory unification is a substantial contribution beyond the individual experiments.

- **The NLM + FP-stability chain correctly identifies what prior implicit-bias accounts miss.** Lyu & Li (2020) and Ji & Telgarsky (2018–2020) proved gradient-weight alignment for homogeneous networks, but did not connect this alignment to the FP precision wall that halts learning. This paper's specific contribution — that gradient alignment under CE loss leads to logit growth that *eventually kills all gradient signal* — is new and consequential.

---

## Weaknesses

### Fatal
None.

### Major

- **StableMax changes the loss geometry, not just numerical stability; the paper does not verify that StableMax-induced grokking produces the same mechanistic solution.** StableMax replaces exponential Softmax with a piecewise-linear ramp, fundamentally altering gradient magnitudes and the optimization landscape. The core claim is that StableMax "removes the numerical barrier" to the same underlying grokking phenomenon — but this is not validated at the mechanistic level. The paper does not compare, e.g., Fourier features or weight structures (as in Nanda et al., 2023) between models grokked via StableMax and those grokked via weight decay. It is possible that StableMax-induced grokking is a different phenomenon that superficially resembles standard grokking by reaching high test accuracy. This directly affects the paper's interpretive claims.

- **No multi-seed statistics.** All presented figures appear to be single runs. Grokking is highly sensitive to random initialization and is notoriously stochastic in its timing. For a paper making strong causal claims about training dynamics, the absence of variance estimates or statistics across seeds is a significant gap — particularly for claims about the *delay* in generalization and the timing of SC onset.

- **⊥Grad projection is underspecified: global vs. per-layer.** Eq. 12 performs a projection using the entire parameter vector $\theta_t \in \mathbb{R}^m$ treated as a single flat vector. For a network with heterogeneous layers (embedding matrices, MLP weights, unembedding matrices), the geometric meaning of a global projection is unclear — the "direction" of a mixed concatenation of embedding and MLP weights has no coherent interpretation. Whether the projection is applied globally or per-layer is not stated, yet this choice substantially affects both the algorithm and its relationship to the NLM theory (which is layer-specific in practice, as Fig. 5 shows stronger alignment in later layers).

- **No systematic quantitative comparison between ⊥Grad and weight decay.** Fig. 6 shows qualitative trajectories, but the paper does not report convergence speed, final test accuracy, or hyperparameter sensitivity across multiple seeds for both methods. Given that ⊥Grad is positioned as a principled, superior alternative to weight decay for grokking tasks, this is necessary for the empirical case to be convincing.

### Minor

- **StableMax design choice is not fully justified against alternatives.** The specific piecewise-linear $s(x) = x+1$ for $x \geq 0$ and $1/(1-x)$ for $x < 0$ is presented with only informal motivation. The paper does not analyze why this particular function is preferred over other absorption-error-resistant choices (e.g., Softplus-based variants). Proposition 1 shows StableMax is equivalent to Softmax after a log-compression, but the properties of this compression (effect on calibration, gradient norms, expressiveness) are not analyzed.

- **The NLM → SC causal link rests on cosine similarity correlation (Fig. 5) without a direct manipulation.** While the paper provides an intervention confirming that SC prevents grokking (App. B.1), the claim that NLM *causes* SC is supported only by the co-occurrence of gradient-weight alignment and logit growth. An explicit intervention — e.g., amplifying the parallel gradient component artificially in a stable setting and showing accelerated SC — would close this gap.

- **StableMax-induced grokking is substantially slower than weight-decay-induced grokking.** Fig. 4 (left) shows generalization at ~40k–80k epochs for StableMax on the 40% split. This is considerably slower than weight-decay-induced grokking in comparable settings. The paper does not discuss this tradeoff, which is relevant to understanding StableMax's practical utility.

- **Computational overhead of ⊥Grad is not discussed.** The projection step in Eq. 12 requires an additional dot product and scaling per iteration. For large-scale models, this is likely negligible, but the paper does not address whether the method integrates cleanly with adaptive moment estimates in Adam (e.g., is the projection applied to raw gradients before or after the Adam preconditioner?).

- **The NLM observation is tightly related to prior work and could be framed more precisely.** The paper cites Lyu & Li (2020) and Ji & Telgarsky (2018–2020) for gradient-weight alignment in homogeneous networks. The distinction between the prior result (alignment proves normalized margin maximization) and the new result (alignment causes logit growth → FP failure) is present but somewhat buried in Section 4.2's discussion. Making this distinction explicit upfront would strengthen the paper's novelty claim.

### Tiny

- Fig. 7's legend contains "LSGD + StatMolux," which appears to be a parsing/transcription artifact that could cause confusion.
- Proposition 2 (⊥Grad is a descent direction) is a standard consequence of projection onto a hyperplane and adds limited theoretical depth; the paper would benefit more from convergence-rate analysis or fixed-point characterization.

---

## Nice-to-Haves

- **Evaluate StableMax on a standard classification task (e.g., CIFAR-10) without grokking.** Without this, readers cannot assess whether StableMax degrades performance or calibration in standard regimes, limiting confidence in it as a general Softmax replacement.

- **Extend experiments to architectures with BatchNorm or LayerNorm.** These normalization layers constrain logit scales and are ubiquitous in modern models. It would be useful to know whether logit growth is inherently bounded in such architectures, and if so, whether SC is still observable.

- **Compare weight norms and solution structure (e.g., Fourier feature analysis) between StableMax-grokked and weight-decay-grokked models.** Even a qualitative comparison would address the open question of whether these methods find the same solution.

- **Provide a float64 vs. StableMax (float32) training time comparison.** The paper uses increased FP precision as an ablation for SC, and StableMax is the proposed practical fix. A direct cost-benefit comparison would clarify why StableMax is the recommended approach.

- **Extend ⊥Grad analysis to include a convergence guarantee or counterexample.** Orthogonal projection can in principle prevent convergence in non-convex geometries; establishing at minimum the conditions under which ⊥Grad converges to a stationary point would increase trust for adoption beyond grokking tasks.

- **Logit magnitude trajectories for all methods.** Plotting max logit per epoch for SGD, weight decay, StableMax, and ⊥Grad in a single figure would provide direct visual evidence for the core claim about uncontrolled logit growth driving SC.

---

## Removed Points

*These points were flagged for removal — treat with caution.*

- **"Binary input experiment is confounded"** (Harsh Critic, Sec. 4.1): The critic argues that changing input dimensionality simultaneously changes memorization capacity, input distribution, expressiveness, and loss geometry. While the confounders are real, the paper's claim is intentionally broad and conceptual — that ease of overfitting (not specific arithmetic structure) is what drives grokking — and the result qualitatively supports this. The experiment is not a precise causal attribution, and the paper doesn't present it as one. Removed as a structural criticism; kept as a minor caveat that the confounders limit the strength of the specific causal claim.

- **"Missing connection to neural collapse"** (Harsh Critic): Interesting academic connection, but neural collapse describes a convergence of last-layer activations to class means under balanced, clean settings — SC is a FP failure mode in logit-growth regimes. The relationship is speculative and not established in the paper. Removed.

- **"NLM direction should be compared against weight normalization as a baseline"** (Harsh Critic): Weight normalization does constrain weight direction updates, but comparing it as an "⊥Grad baseline" requires a full experimental setup. This is a reasonable nice-to-have but not a flaw.

- **"NLM causality intervention needed (no intervention exists)"** (Spark Finder): App. B.1 explicitly provides an intervention (artificially zeroing correct-class gradients reproduces SC's effect). The Spark Finder appears to have missed this appendix. Removed.

- **"Demanding multi-run statistics is standard rigor"**: In general, multi-seed statistics are strongly preferred. However, because the harsh critic also flags this as a weakness (correctly), it is retained as a **Major** weakness rather than removed.

- **"Semantic shift: ⊥Grad doesn't exhibit grokking"** (Positive Reviewer): The paper explicitly frames ⊥Grad as an intervention that *removes the delay* — it never claims ⊥Grad "groks." This is by design and is correctly presented in Figure 1. Not a weakness.

---

## Novel Insights

The most genuinely novel insight in this paper is the identification of a **precision-mediated learning death spiral** in CE-trained neural networks: after 100% training accuracy, gradient updates become dominated by logit-scaling (NLM), which is loss-decreasing but prediction-invariant; this scaling grows logits until floating-point arithmetic silences the gradient signal entirely (SC), permanently halting learning. This mechanism connects three previously separate lines of inquiry — the empirical delay of generalization, the necessity of weight decay for grokking, and the failure of CE+SGD without regularization — through a single mechanistic chain. The secondary insight, that weight decay's role in grokking is primarily *mechanical* (counteracting NLM scaling) rather than purely an implicit bias toward simpler solutions, challenges a dominant narrative in the grokking literature and is supported by the StableMax experiment showing grokking with a growing weight norm.

---

## Suggestions

1. **Report results across multiple seeds** for all main figures (Figs. 2, 4, 6), especially given that grokking timing is highly seed-dependent. At minimum, report variance across 3–5 seeds.
2. **Specify whether ⊥Grad's projection (Eq. 12) is applied globally or per-layer**, and justify the choice. If per-layer, clarify whether embeddings and output matrices are handled separately from MLP weights.
3. **Mechanistically validate StableMax-induced grokking**: compare Fourier feature structure or weight spectra of StableMax-grokked models against weight-decay-grokked models to confirm they find similar solutions.
4. **Explicitly distinguish the NLM-as-observation from NLM-as-new-contribution** in the introduction. Cite Lyu & Li (2020) earlier and state clearly that the novelty is the FP consequence of gradient alignment, not alignment itself.
5. **Add a paragraph on StableMax's behavior under standard (non-grokking) training** — e.g., a note on whether it matches SCE performance on standard CIFAR/language modeling tasks — to clarify its scope as a general-purpose Softmax replacement vs. a grokking-specific tool.
6. **Analyze the interaction of ⊥Grad with Adam's preconditioner**: clarify whether projection is applied before or after Adam's second-moment scaling, as this changes the effective geometry of the update.

---

**Evaluation axes:**

- **Novelty:** High. The SC mechanism is specific, new, and not derivable from prior work on gradient-weight alignment alone. The NLM → SC → training death causal chain is original.
- **Technical soundness:** Moderate-to-high. The formal definitions are careful, Proposition 1 is non-trivial, and the float precision ablation is clean. Gaps remain in the ⊥Grad specification and the theoretical grounding of StableMax's landscape effects.
- **Empirical support:** Moderate. The evidence for SC and its link to grokking failure is strong. The evidence for ⊥Grad as a *better* alternative to weight decay is qualitative and single-run; the StableMax comparison needs mechanistic follow-up. The absence of multi-seed statistics is a meaningful limitation.
- **Significance:** High for the grokking subfield; potentially broader if StableMax generalizes beyond grokking settings.
- **Clarity:** Good overall. The causal structure (NLM → logit growth → SC → no grokking) is clear by the end of Section 4, though it could be surfaced earlier in the introduction. The ⊥Grad specification has an important underspecification around global vs. per-layer projection.

# Actual Human Scores
Individual reviewer scores: [6.0, 8.0, 8.0, 5.0, 8.0]
Average score: 7.0
Binary outcome: Accept
