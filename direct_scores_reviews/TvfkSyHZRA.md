## Summary

This paper proposes that grokking (delayed generalization after prolonged overfitting) is primarily driven by two linked mechanisms: *Softmax Collapse* (SC), a floating-point absorption error in the Softmax that zeros out gradients from correctly-classified samples when logits grow too large, and *Naïve Loss Minimization* (NLM), the tendency of gradients to align with the weight-scaling direction after overfitting, which continuously inflates logits until SC occurs. To validate these hypotheses, the authors introduce *StableMax*, a piecewise-linear replacement for Softmax that avoids absorption errors and enables grokking without regularization (converting complete overfitting into grokking), and *⊥Grad*, an optimizer that projects out the NLM component of the gradient to achieve rapid generalization without the characteristic delay (converting grokking into fast generalization). The paper further shows that these mechanisms explain the efficacy of existing grokking-inducing methods such as weight decay and MSE loss.

---

## Strengths

- **Specific, testable causal hypothesis about floating-point errors.** The core claim — that SC is a *numerical*, not merely a statistical, phenomenon — is tested with a particularly clean causal intervention: training with float16/float32/float64 and observing that SC onset shifts proportionally with precision (Fig. 2). This is direct evidence rather than correlation, and represents a perspective on grokking largely absent in prior work focused on implicit bias and circuit formation.

- **Interventions serve as causal probes.** StableMax (converting complete overfitting to grokking, Fig. 4) and ⊥Grad (converting grokking to fast generalization, Fig. 6) are not merely useful tools — they are causal demonstrations of the hypothesized mechanisms. Each reverses exactly the expected transition in Fig. 1. The fact that StableMax induces grokking with *increasing* weight norms (Fig. 4, middle) is a notable result that directly disentangles weight norm from the necessary condition for grokking, challenging prior accounts.

- **Unified explanation for disparate prior observations.** Section 5.2 provides a clean, parsimonious account of why weight decay induces grokking (it opposes the NLM scaling direction), why MSE loss works on shallow networks (logit overshoot prevents indefinite NLM), and why input dimensionality matters (low-dimensional inputs prevent easy overfitting). This synthesis goes beyond describing the phenomenon to providing predictive explanations.

- **Empirical evidence of NLM in non-homogeneous architectures.** Fig. 5 shows gradient-weight cosine similarities approaching 0.9 in the output layers of MLPs *with* bias terms and in transformers, providing empirical grounding for the NLM concept beyond the theoretically clean homogeneous case.

---

## Weaknesses

- **StableMax conflates numerical stability with optimization geometry change.** Proposition 1 shows that StableMax is equivalent to Softmax with log-compressed inputs, meaning it defines a *different loss function*, not merely a numerically repaired version of the original. The success of StableMax could partly arise from the modified gradient geometry — specifically, the piecewise-linear tail imposes a fundamentally different margin structure than the exponential. The paper does not disentangle these effects. A comparison with float64 Softmax on the same tasks (especially the 40% dataset size shown in Fig. 2a) is notably absent: if float64 alone achieves grokking in that setting, it would confirm the purely numerical story; if not, the loss-geometry change in StableMax is doing additional work. This matters for the core claim.

- **The gap between NLM theory and ⊥Grad application in non-homogeneous models.** The theoretical motivation for ⊥Grad relies on the homogeneity property (Def. 6) under which $\alpha\theta$ is a provable NLM direction. However, ⊥Grad is applied to models with bias terms (Fig. 6b) and transformers (Fig. 6a), which are non-homogeneous. The paper acknowledges this and provides empirical evidence of gradient alignment (Fig. 5), but the projection is still applied to the *entire concatenated weight vector* $\theta$ globally. Whether this global projection removes the correct component in non-homogeneous models is not theoretically characterized. The limitations section mentions quasi-homogeneity but does not close this gap. If ⊥Grad works for the wrong reasons in non-homogeneous settings, the mechanistic interpretation weakens.

- **The paper explains the delay and termination of grokking but not its abruptness.** A defining feature of grokking is the *sudden* transition from near-chance to near-perfect test accuracy. The paper provides an account of why generalization is delayed (NLM dominates gradient) and why it eventually halts (SC), but not why, when generalization does occur, it happens over a very short window. The limitations section acknowledges this gap with a brief mention but it represents a substantial incompleteness in the mechanistic account, particularly for an ICLR audience where prior work has extensively studied the transition dynamics.

- **Narrow experimental scope for the strength of the claims.** The core results are demonstrated on modular arithmetic (addition/subtraction/product mod 113) with a 2-layer MLP, with transformer results only for subtraction mod 113 in Fig. 6a. Sparse parity and a tiny MNIST subset appear in supporting roles. The introduction explicitly invokes grokking in "vision and language" settings, but no evidence is provided that SC or NLM manifests in those settings. The causal claims (SC stops grokking; NLM causes SC) are presented as general explanations yet are validated on a narrow slice of the phenomena.

- **StableMax requires extremely long training (60–80k epochs) in Fig. 4 left.** The delay in grokking with StableMax, while avoiding complete overfitting, is still very substantial. The paper does not discuss what dynamics govern this remaining delay after SC is prevented. If NLM continues under StableMax (the weight norm in Fig. 4 middle continues rising substantially), then a large portion of the NLM dynamics persist even without SC, raising the question of whether StableMax's mechanism is distinct from simply tolerating the NLM-induced logit growth for longer.

---

## Nice-to-Haves

- **Layer-wise projection in ⊥Grad.** The current global projection (Eq. 12) over the entire concatenated weight vector is extremely mild for large models and does not correspond to the per-layer NLM directions. A layer-wise variant — projecting out the $\alpha W_\ell$ direction for each layer $\ell$ — would more directly align with the homogeneity argument and potentially be both more principled and more scalable.

- **Explicit analysis of weight norm behavior under ⊥Grad.** Since $\nabla_\perp \mathcal{L}$ has no component in the $\theta$ direction, ⊥Grad approximately preserves the weight norm (to $O(\eta^2)$). This connects ⊥Grad to normalized gradient descent or Riemannian gradient descent on a sphere; making this connection explicit would situate the method in a known literature and clarify its implicit regularization properties.

- **Ablation on alternative absorption-resistant functions.** The specific functional form of StableMax ($s(x) = x+1$ for $x \geq 0$, $s(x) = 1/(1-x)$ for $x < 0$) is well-motivated informally but not ablated. A comparison with Softplus-based cross-entropy or other linear-tail alternatives would clarify whether the specific design is critical.

- **Validation of ⊥Grad on standard benchmarks.** Showing that ⊥AdamW does not degrade performance on tasks where grokking is not the concern (e.g., CIFAR-10 classification) would strengthen the case that ⊥Grad is a practically safe intervention.

- **Quantitative analysis of the fraction of zero-gradient samples over time.** Visualizing what percentage of training samples yield zero gradient (due to SC) as a function of epoch would provide a more direct and quantifiable measure of SC severity, making the narrative more concrete.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"No comparison between StableMax and ⊥Grad combined" (Harsh Critic):** Figure 7 explicitly shows a combined ⊥SGD + StableMax trajectory (labeled "LSGD + StatMolux" — apparently a rendering artifact). This experiment exists in the paper.

- **"Statistical significance / multiple seeds" (Harsh Critic):** Single-run evaluation is standard practice in the grokking literature and in the broader algorithmic generalization setting. Demanding multi-seed statistics with confidence intervals for this type of controlled mechanistic study would be imposing a norm not standard in this field.

- **"Why wrong-class gradients don't compensate" (Harsh Critic) — partially:** The paper explicitly validates this in Appendix B.1 via a causal intervention (artificially zeroing correct-class gradients replicates the SC-induced plateau). The main text is brief on intuition but the empirical addressal is provided. This could be improved with more intuition in the main text, but it is not an unaddressed concern.

- **"Lack of external related work analysis" (general):** Per review guidelines, missing related works are not flagged as we cannot verify their existence.

---

## Novel Insights

The most genuinely novel insight in this paper — largely absent from prior grokking literature — is that the failure to grok without regularization in standard settings is a *computational* rather than a *statistical* phenomenon: the exponential nature of Softmax, combined with the ease of overfitting in these tasks, guarantees that floating-point arithmetic will eventually collapse the gradient signal before generalization can occur. This reframes regularization's role: weight decay is not primarily reshaping the loss landscape toward simpler circuits (though it may do this too) but is operationally countering the NLM scaling direction that would otherwise drive logits into a numerical dead zone. The causal evidence from precision manipulation is particularly clean and the connection between the known gradient-weight alignment result (Lyu & Li, 2020) and finite-precision failure is a substantive new link. A secondary insight is that grokking is representation-contingent even for the same algorithmic task (Fig. 4, right): changing input representation from one-hot to compact binary removes the overfitting-induced delay entirely, suggesting that what makes a task "a grokking task" is not the algebraic structure but the ease of memorization induced by the input dimensionality.

---

## Suggestions

- Add a float64 experiment on the 40% modular addition setting (currently absent from Fig. 2a, which only shows float16 and float32) to determine whether precision alone can induce grokking in the hardest setting, or whether StableMax's linear tail is doing structural work beyond numerical stabilization.
- Provide a causal experiment that clips logits at a fixed threshold (preventing SC without changing the loss function) to cleanly isolate the numerical-stability component of StableMax's benefit from its loss-geometry change.
- Expand the limitations discussion to explicitly address the sudden-generalization puzzle: the paper explains the *delay* via NLM and the *termination* via SC, but not the *abruptness* of the eventual transition. Even a qualitative hypothesis (e.g., referencing the weight-decay rotational equilibrium of Section 5.2 as a potential sudden-unlock mechanism) would strengthen the mechanistic completeness.
- Consider a brief evaluation of ⊥Grad behavior on weight norm dynamics over training, and whether exploding norms under ⊥Grad (since NLM is removed but no explicit regularization is added) become a concern in longer training runs.

---

**Novelty**: High — the floating-point absorption mechanism is a specific, testable, and largely novel contribution to the grokking literature.

**Technical soundness**: Good — formal definitions are precise, proofs are provided, and the empirical methodology is clean. The main theoretical gap (non-homogeneous case for ⊥Grad) is acknowledged but not resolved.

**Empirical support**: Moderate-to-good — the precision-modulation experiment is a particularly strong causal test, but the experimental scope is narrow (modular arithmetic with one architecture as the primary testbed), and the connection to broader grokking settings claimed in the introduction is asserted rather than demonstrated.

**Significance**: Good — the insights are actionable, the unified explanation is parsimonious, and StableMax/⊥Grad provide new experimental tools for the community. The significance for large-scale practice remains an open question.

**Clarity**: Good — formal definitions are clear, the narrative arc of Fig. 1 is well-constructed, though some figures (e.g., Fig. 2's ambiguous vertical dashed line, Fig. 7's abbreviation artifacts) could be improved.

MY FINAL SCORE: <pineapple>6.8</pineapple>