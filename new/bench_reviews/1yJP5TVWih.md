## Summary

The paper studies rank collapse in deep sequence models (transformers and state space models), using a unifying formulation of their layers. It introduces λ‑skip connections—skip connections scaled by a parameter λ—and proves a lower bound on a standard rank‑collapse metric under a λ‑dependent condition, then shows that removing skip connections (or choosing certain λ) can lead to exponential or doubly‑exponential collapse. Experiments on pretrained Mamba‑2 and Albert, plus small‑scale training runs, illustrate how λ, skip connections, LayerNorm, and gating mechanisms affect the empirical rank‑collapse measure.

## Strengths

- **Addresses a timely and important phenomenon.** Rank collapse / oversmoothing in deep sequence models is clearly important for both theory and practice, and has mostly been analyzed in transformers. The paper usefully extends attention‑centric arguments to SSMs and selective SSMs (e.g., Mamba‑style architectures), which are under‑studied from this perspective.
- **General formalism spanning transformers and SSMs.** Writing both attention and (selective) SSM blocks in the shared form \(O^{(k)} = M^{(k)} V^{(k)}\) and then the overall layer as
  \[
  Y^{(k)} = D^{(k)}\big(M^{(k-1)}Y^{(k-1)} C_V^{(k-1)} + \lambda^{(k)} Y^{(k-1)}\big)
  \]
  allows a single proof (Theorem 4.1) to cover transformers, LTI SSMs, and selective SSMs, plus any architecture that fits this template. This is technically neat and conceptually unifying.
- **Nontrivial theoretical extension to SSMs.** Theorem 4.3 (and its doubly‑exponential counterpart in the appendix) adapt rank‑collapse analysis to selective SSMs. Under Assumption 4.1, the paper shows exponential collapse without skip connections and doubly‑exponential collapse when both skip connections and LayerNorm are removed, mirroring known transformer results. This is, as far as the text indicates, genuinely new.
- **Clarifies what skip connections and λ can and cannot guarantee.** Theorem 4.1 proves a λ‑dependent *lower bound* on the collapse metric; §4.2 constructs explicit small systems (Propositions 4.3.1, 4.3.2) where (i) some λ values still yield collapse despite a skip path, and (ii) the lower bound’s decay rate is essentially tight. These examples sharpen intuitions about the limits of what one can prove without stronger assumptions.
- **Empirical evidence for the role of gating and LayerNorm.** Figure 3 shows convincingly that in pretrained Mamba‑2, gating + LayerNorm together keep the rank‑collapse metric stable across depth, while removing either can lead to collapse or instability. This is a useful empirical insight: gating mechanisms, originally motivated by memory, also help avoid collapse in modern SSMs.
- **λ‑skip is easy to implement and has modest empirical impact on performance.** In small‑scale experiments (Table 1), making λ per‑layer learnable does not catastrophically harm performance and sometimes slightly improves it (e.g., MQAR for Mamba and Mamba‑2). This suggests λ‑skip is at least compatible with standard training, and potentially beneficial in some regimes.

## Weaknesses

### Fatal

None reach the level of “this is not a paper” or a clear mathematical error undermining all results. However, there is a serious mismatch between the paper’s framing (“guarantee to prevent rank collapse”, “necessary and sufficient”) and what is actually proved; this substantially weakens the contribution and, in my view, justifies rejection in its current form.

### Major

- **Overstated “prevention” guarantee vs. what Theorem 4.1 actually shows.**  
  The main theorem gives a *lower bound* of the form
  \[
  \mu(Y^{(K)})^2 \;\ge\; a^K \mu(Y^{(0)})^2
  \]
  for some \(a>0\) and \(K\), provided λ satisfies (7) and the initial condition \(\mu(Y^{(0)})^2 \ge b\). As emphasized in Remark 4.1 and the surrounding discussion, for realistic SSMs like Mamba one must take \(a<1\); indeed the remark works an example with \(a=0.9999\) and \(K=64\), yielding only a slight decay. This means:
  - The bound is compatible with arbitrarily strong *exponential* decay of the collapse metric; it rules out only *faster than exponential* decay.
  - The paper’s phrasing—“guarantee to prevent rank collapse,” “guarantee that rank collapse does not occur in the finite layers setting”—strongly suggests bounding \(\mu(Y^{(K)})\) away from zero, but that is not what is proved. What is guaranteed is that the *lower bound* does not go below \(a^K \mu(Y^{(0)})^2\), which itself decays with K unless \(a=1\). The “ideal choice is \(a=1\)” is mentioned but there is no exhibited non‑toy architecture where \(a=1\) is achievable with finite λ.
  - Empirically, the interesting regimes are those where \(\mu\) is roughly constant or even increases with depth for suitable λ (Figures 1 and 2). Theorem 4.1 is far too weak to explain these behaviors; it only shows things could be much worse.
  
  So the core “prevent”/“no rank collapse” rhetoric is stronger than the mathematics supports. This is not a nit about wording: the primary advertised contribution is a *guarantee to prevent rank collapse*, yet the guarantee is compatible with substantial exponential decay of the metric.

- **The λ condition is opaque and not operationally demonstrated.**  
  The sufficient condition in (7),
  \[
  \lambda^2 - a(S C_M + |\lambda|)^2 > 0,
  \]
  together with the definition of \(b\), depends on abstract constants \(C_M = \sup_k\|M^{(k)}\|_F\), \(S = \sup_k\|C_V^{(k)}\|_F\), \(N,d\), K, and a free parameter \(a\). The paper argues that with LayerNorm, \(C_M\) depends only on weights and sequence length, and notes rough bounds in a footnote. But:
  - There is no worked example where \(C_M\), S, and a are concretely estimated for, say, a standard transformer or a realistic Mamba‑2‑like SSM, and where a specific λ from the experiments is shown to satisfy (7) *and* the \(\mu(Y^{(0)})^2 \ge b\) requirement.
  - The dependence of b on \(1/a^K\) means that for large depth K, b can easily outgrow plausible values of \(\mu(Y^{(0)})^2\), making the condition vacuous. The paper hints at picking a close‑to‑one a (e.g. 0.9999 for K=64) but never instantiates the full inequality with realistic constants.
  - Proposition 4.3.2 suggests \(|\lambda| = \Omega\!\left(\frac{a}{1-a}\right)\) may be required to achieve a given a. For a near 1 this grows large; yet the experiments show that \(|\lambda|\) in the range 5–20 already ensures high μ in practice for Mamba‑2 and Albert. This sizable gap between theory and empirically sufficient λ values is acknowledged (“our condition… is too conservative”) but never analyzed or reconciled.
  
  Overall, while the theorem is mathematically correct as stated, the condition on λ is so loose and abstract that it does not deliver a genuinely actionable design rule at the granularity suggested by the narrative.

- **“Necessary” role of λ‑skip is not actually established.**  
  Section 4.2 is titled “Lambda‑skip connection: necessary to prevent rank collapse?” and the abstract and introduction speak about lambda‑skip connections as “necessary and sufficient” (at least by strong implication). But the paper carefully notes, at the start of §4.2, that no formal necessary condition is provided; instead:
  - §4.2.1 revisits cases **without** skip connections, showing exponential/doubly‑exponential collapse for transformers and selective SSMs. This does support the well‑known fact that some residual/gated path is important but does not show a *specific* λ range is necessary.
  - §4.2.2 gives toy 2×2 examples where rank collapse happens for some λ and not for others, under specially constructed \(M\) and \(Y^{(0)}\). These illustrate that λ can flip a system between collapse and non‑collapse but do not generalize to realistic high‑dimensional architectures.
  - Empirically, Figure 3 shows gating + LayerNorm (with *no* λ‑skip modification) already prevent rank collapse quite effectively in pretrained Mamba‑2.
  
  Thus, what is supported is: (i) skip/gating paths matter; (ii) badly chosen λ can still lead to collapse; and (iii) for at least one small system, the lower bound is tight. That is substantially weaker than a statement that some λ‑skip structure is “necessary” in general. The current framing oversells this aspect.

- **Theory–experiment disconnect: experiments do not test the theorem’s quantitative content.**  
  The experiments are informative about qualitative behavior (e.g., λ=0 is bad, gating+LayerNorm is good), but they do not test the central theorem quantitatively:
  - For Mamba‑2 and Albert in §5.1, λ is varied at inference time on *pretrained* models that were trained without λ‑skip (and with gating in Mamba‑2, which is then removed). Such “Frankenstein” architectures may exhibit interesting μ curves, but there is no attempt to check if the used λ values actually satisfy (7) with any reasonable constants.
  - For the models trained with learnable λ (Table 1), the paper does not report the resulting λ distributions, nor does it measure μ across layers in these trained models. We only see task accuracy, which is mostly flat or slightly mixed relative to λ=1. Thus we cannot tell whether learned λ lands in any theoretically “safe” region or whether it correlates with improved μ.
  - Nowhere are the theoretical constants \(C_M, S, a, b\) instantiated to show that Theorem 4.1’s lower bound is even remotely tight for the architectures probed in Figures 1–3.
  
  As a result, the experiments and the theorem mostly support each other at a qualitative, “skip strengths and gating matter for rank collapse” level, but the claimed precise λ‑dependent guarantee is not empirically exercised.

- **Selective SSM theory relies on restrictive assumptions and simplified LayerNorm.**  
  The selective SSM result (Theorem 4.3) assumes \(A_t = \alpha I\) with α independent of input (Assumption 4.1), and subsequently treats Mamba‑style input dependence and gating only in experiments. The paper is transparent about this, but:
  - Input‑dependent A is precisely the novel ingredient of selective SSMs like Mamba; bypassing it in the main theorem means the strongest practical setting is not fully covered.
  - LayerNorm is modeled without the mean‑subtraction (“shifting”) component (Eq. 4). This simplification follows related theory papers but may affect the dynamics. The paper does not argue in detail why the omission is benign beyond citing related work.
  
  Together, these choices limit how directly one can port the theoretical claims to realistic SSM architectures as implemented.

### Minor

- **Unclear distinction between “no collapse” vs. “slowed collapse”.**  
  Throughout (§1, contributions, §4) the text uses phrases like “prevent rank collapse”, “guarantee that rank collapse does not occur”, sometimes with a brief caveat (“finite layers setting”). A more precise phrasing would distinguish:
  - Slowing collapse (e.g., avoiding double‑exponential and getting exponential with controllable rate); versus
  - Ruling out collapse altogether (maintaining μ above a constant independent of K).
  
  The current language risks confusing these distinct notions.

- **Role of other components (MLPs) only qualitatively discussed.**  
  The paper notes (end of §3.1) that MLPs may improve rank behavior via Lipschitz constants but does not attempt even a coarse bound in the main text. This is fine scope‑wise, but given MLPs are present in all practical architectures, some more explicit speculation or small experiment might help contextualize the sufficiency of λ‑skip.

- **Some experimental choices deserve more explanation.**  
  For example, in Figure 2 the benefit of negative λ is noted and attributed intuitively to negative feedback, but there is no discussion of whether such λ values are stable under training or standard in practice. Similarly, the fairly low MQAR performance of the Linear Transformer baseline (1.6%) suggests something is off in that setup but this is not discussed.

### Trivial

- Minor redundancy in figure captions (e.g., Figure 1 caption text appears twice) and some notational inconsistencies (e.g., referring to constants as both c and S in Remark 4.1) are present but not scientifically important.
- The definition of “collapse rate a” in Def. 4.1 as the decay rate of a lower bound could be rephrased for clarity.

## Nice-to-Haves

- A simple worked‑out numeric example of (7) for a toy but nontrivial architecture (e.g., a 4‑layer, low‑dimension transformer with concrete weight norms) showing how to compute or bound \(C_M, S\), choose a, and get a usable λ range.
- A plot of μ vs depth for the *trained* var‑λ models in Table 1, alongside histograms of the learned λ per layer, to see whether training naturally picks larger/smaller λ than the default and how that relates to collapse.
- Extending the theoretical treatment even heuristically to include gating—e.g., reinterpreting gates as multiplicative residual strengths and discussing how they interact with λ.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **Claim that the main guarantee is “practically vacuous” because it only rules out doubly‑exponential collapse.**  
  While it is true the lower bound is compatible with exponential decay, calling it vacuous is too strong: establishing any nontrivial lower bound across architectures as varied as transformers, LTI SSMs, and selective SSMs is non‑trivial. The paper itself does not assert it keeps μ strictly away from zero; the overclaim is mainly rhetorical. The valid core concern (overstated “prevention” wording, lack of quantitative tightness) is already captured in the major weaknesses.
- **Suggestion that the unifying framework “does not really capture” selective SSMs because \(M^{(k)}\) depends on \(Y^{(k)}\).**  
  Equation (6) explicitly allows \(M^{(k)}\) to depend on the current layer’s input; the derivations of Theorem 4.1 rely only on operator norms and LayerNorm normalization, not on independence from Y. The paper acknowledges additional assumptions when deriving specific SSM collapse results. The correct criticism is about restrictiveness of Assumption 4.1 and lack of quantitative instantiation, not the basic validity of the framework.
- **Implied criticism that using a simplified LayerNorm renders the results inapplicable.**  
  The simplified LayerNorm (normalization without shift) is a standard technical device in prior rank‑collapse work. While it may change constants, there is no evidence in the text that the qualitative behavior would fundamentally differ with full LayerNorm. The more reasonable and already‑captured concern is that this simplification weakens the directness of applicability and should be more explicitly justified.

## Novel Insights

The paper’s most genuinely new insight, beyond prior transformer‑focused rank‑collapse work, is that similar exponential/doubly‑exponential collapse phenomena arise in selective SSMs when skip connections and LayerNorm are removed, and that gating mechanisms in SSMs play a strong stabilizing role against collapse. This broadens the conceptual link between oversmoothing and architectural design from attention‑based models to modern SSMs and suggests that residual‑ and gate‑like pathways are a common antidote across quite different sequence modeling paradigms.

## Suggestions

- **Reframe the main claim more conservatively.**  
  Refocus the narrative on “slowing and lower‑bounding rank collapse” rather than “preventing” it outright, and remove any implication that λ‑skip is formally necessary. This aligns the rhetoric with what Theorem 4.1 and the examples actually establish.
- **Add at least one concrete instantiation of Theorem 4.1.**  
  For a small transformer or SSM with explicit weight norms and depth, estimate \(C_M, S, a\) and compute a feasible λ range. Even if pessimistic, this would clarify how to use the theorem and highlight where the looseness comes from.
- **Clarify the role of gating vs. λ‑skip in the story.**  
  Given the strong empirical effect of gating in Figure 3, it would help to explicitly say: λ‑skip provides a general, analyzable path to lower‑bounding collapse; gating is another powerful mechanism (not modeled in the theory yet) that empirically helps for SSMs. This makes the paper’s contribution more about a family of stabilizing mechanisms than about λ‑skip alone.
- **Report μ and learned λ for the trained var‑λ models.**  
  Show per‑layer μ curves and learned λ distributions for the models in Table 1. This would directly link the training‑time use of λ‑skip to rank‑collapse behavior and either validate or challenge the theoretical intuitions.
- **Consider tightening the SSM assumptions or adding a second, more realistic theorem.**  
  For example, state a separate result for a more Mamba‑like parameterization (even with weaker constants), or at least discuss how input‑dependent A_t alters the key steps of the proof. This would make the selective SSM contribution more compelling.

Regarding the requested axes:

- **Originality:** Moderate to good. Extending rank‑collapse analysis to SSMs and highlighting gating’s role is new; λ‑skip as a mechanism is related to prior residual‑scaling work.
- **Importance:** Moderate. Rank collapse is important; SSMs are central in current sequence modeling. However, the practical impact is hampered by loose bounds and under‑developed quantitative guidance.
- **Support for claims:** Mixed. The mathematical claims are sound as written; however, the rhetoric overstates what is proved (full prevention vs. slowed collapse), and the experiments do not quantitatively validate the theoretical conditions.
- **Soundness of experiments:** Qualitatively informative but not fully aligned with the theory; some setups (e.g., modified pretrained models) are somewhat artificial for making design claims.
- **Clarity:** Generally clear and well‑organized; the main definitions and theorems are stated cleanly. Some conceptual distinctions (types of “prevention”, meaning of “necessary”) need sharpening.
- **Value to the community:** There is real value in the SSM rank‑collapse analysis and the gating experiments, but the paper in its current framing overpromises on what the λ‑skip theory delivers. With reframed claims and tighter connection between theory and practice, it could become a solid contribution.

## Score and Decision

For calibration, I compared against:

- **“Mind the Gap: a Spectral Analysis of Rank Collapse and Signal Propagation in Transformers”** (`X6xzYP2cMk.md`, reject, scores 5,5,6,3): strong theoretical analysis of transformer rank collapse, but reviewers felt experimental validation and practical impact were limited. This paper is comparable in that its theory is interesting but somewhat idealized and experiments do not fully substantiate the strongest claims.
- **“Setting the Record Straight on Transformer Oversmoothing”** (`OCx7dp58H1.md`, reject, scores 6,6,6,5): also analyzes oversmoothing/rank collapse with residual/normalization variations; reviewers cited concerns about simplified assumptions and limited generality. The current paper is similar in ambition and in the gap between analysis and modern architectures.
- **“Understanding and Mitigating Bottlenecks of State Space Models through the Lens of Recency and Over-smoothing”** (`pymXpl4qvi.md`, accept‑poster, scores 6,6,6,6): SSM‑focused analysis with clearer connection to practical architectures and stronger empirical grounding; this seems somewhat stronger overall than the present submission.
- **“Hyper-Connections”** (`9FqARW7dwB.md`, accept‑poster, scores 5,8,6,6): explores modified residual connections with more convincing empirical validation and well‑tuned claims, again somewhat stronger as a package.
- **“Residual Connections Harm Generative Representation Learning”** (`cxKLRM3KhC.md`, reject, scores 6,5,5,6): also modifies residual structure; reviewers found the idea interesting but insufficiently justified empirically, similar in spirit to some concerns here.

Relative to these anchors, this paper sits between the two oversmoothing‑rank‑collapse rejects and the SSM/Hyper‑Connections accept‑posters: it has genuinely interesting ideas and nontrivial theory, but the overclaiming and theory–practice gap are significant. I would place it around a **5.0**—interesting and potentially publishable with substantial reframing and strengthening, but not yet at accept level for a strong venue.

MY FINAL SCORE: <pineapple>5.0</pineapple>  
MY FINAL DECISION: <orange>Reject</orange>