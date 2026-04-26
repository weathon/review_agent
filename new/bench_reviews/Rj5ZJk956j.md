Now I have a thorough understanding of the paper. Let me synthesize the review.

Key verification of critical claims:

1. **Activation frequency confound**: The paper does acknowledge this in Section 7 ("Note however that activation frequencies do not fully explain their effect, since we found that even their negative gate values are influential"). However, the "outsized influence" claim in the abstract and introduction does NOT control for activation frequency. The ablation compares to "random neurons from the same layers" but not to "most frequently activated neurons from the same layers." This is a real concern. However, the conditional ablation result (case iii: negative gate values) partially addresses it by showing effects beyond what activation frequency alone would predict.

2. **Single model behavioral validation**: Confirmed. Section 6 explicitly states they focus on OLMo-7B for behavioral experiments. The weight analysis covers 12 models.

3. **Negative gate value novelty**: The paper acknowledges Kong et al. (2025) concurrently. The claim "first time" is about observing this specific mechanism in the context of weakening neurons.

4. **Cumulative ablation**: The paper doesn't seem to use cumulative ablation per se (no "cumulative" word found). Looking at Section 6.1, they ablate neurons from different RW classes.

Let me re-read the ablation section more carefully. The figures seem to show ablations "up to layer L" which is cumulative.

Actually, looking at the text again: "For attribute rate, the effect is most visible in layers ≈ 10 and onward" - this suggests they are doing cumulative ablation (ablating all weakening neurons up to layer L). But I don't see explicit description of this. The harsh critic mentions cumulative ablation. Let me check.

The text says "In each run we ablate a number of neurons from a different RW class" — this could mean they ablate all neurons of a given class. The phrase "up to layer L" mentioned by the harsh critic might come from the figures which I can't see. But based on the description of the ablation results showing effects varying by layer, it seems they might be doing cumulative ablation. Regardless, this isn't a major issue.

Now, let me make my assessment and write the review.

**Strengths:**
1. Cross-model universality of the cos(w_in, w_out) pattern (Figure 1a) across 12 models — this is clean, striking, well-supported.
2. The RW taxonomy provides a useful conceptual framework for analyzing gated neurons.
3. The conditional ablation method is a genuine methodological contribution.
4. The identification of the negative-gate-value pathway as functionally significant.

**Weaknesses:**

**Major:**
1. The "outsized influence" claim is confounded by activation frequency. The ablation baseline is "random neurons from the same layers," but weakening neurons activate much more frequently. The paper acknowledges the correlation between cos(w_in, w_out) and activation frequency (Section 7, r ≥ -0.71), and notes that "activation frequencies do not fully explain their effect, since we found that even their negative gate values are influential." However, this partial caveat doesn't fully address the confound: the conditional ablation shows negative gate values contribute, but doesn't establish that weakening neurons have "outsize" influence beyond what their high activation frequency would predict. The proper control would be ablating the most frequently activated non-weakening neurons.

2. All behavioral/functional experiments are conducted on a single model (OLMo-7B). The weight-geometry findings extend to 12 models, but the claim about behavioral importance (which is the more consequential claim) rests on one model.

**Minor:**
1. The single case study of a weakening neuron (31.9634) was acknowledged by the authors as "much harder to interpret." More case studies, especially ones where weakening neurons can be clearly interpreted, would strengthen the claim that "weakening" is a coherent functional category.

2. The framing of negative gate values as "surprising" slightly overstates the case, since the non-negligible negative regime of Swish is a known architectural property. However, the functional demonstration is still novel and valuable.

**Trivial:**
- The τ = ±0.5 threshold choice is not deeply justified in the main text.

Now let me evaluate the overall quality. Compared to calibration anchors:

- High anchors (score 6.5-8): Papers like the CRH paper (7.5) showed strong cross-model universality with theoretical backing; NeurFlow (6.5) had novel framework but limited scale.
- Medium anchors (score 5): Pruning paper with cosine similarity (5.0) had simple method but questionable generalizability.
- Low anchors (score 3-4): Papers with overclaiming and confounds such as sJslLVsYNo (3.25) had significant methodological issues.

This paper has a genuinely novel framework (the RW taxonomy), a striking cross-model weight-geometry result, and a genuinely interesting discovery (negative gate values). The main issues are: (1) activation frequency confound in the key behavioral claim, and (2) single-model behavioral validation. These are real limitations but don't invalidate the contribution entirely — the weight-analysis results (which are the main universality claim) are solid across 12 models, and the conditional ablation observation is methodologically novel. The activation frequency confound is partially addressed by the conditional ablation result showing effects from negative gate values (which are rare activations!), though not fully controlled.

I'd put this paper at approximately 5.5-6. The core methodological contribution (RW taxonomy + conditional ablation) is sound and the weight-geometry results are strong and well-validated. The behavioral claims are interesting but have a substantial evidential gap. This aligns with medium-scoring papers in the calibration set that had novel formulations but incomplete controls.

Actually, let me reconsider. The paper has several genuine strengths:
- Novel taxonomy applied to 12 models with consistent findings
- Conditional ablation is a genuine methodological contribution
- The negative gate value discovery is significant for mechanistic interpretability
- The cross-model cos(w_in, w_out) pattern is striking

The main weaknesses:
- The "outsized influence" claim is partially confounded
- Single-model behavioral validation

Given that the weight analysis results across 12 models are the paper's strongest contribution, and the behavioral findings are supportive but not as well-controlled, I think this is in the 5-6 range. The activation frequency confound is a real concern but the conditional ablation result provides *some* evidence (case iii shows effects from infrequent activations). The single-model issue is a limitation but the weight patterns are cross-model.

I'll score this at 5.5 — borderline, leaning towards accept, because the taxonomy and cross-model weight findings are solid contributions even if the behavioral claims are somewhat overclaimed.

Wait, let me reconsider more carefully. The paper's core narrative is about (1) discovering weakening neurons and (2) establishing they have outsize influence. If (2) is significantly weakened by the activation frequency confound, that reduces the paper's impact. But (1) the discovery, taxonomy, and cross-model validation is still strong. And the conditional ablation finding about negative gate values is a separate genuine contribution.

On reflection, I think 5.5 is fair. The paper makes a real contribution with the RW taxonomy and the cross-model pattern, but the behavioral claims need better controls. The conditional ablation methodology is a useful tool for the community.

Let me finalize.</think>Now I have enough information to write the final review. Let me consolidate my findings.

## Summary

This paper introduces a read-write (RW) taxonomy for gated neurons in transformers (e.g., SwiGLU/GeGLU) based on cosine similarities between weight vectors (w_in, w_out, w_gate). Applying this taxonomy across 12 LLMs, the authors discover that "weakening" neurons (where cos(w_in, w_out) ≈ -1) are a small but potentially influential class concentrated in late layers, with "conditional strengthening" neurons dominating early-middle layers. Ablation experiments on OLMo-7B suggest weakening neurons have a disproportionate effect on attribute rate and entropy, and a novel "conditional ablation" method reveals that negative Swish gate values—a previously overlooked regime—contribute substantially to this effect.

## Strengths

- **Cross-model universality of the layer-wise cos(w_in, w_out) pattern**: The median cos(w_in, w_out) transitioning from positive to negative across layers is demonstrated consistently across 12 models spanning 6 families (Figure 1a). This is the paper's most convincing result — simple to compute, clearly replicable, and genuinely striking.

- **The RW taxonomy provides a useful conceptual framework**: The classification of gated neurons into strengthening, weakening, conditional, proportional change, and orthogonal output categories based on weight cosine similarities is intuitive, well-defined, and goes beyond prior work that analyzed only activations or only output weights in isolation (Table 1).

- **Conditional ablation is a genuine methodological contribution**: The four-condition ablation scheme based on signs of x_gate and x_in (Section 6.2) allows attributing behavioral effects to specific activation pathways within a neuron. This is what enabled the discovery that negative gate values are functionally significant, which standard ablation would have masked entirely.

- **Negative gate value discovery**: The finding that case (iii) activations (x_gate < 0, x_in < 0) are responsible for much of the entropy-sharpening effect of weakening neurons is novel and non-trivially interesting for mechanistic interpretability, as it demonstrates that Swish's negative regime serves a functional role beyond training dynamics. The paper acknowledges the concurrent work of Kong et al. (2025).

## Weaknesses

### Fatal

None.

### Major

- **The "outsized influence" claim is confounded by activation frequency.** The paper's central behavioral claim rests on ablations comparing weakening neurons to "random neurons from the same layers" (Section 6.1). However, Section 7 establishes that weakening neurons activate far more often than other classes, with correlations between cos(w_in, w_out) and activation frequency of at least -0.71. Random same-layer neurons will include many rarely-activated neurons, so of course ablating frequently-activated weakening neurons has a larger effect. The paper partially acknowledges this confound ("Note however that activation frequencies do not fully explain their effect, since we found that even their negative gate values are influential") but does not provide the critical control: ablating the most frequently activated *non-weakening* neurons from the same layers. The conditional ablation result showing negative gate values contribute is intriguing evidence that weakening neurons do something beyond just "fire more often," but it doesn't fully establish "outsize influence" — it shows *one pathway* is important, not that the overall effect exceeds what activation frequency alone would predict. This is a meaningful evidential gap for one of the paper's two headline claims.

- **All behavioral claims are validated on a single model (OLMo-7B).** The weight-geometry findings (Section 5) extend across 12 models, but every behavioral and functional claim — the outsized influence of weakening neurons, the entropy-sharpening effect, the negative gate value mechanism — is demonstrated only on OLMo-7B. The paper explicitly notes this ("to save resources, we focus on a single model"), and while the weight patterns suggest the taxonomy is universal, the behavioral consequences of that taxonomy may be model-specific. The claim of universality for behavioral findings cannot be established from a single model.

### Minor

- **The case study of weakening neuron 31.9634 confirms the difficulty of interpreting weakening neurons.** The authors candidly note that the positive-gate activations of this neuron "do not have an obvious semantic relationship" to its output direction. While the negative-gate regime is partially interpretable (reinforcing "again"), the fact that the primary activation mode is hard to interpret weakens the claim that "weakening" constitutes a coherent functional category rather than a weight-geometry label with heterogeneous behaviors underneath.

- **The framing of negative gate values as "surprising" overstates novelty slightly.** The observation that SwiGLU neurons have functionally significant negative-gate activations is genuinely interesting and well-demonstrated. However, the non-negligible negative regime of Swish is a known architectural property (the whole point of Swish over ReLU is smooth non-monotonicity), and Kong et al. (2025) concurrently report related findings. The *discovery of functional significance* is the genuine contribution, not the mere existence of negative activations.

### Trivial

None.

## Nice-to-Haves

- Running ablation experiments on at least one additional model (e.g., Llama-3.1-8B or Gemma-2-2B) would significantly strengthen behavioral claims.
- Adding a "most frequently activated non-weakening neurons" ablation control would cleanly address the activation frequency confound.
- Per-layer or per-neuron ablation effects (rather than cumulative) would improve attribution of effects to specific neurons.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **"All behavioral claims rest on a single model"** — kept in Major because this is a genuine concern for behavioral claims, though the weight analysis across 12 models mitigates it partially.
- **"Cumulative ablation design conflates effects"** — removed as a severity concern. The paper's ablation is class-level (all neurons of a given RW class), not "cumulative up to layer L" in a problematic sense. The effect varying by layer is a natural consequence, not an artifact.
- **"The negative gate value discovery overclaims novelty"** — downgraded from Major to Minor. The discovery *is* novel and interesting; only the "for the first time" / "surprising" framing slightly overstates it given that Swish's negative regime is an architectural given. Kong et al. (2025) is cited concurrently.
- **"Weight preprocessing justification relegated to appendix"** — removed (per rules: appendix references may be stripped by parser; the justification exists in the submission).
- **"τ = ±0.5 threshold not justified"** — removed to Trivial, then removed entirely. The paper discusses random baselines in Section 4.3 and the threshold is consistent with natural clustering in high-dimensional spaces; this is a minor point.
- **Strength Finder claim about "outsized influence despite being numerically rare"** — this strength is partially undermined by the activation frequency confound (weakening neurons are rare in number but activate frequently). Moved to Removed Points since the core claim is contested.
- **"Missing related work"** — removed per rules (no external sources to confirm existence).
- **"Reproducibility concerns about hyperparameters"** — removed per rules.

## Novel Insights

The most novel insight from this work is that gated activation functions like SwiGLU create a rich input-output coupling structure (captured by the RW taxonomy) that is invisible to approaches analyzing only activations or only output projections. The finding that this structure is consistent across 12 models — with a layer-wise strengthening-to-weakening transition — suggests a functional organization principle that emerges from training rather than architecture alone. The negative gate value discovery further suggests that mechanistic interpretability of SwiGLU models requires treating the full Swish function as operative, not reducing it to ReLU-like behavior.

## Suggestions

- Add an ablation of the top-k most frequently activated non-weakening neurons (matched in number and layer distribution to the weakening neurons) to isolate the effect of RW classification from activation frequency.
- Run the conditional ablation experiment on at least one additional model to demonstrate that the negative gate value mechanism is not OLMo-specific.
- Tone down the "outsize influence" language in the abstract and introduction; "disproportionate influence" with a caveat about activation frequency would be more precise.

## Score and Decision

**Evaluation across axes:**

- **Originality**: High. The RW taxonomy for gated neurons and the conditional ablation method are novel contributions. The negative gate value discovery is interesting and timely.
- **Importance of research question**: High. Understanding what gated neurons *do* functionally is important for mechanistic interpretability.
- **Claims well-supported**: Mixed. Weight-geometry claims are well-supported (12 models); behavioral claims are partially confounded and rest on one model.
- **Soundness of experiments**: Weakened by the activation frequency confound and single-model behavioral validation.
- **Clarity**: Good. The writing is clear, figures are informative (even in parsed form), and the taxonomy is well-presented.
- **Value to community**: Solid. The taxonomy and conditional ablation method are useful tools; the cross-model patterns are a valuable empirical finding.

**Calibration anchors:**

- High: NeurFlow (6.5, Accept Poster) — novel framework for neuron groups, good empirical validation. This paper has broader cross-model validation for weight claims but weaker behavioral controls.
- Medium: Cosine similarity pruning paper (5.0, Accept Poster) — simple method, limited controls. This paper is stronger in novelty but similarly limited in controls for behavioral claims.
- Low: Winner-take-most neuron paper (3.25, Reject) — qualitative claims from limited experiments, insufficient quantitative support. This paper is clearly above this level — the cross-model weight analysis and conditional ablation are substantive.

The paper's best contribution (cross-model RW patterns) is solid and replicable. The main behavioral claim has a real but partially addressed confound. The conditional ablation finding about negative gate values is novel and well-evidenced. On balance, this is a solid contribution with a significant but not fatal evidential gap in the "outside influence" claim. I place it slightly above the borderline.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>