Now I have all the information needed. Let me compose the final review.

## Summary

The paper investigates why perturbation-based fidelity metrics (AOPC, AD%, IC%) for saliency maps produce inconsistent results across perturbation types. It identifies two assumptions underlying these metrics — [P1] that perturbing any pixel causes a drop in prediction probability, and [P2] that pixel importance rankings are invariant across perturbation types — and proposes two conformity measures (DROP and PSim) to quantify violations. Across 5 models, 3 datasets, 9 perturbation types, and ~75M predictions, the paper finds DROP ≈ 0.5–0.6 and PSim ≈ 0.27–0.64, concluding that fidelity metrics are "inconsistent and unreliable."

## Strengths

- **Important and under-studied question**: Why perturbation-based fidelity metrics disagree across perturbation types is a genuinely important diagnostic question for the XAI community. Moving beyond Tomsett et al.'s (2020) observation to attempt a systematic diagnosis is a worthwhile goal.

- **Substantial empirical scale**: 5 models (3 standard + 2 adversarially trained), 3 datasets, 9 perturbation types, 2 perturbation schemes, and ~75 million model predictions (Section 4). This breadth makes it difficult to dismiss findings as artifacts of a particular setting, and the consistency of low DROP/PSim across all combinations in Tables 1–2 strengthens generality.

- **Practical finding about Gaussian Blur**: Figures 2, 3, and 5 show that Gaussian Blur variants yield higher DROP probabilities and PSim scores than other perturbation types. This gives practitioners concrete, evidence-based guidance for choosing perturbation types — one of the paper's most actionable contributions (Section 5.3).

- **Extension to adversarially trained models**: Table 2 shows adversarial training does not resolve inconsistency (DROP ≈ 0.53–0.58, PSim ≈ 0.18–0.33), addressing the natural objection that more robust models might satisfy the assumptions.

- **Formalization of P2 is insightful**: The observation that pixel importance rankings should be invariant across perturbation types (Equation 5) and the use of RBO to measure rank consistency is a genuinely useful diagnostic idea, even if the current implementation has issues (see Weaknesses).

## Weaknesses

### Fatal

None.

### Major

- **[P1] is formulated too strongly, making DROP a partial straw man**: The paper claims fidelity metrics assume $p_0 > p_i^\phi$ for *all* pixels $i$ and all perturbation types $\phi$ (Equation 2, Section 2.1). But fidelity metrics like AOPC, AD%, and IC% do not require that *every* pixel perturbation causes a probability drop — they measure *aggregate* behavior, comparing what happens when high-saliency vs. low-saliency pixels are removed. Perturbing an unimportant background pixel and seeing no probability change is the *correct* outcome, not a violation. The DROP ≈ 0.5–0.6 finding on random pixels (Table 1) is largely expected: roughly half of randomly selected pixels are unimportant, so perturbing them correctly produces no drop. This undermines one of the two pillars of the paper's argument. A much more informative version would compute DROP specifically for pixels identified as important by saliency maps — if DROP remains low on those pixels, that would be genuinely concerning.

- **PSim computed on 50 random pixels is largely uninformative about fidelity metric reliability**: The paper selects 50 random pixels per image (Section 4.1), arguing that "a subset of a ranked order list maintains ranking." But most randomly selected pixels are unimportant background pixels whose $\delta p$ values are near zero and dominated by noise. When comparing rankings of such pixels across perturbation types, low RBO scores are expected regardless of whether fidelity metrics work correctly — the ranking of near-zero values is arbitrary. Fidelity metrics care about the *relative importance of the most salient pixels*, not whether the 40th and 42nd least important pixels swap ranks across perturbation types. Computing PSim on the top-K most important pixels (per saliency map) would be a meaningful test; the current setup primarily measures noise sensitivity. This undermines the PSim-based claims throughout Sections 5.1 and 5.3.

- **No direct demonstration that low DROP/PSim leads to contradictory fidelity metric outcomes**: The paper's central claim is that fidelity metrics are "inconsistent and unreliable," but it never directly shows this — e.g., it never demonstrates that AOPC with perturbation φ ranks saliency methods differently than AOPC with perturbation ψ. Tomsett et al. (2020) showed metric-level inconsistency directly; this paper replaces that direct evidence with proxy measures (DROP, PSim) whose relationship to actual metric disagreement is assumed rather than demonstrated. Without showing that low DROP/PSim *predicts* fidelity metric disagreement, the paper's core conclusion is unsupported by direct evidence.

### Minor

- **"Theoretical framework" overclaimed**: Section 1.1 claims the paper "theoretically establishes the scenarios under which such assumptions are violated." Section 2.1 provides notation and restates the assumptions formally but proves no theorems, identifies no specific conditions for violation, and provides no theoretical insight beyond what is already implied by the assumptions themselves. The formalization is useful but adds no explanatory power about *when* or *why* violations occur — it merely re-describes the problem.

- **The paper's own data partially contradicts its strong framing**: Gaussian Blur shows relatively high DROP and PSim (Figures 2, 3, 5), and the paper itself notes that "Gaussian Blur was relatively consistent" (Section 5.3). This supports the view that the problem is *perturbation choice*, not a fundamental flaw in fidelity metrics — yet the paper's conclusion emphasizes unreliability. A more measured framing would acknowledge that fidelity metrics can be consistent when perturbation is chosen appropriately, and the contribution is primarily diagnostic (identifying which perturbations are suitable).

- **Transition from Equation 3 to Equation 5 is an unjustified conceptual leap**: Equation 3 states proportionality within one perturbation type ($\delta p_i^\phi < \delta p_j^\phi$ for $i < j$), while Equation 5 demands rank invariance across perturbation types ($rbo(\mathfrak{R}(\phi), \mathfrak{R}(\psi)) \approx 1$). These are distinct properties. A method could satisfy proportionality for each perturbation individually but yield different rankings because different perturbations naturally measure different aspects of model sensitivity (e.g., zeroing a pixel vs. blurring it). This does not make either ranking "wrong" — it means they measure different things. The paper does not justify why rank invariance across perturbation types should be expected.

### Trivial

- Algorithm 1 and Equations 6–7 have minor notational inconsistencies (the algorithm accumulates into $\delta\mathcal{P}$ and takes $\mu$, whereas the equations define DROP as a ratio), but the intended computation is clear enough.

## Nice-to-Haves

- Compute PSim on the top-K most important pixels (per saliency map) rather than random pixels — this would directly test whether the pixels fidelity metrics *actually rank* have consistent importance across perturbation types.
- Directly measure fidelity metric disagreement (e.g., compute AOPC with each perturbation type and show that low PSim correlates with contradictory saliency method rankings). This is the missing link between the proxy measures and the claimed conclusion.
- Analyze *why* Gaussian Blur is more consistent — if the goal is understanding the *origins* of inconsistency, explaining why one perturbation family behaves differently would be more illuminating than simply noting it does.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **Harsh Critic: Algorithm 1 / Equation mismatch as reproducibility issue** — This is a minor notational inconsistency, not a reproducibility problem. The intended computation is clear. Moved to Trivial.
- **Strength Finder: "Conformity measures directly linked to assumptions" as a core strength** — While the measures are linked to the assumptions, the assumptions themselves are problematic (P1 is too strong; see Major weakness). The strength of "direct linkage" is undermined when what they're linked to is a straw man. Moved to Removed.
- **Strength Finder: "Goes beyond observing inconsistency to diagnosing its cause"** — The paper diagnoses the cause as "assumptions are violated," but this is circular: define assumptions → show they're violated → conclude inconsistency. A genuine diagnosis would explain *mechanistically* why violations occur (e.g., how specific perturbation types interact with decision boundaries). The paper shows *that* assumptions fail, not *why* in a causal sense. Moved to Removed.
- **Harsh Critic: DROP indicator function treats zero change as violation for unimportant pixels** — This is subsumed by the Major weakness about P1 being a straw man. The specific point about the indicator function is correct but is an implication of the broader issue already captured.

## Novel Insights

The paper's most valuable insight is not the one it emphasizes. The Gaussian Blur finding — that certain perturbation types yield much higher DROP and PSim scores than others — actually suggests a more nuanced conclusion than "fidelity metrics are unreliable." It suggests that fidelity metrics *can* be reliable when paired with appropriate perturbation types, and the real contribution is a diagnostic framework for *which* perturbation types are suitable for a given model-dataset combination. This reframes the contribution from "fidelity metrics are broken" to "here is how to check whether a perturbation type is appropriate before using a fidelity metric," which is more constructive and better supported by the data.

## Suggestions

- Recompute DROP only for pixels identified as important by saliency maps. If those pixels still show DROP ≈ 0.5, the conclusion would be far stronger and less vulnerable to the straw-man critique.
- Reframe [P1] as an assumption about *important* pixels rather than *all* pixels. The current formulation attacks a claim that fidelity metrics don't actually make.
- Directly demonstrate metric-level disagreement: compute AOPC/AD%/IC% with each perturbation type on the same saliency maps and show that rankings of saliency methods change. This would close the gap between the proxy measures and the claimed conclusion.
- Reframe the paper's thesis from "fidelity metrics are unreliable" to "perturbation choice critically affects fidelity metric reliability, and here is a diagnostic framework for checking suitability." This is both better supported by the data and more constructive.

## Score and Decision

**Calibration anchors used:**

| Anchor | Path | Avg Score | Comparison |
|--------|------|-----------|------------|
| UNI (perturbation baselines for attributions) | PBjCTeDL6o | 8.0 | Much stronger: novel method + real theory + direct demonstrations. Paper under review lacks comparable novelty and direct evidence. |
| Don't trust your eyes (feature viz reliability) | OZWHYyfPwY | 7.0 | Stronger: directly demonstrates unreliability via adversarial circuits + theoretical proofs. Paper under review only shows proxy measures. |
| FEI (hidden-layer faithfulness) | L7jtdGhWzT | 4.67 | Comparable: addresses gaps in faithfulness evaluation but has issues with its own framework. Paper under review has similar pattern — identifies real problem but has structural issues with its diagnostic. |
| AIM (adversarial info masking for EEG XAI) | B5i88Tj1nk | 4.67 | Comparable: identifies gaps in faithfulness frameworks, has empirical contributions, but core framework is weakened by methodological issues. |
| Global-to-Local Support Spectrums | cxB0fPNZkx | 3.0 | Weaker: straw-man criticism with anecdotal evaluation and no baselines. Paper under review is better — has real empirical scale and a genuine question. |
| EXAGREE | wJVZkUOUjh | 2.0 | Much weaker: deeply flawed formulation, unclear problem setup. Paper under review is clearly better. |

The paper addresses a genuinely important question with impressive empirical scale, but its core argument is structurally weakened by the straw-man formulation of P1, the uninformative random-pixel sampling for PSim, and the absence of any direct demonstration that the proxy measures predict actual fidelity metric disagreement. It sits above the low-scoring anchors (which have fundamental formulation problems) but below the medium-scoring accepted papers (which have clearer contributions). It is comparable to the B5i88Tj1nk (AIM) and L7jtdGhWzT (FEI) anchors at ~4.67, both of which identify real gaps in faithfulness evaluation but have issues with their own frameworks. The paper under review has a larger empirical scale than those anchors but also more severe structural issues with its core argument. I place it slightly below those anchors.

MY FINAL SCORE: <pineapple>4.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>