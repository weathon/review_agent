Now I have all the information needed. Let me compose the final review.

## Summary

This paper studies how incorrect L0 settings in Sparse Autoencoders (SAEs) lead to feature mixing rather than correct feature disentanglement. Using toy models with known ground-truth features, the authors demonstrate that when L0 is too low, SAEs mix correlated features to improve reconstruction (feature hedging), and when L0 is too high, they find degenerate solutions that also mix features. The key result is that ground-truth SAEs with correct features achieve *worse* MSE than trained SAEs with corrupted, polysemantic latents at low L0, directly undermining the sparsity–reconstruction tradeoff evaluation paradigm. The paper also proposes decoder pairwise cosine similarity (c_dec) as a proxy metric to detect incorrect L0, validated in toy models and partially in LLMs.

## Strengths

- **The MSE comparison (Section 3.4) is a genuinely important result.** A ground-truth SAE with correct features achieves MSE of 4.88 vs. 2.73 for a trained SAE with corrupted latents at L0=5, and Figure 4 shows this holds across all L0 values below the true L0. This directly demonstrates that the standard sparsity–reconstruction tradeoff evaluation actively favors incorrect features, with concrete practical implications for how the field evaluates SAEs.

- **The toy model demonstration of low-L0 feature mixing is clean and well-controlled.** Section 3.1 uses ground-truth initialization to rule out local-minima explanations, and the positive/negative correlation inversion (Figures 2–3) cleanly isolates the directionality of feature mixing. This goes beyond showing degradation — it reveals the specific mechanism.

- **The asymmetric effects of too-low vs. too-high L0 (Section 3.2, Figure 1) provide practical nuance.** When L0 is too high, the SAE still learns many correct latents; when L0 is too low, every latent is corrupted. This asymmetry usefully prioritizes which failure mode matters more.

- **The paper identifies a real and underappreciated problem.** The common practice of treating L0 as a free parameter along a tradeoff curve is widespread, and the demonstration that low L0 systematically degrades feature quality addresses a genuine gap in SAE methodology. The JumpReLU "sticking" observation (Section 3.6, Figure 7) — that L0 naturally settles near the correct value across a range of sparsity coefficients — is an interesting finding with practical implications.

## Weaknesses

### Fatal
None.

### Major

- **The c_dec metric does not reliably identify the correct L0 via its minimum in real LLMs, undermining the paper's practical contribution.** For Gemma-2-2b layer 5 (Section 4), the authors acknowledge "a long shallow region with the global minimum actually appearing in that shallow region," forcing reliance on an ad hoc "elbow" heuristic rather than the metric's minimum. The abstract claims c_dec "can help guide the search for the correct L0," but if practitioners must eyeball an elbow rather than use the metric's defined extremum, the metric is not doing the advertised job reliably. The Discussion (Section 6) honestly acknowledges this — "we do not view this as a perfect guide" and "the metric can sometimes remain nearly flat for a wide range of L0" — but the abstract's framing overstates what the evidence supports for real LLMs.

- **The LLM evidence is too thin to support the paper's sweeping field-wide claims.** The abstract states "most commonly used SAEs have an L0 that is too low," but this is supported by: (i) c_dec sweeps on two small models (Gemma-2-2b, Llama-3.2-1b) at 2–3 layers total, (ii) sparse probing on those same layers, and (iii) a "cursory search" of Neuronpedia (Section 6). No systematic survey of existing SAEs is provided. The models are both small (1–2B parameters), and validation at more layers, larger models (where SAEs are most used in practice), and additional downstream evaluations beyond sparse probing would be needed to support general claims about the field's L0 choices.

- **The "true L0" concept is ill-defined for real LLMs, and this tension is not fully resolved.** In toy models, the true L0 is well-defined because the data-generating process has a fixed expected number of active features. In real LLMs, the number of active concepts varies across tokens and contexts. Section 4.2 acknowledges "there is likely a range of L0s where some latents are firing more than they ideally should while other latents are firing less than they ideally should," but this is treated as an observation rather than a fundamental challenge. If there is no single correct L0, then categorical claims like "most commonly used SAEs have an L0 that is too low" (Abstract) and "L0 must be set correctly" (Section 6) are structurally weaker — the problem may be more nuanced than "too low vs. correct."

### Minor

- **The high-L0 feature mixing mechanism is observed but not explained mechanistically.** The low-L0 mechanism (hedging due to insufficient budget) is clearly articulated in Section 3.1. The high-L0 case (Section 3.2) is described as finding "degenerate solutions that mix features" but no mechanistic account is given for *why* too many active latents would cause mixing rather than simply having some latents fire on noise. This asymmetry in explanation quality is notable.

- **The decoder projection histogram analysis (Section 4.2, Figure 9 right) is speculative.** The claim that at L0=750 "some latents become more monosemantic while other latents mix underlying features" is supported only by visual inspection of a histogram shape, not by per-latent analysis that would directly verify this interpretation.

### Trivial
None.

## Nice-to-Haves

- Comparison of c_dec against MDL-SAEs and AFA-SAEs for L0 selection, to assess whether c_dec offers advantages over existing approaches mentioned in Related Work.
- Validation on at least one larger model (e.g., 7–9B parameters) where SAEs are commonly used in practice.
- Investigation of whether the JumpReLU "sticking" phenomenon (Section 3.6) provides a more reliable and practical approach to finding the correct L0 than c_dec, since it requires no sweep and already shows promising behavior.
- Per-latent analysis of the "simultaneously too high and too low" phenomenon in Section 4.2, to move beyond histogram-based speculation.

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Cross-reference error in Section 3.3 ("As we discussed in Section 3.3" — self-reference, should reference Section 3.1):** This is a trivial formatting/parsing issue. The content is still readable and the intended reference is clear from context.

- **c_dec scalability concern (quadratic in h for large SAEs):** This is a generic concern that could apply to many metrics. The paper uses h=32768 which is practical, and the concern is speculative without evidence that it's actually a problem.

- **No error bars across training runs:** Single-run evaluation is standard in the SAE literature for large-scale experiments; demanding variance estimates here is a methodological standard not typical in this field.

- **The ground-truth SAE comparison at L0=5 is "unfair" or "unnatural":** The ground-truth SAE at L0=5 is the natural comparison object — it has correct features but is constrained by the L0 budget. The point of the comparison is precisely that the correct features achieve worse MSE, which is the central finding. Whether this comparison is "natural" is irrelevant to whether it's informative.

- **Missing comparison with MDL-SAEs and AFA-SAEs:** This is a valid nice-to-have but not a major weakness, since c_dec is a diagnostic metric while MDL-SAEs and AFA-SAEs are alternative training approaches with different goals (MDL-SAEs explicitly assume there is no correct L0).

- **Demand for multiple downstream evaluations beyond sparse probing:** Sparse probing is a well-established evaluation in the SAE literature. Requesting additional evaluations is reasonable but not a major weakness given that sparse probing already provides meaningful signal.

- **Missing related works:** Not verifiable; removed per rules.

- **"Section 3.3 and 3.4 content seem partially swapped":** The section titles and content are reasonably aligned. Section 3.3 discusses *why* MSE incentivizes mixing (conceptual argument), while Section 3.4 provides the quantitative demonstration. This is not a content swap but a logical progression.

- **"Training on 500M tokens is modest":** This is a generic criticism; 500M tokens is standard for SAE training experiments in the literature.

## Novel Insights

The most novel insight is the specific directionality of feature mixing at low L0: positively correlated features mix in positive components, while negatively correlated features mix in negative components (Section 3.1, Figures 2–3). This means that low-L0 SAE latents don't just become "noisy" — they become systematically biased, acquiring spurious positive components of correlated features and spurious negative components of anti-correlated features. This has implications beyond evaluation methodology: it suggests that interventions based on low-L0 SAE latents could systematically inject correlated feature components, potentially causing predictable and directional errors rather than random noise.

## Suggestions

- Moderate the abstract's claim from "most commonly used SAEs have an L0 that is too low" to "our evidence suggests commonly used SAEs may have an L0 that is too low," reflecting the limited scope of the LLM validation and the "cursory search" supporting this claim.
- In Section 3.3, fix the self-reference "As we discussed in Section 3.3" to reference Section 3.1 where the low-L0 mixing was first demonstrated.
- Add a brief discussion of when c_dec's minimum is reliable vs. when the elbow heuristic is needed, ideally with a characterization of what data properties predict each case.

## Score and Decision

**Calibration anchors:**

- **High (>7):** 7cMzTpbJHC (Temporal superposition in RNNs, 7.5, Oral) — much deeper theoretical framework with phase transitions and nonlinear extensions. This paper is clearly below that level due to thinner theory and weaker LLM validation.
- **Medium (4–6):** DSOTgzeH3w (SAE theoretical limits, 6.0, Poster) — has closed-form solutions; our paper has a more practical/impactful finding (MSE is misleading) but less theoretical depth. 33wY6AI13k (Amortized inference pathologies, 5.0, Poster) — similar topic (SAE pathologies), our paper has a cleaner core finding but weaker practical contribution. soMC0uESuz (SAE bimodality, 5.0, Reject) — proposes proxy metric with weak validation; our paper is clearly better (cleaner demonstration, more important finding, better LLM evidence). EjInprGpk9 (SAE feature inconsistency, 5.5, Poster) — identifies SAE reliability issues; comparable contribution level.
- **Low (<3):** mqNKv0brqk (CE-Bench, 1.0, Withdrawn) — proxy metrics with no validation linking to interpretability; our paper is clearly above with its toy model grounding and MSE comparison. 9050ZOHwxV (MUI, 2.5, Reject) — proxy metric with unclear theoretical grounding; our paper has much stronger theoretical motivation.

This paper sits above the rejected proxy-metric papers (soMC0uESuz at 5.0, CE-Bench at 1.0) because the core finding (MSE is misleading; ground-truth SAEs are outperformed by corrupted ones) is well-demonstrated and practically important. It sits below the stronger theoretical papers (6.0+) due to limited LLM evidence and a practical metric with reliability issues. The MSE comparison result elevates it above a standard "we found a problem" paper. Relative to the 5.0–5.5 band anchors, the core finding warrants the higher end, but the overclaimed LLM conclusions and unreliable metric pull it back.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>