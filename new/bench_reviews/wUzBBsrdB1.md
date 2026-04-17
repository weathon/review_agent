Now I have enough context to write the final review. Let me synthesize everything carefully.

## Summary

This paper demonstrates that setting the L0 hyperparameter incorrectly in Sparse Autoencoders (SAEs) leads to systematically corrupted features: too-low L0 causes SAEs to mix correlated features (feature hedging), while too-high L0 causes degenerate solutions that also corrupt features. Through toy model experiments with known ground truth, the authors show that the standard sparsity–reconstruction tradeoff evaluation is misleading—MSE loss actively incentivizes feature mixing at low L0, causing a ground-truth SAE to score worse than an incorrect one. They propose a proxy metric, decoder pairwise cosine similarity (c_dec), that is minimized near the correct L0 in toy models and coincides with peak sparse probing performance on Gemma-2-2b and Llama-3.2-1b.

## Strengths

- **Clear, compelling toy model demonstration of feature mixing at incorrect L0.** The experiments in Sections 3.1–3.2 are well-designed and rigorously controlled. Initializing the low-L0 SAE at the ground-truth solution (Figure 2) convincingly shows the mixing is driven by gradient pressure rather than initialization. The finding that low L0 affects nearly every latent while high L0 preserves many correct ones is a nuanced and important asymmetry (Section 3.2).

- **Important critique of sparsity–reconstruction tradeoff evaluation.** The demonstration that a ground-truth SAE achieves *worse* reconstruction than a feature-mixing SAE at low L0 (Figures 4–5) directly challenges a core evaluation paradigm in the SAE literature. This is a significant conceptual contribution: if reconstruction can reward incorrect features, it undermines a widely-used evaluation approach across Cunningham et al. (2024), Gao et al. (2024), and Rajamanoharan et al. (2024).

- **c_dec is simple, intuitive, and scalable.** The decoder pairwise cosine similarity metric has a clear mechanistic motivation—mixed features reduce decoder orthogonality—and requires only the decoder matrix, making it easy to compute and requiring no labels or downstream tasks. Its minimum coincides with the true L0 in toy models (Figure 6) and with peak sparse probing performance in real LLMs (Figure 8).

- **Interesting observations about JumpReLU behavior.** The "stickiness" of L0 under JumpReLU near the correct value (Section 3.6) and the differing high-L0 behavior between JumpReLU and BatchTopK SAEs (Section 4.1) are novel architectural insights that could influence future SAE design choices.

- **Practically relevant findings.** The observation that commonly used SAEs likely have too-low L0 (with most on Neuronpedia having L0 < 100 while the paper's analysis suggests optimal values around 200) is a direct, actionable finding for the interpretability community.

## Weaknesses

### Major:

- **Overclaiming about a single "correct L0" in real LLMs.** The paper's central narrative—that there exists a "true L0" whose violation makes features "incorrect"—is established only in toy models where it is defined by construction (Section 3: "we know how many features are firing on average. We refer to this as the *true L0*"). In real LLMs, the paper shows that extreme L0 values degrade sparse probing performance and increase c_dec, but this establishes only an empirical optimum for a specific metric, not an intrinsic "true L0" of the model. The paper even presents evidence (Section 4.2) that "there is likely a range of L0s where some latents are firing more than they ideally should while other latents are firing less," suggesting the "correct L0" concept itself may be oversimplified. The abstract's claim that "L0 must be set correctly to train SAEs with correct features" leaps beyond the evidence—from toy model ground truth to LLM validation via a single downstream metric.

- **Limited LLM experimental scope for the breadth of claims made.** The paper makes broad claims ("most commonly used SAEs have an L0 that is too low," "L0 must be set correctly") but validates only on BatchTopK SAEs with h=32768 on Gemma-2-2b and Llama-3.2-1b at specific layers, using 500M–1B Pile tokens. This is insufficient to support claims about "most SAEs used by researchers today." The statement about common SAEs having too-low L0 rests on a "cursory search" of Neuronpedia (Section 6), not a systematic survey. The h=32768 width is fixed throughout, entangling L0 effects with width effects.

- **c_dec validation in real models is thin for its role as central contribution.** The paper validates c_dec against sparse probing performance, which measures how useful latent codes are for linear probes—but this is not the same as monosemanticity or interpretability. A representation that mixes semantically related features might still perform well on many probing tasks. No direct interpretability evaluation of features across L0 is provided (e.g., feature-level coherence, causal interventions, or even max-activating examples at different L0s). Furthermore, the authors acknowledge c_dec "can sometimes remain nearly flat for a wide range of L0" (Section 6), and in practice they rely on an informal "elbow" heuristic that is neither formally defined nor evaluated for robustness—making the method not reliably reproducible or automatable.

- **No demonstration of causal link between low L0 and incorrect features in real LLMs.** In toy models, the paper provides a compelling causal story by directly inspecting decoder columns against ground truth. For real LLMs, no individual feature analysis is shown—no examples of specific latents becoming more/less monosemantic as L0 changes, no activation pattern analysis, and no causal intervention evidence. The inference from "toy models + c_dec shape + sparse probing shape → low L0 yields incorrect features" is plausible but not demonstrated. The abstract's claim that "the SAE fails to disentangle the underlying features of the LLM" when L0 is incorrect is not directly supported for real LLMs.

### Minor:

- **Self-referential text error.** Section 3.3 states "As we discussed in Section 3.3, when the L0 of the SAE is lower than optimal, the SAE can find ways to 'cheat' by engaging in feature hedging"—this is a circular reference. The intended reference appears to be Section 3.1 or the feature hedging discussion therein.

- **The "sparsity–reconstruction tradeoff is unsound" framing is somewhat overstated.** The paper demonstrates that reconstruction alone is an *insufficient* proxy for feature quality and can actively reward pathological mixing at low L0. But saying the tradeoff plots are "not a sound method" (Section 1) overreaches—these plots remain informative about coding efficiency and the high-L0 regime. A more precise claim—"sparsity–reconstruction plots are insufficient and can actively reward pathological solutions at low L0"—is what the evidence supports.

## Nice-to-Haves

- Feature-level case studies in real LLMs at different L0 values (e.g., max-activating examples for the same latent at low vs. "correct" L0), which would make the abstract claim of mixing concrete and verifiable for practitioners.
- Ablations over SAE width (h), as all LLM results fix h=32768 and it is unclear how c_dec behavior depends on width.
- Experiments on larger models (7B+), since the claim about "most SAEs used today" directly implicates SAEs on frontier-scale models.
- Formalization of the "elbow" heuristic for identifying the correct L0 from c_dec curves.
- Comparison with alternative L0 selection methods (MDL-SAEs, AFA-SAEs) mentioned as related work.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Unrealistic evaluation setup for the sparsity-reconstruction critique"** (Harsh Critic, Issue 2, part 1): The harsh reviewer claims the paper's comparison between a trained SAE and a ground-truth SAE at the same low L0 is a "straw man" because "practitioners compare learned SAEs at different L0, not 'true feature' SAEs forced into suboptimal sparsity." This misreads the argument. The paper's point is conceptual: if a metric rewards incorrect features over correct ones at the same L0, the metric is misleading about quality. This does not depend on practitioners comparing against ground truth—it shows the metric can systematically point in the wrong direction. The comparison between trained and ground-truth SAEs at the same L0 is the correct way to demonstrate this.

- **"Demand for confidence intervals/statistical tests across seeds"** (Spark reviewer): While multiple seeds would strengthen the paper, SAE training at scale is expensive, and single-run evaluation is standard practice in this field. This is a nice-to-have rather than a core flaw.

- **"Demand for baselines like random decoder or dense autoencoder"** (Human Finder): c_dec is used comparatively across an L0 sweep for the same architecture, not to determine absolute quality. A random decoder baseline would not elucidate whether c_dec identifies the correct L0 within a sweep.

- **"Non-linear features invalidate the framework"** (spark reviewer): The paper explicitly cites Engels et al. (2025) in related work and its framework is predicated on the Linear Representation Hypothesis, which is the standard assumption in SAE research. This is a scope limitation, not a flaw.

- **"Missing related works"** (implicit across reviewers): Per instructions, I do not have external sources to confirm the existence of missing related work.

## Novel Insights

The paper makes a striking observation that has not been clearly articulated in prior work: the sparsity–reconstruction tradeoff is not merely a tradeoff between two desirable properties (sparsity vs. reconstruction)—at low L0, it is a *tradeoff between reconstruction and correctness*. This is a qualitatively different and more concerning failure mode than simply "reconstruction degrades at high sparsity." The paper correctly identifies that this asymmetry (low L0 is worse than high L0 for interpretability, contrary to common intuition) has been systematically under-appreciated. The JumpReLU "stickiness" near the correct L0 is a genuinely novel architectural insight that suggests JumpReLU's per-latent threshold mechanism provides implicit L0 regularization.

## Suggestions

- Reframe the central claim from "there exists a correct L0" to "there exists a range of L0 values that avoid systematic feature corruption, and current SAEs systematically fall below this range." This is both more accurate and more practically useful.
- Add at least one concrete example of a real LLM feature that changes character between low and "correct" L0 values—e.g., showing max-activating examples for a specific latent at L0=30 vs. L0=200. Even one such case study would significantly strengthen the claim that mixing occurs in practice, not just in toy models.
- Formally define the "elbow" detection heuristic for c_dec curves and evaluate its robustness, since this is the practical method readers will attempt to apply.

## Evaluation on Key Axes

**Originality:** Moderate-to-high. The finding that low L0 causes feature mixing through correlation exploitation extends feature hedging (Chanin et al., 2025) specifically to the L0 setting. The critique of sparsity-reconstruction tradeoffs as misleading is novel and important. c_dec is a straightforward metric.

**Importance of research question:** High. L0 selection is a universal hyperparameter decision in SAE training, and showing it has systematic consequences for feature correctness is of direct practical importance.

**Claims well supported:** Partial. Toy model claims are very well supported; LLM claims are only indirectly supported via sparse probing and c_dec, without direct feature-level analysis.

**Soundness of experiments:** Good for toy models; limited for real models (small scale, few layers, fixed width, single evaluation metric).

**Clarity:** Generally clear writing; minor errors (self-referential Section 3.3), but overall well-structured and accessible.

**Value to community:** High for the conceptual/cautionary contribution that low L0 is dangerous; moderate for c_dec as a practical tool given its limitations.

## Score and Decision

**Calibration comparison:**

- "A is for Absorption" (similar: SAE failure mode, ground-truth validation, limited model scope): Scores 6-8 avg ~7.5, but rejected. This paper has a clearer mechanistic explanation but weaker LLM evidence.
- "SAEs Do Not Find Canonical Units" (SAE criticism, practical implications): Scores 6-8 avg ~7, accepted as poster. That paper introduced two novel methods (stitching, meta-SAEs); this paper introduces one metric (c_dec) with significant validation gaps.
- "Incidental Polysemanticity" (toy model results, limited real-world validation): Scores 5-6 avg ~5.7, rejected. Very similar profile to this paper—strong toy model, limited real model evidence.
- "Rethinking Evaluation of SAEs" (SAE evaluation metrics): Scores 5-6 avg ~5.75, accepted as poster. Similar contribution level—new evaluation perspective with moderate empirical backing.
- "Measuring Feature Sparsity" (metric validation issues, limited scope): Scores 3-5 avg ~3.5, rejected. This paper is clearly stronger.

This paper falls between "Incidental Polysemanticity" (rejected at ~5.7) and "SAEs Do Not Find Canonical Units" (accepted at ~7). The core insight is important and the toy model work is strong, but the overclaiming about "correct L0" in real LLMs, thin LLM validation, and inconclusive c_dec behavior in practice are notable weaknesses. It is modestly stronger than "Incidental Polysemanticity" due to the c_dec metric and the sparsity-reconstruction critique, but weaker than "SAEs Do Not Find Canonical Units" which had stronger methodology.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>