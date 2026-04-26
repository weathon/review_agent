Now I have enough information. Let me carefully synthesize the review.

The key issue is: is the comparison unfair? The paper frames its Table 1 comparison as "unconditional trajectory generation" (Section 4.1), and the task definition in Section 2 defines p(x) = Σ p(x|H)p(rc). The TDDM method generates trajectories by first sampling a region, then generating conditioned on the spatial prior H for that region. The baselines are unconditional models that don't use spatial priors. The harsh critic argues this is an unfair comparison because TDDM gets oracle spatial information that baselines don't.

However, I need to think about this carefully. The paper's claim is about a new methodology for trajectory generation that factorizes "where" and "how." This factorization is the core contribution. The comparison is against unconditional models because the paper's point is that unconditional models struggle with distributional coverage and proportionality, and the factorization addresses this. The ablation (Table 2) explicitly shows TDDM without spatial priors has KLsym=1.334 vs Diffusion-TS 1.153 and DiffTraj 1.232. This is important — the unconditional component of TDDM doesn't outperform baselines.

But the paper's abstract and conclusion claim TDDM "consistently outperforms leading baselines" and achieves "4× lower KL divergences." This is true in the aggregate comparison, but it's misleading without acknowledging that the gains come primarily from the spatial prior conditioning. The paper does acknowledge this in the ablation but frames the main comparison as a methodological win rather than a conditioning advantage.

The zero-shot claim: Algorithm 2 line 3 requires H = f(rc, Xtarget), which requires target-city trajectory data to compute the spatial prior. The paper calls this "zero-shot," but it requires aggregated trajectory data from the target city. This is a valid concern — it's not truly zero-shot in the typical ML sense (no data from the target domain at all). However, the paper does explain that H is an aggregate marginal (not individual trajectories), which is a different kind of information input. The paper's narrative about data scarcity is about individual trajectory data, not about aggregate spatial distributions.

Let me now write the final review.</think>Now I have a thorough understanding of the paper and calibration anchors. Let me write the final consolidated review.

## Summary

The paper introduces the Temporal Deaggregation Diffusion Model (TDDM), a hierarchical trajectory generation framework that factorizes generation into spatial priors (marginal occupancy heatmaps over geographic regions) and temporal dynamics (a diffusion model that deaggregates these priors into individual trajectories). Through Procrustes-style canonicalization of geographic regions, the model learns location- and orientation-invariant temporal dynamics that can transfer across cities. Evaluation spans three cities on different continents with six metrics covering fidelity, coverage, and proportionality.

## Strengths

- **Conceptually clean factorization**: Decoupling "where" from "how" in trajectory generation is an intuitive and well-motivated decomposition. The ablation (Table 2) confirms spatial priors are crucial — removing them degrades KLsym from 0.277 to 1.334 (~5×), validating the premise that spatial marginals carry substantial information.

- **Elegant canonicalization design**: The Procrustes-style similarity transform (translation, rotation, scaling into [-1,1]²) enables a single model to handle arbitrary geographic regions without equivariant architectural changes. The city-to-city transfer results (Table 3: Porto-trained model achieves KLsym 0.335 on unseen cities) demonstrate this works in practice.

- **Comprehensive evaluation**: Three datasets spanning different continents and travel modes (pedestrian vs. taxi), six metrics covering distinct quality dimensions (fidelity via TSTR, coverage via KL, proportionality via Density/Trip, structure via Pattern), plus ablations, map-matching validation, and OOD experiments. This is broader than typical trajectory generation papers.

- **Transparent about limitations**: The paper honestly discloses that (1) Length error degrades in OOD settings (0.004→0.060–0.109), (2) 1×1 km regions worsen Length error (0.150), and (3) the model without spatial priors is inferior to baselines on coverage metrics.

- **Interesting empirical finding**: Porto as a "universal source" dataset — a model trained on Porto generalizes better to other cities than one trained on 25% of the target city (KLsym 0.335 vs. 0.545). This is a practically useful insight enabled by the factorization.

## Weaknesses

### Fatal
None.

### Major

- **Misleading framing of Table 1 as a fair methodological comparison**: The paper presents Table 1 as evidence that "TDDM consistently surpasses existing GAN-, VAE-, and diffusion-based methods" (Section 4.1) and claims "4× lower KL divergences." However, TDDM conditions on oracle spatial priors H (computed from training data) while all baselines are unconditional models that receive no such information. The ablation in Table 2 makes clear that the unconditional version of TDDM (KLsym 1.334) actually *underperforms* Diffusion-TS (1.153) and DiffTraj (1.232) on distributional coverage. The large gap in Table 1 is thus largely attributable to the information advantage from spatial priors, not from a superior temporal dynamics model. The paper's contribution is better described as "spatial-temporal factorization + priors" rather than "a better trajectory model." This does not invalidate the method, but the current framing overclaims. The paper should explicitly acknowledge this asymmetry in Table 1 and present the unconditional TDDM result alongside the conditioned baseline results, rather than relegating it to an ablation.

- **"Zero-shot generalization" requires target-city trajectory data**: Algorithm 2 (line 3) computes H = f(rc, Xtarget), meaning the spatial prior for transfer experiments is derived from actual trajectory observations in the target city. While these are aggregated marginals — not individual trajectories — the paper's motivating narrative discusses cities where "existing data cannot be shared or new environments lack sufficient observations" (Section 1). If you already have enough trajectory data to compute a reliable 64×64 marginal heatmap, you likely have enough to fine-tune a baseline model. The paper does not test what happens when H is approximated from non-trajectory sources (e.g., population density, road networks), leaving the practical value of zero-shot transfer under true data scarcity unvalidated. This is a meaningful gap between motivation and evaluation.

### Minor

- **Missing conditioned-baseline comparison**: The paper does not compare against any baseline equipped with the same spatial prior information TDDM uses. Even a simple baseline (e.g., Diffusion-TS + spatial prior concatenated as additional conditioning) would clarify whether TDDM's factorization architecture contributes beyond the information content of H. The lack of this comparison means we cannot determine whether the factorization design itself is important, or whether any model given H would perform comparably.

- **Per-dataset results only in appendix**: Table 1 averages across three heterogeneous datasets (pedestrian Beijing, taxi Porto, taxi San Francisco). Averaging can mask important per-dataset failures. The paper references Table 7 in the appendix, but given that datasets differ in transport mode and geography, key per-dataset results should appear in the main comparison.

- **Tokenization underspecification**: The paper states they "split marginal distribution into several regions, linearly project" following ViT (Section 3), but does not detail the number of patches, patch size, or how 4096 cells map to tokens. Given the quadratic attention cost discussed, this affects both computational assessment and reproducibility.

## Nice-to-Haves

- Approximate spatial priors from non-trajectory sources (e.g., OpenStreetMap road networks, population density) and test whether TDDM still works in genuinely data-scarce settings, validating the motivating use case.
- Compare a baseline diffusion model (e.g., Diffusion-TS) conditioned on the same H to isolate the architectural contribution.
- Visualize TDDM-generated trajectories with and without spatial priors side-by-side to make the prior's contribution visually obvious.

## Removed Points

- **"DiffTraj code is publicly available" — unfair comparison allegation too strong**: The harsh critic claims DiffTraj is publicly available and should have been conditioned. This is a knowledge-gap claim about code availability. The paper states they excluded methods without "reproducible source code" and the rule says we should not question referenced entities' availability. **Kept the weaker version** (lack of conditioned baseline) as a minor weakness rather than a fatal structural flaw.
- **"TSTR scores are too close to differentiate"**: The harsh critic notes TSTR overlaps substantially (0.011±0.006 vs 0.013±0.005). This is valid but the paper does not make strong TSTR claims; the headline wins are on KL/Density/Trip. Not a significant issue.
- **Formatting issues, tokenization reproducibility nitpick**: Removed the harsh critic's stronger reproducibility claims; kept the underspecification as minor only.
- **Missing related works**: Removed per instructions — no external sources to confirm relevance.
- **"Unconditional baselines are unfair comparison" — softened**: The harsh critic frames this as purely unfair. In my assessment, this is better described as a *framing issue* rather than methodologically invalid. TDDM's contribution IS the factorization framework (spatial priors + temporal dynamics); it's reasonable to compare the full pipeline against unconditional baselines, but the paper should not claim the temporal dynamics model itself is superior when the ablation shows otherwise. Kept as Major rather than Fatal.

## Novel Insights

The paper surfaces an underappreciated empirical finding: spatial marginal information alone carries enormous predictive power for trajectory distributions — conditioning on it yields a 5× improvement in KLsym. However, the temporal dynamics model (without the prior) is actually *worse* than existing diffusion baselines. This suggests the main value of TDDM is not in learning better temporal dynamics, but in providing a principled mechanism to inject highly informative spatial statistics into the generation process. The more interesting and underdiscussed question is whether the canonicalization mechanism matters, or whether simply appending a heatmap to any diffusion backbone would achieve similar gains.

## Suggestions

- Reformat Table 1 to include the "w/o spatial prior" ablation result alongside baselines, making the information asymmetry transparent. A simple additional column would suffice.
- Add a row or section demonstrating a Diffusion-TS or DiffTraj model conditioned on H (even naively, e.g., concatenating rasterized H as channels) to isolate the contribution of TDDM's architecture from the information content of H.
- For zero-shot claims, include an experiment where H is approximated from a simple proxy (e.g., uniform distribution weighted by road network density) rather than computed from target-city trajectories, to test robustness under genuine data scarcity.

## Score and Decision

**Calibration anchors used:**

| Paper | Score | Comparison |
|-------|-------|------------|
| Two-Stage Diffusion Models (fH2wf2w2Ss) | 4.0 | Conditional pipeline for unconditional generation; reviewers questioned novelty of conditioning advantage. Overclaims relative to baselines. Similar unconditional-vs-conditional framing issue as TDDM, but TDDM has much stronger empirical results and a genuine methodological contribution (canonicalization). |
| Multi-Scale Wavelet GAN (pXEnurdRAx) | 4.0 | Unfair comparison flagged (extra GAN steps not counted). TDDM's comparison is more transparent (ablation explicitly shows unconditional performance). |
| Matryoshka Diffusion (tOzCcDdH9O) | 6.25 | Hierarchical diffusion with strong results but unfair baseline comparisons flagged. TDDM is comparable in having strong results but a significant framing overclaim. |
| TSFlow (uxVBbSlKQ4) | 6.75 | Time-series flow matching with GP priors. Accepted poster. Has solid results but limited novelty concerns. |
| Trajeglish (Z59Rb5bPPP) | 6.0 | Trajectory generation, accepted poster. Moderate novelty, reasonable baselines, moderate improvement. |
| Zero-Shot Causal Models (x3F8oPxKV2) | 6.25 | Zero-shot overclaim concerns, rejected. TDDM's "zero-shot" claim is weaker than it seems, but the method itself is sound. |

TDDM has a genuine methodological contribution (the spatial-temporal factorization + canonicalization framework), strong empirical results when including the priors, transparent ablations, and an interesting practical finding (Porto as universal source). However, the Major weaknesses — overclaiming the unconditional comparison and the zero-shot framing — are significant. The paper's real contribution is the *framework*, not a better unconditioned trajectory model, and this should be positioned honestly. Compared to papers with similar profiles (overclaim + solid method), this falls in the 5-6 range: the work is above average but the framing issues are real and non-trivial.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>