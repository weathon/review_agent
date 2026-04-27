Now let me do the calibration searches.Now I have enough information to write the final review. Let me synthesize everything.

**Summary of verified facts from the paper:**
1. TDDM conditions on H (spatial prior from target trajectories) — confirmed from Algorithm 2, line 3
2. Table 1 has no conditioned baselines — confirmed: all baselines (TimeGAN, TimeVAE, COSCI-GAN, Diffusion-TS, DiffTraj) are unconditional
3. Table 3 (OOD) contains no external baselines — confirmed: only TDDM variants shown
4. Ablation (Table 2) isolates spatial prior but not canonicalization
5. Algorithm 2 line 9 sign may be parser artifact — per rules, remove
6. Paper says in abstract "enables generation without sample-specific conditions" — technically correct (H is aggregate, not per-sample)
7. Paper uses "zero-shot" in contribution bullet — but requires target-domain trajectory data for H

---

## Summary
TDDM is a diffusion-based trajectory generation framework that factorizes mobility into "where" (spatial occupancy priors H, computed from real trajectories) and "how" (temporal dynamics learned via a transformer denoising model). The method uses similarity-transform canonicalization to enable parameter sharing across geographic regions. Evaluation spans three cities across three continents, with TDDM showing large KL divergence improvements over unconditional baselines and claimed zero-shot cross-city transfer.

---

## Strengths

- **Principled spatial-temporal factorization with formal treatment**: Equations 1–5 formally express TDDM as a mixture model over region priors. The decoupling of "where" (marginal spatial prior H) from "how" (temporal denoising process) is a conceptually clean and principled design choice that gives both controllability and interpretability, with the ablation study in Table 2 providing concrete mechanistic evidence for the spatial prior's role (KL_sym jumps 4.8× when H is removed).

- **Cross-continental multi-city benchmark with complementary metrics**: The three-city evaluation (Beijing walking/cycling, Porto taxis, San Francisco taxis) spans qualitatively different urban environments. The multi-metric suite — TSTR, KL(S‖R), KL(R‖S), JS, Density, Trip, Length, Pattern — captures genuinely distinct quality dimensions (fidelity, support coverage, proportionality, usefulness), and the paper demonstrates how relying on a single metric would mask the Length vs. KL tradeoff visible in the OOD setting.

- **Thorough preprocessing and ablation discipline**: Map matching is applied uniformly to all baselines, and its impact is explicitly ablated (Table 9), providing evidence that TDDM's gains persist without map matching. This level of experimental hygiene is above average for the trajectory generation literature.

- **Interesting cross-city transfer finding**: The observation that Porto-trained TDDM generalizes better to other cities (KL_sym 0.335) than a model trained on 25% of the local target city (KL_sym 0.545) is a substantive empirical finding with practical implications for dataset selection in mobility modeling.

---

## Weaknesses

### Fatal
None.

### Major

- **No spatially-conditioned baseline in Table 1 — the core comparison is not apples-to-apples.** TDDM is conditioned on H (a spatial marginal distribution derived from real training trajectories), while all five baselines (TimeGAN, TimeVAE, COSCI-GAN, Diffusion-TS, DiffTraj) are fully unconditional. The headline result — KL_sym 0.277 vs. 1.153 for Diffusion-TS — measures the improvement from using spatial conditioning, not from TDDM's architectural choices. The ablation in Table 2 confirms this directly: removing H from TDDM causes KL_sym to jump from 0.277 to 1.334, nearly matching the baselines. The paper's framing as "TDDM consistently outperforms leading baselines" conflates the framework's conditioning signal with the model's intrinsic capability. A spatially-conditioned version of Diffusion-TS (receiving the same H token stream via cross-attention or concatenation) would be the necessary comparison to isolate whether TDDM's architecture contributes beyond the information in H. Without it, the central empirical claim cannot be cleanly attributed to the proposed method.

- **No OOD baselines in Table 3 — the cross-region generalization claim is unvalidated against alternatives.** Table 3 reports only TDDM variants (25% vs. 100% local training, and city-to-city). No baseline method appears, making it impossible to determine: (i) whether Diffusion-TS trained on Porto would also generalize to Beijing; (ii) whether the intra-city generalization is simply driven by having H in a geographically contiguous region, regardless of model; (iii) whether the performance levels in Table 3 represent genuine superiority or just floor effects. The claim "TDDM demonstrates robust fidelity and distributional generalization across cities" is unvalidated in the comparative sense.

- **"Zero-shot" framing overstates the practical generalization capability.** Algorithm 2, Line 3 explicitly reads: *"Compute heatmap H = f(r_c, X_target)"* — H is computed from real trajectories in the target domain. The paper's distinction ("the model ε_θ never receives individual target trajectories, only their aggregate spatial distribution") is technically true but practically misleading: collecting enough target-city GPS trajectories to estimate a reliable 64×64 H requires substantial data collection. The contribution bullet ("out-of-distribution zero-shot performance") and the abstract phrase ("supporting transfer to new regions") create the impression of data-free transfer. The paper never demonstrates H estimation from non-trajectory sources (e.g., road networks, population density grids), which would be a genuinely zero-shot scenario and a natural realization of the stated design philosophy.

### Minor

- **Canonicalization contribution is not isolated.** The canonicalization via similarity transform is described as a key contribution enabling generalization across regions and cities, but Table 2 only ablates the spatial prior — there is no experiment comparing TDDM with and without the similarity transform (but otherwise identical). The contribution of canonicalization vs. spatial prior conditioning alone to the OOD results therefore cannot be separated from the paper's current experiments.

- **Variance not reported for most metrics.** Only TSTR reports standard deviations across datasets; all KL, JS, Density, Trip, Length, and Pattern scores in Tables 1 and 3 are single-run point estimates. With only three datasets, it is impossible to assess whether numeric differences in these tables (e.g., Pattern: 0.917 vs. 0.907) are meaningful or within noise.

- **The "Porto as universal source" finding lacks supporting analysis.** The observation that Porto-trained models outperform locally trained models on KL metrics (Section 4.3) is potentially the paper's most practically useful finding, but it is discussed only speculatively. No analysis of road network statistics, trajectory length distributions, or source-target distributional distance is offered to explain or characterize this phenomenon.

### Trivial
None beyond what's already noted.

---

## Nice-to-Haves

- A visualization of KL(S‖R) as a function of how many target trajectories are used to estimate H (e.g., 10, 100, 1000 trajectories) would make the practical data requirement concrete and support the aggregate-conditioning framing.
- Demonstrating H estimation from road network density (e.g., OpenStreetMap) or population grids for a city with no trajectory data would be a compelling validation of the stated design philosophy.
- A per-city breakdown of all Table 1 results in the main body (not only in the appendix) would help readers understand which cities drive the average improvements.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Algorithm 2 sign error (line 9)**: The harsh reviewer noted that `x_{t-1} = (1/√α_t)(x_t + ε_θ(…)) + σ_t z` appears to use addition where DDPM uses subtraction. Per our rules, formatting/parsing artifacts should not be penalized — the sign is most likely a parser artifact from the original PDF rendering. Removed.

- **KL divergence grid circularity concern**: The reviewer noted that if KL is computed on the same 64×64 grid as H, the comparison is circular. However, the paper states KL measures the spatial marginal distribution quality — this is precisely what the model is designed to optimize. The concern would only be circular if the grid used to evaluate KL was *also* used to compute H in a way that creates a direct algebraic link. This is speculative and lacks sufficient evidence to constitute a verified weakness. Removed as unverified.

- **Strength Finder — "canonicalization enables location-invariant learning"**: While the paper describes this design choice, there is no ablation verifying its contribution. The paper's own ablation only isolates H, not the canonicalization. Moved: Cannot be confirmed as a demonstrated strength.

- **Strength Finder — "zero-shot cross-city generalization" as a strength**: Partially contradicted by the verified weakness that H requires target-domain trajectory data. Results are real but the "zero-shot" framing is contested. The cross-city transfer results are genuine, but the zero-shot characterization is misleading. Kept only as a result (cross-city transfer works), not as a methodological claim.

---

## Novel Insights

The most genuinely interesting finding in this paper — one not fully appreciated by either reviewer — is the Porto universality result: a model trained on Porto taxi trajectories outperforms locally-trained models on distributional metrics in other cities, even when local training used 25% of the city data. This suggests that source-dataset selection for mobility modeling may follow a "rich-get-richer" dynamic: cities with high trajectory density and road-network diversity may be disproportionately useful as universal pre-training sources, independent of target-city characteristics. This has direct implications for data-scarce mobility settings (e.g., developing-world cities) where collecting local trajectory data is expensive. Formalizing this through a source-diversity score or road-network coverage metric could be a significant contribution in its own right.

---

## Calibration Anchors

**Retrieved anchors:**

1. `/home/wg25r/review_agent/human_reviews/WeJEidTzff.md` — avg 6.75 — Large-scale urban OD flow generation benchmark with standardized evaluation across 3,333 areas; accepted as poster. This paper makes a dataset+benchmark contribution with comprehensive baselines. TDDM is comparable in benchmark scope but weaker on the comparison rigor (no conditioned baseline, no OOD baselines).

2. `/home/wg25r/review_agent/human_reviews/Pxik3T6Mn9.md` — avg 4.50 — Human mobility modeling with transformers and uncertainty estimation; rejected. Criticized for missing baselines and limited analysis. TDDM has similar issues (missing OOD baselines) but stronger technical contribution.

3. `/home/wg25r/review_agent/human_reviews/HV67MnnXkL.md` — avg 4.00 — Mobility benchmark dataset paper; rejected. Thin methodological contribution, incomplete evaluation. TDDM is more methodologically developed than this paper.

4. `/home/wg25r/review_agent/human_reviews/wM2sfVgMDH.md` — avg 7.50 — Diffusion-based autonomous driving planner; accepted oral. Strong empirical results, comprehensive evaluation, clearly separated contributions. TDDM is weaker because the core comparison is not properly isolated.

5. `/home/wg25r/review_agent/human_reviews/9UGfOJBuL8.md` — avg 7.33 — Conditional diffusion for longitudinal disease modeling; accepted spotlight. Clear conditioned generative model with strong empirical evidence and proper baselines. TDDM has a similar idea (conditioning diffusion on aggregate statistics) but weaker comparative evidence.

**Scoring reasoning:**

The paper has a genuine, principled contribution (spatial-temporal factorization, benchmark, OOD transfer finding) and strong results on the metrics it is evaluated on. However, three substantive issues materially limit confidence in the headline claims: (1) the comparison in Table 1 conflates conditioning signal with architectural contribution; (2) no baselines appear in Table 3; (3) the zero-shot framing is misleading. These are not fatal — the paper does show that conditioning on H dramatically improves distributional alignment, and the benchmark is a real contribution — but they require significant additional work.

The closest analogues are WeJEidTzff.md (6.75, accepted, a benchmark-plus-model paper with some gaps) and Pxik3T6Mn9.md (4.50, rejected, weaker mobility paper with missing baselines). TDDM is stronger than Pxik3T6Mn9 on methodological contribution but has real gaps similar to that paper's missing baselines. Relative to WeJEidTzff, TDDM has a stronger model component but weaker comparison design. I position TDDM slightly below WeJEidTzff because the missing conditioned baseline and OOD baselines are more significant than the issues in WeJEidTzff (which had an anonymity violation but sound methodology).

## Score and Decision

Anchoring: WeJEidTzff (6.75, accepted, similar urban-generation benchmark scope) is the best topical anchor. TDDM falls slightly below due to the missing baselines and the misleading zero-shot framing, which constitute methodological gaps an area chair would require to be addressed. However, the core contribution — conditioning on spatial marginals to improve trajectory generation — is sound, the ablation is convincing, and the benchmark is valuable. This places the paper in the borderline accept/reject range.

**Score: 4.5**
The paper has real contributions and solid empirical results within its framing, but the major weaknesses (no conditioned baseline in Table 1, no baselines in Table 3, overstated zero-shot claim) cannot be addressed in a rebuttal and require additional experiments. In its current form it does not meet the bar for acceptance at a top venue.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>