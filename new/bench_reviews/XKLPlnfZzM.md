Let me carefully verify the harsh critic's most critical point — whether TDDM's comparison with unconditional baselines is unfair — by re-reading the paper's own framing and checking the actual experimental setup.

The paper's abstract says: "We introduce the Temporal Deaggregation Diffusion Model (TDDM), a hierarchical framework that first represents mobility using spatial priors, which are marginal distributions over geographical occupancy, and then deaggregates them into trajectories."

Section 2 defines unconditional generation. Section 4.1 says "The task of unconditional trajectory generation is to learn a generative model which generate a high quality (I)-(V) synthetic dataset of trajectories."

However, TDDM's generation process is conditional: it conditions on spatial priors H. The baselines (TimeGAN, TimeVAE, COSCI-GAN, DiffTraj, Diffusion-TS) are described as unconditional.

The harsh critic raises a critical structural point: TDDM is given spatial occupancy priors H as conditioning information, while baselines receive no such information. This means the comparison is asymmetric. Let me also verify the ablation study claims.

Table 2 shows that removing spatial priors ("w/o spatial prior") leads to KL_sym jumping from 0.277 to 1.334, while TSTR barely changes (0.011 vs 0.011). This confirms the critic's point: the massive KL improvement comes primarily from having H as conditioning, not necessarily from better trajectory modeling.

Now let me assess each of the harsh critic's critical issues:

**Issue 1 (Unconditional vs. conditional mismatch)**: This is a valid concern. TDDM is fundamentally a conditional model p(x|H), while baselines are unconditional. The paper frames Section 4.1 as "unconditional trajectory generation" but TDDM operates conditionally. However, I need to think about whether this comparison is truly unfair. The paper's key contribution is precisely this factorization — giving the model spatial occupancy information and having it generate temporal trajectories. The relevant question is: does this factorization have value? 

The ablation study shows that without H, TDDM still works (TSTR unchanged, KL degrades). But no baseline is given H as conditioning. The point is valid that the comparison is asymmetric. However, the factorization itself IS the contribution — the idea that you CAN condition on spatial marginals to get better generation. The claim about "unconditional" is misleading, but the method itself is not invalid. The comparison does show TDDM with priors outperforms baselines without priors, which is the intended comparison for the proposed approach. But calling it "unconditional" and not acknowledging the asymmetry is problematic.

**Issue 2 (Zero-shot requires target data)**: This is partially valid but partially overblown. Algorithm 2 does compute H from X_target. But the model is never trained on X_target — only the spatial prior H is computed from it. This is analogous to giving the model a map of where people go in a new city without showing it any trajectories. The model still needs to generate realistic temporal dynamics from scratch. The critic says "H is essentially the answer key" — but H only gives marginal occupancy (where people spend time), not how they move through time (the actual trajectories). So H provides significant information about spatial distribution but zero information about temporal dynamics. The question is whether calling this "zero-shot" is appropriate. In many contexts, "zero-shot" means no gradient updates on target data, which is satisfied. But it's misleading if it implies no access to target data at all.

**Issue 3 (KL metrics privilege spatial prior information)**: This is valid. The very metrics where TDDM excels most (KL, JS, Density, Trip) are measuring spatial distribution properties that H explicitly encodes. This doesn't make the results meaningless, but it does mean the gains should be contextualized appropriately.

**Issue 4 (Data split transparency)**: Valid concern about train/test leakage ambiguity.

Now let me also check the Neutral Reviewer and Spark points against the actual paper.

Checking the paper for standard deviations in tables — Table 1 shows TSTR with ±std but other metrics without. This is valid.

Let me now look at the previous version's reviews more carefully.

From the earlier submission (dDdxbdhMsY.md), scores were 3, 6, 6 with rejection. The paper has addressed several weaknesses from the prior review (added more metrics, added ablation study). But the core concern about asymmetric comparison and zero-shot claims appears to be new criticisms that weren't raised before.

Let me finalize my assessment. The paper makes a genuine contribution with the spatial-temporal factorization idea. The method is well-motivated and the canonicalization approach for cross-region transfer is clever. However, the evaluation framing has significant issues:

1. The "unconditional" comparison is asymmetric — TDDM gets H, baselines don't.
2. The "zero-shot" claim requires access to target-city data to compute H.
3. The KL/JS metrics where TDDM excels are essentially measuring what H encodes.

These are substantive but not fatal issues. The core idea is sound; the claims just need to be tempered. The paper still demonstrates:
- A novel and useful factorization approach
- Strong improvement in spatial distribution metrics
- Competitive temporal fidelity (TSTR)
- Some generalization capability (even if not truly zero-shot without any target data)

Compared to calibration papers:
- The prior version scored 3, 6, 6 (rejected)
- DiffMove (trajectory diffusion, similar area) scored 5, 6, 8, 6, 6 (rejected)
- This paper addresses prior weaknesses and has a stronger contribution

I'll give this a score around 5.5-6. The idea is good but claim framing issues are significant.</think>Now I have enough context to produce the final meta-review.

## Summary

TDDM factorizes large-scale trajectory generation into spatial priors (marginal occupancy distributions over regions) and temporal dynamics (a diffusion model that generates trajectories conditioned on those priors). Regions are canonicalized via similarity transforms to enable parameter sharing, supporting both in-distribution generation and cross-city transfer. The paper evaluates on three cities (Beijing, Porto, San Francisco) with a comprehensive metric suite and shows strong distributional alignment and zero-shot generalization performance.

## Strengths

- **Principled spatial–temporal factorization**: Decomposing trajectory generation into "where" (spatial priors H) and "how" (temporal dynamics via diffusion) is conceptually clean and well-motivated. It naturally enables controllability via spatial priors and cross-region transfer via canonicalization, which is a genuine architectural insight.

- **Strong empirical gains in distributional metrics**: TDDM achieves substantial improvements over baselines on KL-sym (0.277 vs. 1.153 for the next-best Diffusion-TS), JS (0.059 vs. 0.198), and Pattern score (0.917 vs. 0.907), with competitive TSTR fidelity (0.011 vs. 0.013). These results are consistent across all three datasets.

- **Meaningful ablation study**: Table 2 cleanly demonstrates the role of spatial priors (removing them degrades KL metrics by ~5× while TSTR stays flat at 0.011), and reveals a genuine tradeoff between region size (1×1 km vs. 3×3 km) in local coherence vs. global realism. This addresses a prior weakness from the earlier submission.

- **Cross-city generalization capability**: The city-to-city transfer experiments (Table 3) show that a model trained on Porto can generate trajectories for unseen cities with Pattern ≥ 0.915 and competitive KL values. The unexpected finding that Porto transfer sometimes outperforms 25% local data is practically interesting.

- **Comprehensive evaluation framework**: Harmonizing five qualities (fidelity, diversity, proportionality, usefulness, generalization) with concrete metrics across three multi-continent datasets is a genuine benchmarking contribution.

## Weaknesses

### Major

- **Asymmetric conditioning in the "unconditional" comparison with baselines (§4.1)**: TDDM is a *conditional* model p(x|H) that receives per-region 64×64 occupancy heatmaps H at both training and inference, while all baselines (TimeGAN, TimeVAE, COSCI-GAN, DiffTraj, Diffusion-TS) are genuinely unconditional p(x) models receiving no such information. The paper frames §4.1 as "unconditional trajectory generation" but TDDM explicitly conditions on H, which encodes precisely the spatial statistics that KL, JS, Density, and Trip metrics evaluate. The ablation in Table 2 confirms this: removing H leaves TSTR unchanged (0.011) but degrades KL-sym by 5× (1.334 vs. 0.277), meaning the dramatic distributional gains come primarily from the conditioning signal, not from fundamentally better temporal trajectory modeling. No baseline is given comparable spatial priors, so the comparison does not isolate the contribution of the architecture/factorization from the contribution of privileged conditioning information. The paper should either (a) acknowledge this asymmetry and reframe the claims accordingly, or (b) provide a comparison where baselines also receive aggregate spatial statistics.

- **"Zero-shot" generalization claims require access to target-city trajectory data (§4.3, Algorithm 2)**: The paper emphasizes "zero-shot generalization to new regions/cities without retraining or finetuning" (abstract, §4.3). However, Algorithm 2 line 3 computes H from X_target — the full trajectory dataset of the target city/region. While the parameterized model is never retrained, calling this "zero-shot" overclaims what the experiments demonstrate. Each H is a high-bandwidth summary of spatial occupancy computed from the very data whose distribution is being evaluated. The experiments show transfer of *temporal dynamics* conditioned on an *oracle spatial prior*, not generalization to data-sparse or data-absent regions (the motivating use case in §1). This should be acknowledged honestly — the demonstrated capability is "given a spatial occupancy map of a new city, a model trained elsewhere can generate plausible trajectories consistent with that map," which is useful but different from "zero-shot transfer to new cities."

### Minor

- **No comparison in the zero-shot setting with alternative approaches (§4.3)**: The OOD generalization results in Table 3 show only TDDM variants. Without a baseline comparison (e.g., directly applying a diffusion model trained on one city to another city), it is unclear whether the factorization rather than simply the canonicalization enables the transfer. The paper attributes transfer capability to the spatial-temporal factorization, but canonicalization alone might account for much of it.

- **Missing canonicalization ablation**: The paper claims canonicalization "enables parameter sharing across geographic regions" (§3) but never ablates it. Training TDDM without the similarity transform (using raw coordinates or translation only) and evaluating cross-region transfer would directly validate this architectural choice. This was also flagged in the prior submission's review.

- **Cross-region trajectory coherence is not evaluated**: TDDM generates trajectories independently per region. Real trajectories that span multiple regions are not discussed. It is unclear how the method handles boundary stitching, whether artifacts arise at region borders, or whether evaluation considers only intra-region trajectory segments.

- **Missing variance estimates for most metrics**: Only TSTR includes ±std. Distributional metrics (KL, JS, Density, Trip, Length, Pattern) report point estimates without variance, making it difficult to assess statistical significance of improvements.

- **Limited novelty of individual components**: Each building block — denoising diffusion with transformers, ViT-style patch tokenization, Procrustes-like canonicalization — is individually well-established. The contribution is their composition and the factorization insight, which is meaningful but incremental.

## Nice-to-Haves

- Computational cost analysis (training/sampling time, memory, scalability with number of regions or grid resolution) — essential for a method claiming "large-scale" generation but not standard in the field's evaluation conventions.
- Sensitivity analysis of spatial prior quality (e.g., computing H from 5%, 10%, 50% of target data) to test how much target information is practically needed.
- Per-city results in the main body rather than only in the appendix, to expose city-specific failure modes.
- Evaluation without map-matching as a primary rather than appendix condition, given the strong dependence shown in Table 9.

## Removed Points

These points are flagged to be removed; treat them with caution:

- **"Baselines without code shouldn't be compared" (Spark)**: The paper cites TrajGen and ControlTraj in the appendix as related work without reproducible code. This is standard practice — citing non-reproducible work alongside reproducible baselines is not a flaw.

- **"Privacy evaluation missing" (Spark)**: The paper explicitly does not claim privacy preservation as a contribution; the earlier version included k-anonymity but this version scopes it out ("this work focuses exclusively on improving fidelity and cross-region generalization"). Criticizing a scoped-out direction is scope creep.

- **"Standard deviations missing for KL/JS" (Harsh Critic, via earlier review)**: While reporting variance is good practice, large-scale diffusion model studies typically report point estimates for distributional metrics, and the improvements (4× lower KL) are large enough that variance is unlikely to change conclusions.

- **"Map-matching dependency limits applicability to developing regions" (Neutral Reviewer)**: This is a data preprocessing concern, not a methodological flaw. All models benefit from map-matching (Table 9), and the comparison remains fair.

- **"No theoretical proof" (UTSD review transfer)**: This is an empirical paper claiming empirical contributions; demanding theoretical proofs is beyond its scope.

## Novel Insights

The paper reveals an interesting decomposition insight: spatial distributional metrics (KL, JS, Density) and temporal fidelity metrics (TSTR) probe fundamentally different aspects of trajectory quality, and TDDM's factorization decouples them. The ablation (Table 2) shows that spatial priors are almost entirely responsible for distributional gains while contributing negligibly to temporal fidelity (TSTR unchanged). This suggests that future trajectory generation work should clearly separate these two evaluation axes and that unconditional baselines may be stronger on temporal dynamics than their KL scores suggest.

## Suggestions

1. **Reframe the "unconditional" comparison honestly**: Either acknowledge that TDDM is a conditional model given spatial priors and compare it as such, or augment baselines with comparable aggregate spatial conditioning to create an apples-to-apples comparison.
2. **Temper "zero-shot" claims**: Clarify that transfer requires computing a spatial prior from target-city data. Consider adding an experiment with degraded priors (reduced data, coarser grids, or simulated noisy priors) to quantify how much target information is needed.
3. **Add canonicalization ablation**: Remove the similarity transform and show generalization results — this would validate the core architectural claim.
4. **Report per-city results in the main text**: The average across three diverse cities hides potentially informative variation.

## Score and Decision

**Calibration**: The prior version of this work (dDdxbdhMsY) received scores 3, 6, 6 and was rejected. The current version addresses several prior weaknesses (added ablation study, expanded metrics, improved presentation) and adds cross-city transfer experiments. DiffMove, a trajectory diffusion model in a similar space, received scores 5, 6, 8, 6, 6 (rejected) with concerns about novelty and limited evaluation. This paper has a stronger methodological contribution (the spatial-temporal factorization) and more comprehensive evaluation, but has a more significant framing issue (conditional vs. unconditional comparison). Papers with novel problem factorization but flawed evaluation framing typically land in the 5-6 range based on the calibration examples.

The core idea — spatial-temporal factorization with canonicalization — is genuine and interesting. The asymmetric comparison and overclaimed "zero-shot" framing are significant but not fatal; they undermine the magnitude of the claims, not the validity of the approach. The empirical contribution (benchmark, multi-city evaluation, ablation) is solid. This is a paper with a good idea that oversells its results through framing issues, roughly comparable in substance to papers scoring 5.5-6.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>