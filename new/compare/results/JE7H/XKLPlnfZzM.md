---
job_id: 8f246750-7980-499a-8ce7-5ab6d0a74cb3
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: XKLPlnfZzM.pdf
paper: Learning to Deaggregate: Large-Scale Trajectory Generation with Spatial Priors
main_score_norm: 0.6
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The work is on diffusion-based generative models, representation learning, and generalization for spatio‑temporal trajectories, which fits squarely within ICLR topics (generative models, representation learning, benchmarks).

## Minimum Quality
Pass ✅.  
The paper is in English and has all major sections (Abstract, Introduction, Problem Definition, Method, Experiments, Conclusion, Ethics). The method is technically nontrivial, experiments are extensive with strong baselines, and there are no obvious fatal methodological errors or misuse of test data.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I did not find any hidden prompts or attempts to manipulate automated reviewing.

---

# Expected Review Outcome:

## Summary

The paper proposes the Temporal Deaggregation Diffusion Model (TDDM), a trajectory generator that factorizes “where” and “how” people move by conditioning a diffusion model on spatial occupancy priors (heatmaps) and using similarity transforms to canonicalize regions. Trajectories are generated region‑by‑region in normalized coordinates, then mapped back to geographic space, enabling parameter sharing across regions and cities. On three real‑world mobility datasets (Geolife, Porto, Cabspotting) and a standardized metric suite, TDDM outperforms GAN-, VAE-, and diffusion‑based baselines on distributional metrics while maintaining comparable or better sample‑level fidelity, and shows promising zero‑shot generalization to unseen regions and cities.

## Strengths

1. **Clear and coherent spatial–temporal factorization with canonicalization.**  
   The core idea of separating spatial marginals (heatmap priors) from temporal dynamics, combined with Procrustes‑style similarity transforms over fixed‑size regions, is well articulated in Section 3. This is more than “just conditioning”: the model is trained only in normalized local coordinates and never sees absolute lat/long, so it genuinely reuses temporal dynamics across locations. The mixture formulation in Eq. (5) formalizes this region‑based view and explains how city‑scale generation reduces to independent region sampling.

2. **Architecture and tokenization are thoughtfully designed and reasonably simple.**  
   The use of a transformer encoder that jointly attends over trajectory tokens, prior‑heatmap tokens, and a denoising‑step token (Figure 3, Page 5) is a clean, modular design. The tokenization strategy for the heatmap (ViT‑style patches with positional encodings) is standard but well adapted; importantly, it provides a clear path to inject richer priors (e.g. road hierarchy, temporal marginals) later, as the authors note in the conclusion.

3. **Strong empirical gains on multiple, well‑chosen metrics.**  
   The unconditional generation benchmark (Section 4.1) is quite comprehensive. Table 1 (Page 8) shows that TDDM beats all baselines by a large margin on symmetric KL and JS (e.g., KL\(_\text{sym}\) 0.277 vs 1.153 for Diffusion‑TS and 1.232 for DiffTraj; JS 0.059 vs 0.198/0.209), and also on Density/Trip errors. TSTR is at least as good as the best diffusions (0.011 vs 0.013–0.014), indicating that downstream usefulness is not sacrificed by imposing spatial priors. Per‑city results in Table 7 (Appendix, Page 41) corroborate that this holds across Geolife, Porto, and Cabspotting.

4. **Ablation convincingly shows the importance of spatial priors (and not just “a bigger model”).**  
   Table 2 (Page 9) is one of the more informative pieces of evidence: removing the spatial prior but keeping the rest of the architecture essentially intact leads to KL\(_\text{sym}\) exploding from 0.277 to 1.334 and JS from 0.059 to 0.228, while TSTR stays almost identical. This cleanly supports the claim that the priors specifically improve distributional alignment (coverage + proportionality) rather than simply improving sample fidelity. The 1×1 km vs 3×3 km comparison also highlights a nontrivial tradeoff between local pattern score and length distribution.

5. **Generalization experiments are stronger than what is often seen in trajectory‑generation work.**  
   Section 4.3 and Table 3 (Page 10) evaluate both intra‑city (25%‑coverage training) and city‑to‑city transfer without any fine‑tuning, conditioning only on priors computed from target‑city trajectories. The intra‑city setting shows TSTR remaining almost identical (0.010 vs 0.010), with KL\(_\text{sym}\) and JS degrading but staying in a reasonable range (0.545 / 0.106 vs 0.278 / 0.059). The cross‑city results (Table 12, Page 46) are particularly interesting: training on Porto and evaluating on other cities gives KL\(_\text{sym}\) ≈ 0.263–0.407 with Pattern ≥ 0.93, often competitive with intra‑city 25% training. This is a meaningful demonstration of the claimed “zero‑shot” capability.

6. **Visualization quality strongly supports quantitative claims.**  
   Figures 2, 4, and 5 (Pages 3 and 24) and the per‑city detail plots (Figures 6–8, Pages 25–27) do a good job of visually conveying that TDDM both fills the road network more faithfully and avoids hallucinating large off‑road blobs. In Figure 2, for Porto, TDDM’s log‑density heatmap most closely matches the real one, with distinct “holes” where roads are absent, whereas Diffusion‑TS/DiffTraj blur over and GAN/TimeVAE variants either miss network extremities or oversmooth everything. This visual evidence matches the density‑ and pattern‑based metrics.

7. **Benchmarking and evaluation framework are a significant contribution by themselves.**  
   The paper systematically harmonizes multiple quality dimensions (fidelity, diversity, proportionality, usefulness, generalization) via KL‑based metrics + DiffTraj metrics + TSTR, and does so across three cities on different continents. Section E (Appendix) details the 256×256 discretization for KL and relates each metric back to those conceptual dimensions. Even if one is not sold on TDDM per se, the benchmark is likely to be reused by others.

## Weaknesses

1. **Dependence on access to target‑region trajectories undermines part of the motivation.**  
   A central motivation in the Introduction (Page 1) is data scarcity and privacy constraints: “data cannot be shared” and “new environments lack sufficient observations.” However, the zero‑shot generation procedure in Algorithm 2 (Page 6) explicitly computes \(H = f(r_c, \mathbb{X}_\text{target})\) from target‑city trajectories. This means TDDM needs reasonably dense target data to estimate spatial priors; it does *not* operate from maps alone or from extremely sparse counts. This weakens the privacy / “no data” story and should be acknowledged more honestly: the method assumes that at least aggregate positional data (enough to form 64×64 heatmaps per 3×3 km region) are available in the target city, which is a nontrivial requirement in many sensitive mobility applications. Related, the “city‑to‑city zero‑shot” label is slightly misleading because some target trajectories are in fact used to estimate priors (even if individual paths are not fed into the model).

2. **Some mathematical notation and formulations are sloppy and obscure the true generative story.**  
   Several equations in Section 3 could use clarification and, in their current form, are somewhat inconsistent:
   - Eq. (1): \(p(x) = \int p(x \mid H)p(H)\, dl\). There is no variable \(l\) defined, and later \(p(H)\) is set to \(p(H = f(r_c, \mathbb{X})) = p(r_c)\). What is formally being integrated here is unclear. If this is meant to be marginalization over regions or priors, it should be expressed as a sum over discrete regions, not an integral with a mysterious \(dl\).
   - Eq. (2): \(p(r_c) \propto \sum_{x\in\mathbb{X}}\sum_n \mathbbm{1}(T_{r_c} x[n] \in [-1,1]^D)\). This defines a proportionality but never specifies the normalizing constant or whether trajectories are weighted by duration; later, Eq. (5) replaces this with a sum over \(r_c\) but still does not fully tie this to a *proper* mixture \(p(x) = \sum_{r_c} p(r_c) p(x \mid H_{r_c})\) with \(p(r_c)\) normalized.
   - Eq. (3): The definition of \(H_{i,j}\) uses \(\mathbbm{1}_{r_{c_{i,j}}}(x[n])\) but the condition in Eq. (4) is written as \(\mathbbm{1}_{r_{c_{1,j}}}\), which is almost certainly a typo. More importantly, the denominator \(\sum_{x,n} \mathbbm{1}(x[n] \in \mathcal{R}_{r_c})\) implicitly defines the region \(\mathcal{R}_{r_c}\); this set is not clearly defined in the text.
   - There is an inconsistency between mapping regions to \([-1,1]^D\) in the canonicalization paragraph (Page 4) and “Normalize \(X_{r_c}\) to [0, 1]^D” in Algorithm 1, line 6 (Page 6); clear notation should make it obvious whether the model sees \([-1,1]^D\) or \([0,1]^D\).
   None of these are fatal, but together they make the probabilistic framing more hand‑wavy than it needs to be. A tightened presentation of \(p(r_c)\), \(H_{r_c}\), and the exact coordinate normalization would significantly improve rigor.

3. **The “universal” temporal dynamics assumption is not empirically probed or stress‑tested.**  
   The canonicalization strategy relies on a strong assumption: that local temporal dynamics are transferable across cities once coordinates are normalized and only the spatial marginal differs. The authors themselves note that Length error degrades in cross‑city settings (Table 3, Page 10; and more starkly in Table 12, Page 46, where Length goes up to 0.148/0.193 in some transfers), especially between cities with different typical trip length distributions. However, there is no explicit ablation that turns *off* canonicalization (e.g., training in absolute coordinates) or that perturbs similarity transforms to test robustness to misalignment. Without such comparisons, it is difficult to tease apart how much of the gain comes from spatial priors alone versus canonicalization and whether the invariance assumption actually holds beyond the three studied cities.

4. **Limited comparison to *conditional* diffusion baselines and closely related spatial‑temporal diffusion work.**  
   While the paper compares against DiffTraj and ControlTraj conceptually in Appendix A, the main empirical baselines (Table 1) are unconditional or simple TS diffusion models (DiffTraj, Diffusion‑TS, TimeVAE, etc.). Conditional trajectory diffusion methods that exploit road networks or trip attributes (e.g., ControlTraj) and more recent spatiotemporal diffusion architectures (e.g., ST‑DiffTraj, CDSTraj, Geometric Trajectory Diffusion Models) are not included empirically and are only partially discussed. Given that TDDM’s core claim is about spatial priors enabling better coverage/generalization, it would be particularly important to show either:
   - that these conditional models *cannot* be easily adapted to cross‑city zero‑shot via similar priors, or  
   - that, when used in a comparable setting, they are still outperformed by TDDM.  
   As it stands, the novelty and advantage over the broader set of diffusion‑based trajectory models is somewhat under‑positioned.

5. **The “deaggregation” story ignores how priors are estimated and their availability in truly unseen regions.**  
   The paper treats the prior \(H = f(r_c, \mathbb{X})\) as if it were an exogenous given, but in practice computing \(H\) in a new city requires either (a) collecting enough trajectories to estimate marginals, or (b) using a road graph / proxy model. Neither scenario is seriously discussed. For example, in Figure 1 (Page 2), the bottom panel suggests generating in an unseen dashed rectangle, but the text does not clarify whether that rectangle’s prior was estimated from withheld data (which contradicts the “unseen” claim) or from some external signal. This is critical to understanding what “generalization to new regions” really means and how the method would be used in settings with strict data access constraints.

6. **Ablations stop short of exploring the full design space of priors and regionization.**  
   Table 2 examines only two region sizes (3×3 km vs 1×1 km) and the presence/absence of priors. Yet several aspects of the design are potentially important and are only justified qualitatively:
   - The choice of 64×64 grid for the 3×3 km regions: what happens with 32×32 or 128×128? The complexity discussion in Section D.1 (Page 20) mentions quadratic cost in token count, but there is no empirical tradeoff curve between spatial resolution and performance.
   - Overlapping region sampling vs a rigid grid at sampling time: Algorithm 1 uses random translations/rotations with arbitrary overlap, whereas Algorithm 2 participants are on a grid. This asymmetry might produce boundary artifacts or mismatched distributions. Visualizations like Figures 12–14 (Pages 31–33) *suggest* that transitions are fine, but there is no quantitative assessment of border consistency.
   - Priors only encode point density, not directions or speeds; the authors speculate about adding richer marginal information in the conclusion, but there is no ablation that modifies \(H\) to, say, coarsen or smooth it and show what breaks. This makes it harder to judge *how much* information is truly needed in the prior.

7. **Evaluation still conflates training and test distributions, raising questions about memorization/generalization on the *same* city.**  
   Section E.6 (Page 23) argues that held‑out test sets are “currently infeasible” and that replicating the training distribution is already hard enough. However, this is exactly where one would want to see TDDM’s purported advantage in avoiding memorization thanks to aggregate priors: e.g., train on a subset of trajectories in Geolife and evaluate on held‑out users/trajectories in the same geographic extent. Without such a split, most of the “in‑distribution” results (Table 1, Figure 2) test matching of the *full* training distribution, which is closer to a density‑estimation objective than a true generalization test. This does not invalidate the KL‑based improvements, but it makes the “generalization” claims within a city somewhat weaker.

8. **Minor but notable clarity issues.**  
   - The positional encoding equations (8)–(9) are mis‑typed: \(\text{PE}_{(pos,2i)} = \sin(-e^i \frac{\log(10000)}{d/2 - 1})\) appears dimensionally nonsensical; presumably pos should appear inside the sine/cosine, not an exponent of \(e^i\). Since the implementation is likely using standard transformer PEs, this is mostly cosmetic but confusing for a reader trying to reproduce details from the text alone.
   - Algorithm 1, line 4 uses “Find contiguous subsequences of trajectories in X that lie within \(r_c\)” without specifying how partial trajectories crossing region boundaries are handled when sampling vs counting in priors. This matters if one wants to interpret the mixture \(p(x) = \sum_{r_c} p(r_c) p(x \mid H_{r_c})\) as generating full trips.

## Potentially Missing Related Work

The following directly related papers are not cited and should be discussed:

1. **Liao et al., “CDSTraj: Characterized Diffusion and Spatial-Temporal Interaction Network for Trajectory Prediction in Autonomous Driving,” 2024.**  
   This work uses diffusion models with explicit spatio‑temporal interaction networks for trajectory prediction. It is related because it integrates spatial structure and temporal dynamics in a diffusion framework. It should be discussed in the extended related work (Appendix A) as another diffusion‑based approach that models spatial‑temporal interactions, with clarification on how TDDM’s spatial priors differ (e.g., aggregate occupancy vs pairwise interactions) and whether CDSTraj’s ideas could complement TDDM’s prior design.

2. **Liao et al., “ST-DiffTraj: A Spatiotemporal-Aware Diffusion Model for Trajectory Generation,” 2025.**  
   This is highly relevant: it proposes spatiotemporal‑aware diffusion for trajectory generation, closer to the unconditional generative task studied here. It should be cited in Section 1 and Appendix A, with a comparison focusing on how ST‑DiffTraj handles spatial structure (likely via explicit spatiotemporal encoders) versus TDDM’s aggregate‑prior factorization. If possible, a brief empirical comparison (even if approximate) would strengthen positioning.

3. **Han et al., “Geometric Trajectory Diffusion Models,” 2026.**  
   This work develops diffusion models for geometric trajectories (e.g., 3D curves), which is conceptually close to modeling road or motion paths. It should be mentioned in Appendix A with a discussion of how geometric trajectory diffusion compares to TDDM’s canonicalization and region‑based formulation, and whether their geometric constraints could be used as alternative priors.

## Questions

1. **Clarification on how priors are obtained in “unseen” regions.**  
   In Figure 1 (bottom) and the intra‑city 25% experiments (Figures 12–14), what exactly is used to estimate \(H\) in regions where no training trajectories were used for model fitting? Are these priors computed from all data (including the held‑out 75%) solely for evaluation, or from a restricted subset consistent with the training scenario? Please spell out precisely which data are available when computing \(H\) during “zero‑shot” generation.

2. **Can you provide a more rigorous definition of \(p(r_c)\) and \(p(x \mid H_{r_c})\)?**  
   It would be helpful to see a fully specified probabilistic model: concretely, how is \(p(r_c)\) normalized in Eq. (2)/(5), and how does the sampling procedure in Algorithm 2 correspond to that distribution? Are trajectories generated independently per region, and if so, how is the effective mixture weight per region derived from Eq. (2) and line 4 in Algorithm 2?

3. **What is the effect of canonicalization alone, without priors?**  
   A key missing ablation is a version of the model that uses similarity transforms and the same transformer but no \(H\) token. How does that compare to “w/o spatial prior” in Table 2? If you have run such an experiment, please report it; if not, could you outline why it is nontrivial or what you would expect qualitatively?

4. **How sensitive is TDDM to the quality or sparsity of priors?**  
   Have you tried artificially sparsifying or corrupting the priors \(H\) (e.g., using only 10% of trajectories to compute \(H\), or blurring the heatmap) and measuring how KL / Pattern / TSTR change? Such results would greatly help practitioners understand what level of aggregate information is needed for reasonable performance, especially in data‑scarce regions.

5. **Could conditional baselines be adapted to use the same priors?**  
   For diffusion‑based baselines that already accept conditioning (DiffTraj, ControlTraj, perhaps ST‑DiffTraj), is there a conceptual barrier to using the same heatmap prior \(H\) as an additional condition? Do you expect their architectures to fail in cross‑city zero‑shot settings even with such priors, and if so, why? Any empirical or theoretical justification here would clarify the distinctiveness of TDDM.

## Flag For Ethics Review

- No ethics review needed.

## Details Of Ethics Concerns

N/A. The paper uses public trajectory datasets with stated licenses and discusses dual‑use and privacy concerns in an Ethics Statement. The method itself is a generic generative model; there is no explicit handling of privacy, but this is acknowledged as a limitation rather than misrepresented as privacy‑preserving.

## Soundness Rating

3: good.  
The methodology and experiments are generally sound and well executed, with extensive baselines and ablations. Some probabilistic formulations and assumptions (e.g., prior estimation and canonicalization) are not fully formalized or stress‑tested, but there are no obvious fatal flaws.

## Presentation Rating

3: good.  
The paper is well organized, with clear figures and thorough experimental tables. However, several equations are mis‑typed or ambiguous, and some key assumptions (availability of priors, true meaning of “unseen regions”) are not articulated as clearly as they should be.

## Contribution Rating

3: good.  
The combination of spatial occupancy priors, canonicalization, and a transformer‑based diffusion model for large‑scale trajectory generation, together with a strong benchmark and metric suite, is a meaningful contribution. The main limitation is that the work does not fully situate itself against the latest spatiotemporal diffusion models and could be clearer about what is practically required to obtain spatial priors.

## Overall Rating

6: Marginally above the acceptance threshold. But would not mind if paper is rejected.  
The paper offers a solid and practically relevant improvement in trajectory generation with a clean architectural idea and strong empirical evidence, especially on distributional metrics and zero‑shot style generalization. At the same time, some of the conceptual framing (data scarcity/privacy, nature of “unseen” regions) is overstated relative to what the method actually assumes, and the mathematical formulation could be tightened. With clearer positioning and a bit more rigor on the prior/canonicalization story, this would be an easy accept.

## Reviewer Confidence

4: confident.  
I am familiar with diffusion models, spatio‑temporal generative modeling, and trajectory simulation. I carefully read the main paper and appendices, including equations and experiments, though I did not attempt to re‑implement the method.