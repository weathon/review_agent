Now I have a good understanding of the paper and the calibration landscape. Let me synthesize the final review.

Key observations from reading the paper:

1. **Copy-paste artifact formalization**: The paper introduces M_CP metric (Eq. 2) - a normalized angular distance that measures whether generated face embedding is closer to reference or ground truth. This is a reasonable metric.

2. **MultiID-2M dataset**: ~3k identities, ~1M references (~400 per identity), 500k labeled group photos, 1.5M unlabeled. This is a substantial dataset contribution.

3. **MultiID-Bench**: 435 test cases, rare long-tail identities with no overlap to training. Uses Sim_GT as primary metric (instead of Sim_Ref which incentivizes copy-paste).

4. **WithAnyone model**: Built on FLUX, 4-phase training. Key innovations: GT-aligned ID loss, ID contrastive loss with extended negatives (4096 negatives from labeled pool), paired training (Phase 3).

5. **Results (Table 1)**: WithAnyone achieves SimGT=0.460, which is best among ID-specific models (InstantID=0.464 is comparable). The main advantage is low CP=0.144 (vs InstantID 0.337, PuLID 0.315). Aesthetics is lowest at 4.783.

6. **User study**: Only 10 participants, 230 groups. "Moderate positive correlation" with human judgments.

7. **OmniContext result**: "highest score among ID-specific reference models" - qualified claim.

8. **Training pipeline**: λ_ID = λ_CL = 0.1 across all phases. Temperature τ and negative count M mentioned but not explicitly stated (4096 negatives vs 63 in ablation, mentioned in text).

9. **"Breaking the trade-off"**: This is a strong claim. The paper argues most methods lie on a SimGT vs CP trade-off curve and WithAnyone breaks it.

Let me now carefully evaluate the reviewers' criticisms against the actual paper content.

**Harsh Critic's points:**

1. **Evaluation heavily favors authors' own metric/benchmark**: The paper does use MultiID-Bench and M_CP as central. But it also reports OmniContext results and standard metrics (CLIP-I, CLIP-T, Aes, SimGT, SimRef). The user study is small. The claim about "moderate positive correlation" without quantitative details is a valid concern. However, the paper explicitly mentions "further details of the study design, ranking protocol, and statistical analysis are provided in Appendix H" - we just can't see it.

2. **MultiID-2M/Bench under-specified**: The paper says rare long-tail identities with no overlap. The exact deduplication procedure is in the appendix. Since we can't see the appendix, this is partially valid but partially addressed by the reference to appendix.

3. **Copy-paste not rigorously disentangled from legitimate high-fidelity**: This is a thoughtful concern. The paper defines CP as a relative measure (how much closer the generation is to reference vs GT). The paper uses a single GT image. However, the paper's motivation (Fig 2) specifically argues that natural variations in the same person's photos already reduce Sim significantly, and copy-paste generation is even closer to reference than natural variation would produce. This is a conceptual concern worth raising but the paper does address it at a high level.

4. **Overclaims of "state-of-the-art" and "breaking the trade-off"**: Looking at table 1, SimGT=0.460 vs InstantID=0.464 - WithAnyone is not the highest on SimGT! But it does have dramatically lower CP. The "breaking trade-off" claim is visual in Fig 5. The margins on SimGT are small. This is a valid concern about overclaiming.

**Neutral Reviewer's points:**

1. **Limited novelty of contrastive loss**: Valid - InfoNCE is standard, the contribution is the extended negative pool enabled by the dataset.

2. **Small user study**: Valid - only 10 participants.

3. **Ethical concerns about celebrity images**: Valid but common in this field.

4. **Incomplete comparison with DynamicID**: The paper explains DynamicID is excluded due to unavailability. This is a REMOVE per the rules.

5. **CP metric edge cases**: Valid concern about when θ_tr is small.

6. **Lower aesthetic scores**: Valid - 4.783 is the lowest. This is a real trade-off.

**Spark's points:**

1. **Non-celebrity evaluation**: Valid suggestion but somewhat out of scope.

2. **Comparison with reduced ID loss weight**: Valid as an ablation - would test if the paired training is necessary.

3. **Scaling analysis for negative pool**: Only 63 vs 4096 tested.

4. **Statistical validation of user study**: Valid.

5. **Memorization audit**: Interesting but somewhat out of scope.

Now let me write the final review.</think>Now I have a thorough understanding of the paper and relevant calibration papers. Let me write the final review.

## Summary

WithAnyone addresses the "copy-paste" artifact in identity-consistent image generation—where models replicate reference faces rather than preserving identity across natural variations—and makes three main contributions: (1) MultiID-2M, a large-scale paired dataset with ~3k celebrity identities and ~400 reference images each; (2) MultiID-Bench, a standardized benchmark with a novel Copy-Paste metric (M_CP) that measures relative bias toward the reference versus ground truth; and (3) a FLUX-based model trained with GT-aligned ID loss, ID contrastive loss with extended negatives, and a four-phase paired-training pipeline that reduces copy-paste artifacts while maintaining competitive identity similarity.

## Strengths

- **Important problem formalization.** The copy-paste artifact is a real, under-appreciated failure mode in ID-consistent generation. The paper's observation (Fig. 2) that natural face variation already reduces similarity, and that reconstruction-based training exacerbates over-copying, is well-motivated and clearly articulated. Introducing Sim_GT as the primary metric (rather than Sim_Ref, which rewards copying) is a principled improvement over prior evaluation.

- **Substantial dataset and benchmark contribution.** MultiID-2M provides ~500k labeled group photos with paired references per identity, filling a genuine data gap. The construction of paired, multi-reference-per-identity data from celebrity web images with ArcFace clustering is a non-trivial engineering effort that will benefit the community if released as claimed.

- **Effective training pipeline.** The four-phase training progression from reconstruction to paired tuning is well-designed. The GT-aligned ID loss is a clever practical contribution (avoiding noisy landmark extraction from generated images, enabling ID loss at all noise levels), and the extended InfoNCE negative pool leverages the labeled dataset effectively. Ablations (Table 3) confirm each component's contribution, with Phase 3 reducing CP from 0.239 to 0.161 and FFHQ-only achieving only 0.224 Sim_GT.

- **Competitive results with clear CP advantage.** WithAnyone achieves Sim_GT=0.460 (competitive with the best ID-specific models like InstantID at 0.464) while dramatically reducing copy-paste (CP=0.144 vs. InstantID 0.337, PuLID 0.315). Fig. 5 visually demonstrates WithAnyone's position off the regression curve, supporting the claim of improved trade-off.

## Weaknesses

### Major:

- **The "breaks the trade-off" and "state-of-the-art" claims are overstated given modest margins and absent statistical analysis.** Looking at Table 1, WithAnyone's Sim_GT (0.460) is essentially tied with InstantID (0.464) and UMO (0.458)—the difference is within noise of embedding-based metrics. The paper's real advantage is in M_CP, which is measured on the authors' own newly introduced metric. No standard deviations, confidence intervals, or significance tests are reported anywhere. The "regression curve" in Fig. 5 is described only qualitatively, with no fit statistics. The OmniContext result is qualified as "best among ID-specific reference models," but this excludes stronger general models. The evidence supports "competitive with the best existing models and substantially better on our new copy-paste metric," not an unambiguous break from a fundamental trade-off.

- **The Copy-Paste metric (M_CP) and benchmark lack rigorous validation.** The user study has only 10 participants, and the paper describes only a "moderate positive correlation" with human judgments—no correlation coefficients, confidence intervals, or inter-rater reliability statistics are reported in the main text (deferred to Appendix H). The metric is defined relative to a single GT image per case, but natural variation across multiple valid photos of the same person means a generation legitimately different from the one GT could be penalized. No sensitivity analysis (e.g., behavior when θ_tr is small, robustness to embedding noise, or failure cases where the metric disagrees with human perception) is provided. Since the central claim hinges on this metric, these gaps are significant.

- **Dataset construction and identity labeling lack transparency in the main text, impacting trust in the core claims.** The paper states benchmark identities are "rare, long-tail" with "no overlap to training data" but does not operationally define how overlap is detected or how identities are disambiguated. Identity clustering relies on ArcFace embeddings, and no error rate or manual verification is reported. Since both training signals and evaluation metrics (Sim_GT, Sim_Ref, M_CP) depend on the same embedding family, methods that overfit to ArcFace-like features could be systematically favored. Details are deferred to appendices that are not present in the main submission.

### Minor:

- **Aesthetic quality is notably lower than several baselines.** WithAnyone achieves Aes=4.783 on the single-person subset, the lowest among all compared methods (FLUX.1 Kontext: 5.319, GPT-4o: 5.344). This potential trade-off between reduced copy-paste and perceptual quality is not discussed, despite being visible in the reported numbers.

- **Limited novelty of individual technical components.** The InfoNCE contrastive loss is standard; the main novelty is scaling negatives via the labeled dataset (from 63 to 4096). The GT-aligned ID loss is a practical engineering contribution. The overall contribution relies heavily on the combination of dataset + benchmark + training recipe rather than on architectural innovation.

- **Comparison fairness concerns due to base model and data scale differences.** WithAnyone is built on FLUX, a much more powerful backbone than many baselines (e.g., those based on Stable Diffusion 1.5/SDXL). Additionally, the model is trained on large proprietary paired data (MultiID-2M + FFHQ + CelebA-HQ + FaceID-6M), while some baselines may not have comparable data advantages. Disentangling the contribution of the proposed training recipe from foundation model and data scale effects is not fully addressed.

- **Missing ablations for key design choices.** The extended negative pool is tested at only two points (63 vs. 4096); no intermediate scaling analysis is provided. The hyperparameters λ_ID = λ_CL = 0.1 lack sensitivity analysis. Reference-count sensitivity (few vs. many references per identity) is not examined.

- **The conceptual boundary between "harmful copy-paste" and "legitimate high-fidelity matching" is under-explored.** In many personalization use-cases (avatars, profile photos), staying very close to the reference is precisely what users want. The paper pathologizes high Sim_Ref without adequately analyzing scenarios where it is desirable, which weakens the claim that the copy-paste artifact is universally harmful.

### Trivial:

- The four-phase training details (step counts like "~20k", "~40k") are reported approximately, and some hyperparameters (τ temperature value) are not explicitly stated in the main text.

## Nice-to-Haves

- Evaluate on non-celebrity identities to test generalization beyond memorizable public figures.
- Add statistical analysis (confidence intervals, significance tests) to the user study and quantitative tables.
- Provide a sensitivity analysis of M_CP when θ_tr is small or when multiple GT images are available.
- Include an ablation where ID loss weight is simply reduced (without paired training) to test whether the proposed paradigm is necessary beyond a simpler fix.
- Discuss the perceptual importance of Sim_GT values (e.g., what does 0.46 vs. 0.52 mean perceptually?).

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **DynamicID exclusion**: The neutral reviewer criticized omitting DynamicID as a baseline. The paper explicitly states DynamicID was excluded "due to unavailability of code and pretrained models" (footnote 1). Per the rules, we do not question the availability of cited systems. However, the reviewer's broader point about missing controllable baselines retains some relevance—though specific baselines not runnable due to unavailable code are not the authors' fault.

- **Reproducibility concerns about hyperparameters**: The harsh critic raised extensive reproducibility concerns (batch sizes, exact step counts, temperature τ). These are standard in large-scale generative model papers and not a differentiating weakness. The λ values are stated and key architecture details are in the appendix reference.

- **Ethical concerns about celebrity images and dual-use**: While valid societal considerations, these are standard concerns for any identity-preserving generation paper and not a unique scientific weakness. The paper includes a substantive ethics section addressing licensing and mitigation.

- **Demands for non-celebrity test identities**: The paper scopes its contribution to multi-identity celebrity generation with paired references. Demanding evaluation on entirely different identity distributions (e.g., ordinary individuals with few references) expands the scope beyond what the paper claims. Raised as a nice-to-have instead.

- **Formatting/style nitpicks**: Removed per rules.

## Novel Insights

The paper's most insightful contribution is the formalization and measurement of copy-paste as a relative phenomenon (M_CP): rather than measuring absolute similarity to reference, it measures whether the generated embedding is biased toward the reference more than the GT, normalized by the reference-to-GT distance. This reframing exposes how Sim_Ref implicitly incentivizes copying and provides a more nuanced lens for evaluating ID-customization methods. The empirical finding that most existing methods cluster on a Sim_GT vs. M_CP regression curve (Fig. 5), with high-identity-fidelity methods paying a copy-paste penalty, is a meaningful observation even if the claim of "breaking" this trade-off is overstated.

## Suggestions

- **Report statistical measures** (standard deviations across test cases, bootstrap confidence intervals for metric differences) to establish whether the reported improvements are meaningful.
- **Validate M_CP rigorously**: report correlation coefficients with human judgments (not just "moderate positive"), analyze failure cases, and test behavior when reference and GT are already very similar.
- **Tone down claims**: replace "breaks the trade-off" with "improves the trade-off" and "state-of-the-art" with more precise qualifiers (e.g., "best reported copy-paste score among ID-specific models on our benchmark").
- **Add an ablation** testing whether simply reducing λ_ID in a reconstruction-only model achieves comparable CP reduction, or whether the paired training design is specifically responsible for the improvement.
- **Acknowledge the aesthetic quality trade-off** explicitly and discuss whether it is an inherent cost of reducing copy-paste or could be mitigated in future work.

## Score and Decision

**Calibration comparison:**

- **MS-Diffusion** (scores 6/6/6/6/6, Accept Poster): Similar multi-subject personalization paper with incremental technical novelty and a benchmark contribution. WithAnyone has a more novel problem framing (copy-paste) and more substantial dataset contribution, but similar issues with novelty of individual components.

- **RetriBooru** (scores 5/5/3/5, Withdrawn/Reject): Similar motivation (avoiding "leakage" / copy-paste in reference conditioning), similar dataset contribution. Weaker empirical validation. WithAnyone is stronger empirically.

- **One-Prompt-One-Story** (scores 8/8/8/8/5, Accept Spotlight): Novel training-free insight for consistent generation. Much higher novelty per component. WithAnyone has a weaker novelty ratio (more engineering, less conceptual breakthrough).

- **FUSION IS ALL YOU NEED** (scores 3/6/3/3, Reject): Identity-preserving generation with copy-paste problems, limited novelty, unconvincing quantitative results. WithAnyone is substantially stronger—better problem framing, metric, dataset, and empirical results.

- **VideoAlchemy** (scores 3/6/5/5, Withdrawn/Reject): Similar dataset + method + benchmark package, similar concerns about base model advantage and novelty. WithAnyone has a clearer methodological story and more thorough evaluation.

WithAnyone is a solid engineering contribution with a useful problem formalization, but the overclaiming ("breaks the trade-off," "state-of-the-art"), the weakly validated novel metric, the small user study, and the moderate novelty of individual components prevent it from reaching the level of clearly strong papers. It is comparable to MS-Diffusion (6-range) but with some additional concerns about self-evaluation validity. The dataset/benchmark contribution is substantial, and the paper addresses a real problem, but the evidence doesn't fully support the strongest claims.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>