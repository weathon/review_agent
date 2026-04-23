## Summary

SoloPose proposes a one-stage, many-to-many spatio-temporal transformer for 3D human pose estimation from monocular video, supported by two additional contributions: the AugMotion Toolkit for merging multiple 3D pose datasets into a universal coordinate system (yielding Human7.1M), and HeatPose, a kinematically-aware 3D GMM heatmap representation. The paper reports state-of-the-art MPJPE on both Human3.6M and Human7.1M testing sets.

## Strengths

- **AugMotion Toolkit addresses a real and important problem**: Merging multiple 3D pose datasets with incompatible coordinate systems is a practical challenge. The pipeline of key-frame selection (k-means), universal coordinate definition via anatomical landmarks (Sec 3.2), and Kabsch alignment (Sec 3.3) is a reasonable engineering solution. The ablation in Table 2 quantifies its impact: removing AugMotion increases MPJPE by 12.9 on Human3.6M testing, confirming data augmentation as the largest contributor to performance.

- **HeatPose is a conceptually sound idea**: Incorporating kinematically adjacent keypoints as side Gaussian distributions in the heatmap target (Sec 4.2, Eq. 6–8) provides richer gradient signal. Table 2 shows removing HeatPose increases MPJPE by 4.7 on Human3.6M testing, demonstrating a measurable improvement.

- **The paper transparently includes the "SoloPose only trained on Human3.6M" ablation** (Table 2, 38.9 MPJPE), which allows readers to assess the model's standalone contribution separate from the data advantage—though the paper does not honestly interpret this result (see Weaknesses).

## Weaknesses

### Fatal

None.

### Major

- **The central claim of "superior results" is confounded by training data**: SoloPose is trained on Human7.1M (four datasets combined), while all baselines (P-STMO, STCFormer, KTPFormer, FinePOSE) are trained only on Human3.6M. The paper's own ablation reveals that SoloPose trained on Human3.6M alone achieves 38.9 MPJPE on Human3.6M testing, which is **substantially worse** than FinePOSE w/ CPN (31.9) and KTPFormer w/ CPN (33.0). The headline numbers (26.0 MPJPE on H3.6M, 22.7 on Human7.1M) are primarily the product of training on 4–5× more data, not of a superior architecture. Without training baselines on Human7.1M as well, or comparing under matched training conditions, the claim of model superiority is unsupported.

- **The ablation discussion in Section 5.4.2 cherry-picks baselines**: The paper states "Our results of MPJPE and P-MPJPE are still 3.9% and 5.9% lower than the two SOTA methods on the Human3.6M testing dataset," comparing SoloPose-H3.6M (38.9) only against P-STMO (42.1) and STCFormer (40.5)—the two weakest baselines. It omits KTPFormer (33.0) and FinePOSE (31.9), against which SoloPose-H3.6M loses by a large margin. This selective comparison misrepresents the model's standalone performance.

- **Human7.1M testing comparison is inherently unfair to baselines**: Table 2 reports SoloPose at 22.7 MPJPE vs. FinePOSE w/ GT at 26.1 on Human7.1M testing, but baselines were never trained on data from this distribution (which includes MADS, AIST Dance++, and MPI INF 3DHP). Testing models on a distribution they were never trained on while the proposed model was does not constitute a valid benchmark. These numbers cannot support claims about model quality.

### Minor

- **HeatPose ablation confounds representation and loss function**: Section 5.4.1 removes HeatPose and simultaneously switches from cross-entropy to MSE loss. The 4.7 MPJPE improvement may come from the GMM structure, the cross-entropy loss, or both. A proper ablation would isolate each factor independently.

- **"Cost-efficient" claim is unsupported**: The paper introduces SoloPose as a "cost-efficient" model (line 47) but reports no computational metrics (FLOPs, parameters, inference time). Without these, the efficiency claim cannot be evaluated. This is especially relevant because SoloPose uses a frozen CLIP model (a large pre-trained network) as its spatial backbone.

- **The "avoids non-convex problems" claim for cross-entropy loss is incorrect**: Section 4.2 states that "using a cross-entropy loss function methodology avoids non-convex problems." Cross-entropy over a discretized 3D volume with GMM targets is not convex; this claim is misleading.

- **The one-stage advantage is asserted but not empirically demonstrated**: The paper argues that two-stage methods "pass on first stage errors," but SoloPose itself relies on frozen CLIP features—a pre-trained upstream module. No experiment isolates the benefit of one-stage processing (e.g., comparing SoloPose with a 2D detector upstream vs. CLIP, or analyzing how CLIP feature noise affects downstream performance).

### Trivial

- The universal coordinate system definition uses average body proportions (Eq. 1), which could introduce systematic scaling errors for individuals whose proportions differ significantly from the mean. This limitation is not acknowledged.

## Nice-to-Haves

- Train existing baselines (especially FinePOSE, KTPFormer) on Human7.1M and compare—this would isolate the architectural contribution from the data contribution and substantially strengthen (or clarify) the paper's claims.
- Report computational cost metrics (FLOPs, parameters, inference time) to support the "cost-efficient" framing.
- Test on in-the-wild images/videos to support the motivation about real-world applications.
- Disentangle the HeatPose ablation by testing: (a) standard heatmap + cross-entropy, (b) HeatPose + MSE, (c) HeatPose + cross-entropy.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **"SoloPose outperforms all two-stage methods even with GT input" (Strength Finder)**: This is based on the Human7.1M testing comparison where baselines were never trained on that distribution. It is an unfair comparison and cannot be claimed as a strength.
- **Missing hyperparameters (constant c, volume dimensions w×h×d, CLIP variant, Swin configuration, learning rate, training schedule)**: These are reproducibility nitpicks about implementation details impractical to fully include in a submission. Per the rules, these are removed.
- **No variance/std reported, no statistical significance tests**: Large-scale benchmark evaluation in this field typically reports single-run results; demanding confidence intervals is not standard practice.
- **N=30 vs N=243 frames confounds the comparison**: This is a design choice difference, not an unfair advantage. SoloPose uses fewer frames; if anything, this makes the comparison harder for SoloPose, not easier. The harsh critic's framing of this as a confound that hurts the comparison is inverted.
- **Human3.6M coordinate system misunderstanding (Figure 1)**: The harsh critic suggests Figure 1 may reflect the authors' incorrect data processing rather than an inherent dataset flaw. This is speculative and cannot be verified from the paper alone; the authors may have legitimate reasons for their observation.
- **Formatting/style issues**: Removed per rules.
- **Missing related works**: Cannot verify existence; removed per rules.

## Novel Insights

The paper's most revealing finding is an accidental one: the ablation in Table 2 functions as a natural experiment showing that data scaling (via AugMotion) contributes roughly 2.7× more to performance reduction than the architectural innovation (HeatPose). This quantifies what many in the community suspect—that data quality and diversity often dominate architectural choices in 3D pose estimation. A more honest framing around this finding, rather than overclaiming model superiority, would have strengthened the paper considerably.

## Suggestions

- **Most important**: Re-run FinePOSE and KTPFormer (the strongest baselines) on Human7.1M training data. If these baselines also improve substantially, it confirms data is the main driver. If SoloPose still wins under matched training data, the architectural claim becomes credible.
- Re-frame the contribution: center the paper on the AugMotion Toolkit as the primary contribution (since the ablation shows it is), and present SoloPose as a model designed to leverage this augmented data.
- Fix the misleading statement in Section 5.4.2 to acknowledge that SoloPose-H3.6M does not beat the strongest baselines (KTPFormer, FinePOSE).
- Remove or soften the "cost-efficient" claim until computational metrics are provided.

## Evaluation

**Originality**: Limited. SoloPose's architecture is a straightforward combination of frozen CLIP + Swin transformer blocks with 3D relative position embeddings. HeatPose (kinematically-aware GMM heatmap) is the most novel component but is incremental. AugMotion is engineering rather than scientific novelty.

**Importance of research question**: Moderate. 3D human pose estimation and dataset diversity are important problems, but the paper's approach does not convincingly advance the state of the art on the model side.

**Claims well supported**: No. The central claim of "superior results" is confounded by training data differences. Under fair comparison, the model underperforms the best baselines.

**Soundness of experiments**: Weak. The main comparison is unfair (different training data), the Human7.1M testing comparison is invalid for baselines, the ablation discussion cherry-picks baselines, and the HeatPose ablation confounds two variables.

**Clarity of writing**: Adequate. The paper is generally readable but makes several unsupported or incorrect claims (cost-efficient, avoids non-convex problems, superior results).

**Value to community**: Moderate potential. The AugMotion Toolkit could be useful if released, and HeatPose is a reasonable idea. But the evaluation problems undermine confidence in the reported results.

## Calibration

Anchors used:
- **V-JEPA** (`/home/wg25r/review_agent/human_reviews/WFYbBOEOtv.md`, avg 4.0, Reject): Trained on more data than baselines, making fair comparison impossible. Very similar unfair-comparison pattern to SoloPose. SoloPose has a similar level of overclaiming but with a more legitimate toolkit contribution.
- **LC-PCFG** (`/home/wg25r/review_agent/human_reviews/63r6HyqyRm.md`, avg 2.33, Reject): Imports billions of extra training data from LLM, making comparison unfair. More severe than SoloPose since the "contribution" is essentially just using an LLM.
- **PF-LRM** (`/home/wg25r/review_agent/human_reviews/noe76eRcPC.md`, avg 8.0, Accept spotlight): Strong method with potential data confounding but overall solid evaluation. Far above SoloPose in methodology and fairness.
- **WiMTR** (`/home/wg25r/review_agent/human_reviews/q3WzT2mrhB.md`, avg 4.0, Reject): Results cannot be compared to SOTA due to dataset differences. Similar fairness issue.
- **UniPose** (`/home/wg25r/review_agent/human_reviews/v2J205zwlu.md`, avg 5.0, Reject): Dataset-merging for pose estimation with some methodological issues. SoloPose has a more serious overclaiming problem.
- **Real3D** (`/home/wg25r/review_agent/human_reviews/Ffuw2ryqpz.md`, avg 5.0, Withdrawn): Potentially unfair comparison from more fine-tuning data, but the method has genuine architectural novelty.

SoloPose sits between the 2.33 (LC-PCFG) and 5.0 (UniPose/Real3D) anchors. It has real contributions (AugMotion toolkit, HeatPose) that LC-PCFG lacks, but its overclaiming is more severe than UniPose's. The unfair comparison is the dominant issue, similar to V-JEPA (4.0). I place SoloPose slightly below V-JEPA because V-JEPA at least has more methodological novelty in applying JEPA to video, while SoloPose's architecture is a straightforward combination of existing components.

MY FINAL SCORE: <pineapple>3.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>