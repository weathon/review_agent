Now let me carefully construct the final review. Let me verify the key claims against the paper:

1. **Table 1 row 4**: The paper states row 4 is "nuScenes Train (Sub 50%)" for map training and "nuScenes Train (Another Sub 50%)" for motion training, with evaluation on nuScenes Val. This achieves 0.6373 minADE, very close to the proposed split's 0.6308. The harsh reviewer is CORRECT that this undermines the spatial overlap concern, since row 4 doesn't remove spatial overlap but still achieves similar performance. However, the train-val gap (inference on training data) is addressed by row 4 as well. The proposed split additionally addresses spatial overlap, which matters for generalization evaluation.

2. **Boundary-free baseline**: Looking at Eq. 1 and the description in Section 3.3, it's indeed sparse - one equation and a brief description. The paper doesn't detail architecture specifics (sampling points, heads, etc.). But for a baseline contribution in a benchmark paper, this might be acceptable given the promised open-source code.

3. **Train-val gap claim**: Table 1 shows that the proposed split (row 1, 0.6308 minADE) vs. default (row 3, 0.6839). Row 4 (sub-50% split, 0.6373) also does well. Row 2 (trained on combined, worse at 0.7006) confirms that in-distribution map data hurts. This is actually some evidence but not as direct as measuring map quality directly.

4. **Overclaiming "misconception"**: The paper uses very strong language ("mis-usage and misunderstanding"), but the experimental evidence shows the proposed benchmark is more demanding and reveals more, not that prior protocols were "wrong."

5. **Model diversity**: Only HiVT and DenseTNT are used as motion models, and MapTR/MapTRv2-CL as map models. These are relatively standard but not the most recent.

6. **Single dataset**: nuScenes only, but the paper acknowledges this and explains why (it's the only dataset with all required modalities).

Now let me formulate the review.</think>Now I have a solid calibration. Let me write the final review.

## Summary
This paper identifies several issues with the current online map-based motion prediction protocol: (1) inappropriate dataset splits causing a train-validation distribution gap, (2) misaligned perception ranges between online mapping models (~30×60m) and motion prediction needs (>100m), and (3) non-discriminative metrics that evaluate only the ego vehicle and include trivially-predictable static agents. The authors propose OMMP-Bench with a new spatially-disjoint data split, refined metrics evaluating moving non-ego agents at different distances, and a "boundary-free" baseline that uses deformable attention from agents to raw image features to compensate for out-of-range map coverage.

## Strengths

- **Important and timely problem identification.** The paper surfaces genuine issues in the emerging online map-based motion prediction paradigm. The range mismatch (maps covering ~30×60m while agents can be 100m+ away) and the metric triviality (static agents have minADE ≈ 0.002) are well-demonstrated with concrete evidence (Tables 2, 3, 6). These are real problems that the community should address.

- **Systematic empirical analysis.** The mapping of map perception range vs. accuracy (Table 2), the comparison of GT maps at different ranges (Table 3), and the decomposition by agent type and distance (Table 6) provide actionable insights for both the mapping and prediction communities. Table 5 showing centerlines are nearly as effective as the full element set is a useful empirical finding.

- **Principled metric refinements.** The decision to evaluate moving non-ego agents and to stratify by close/far is well-motivated and aligned with standard practices in Argoverse and Waymo benchmarks. The stark gap between static (0.002) and moving (0.5585–0.6997) minADE in Table 6 clearly justifies excluding static agents.

- **Comprehensive benchmark results.** Table 7 evaluates 16 method combinations (2 map models × 2 motion models × 4 integration methods) with stratified metrics, providing a useful reference point for the field.

## Weaknesses

### Major:

- **Train-val gap claim is not conclusively demonstrated, and the proposed split's advantage is ambiguous.** The paper's central claim that the default protocol creates a "severe train-val gap" is not directly measured—no map quality metrics (e.g., mAP) are reported on training vs. validation scenes under either split, and no motion model loss curves or overfitting analysis are provided. Table 1 shows the proposed split (row 1: minADE 0.6308) improves over the default (row 3: 0.6839), but row 4 (sub-50% split: 0.6373) achieves nearly identical performance without spatial disjointness. This suggests the improvement comes primarily from avoiding inference on the map model's training data, not from removing spatial overlap—an important distinction that the paper does not disentangle. The paper merges two distinct problems (spatial overlap and train-set inference) without isolating their effects.

- **Boundary-free baseline is underspecified and its claimed mechanism is not isolated.** Equation 1 provides only a single deformable attention formula with minimal architectural details (number of sampling points, heads, resolution levels, backbone details, how BEV map info and image features are combined, occlusion handling). More critically, the paper claims the baseline specifically "addresses the issue of agents extending beyond the map's perceptual boundaries," but Table 7 shows improvements on both close AND far agents (e.g., MapTRv2-CL+HiVT: close minADE drops from 0.5585 to 0.5175, far from 0.6999 to 0.6274). Without an ablation applying image features only to far agents or comparing to the same deformable attention applied to BEV features within map range, it is impossible to determine whether the gains come from solving the range mismatch or from simply adding a more powerful sensory pathway. The causal attribution remains unestablished.

- **Overclaiming relative to evidence.** The paper repeatedly frames the existing protocol as suffering from "misconceptions," "mis-usage," and "long-standing" problems that its benchmark "solves." The evidence supports that OMMP-Bench is a more demanding and informative evaluation, but it does not demonstrate that prior conclusions from Gu et al. were fundamentally wrong—merely that they were narrower in scope. For instance, the paper does not show whether method rankings change under its proposed metrics (a key indicator of whether prior evaluations were misleading). The gap between the rhetorical framing and the empirical support is a structural issue.

### Minor:

- **Limited model diversity.** Only two motion prediction models (HiVT, DenseTNT) and two map models (MapTR, MapTRv2-CL) are evaluated. These are relatively older models; newer architectures like MTR++, QCNet, or StreamMapNet are absent, making it unclear whether findings generalize.

- **Small validation set and no distribution analysis.** The motion validation set has only 86 scenes. No analysis is provided on whether this split maintains similar scenario distributions (intersections, weather, traffic density) as the original nuScenes split, or whether 86 scenes provide statistically stable rankings.

- **"Close" vs. "far" threshold lacks precise definition.** The paper states agents are classified by whether they fall "within the perception range of online mapping models," but the exact distance computation and threshold per agent are not specified, and no sensitivity analysis to this threshold is provided.

- **Not evaluating online map accuracy directly.** The benchmark explicitly skips evaluating map quality (stating output forms vary across models) and focuses only on downstream motion prediction. This limits the ability to attribute performance differences to online mapping quality vs. motion model design or fusion strategy.

### Trivial:
- Minor grammatical issues throughout (e.g., "We do not evaluate" stated as "We no not evaluate" in Appendix A).

## Nice-to-Haves

- Running motion prediction with GT maps at varying ranges for close vs. far agents separately would provide an upper-bound analysis isolating whether the core issue is missing maps or inherent difficulty of far agents.
- Ablation applying image features only to far agents vs. all agents would clarify the claimed mechanism.
- Reporting map quality metrics (e.g., mAP) under different splits would directly validate the train-val gap claim rather than relying on indirect evidence from motion prediction endpoints.
- Including more recent motion prediction models (MTR++, QCNet) would strengthen benchmark credibility.
- Computational cost analysis (latency, memory) for the image feature baseline would help assess practical viability.

## Removed Points

These points are flagged to be removed; treat them with caution.

- **"The paper uses old baselines (HiVT and DenseTNT), which are too old"** — HiVT (CVPR 2022) and DenseTNT (ICCV 2021) are reasonable, widely-used baselines in this domain. While newer models exist, this is a minor weakness (generalizability concern), not a fatal flaw. Downgraded to Minor.

- **"Significant training data reduction not discussed"** — The new split uses 367+397 = 764 scenes for training (vs. nuScenes' original ~700), so the total training data is actually comparable. The concern about reduced map training data (367 scenes) is valid but moderate; many online mapping papers train on subsets. This is a valid observation but not a major weakness.

- **"Only evaluated on nuScenes"** — The paper explicitly acknowledges this and explains why (nuScenes is the only dataset with all required modalities). This is a scope limitation, not a methodological flaw. Moved to Nice-to-Have.

- **"No comparison with simply expanding the map perception range"** — The paper actually does address this: Table 2 shows map accuracy degrades with expanded range, and Table 3/Fig. 6 shows that long-range GT maps help but long-range predicted maps don't. The specific ablation of running motion prediction WITH expanded-range online maps is missing but Table 2's mAP drop suggests this would be unproductive. The analysis is incomplete but the direction is covered.

- **"Should compare against end-to-end methods like UniAD"** — The paper explicitly scopes this out (Section 3.1, "the study of online mapping's influence... introduces additional complexity by coupling motion prediction performance with that of object detection and tracking"). This is a reasonable scoping decision.

- **"Formatting/style nitpicks"** — Removed per hard rules.

- **"No confidence intervals or statistical testing"** — Single-run evaluation is the norm in this community for large-scale autonomous driving benchmarks. Reporting variance across multiple runs would be nice but is not standard practice. Moved to Nice-to-Have.

## Novel Insights

The most insightful finding is the fundamental tension between online mapping range and motion prediction requirements: current online map models cannot simultaneously provide both accurate and extended-range coverage (Table 2: mAP drops from 0.124 to 0.014 when range expands from 30×60m to 100×100m). This is not just a scaling problem—it reveals a structural limitation of current camera-based BEV mapping approaches that cannot simply be engineered around. The observation that Table 1 row 4 (simple half-split without spatial disjointness) performs nearly as well as the proposed spatially-disjoint split suggests that the train-val gap issue is driven more by the map model performing inference on its own training data than by spatial overlap—a nuance the paper itself does not highlight but that emerges from its own experimental design.

## Suggestions

- **Directly measure and report map quality (mAP) under the default vs. proposed splits** to validate the train-val gap claim. Even a brief analysis showing the map accuracy distribution shift would substantially strengthen this core argument.
- **Ablate the boundary-free baseline** by applying image features only to far agents, and compare to a variant where equally many parameters of additional capacity are added to the base model. This would help isolate whether the improvement is specifically from range extension or merely from added model capacity.
- **Report method rankings under different metric configurations** (ego-only, all-moving, close-only, far-only) to demonstrate whether prior evaluations produced misleading conclusions, not just less informative ones.

## Score and Decision

Comparative calibration:
- ESDMotion (5,5,6,5): Similar domain (motion prediction with alternative maps), rejected for limited novelty, single dataset, and incremental methods. Score ~5.25.
- Leveraging Uncertainty for Trajectory Prediction (3,6,5,6): Directly related to online map uncertainty for trajectory prediction, rejected. Score ~5.
- CTBench (5,3,6,5): Benchmark paper correcting evaluation practices, rejected for limited scope and incomplete analysis. Score ~4.75.
- Uni-Map (6,8,6,5): Online HD map construction, rejected mainly for limited theoretical insight despite strong experiments. Score ~6.25.

This paper sits between the ESDMotion/CTBench tier (score ~5) and the Uni-Map tier (score ~6). It has genuinely useful benchmark contributions but with overclaimed evidence and underspecified technical contributions. The benchmark design decisions (split, metrics, agent selection) are reasonable and the identified problems are real, but the core claims about "solving misconceptions" and "eliminating train-val gaps" go beyond what the experiments establish. The technical novelty of the boundary-free baseline is incremental. Overall, this is a solid but not exceptional contribution—valuable for the community but needing stronger empirical grounding and more measured claims.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>