---
job_id: 8f94c4f0-3d9b-4176-bb9e-168634605a75
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: M14YpuTejd.pdf
paper: Understanding the Task and Data Misconceptions in Online Map Based Motion Prediction for Autonomous Driving and a Boundary-Free Baseline
main_score_norm: 0.4
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper is squarely in ML for autonomous driving, focusing on representation learning for online mapping and motion prediction, and proposes a benchmark and baseline; this is fully within ICLR’s scope (representation learning, benchmarks, applications to autonomy).

## Minimum Quality
Pass ✅.  
The paper is in English and has all core sections (Abstract, Introduction, Related Work, Method/Benchmark description, Experiments, Results/Analysis, Conclusion). The methodology is straightforward, with no obvious fatal theoretical or experimental flaws. While there are significant weaknesses (limited scope, modest novelty, missing related work), they do not reach the level of desk-reject.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I see no attempts to manipulate automated reviewing systems or hidden prompts; the content is a standard research paper.

---

# Expected Review Outcome:

## Summary

The paper revisits the recently proposed “online map based motion prediction” protocol and argues that current practice on nuScenes suffers from three main misconceptions: an unsuitable dataset split for the two-stage map→motion pipeline, misaligned spatial ranges between map prediction and motion prediction, and non-discriminative evaluation metrics. The authors introduce OMMP-Bench, a re-partitioned nuScenes-based benchmark with new evaluation protocols, and propose a simple “boundary-free” baseline that augments motion prediction with deformable attention over image features to mitigate missing-map issues for faraway agents. They further analyze the impact of different online map element types on downstream motion prediction.

## Strengths

1. **Clear identification of protocol issues, backed by concrete evidence.**  
   - The paper convincingly argues that the standard nuScenes train/val split is ill-suited for two-stage online map → motion prediction training, because the map model is both trained and inferred on the same “train” scenes while motion is evaluated with map predictions on unseen “val” scenes.  
   - **Figure 3** (upper vs lower) visually clarifies how the default protocol leads to a train–val gap (map model infers on its training set for motion training but on unseen data for evaluation), whereas their split disentangles map-train from motion-train/val. This figure significantly improves clarity of the data-flow argument.
   - **Table 1** empirically supports the argument: their “Split 1 (Ours)” improves minADE/FDE and MR over the default split 3, showing that aligning map accuracy distributions between motion train and val matters.

2. **Reasonable and concrete benchmark design improvements.**  
   - The re-partitioned OMMP-Bench split (map train / motion train / motion val) is spatially disjoint, addressing both train–val map overfitting and geographic leakage, which were highlighted in prior mapping work. **Figure 4** nicely visualizes the overlap issue in the original split versus the proposed one; the large fraction of overlapping regions in nuScenes’ official split is made very clear.
   - The evaluation focuses on non-ego moving agents and further distinguishes agents inside vs outside the map perception range. **Table 6** demonstrates that static agents trivially lower error metrics to near-zero (e.g., static minADE 0.002 for HiVT+MapTR), confirming that including them hides differences between models.

3. **Insightful analysis of perception-range mismatch.**  
   - The paper quantitatively shows that increasing the map perception range with current online mapping architectures severely degrades map quality (Table 2, mAP dropping from 0.164 to 0.002 for MapTRv2-CL).  
   - At the same time, motion prediction benefits from longer-range ground-truth maps (Table 3, modest but consistent minADE/FDE improvements when extending GT maps from 30×60 m to 100×100 m).  
   - Together with the visualization in **Figure 5** and **Figure 6**, this forms a coherent argument that current online mapping models cannot simply be naively extended in range to satisfy the needs of motion prediction, and that faraway agents indeed suffer from missing context.

4. **Simple, interpretable boundary-free baseline with consistent gains, especially for far agents.**  
   - The deformable-attention-based image feature augmentation is conceptually simple and easy to implement. Equation (1),  
     \[
     \hat{A}_i = \operatorname{DeformAtt}(A_i, p_i, I_{T(i)}),
     \]  
     is compact and clearly identifies the three inputs (agent feature, projected agent position, corresponding image feature map).
   - **Table 4** and **Table 7** show that this image-feature (“img”) baseline improves minADE over “base”, “unc” and “bev” in most settings, with more pronounced improvements for far non-ego agents. For example, in Table 7, MapTRv2-CL+HiVT for “Moving Non-Ego Far” improves minADE from 0.7242 (bev) to 0.6274 (img), and MapTR+HiVT improves from 0.6738 to 0.6318. This supports the claim that image features help compensate for missing map coverage.

5. **Useful analysis of which map elements matter.**  
   - **Table 5** provides an ablation on the impact of dividers, boundaries, pedestrian crossings, and centerlines. It shows that including centerlines alone gives strong performance (minADE 0.6631) and that combining all elements yields the best result (0.6308). This is a valuable practical insight for both online mapping and motion prediction communities, as it articulates that centerlines are particularly informative while other semantics give incremental gains.

6. **Reasonable baselines and simple experimental setup.**  
   - The paper evaluates HiVT and DenseTNT as motion models, and MapTR / MapTRv2-CL as map models, plus prior online-map-based methods “unc” (uncertainty) and “bev” (BEV feature) in **Table 7**. These are canonical and widely used components in the field, so results are interpretable and reusable.

## Weaknesses

1. **Limited scope: single dataset, narrow set of models, and lack of ablation on design choices in OMMP-Bench.**  
   - OMMP-Bench is entirely built on nuScenes; there are no experiments or even small-scale sanity checks on other motion datasets (e.g., Waymo Open Motion, Argoverse2) where trajectory data exist but HD maps differ. Even if full online mapping is not possible there, some cross-dataset analysis or discussion would help argue generality. Currently, the benchmark risks being seen as a re-split of nuScenes with modified metrics, rather than a broadly applicable protocol.  
   - Only two map models (MapTR and MapTRv2-CL) and two motion models (HiVT and DenseTNT) are used. For a benchmark paper, one would expect a somewhat broader set of motion models (e.g., at least one raster-based or GNN-based vector method) to show that the insights are not architecture-specific. **Table 7** covers four combinations, but they are all Transformer-style motion models and high-capacity vectorized map models; classical baselines are absent.  
   - The benchmark’s own design choices are only partially validated. For example, evaluation horizon is fixed to 3 seconds; the paper does not show results for different horizons (e.g., 5 s) or discuss why 3 s is preferred. Similarly, the choice of evaluating only vehicles (no pedestrians, cyclists) is not justified, even though nuScenes has multiple agent types. This weakens the claim that OMMP-Bench is a fully “well-defined” standard, rather than one particular viewpoint.

2. **The “train–val gap elimination” claim is somewhat overstated and not fully disentangled from simple data re-splitting effects.**  
   - **Table 1** is used to support the claim that the proposed split “eliminates the train–val gap”. However, the comparison between splits is confounded. Split 4 (“nuScenes Train (Sub 50%) / Another Sub 50%”) already partially decouples map-train from motion-train by random partitioning and achieves minADE 0.6373, close to the proposed split (0.6308) and better than the default split 3 (0.6839).  
   - This suggests that part of the benefit may come from simply avoiding map-inference on the map model’s training scenes for motion-training, not necessarily from the specific spatial disjoint design adopted here. The paper does not provide a more systematic analysis (e.g., gradually varying overlap percentage or comparing random vs geometric splits) so “eliminated train-val gap” feels too strong.  
   - Also, OMMP-Bench does not provide any explicit quantitative measure of train–val map prediction accuracy mismatch before/after the new split, beyond the final motion metrics. Directly reporting map accuracy (e.g., mAP or other consistent map metrics) on train vs val under different splits would better support the core narrative.

3. **Boundary-free baseline is very simple and its benefits, though consistent, are modest; analysis is shallow.**  
   - The image-feature baseline uses a single deformable attention layer (Equation (1)) but the paper does not probe design variants: number of attention layers, number of sampled points, per-view vs multi-view aggregation, or fusion strategies with map and agent features. Without these ablations, it is difficult to conclude that “image features are not constrained by perception range and could supplement environmental information” in a principled way, beyond the obvious fact that image crops cover larger FOV.  
   - The gains reported in **Table 4** are relatively small: for HiVT+MapTR, minADE improves from 0.6375 to 0.6163 (≈3.3% relative) and minFDE from 1.2592 to 1.2344. For HiVT+MapTRv2-CL, improvements are larger but still modest on global metrics. The paper emphasizes bigger relative improvements on far agents in **Table 7**, but does not provide detailed error breakdowns (e.g., histograms vs distance, failure mode visualizations) or qualitative examples beyond **Figure 7**, which only shows one or two scenes.  
   - There is no ablation comparing image-features-only vs map+image; it is unclear if the model is over-relying on appearance cues when maps are noisy. For example, it would be instructive to disable maps entirely and see how much of the performance can be recovered via image features alone. Without such experiments, practical guidance for practitioners is limited.

4. **Evaluation protocol leaves out important aspects of motion prediction and may introduce its own biases.**  
   - The benchmark evaluates only vehicles, and further only those that move more than 2 meters in 3 seconds. This “moving only” criterion, adopted from Argoverse/Waymo, is defensible, but the downstream consequences are not analyzed. For instance, heavily congested urban environments may have many “slow but critical” agents whose movement is marginally above or below 2 m; ignoring them may bias the benchmark toward free-flow conditions.  
   - There is no explicit handling of multi-agent interaction complexity, such as number of neighbors, intersection vs highway scenarios, lane merges, etc. **Table 6** only partitions by ego vs non-ego and close vs far; thus, the metrics may still pool fundamentally different interaction regimes. Given that the paper’s main theme is “misconceptions in task and data”, it is a missed opportunity not to go deeper on scenario conditioning.  
   - The benchmark only considers 3-second prediction horizons and minADE/minFDE/MR metrics. Other widely used metrics such as brier score, NLL, collision rate, or off-road rate are not reported. This is arguably acceptable for a first benchmark, but it does limit how comprehensively one can assess models, especially given that safety-oriented behaviors (e.g., less aggressive but safer trajectories) often manifest in those metrics.

5. **Mathematical and algorithmic description is terse; some details are underspecified.**  
   - The core new methodic component, deformable attention over image features (Sec. 3.3), is only given by Equation (1) with a high-level description of projections. Crucial implementation details are missing from the main paper:  
     - How many sampling points per query are used in DeformAtt?  
     - Are multi-scale image features used (like in Deformable DETR), or just a single feature level?  
     - How exactly is \(p_i\) represented (normalized coordinates, camera index, depth assumption)?  
     - How are multi-view ambiguities handled when an agent is visible in multiple cameras (currently T(i) returns a single view)?  
   - These omissions make it difficult to reproduce the baseline from the main paper alone and to interpret its computational cost relative to the BEV-feature method of Gu et al. (2024b). For a benchmark paper proposing a “new baseline”, more algorithmic clarity and a brief complexity analysis would strengthen the contribution.

6. **Some claims about “all map elements” and their usage in prior work are oversimplified.**  
   - In Sec. 3.5 and **Table 5**, the authors conclude that feeding all map elements is best, and they criticize Gu et al. (2024a) for using only one element type at a time. However, they do not quantify the cost tradeoffs (memory, runtime) nor discuss whether some tasks or environments may actually benefit from a subset, especially when maps are noisy.  
   - The observation that centerlines are most beneficial is interesting, but the model architecture is tuned only once; there may be confounding effects where certain heads or embedding dimensions are optimized more for centerlines. An additional experiment where the map encoder capacity is fixed per total number of vectors (not per semantic type) would disentangle semantic richness from capacity.  
   - Also, OMMP-Bench’s protocol disallows offline maps “in any form” (Appendix A), which is reasonable for the online-map study, but this design choice is not contrasted with hybrid setups (offline+online) that might be more realistic in practice.

7. **Missing and incomplete related work coverage, especially on recent online map / SD-map based motion prediction.**  
   - The related work section largely focuses on classic motion prediction (VectorNet, LaneGCN, HiVT/MTR) and early online mapping models (HDMapNet, VectorMapNet, MapTR, StreamMapNet). It omits several very directly relevant works that also explore online or low-quality map based motion prediction, structured uncertainty in online maps, or SD map based forecasting (see “Potentially Missing Related Work” below). Given that the paper is positioned as a “clarification of misconceptions” in a growing subfield, a more complete literature treatment is necessary.  
   - In particular, recent works that also tackle range limitations or exploit online-map memory are highly relevant to the proposed boundary-free baseline and should be discussed, both conceptually and empirically (even if full baselines are difficult).

8. **Benchmark status and reproducibility details are deferred, and some design decisions are opaque.**  
   - The paper states that “we will open source the related code and checkpoints,” but at review time this is a promise, not a fact; no URLs or anonymized repository info are provided.  
   - The process for manually splitting nuScenes into map-train / motion-train / motion-val is only briefly described and not formally specified in the main text. **Figure 4** is helpful, but precise rules (distance thresholds, handling of overlapping tiles, treatment of rare cases) are not included, raising concerns about reproducibility and exact comparability.  
   - Appendix A mentions that OMMP-Bench does not enforce or even measure map metrics, instead evaluating map models only via downstream motion. While this is a design choice, it removes an important diagnostic tool: users cannot tell whether a motion model’s failure is due to a bad map model, a bad motion model, or their interaction.

Overall, these weaknesses mean the work is valuable as an early attempt at cleaning up the protocol, but it falls short of being a definitive benchmark or a strong methodological paper.

## Potentially Missing Related Work

1. **Zhang, Z., Jia, X., Li, Q. (2025). “Revisiting Standard-Definition Map Based Motion Prediction in the Era of End-to-End Autonomous Driving.”**  
   - This work studies motion prediction with standard-definition maps and discusses how lower-fidelity mapping can still support accurate forecasting. It is directly related to the authors’ focus on suboptimal or partial map inputs. It should be cited in the Related Work section and discussed as an alternative perspective on “relaxing HD map assumptions,” possibly in Sec. 2 and/or Sec. 3.3.

2. **Liao, W., Chen, X., He, T. (2025). “Enhanced Motion Prediction by Online Mapping Memory.”**  
   - This paper explicitly addresses leveraging online mapping memory to improve motion prediction, which is highly pertinent to the authors’ argument about misaligned ranges and online map quality. It should be discussed in the “Online Map Based Motion Prediction” paragraph, and compared conceptually to the proposed image-feature baseline (both aim to mitigate limitations of online maps, but via different mechanisms).

3. **Zhang, G., Lin, J., Wu, S. (2023). “Online Map Vectorization for Autonomous Driving: A Rasterization Perspective.”**  
   - Presents an online map vectorization framework that bridges raster and vector representations. Since the authors discuss different online map output formulations (Sec. 3.5), this work is directly relevant and should be cited there, with a short discussion on how alternative representations might interact with motion prediction.

4. **Upreti, M., Girijal, R., B A, N. (2026). “Map-Less Yet Accurate: Trajectory Prediction for Traffic Agents Using Online HD Map Reconstruction for Autonomous Driving.”**  
   - Proposes a trajectory prediction framework that reconstructs HD maps online, closely aligned with the overall topic of the paper. It should be cited in Sec. 2 and contrasted with OMMP-Bench: for example, how their protocol handles or avoids train–val leakage, and whether they also confront range limitations.

5. **Zhang, Z., Jia, X., Li, Q. (2024). “ESDMotion: End-to-end Motion Prediction Only with SD Maps.”**  
   - Introduces motion prediction based on SD maps, again relaxing dependence on perfect HD maps. It should be referenced when discussing the broader landscape of “weaker map input” approaches and in comparison to the idea of supplementing online HD maps with image features.

6. **Gogoi, P., Janjoš, F., Yang, B. (2026). “Uncertainty Matters: Structured Probabilistic Online Mapping for Motion Prediction in Autonomous Driving.”**  
   - Focuses on probabilistic online mapping and explicitly targets motion prediction as a downstream task. This is strongly related to both MapUncertaintyPrediction (Gu et al., 2024a) and OMMP-Bench’s emphasis on the quality of online maps. It should be cited in Sec. 2 and discussed as an alternative to simply changing data splits and metrics.

These works should be integrated into the Related Work section and, where appropriate, into the discussion and conclusion, clarifying how OMMP-Bench differs and what unique insights it offers compared to these concurrent ideas.

## Questions

1. **Quantifying the train–val map accuracy gap.**  
   - Can you provide explicit map accuracy metrics (e.g., mAP, IoU, or vector-level metrics) for the map model on (a) map-train scenes, (b) motion-train scenes, and (c) motion-val scenes, under both the default split and OMMP-Bench split? This would directly show how much the gap is reduced by your partition.

2. **Details of the deformable attention baseline.**  
   - Please clarify:  
     - How many sampling points per query does DeformAtt use, and at how many feature scales?  
     - How is \(p_i\) represented numerically (e.g., normalized coordinates in [0,1]^2, depth assumptions)?  
     - How do you choose T(i) if an agent is visible in multiple cameras? Have you tried aggregating across all visible views?  
   - A small ablation on these design options could significantly strengthen the methodological contribution of the “img” baseline.

3. **Maps-only vs image-only vs combined fusion.**  
   - Have you evaluated models that use only image features (no online map) and models with both map + image features vs map-only? Such a comparison would help understand how much of the benefit comes from mitigating mapping range vs adding extra raw context.

4. **Sensitivity to the “moving > 2 m in 3 s” threshold.**  
   - How sensitive are your conclusions to the 2 m cutoff? For example, if you change it to 1 m or 3 m, do the relative rankings in **Table 7** remain stable? Providing at least one sensitivity experiment or analysis would make the benchmark definition more robust.

5. **Generalization beyond nuScenes.**  
   - Do you foresee any obstacles to applying OMMP-Bench-style splits and metrics to other datasets such as Waymo or Argoverse2 once suitable raw sensor data is available? Could you outline which parts of your design are nuScenes-specific (e.g., perception range, semantic classes, log structure) and which are generic?

6. **Complexity and runtime comparison for baselines.**  
   - What is the runtime and memory overhead of the image-feature baseline compared to “unc” and “bev”? Especially for far agents, deformable attention over full-resolution feature maps could be costly. Providing some numbers would help practitioners decide whether the gains in **Table 7** are worth the added cost.

Clarifying these points in the rebuttal and, ideally, adding some targeted experiments would make me more positive about both the benchmark and the proposed baseline.

## Flag For Ethics Review

No ethics review needed.

## Details Of Ethics Concerns

N/A.

## Soundness Rating

2: fair.  
The empirical methods and comparisons are broadly sound and internally consistent, and Equation (1) is technically reasonable, but several key claims are under-analyzed (e.g., elimination of train–val gap, superiority of the new baseline), and important implementation details are missing from the main text.

## Presentation Rating

3: good.  
The paper is generally well written and organized; figures like **Figures 3, 4, 5, 6, 7, 8** and tables like **Tables 1, 4, 5, 6, 7** substantially help understanding. However, some methodological details are too terse, and related work coverage is incomplete.

## Contribution Rating

2: fair.  
The benchmark corrections (split, metrics) and the identification of protocol issues are useful, but they are relatively incremental re-interpretations of an existing setup on a single dataset. The boundary-free baseline is simple and provides modest gains without deep analysis. Overall impact is moderate rather than strong.

## Overall Rating

4: Marginally below the acceptance threshold. But would not mind if paper is accepted.  
The paper raises valid and timely concerns about how online map based motion prediction is currently evaluated, and OMMP-Bench plus the image-feature baseline provide some practical improvements. However, the scope is limited to nuScenes, the benchmark design choices are not thoroughly validated, the methodological novelty is modest, and several important recent related works are missing. With stronger analysis, broader validation, and clearer positioning, this line of work could become a solid contribution, but in its current form it falls just short of ICLR standards for the main track.

## Reviewer Confidence

4: confident.  
I am familiar with motion prediction and online mapping literature, have checked the equations and experimental protocol carefully, and feel reasonably certain about the assessment, though some implementation details not in the main text could alter fine-grained judgments.