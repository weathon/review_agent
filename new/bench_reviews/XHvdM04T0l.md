Now I have enough calibration context. Let me compose the final consolidated review.

## Summary
The paper proposes G-SFormer, an architecture for 2D-to-3D pose lifting that combines a Part-based Adaptive GNN (grouping joints into 5 body parts with learned graph topology) for spatial modeling and a Skipped Transformer (performing self-attention on stride-separated framesets) for temporal modeling. The method achieves competitive MPJPE on Human3.6M, MPI-INF-3DHP, and HumanEva with dramatically lower FLOPs compared to heavy transformer baselines like MixSTE and KTPFormer.

## Strengths

- **Genuine and dramatic efficiency improvement over heavy baselines.** Table 2 shows G-SFormer-L achieves 40.5mm MPJPE with 2,366M FLOPs vs. KTPFormer's 40.1mm at 278,119M FLOPs (~0.85% cost) and STFormer-L's 40.5mm at 156,392M FLOPs. These are real and substantial efficiency gains that matter for practical deployment.

- **Consistent performance across three benchmarks and multiple model scales.** G-SFormer variants (S, standard, L) achieve competitive or near-SOTA results on Human3.6M (Table 1), MPI-INF-3DHP (Table 3), and HumanEva (Table 4), with comprehensive evaluation under two protocols (MPJPE and P-MPJPE).

- **Sound architectural design that addresses a real bottleneck.** The Skipped Self-Attention mechanism reduces the quadratic complexity of temporal attention from O(T²) to O(T²/m), and the Part-based GNN reduces spatial computation from O(J²) to O(N_p²) where N_p=5. These are well-motivated reductions that preserve global structure while cutting cost.

- **Meaningful ablation studies.** Table 6 provides ablations over spatial module variants (Spatial-MLP, Joint-wise GCN), temporal variants (VT-Conv, VT-StridedConv), and skip factor m, showing consistent degradation when proposed components are replaced. Table 5 evaluates data completion strategies.

## Weaknesses

### Fatal
None.

### Major

- **The "robustness" claim in the title and abstract is not quantitatively substantiated.** The paper title promises "efficient and robust" estimation, the abstract claims "outstanding robustness to inaccurately detected 2D poses," and Section 4.3.2 is titled "Qualitative Comparison for Robustness." However, the only evidence for robustness is visual comparison on in-the-wild videos (Figs 1b, 5) with no controlled quantitative experiment. There is no evaluation with synthetic noise injection (Gaussian perturbation, random joint dropout, left-right flips), no evaluation on benchmarks containing noisy 2D detections with ground truth 3D labels, and no comparison of error-vs-noise-level curves across methods. For a claim placed in the paper's title, this is a significant evidentiary gap.

- **The "~1% computational cost" claim is selectively framed.** The abstract and conclusion state the model uses "around 1% computational cost" compared to "the state-of-the-art methods." This is accurate relative to the heaviest baselines (KTPFormer at 278K MFLOPs, MixSTE at 27K MFLOPs), but misleading relative to efficiency-focused methods already in the literature: P-STMO 81f (+PT) uses 493M FLOPs vs G-SFormer-S's 361M FLOPs — a ratio of ~73%, not 1%. Similarly, TM2P 81f uses 392M FLOPs. The paper does present Table 2 with all these numbers, so it is transparent, but the headline claim in the abstract glosses over this nuance and creates an impression that all SOTA methods are two orders of magnitude more expensive, which is not the case. This should be clearly qualified.

- **The Part-based Adaptive GNN is not fully "adaptive" and its design is not adequately ablated.** The paper claims "a flexible graph topology is learned by a totally data-driven approach" (Section 3.1), but the 5-part grouping of joints is hardcoded, not learned. Only the 5×5 adjacency matrix over these fixed parts is learned. This is partial adaptivity at best. Additionally, there is no ablation over the number or definition of parts (e.g., 3, 7, or alternative groupings), making it unclear whether the hand-chosen 5-part partition is near-optimal or arbitrary. Table 6 shows replacing the part-based GNN with a simple Spatial-MLP only costs 0.9mm, suggesting the contribution of the graph structure specifically (vs. just mixing features) is modest.

### Minor

- **The joint-wise GCN ablation (Table 6, row 2) performs worse than Spatial-MLP (44.4mm vs 43.6mm), which is unusual and unexplained.** The paper attributes this to "redundant spatial connections," but an alternative explanation is that the joint-wise GCN was not properly tuned. If the comparison baseline itself is suboptimal, the relative improvement of the part-based design is less convincing.

- **SSA's attention sparsity pattern has an unacknowledged limitation.** SSA performs self-attention only within each stride-separated frameset; frames with indices that differ by less than m but are in different residue classes never directly attend to each other. Cross-residue interaction must be mediated through the MLP and stacked layers. The paper does not discuss this architectural limitation, which likely explains the performance degradation for large m (Table 6, m=7,9).

- **Pre-training and reprojection refinement are mixed into headline comparisons without sufficient clarity.** In Table 1, G-SFormer-L (+PT†) with reprojection refinement achieves 39.9mm and is described as outperforming KTPFormer (40.1mm). However, it is unclear whether KTPFormer uses pre-training or reprojection. The "+PT" and "†" markers are present, but the comparison narrative does not explicitly flag the asymmetry.

- **Evaluation is limited to indoor/lab benchmarks.** No evaluation on in-the-wild datasets like 3DPW. While the qualitative Fig. 1b shows in-the-wild results, quantitative evaluation on such datasets would strengthen the generalization argument.

### Trivial

- The Data Rolling and Sinusoidal Positional Encoding contribute only 0.15–0.23mm improvement (Table 5). While parameter-free, the practical impact is minimal.

- In Eq. (2), the use of |f_{pi} + f_{pj}| (absolute value of sum) rather than a more standard attention mechanism (dot product, difference) is not discussed or justified.

## Nice-to-Haves

- Quantitative robustness experiments with varying noise levels (Gaussian perturbation, random joint dropout) comparing G-SFormer against baselines — this would substantiate the title's "robust" claim.
- Ablation over different part groupings (number and definition of parts) to validate the 5-part design.
- Wall-clock inference time comparisons, not just FLOPs, to confirm practical efficiency gains (especially for SSA, which involves non-contiguous memory access patterns).
- Comparison with other efficient attention mechanisms (e.g., windowed attention, token pruning) beyond convolution-based striding.

## Removed Points

- **"Not yet released" / reproducibility concerns about models, baselines, or AMASS pretraining** — The paper cites these as existing; per review rules, we treat all cited entities as real and available.

- **Demand for complete hyperparameter disclosure in the main text** — The paper references detailed settings in the Appendix. This is standard and not a weakness.

- **Formatting/style nitpicks** — Removed per rules.

- **Claim that the paper should compare against P-STMO at a matched budget to prove efficiency** — The paper already presents Table 2 with P-STMO and TM2P in the same table, showing their accuracy and cost alongside G-SFormer. Readers can judge the trade-off. Demanding budget-matched comparisons with every lightweight baseline is scope creep beyond what's reasonable.

- **Spark reviewer's "FLOPs for KTPFormer seem anomalously high, verify"** — Per rules, we accept cited numbers from the paper and do not flag unverifiable concerns.

- **Spark reviewer's "compare against parameter-matched baselines (e.g., standard Transformer with comparable parameter count)"** — This is essentially what the VT-Conv/VT-StridedConv ablations in Table 6 already do. Requesting additional matched baselines beyond the existing ablations is excessive.

- **Neutral reviewer's "investigate adaptive grouping (e.g., Slot Attention)"** — This is a nice-to-have suggestion, not a weakness of the current work.

## Novel Insights

The Skipped Self-Attention mechanism reveals an interesting trade-off in temporal modeling for pose sequences: processing stride-separated framesets independently captures long-range periodic structure but inherently sacrifices fine-grained local temporal continuity. The fact that m=3 works best (Table 6) — balancing global context from stride-3 subsequences with sufficient local information through stacked layers and the joint residual pathway — suggests that skeletal motion may have a natural "temporal resolution budget" that can be allocated differently than standard dense attention. This insight could generalize to other sequential dense prediction tasks where temporal redundancy is high.

## Suggestions

- Add a quantitative robustness evaluation: inject Gaussian noise (σ ∈ {0, 5, 10, 20, 40}mm) and random joint dropout (0%, 10%, 20%, 30%) into CPN-detected 2D inputs, and report MPJPE curves for G-SFormer vs. MixSTE and PoseFormerV2. This directly addresses the core "robustness" claim.
- Qualify the "1% computational cost" statement in the abstract to specify that it refers to comparison with the heaviest transformer-based SOTAs, and acknowledge that efficiency-focused prior methods like P-STMO operate in a similar FLOPs regime.
- Ablate the number and definition of body parts (e.g., 3, 5, 7 parts) to validate this design choice.

## Score and Decision

**Calibration anchors:**
- PrML (3D HPE, small improvements, novelty concerns): scores 3/6/6, rejected
- Human Pose via Parse Graph (body structure modeling, missing ablations on parse graph): scores 3/3/5/3, rejected
- Efficient Multi-Task Transformer for 3D Face Alignment (efficiency claims without runtime, marginal improvements): scores 5/5/5/5, rejected
- CHAMP (3D HPE, diffusion-based, solid but incremental, indoor-only): scores 6/6/8/6, accept poster

G-SFormer is clearly above PrML and Parse Graph — it has a coherent and well-specified architectural contribution with dramatic efficiency gains demonstrated across 3 benchmarks. It is somewhat comparable to CHAMP in terms of domain and experimental scope (indoor benchmarks, competitive improvements). However, G-SFormer has a more significant weakness: the "robustness" claim in its very title lacks quantitative evidence, and the "1% cost" framing is misleading in its generality. These issues are more substantial than CHAMP's concerns.

The paper makes a genuine and valuable efficiency contribution, but overclaims on two of its three pillars (robustness has no quantitative evidence; efficiency framing is selective). The accuracy contribution is competitive but not decisively SOTA. I place this below CHAMP (which had ~6 average scores) but above the PrML/Parse Graph papers (which had ~3-5 scores). A score of 5 reflects a paper with real engineering merit but overclaimed contributions that need to be toned down or substantiated.

MY FINAL SCORE: <pineapple>5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>