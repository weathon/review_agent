=== CALIBRATION EXAMPLE 43 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- **Accuracy:** The title accurately reflects the core contribution: a hierarchical self-supervised framework (JEPA-based) for trajectory similarity.
- **Clarity of Problem/Method/Results:** The abstract clearly states the limitation of single-scale trajectory embeddings, introduces the three-layer hierarchical JEPA architecture, and reports strong zero-shot and similarity search performance.
- **Supported Claims:** The claim of a "first architecture to explicitly unify both fine-grained and abstract trajectory patterns within a single model" is slightly overstated. Multi-scale representation learning exists in trajectory literature (e.g., hierarchical attention networks, multi-granularity contrastive learning). The novelty lies specifically in the *JEPA-based* hierarchy with attention-weight injection, not the mere existence of multi-scale modeling. The abstract should temper this to avoid implying no prior work has attempted hierarchical trajectory representations. Otherwise, the claims align with the results in Table 1.

### Introduction & Motivation
- **Motivation & Gap:** The gap is well-motivated: trajectory data inherently contains local kinematic details and global mobility patterns, yet most SSL frameworks produce flat, single-scale embeddings. The transition to self-supervised trajectory learning and the limitations of existing contrastive/JEPA baselines are clearly articulated.
- **Contributions:** The three contributions are stated. However, Contribution 1 ("first architecture to explicitly unify...") repeats the overclaim from the abstract. Contribution 2 accurately describes the technical novelty (top-down attention injection). Contribution 3 accurately summarizes the empirical scope.
- **Claims Calibration:** The introduction over-claims novelty by not sufficiently acknowledging that hierarchical representation learning via top-down or bottom-up fusion is a well-established paradigm in CV (Feature Pyramid Networks, Hierarchical ViTs) and even in some trajectory models. The paper should frame its contribution more precisely: adapting JEPA's predictive objective to a multi-granular trajectory setting with a novel *attention-coefficient injection* mechanism, rather than claiming the hierarchy itself is unprecedented.

### Method / Approach
- **Clarity & Reproducibility:** The pipeline (Conv1D+Pool $\rightarrow$ 3-level abstractions $\rightarrow$ JEPA context/target/predictor branches $\rightarrow$ loss) is structurally clear. Appendix A.3 provides hyperparameters, aiding reproducibility.
- **Key Assumptions & Justification:** The model assumes 1D convolutions with stride-2 pooling adequately capture mesoscopic patterns (Eq. 3-5). While reasonable, no ablation justifies this choice over temporal pooling or clustering-based aggregation. More critically, the hierarchical interaction mechanism (Eq. 12-14) assumes attention coefficient matrices can be treated as 2D spatial maps amenable to bilinear interpolation. Attention weights represent discrete token-to-token dependencies, not continuous spatial densities. Interpolating them to match differing sequence lengths ($n^{(l-1)} \neq n^{(l)}$) breaks the probabilistic row-sum constraints of softmax attention and mixes semantically distinct token indices. This design choice lacks theoretical or empirical justification.
- **Logical Gaps:** 
  - Eq. 14 uses a single learnable scalar $\sigma$ to weight the interpolated high-level attention against the low-level attention. Attention matrices vary significantly across heads, layers, and trajectory segments. A global scalar gating mechanism is overly simplistic and may not adaptively modulate local vs. global focus. A position-wise or head-wise gating mechanism would be more principled.
  - The gradient flow during EMA updates of target encoders vs. training of context encoders is mentioned in Appendix A.3 but omitted from the main text, obscuring the exact JEPA training dynamics for readers.
- **Edge Cases/Failure Modes:** The method assumes trajectories are processed as fixed-sequence inputs after grid tokenization. How the model handles highly irregular sampling rates or missing segments (beyond the artificial distortion in Section 4) is not analyzed. If MaxPool1D drops critical anchor points, the abstraction hierarchy may misalign with ground truth semantics.

### Experiments & Results
- **Testing Claims:** The experiments directly test the core claims: similarity computation (Table 1), robustness to downsampling/distortion (Table 1), zero-shot transfer (TKY/NYC/AIS(AU)), and downstream fine-tuning (Table 2). The evaluation protocol aligns with ICLR standards for representation learning papers.
- **Baselines:** TrajCL, CLEAR, and T-JEPA are strong, recent, and directly relevant. Comparing against T-JEPA is particularly important as it establishes the value of adding hierarchy to JEPA.
- **Missing Ablations:** 
  - The primary technical novelty is the *attention upsampling & injection* mechanism. The ablation (Table 3) compares it only to direct embedding concatenation (`HiT_emb`) and removing interaction (`HiT_no_attn`). A crucial missing baseline is **cross-attention** between levels, which is the standard, more principled way to fuse hierarchical features in Transformers. Without this, it's unclear if attention interpolation is genuinely better than standard cross-level feature alignment.
  - No ablation on the choice of interpolation method (bilinear vs. nearest vs. learnable upsampling layers).
- **Statistical Rigor:** Results are reported as single-point estimates. ICLR increasingly expects variance metrics (std/dev over multiple random seeds) or confidence intervals for deep learning experiments, especially when claims of "superior zero-shot performance" hinge on marginal differences (e.g., GeoLife results where HiT-JEPA is ~2.8% worse than T-JEPA).
- **Claims vs Results:** The paper claims "comparative trajectory similarity search and remarkably superior zero-shot performance." Table 1 supports strong zero-shot performance, but on in-distribution tasks (Porto, GeoLife), gains are marginal or non-existent compared to T-JEPA. The post-hoc explanation that "relying on these features undermines generalization" (Section 4.1.1) to explain why TrajCL wins on Porto is speculative and not empirically validated within the paper.

### Writing & Clarity
- **Confusing Sections:** Section 3 ("Hierarchical interactions") is dense and notationally heavy. The transition from computing $A^{(l)}$ (Eq. 9-11) to upsampling it via bilinear interpolation (Eq. 13) and then fusing it with $A^{(l-1)}$ (Eq. 14) is difficult to follow. It is unclear whether $A^{(l)}$ refers to raw attention logits or post-softmax weights. If post-softmax, interpolation will destroy the unit-sum property, making the injected "spotlight" mathematically inconsistent with standard attention.
- **Figures/Tables:** Figures 4 and 5 are informative for interpretability but rely on subjective labeling ("origin anchoring", "destination intent") without quantitative validation (e.g., correlation with human-annotated waypoints or trip purpose). Table 2 is clear for downstream fine-tuning, though the averaging methodology across different metrics (HR@5, HR@20, R5@20) across spatial distances should be explicitly defined (arithmetic mean?).

### Limitations & Broader Impact
- **Acknowledged Limitations:** The authors correctly note that their hierarchical interaction method is specific to Transformer-based JEPAs and suggest future work generalizing to CNNs/Mambas. The ethics statement appropriately addresses location privacy via hexagonal grid blurring.
- **Missed Fundamental Limitations:**
  1. **Discretization Bias:** The model depends heavily on H3 hexagonal grid resolution. Sensitivity analysis is missing. Fine resolutions increase vocabulary size and sparsity; coarse resolutions lose local details. The hierarchical design does not mitigate this fundamental tokenization bottleneck.
  2. **Quadratic Complexity Scaling:** While the paper notes 1-layer Transformers are efficient (Appendix A.7), applying self-attention at three hierarchical levels still scales quadratically with sequence length. Scaling to trajectories >500 points (common in long-haul logistics or maritime tracking) is not discussed.
  3. **Heuristic Interaction Mechanism:** The core limitation is the lack of a learnable, structurally sound fusion mechanism for attention weights. The current approach is a heuristic approximation that may not generalize across diverse trajectory modalities (e.g., pedestrian vs. vessel).
  4. **Societal Impact:** Beyond privacy, the paper misses discussion of potential misuse in surveillance or predictive policing, especially given the zero-shot transfer capabilities across cities.

### Overall Assessment
HiT-JEPA presents a well-motivated empirical study on hierarchical self-supervised trajectory representation learning, with extensive evaluations across multiple datasets, robustness checks, and downstream fine-tuning. The experiments are comprehensive, and the zero-shot generalization results are compelling for the trajectory community. However, from an ICLR methodological perspective, the paper's core technical novelty—the bilinear interpolation and scalar weighting of attention coefficient matrices across hierarchical levels (Eq. 12-14)—lacks theoretical grounding and empirical justification against standard hierarchical fusion techniques like cross-attention or feature pyramid alignment. Treating discrete token-attention maps as interpolatable 2D fields breaks the probabilistic interpretation of attention and raises concerns about the inductive bias of the model. Additionally, the absence of statistical variance reporting and the lack of a cross-attention ablation weaken the empirical claims. The contribution stands as a strong empirical benchmark for trajectory SSL, but the methodological soundness of the key architectural design needs to be either rigorously justified, compared against principled alternatives, or tempered in the claims to reflect its heuristic nature.

# Neutral Reviewer
## Balanced Review

### Summary
The paper proposes HiT-JEPA, a hierarchical self-supervised trajectory representation framework that jointly captures fine-grained point-level details and high-level global semantics using a three-layer Joint Embedding Predictive Architecture. The model introduces a "top-down spotlight" mechanism that upsamples and blends high-level attention maps into lower layers to guide multi-scale feature learning. Extensive experiments demonstrate strong performance in trajectory similarity search, robustness to noise/downsampling, and zero-shot generalization across urban and maritime datasets.

### Strengths
1. **Clear motivation and well-structured architecture:** The paper effectively identifies the single-scale limitation of existing trajectory models and addresses it with a principled three-level hierarchy (point, segment, route) combined with JEPA's latent-space prediction paradigm, avoiding noisy point-level reconstruction.
2. **Comprehensive empirical evaluation:** Results cover standard self-similarity, robustness to varying downsampling/distortion rates (Table 1), and downstream fine-tuning against multiple heuristic distance metrics (Table 2). The zero-shot transfer experiments to TKY, NYC, and AIS(AU) demonstrate strong cross-domain generalization.
3. **Rigorous ablation and hyperparameter analysis:** Ablations validate critical design choices, particularly showing that direct embedding concatenation across levels causes representation collapse, while attention-blending preserves performance (Table 3). Sensitivity analysis for loss weighting coefficients (Tables 5-7) shows the model's stability.
4. **Strong reproducibility practices:** The paper includes a complete code repository, detailed hyperparameters, optimization schedules, dataset statistics, and explicit training/inference protocols (Appendices A.1-A.4), fully aligning with ICLR reproducibility expectations.

### Weaknesses
1. **Hierarchical interaction mechanism lacks comparative baselines:** The upsampling and linear blending of attention weights across levels (Eq. 13-14) is intuitively motivated but conceptually similar to feature pyramid fusion in vision. Without comparison to standard multi-scale fusion methods (e.g., cross-attention between levels, learned gating, or additive feature aggregation), it's unclear if this specific design is optimal or merely one of many viable options.
2. **Evaluation protocol may conflate temporal continuity with semantic similarity:** The self-similarity setup splits each trajectory into odd/even point sequences for query and database pairs. While common in the domain, this risks allowing models to exploit trivial adjacent-point continuity rather than learning true spatial mobility semantics, potentially inflating mean-rank scores.
3. **Incomplete efficiency and scalability analysis:** Table 8 reports training time per iteration, but omits critical metrics for modern ML submissions: parameter count, memory footprint, and inference latency. Running three parallel Transformer branches with predictors and VICReg regularization likely increases compute overhead, which matters for real-time trajectory applications.
4. **Zero-shot generalization drivers are under-analyzed:** The strong cross-domain performance is attributed to hierarchical abstraction, but alternative factors like H3 hexagonal tokenization, spatial invariance, or dataset-specific preprocessing are not disentangled. A controlled experiment isolating the tokenizer vs. hierarchy contributions would strengthen the claim.

### Novelty & Significance
- **Novelty:** Moderate to High. Adapting JEPA to trajectory data is a timely extension of recent T-JEPA work, and the explicit multi-scale hierarchy tailored for sparse/irregular GPS sequences is novel for the spatial-temporal domain. The attention-upsampling interaction is a practical engineering contribution but incremental relative to hierarchical fusion techniques in CV/NLP.
- **Clarity:** High. The paper is well-organized, mathematical notation is mostly consistent, and figures/tables directly support claims. Minor density in notation (e.g., multi-level index notation) could be eased with a short algorithmic pseudocode block.
- **Reproducibility:** High. All necessary details (code, splits, hyperparameters, hardware, training schedules) are provided. The reproducibility statement and ethical considerations meet ICLR standards.
- **Significance:** High for spatiotemporal/urban computing, moderate for general ML. The method addresses a clear gap in multi-scale trajectory representation and demonstrates strong zero-shot and downstream utility. While the broader ML community may view it as a domain-specific application of hierarchical self-supervised learning, the empirical rigor and cross-transfer results make it a valuable contribution to representation learning for irregular sequential data.

### Suggestions for Improvement
1. Add baseline comparisons for the hierarchical fusion mechanism, such as cross-level cross-attention, learned gating, or simple feature concatenation with layer normalization, to conclusively demonstrate the superiority of attention upsampling and blending.
2. Discuss the limitations of the odd/even trajectory split in the self-similarity evaluation and, if possible, include an alternative evaluation using independently sampled trajectory pairs or a small human-annotated similarity subset to verify semantic alignment beyond temporal continuity.
3. Report model size (total trainable parameters), GPU memory consumption during training/inference, and latency per trajectory to provide a complete efficiency profile, which is essential for assessing practical deployment feasibility.
4. Conduct a controlled ablation disentangling the contribution of H3 spatiotemporal tokenization from the hierarchical architecture in zero-shot settings (e.g., flat JEPA with H3 vs. hierarchical JEPA with H3 on check-in/maritime data).
5. Include a concise Algorithm 1 pseudocode summarizing the forward pass, hierarchical attention fusion step, and multi-level loss computation to improve readability and help ICLR reviewers quickly verify implementation correctness.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Zero-shot spatial embedding protocol isolation:** Explicitly report whether H3 node2vec embeddings are retrained or frozen for target cities (TKY/NYC/AIS); if retrained, the zero-shot claim is invalid, and if frozen, cross-city transfer over disjoint spatial graphs is geographically implausible.
2. **Strong multi-scale non-hierarchical baseline:** Compare HiT-JEPA against running T-JEPA independently at multiple sequence strides and concatenating the outputs to isolate whether gains stem from hierarchical attention fusion or merely from ingesting multi-scale inputs.
3. **Ablation portability beyond Porto:** Table 3 ablations are conducted exclusively on Porto; repeat these on T-Drive and GeoLife to prove the hierarchical interaction module generalizes to varying data densities, sampling rates, and trajectory domains.
4. **Inference latency and memory profiling:** Report inference throughput, peak VRAM usage, and parameter count, as the bilinear upsampling of attention matrices introduces non-standard computational overhead that challenges real-world scalability claims.

### Deeper Analysis Needed (top 3-5 only)
1. **Level-wise contribution decoupling:** Quantify the distinct impact of each abstraction level ($T^{(1)}$ vs $T^{(3)}$) on performance under specific perturbations to validate whether the hierarchy truly provides complementary robustness rather than redundant features.
2. **Collapse mechanism verification:** Provide latent space eigenvalue spectra or variance/covariance distributions for the direct concatenation variant versus HiT-JEPA to empirically prove *why* attention weighting prevents representation collapse while embeddings alone fail.
3. **Sensitivity to H3 resolution choices:** Analyze how varying the H3 hexagonal resolution affects retrieval quality to determine if reported gains are driven by the hierarchical architecture or by careful manual tuning of spatial discretization per dataset.

### Visualizations & Case Studies
1. **Failure mode reconstruction examples:** Show masked trajectories where predictions significantly diverg from ground truth, specifically illustrating cases where top-down attention misaligns or propagates noise, to expose model limitations.
2. **Quantitative cross-layer attention alignment:** Replace single-trajectory cherry-picked heatmaps with batch-level distributions of cross-layer attention similarity scores to statistically prove systematic coarse-to-fine semantic evolution across the hierarchy.
3. **Semantically annotated embedding projections:** Augment the t-SNE (Fig. 9) by coloring clusters with external attributes (e.g., trip duration, speed variance, or road network density) to confirm embeddings capture behavioral semantics rather than merely geometric proximity.

### Obvious Next Steps
1. **Applied downstream task validation:** Evaluate representations on real urban computing tasks beyond heuristic approximation, such as transport mode classification or map-matching, to demonstrate practical utility for deployment outside synthetic benchmarks.
2. **Explicit spatiotemporal tokenization:** Treat time intervals or velocity as separate tokens alongside spatial H3 embeddings, as the current architecture overemphasizes spatial geometry and struggles to justify robustness claims under highly irregular temporal sampling.
3. **Generalization of the interaction module:** Implement the proposed top-down hierarchical interaction with alternative backbones (e.g., Mamba or CNNs) to validate the interaction design as a general architectural contribution rather than an artifact specific to Transformer attention matrices.

# Final Consolidated Review
## Summary
This paper introduces HiT-JEPA, a hierarchical self-supervised framework for learning multi-scale urban trajectory representations by extending Joint Embedding Predictive Architectures (JEPA) to three levels of semantic abstraction: point, segment, and route. The model employs a top-down hierarchical interaction mechanism that upscales and blends attention weights from coarser to finer layers to guide local feature learning. Extensive experiments on diverse trajectory datasets demonstrate strong performance in similarity retrieval, robustness to spatial-temporal distortions, and notable zero-shot transfer across domains.

## Strengths
- **Comprehensive empirical evaluation demonstrating robustness and generalization:** The paper rigorously tests representations across varying database sizes, downsampling, and distortion rates, with results (Table 1) showing consistent stability under noise. Zero-shot transfer to check-in and maritime datasets, combined with strong downstream fine-tuning across multiple heuristic distance metrics (Table 2), provides compelling evidence of the representation's generalizability.
- **Effective adaptation of JEPA to irregular sequential spatial data:** By operating purely in latent space and avoiding noisy point-level reconstruction, the framework successfully captures complementary local kinematics and global mobility intent. The explicit three-level hierarchy directly addresses the single-scale bias prevalent in contrastive and generative trajectory SSL baselines.
- **Strong reproducibility and transparent experimental protocol:** The authors provide a fully accessible codebase, detailed hyperparameter schedules, dataset statistics, and explicit training/inference procedures in the appendix, meeting and exceeding standard reproducibility expectations for ICLR submissions.

## Weaknesses
- **Mathematically inconsistent and unbenchmarked core fusion mechanism:** The primary technical novelty relies on bilinearly interpolating post-softmax attention coefficient matrices across hierarchical levels (Eq. 13-14) and mixing them via a scalar gate. Interpolating stochastic attention weights inherently breaks row-stochastic constraints, producing mathematically malformed attention distributions. Furthermore, the paper lacks comparison to standard, well-established hierarchical fusion techniques (e.g., cross-attention, FPN-style feature concatenation, or learned gating), making it impossible to determine whether performance gains stem from this specific heuristic or simply from ingesting multi-scale features.
- **Questionable zero-shot transfer protocol validity:** The paper claims strong cross-city zero-shot generalization by loading a model pretrained on Porto into TKY, NYC, and AIS(AU). However, spatial tokenization relies on H3 hexagonal graphs pre-embedded via node2vec on Porto’s specific road topology. Since H3 cell indices are globally defined but graph connectivity and mobility patterns are highly city-specific, reusing or transferring these structural priors across disjoint geographic networks introduces significant distribution shift. Without clarifying how spatial embeddings are adapted or why they remain valid for unseen topologies, the zero-shot claims may reflect tokenizer artifacts rather than true semantic robustness.
- **Self-similarity evaluation protocol conflates temporal continuity with semantic alignment:** The query/database pairs are constructed via odd/even index splits of the same trajectory. While common in the subfield, this setup heavily rewards models that exploit trivial point-to-point temporal proximity rather than learning holistic trajectory shape or semantic similarity, potentially inflating mean-rank improvements and masking true representation quality.

## Nice-to-Haves
- **Level-wise contribution decoupling:** Quantifying the isolated impact of each abstraction layer ($T^{(1)}$ vs $T^{(3)}$) under specific perturbations would clarify whether the hierarchy yields complementary synergistic information or redundant features.
- **Complete efficiency profiling:** Reporting trainable parameter counts, peak VRAM consumption, and inference latency per trajectory is essential for assessing real-world scalability, especially given the overhead of running three parallel JEPA branches with attention upsampling.
- **Algorithmic pseudocode:** A concise forward-pass summary detailing the abstraction pipeline, attention upsampling, re-normalization step, and multi-level loss aggregation would significantly improve readability and verification.

## Novel Insights
The paper effectively reframes trajectory representation as a cross-scale semantic alignment problem rather than a purely metric learning or generative task. By conceptualizing hierarchical attention propagation as a "top-down spotlight," HiT-JEPA demonstrates that explicitly grounding fine-grained spatial transitions in coarse route intent yields embeddings that remain stable under severe temporal sparsity and coordinate distortion. This challenges the prevailing assumption that flat, single-granularity encoders are sufficient for spatial-temporal SSL and highlights the value of latent-space predictive objectives for irregular geometric sequences.

## Potentially Missed Related Work
- Cross-attention / Transformer-based feature pyramid fusion methods in vision (e.g., ViT multi-stage variants, cross-level attention gating) — highly relevant as principled alternatives to attention interpolation.
- Multi-granularity trajectory clustering and representation frameworks in spatial computing — relevant for contextualizing the "first to unify" claim and providing additional baseline comparisons for hierarchical trajectory modeling.

## Suggestions
1. **Normalize attention matrices post-interpolation:** After bilinearly upsampling $A^{(l)}$, apply row-wise softmax re-normalization to restore valid probability distributions before blending with lower-level attention.
2. **Add standard hierarchical fusion baselines:** Compare the proposed attention-blending mechanism against cross-level cross-attention, simple concatenation with layer normalization, and learned gating to isolate the true source of empirical gains.
3. **Clarify the zero-shot spatial tokenization protocol:** Explicitly detail whether H3/node2vec embeddings are shared, re-initialized, or frozen during zero-shot evaluation. If retrained, update the zero-shot claim accordingly; if frozen, analyze cross-city spatial alignment quality or use a geometry-invariant tokenizer.
4. **Introduce independent semantic similarity evaluation:** Augment the odd/even split protocol with a subset of independently sampled trajectory pairs or human-annotated similarity judgments to validate that learned embeddings capture shape/semantic affinity beyond temporal aliasing.

# Actual Human Scores
Individual reviewer scores: [4.0, 6.0, 2.0, 4.0]
Average score: 4.0
Binary outcome: Reject
