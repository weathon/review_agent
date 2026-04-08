=== CALIBRATION EXAMPLE 1 ===

# Harsh Critic Review
## Section-by-Section Critical Review of PARDIFF

---

### Title & Abstract

The title claims "order-agnostic" generation, but the method introduces a structured *partial* order via block-wise ranking (Algorithm 1). The generation is order-agnostic only *within* blocks; the block-level AR progression imposes a global topological ordering. This distinction is subtle but materially important, and the title overstates the contribution.

The abstract makes several sweeping claims: "paradigm shift," "state-of-the-art results on diverse benchmarks," and "real-time applications like drug–drug interaction analysis." None of these are fully justified in the paper. Benchmarks are limited to three molecular datasets (QM9, ZINC-250K, MOSES); the one non-molecular experiment (grid graphs, Figure 1.1) is purely qualitative and anecdotal. The drug–drug interaction claim has no corresponding experiment whatsoever.

The GitHub link in the abstract (`llmresearch678/Pardiff_M_1`) directly violates ICLR's double-blind review requirements. This alone should trigger desk rejection.

---

### Introduction & Motivation

The framing is reasonable: the AR vs. diffusion trade-off is a real and recognized problem. However, several references are confused or mis-attributed. GDSS is cited as "You et al. (2023)" in the introduction but the reference list attributes GDSS to Jo et al. (2022), and "You et al. (2023)" is a different entry entirely. EDP-GNN is attributed as Vignac et al. (2022b), which is actually "Equivariant Discrete Diffusion for Graph Generation," not EDP-GNN. These are not trivial errors; they suggest the literature review was not carefully checked.

The claim "No prior approach fully unifies scalability, permutation-invariance, and structural expressivity in a single, efficient framework" ignores SWINGNN (Yan et al. 2023), which achieves competitive results and is later presented as a baseline — making this claim misleading at best.

---

### Method / Approach

**Section 2 (Diffusion Formulation):** The discrete diffusion setup follows DIGRESS closely, which the paper acknowledges. The VLB objective (Equation 1) and CE loss (Equation 2) are standard; there is no novel technical contribution in the diffusion formulation itself.

**Critical internal inconsistency:** Section 2.4 states "We fix the maximum number of diffusion steps to T = 40," while Section 3 states "using a fixed schedule length of T = 50." This contradiction is not a minor typo — it undermines confidence in the reproducibility of the reported results.

**Section 2.1 (Block Ranking, Algorithm 1):** The weighted degree hash $w_K(v) = \sum_{k=1}^K \delta_k(v) \cdot |V|^{K-k}$ is proposed to reduce rank collisions. However:
- The paper acknowledges collision can still occur ("rank collisions"), but provides no quantitative analysis of collision rates on real datasets (e.g., molecular graphs have many structurally equivalent atoms).
- In highly symmetric molecules (e.g., benzene, ferrocene), every atom is in the same orbit — the ranking degenerates and all nodes receive identical ranks, collapsing the block-wise structure.
- The computational cost of computing K-hop neighborhoods for all nodes is not analyzed, particularly for large graphs in MOSES (~1.9M molecules, some with many atoms).

**Theorem 1 (Permutation Consistency):** The theorem is stated correctly for a collision-free ranking. But since collisions are acknowledged as possible, the permutation-consistency guarantee is conditional in a way not made explicit. The "proof sketch" in the main paper simply restates that the K-hop neighborhood is invariant under relabeling — this is not a rigorous proof. The proof is deferred to an appendix that is not included in this submission, making the claim unverifiable.

**Theorem 2 (Symmetry Bottleneck):** This is a well-known result in the graph representation learning literature (see Maron et al. 2019; Keriven & Peyré 2019; Garg et al. 2020). Presenting it as a theorem contribution of this paper is misleading. It is essentially a literature fact restated in slightly different notation.

**Section 2.3 (Symmetry-Breaking via Diffusion):** This section creates a significant logical tension with Theorem 3. Section 2.3 argues that structured noise injection deliberately breaks symmetry to escape automorphic orbits (akin to simulated annealing). But Theorem 3 claims the full generative model is *invariant* under node permutations. These two claims are in direct conflict: if the mechanism genuinely breaks automorphic symmetry (i.e., distinguishes nodes in the same orbit), the resulting distribution *cannot* be permutation-invariant in the sense required by Theorem 3. The paper does not resolve this tension. The proof of Theorem 3 is again deferred to a missing appendix.

**Section 2.4 (Hybrid Transformer Architecture):** This is the section with the greatest opacity. The paper says it "merges the transformer-based global reasoning of GRIT with a lightweight approximation of higher-order interactions inspired by PPGN," but provides no architectural diagram, no equation for the combined layer, no specification of depth/width, no attention mechanism details, and no ablation comparing this hybrid to GRIT or PPGN alone. The masked attention (MA) and masked bilinear (MB) operations are defined but the derivation of MB's correctness is deferred to the missing appendix. This is insufficient for reproducibility.

**Algorithm 2 (Block Size Predictor) — Exposure Bias:** The block size predictor $g_\alpha$ is trained to predict the next block size given the current ground-truth block $C_i$, using cross-entropy against the true next block $C_{i+1}$. At inference (Algorithm 4), predictions are made autoregressively conditioned on previously *generated* (possibly imperfect) blocks. This is the classic training/inference distribution mismatch (exposure bias) well-known in sequence generation. The paper does not mention this issue or how it is addressed.

---

### Experiments & Results

**Table 1 (QM9) — Text vs. Table Inconsistency:** Page 8 reports: "achieving state-of-the-art scores on VAL (98.1%), AL (98.9%), and molecular accuracy or MOL (88.5%)." Table 1, however, shows PARDIFF's VAL = **98.9%**, AL = **99.2%**, MOL = **90.3%**. Every single number in the narrative differs from the table. This is a major red flag suggesting the text was not updated to match final results, or numbers were changed without cross-checking.

**Table 3 (MOSES) — Missing Data:** The VAL and UNI columns for PARDIFF in Table 3 are entirely blank. The text claims "perfect VAL and UNI" but these values do not appear in the table. This is not a PDF parsing artifact — the empty cells are structurally present in the Markdown table. It is unclear whether this is an oversight or a deliberate omission.

**Surpassing "Dataset Optimum" on QM9:** PARDIFF claims 98.9% VAL and 100% UNI, exceeding the reported "DATASET (OPTIMAL)" row (97.8%, 100.0%). Surpassing the distribution of the training data is physically difficult to interpret. Typically, if the dataset contains some invalid molecules, a model that generates *only* valid ones is trivially rewarded by this metric. This needs careful discussion: is PARDIFF generating the *right* distribution, or just filtering toward validity?

**Missing Quantitative Non-Molecular Experiments:** The introduction motivates the work through social networks, sensor meshes, and cyber-physical graphs. Grid graph generation (Figure 1.1) is shown only qualitatively. There are no quantitative benchmarks on generic graph generation tasks (e.g., Erdős-Rényi, planar, stochastic block model, community graphs) that have been standard in the graph generation literature (GRAPHRNN, GRAN, DiGress all report these). This significantly limits the claim of "diverse domains."

**Baseline Coverage:** GDSS appears only in Table 2; GRAPHARM appears in Table 2 but not Tables 1 or 3. Why are different baselines omitted across tables? More importantly, SWINGNN-L is compared only on ZINC (Table 2); it was also evaluated on MOSES in the SWINGNN paper. The missing comparison on MOSES is suspicious given that SWINGNN achieves an FCD of ~0.5 on MOSES, potentially competitive with PARDIFF's 0.39.

**No Statistical Significance or Error Bars:** All results are reported as point estimates with no variance. For improvements as small as FCD 1.62 vs. 1.99, statistical significance cannot be assessed.

**No Meaningful Ablation in Main Paper:** The paper says "ablation results are provided in the APPENDIX," but the appendix is absent from this submission. No ablations appear in the main text: no comparison of the AR+diffusion design against pure diffusion alone, no effect of block number K, no effect of T (number of diffusion steps — which has an internal inconsistency anyway), no comparison of the hybrid architecture against GRIT or PPGN alone.

**Hardware Claim:** The experiments were run on "NVIDIA RTX 5080," a GPU that is not commercially available as of standard timelines, raising questions about the experimental setup's authenticity.

---

### Writing & Clarity

The method description in Section 2.4 is too vague to support reproducibility — no architectural diagram, no layer specification, no code. Given that the GitHub repository uses a generic username (llmresearch678), the code's correspondence to the paper is not verifiable in a blind review. Sections 2.2–2.3 introduce the symmetry bottleneck and annealing analogy, which are conceptually interesting but mathematically underdeveloped.

The conclusion reads more like a press release ("Game Changer," "paradigm shift") than a scientific assessment. There is no discussion whatsoever of limitations, failure modes, or boundary conditions.

---

### Limitations & Broader Impact

The paper has **no limitations section**. Notable absent discussions include:
- Failure on highly symmetric graphs (automorphism collapse in the ranking)
- Computational cost at inference (block-by-block diffusion with T=40–50 steps each is expensive)
- The exposure bias issue in block size prediction
- Scope of valid application (molecular vs. arbitrary graphs)
- Negative applications (e.g., generation of harmful chemical structures)

---

## Overall Assessment

PARDIFF presents an interesting conceptual combination of AR block-wise generation and discrete diffusion for graphs. The core idea — using a structural ranking to define partial orderings, then applying shared block-level diffusion — is reasonable and potentially valuable. However, the submission has serious problems that collectively place it well below the ICLR acceptance bar. The most critical issues are: (1) a double-blind violation via the GitHub link; (2) a missing appendix that is referenced for all proofs, derivations, and ablations; (3) a direct self-contradiction between text and Table 1 results; (4) blank cells in Table 3 with unsubstantiated claims; (5) an internal inconsistency on a key hyperparameter (T=40 vs. T=50); (6) a logical conflict between the symmetry-breaking mechanism (§2.3) and the permutation-invariance theorem (Theorem 3); and (7) no quantitative evaluation on non-molecular benchmarks despite broad domain claims. Even setting aside these integrity concerns, the technical contribution is incremental over GRAN/DiGress, the architecture is under-specified, and the experimental design lacks ablations and statistical rigor. In its current form, this paper should not be accepted.

# Neutral Reviewer
## Balanced Review

### Summary
The paper introduces PARDIFF, a graph generation framework that reconciles autoregressive controllability with diffusion-based permutation invariance by generating graphs through a learned, block-wise structural decomposition. It combines a discrete diffusion denoiser conditioned on dynamically predicted block sizes with a causally masked, block-parallel training scheme and a symmetry-breaking noise injection strategy. Empirical evaluations on QM9, ZINC-250K, and MOSES show competitive or state-of-the-art scores in chemical validity, uniqueness, novelty, and distributional fidelity compared to leading baselines like DIGRESS and GDSS.

### Strengths
1. **Addresses a Core Graph Generation Dilemma:** The block-wise decomposition circumvents the factorial ordering problem of AR models while preserving directional generation, effectively mitigating permutation bias without resorting to canonical ordering heuristics (Sec. 2.1, Alg. 1).
2. **Clear Theoretical Motivation for Symmetry Breaking:** The paper correctly identifies the automorphism orbit bottleneck for equivariant nets (Theorem 2) and proposes a structured noise-injection mechanism to escape representational degeneracy. This aligns well with energy-landscape intuition and is grounded in permutation-consistency proofs (Theorem 1 & 3).
3. **Strong Empirical Performance:** Results across three standard molecular benchmarks consistently outperform or closely track recent diffusion and hybrid methods in validity, uniqueness, and chemical similarity metrics (Tables 1–3), with particularly notable FCD and scaffold similarity improvements on MOSES.
4. **Practical Efficiency via Causal Block Parallelization:** The masked attention/bilinear update scheme (Sec. 2.4) allows simultaneous conditioning over multiple blocks, claiming >10× training speedups. This engineering contribution directly tackles the sequential latency bottleneck of traditional AR graph models.
5. **Code Availability:** Providing a public repository aligns with open-science norms and facilitates independent verification of the pipeline.

### Weaknesses
1. **Overstated Theoretical Novelty:** Theorem 2 is a direct restatement of the well-known representational limit of permutation-equivariant maps under graph automorphisms (closely tied to the 1-WL expressivity bound in GNN literature). Presenting it as a central theoretical contribution without citing foundational works on equivariant expressivity boundaries misrepresents its originality.
2. **Insufficient Architectural & Ablation Details in Main Text:** Core design choices (hybrid transformer construction, block-size predictor architecture, hyperparameter sensitivities, layer counts, hidden dimensions) are deferred to the appendix. The main text lacks a concise architecture diagram, making it difficult for reviewers to assess the true contribution vs. engineering tuning.
3. **Missing Statistical Rigor & Inference Benchmarks:** ICLR expects mean ± std across multiple random seeds for generative metrics. Tables report single-point numbers with perfect/uniformly rounded values (e.g., UNI 100.0, UNI 99.998), raising concerns about overfitting to evaluation scripts or single-run variance. Additionally, while training speed is claimed, inference sampling time per molecule and memory footprint during generation are unreported.
4. **Inconsistencies & Unclear Metrics:** The diffusion steps are stated as `T=40` in Sec. 2.4 but `T=50` in Sec. 3. The `MOL` score in Table 1 reportedly exceeds "reference dataset accuracy (87.0%)", which is mathematically ambiguous for a molecular accuracy metric and requires precise definition. The claim of directly learning `p_φ(G_0|G_t)` is standard `x_0`-prediction in modern DDPM literature, not a novel objective derivation.
5. **Hyperbolic Language & Reporting Artifacts:** Phrases like "paradigm shift" and "game changer" (Sec. 4) conflict with ICLR's expectation for measured, scientific tone. Additionally, listing an unreleased GPU (RTX 5080) suggests insufficient proofreading and raises questions about experimental provenance.

### Novelty & Significance
**Novelty:** Moderate to High. The integration of dynamic structural ranking, conditional discrete diffusion over predicted blocks, and causal parallel masking forms a coherent pipeline that meaningfully advances the AR-diffusion hybrid space. However, individual components (block-wise generation, discrete graph diffusion, masking for parallelism) exist independently; the primary novelty lies in their systematic unification and the practical parallel training scheme.
**Clarity:** Fair to Good. The motivation and high-level pipeline are intuitive, but mathematical notation is occasionally imprecise, and key architectural specifications are pushed to supplements. Inconsistent hyperparameters and ambiguous metric definitions detract from readability.
**Reproducibility:** Conditional Positive. Code is provided, and training splits/baselines are standard. To meet full ICLR reproducibility standards, the authors must clarify the exact diffusion step count, provide a reproducible configuration file (or table), and report multi-seed statistical variance.
**Significance:** High if validated. Scalable, permutation-invariant graph generation is a bottleneck for molecular design and network synthesis. If PARDIFF's claimed efficiency and chemical fidelity hold under rigorous evaluation, it offers a practical, trainable alternative to monolithic diffusion or biased AR generators.

### Suggestions for Improvement
1. **Strengthen Experimental Rigor:** Report all metrics as mean ± standard deviation over at least three random seeds. Include explicit generation throughput metrics (e.g., seconds per 1k molecules, peak VRAM during sampling) and compare them against baselines to substantiate efficiency claims.
2. **Frontload Architecture & Ablations:** Add a clear architecture schematic and a dedicated hyperparameter table in the main text. Summarize key ablation results in the main manuscript (e.g., impact of block-size prediction accuracy, effect of removing causal masking, contribution of symmetry-breaking noise vs. uniform noise).
3. **Contextualize Theory & Standardize Notation:** Explicitly cite foundational literature on equivariant map limitations and WL expressivity when discussing Theorem 2. Resolve the `T=40` vs `T=50` discrepancy and mathematically define the `MOL` metric and how baseline "dataset accuracy" thresholds are computed.
4. **Refine Claims & Academic Tone:** Remove promotional language ("game changer," "paradigm shift"). Frame contributions realistically as a structured hybrid approach with practical parallelization benefits. Ensure all claims are proportionate to empirical evidence and align with standard diffusion parameterization terminology (e.g., clarify that `x_0`-prediction is used rather than claiming independent step estimation as novel).

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1.  **Correct Validity Metrics:** Table 1 shows PARDIFF validity (98.9%) exceeding the Dataset Optimal (97.8%), which is impossible if the dataset represents the ground truth distribution; recalculate these metrics or explain how generated data surpasses ground truth validity.
2.  **Topological Structure Metrics:** Add graph topology metrics (degree distribution KL, clustering coefficients, orbit counts) to substantiate claims of "structural fidelity" beyond chemical validity (VAL/FCD).
3.  **Wall-Clock Inference Time:** Provide actual inference latency benchmarks against DIGRESS and GDSS, as parameter count (Table 2) does not reflect the computational cost of PPGN's $O(n^3)$ operations during diffusion.
4.  **Ranking Module Ablation:** Include an ablation study comparing the proposed structural ranking against random ordering or BFS/DFS to prove the learned decomposition drives performance gains.

### Deeper Analysis Needed (top 3-5 only)
1.  **Tie-Breaking Mechanism:** Explicitly analyze how Algorithm 1 handles nodes with identical structural scores ($w_K(u) = w_K(v)$) without using node indices, as index-based tie-breaking violates the claimed permutation invariance in Theorem 1.
2.  **Parallelism vs. Causal Masking:** Resolve the contradiction between Section 2.1.1 claiming "partially parallel generation within blocks" and Section 2.4 applying causal masks ($M_{ij}=1$ if $i \geq j$) which enforce sequential dependency.
3.  **Symmetry Breaking vs. Invariance:** Clarify how injecting noise to break symmetry (Section 2.3) does not conflict with the strict permutation invariance guarantee claimed in Theorem 3 during the deterministic ranking phase.

### Visualizations & Case Studies
1.  **Symmetric Graph Stress Test:** Display generation results on highly symmetric graphs (e.g., fullerenes or regular lattices) to visually verify if the model collapses or fails due to the equivariance bottleneck described in Theorem 2.
2.  **Block Growth Sequence:** Visualize the intermediate steps of graph construction (Block 1 $\to$ Block 2 $\to$ Final) to verify if the learned decomposition aligns with chemical intuition (e.g., scaffold first, then substituents).
3.  **Orbit Embedding Separation:** Plot t-SNE projections of node embeddings within the same automorphism orbit to demonstrate whether the "symmetry-breaking" noise successfully distinguishes structurally identical nodes.

### Obvious Next Steps
1.  **Fix Metric Anomalies:** Immediately audit and correct Table 1 where generated validity exceeds the dataset's own validity, as this undermines trust in all reported SOTA claims.
2.  **Formalize Theoretical Claims:** Revise Theorem 1 and Algorithm 1 to formally address tie-breaking logic, ensuring the proof of permutation consistency holds under all structural conditions.
3.  **Integrate Ablation Studies:** Move the referenced appendix ablations into the main text to validate the contribution of the block-wise design versus standard monolithic diffusion.
4.  **Clarify Computational Complexity:** Reconcile the use of PPGN (high memory complexity) with the claimed efficiency gains, providing a clear analysis of memory usage during training on large graphs like MOSES.

# Final Consolidated Review
## Summary
PARDIFF proposes a hybrid autoregressive-diffusion framework for graph generation that decomposes graphs into blocks via a learned structural ranking, then applies equivariant discrete diffusion within each block. The approach aims to combine the controllability of autoregressive models with the permutation invariance of diffusion models, using masked parallel training for efficiency. Experiments focus on molecular benchmarks (QM9, ZINC-250K, MOSES).

## Strengths
- **Core idea addresses a genuine tension**: The block-wise decomposition approach provides a principled way to reconcile autoregressive directional generation with permutation invariance, avoiding factorial ordering complexity while maintaining coherent structural growth (Sec. 2.1, Algorithm 1).
- **Strong empirical results on molecular benchmarks**: PARDIFF achieves competitive or state-of-the-art validity, uniqueness, and FCD scores across QM9, ZINC-250K, and MOSES (Tables 1-3), with particularly notable FCD improvements (e.g., 1.62 on ZINC-250K vs. 14.66 for GDSS).
- **Practical efficiency via masked parallelization**: The causal masking scheme (Sec. 2.4) enables single-pass computation for all conditional block probabilities, claiming >10× training speedup—a meaningful engineering contribution addressing AR model latency.

## Weaknesses
- **Double-blind violation**: The abstract contains a GitHub repository link (`github.com/llmresearch678/Pardiff_M_1`), which directly violates ICLR's anonymous submission policy. This is grounds for rejection regardless of paper quality.

- **Missing appendix undermines reproducibility**: Proofs for Theorems 1 and 3, derivations of masked bilinear operations, and all ablation studies are deferred to an appendix not included in the submission. Critical architectural details (depth, width, attention specifics) are absent from the main text, making the contribution impossible to fully evaluate.

- **Internal inconsistencies in hyperparameters and results**: Section 2.4 states diffusion steps T=40 while Section 3 states T=50. Table 1 text reports VAL=98.1%, AL=98.9%, MOL=88.5% while the table shows VAL=98.9%, AL=99.2%, MOL=90.3%. Table 3 has blank cells for PARDIFF's VAL and UNI columns despite text claiming "perfect" scores. These inconsistencies erode trust in the reported results.

- **Claimed GPU does not exist**: Experiments report using "NVIDIA RTX 5080"—a GPU model that does not exist commercially. This raises serious questions about experimental provenance.

- **Theorem 2 is not novel**: The symmetry bottleneck for equivariant networks (nodes in the same automorphism orbit receive identical representations) is a well-established result in the GNN literature (Maron et al. 2019; Keriven & Peyré 2019; Garg et al. 2020). Presenting it as a theorem contribution without citing foundational work is misleading.

- **Logical tension between symmetry-breaking and permutation invariance**: Section 2.3 explicitly introduces noise injection to "break symmetry" and escape automorphism orbits, while Theorem 3 claims the full generative model is permutation-invariant. If the mechanism genuinely breaks automorphic symmetry during generation, the resulting distribution cannot simultaneously satisfy permutation invariance in the sense Theorem 3 requires. This contradiction is not resolved.

- **Tie-breaking in Algorithm 1 is unspecified**: Algorithm 1 selects nodes with minimum structural weight $w_K(v)$, but does not specify how ties are broken when multiple nodes share the same weight. Index-based tie-breaking would violate Theorem 1's permutation-consistency claim.

- **No exposure bias mitigation**: The block size predictor (Algorithm 2) trains on ground-truth block sizes but predicts autoregressively from generated blocks at inference (Algorithm 4). This training/inference distribution mismatch is unaddressed.

- **No quantitative non-molecular experiments**: The introduction claims contribution for "social networks, biochemical systems, recommendation engines, and cyber-physical infrastructures," but experiments are limited to molecular datasets. Grid graphs (Figure 1.1) are purely qualitative.

- **No statistical significance or multi-seed evaluation**: All results are single-point estimates with no variance reported. Small improvements (e.g., FCD 1.62 vs. 1.99) cannot be assessed for statistical significance.

## Nice-to-Haves
- **Ablation studies in main text**: Compare against pure diffusion without block decomposition, vary the number of diffusion steps, and ablate the hybrid architecture against GRIT or PPGN alone.
- **Inference latency benchmarks**: Report wall-clock sampling time per molecule, as parameter count alone does not capture PPGN's $O(n^3)$ memory complexity during diffusion.
- **Graph topology metrics**: Degree distribution KL divergence, clustering coefficients, and orbit counts would strengthen structural fidelity claims beyond chemical metrics.

## Removed Points
*These points are flagged to be removed; treat them with caution.*

- **Baseline coverage concerns**: The harsh critic questioned why different baselines appear in different tables. However, this is common practice when some methods are only evaluated on specific benchmarks in their original papers. MOSES comparisons include DIGRESS and CONGRESS—relevant baselines for that benchmark.

- **Claim that surpassing dataset validity is "impossible"**: The spark finder claims Table 1's PARDIFF validity (98.9%) exceeding dataset optimal (97.8%) is "impossible." This is incorrect reasoning: a generative model can produce higher validity than the training distribution if it learns to filter toward chemically valid structures. The metric requires explanation but is not mathematically impossible.

- **Generic promotional language complaints**: Criticisms about "paradigm shift" and "game changer" language are stylistic preferences. While not ideal, they do not constitute substantive weakness.

## Novel Insights
The tension between symmetry-breaking for expressivity and permutation invariance for robustness reveals a fundamental design choice that this paper does not cleanly resolve. A cleaner approach would be to argue that permutation invariance holds *over the ranking function* (which it does), while acknowledging that intra-block symmetry-breaking is intentional to improve expressivity—rather than claiming both properties simultaneously. Additionally, the block-wise decomposition conceptually mirrors curriculum learning: generating easier structures first (high-connectivity cores) before harder peripheral structures. This analogy could strengthen future work.

## Suggestions
1. **Remove the GitHub link immediately** and provide code via anonymous repository for review.
2. **Correct all inconsistencies**: reconcile T=40 vs. T=50, align Table 1 numbers with text, and fill missing values in Table 3.
3. **Clarify the real GPU used**—RTX 5080 does not exist and must be corrected.
4. **Move proofs and ablations into the main submission**: currently unverifiable claims include Theorems 1 and 3, masked bilinear derivation, and all ablation experiments.
5. **Resolve the symmetry/invariance contradiction**: either explain how both can hold or reframe the claims appropriately.
6. **Specify tie-breaking in Algorithm 1** and prove permutation consistency holds under that scheme.

# Actual Human Scores
Individual reviewer scores: [0.0, 0.0, 0.0, 2.0]
Average score: 0.5
Binary outcome: Reject
