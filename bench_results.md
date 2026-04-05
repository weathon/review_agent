# ICLR Benchmark Results

Date: 2026-04-04 22:03
Critic/Merger: qwen/qwen3.6-plus:free (OpenRouter)
Neutral: qwen/qwen3.6-plus:free, Related Work: qwen/qwen3.6-plus:free:online (OpenRouter)

## bm3rbtEMFj

- GT: Accept (Poster) (avg 5.0)
- Predicted: N/A (5.8/10)
- Match: N/A

### Final Review

## Summary
This paper introduces ELMUR, a transformer-based policy architecture that augments each layer with a structured external memory track, employing bidirectional cross-attention and a non-learnable LRU update mechanism for long-horizon partially observable tasks. Evaluated exclusively via offline imitation learning, the method achieves perfect success on T-Maze corridors up to one million steps and demonstrates consistent gains over strong sequence modeling and offline RL baselines on POPGym and MIKASA-Robo. The work is supported by formal retention bounds and thorough mechanistic probing that confirms explicit, stable slot utilization rather than parametric memorization.

## Strengths
- **Strong empirical validation on memory-intensive benchmarks:** ELMUR achieves 100% success on T-Maze up to 1,000,000 steps and consistently outperforms Decision Transformer, RATE, CQL, and Diffusion Policy across diverse POPGym puzzles and sparse-reward robotic manipulation tasks.
- **Rigorous interpretability and memory probing:** Detailed post-hoc probing, PCA trajectory analysis, and cross-attention heatmaps conclusively demonstrate targeted one-shot writes, stable retention across long delays, and head specialization, ruling out performance artifacts from scale or overparameterization.
- **Clear theoretical grounding:** Propositions 1 and 2 derive closed-form expressions for exponential forgetting, half-life, and norm boundedness under convex blending. This provides transparent, actionable intuition linking hyperparameters ($\lambda$, $M$) to retention dynamics, elevating the work beyond purely empirical engineering.

## Weaknesses
- **Significant mismatch between RL motivation and IL-only evaluation:** The introduction heavily frames the problem around online RL challenges (exploration, sparse rewards, sample inefficiency, sim-to-real gap), yet all experiments are conducted under offline Behavior Cloning with expert trajectories. This entirely sidesteps how detached segment-level memory, convex blending, and LRU updates interact with policy gradients, credit assignment, or distribution shift from suboptimal data. The core claim of solving "long-horizon RL problems" is therefore unsubstantiated for the intended regime.
- **Linear memory scaling bottlenecks true scalability:** While the architecture decouples inference complexity from sequence length, the empirical results (Fig. 6c) show a sharp performance cliff when memory slots $M$ fall below the required number of segments $N$. The "100,000× horizon extension" claim mathematically holds, but practically shifts the bottleneck from quadratic attention to linear explicit memory capacity. Without adaptive slot allocation or content-aware addressing, the method does not fundamentally improve memory efficiency for open-ended or highly complex trajectories.
- **Rigid write-throughput and recency-only eviction:** The LRU mechanism restricts updates to exactly one slot per segment using a fixed recency heuristic. This creates a strict write-throughput ceiling for multi-factor POMDPs and risks catastrophic loss of rare but critical events that do not align with recency metrics. The non-learnable blending parameter $\lambda$ further limits adaptability, with ablations showing notable instability at intermediate values ($\lambda \approx 0.4$–$0.6$), raising concerns for multi-task or dynamic-horizon deployment.
- **Lack of compute-matched and modern backbone comparisons:** DMamba is listed as a baseline but omitted from main comparison tables. More critically, there is no FLOP or VRAM-matched baseline comparing ELMUR against a standard transformer with an extended context window, nor against modern long-sequence backbones (e.g., xLSTM, Mamba, RWKV). Reported step latency (6.8 ms) is hardware-dependent and ignores MoE routing overhead, total parameter count, and memory slot footprint, leaving the "efficient and scalable" assertion partially unverified.

## Nice-to-Haves
- Extend the memory capacity sensitivity analysis (Fig. 6c–d) across all three benchmarks to confirm the $M \ge N$ requirement is a universal bottleneck rather than a task-specific artifact.
- Provide a brief discussion on safety and failure modes for deployed agents, specifically how persistent LRU memory could compound distribution shift or retain stale/outdated commands over multi-million step horizons.
- Clarify empirically how layer normalization and gradient clipping maintain the theoretical $\|m^{new}\| \le C$ assumption during long-horizon training, particularly when MoE experts exhibit high activation variance.

## Novel Insights
The paper effectively demonstrates that for long-horizon POMDPs, explicit temporal separation of read/write operations combined with simple recency-based eviction can outperform both implicit sequence modeling and differentiable memory addressing. The mechanistic analyses reveal a highly structured, interpretable dynamic where the network performs near-instant, targeted writes to designated slots and maintains stable, linearly decodable representations without continuous overwriting. This suggests that in sparse-reward or partially observable regimes, stability and bounded retention may be more critical than complex, content-adaptive memory routing, offering a compelling design principle for efficient sequential decision-making architectures.

## Potentially Missed Related Work
- **TiTANS (Behrouz et al., 2024)** and **xLSTM (Beck et al., 2024)** — relevant for test-time adaptation and long-horizon implicit memory scaling without external memory tracks.
- **Compressive Transformer (Rae et al., 2019)** and **Memorizing Transformer (Wu et al., 2022)** — important baselines for explicit context caching and retrieval-augmented sequence modeling that share similar retention goals.
- Present as architectural alternatives; omission does not invalidate the contribution but contextualizes it within broader long-context literature.

## Suggestions
1. Align experiments with the stated RL motivation: evaluate ELMUR under at least one online RL algorithm (e.g., PPO/SAC) or with suboptimal offline datasets, and discuss how segment-level recurrence and memory detachment affect policy gradient stability or sample efficiency.
2. Introduce compute-normalized comparisons: report peak VRAM, total FLOPs per episode, and include a matched-context standard transformer and at least one modern state-space backbone. This will rigorously test whether gains stem from architectural design or resource allocation.
3. Explore lightweight adaptive mechanisms: replace the fixed single-slot LRU update with a content-aware routing or top-$k$ write scheme, and propose a dynamic $\lambda$ adjustment based on slot attention entropy or prediction uncertainty. This would mitigate the $M < N$ cliff and improve multi-horizon robustness.

---

## M14YpuTejd

- GT: Withdrawn (treated as Reject) (avg 4.0)
- Predicted: N/A (4.8/10)
- Match: N/A

### Final Review

## Summary
This paper identifies critical methodological flaws in the emerging two-stage online map-based motion prediction protocol, specifically highlighting the train-validation distribution gap caused by overlapping dataset splits and the mismatch between limited map perception ranges and long-range motion forecasting. To address these, the authors introduce OMMP-Bench, featuring a spatially disjoint data partition, refined metrics targeting non-ego moving agents stratified by distance, and a boundary-free baseline that retrieves raw image features via deformable attention to compensate for out-of-range agents.

## Strengths
- **Rigorous Protocol Diagnosis & Correction:** The identification of the implicit train-val gap in two-stage training (where the online mapper infers on its own training data, masking severe distribution shifts during evaluation) is a crucial methodological correction. Table 1 empirically demonstrates the performance collapse under the flawed protocol and the recovery achieved by the proposed disjoint split.
- **Metric Realignment Towards Safety-Critical Targets:** Shifting evaluation from the ego-vehicle to all moving non-ego agents directly aligns the benchmark with the collision-avoidance objectives of downstream planning. Stratifying results by distance relative to the map perception boundary provides highly actionable diagnostics for failure modes that ego-centric metrics completely obscure.
- **Pragmatic & Effective Baseline Design:** The proposed deformable attention mechanism to extract environmental cues from raw multi-view image features elegantly circumvents the hard perceptual boundaries of vectorized map predictors. Consistent performance gains across multiple backbone configurations (Table 7) validate its utility, particularly for distant agent forecasting where map context is absent.

## Weaknesses
- **Reproducibility of the Data Partition:** The reliance on a "manual check" to ensure spatial disjointness between the map train and motion sets is scientifically insufficient for a proposed benchmark. Without a programmatic algorithm (e.g., coordinate distance thresholds, spatial hashing criteria, or explicit scene clustering logic), the partition cannot be independently verified, replicated, or adapted to other datasets, severely limiting OMMP-Bench's utility as a community standard.
- **Incomplete Specification of the Image Baseline:** The core methodological contribution lacks critical implementation details. Equation 1 leaves geometric projection variables undefined, and the paper omits how 3D agent trajectories are mapped across multiple camera views, how agents falling outside camera FOV or under occlusion are handled, and precisely where the aggregated features inject into existing motion predictors. More critically, motion prediction relies on temporal history; the paper provides no explanation of how image features are cached or aggregated across time steps, introducing a major architectural ambiguity.
- **Lack of Statistical Rigor and Efficiency Analysis:** All results (Tables 1, 4, 6, 7) are reported as single-run point estimates. Given the well-documented training variance in trajectory prediction, the absence of standard deviations, confidence intervals, or multi-seed averages fundamentally weakens the statistical validity of claimed improvements. Furthermore, deploying per-agent dense image attention incurs substantial computational overhead; without reporting inference latency, FLOPs, or memory footprint, the practical viability of the baseline for real-time autonomous deployment remains unproven.
- **Oracle Assumptions and Narrow Empirical Scope:** The evaluation is restricted to nuScenes and a limited architecture matrix (MapTR variants × HiVT/DenseTNT). More critically, the protocol assumes online mappers will provide comprehensive element recall, testing under an "oracle-like" condition rather than simulating realistic topological failures (e.g., hallucinated centerlines, missing boundaries). This bypasses the compound error analysis necessary to understand how the benchmark translates to full-pipeline deployment.

## Nice-to-Haves
- Conduct controlled map corruption experiments (e.g., systematic lane dropout, boundary offset, topology breaks) on ground-truth inputs to explicitly quantify how specific topological errors degrade motion forecasting and validate the benchmark's diagnostic value.
- Introduce a capacity-matched ablation (e.g., expanded MLPs or learnable positional embeddings) to verify that baseline gains stem from the semantic content of raw image features rather than simply increased parameter count.
- Plot minADE and Miss Rate as continuous functions of agent-ego distance to visually characterize the error decay curve and demonstrate whether the image-feature baseline genuinely recovers performance at the map boundary or merely flattens the distribution.
- Visualize the deformable attention maps for far-range agents to verify the model attends to drivable regions and semantic lane markings rather than high-frequency background textures.

## Novel Insights
The work compellingly demonstrates that current evaluation pipelines for online map-based motion prediction artificially inflate performance through implicit data leakage and ego-centric myopia. By exposing the hard boundary limitation of vectorized map decoders, the authors reveal that long-range forecasting fundamentally requires environmental context that discrete geometric representations fail to provide at scale. The proposed boundary-free approach suggests a pragmatic paradigm shift: rather than aggressively extending the brittle perception range of online mappers, downstream predictors can more robustly compensate by anchoring directly to continuous, range-unconstrained sensor features. This effectively decouples environmental awareness from the discrete hallucination and dropout inherent in vectorized mapping, pointing toward feature-hybrid architectures as a more reliable path for co-developing mapping and prediction modules.

## Potentially Missed Related Work
- **Dense occupancy grids and world models (e.g., BEVWorld, Occ3D)** — These representations naturally bypass vectorized boundary constraints and provide explicit 3D semantic context for out-of-range agents. Comparing the proposed image-feature baseline against dense occupancy inputs would contextualize its advantage and limitations.
- **Implicit map reasoning in end-to-end predictors** — While the paper deliberately isolates mapping from prediction, contrasting OMMP-Bench's findings with how integrated architectures (e.g., UniAD, VAD) implicitly handle missing map topology could strengthen the discussion on when decoupled vs. joint training is appropriate.

## Suggestions
- Replace the "manual check" description with a fully reproducible partitioning script. Release exact scene indices, spatial bounding coordinates, and a clear overlap-distance threshold so researchers can regenerate the split deterministically.
- Fully specify the geometric transformation pipeline in Equation 1, detail the temporal feature aggregation strategy (e.g., sliding window, recurrent caching, or frame-averaging), and explicitly document the injection layer and tensor dimensions used to fuse image features into HiVT/DenseTNT.
- Rerun all key experiments across ≥3 random seeds, reporting mean ± standard deviation with statistical significance testing for claimed improvements. Add a comprehensive table quantifying the latency, FLOPs, and memory overhead of the proposed baseline relative to the base models to enable fair systems-level evaluation.

---

## ahpO7S1Ppi

- GT: Reject (avg 3.5)
- Predicted: N/A (5.3/10)
- Match: N/A

### Final Review

## Summary
This paper introduces PCTX, a personalized context-aware tokenizer for generative recommendation that replaces static semantic IDs with user-history-conditioned identifiers. It addresses the inherent similarity bias in autoregressive decoding by clustering context-aware item representations, merging infrequent IDs, and training a GR model with data augmentation and multi-facet inference. The method demonstrates consistent, statistically significant improvements over strong static and local-context baselines across three Amazon benchmark datasets.

## Strengths
- **Clear architectural motivation:** Correctly identifies that static semantic IDs impose a universal similarity metric via shared prefix probabilities in autoregressive decoders. Injecting personalization upstream during tokenization, rather than relying solely on the decoder to learn user-specific intent from fixed IDs, is a conceptually sound shift.
- **Principled sparsity-personalization balance:** Introduces a cohesive, empirically validated pipeline (adaptive clustering, duplicate/frequency-based ID merging, and stochastic augmentation) to prevent over-fragmentation of the training signal. Ablation studies rigorously confirm that each component is necessary for maintaining performance.
- **Rigorous empirical protocol:** Outperforms state-of-the-art GR baselines (TIGER, LETTER, ActionPiece) across all metrics with statistical significance, using matched parameter counts, compute budgets, and evaluation settings. Code and detailed implementation notes are fully provided.

## Weaknesses
- **Strictly two-stage, non-end-to-end architecture:** The tokenizer is entirely offline. Context representations are extracted from a frozen auxiliary model, clustered, fused with text features, and quantized via RQ-VAE *before* training the GR backbone. This decoupling prevents co-adaptation between the tokenization and generative objectives, capping potential performance and introducing rigidity to distributional shifts. While acceptable as a preprocessing pipeline, it leaves the information gap unquantified and ignores whether joint optimization could yield superior or more robust representations.
- **Over-engineered centroid allocation:** The adaptive grouping strategy (Appendix B) relies on four manually tuned hyperparameters ($T, K, C_{start}, \delta$) with Gamma-distribution priors and arithmetic progressions. While stable across three similarly scaled datasets, this heuristic lacks theoretical grounding and introduces substantial tuning overhead for scaling to heterogeneous, domain-shifted catalogs.
- **Failure to isolate personalization from label-smoothing effects:** The paper compares against static 1-ID baselines and includes a "random ID target" ablation, but omits a critical control: a *context-agnostic* multi-ID baseline with identical data augmentation. Without this, it remains ambiguous how much of the gain stems from genuine intent-aware personalization versus increased target vocabulary diversity acting as implicit regularization or label smoothing.
- **Narrow evaluation scope and unreported overhead:** Experiments cap interaction histories at length 20 (avg ~8.9), which artificially constrains the contextual signal that the method is explicitly designed to leverage. Furthermore, no computational analysis (offline pipeline cost, training throughput, inference latency/memory during multi-facet beam search) is provided, making it impossible to assess the practical trade-off against static tokenizers in production-scale systems.

## Nice-to-Haves
- Ground the explainability claims by computing mutual information between personalized semantic IDs and explicit item metadata (categories, price tiers, genres) or user cluster labels, rather than relying solely on LLM-as-a-judge accuracy scores.
- Visualize top-K beam search decoding paths for multifaceted items, mapping distinct semantic ID prefixes to categorically different recommended items to demonstrate active intent disentanglement during generation.
- Include concrete failure cases where personalized clustering produces semantically inconsistent or redundant IDs, exposing the true operational boundaries when user history is noisy or items lack distinct facets.

## Novel Insights
The paper effectively reframes the inductive bias of autoregressive recommendation by treating item representations as malleable intent facets rather than static points in a universal embedding space. By demonstrating that semantic IDs can act as dynamic, context-dependent routing signals, the work decouples universal item similarity from personalized recommendation logic without altering the underlying generative architecture. This suggests that injection of personalization at the *lexical* level can yield stronger gains than attempting to recover user-specific similarity solely through decoder-level attention or prompt conditioning.

## Potentially Missed Related Work
- *Kudo (2018) Subword Regularization & Kudo & Richardson (2018) SentencePiece* — The stochastic target replacement ($\gamma$) closely parallels subword regularization strategies in NLP. Explicitly framing PCTX's augmentation as a recommendation-specific variant of multi-candidate sampling could strengthen the theoretical grounding of the training procedure.

## Suggestions
- Add a "Static Multi-ID + Augmentation" baseline to strictly isolate the contribution of *contextual personalization* from the effects of mere token diversity and implicit label smoothing.
- Implement or thoroughly discuss a teacher-student distillation scheme where the context encoder or codebooks receive gradient signals fine-tuned via the GR loss, providing an empirical bound on the performance gap introduced by the decoupled pipeline.
- Report explicit computational metrics: offline pipeline runtime, training throughput comparison, and inference latency/memory overhead during multi-facet beam search. Additionally, extend evaluation to longer sequence lengths or variable-length settings to verify that context aggregation scales effectively beyond short interaction histories.

---

## dCtkwjkK0E

- GT: Reject (avg 2.0)
- Predicted: N/A (3.2/10)
- Match: N/A

### Final Review

## Summary
This paper proposes a dataset-centric active learning framework for flow matching models with continuous conditions, targeting expensive scientific simulation domains like aerodynamic shape design. By modeling the velocity network as a piecewise-linear interpolator, the authors derive a theoretical trade-off between dataset composition and generation quality, yielding two antagonistic query strategies ($Q_D$ for diversity, $Q_A$ for accuracy) and a hybrid weighting mechanism evaluated on synthetic and CFD datasets.

## Strengths
- **Well-Motivated Scientific Application & Data-Centric Insight:** The paper correctly identifies that uniform label coverage is often infeasible in CFD-driven shape design due to high simulation costs. The conceptual insight—that label-space density governs accuracy while intra-label sample multiplicity governs diversity—provides an intuitive, geometric lens for managing generative performance without model-internal diagnostics.
- **Computationally Decoupled Query Design:** By relying on lightweight RBF surrogates and dataset-level distance/entropy calculations instead of iterative gradient or uncertainty computations through the deep flow matching model, the method significantly reduces the computational overhead of the selection step itself.

## Weaknesses
- **Unverified Core Theoretical Assumption:** The entire analytical framework and the derivation of the diversity-accuracy trade-off hinge on the hypothesis that trained flow matching networks behave as exact Continuous Piecewise-Linear (CPWL) interpolators. This is stated without empirical validation. Modern flow matching architectures trained with standard optimizers produce smooth, non-linear approximations; without demonstrating that the trained velocity fields actually exhibit condensation to piecewise-linear boundaries, the theoretical bounds and the strict "dataset composition" mechanism remain heuristic rather than rigorous. This disconnect fundamentally weakens the claim of "rigorous theoretical characterization."
- **Flawed Diversity Metric Undermines Empirical Claims:** The proposed diversity metric (Equation 8) is simply the average pairwise Euclidean distance of generated shapes. This metric is known to be highly sensitive to outliers and boundary noise, often rewarding topologically broken or scattered samples rather than meaningful mode coverage. The paper's finding that $Q_D$ outperforms a model trained on the full dataset strongly suggests the strategy exploits this metric vulnerability to select extreme geometries, which casts serious doubt on the validity of the reported diversity improvements.
- **Insufficient Baselines for a Deep Generative Setting:** The empirical evaluation compares the proposed strategies only to classical AL methods (Coreset, Query-by-Committee with SVR/Random Forest/XGBoost, and Anchor). These surrogates operate on classical feature spaces and lack the representational capacity to capture deep FM uncertainty. The absence of modern deep AL standards (e.g., BADGE, MC-Dropout ensemble variance on the velocity field, or recent diffusion/FL-specific uncertainty sampling) makes it unclear whether the proposed heuristics genuinely outperform contemporary deep learning-centric selection strategies.
- **Lack of Downstream Physical Validation for Stated Application:** The paper positions aerodynamic shape design as the primary use case, yet evaluates success solely via condition MSE. There is no verification of physical realizability, mesh validity, or actual aerodynamic performance of the queried shapes. A strategy achieving low condition MSE could easily return degenerate or non-aerodynamic geometries, rendering the results practically irrelevant to engineering pipelines where functional validity is paramount.
- **Overstated Efficiency Claims:** While the paper argues that decoupling the query from the FM training saves computational budget, the evaluation protocol still requires training the 8-layer, 512-hidden-unit flow matching model for 4,000,000 steps per AL round. The RBF surrogate overhead is negligible compared to this target model training. Without FLOP/wall-clock accounting or budget-scaling curves showing drastic reductions in required simulation/ training cycles, the practical efficiency gains are unsubstantiated.

## Nice-to-Haves
- Provide standard error bars or variance shading across multiple random seeds for the AL learning curves, as active selection can be highly sensitive to initialization.
- Clarify how the initial labeled pool size, total query budget, and stopping criteria scale with dataset size and dimensionality.
- Report the predictive accuracy ($R^2$ or MAE) of the RBF surrogate on held-out shapes, as query quality is directly bottlenecked by surrogate reliability.

## Novel Insights
The paper's strongest conceptual contribution is reframing the diversity-accuracy trade-off not as a model-internal regularization or loss-weighting problem, but as a geometric property of dataset composition in the label space. By positing that generation accuracy is bounded by the diameter of label subregions while diversity scales with the multiplicity of samples sharing identical conditions, it decouples query design from expensive gradient computations. Even if the strict theoretical derivation requires future validation, this dataset-centric perspective offers a fresh, computationally tractable paradigm for steering conditional generative models where label acquisition is the true bottleneck.

## Potentially Missed Related Work
- *Active Learning for Diffusion/Flow Models (e.g., uncertainty-aware sampling via score ensembles or trajectory variance)* — Directly relevant as modern baselines that leverage the generative process itself rather than classical surrogates.
- *Critiques of Distance-Based Diversity Metrics* — Relevant to contextualize why mean pairwise Euclidean distance fails to capture semantic diversity in high-dimensional shape spaces and how spectral/covariance-based metrics (like the proper Vendi score) address this.
- *Bayesian Optimization for Aerodynamic Design* — Highly relevant for comparing against established scientific ML pipelines that already handle expensive simulation budgets and constraint filtering.

## Suggestions
- Replace the average pairwise Euclidean distance with a robust, community-accepted diversity metric (e.g., the spectral Vendi score, 1-NN precision/recall, or Fréchet-based shape distances) to ensure diversity claims reflect meaningful geometric variation rather than outlier inflation.
- Incorporate strong deep AL baselines (e.g., BADGE, velocity field ensemble variance) and perform an ablation on the RBF surrogate's impact to prove the proposed strategies' robustness independently of surrogate quality.
- Empirically validate or clearly contextualize the CPWL assumption: report Jacobian condition numbers or linear approximation errors of the trained velocity networks across checkpoints. If the networks exhibit significant smoothness, reframe the theoretical analysis as motivational intuition rather than strict guarantees, and discuss the implications.
- Integrate a physical feasibility pipeline: run generated shapes through a basic mesh validation or fast CFD solver to report validity rates alongside condition MSE, ensuring the method meets the practical requirements of the claimed engineering application.

---

## D5PJX02Jki

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (5.3/10)
- Match: N/A

### Final Review

## Summary
This paper introduces RoPE++, a modification to standard Rotary Position Embeddings that recovers the discarded imaginary component of the complex-valued attention dot product. By mathematically reformulating the imaginary term as a $-\pi/2$ query rotation, the method computes a parallel set of attention heads that preserve RoPE's absolute-relative duality. Two deployment variants are proposed: RoPE++EH (equal head count, halved KV cache/parameters) and RoPE++EC (equal cache, doubled heads). Empirical evaluations on 376M–1.5B models demonstrate consistent long-context improvements, stable scaling up to 50B tokens, and compatibility with established interpolation-based extrapolation techniques.

## Strengths
- **Mathematical Elegance & Implementation Simplicity:** The core derivation—recovering the imaginary component via a parameter-free $-\pi/2$ query rotation—is analytically clean and directly compatible with existing FlashAttention kernels and GQA/MHA architectures without altering KV caching logic for keys/values.
- **Practical Efficiency-Aware Design:** The explicit EH/EC variants directly address real-world deployment constraints (memory-bound inference vs. compute-bound training), with empirical evidence showing meaningful cache reduction and throughput gains for EH, while EC delivers measurable long-context accuracy improvements under fixed cache budgets.

## Weaknesses
- **Lack of Compute/Parameter-Matched Baselines for RoPE++EC:** The EC variant doubles the number of attention groups and scales the output projection $W_o$ while maintaining KV cache parity. The paper compares RoPE++EC against vanilla RoPE but omits a capacity-matched control (e.g., standard RoPE with doubled heads or increased intermediate/hidden dimensions). Without this, it remains ambiguous whether the observed gains originate from the inductive bias of the recovered imaginary component or simply from increased parameter count and attention capacity.
- **Sub-Billion Scale Validation Limits Generalizability:** All experiments cap at 1.5B parameters trained on ≤50B tokens. Positional encoding benefits, head redundancy dynamics, and optimization landscapes are known to shift non-linearly at production scales (7B+). The absence of validation beyond this threshold raises concerns that the observed trends may reflect under-optimized or scale-limited training regimes rather than a fundamental architectural improvement.
- **Over-Reliance on Synthetic Long-Context Metrics & Training Dependence:** The primary long-context evaluation relies exclusively on RULER and BABILong. While useful for tracking capability retention, these synthetic benchmarks do not adequately capture retrieval degradation, multi-hop reasoning, or discourse coherence measured in modern LLM suites (e.g., LongBench v2, InfiniteBench). Furthermore, the method explicitly requires continued pre-training and offers no plug-and-play zero-shot extrapolation; this training dependency is downplayed in favor of cache-parity framing, limiting immediate adoption for inference-only context extension.

## Nice-to-Haves
- Deeper representational analysis (e.g., SVCCA, cross-layer attention correlation matrices) to rigorously quantify functional orthogonality between real and imaginary heads, complementing the coarse noise-injection proxy.
- Formal pseudocode clarifying gradient routing and softmax normalization when K/V projections are shared across real and imaginary groups under GQA.
- Distance-dependent perplexity/accuracy breakdowns mapping token prediction fidelity to relative offsets $(t-s)$ to empirically verify the theoretical "slow decay" property of imaginary attention.
- A practitioner decision framework outlining when EH vs. EC is preferable based on hardware bottlenecks (memory-bound vs. compute-bound) and target sequence lengths.

## Novel Insights
The paper reframes RoPE not merely as a relative-position routing mechanism, but as a complex-phase separation problem. By exposing the intrinsic mathematical structure of RoPE's complex dot product, the authors identify that the standard practice of truncating to the real component discards a phase-shifted representation that naturally exhibits slower characteristic decay. The insight that this discarded term can be resurrected via a deterministic, learnable-parameter-free query rotation—effectively splitting semantic aggregation (real) and long-range tracking (imaginary) into parallel, structurally constrained attention streams—offers a principled alternative to heuristic interpolation or sparse attention patterns for long-context modeling.

## Potentially Missed Related Work
- Complex-valued attention optimization & phase-aware routing in sequence models (e.g., complex transformer variants in multimodal/speech literature could be better contrasted to clarify why RoPE's phase loss is uniquely problematic for text-based positional encoding).
- No critical missed foundational works; the bibliography adequately covers RoPE variants and complex NN surveys. However, deeper discussion of how phase separation in attention aligns with Fourier/spectral attention theory would strengthen theoretical grounding.

## Suggestions
1. **Add a capacity-matched control:** Train a standard RoPE baseline with doubled attention heads/increased $W_o$ size matching RoPE++EC's parameter budget to isolate the phase-shift inductive bias from trivial capacity scaling.
2. **Validate at ≥7B scale or provide rigorous scaling laws:** Extend pre-training or continued fine-tuning to at least one 7B-class model to confirm that attention complementarity and cache-accuracy trade-offs hold in modern LLM regimes.
3. **Incorporate real-world long-context benchmarks:** Supplement RULER/BABILong with LongBench v2 or InfiniteBench to demonstrate robustness on multi-document QA, code completion, and long-form summarization tasks that better reflect production workloads.
4. **Clarify compute vs. memory trade-offs explicitly:** Front-load the training/inference FLOP penalty of RoPE++EC (~15–35% TGS reduction at 32k context per Table 11) in the main text to ensure practitioners have a complete efficiency budget before adoption.

---

## X2yzXtH4wp

- GT: Accept (Poster) (avg 5.3)
- Predicted: N/A (4.4/10)
- Match: N/A

### Final Review

## Summary
This paper introduces Ambig-SWE, a benchmark evaluating LLM-based software engineering agents on handling underspecified instructions through a three-stage pipeline: detecting missing information, asking clarifying questions, and integrating responses to resolve tasks. By comparing fully specified, hidden, and interactive settings across multiple proprietary and open-weight models, the authors demonstrate that interaction significantly recovers performance, while exposing model-specific failures in default detection and feedback integration.

## Strengths
- **Structured diagnostic pipeline for agent interaction:** Rather than treating "interactivity" as a monolithic metric, the paper cleanly decomposes it into detection, questioning, and integration. This framework is methodologically sound and directly targets a critical gap in prior ambiguity research, which largely focuses on single-turn or single-missing-detail scenarios.
- **Actionable behavioral insights beyond resolve rates:** The analysis effectively surfaces non-trivial agent strategies, notably the superior "exploration-first" approach of Claude models versus the rigid, turn-wasting protocol-following observed in Qwen 3. The finding that higher information extraction volume does not correlate with higher task success challenges conventional assumptions about interactive agent design.

## Weaknesses
- **Confounded turn budgets undermine cross-model comparisons:** The study allocates 100 interaction turns to Claude Sonnet 4 and Qwen 3 Coder while restricting all other models to 30 turns (§3.1), justifying this as accounting for "greater reasoning capacity." This design fundamentally conflates interaction capability with raw token budget and exploration depth. Consequently, performance gaps in RQ1 and RQ2 cannot be reliably attributed to superior ambiguity handling rather than simply deeper codebase traversal. For a top-tier venue, this compute confound requires strict parity controls or turn-normalized success curves.
- **Statistical rigor and subsample analysis are insufficient:** The Hidden evaluation for Sonnet 4 uses only 100/500 instances (Footnote 4), yet Table 4 reports Wilcoxon signed-rank tests comparing Hidden vs. Interaction across settings. The Wilcoxon test strictly requires 1:1 paired observations; the paper does not clarify if the 100 instances were identically matched or randomly split, casting doubt on the reported p-values. Additionally, Table 1's analysis of navigational vs. informational question efficacy lacks confidence intervals, stratified difficulty controls, or significance testing, making claims about performance degradation statistically fragile given high instance heterogeneity in SWE-Bench.
- **Detection evaluation conflates ambiguity recognition with prompt compliance:** RQ2 measures underspecificity detection by varying instruction phrasing (Neutral, Moderate, Strong). Table 2 shows dramatic FNR shifts based purely on prompt wording (e.g., Qwen 3's 100% FNR across all prompts), which primarily reflects instruction-following brittleness and misalignment with training objectives rather than true representational inability to detect missing information. Without a standalone, prompt-decoupled classification task or an oracle baseline that bypasses questioning, the paper cannot disentangle "failure to detect" from "refusal/unwillingness to interact."
- **Unsubstantiated speculation on data leakage and synthetic fidelity gaps:** The authors speculate in §3.2 that higher Hidden-state success may stem from "data leakage," providing no contamination checks, repository overlap analysis, or versioning controls to support this claim. Furthermore, the GPT-4o summarization aggressively removes stack traces, error messages, and conversational context (§2.1), creating a synthetic distribution that diverges significantly from naturally underspecified developer tickets. While pragmatically motivated, this limits ecological validity and overemphasizes navigational gaps while underrepresenting real-world reproducibility ambiguities.

## Nice-to-Haves
- An ablation injecting oracle ground-truth details directly into the agent's context to isolate the performance ceiling of information integration from the bottleneck of question formulation.
- A granular failure-mode taxonomy for the Hidden setting (e.g., incorrect file localization vs. hallucinated API usage vs. architectural missteps) to causally link underspecificity to patch rejection.
- Efficiency metrics quantifying the token/compute cost per resolved task, including analysis of diminishing returns after $N$ clarification turns.

## Novel Insights
The paper compellingly demonstrates that successful agentic interaction in software engineering depends less on the volume of extracted information and more on the strategy of adaptive integration. Models that independently explore the codebase before querying (e.g., Claude Sonnet) outperform those that aggressively interrogate the user but rigidly execute pre-programmed protocols (e.g., Qwen 3), even when the latter extract more raw details. This reveals a misalignment in current agentic training paradigms, which optimize for autonomous task completion in isolation rather than dynamic, context-aware incorporation of human feedback. The brittle dependence on strong interaction prompts further suggests that underspecificity handling is not an emergent property of scale, but a neglected alignment objective requiring explicit curriculum or preference optimization.

## Potentially Missed Related Work
- *ClarifyGPT* (Mu et al., 2023) and *Clamber* (Zhang et al., 2024) for instruction ambiguity resolution in code/NLG; contrasting their single-turn clarification mechanisms with multi-turn agentic trajectories would strengthen the positioning.
- Recent work on *RLHF/DPO for tool-use and feedback integration* (e.g., trajectory-level fine-tuning for agent loops) could contextualize why open-weight models exhibit rigid protocol-following versus adaptive exploration.

## Suggestions
- **Control exploration depth across all models:** Re-run core experiments with uniform turn budgets or report performance as a function of normalized step counts to isolate genuine interaction gains from compute advantages.
- **Decouple detection from prompting:** Add a binary classification task where the model solely labels inputs as underspecified vs. complete without generating solutions, or report how many interactions in RQ1 actually changed the trajectory vs. redundant confirmations.
- **Replace speculative leakage claims with empirical checks:** Provide training cutoff dates relative to SWE-Bench Verified repository commits, or remove the leakage hypothesis if unverified. Clearly state the 1:1 pairing methodology for the Sonnet 4 Wilcoxon tests and add bootstrap confidence intervals to Figure 3 and Table 1.

---

## ZNAY3ivd62

- GT: Reject (avg 4.0)
- Predicted: N/A (4.6/10)
- Match: N/A

### Final Review

## Summary
GUI-Spotlight introduces an iterative, tool-augmented visual grounding pipeline for GUI agents that dynamically crops and refines screen regions across multiple reasoning turns. The model is trained via a three-stage process combining SFT warmup, a modified GSPO objective, and a cross-entropy auxiliary loss to stabilize multi-turn policy updates. Evaluated on ScreenSpot-Pro, UI-Vision, and OSWorld-G, the approach reports competitive accuracy among 7B models while claiming high sample efficiency (~18.5K fine-tuned samples).

## Strengths
- **Transparent algorithmic exploration:** Section 4 provides a systematic, empirical comparison of RL variants and reward designs, explicitly documenting negative results and discarded modifications. This candid reporting of what *does not* work is highly valuable for the agentic VLM community.
- **Effective training stabilization recipe:** The auxiliary cross-entropy loss ($J'$) applied to format-valid completions demonstrably prevents policy collapse and syntactic degradation during multi-turn RL, as evidenced by the stable training curves in Figure 3.
- **Validated multi-step reasoning over naive prompting:** The ablation in Section 5.4 convincingly shows that training iterative tool coordination yields substantial accuracy gains over both single-shot inference and untrained multi-turn prompting/cropping, confirming that the gains stem from learned policy optimization rather than increased context alone.

## Weaknesses
- **Unquantified inference overhead and compute trade-offs:** The iterative agentic loop inherently multiplies token consumption, memory footprint, and wall-clock latency. The paper specifies a 15k token limit for Stage 3 but provides zero metrics on average refinement steps, total inference tokens per query, or runtime latency. Without an accuracy-vs-compute Pareto analysis, the method's practical viability and deployment feasibility are unassessable.
- **Template-driven dataset limits compositional generalization:** The high-resolution dataset relies on rigid, template-based instructions (e.g., “Click the ‘L’ button”, Appendix A.4). This reduces the task to simple keyword-to-element matching and entirely bypasses spatial, contextual, or multi-hop reasoning required in real-world GUI navigation. It remains highly questionable whether the model learns robust grounding or merely overfits to direct lexical cues.
- **Lack of statistical validation for stability and SOTA claims:** All benchmark results are reported as single-point accuracies. Given the extreme seed-sensitivity and hyperparameter fragility of GSPO/GRPO, the marginal absolute gains (~2.2%) over existing 7B baselines and claims regarding training stability lack error bars or multi-seed validation, rendering them statistically unsubstantiated.
- **Missing tool-level ablations marginalize the core contribution:** The framework hinges on "coordinated multi-tool use," yet provides no ablation removing individual tools (`crop`, `extract`, `find_color`), nor any analysis of invocation distributions across success/failure cases. It is unclear whether the multi-tool design enables genuine adaptive reasoning or if performance is overwhelmingly driven by a single primitive.

## Nice-to-Haves
- Quantify coordinate offset error propagation across multi-turn crops to assess compounding drift on high-density layouts.
- Provide gradient magnitude or rollout variance statistics to explain the heuristic decay of $\lambda$ between Stages 2 and 3, and the sparse vs. dense Answer reward trade-offs.
- Release exact tool hyperparameters (color thresholds, window strides, quadrant boundaries) and full system prompts to enable strict reproducibility.

## Novel Insights
The paper’s most substantive observation is not the cropping mechanism itself, but the empirical finding that RL collapse in multi-turn, tool-augmented VLMs can be reliably mitigated by a simple auxiliary cross-entropy loss conditioned on syntactic validity. This, combined with strict data filtering and bucketed sampling, suggests that sample-efficient agentic grounding relies less on algorithmic complexity and more on constraining the policy's exploration space to structurally sound trajectories during early RL phases. The negative results further highlight that many advanced RL modifications (e.g., top-$p$ uncertainty filtering, continuous reference policy updates) actively degrade performance in this specific grounding domain, offering a practical pruning list for future agentic RL work.

## Potentially Missed Related Work
- Test-time scaling and iterative refinement methods in vision-language models (e.g., dynamic region proposal, chain-of-thought with adaptive cropping). Relevant for properly contextualizing the novelty and overhead of the multi-step pipeline.
- Process-supervision and step-wise reward decomposition in agentic RL. Could strengthen the analysis of intermediate tool rewards vs. terminal grounding accuracy.

## Suggestions
- Report average/maximum refinement steps, total token consumption, and wall-clock latency per query across all benchmarks, and plot an accuracy-vs-compute curve against single-pass baselines to explicitly justify the iterative design.
- Add a tool-ablation study (removing `extract` and `find_color` individually) and provide failure-mode correlation analysis to prove the necessity of multi-tool coordination and characterize when the policy defaults to brittle heuristics.
- Validate key empirical claims with multi-seed runs (≥3) and report mean ± standard deviation in all result tables to substantiate statistical significance over competing 7B models.

---

## xFo13SaHQm

- GT: Accept (Poster) (avg 6.5)
- Predicted: N/A (4.7/10)
- Match: N/A

### Final Review

## Summary
This work addresses "copy-paste" artifacts in identity-consistent image generation, a failure mode where models rigidly replicate reference appearances instead of synthesizing natural variations. The authors release MultiID-2M (a large-scale paired celebrity dataset) and MultiID-Bench (an evaluation framework featuring a novel Copy-Paste metric, $M_{CP}$), and propose WithAnyone: a FLUX-based model trained via a 4-phase curriculum, a ground-truth-aligned identity loss, and an extended-negative contrastive objective. Empirical results suggest a Pareto improvement over existing baselines in balancing Sim(GT) with artifact reduction.

## Strengths
- **Principled Benchmark & Metric Design:** The formalization of the copy-paste artifact via $M_{CP}$ (Eq. 2), which prioritizes Sim(GT) over Sim(Ref) and normalizes by reference-target angular distance, correctly reframes how identity customization should be evaluated. The correlation analysis with human rankings (Appendix H) provides useful initial validation.
- **High-Quality Data Curation & Phased Training Pipeline:** MultiID-2M fills a tangible gap by providing cross-paired multi-ID references. The 4-phase training strategy (reconstruction → caption alignment → paired de-biasing → quality refinement) is logically motivated, and ablations (Table 3, Fig. 5) demonstrate consistent, measurable reductions in copy-paste scores without collapsing identity fidelity.

## Weaknesses
- **Structural Train-Test Mismatch in Landmark Supervision:** The core Ground-Truth Aligned ID Loss (Sec. 5.1, Eq. 14) relies on GT facial landmarks to align generated faces during training, circumventing noisy intermediate denoising. This creates a hidden distribution shift: at inference, landmarks must be predicted from generated outputs, and the training objective inherently penalizes valid geometric deviations (e.g., extreme pose/expression changes) that diverge from the GT alignment. The paper claims this "implicitly supervises generated landmarks" but provides zero empirical validation on extreme pose controllability or inference-time alignment robustness. Without this evidence, the claimed "controllability" remains constrained to variations close to the training distribution's alignment prior.
- **Insufficient Statistical Rigor & Metric Fragility:** The primary trade-off claim rests on a 435-case benchmark with no reported generation variance, confidence intervals, or multi-seed error bars. For a paper asserting it "breaks" an observed fidelity-variation trade-off, this lacks statistical grounding. Furthermore, $M_{CP}$ divides by $\max(\theta_{tr}, \varepsilon)$. When reference and GT images are visually near-identical ($\theta_{tr} \approx 0$), the metric becomes numerically volatile and ranking-sensitive to epsilon scaling. The frequency of such edge cases in the benchmark is unreported, leaving the stability of the reported rankings unverified.
- **Dataset Contamination Risks & Demographic Bias:** Pair formation relies on automated ArcFace clustering with fixed similarity thresholds (Sec. 3 states 0.4, App C.1 states 0.5), yet the precision/recall of this matching is never quantified. False-positive pairings directly poison the InfoNCE contrastive objective by treating distinct identities as the same person, artificially inflating identity similarity claims. Combined with the heavy celebrity and geographic skew (Fig. 13) and absence of systematic out-of-distribution evaluation, generalization to non-public or diverse real-world demographics remains anecdotal rather than rigorously demonstrated.
- **Uncharacterized Computational Overhead & Missing Efficiency Context:** The pipeline requires ~100k steps on 8x H100 GPUs with dynamic negative sampling up to 4096 identities. The paper provides no benchmarking of training FLOPs, VRAM peak, inference latency, or trainable parameter count relative to lightweight baselines (e.g., PuLID, InstantID, LoRA adapters). Without quantifying the performance-to-cost ratio, it is unclear whether the metric gains are driven by superior methodology or brute-force compute and extensive data curation, limiting practical reproducibility for standard academic settings.

## Nice-to-Haves
- Controlled ablation scaling the negative pool size (e.g., 256 → 1k → 4k) to identify performance saturation points and justify the 4096 default.
- Development of a reference-only, inference-time proxy metric for controllability that does not require ground-truth targets.
- Explicit failure-case gallery documenting behavior under heavy occlusion, profile views, or stylized prompts, paired with corresponding SimGT/CP scores to define operational boundaries.

## Novel Insights
The paper correctly identifies that reconstruction-heavy training objectives inherently optimize for pixel/trivial feature replication rather than high-level identity manifolds, coining the "copy-paste" artifact to describe this degenerate optimization behavior. The GT-aligned landmark loss is a pragmatic engineering compromise: by trading strict geometric independence for stable, full-timestep gradient signals during flow-matching, it accelerates convergence and mitigates the computational bottleneck of iterative denoising for face extraction, though it introduces the aforementioned alignment priors.

## Potentially Missed Related Work
- **Suggest considering:** Recent work on landmark-free or uncertainty-aware ID preservation methods that decouple identity alignment from geometric detectors, as they directly address the train-test landmark distribution shift highlighted here. The framing could be strengthened by contrasting against these approaches.

## Suggestions
1. **Add statistical rigor:** Report generation variance, bootstrap confidence intervals, or multi-seed error bars across the 435 test cases to statistically substantiate the Pareto shift claim.
2. **Validate alignment robustness:** Conduct an explicit ablation comparing GT-aligned vs. predicted-landmark-aligned ID loss during training and inference, particularly on prompts requesting significant pose/expression divergence, to prove controllability isn't an artifact of the supervision scheme.
3. **Quantify efficiency & dataset reliability:** Provide a compute/latency comparison table against key baselines, and report empirical clustering precision/recall (or a sensitivity analysis on the ArcFace threshold) to rule out label noise driving the contrastive gains.

---

## RpDJz00zNh

- GT: Reject (avg 4.5)
- Predicted: N/A (5.2/10)
- Match: N/A

### Final Review

## Summary
This paper proposes ConciseHint, an in-reasoning intervention framework that reduces the verbosity of large reasoning models by continuously injecting concise hints during autoregressive generation. The method features adaptive hint scheduling (intensity inversely correlated with current output length) and dynamic injection positioning to balance efficiency and accuracy. Experiments across multiple state-of-the-art models and reasoning benchmarks demonstrate 40–65% token reduction with maintained accuracy and seamless compatibility with existing efficiency baselines.

## Strengths
- **Novel "in-reasoning" steering paradigm:** The work successfully identifies and operationalizes an orthogonal axis to efficiency improvement. Unlike static input prompts or post-training compression, continuous mid-generation intervention directly modulates the reasoning trajectory, offering a fresh mechanism to curb overthinking and redundant self-correction.
- **Strong empirical validation and modularity:** Results span multiple architectures (Qwen3 1.7B–30B, DeepSeek-R1-14B) and diverse benchmarks (GSM8K, AIME24, GPQA-Diamond). The framework consistently lowers token usage by substantial margins and acts as a plug-and-play accelerator that composes effectively with prompting, early-exit, and token-filtering methods, demonstrably raising the overall efficiency upper bound.
- **Well-motivated adaptive heuristics with rigorous ablations:** The linear interval scheduling and dynamic position selection are simple yet highly effective. Ablations provide concrete evidence that fixed intervals severely degrade hard-task performance (Table 3) and static injection positions cause either accuracy collapse or compute inflation (Table 4), validating the necessity of the proposed scheduling logic.

## Weaknesses
- **Fundamentally reactive and fragile complexity proxy:** Using current generation length $l_k$ as a proxy for query complexity assumes monotonic correlation, which is frequently violated in practice. If a model enters a repetitive verification loop or hallucinates on an easy query, $l_k$ inflates, causing the hint interval to widen and intensity to drop precisely when intervention is most needed. Conversely, all queries receive maximum intensity early on, which risks disrupting the initial planning steps of genuinely complex problems. Without a mechanism to detect generation stagnation or estimate input difficulty independently of output length, the adaptation mechanism can actively undermine its own goals.
- **Incomplete baseline positioning and lacking statistical rigor:** The main results (Table 1) omit direct comparisons to recent SFT and RL-based length-compression methods (e.g., CoT-Valve, TokenSkip), relegating strong contemporary baselines like AlphaOne and O1-Pruner to the appendix. For ICLR, positioning against the current SOTA efficiency frontier in the primary evaluation is necessary. Furthermore, reporting accuracy on AIME24 (30 items) without standard deviations, confidence intervals, or significance tests renders minor accuracy fluctuations (-0.67% to +0.91%) statistically unverifiable. Claims of "maintained performance" cannot be rigorously evaluated under these reporting standards.
- **Unrealized system-level throughput implications:** While single-query end-to-end latency is shown to improve, the proposed chunked generation inherently fragments the KV cache. The paper claims relative per-query overhead is negligible (<0.3%) but does not analyze how mid-stream insertions and forced re-prefilling interact with continuous batching, PagedAttention, or memory bandwidth under high concurrency. In production serving, cache fragmentation typically dominates latency considerations; demonstrating sustained throughput gains, not just single-query speedups, is required to substantiate deployment readiness.
- **Insufficient reproducibility for the trained variant (ConciseHint-T):** While the training-free text variant is fully reproducible, ConciseHint-T lacks critical specifications: exact dataset size, number of epochs/steps, learning rate schedule, optimization method, and the precise mechanism for embedding injection during inference (e.g., soft prompt hooking vs. token-space quantization). Without these details, the claim that learned embeddings capture transferable concise patterns cannot be independently verified or utilized by the community.

## Nice-to-Haves
- A control experiment injecting random noise or semantically neutral tokens to isolate whether efficiency gains stem from the specific "concise" semantic signal or merely from breaking the autoregressive generation rhythm.
- Visualization of the optimized hint embeddings (e.g., nearest-neighbor tokens or trajectory in activation space across $\gamma$ values) to verify they encode meaningful compression patterns rather than collapsing into trivial EOS or punctuation biases.
- Reporting accuracy-vs-token Pareto curves for strong baselines (e.g., varying confidence thresholds in early-exit methods) alongside the $\gamma$ interpolation curve to provide a unified efficiency-quality frontier comparison.

## Novel Insights
The work effectively demonstrates that LRM verbosity is not a static property but a dynamically steerable trajectory. By showing that shorter, intermittently interrupted reasoning chains can preserve or occasionally exceed baseline accuracy, ConciseHint challenges the prevalent assumption in modern LRMs that longer CoT inherently yields better reasoning quality. This shifts the paradigm from passive length penalization to active, trajectory-aware correction, suggesting that efficiency and reasoning fidelity can be co-optimized through timely, lightweight interventions rather than solely through pre-training data curation or reward shaping.

## Potentially Missed Related Work
- **TokenSkip (Xia et al., 2025)** and **CoT-Valve (Ma et al., 2025)** directly learn compressible reasoning steps via SFT and should be moved from related work discussion into the main empirical comparison to rigorously benchmark the training-free vs. compression paradigms.
- **Adactrl (Huang et al., 2025)** and **Thinkless (Fang et al., 2025)** explore difficulty-aware budgeting and adaptive thinking boundaries. While they differ methodologically, they address the same core tension between query complexity and generation length, and contrasting their proactive estimation approaches with ConciseHint's reactive scheduling would strengthen the technical narrative.

## Suggestions
- **Decouple complexity estimation from generation length:** Integrate a lightweight, input-aware complexity predictor (e.g., initial prefix perplexity or a cheap classifier) to seed $\alpha$ and $\beta$, or implement a stagnation detector (n-gram repetition or entropy drop) that resets or overrides the interval scheduler when the model enters repetitive loops.
- **Strengthen empirical positioning and statistical reporting:** Move AlphaOne and O1-Pruner into the main comparison table. Report mean ± standard deviation across all runs, and apply paired statistical tests (e.g., bootstrap or McNemar's) to verify that accuracy preservation on small benchmarks (AIME) is statistically significant rather than noise-level variance.
- **Provide complete training and serving specifications:** Add a dedicated appendix section detailing the ConciseHint-T training protocol (dataset statistics, hyperparameters, optimizer, compute hours) and explicitly describe the inference pipeline for embedding injection. Include a throughput analysis using vLLM continuous batching under varying concurrent request loads to demonstrate how KV cache fragmentation scales and quantify the actual serving efficiency gain.

---

## j3htU5i01r

- GT: Reject (avg 4.0)
- Predicted: N/A (3.8/10)
- Match: N/A

### Final Review

## Summary
The paper proposes a compositional meta-learning framework that models tasks as sequences of reusable neural modules selected by a gating RNN. By formalizing the architecture as a probabilistic generative model, it replaces gradient-based test-time fine-tuning with particle-filter-based state inference, demonstrating single-episode task acquisition and robustness to sparse feedback on synthetic rule-learning and motor-control tasks.

## Strengths
- **Principled Probabilistic Formulation:** The explicit separation of within-module dynamics ("syllables") and between-module transition statistics ("grammar") is theoretically clean. Casting test-time adaptation as Bayesian state estimation over latent module sequences rather than weight-space optimization is an elegant and well-motivated departure from standard gradient-centric meta-learning.
- **Effective Demonstration of Hypothesis-Constrained Inference:** The controlled experiments convincingly show how learned gating priors naturally constrain the hypothesis space during gaps in feedback, enabling accurate long-horizon sequence inference from sparse signals without parameter updates.
- **Exemplary Reproducibility & Transparency:** Full code, trained weights, hyperparameter specifications, and detailed derivations for the particle-filter training loop are provided, meeting the highest standards for methodological openness and enabling immediate replication.

## Weaknesses
- **Highly Engineered Tasks Mask True Compositionality:** The synthetic tasks use rigid, fixed-duration chunks (operations repeat exactly 3, 4, or 5 timesteps). This structure strongly suggests the gating network may learn a step-counter rather than generalizable transition grammar. Without testing robustness to stochastic durations, variable chunk lengths, or probabilistic transitions, the core claim of compositional generalization remains unverified. This matters because true compositionality requires recombination across flexible temporal structures, not memorized repetition schedules.
- **Unquantified Inference Compute & Scalability Limits:** Eliminating parameter updates is traded for online particle filtering (K=250) with full RNN state tracking per timestep. The paper provides no analysis of wall-clock latency, memory footprint, or FLOPs compared to few-shot gradient updates. Furthermore, standard particle filters suffer from weight degeneracy as sequence length increases, yet the paper does not discuss where this approximation breaks down. This matters because without a compute-sample efficiency trade-off analysis, the practical viability of the framework for real-world horizons or continuous control is unclear.
- **Missing Critical Ablations & Distributional Reporting:** Test-time success is illustrated primarily through single trajectory visualizations rather than statistical distributions across seeds/tasks. Crucially, there is no ablation on particle count ($K$) or module capacity to show how inference fidelity degrades as resources tighten. This matters because particle-based inference is inherently stochastic; reporting only cherry-picked or averaged trajectories obscures posterior variance and potential failure modes, which is unacceptable for evaluating a probabilistic reasoning system.
- **Severe Training Fragility & Hard-Coded Capacity:** The authors acknowledge a fundamental "chicken-and-egg" instability in joint module-gating optimization, mitigated only by carefully tuned low-weight initialization (`winit=0.01`). Coupled with a fixed, oracle-known number of modules, the framework lacks any mechanism for dynamic capacity allocation, pruning, or automatic component discovery. This matters because real compositional domains rarely expose the exact number of underlying primitives, and reliance on brittle initialization undermines the method's robustness for open-ended deployment.

## Nice-to-Haves
- Report wall-clock inference time, memory usage, and training variance across standard random initializations (e.g., Kaiming/He) to empirically characterize the method's computational profile and optimization robustness.
- Include a comparison against modern sparse MoE routing or learned in-context architectures to better contextualize the gains relative to current state-of-the-art modular systems.

## Novel Insights
The paper's strongest conceptual contribution is reframing rapid meta-learning as constrained hypothesis testing over a learned discrete combinatorial space. By treating modules as atomic computations and gating statistics as a generative grammar, it elegantly demonstrates how explicit inductive biases can compress the hypothesis space enough to allow single-shot inference. This bridges classical switching dynamical systems with modern neural routing, suggesting that sample efficiency in meta-learning may come from architectural constraints on *search* rather than optimization landscape smoothing. However, the current implementation's reliance on strictly sequential, non-overlapping, fixed-duration primitives reveals a fundamental gap between this idealized probabilistic parsing and the continuous, concurrent, and stochastic blending characteristic of real-world skills or language.

## Potentially Missed Related Work
- **Infinite/Nonparametric HMMs & Hierarchical Dirichlet Processes for Switching Dynamical Systems** — relevant for addressing the fixed module count and providing principled priors for dynamic component allocation during test-time inference.
- **Differentiable Routing & Soft-MoE Variants (e.g., ST-MoE, Gumbel-Sinkhorn routing)** — relevant for contextualizing the training instability observed with hard categorical routing and for comparing gradient dynamics in sparse modular architectures.

## Suggestions
1. **Probe Grammatical Abstraction vs. Step Counting:** Modify the task distributions to include randomized or variable chunk durations during training/testing. If performance collapses when durations deviate from the learned fixed counts (3/4/5), it confirms the gating network merely memorizes repetition statistics rather than learning transferable transition rules.
2. **Systematically Characterize the Particle Filter Trade-offs:** Run inference sweeps over $K \in \{10, 50, 100, 500\}$ and report accuracy, runtime, and posterior entropy/weight degeneracy metrics across seeds. This will establish practical scaling limits and validate whether the probabilistic framework remains robust under tighter computational budgets.
3. **Benchmark Against Stronger Modern Baselines:** Evaluate on slightly higher-dimensional, continuous benchmarks (e.g., simplified robotic manipulation or sequential vision tasks) and compare against transformers with in-context learning and continuous latent-conditioning models. This will isolate whether the observed sample efficiency stems from compositional inference or simply from avoiding weight-space optimization on small synthetic tasks.

---

## ngOOlatCK6

- GT: Reject (avg 5.3)
- Predicted: N/A (4.8/10)
- Match: N/A

### Final Review

## Summary
This paper addresses the problem of identifying the minimal set of candidate nodes for single-node conditional interventions in a causal bandit setting. The authors theoretically characterize this set as the Lowest Strict Common Ancestor (LSCA) closure of the target variable's parents, prove its uniqueness and equivalence to deterministic atomic-intervention superiority, and propose C4, a linear-time algorithm to compute it. Empirical results demonstrate substantial search-space pruning in sparse graphs and reduced node-selection regret when integrated into a contextual UCB bandit baseline.

## Strengths
- **Rigorous and Novel Graphical Characterization:** The paper provides the first complete theoretical characterization of the minimal search space for conditional (policy-driven) interventions. The equivalence between conditional-intervention superiority and deterministic atomic superiority (Proposition 4) is a clever theoretical bridge that elegantly reduces an infinite policy-space optimization problem to a tractable graph-theoretic one.
- **Provably Correct and Computationally Optimal Algorithm:** The C4 algorithm computes the exact minimal set (mGISS) in $O(|V|+|E|)$ time using an elegant connector-based recursion. The correctness proof (Theorem 16) and the Λ-structure characterization (Theorem 12) are mathematically sound, well-structured, and offer a clean alternative to heuristic common-ancestor approaches.
- **Clear Scope and Positioning:** The paper correctly delineates its contribution from prior work on hard/multi-node interventions (e.g., Lee & Bareinboim) and explicitly frames C4 as a pre-processing step that can be plugged into any downstream causal bandit algorithm, avoiding unnecessary overlap with policy-learning literature.

## Weaknesses
- **Selection Bias in Pruning Evaluation:** The empirical pruning experiments exclusively target nodes $Y$ with the maximum number of ancestors (Section 6, Footnote 8). This design choice systematically inflates reported space-reduction ratios, as deeper target nodes naturally induce larger ancestral subgraphs where LSCA pruning is most aggressive. The claims of "substantial" real-world utility lack validation on a representative distribution of target node depths, raising concerns about generalizability in practical causal graphs where targets may be shallow or sparsely connected.
- **Narrow Regret Metric and Policy Learning Disconnect:** The experiments measure cumulative regret *only* with respect to node selection, deliberately ignoring the policy learning component over $\mathcal{Z}_X$. While this isolates C4's effect, it misrepresents the end-to-end challenge of conditional causal bandits. The used tabular UCB per context does not scale with $|\mathcal{Z}_X|$, and without analyzing how $|mGISS|$ interacts with the exponential context space, the claim that pruning "accelerates convergence" remains empirically unsubstantiated for realistic, high-dimensional conditioning sets.
- **Absence of Formal Sample Complexity Guarantees:** For a venue like ICLR that emphasizes sequential decision-making theory, empirical regret curves are insufficient. The paper lacks a formal pseudo-regret bound explicitly showing how the reduction from $|An(Y)|$ to $|mGISS|$ translates into provable sample complexity improvements or tighter convergence rates. Without this, the theoretical contribution remains decoupled from the bandit learning dynamics it aims to support.

## Nice-to-Haves
- **Robustness Analysis for Graph Uncertainty:** While the paper briefly notes applying C4 to candidate graph families, a formal or empirical sensitivity analysis showing how mGISS degrades under minor edge perturbations or latent confounding would significantly strengthen practical deployment guidelines.
- **Downstream Algorithm Integration:** Testing C4 alongside more advanced causal bandit algorithms that explicitly model do-effects or share statistical strength across contexts (rather than independent tabular UCB instances) would better demonstrate its utility as a true pre-processor.

## Novel Insights
The paper successfully reframes a dynamic, policy-dependent intervention selection problem as a static graph traversal task. The introduction of Λ-structures as a necessary and sufficient condition for the LSCA closure offers a visually and mathematically intuitive lens that surpasses traditional Lowest Common Ancestor heuristics, which fail in multi-path ancestral configurations (as correctly illustrated in Figure 1d). The equivalence proof (Proposition 4) is particularly insightful, demonstrating that worst-case SCM constructions for deterministic atomic interventions inherently bound the optimal conditional policy space, a result that cleanly divorces graphical structure from noise distribution assumptions.

## Potentially Missed Related Work
- Works on robust intervention design under partial observability or Markov equivalence classes (e.g., algorithms operating on CPDAGs or essential graphs) could be discussed to contextualize how mGISS might extend to settings where the exact causal DAG is not identified.
- Recent advances in contextual bandit regret scaling with action space cardinality (e.g., linear or non-linear contextual bandits) could be referenced to ground the discussion on how $|mGISS|$ theoretically impacts regret bounds, even if not directly causal.

## Suggestions
1. **Derive or Cite a Formal Regret Bound:** Provide a theoretical analysis bounding the pseudo-regret of a contextual bandit using the pruned action set, explicitly showing the dependence on $|mGISS|$ rather than just plotting empirical curves. This is critical for ICLR's theoretical standards.
2. **De-bias Target Node Selection:** Re-run pruning evaluations over a randomized or stratified sample of target nodes $Y$ (varying depth, in-degree, and ancestor count) rather than cherry-picking maximal-ancestor nodes. Report mean/median pruning fractions to establish realistic expectations.
3. **Address Context Space Scalability:** Include an analysis or experiment demonstrating how the computational and statistical burden of the downstream policy learner scales with $|Z_X|$, and discuss practical strategies (e.g., function approximation, context discretization, or independence assumptions) to prevent the policy learning bottleneck from negating the upstream pruning gains.

---

## USyGD0eUod

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (4.6/10)
- Match: N/A

### Final Review

## Summary
This paper conducts a systematic sanity check for sparse autoencoder (SAE) evaluation by training SAEs on both trained and randomly initialized transformers (Pythia 70M–6.9B) and measuring their performance with standard reconstruction and auto-interpretability (fuzzing AUROC) metrics. The authors find that aggregate auto-interpretability scores often fail to separate trained from random models, sometimes even inflating scores for randomized variants due to their tendency to learn simplistic, single-token features. As a corrective, they propose token-distribution entropy as a proxy for feature abstractness and recommend routine use of randomized baselines in SAE evaluation.

## Strengths
- **Timely and rigorously executed methodological sanity check:** The paper adapts the canonical weight-randomization test to mechanistic interpretability, exposing a critical blind spot in the field's reliance on auto-interpretability leaderboards. The systematic comparison across five initialization variants, six model scales, and comprehensive hyperparameter ablations (Appendices C–G) provides a robust empirical foundation.
- **Clear, reproducible experimental pipeline:** Use of standardized TopK SAEs, open-source evaluation frameworks (EleutherAI/delphi, sparsify), explicit hyperparameters, and transparent compute logging (Appendix K) sets a strong baseline for community reproducibility.
- **Constructive diagnostic proposal:** The introduction of token-distribution entropy successfully correlates with feature lexical diversity and abstractness, providing a cheap, computable signal that diverges meaningfully between trained and random SAEs across layers (Section 3, Figure 20).

## Weaknesses
- **Overstated claims vs. nuanced empirical reality:** The title and abstract assert that metrics "do not distinguish" trained and random transformers, yet the data consistently shows partial or complete separation depending on the metric and scale. Appendix B demonstrates clear AUROC gaps for Pythia 70M–160M. Figure 2 and Section 3 show that reconstruction quality, $L_1$ norms, and CE loss recovery systematically differ across variants. Moreover, randomized variants sometimes achieve *higher* fuzzing AUROC than trained models, indicating a bias toward simpler features rather than mere indistinguishability. This framing overstates the negative result and obscures actionable insights for practitioners.
- **Uncontrolled LLM explainer confound:** The central finding relies entirely on a single 70B-parameter LLM generating explanations and computing fuzzing AUROC. High scores on random SAEs likely reflect the explainer's well-documented tendency to construct plausible post-hoc narratives for any sufficiently sparse or token-concentrated activation pattern. Without controls (e.g., testing smaller explainers, perturbing prompts, or feeding artificially sparse but semantically void activation patterns), the paper cannot disentangle whether auto-interpretability is measuring SAE-learned structure or LLM scorer bias. This is a fundamental methodological gap for a paper whose core claim is that these scores are "insufficient proxies."
- **Proposed entropy metric lacks functional grounding:** Token-distribution entropy is positioned as a proxy for feature "abstractness" and computational relevance, but remains purely correlational. The paper provides no activation patching, ablation, or steering experiments to demonstrate that high-entropy trained features actually drive model behavior, or that low-entropy random features are computationally inert. Without causal validation, the entropy metric risks capturing superficial lexical frequency or embedding alignment rather than genuine mechanistic decomposition.
- **Statistical robustness and architectural scope:** Scaling conclusions (e.g., AUROC convergence at 6.9B, layer-wise trend reversals) rely on single-seed runs for all models ≥410M due to compute constraints. Given the known high variance in SAE convergence, single-seed trends at scale weaken confidence in the scaling narrative. Additionally, experiments are restricted to TopK SAEs. Modern variants (Gated, JumpReLU, Switch-SPD) enforce sparsity and optimize differently; without testing at least one alternative architecture, the generalizability of the auto-interpretability failure to the broader SAE ecosystem remains unverified.

## Nice-to-Haves
- Develop and release a lightweight composite diagnostic combining fuzzing AUROC, token entropy, and a basic causal consistency score to provide practitioners with an immediate alternative to raw auto-interpretability leaderboards.
- Include side-by-side activation heatmaps and context distribution plots for matched-AUROC latents (trained vs. random) to visually ground the claim that random features collapse to trivial token/pattern matching.

## Novel Insights
The paper reveals that automated interpretability scores may not be measuring learned computational structure at all, but rather an LLM's capacity to assign narrative coherence to statistically sparse activation patterns. This reframes the evaluation problem: high auto-interpretability is not evidence of mechanistic insight, but evidence of pattern concentration. Consequently, meaningful SAE validation requires moving beyond score maximization toward complexity-aware metrics and rigorous null-model benchmarking that explicitly filters out features explainable by data sparsity or architectural priors alone.

## Potentially Missed Related Work
- Recent analyses of post-hoc rationalization bias in LLM-based neuron description (e.g., studies demonstrating that larger explainers confidently justify arbitrary sparse patterns) would strengthen the discussion of explainer confounds.
- Work on causal feature validation in SAEs (e.g., activation patching pipelines that quantify feature utility for downstream tasks) is relevant for grounding the entropy proposal and could be added to contextualize the move from correlational to functional evaluation.

## Suggestions
1. **Temper framing and highlight discriminative signals:** Revise the title and abstract to reflect the nuanced findings (e.g., "Aggregate auto-interpretability scores alone fail to isolate learned computation and can be biased by sparsity"). Explicitly summarize which standard metrics retain partial discriminative power so practitioners know what to report alongside fuzzing AUROC.
2. **Isolate LLM explainer bias:** Add a control experiment where you generate artificially sparse activation patterns (or shuffle high-scoring latents across contexts) and run the same auto-interpretability pipeline. Demonstrate whether AUROC remains high, confirming scorer bias. Run a sensitivity analysis with a smaller explainer (8B vs 70B) to test score degradation on random features.
3. **Functionally validate the entropy proxy:** Perform targeted activation patching or steering on a small subset of high-entropy trained latents vs. low-entropy random latents. Show that the former causally impact model outputs while the latter do not. This would directly bridge the correlational observation to the claim of "computational relevance."
4. **Clarify scaling claims or add multi-seed validation:** If compute precludes multi-seed runs at 6.9B, explicitly bound scaling conclusions as preliminary trends rather than definitive results. Alternatively, report variance estimates from the available 70M/160M multi-seed runs and discuss how variance likely scales with dictionary size and depth.

---

## U6ROetm5nW

- GT: Withdrawn (treated as Reject) (avg 4.5)
- Predicted: N/A (4.3/10)
- Match: N/A

### Final Review

## Summary
This paper introduces the first continuous time-space tradeoffs for high-dimensional Kernel Density Estimation (KDE) by integrating asymmetric Locality-Sensitive Hashing (LSH) into the density-constrained approximate nearest neighbor framework. The authors derive asymptotic query time exponents of $\approx 0.05$ (at space $\approx 1/\mu^{4.15}$) and $\approx 0.1865$ (at linear space $\approx 1/\mu$), numerically improving upon prior data-independent bounds while maintaining a cleaner analysis than complex data-dependent alternatives.

## Strengths
- **First formal KDE time-space tradeoff:** The work successfully generalizes the KDE-to-ANN reduction to accommodate asymmetric LSH, yielding a parameterized family of space-query tradeoffs ($\delta \geq 0$) that rigorously extends the theoretical limits of previous KDE data structures.
- **Improved linear-space baseline:** For $\delta=0$, the method achieves a query exponent of $0.1865$, strictly improving upon the $0.25$ bound of prior data-independent approaches without relying on the significantly more complex machinery of adaptive/data-dependent LSH.
- **Transparent numerical validation:** The optimization constraints are explicitly defined and accompanied by a reproducible grid-search script, allowing independent verification of the reported exponent curves, threshold behavior, and plateau dynamics.

## Weaknesses
- **Lack of analytical depth and over-reliance on numerical optimization:** The core theoretical claims—specifically the query exponents $0.05$ and $0.1865$, and the tradeoff plateau at $\delta \approx 3.15$—are derived entirely via numerical grid search. The paper provides no closed-form approximations, analytical bounds, or theoretical characterization of the optimization landscape $\xi(\delta)$. This leaves the mathematical behavior opaque and reduces a central theoretical contribution to a computational artifact rather than a fundamental provable insight.
- **Astronomically impractical space complexity and unanalyzed hidden constants:** The primary result ($\approx 1/\mu^{0.05}$ query time) requires space scaling as $1/\mu^{4.15}$. For even moderate density values (e.g., $\mu = 10^{-4}$), this implies $\sim 10^{16}$ storage, rendering the regime entirely infeasible. Furthermore, the analysis aggressively suppresses polynomial dependencies on $d$, $1/\epsilon$, and asymmetric LSH tree-depth overheads. In realistic high-dimensional settings, these hidden constants and memory bandwidth constraints will dominate asymptotic gains, a practical disconnect that is entirely unaddressed.
- **Heavy deferral of technical proofs and unproven barrier claims:** Critical derivations—including the intermediate-scale collision bounds (Lemma 31), threshold function $\theta(\delta)$, and handling of boundary distance scales—are heavily offloaded to appendices, making the main text non-self-contained. Additionally, the paper asserts that constant-query-time KDE is impossible under current techniques based on heuristic collision arguments but explicitly leaves the lower bound as an open problem. Without a formal hardness proof or tight analysis, the claimed plateau at $\sim 0.05$ reads as a limitation of the chosen asymmetric LSH instantiation rather than a fundamental algorithmic barrier.

## Nice-to-Haves
- Provide a concrete constant-factor or memory-bound calculation demonstrating for which $(\mu, n)$ ranges the proposed structure fits within standard hardware constraints, or include a minimal empirical benchmark against uniform sampling/symmetric LSH to contextualize asymptotic gains.
- Move the core collision probability derivation and threshold parameter analysis into the main text to improve accessibility for readers outside the specialized sublinear algorithms community.

## Novel Insights
The paper effectively maps the multi-scale KDE recovery problem onto an asymmetric LSH parameter space, revealing a rigid structural constraint: aggressively increasing space to accelerate query retrieval only reduces the exponent until it hits a hard floor ($\approx 0.05$). This occurs because the overhead of scanning intermediate distance bands grows faster than the benefits gained from allocating higher query exponents to closer scales. This observation shifts the optimization paradigm for KDE data structures from pure time minimization to navigating a strictly bounded time-space frontier dictated by intermediate-scale hash collisions.

## Potentially Missed Related Work
- **Charikar et al. (2024) & Indyk et al. (2025):** Recent advances in quasi-Monte Carlo data structures and sparse kernel matrix-vector multiplication offer alternative acceleration pathways with different space-time profiles and often tighter dependence on $\epsilon$. Contrasting these asymptotic regimes would better clarify the unique niche (if any) of the asymmetric LSH tradeoff curve in modern ML workloads.

## Suggestions
- Derive provable analytical upper/lower bounds for $\xi(\delta)$ to replace the pure grid-search characterization, or formally prove the claimed impossibility of constant-query-time KDE to substantiate the plateau observation as a fundamental limit rather than an artifact of the chosen ANN construction.
- Explicitly track and bound the suppressed $\text{poly}(d, 1/\epsilon)$ factors, LSH tree construction overhead, and memory access patterns. Provide a realistic complexity breakdown to demonstrate whether the theoretical improvements survive in finite-sample, high-dimensional regimes relevant to machine learning.
- Analyze the robustness of the geometric level-set partitioning when the baseline $\mu$ approximation contains bounded multiplicative error, as practical deployments rarely guarantee exact a priori density estimates and the sampling rates are highly sensitive to this parameter.

---

## bH5M0ts8Y6

- GT: Accept (Poster) (avg 5.0)
- Predicted: N/A (4.8/10)
- Match: N/A

### Final Review

## Summary
This paper proposes VINCIE, a framework that learns in-context, multi-turn image editing by converting raw videos into interleaved multimodal sequences and training a Diffusion Transformer on next-image and segmentation prediction tasks. It introduces MSE-Bench for evaluating multi-turn editing and demonstrates scalability, artifact mitigation across long editing chains, and emergent zero-shot capabilities like story generation and multi-concept composition.

## Strengths
- **Compelling & Scalable Data Paradigm:** The core premise of extracting editing supervision from raw video dynamics, bypassing synthetic pairwise generation, is well-motivated and empirically validated. Figure 5 provides clear evidence of log-linear performance gains on Turn-4/5 as training data scales to 10M sessions, supporting the feasibility of video as a native, scalable editing curriculum.
- **Effective Mitigation of Cumulative Artifacts:** The model directly addresses a critical failure mode in sequential editing. Table 4 and Figure 6 convincingly show that leveraging full interleaved context nearly halves pixel-wise drift (L1/L2) and prevents the compounding of visual degradation that plagues single-turn baselines.
- **Logically Grounded Proxy Task Design:** The joint formulation of next-image prediction (NIP), current segmentation prediction (CSP), and next segmentation prediction (NSP) provides a principled mechanism for teaching spatial grounding and layout planning. The ablation on camera motion disentanglement (Appendix Table 9) further demonstrates thoughtful engineering to separate object dynamics from background shifts.

## Weaknesses
- **Fundamental Contradiction in Training Claims:** The abstract and introduction repeatedly assert that the model is learned "solely" or "exclusively" from video data. However, Figure 14's caption explicitly states training on "interleaved session data from video, T2I data, and T2V data," and Appendix C.7 confirms extensive supervised fine-tuning (SFT) on external pairwise datasets (OmniEdit, SEED, X2I2). The strongest reported results (Tables 1 & 2) rely on this SFT phase. Without cleanly isolating the video-only contribution, the central premise is significantly overstated, and the actual marginal gain from the proposed pipeline remains ambiguous.
- **Statistically Fragile Benchmark Evaluation:** MSE-Bench comprises only 100 test instances evaluated primarily via GPT-4o. The reported human-auto correlation is moderate at best (Pearson *r* ≈ 0.486), which is insufficient to claim reliable benchmarking for long-horizon editing. Furthermore, Table 3 in the main text explicitly uses an "intermediate checkpoint," invalidating direct comparisons with fully converged baselines and severely undermining the rigor of the segmentation ablation claims.
- **Unquantified Foundation Model Dependency & Annotation Noise:** Performance is tightly coupled to a proprietary "in-house" 3B/7B video foundation model initialized from T2V data, making it impossible to disentangle learned editing dynamics from inherited generative priors. Additionally, the VLM transition annotation accuracy is acknowledged at ~75%, yet the paper provides no analysis on how mislabeled transitions or inaccurate RoE masks propagate through multi-turn sequences, leaving a critical gap in robustness evaluation.

## Nice-to-Haves
- Provide concrete efficiency metrics (inference latency, memory throughput) comparing block-wise causal attention against full attention, to better justify its architectural exploration despite current performance trade-offs.

## Novel Insights
The paper effectively reframes video not merely as a generation target, but as an unsupervised curriculum for learning temporal consistency and localized editing dynamics. By treating sparse video transitions as self-supervised editing instructions, VINCIE demonstrates that interleaved multimodal sequences can naturally teach grounding and planning, enabling the model to internalize a "chain-of-editing" that structurally mitigates error accumulation. This suggests that temporal priors in foundation video models are fundamentally aligned with compositional image editing, provided they are unlocked via appropriate multi-turn sequence modeling and explicit spatial proxy tasks rather than isolated pairwise supervision.

## Potentially Missed Related Work
- None identified; the coverage of video-derived editing consistency datasets (UniReal, RealGeneral, FramePainter) and in-context generative frameworks is adequately contextualized.

## Suggestions
- **Strictly Isolate Contribution Claims:** Restructure Tables 1 & 2 to explicitly separate zero-shot video-only pre-training results from T2I/T2V mixed training and SFT-enhanced results. Report the exact marginal gain attributable to the proposed video pipeline to validate the "video-only" narrative.
- **Replace Non-Comparable Ablations & Strengthen Evaluation Benchmarks:** Move intermediate checkpoint results (Table 3) to the appendix or re-run experiments on converged models with reported variance. Expand MSE-Bench to 300+ instances, conduct comprehensive human evaluation across all Turns 1-5 for all baselines, and categorize GPT-4o failures to establish statistical reliability.
- **Conduct Error Propagation & Noise Sensitivity Analysis:** Provide a turn-by-turn failure breakdown categorizing degradation types (spatial drift, instruction confusion, identity collapse). Additionally, include a sensitivity analysis on how varying VLM/segmentation accuracy impacts final editing success, offering practical guidance for scaling the data pipeline.

---

## WhO6Km5Rku

- GT: Withdrawn (treated as Reject) (avg 4.0)
- Predicted: N/A (3.3/10)
- Match: N/A

### Final Review

## Summary
QubitCache proposes a hybrid KV-cache compression framework that retains 15% of semantically critical tokens in classical memory while encoding the attention patterns of the remaining 85% into quantum-inspired amplitude-encoded states (simulated classically). By reconstructing soft attention probabilities from these states and interpolating value vectors, the method claims 7× memory reduction with minimal performance degradation, particularly on multi-hop reasoning tasks.

## Strengths
- **Conceptually aligned with attention topology importance:** The core premise that preserving relational attention structure outperforms binary token eviction is well-motivated and supported by the ablation study (Table 4), which shows a catastrophic 20.4% F1 drop when attention-based selection is removed, versus only ~4% when the quantum encoding is removed. This empirically validates that importance-weighted pattern retention drives compression success.
- **Comprehensive multi-model and multi-benchmark evaluation:** The paper provides systematic results across five language models (4B–8B, scaling to 30B/70B) and six long-context benchmarks. The consistent 92–97% baseline retention despite aggressive 15% token preservation demonstrates practical robustness across diverse architectures and tasks.
- **Clear implementation transparency and classical simulation acknowledgment:** The authors explicitly state that the framework operates via classical simulation on GPUs, providing concrete optimization details (gate fusion, parallel segment encoding, adaptive shot allocation, Appendix A.1.4) that facilitate reproducibility and immediate hardware deployment.

## Weaknesses
- **Static attention prior breaks query-dependent reasoning:** The method computes amplitudes $\alpha_i$ by aggregating historical attention weights $A_{j,i}$ across layers/heads (Eq. 3–5), producing a fixed, query-independent distribution $p_j = |\langle j|\psi\rangle|^2$. During autoregressive generation, this distribution is used as a proxy for attention without modulation by the current query $Q_t$. This fundamentally severs the adaptive mechanism of self-attention, making the method's success on multi-hop reasoning surprising and theoretically misaligned. The paper neither explains how $Q_t$ modulates compressed tokens nor ablates this critical design choice against a classical soft-prior baseline.
- **Misleading complexity claims and unproven quantum advantage:** The abstract claims "compression beyond classical information-theoretic limits" and reports $O(\log N)$ memory complexity, but the classical simulation of $N/512$ segments storing 512 amplitudes each intrinsically scales as $O(N)$ in memory and preprocessing compute. The "quantum-inspired" formulation adds circuit simulation overhead without delivering representational gains over straightforward classical softmax/probabilistic weighting (as evidenced by the marginal ~4% ablation difference). Theoretical guarantees for "rank-$r$ structure with bounded reconstruction error" are mentioned but entirely omitted from the manuscript.
- **Missing system metrics and uneven baseline comparisons:** The paper reports only memory footprint, omitting end-to-end decoding latency, tokens/second throughput, and FLOP counts. For an inference optimization method, classical simulation of amplitude encoding and measurement per step likely introduces substantial decoding overhead that could negate memory savings in production. Furthermore, QubitCache at 15% retention is compared against baselines at ~50% retention, leaving the true performance-compression Pareto frontier unverified.

## Nice-to-Haves
- **Attention heatmap & multi-hop chain visualizations:** Directly comparing reconstructed vs. full attention matrices would clarify whether the method recovers sharp, task-critical spikes or merely produces smoothed approximations that LLMs tolerate.
- **Head/layer specialization analysis:** Investigating how cross-layer/cross-head averaging of attention scores affects specialized heads (e.g., positional vs. syntactic) would strengthen the claim of topological preservation.
- **Theoretical proof appendix:** Providing the complete derivation for the bounded reconstruction error and rank-preserving claims would significantly bolster theoretical credibility.

## Novel Insights
The paper inadvertently reveals that LLMs exhibit remarkable tolerance to soft, historically aggregated attention priors when coupled with smooth value interpolation. The performance retention stems not from quantum mechanics, but from replacing discontinuous binary eviction with a continuous, distance-weighted attention distribution. This suggests that future KV-cache methods need not focus on exact token reconstruction, but rather on maintaining low-rank, probabilistic approximations of the attention graph to sustain long-context coherence.

## Potentially Missed Related Work
- SnapKV (Li et al., 2024), DuoAttention (Xiao et al., 2024), Quest (Tang et al., 2024) — Recent methods that similarly exploit attention sparsity, dynamic token clustering, or streaming KV buffers at high compression ratios. Including these would position QubitCache accurately against the current state-of-the-art.
- KVQuant / KIVI (Liu et al., 2024) — Modern quantization techniques achieving <2-bit precision with minimal degradation. Comparing against these at matched 10–15% effective footprint would isolate whether attention-pattern preservation truly outperforms aggressive numerical compression.

## Suggestions
1. **Reframe the method as a classical probabilistic attention prior:** Either remove the quantum circuit formalism and present the approach as a normalized, history-weighted soft-attention retention strategy, or provide a rigorous ablation proving that the specific amplitude-encoding/measurement pipeline yields statistically significant gains over a direct classical implementation with identical compute/memory budgets.
2. **Add comprehensive system benchmarks:** Report wall-clock decoding latency (ms/token), throughput (tokens/sec), and theoretical/empirical FLOP overhead for QubitCache vs. baselines. Explicitly detail the classical simulation memory budget (statevector storage, interpolation buffers) to correct the $O(\log N)$ complexity claim.
3. **Equalize compression ratios for fair comparison:** Evaluate all baselines at matched 15% retention (or equivalent memory footprint) to properly situate QubitCache on the efficiency-accuracy frontier. Clarify that the reported "15–25% improvement" is relative to 50%-retention baselines.
4. **Address query-independence:** Provide theoretical justification or empirical analysis for using historical attention aggregation as a proxy for future query responses. If dynamic query modulation is infeasible, explicitly bound the method's applicability to tasks where attention topology remains relatively stable across generation steps.

---

## ZBhZT307xx

- GT: Withdrawn (treated as Reject) (avg 3.0)
- Predicted: N/A (4.6/10)
- Match: N/A

### Final Review

## Summary
This paper systematically evaluates rule-based and model-based verifiers in the Reinforcement Learning with Verifiable Rewards (RLVR) pipeline for mathematical reasoning. The authors demonstrate that rule-based verifiers suffer from significant false negative rates that worsen with stronger policy models, while model-based verifiers achieve higher static accuracy but are highly susceptible to reward hacking during dynamic RL training. Through a hybrid verifier design and adversarial probing, the study reveals a stark disconnect between static verification metrics and dynamic training robustness.

## Strengths
- **Timely and high-impact problem formulation:** The paper directly addresses a critical, underexplored bottleneck in modern RLVR pipelines (e.g., DeepSeek-R1, SimpleRL-Zoo): verifier reliability as a reward signal. The motivation is tightly coupled to current community practices and clearly justifies why static evaluation metrics fail to predict dynamic RL stability.
- **Rigorous empirical progression and practical design:** The methodological pipeline logically advances from static classification evaluation to hybrid RL integration and culminates in an adversarial probing study. The hybrid verifier architecture (rule-based filter + model-based fallback) is computationally pragmatic and yields a verifiable +2.3 absolute point improvement over pure rule-based baselines, demonstrating clear engineering value.
- **Clear diagnostic finding on verifier architecture differences:** The empirical result that discriminative verifiers (e.g., xVerify) maintain robustness under RL pressure while generative CoT verifiers collapse under simple syntactic exploits provides a highly actionable diagnostic for practitioners designing reward systems.

## Weaknesses
- **Statistical fragility of core RL claims:** All dynamic training curves, the reported +2.3 improvement, and the training collapse phenomenon rely on single-run experiments due to computational constraints. Without variance reporting across at least 3 random seeds, it is impossible to distinguish genuine reward hacking dynamics from stochastic training instability, hyperparameter sensitivity, or dataset sampling variance. In the context of ICLR's empirical standards, this severely undermines the reproducibility and confidence of the paper's central RL findings.
- **Confounded experimental design regarding verifier prompting:** The static and RL evaluations apply different prompting strategies: untrained verifiers receive a simplified prompt (ground truth + extracted answer), while trained verifiers receive the full original question context (Appendix E). This uncontrolled confounder makes it impossible to isolate whether the observed "accuracy-RL mismatch" and heightened hacking susceptibility stem from the fine-tuning process itself or from prompt-induced context leakage and instruction-following degradation. The causal link between fine-tuning and vulnerability is therefore unsupported.
- **Superficial adversarial probing scope:** The identified hacking patterns are exclusively syntactic and formatting-based (e.g., empty braces, gibberish strings, HTML tags, markdown injection). While effective against current generative judges, they fail to stress-test *semantic* or *reasoning* vulnerabilities, such as plausible but mathematically flawed reasoning traces, step-skipping, or numerically close incorrect answers. This limits the depth of the robustness analysis and leaves open whether stronger policy models would exploit structural reasoning flaws rather than trivial token artifacts.
- **Diagnostic rather than prescriptive contribution:** The paper thoroughly documents failure modes but provides zero mechanistic explanation for why static fine-tuning paradoxically increases RL susceptibility (e.g., decision boundary compression, overconfidence on out-of-distribution tokens, or reward landscape smoothing). Furthermore, it omits testing of standard RLVR stabilization baselines (e.g., non-zero KL regularization, DAPO, or response formatting constraints). Without these, the claim that model-based verifiers are inherently vulnerable remains unproven; the observed collapse may simply be an artifact of an unregularized training setup.

## Nice-to-Haves
- Release the constructed adversarial pattern suite as a standardized benchmark with automated evaluation scripts to facilitate reproducible robustness testing across future verifier architectures.

## Novel Insights
The paper crystallizes a critical paradox in modern RLVR: optimizing verifiers for higher static classification accuracy can actively degrade their reliability as dynamic reward signals. Specifically, fine-tuning generative CoT verifiers to reduce overthinking and improve recall inadvertently compresses their rejection boundaries for out-of-distribution or malformed inputs, making them highly susceptible to trivial syntactic exploits. This reveals that static equivalence judgments lack any grounding in the adversarial optimization pressures inherent to RL, fundamentally challenging the community's current reliance on offline accuracy metrics as proxies for training stability.

## Potentially Missed Related Work
- **Baker et al. (2025)** is already cited, but work on *process-based reward models* (e.g., Lightman et al., 2023; Uesato et al., 2022) and *adversarial robustness of LLM judges* (e.g., Liu et al., 2023 on hallucination-based judge manipulation) should be contextualized to frame syntactic hacking within the broader literature on LLM evaluation vulnerabilities.
- Studies on *reward hacking in alignment RLHF/RLAIF* (e.g., Denison et al., 2024) could provide theoretical grounding for why zero-KL, unconstrained optimization accelerates verifier exploitation.

## Suggestions
- **Decouple prompt conditioning from architecture testing:** Re-evaluate both trained and untrained verifiers under identical prompting conditions. Without this controlled ablation, the attribution of RL instability to fine-tuning remains speculative.
- **Expand adversarial patterns to semantic/mathematical exploits:** Construct and test reasoning-level attacks (e.g., correct final answer derived from flawed steps, plausible but incorrect intermediate derivations) to determine if generative verifiers are vulnerable to deeper logical deception or merely surface-level token manipulation.
- **Introduce a baseline RL mitigation ablation:** Run the same GRPO pipeline with standard stabilizers (e.g., KL coefficient > 0, reward clipping, or explicit output format constraints via regex). Demonstrate whether the observed reward hacking is an inherent verifier flaw or a correctable artifact of unregularized optimization. This would transform the paper from a descriptive audit to a prescriptive guide.

---

## khHNHzRjMy

- GT: Reject (avg 3.0)
- Predicted: N/A (2.8/10)
- Match: N/A

### Final Review

## Summary
This paper introduces EmoSign, a dataset of 200 ASL video clips annotated for sentiment, emotion intensity, and qualitative emotional cues by Deaf native signers. The authors benchmark four multimodal LLMs across caption-only, video-only, and combined conditions, reporting heavy text reliance, systematic fallback biases toward positive/neutral labels, and poor visual grounding for emotional reasoning.

## Strengths
- **Expert-driven, culturally grounded annotation:** The use of Deaf native ASL signers with professional interpretation experience directly addresses a known failure mode in prior work (e.g., FePh), where hearing annotators systematically misinterpret non-manual markers. The inclusion of open-ended cue descriptions provides rare, linguistically informed insight into how emotion is spatially and kinematically encoded in ASL.
- **Diagnostic modality ablation revealing architectural failure modes:** The clean video/caption/video+caption setup systematically exposes that SOTA multimodal LLMs do not perform genuine visual affective reasoning on sign language; instead, they collapse heuristically to text-driven priors or safe neutral/positive defaults when linguistic scaffolding is removed. This finding is highly relevant for multimodal alignment research.

## Weaknesses
- **Dataset construction directly confounds the core multimodal claim:** Selecting the 100 most positive and 100 most negative clips via VADER on English captions creates an artificially bimodal distribution with high text-label correlation. Consequently, strong caption-only performance does not reveal model capability, and weaker video-only performance cannot be confidently attributed to a failure of visual integration rather than an expected consequence of pre-selecting text-predictive samples. The benchmark lacks text-label mismatch or purely visually expressive clips to validate the claimed visual reasoning deficit.
- **Unreliable labels for critical negative emotions:** Inter-annotator agreement is moderate overall (α=0.593) and drops to poor levels for negative emotions (α=0.11–0.55). Comparing these scores to Fleiss’ κ on entirely different datasets (MELD/IEMOCAP) is methodologically invalid. Low agreement on high-impact labels (anger, frustration, fear) means model errors on these classes cannot be distinguished from annotation noise, severely undermining the dataset's utility as a rigorous benchmark for accessibility-critical systems.
- **Insufficient empirical rigor and narrow baseline scope:** The paper lacks standardized train/val/test splits, confidence intervals, or statistical significance testing, which is unacceptable for N=200 benchmarking where variance heavily dictates reported rankings. Furthermore, evaluating only zero-shot MLLM prompts without any fine-tuned ablations, vision-centric baselines (e.g., kinematic/pose features, optical flow classifiers), or quantitative grounding metrics restricts the empirical contribution to a superficial prompt-probing exercise rather than a foundational benchmark paper.
- **Anecdotal grounding evaluation:** The emotion cue grounding analysis relies on manual inspection of "several randomly selected videos" without quantitative overlap metrics, attention map validation, or systematic comparison to annotator-identified frames. This renders the grounding claims entirely qualitative and impossible to track or improve in future iterations.

## Nice-to-Haves
- Consistent prompting pipelines or strict automated output parsing across all models to eliminate confounding variables in cross-architecture comparisons.
- Explicit analysis of clips where grammatical non-manual markers (e.g., raised eyebrows for questions) co-occur with or contradict emotional cues, directly testing the disentanglement challenge highlighted in the motivation.
- Reporting inference latency, parameter counts, and API/compute trade-offs to contextualize the practical feasibility of deploying such models for real-time accessibility applications.

## Novel Insights
The paper compellingly demonstrates that state-of-the-art multimodal LLMs do not comprehend visual affect in sign language but instead construct post-hoc textual rationalizations anchored by captions. When forced into video-only conditions, these models abandon nuanced affective reasoning and default to positive/neutral safety priors, exposing a fundamental misalignment in vision-language pretraining: current architectures are optimized for spoken-language prosody/text synchronization and fail entirely on visual-manual languages where linguistic and affective signals are spatially and temporally co-articulated. This reveals that strong multimodal performance on conventional benchmarks is likely driven by lexical shortcuts rather than genuine spatiotemporal visual reasoning.

## Potentially Missed Related Work
- None identified. The paper adequately covers prior sign language translation datasets, FePh, and contemporary multimodal emotion benchmarks.

## Suggestions
- **Decouple text and visual signals:** Create a controlled ablation subset where emotional keywords in captions are masked, neutralized, or sentiment-flipped. This is necessary to prove that model failures in the video condition stem from poor visual integration rather than the expected removal of strong lexical cues.
- **Establish formal benchmarking rigor:** Define explicit train/validation/test splits and report bootstrapped confidence intervals or significance tests across all tables. Without statistical quantification, performance gaps between modalities and models remain indistinguishable from sampling variance on a 200-clip dataset.
- **Replace qualitative grounding with quantitative evaluation:** Have annotators label relevant temporal segments or spatial bounding boxes for emotion cues, then compute temporal IoU or attention overlap between model outputs and ground truth. Supplement this with at least one fine-tuned or classical vision baseline (e.g., optical flow + classifier, or LoRA-adapted MLLM) to isolate zero-shot prompting limitations from fundamental architectural gaps.

---

## v05SW2X3IC

- GT: Accept (Poster) (avg 5.3)
- Predicted: N/A (3.6/10)
- Match: N/A

### Final Review

## Summary
This paper proposes a learnable three-channel codec grounded in the Gray-Wyner network, designed to disentangle common and task-specific information for efficient multi-task machine inference. It derives theoretical bounds linking Wyner’s and Gács-Körner lossy common information via interaction information and introduces a tunable Lagrangian parameter $\beta$ to optimize the tradeoff between transmit and receive rates. Empirical evaluation on synthetic data, controlled MNIST variants, and standard vision benchmarks demonstrates rate reductions over independent coding baselines.

## Strengths
- **Principled theoretical formulation:** The paper provides a clear, mathematically rigorous mapping of the classical Gray-Wyner rate region to a differentiable neural optimization objective. The derivation of the $\beta$-parameterized tradeoff (Theorem 2) offers a structured way to think about rate allocation in multi-terminal machine vision.
- **Controlled validation on synthetic data:** The synthetic experiments successfully demonstrate that the proposed architecture respects theoretical mutual information bounds, and the MNIST PMF analysis cleanly illustrates how the model adapts to fully dependent and fully independent task distributions.
- **High reproducibility standards:** The inclusion of complete code, explicit architectural diagrams, and granular training hyperparameters aligns with rigorous empirical standards and facilitates direct replication of the reported rate-distortion curves.

## Weaknesses
- **Collapse of the core $\beta$ tradeoff on real vision tasks:** The paper's central claim is that the proposed objective empirically explores the transmit-receive tradeoff via $\beta \in (1, 2)$. However, the auxiliary matching loss (Eq. 15) destabilizes common channel utilization outside $\beta=1$, forcing the authors to fix $\beta=1$ (and even override to $0.1$ for MNIST) for all real-data experiments (Appendix D.5). This leaves the primary theoretical mechanism entirely unvalidated on the datasets that matter most for the claimed application, creating a significant theory-practice disconnect.
- **Unverified theoretical bounds and information-theoretic assumptions:** Theorem 1 bounds lossy common information using interaction information under the strong assumption that learned latents contain no "nuisance" or redundant private information (type a.2/b.2 terms). Crucially, the paper never measures interaction information, mutual information, or task-relevant information in the actual discrete latents $Y_0, Y_1, Y_2$. Without empirical quantification, the theoretical bounds remain mathematically isolated from the neural implementation.
- **Limited experimental scope and unfair baseline comparisons:** The evaluation freezes all task-specific decoders and adds a pixel-reconstruction loss, effectively reducing the framework to source compression for fixed models rather than representation learning. More critically, the paper does not match parameter counts, FLOPs, or latency against baselines, nor does it compare against modern multi-task learned codecs (e.g., Guo et al., 2024; Chamain et al., 2021). It remains unclear whether the observed rate savings stem from principled Gray-Wyner routing or simply higher model capacity and gradient stabilization from the auxiliary loss.

## Nice-to-Haves
- Visualizing channel-specific salience or attention maps to qualitatively verify whether $Y_0$, $Y_1$, and $Y_2$ actually attend to semantically distinct regions or encode overlapping features.
- Reporting encoding/decoding latency metrics, particularly given the multi-channel masked auto-regressive entropy models which introduce sequential bottlenecks.
- Clarifying the "six vision benchmarks" phrasing in the abstract to accurately reflect the dataset/task combinations actually used.

## Novel Insights
The work effectively reframes multi-task distributed inference through the lens of the Gray-Wyner achievable region, explicitly surfacing the often-overlooked tension between optimizing for joint transmission ($R_t$) versus independent retrieval ($R_r$). By parameterizing this tradeoff and introducing a branch-compatibility mechanism, the paper reveals a critical practical constraint: strict mathematical isolation of common information in high-dimensional continuous latents is highly sensitive to auxiliary alignment losses. This observation underscores that theoretical disentanglement in neural compression requires softer, statistical matching priors rather than rigid MSE constraints, offering a valuable cautionary insight for future multi-branch representation architectures.

## Potentially Missed Related Work
- **Empirical mutual information / interaction information estimators in deep networks:** Methods like MINE or CLUB could bridge the gap between Theorem 1 and the learned latents, allowing empirical verification of the proven bounds.
- **Dynamic routing and mixture-of-experts for multi-task compression:** Recent works on adaptive bitrate allocation or conditional feature routing share the goal of efficient multi-task representation but approach the problem without classical IT constraints.
- **Information bottleneck approaches to multi-task learning:** Variants that explicitly optimize for shared vs. private information under distortion constraints (e.g., Variational Information Bottleneck for multitask settings) provide complementary perspectives on the disentanglement objective.

## Suggestions
1. **Decouple the auxiliary loss from the rate tradeoff:** Replace the rigid element-wise MSE matching term (Eq. 15) with a softer statistical alignment constraint (e.g., correlation penalty, contrastive loss, or optimal transport) to enable stable optimization across the full $\beta \in [1, 2]$ range. Empirically demonstrating the transmit-receive curve on vision tasks is essential to support the paper's core claim.
2. **Ensure capacity-matched baselines and include modern competitors:** Report parameter counts, memory footprint, and inference FLOPs. Add comparisons to recent multi-task learnable codecs to establish whether the Gray-Wyner routing offers genuine advantages beyond naive Independent/Joint splits.
3. **Empirically validate the information bottleneck:** Use linear probing or neural MI estimators to quantify the actual task-relevant information contained in $Y_0$ vs. $Y_1/Y_2$. Demonstrating that $Y_0$ genuinely captures shared semantics rather than redundant pixel artifacts would solidify the disentanglement claim.
4. **Jointly optimize task decoders:** Remove the frozen-head constraint or include a variant with trainable synthesis/task heads to demonstrate true representation learning capabilities rather than compression performance dependent on pixel-reconstruction regularization.

---

## JEN4nsDgh9

- GT: Reject (avg 3.5)
- Predicted: N/A (3.8/10)
- Match: N/A

### Final Review

## Summary
This paper introduces a benchmark for evaluating zero-shot text-to-image (T2I) models on generating visual depictions for WordNet taxonomy concepts. It proposes nine evaluation metrics spanning preference-based rankings (human and GPT-4), hierarchy-aware CLIP similarities derived from information-theoretic formulations, and standard distributional scores. Across 12 open-source models and multiple concept subsets, the study finds that T2I model rankings diverge significantly from standard vision benchmarks, with FLUX and Playground-v2 consistently outperforming, and provides a publicly released dataset of generated images for structural lexical resources.

## Strengths
- **Comprehensive, Multi-Dimensional Evaluation Protocol:** The benchmark thoughtfully spans preference judgments, hierarchical semantic alignment, and generation quality across diverse concept partitions (common, rare/abstract, and LLM-predicted). The strong empirical correlation ($\rho \approx 0.91$) between the proposed hierarchy-aware metrics and human annotator rankings demonstrates their practical utility as diagnostic signals for taxonomy visualization.
- **High-Quality Empirical Resource & Open Release:** The systematic testing of 12 models reveals actionable insights for model selection, notably that standard T2I benchmarks poorly proxy performance on structured lexical tasks. The commitment to releasing the full prompt set, pairwise preference logs, and a complete WordNet-3.0 image generation dataset aligns well with community standards for reproducible, resource-driven research.

## Weaknesses
- **Fundamental Disconnect Between Theoretical Framing and Metric Implementation:** The paper formally derives novel metrics (Lemma, Hypernym, Cohyponym Similarity, Specificity) from probabilistic definitions grounded in KL Divergence and Mutual Information, complete with theorems and proofs in Appendix D. However, Section 4.2 explicitly states that these probability densities are approximated using raw CLIP cosine similarity: $P(X=x|v) \approx \text{sim}(C(v), C(x))$. Cosine similarity is a bounded geometric measure, not a valid probability distribution over continuous image space. This substitution invalidates the mathematical guarantees of the derivations. The claimed theoretical grounding is therefore heuristic, and presenting the theorems as formally applicable to the implemented metrics misrepresents the methodological rigor expected at ICLR.
- **Unmitigated Systemic Bias in LLM-as-a-Judge Evaluation:** The pairwise GPT-4 evaluation relies solely on random prompt assignment and assumes the Bradley-Terry model will inherently compensate for a documented, strong positional bias (Fig. 5 & 12). This contradicts established best practices for robust LLM judging, which require position swapping, output anonymization, and majority voting to ensure reliable latent strength estimation. Given the explicit finding that raw pairwise battle correlations are near zero, the resulting GPT-4 ELO rankings and downstream superiority claims remain statistically precarious without explicit bias mitigation.
- **Unvalidated Reliance on Synthetic Taxonomy Nodes Undermines "Automation" Claims:** A core motivation is assessing readiness for "automating taxonomy enrichment." Yet, the evaluation on the "Predicted" subset uses LLM-generated concepts and GPT-4-synthesized definitions without any factual validation or confidence scoring. Hallucinated, structurally invalid, or linguistically malformed prompts will systematically penalize image generators, conflating upstream linguistic errors with downstream visual synthesis failures. The absence of a human audit or filtering protocol for this subset severely undermines the validity of conclusions drawn regarding automated taxonomy curation.

## Nice-to-Haves
- Replace or augment the naive top-1 Wikimedia retrieval baseline with a modern semantic retrieval baseline (e.g., CLIP-based cross-modal search) to more rigorously validate the claim that generation outperforms retrieval for taxonomy expansion.
- Provide quantitative failure breakdowns (e.g., rates of parent-concept bleed, literal text injection, abstract hallucination) rather than relying solely on qualitative error grids in the appendix.
- Clarify experimental protocol specifics: exact number of pairwise battles per model pair, seed variance reporting, and compute budget, to ensure full reproducibility.

## Novel Insights
The work surfaces a critical dichotomy between standard T2I optimization objectives and lexical hierarchy preservation. Models fine-tuned for aesthetic fidelity or fast inference (e.g., distilled architectures) often excel on CLIP-based alignment metrics but fail to capture the fine-grained conceptual distinctness prioritized by human evaluators. This demonstrates that visual taxonomy generation rewards semantic specificity and hierarchical coherence over photorealism, challenging the assumption that state-of-the-art aesthetic generators will naturally generalize to structured knowledge graph visualization tasks.

## Potentially Missed Related Work
- **Concept-level Benchmarks (e.g., ConceptBed/ICCV 2024, Liao et al. AAAI 2024):** These works evaluate text-conditioned concept learning and abstract T2I generation. Explicitly contrasting taxonomy-aware hierarchical metrics against standard concept-level evaluation protocols would better contextualize the paper's methodological novelty.
- **Reference-Free Grounding Metrics (e.g., TIFA, VQAScore, PickScore):** Modern VLM-based evaluation metrics that verify visual-textual alignment via question-answering could serve as a more robust, interpretable alternative to raw CLIP similarities for validating taxonomy concept depiction.

## Suggestions
1. **Reframe Metric Formulation & Theoretical Claims:** Explicitly state that Lemma, Hypernym, Cohyponym, and Specificity scores are *empirical heuristics inspired by* KL divergence and mutual information. Remove or heavily caveat Appendix D's theorems to reflect that cosine similarity does not satisfy the probabilistic axioms required for the mathematical guarantees. Report these as validated empirical proxies rather than theoretically grounded probabilities.
2. **Harden the LLM Judge Protocol:** Implement standard debiasing for the GPT-4 evaluator: enforce strict left/right position swapping across multiple calls, anonymize model names/outputs in the prompt, and compute Krippendorff's $\alpha$ alongside Spearman correlation. Only publish ELO scores derived from a bias-mitigated evaluation loop.
3. **Audit & Stratify LLM-Predicted Subsets:** Manually validate a statistically significant sample of the TaxoLLaMA-generated concepts and GPT-4 definitions. Filter or stratify the "Predicted" subset into high-confidence vs. low-confidence generations, and report evaluation metrics separately. This isolates true image synthesis capability from upstream linguistic noise and salvages the validity of the taxonomy automation claims.

---

## b6qQmQ2F13

- GT: Accept (Poster) (avg 5.0)
- Predicted: N/A (5.4/10)
- Match: N/A

### Final Review

## Summary
This paper systematically investigates how to optimally allocate a fixed memory budget across model weights, generation length, parallel scaling, and KV cache compression for long-context reasoning LLMs. Through an extensive sweep of ~1,700 configurations across multiple model families and reasoning benchmarks, the authors identify a scale-dependent inflection point (~4.2 GB effective weight size) that partitions deployment strategies: smaller effective models benefit from higher weight precision/capacity, while larger models benefit from extended test-time compute and KV cache management.

## Strengths
- **Comprehensive empirical scope and reproducibility:** The evaluation spans three reasoning model families, four challenging benchmarks, and five critical deployment axes (size, precision, token budget, parallelism, KV compression). The inclusion of explicit memory accounting equations, open-source code, and cross-quantization-backend validation (GPTQ, AWQ, FP8) establishes a strong, reproducible foundation.
- **Actionable, task-dependent precision heuristic:** The paper effectively challenges the industry-wide "4-bit is optimal" dogma by demonstrating that mathematical reasoning and code generation retain accuracy only with higher-weight precision, whereas knowledge-intensive tasks prioritize parameter capacity at 4-bit. This contrast is clearly supported by the Pareto analyses across AIME25/LiveCodeBench versus GPQA-Diamond.

## Weaknesses
- **Statistical fragility of the claimed crossover threshold:** The paper presents deterministic Pareto frontiers and treats the ~8-bit 4B (~4.2 GB) inflection point as a precise decision boundary, yet provides no confidence intervals, error bands, or significance testing across the 32-run averages. Given that frontier points often differ by 1–3% accuracy and benchmark variance is high (especially for AIME25 and LiveCodeBench), the statistical reliability of this threshold is unverified. Without variance-aware plots or bootstrap testing, the claimed strategic shift risks being an artifact of sampling noise rather than a robust principle.
- **Confounded task-difficulty vs. intrinsic precision sensitivity:** The claim that "math/code requires higher precision while knowledge needs lower precision" is plausible but inadequately disentangled from benchmark saturation effects. MATH500 and GPQA saturate quickly for models above certain capacities, naturally flattening the Pareto curve, while AIME25 demands deeper reasoning where extended context or higher precision delays saturation. The analysis does not control for task difficulty or reasoning depth, making it unclear whether precision sensitivity is a fundamental property of task type or merely a byproduct of how quickly accuracy ceilings are reached.
- **Internal threshold inconsistency and budget-forcing artifacts:** Section 4 establishes the memory-optimal shift at ~8-bit 4B, but Section 5 abruptly shifts this to ~8-bit 8B when discussing KV cache eviction vs. quantization. This discrepancy is not reconciled and undermines the claim of a unified, scale-dependent principle. Furthermore, the reliance on prompt-injected budget forcing artificially inflates KV cache growth without guaranteeing proportional gains in logical reasoning depth. On easier benchmarks (notably MATH500), forced continuation introduces non-monotonic accuracy drops, potentially skewing the memory-accuracy Pareto frontier and overestimating the memory penalty of longer generation.
- **Theoretical memory model diverges from production serving dynamics:** The memory accounting assumes idealized additive costs (`M_weights + M_kv`), deliberately ignoring activation overhead, KV fragmentation, vLLM block-table paging, and the compute/latency costs of online methods like R-KV scoring or on-the-fly dequantization. While acknowledged as a scope limitation, relegating latency/throughput to Appendix C.1 creates a disconnect between the paper's "deployment-ready" framing and the reality that memory-optimal byte allocations rarely translate to throughput-optimal serving configurations.

## Nice-to-Haves
- Include a brief qualitative analysis of reasoning traces (e.g., self-correction frequency, step coherence, or calculation error rates) under forced long-context vs. high-precision regimes to verify whether increased KV cache actually translates to deeper logical chains or merely longer, meandering generations.
- Extend the KV compression comparison to include heavy-hitter retention policies (e.g., SnapKV, H2O) or dynamic bitwidth quantization (e.g., KIVI) to test whether the eviction > quantization advantage for small models is robust to algorithmic choices beyond R-KV/StreamingLLM and static HQQ.
- Provide component-level precision ablations (e.g., isolating MLP vs. attention precision) for math/code tasks to pinpoint whether quantization degradation stems from numerical fidelity loss in arithmetic operations or capacity collapse in reasoning pathways.

## Novel Insights
The work reframes test-time scaling from a traditional FLOPs-driven perspective to a strict memory-constrained allocation problem, revealing that "effective size" (parameters × precision) dictates deployment efficiency in a non-linear manner. The key conceptual leap is recognizing that KV cache dominance flips the optimization landscape: for smaller reasoning models, compressing weights to fund longer context is a false economy because the KV cache grows faster than reasoning quality improves, whereas for larger models, the parameter-rich architecture can tolerate longer, redundant generation where test-time compute becomes the primary bottleneck. This scale-dependent dichotomy provides a necessary corrector to scale-agnostic compression heuristics.

## Potentially Missed Related Work
- Kumar et al. (2024), *Scaling laws for precision* — Highly relevant for contextualizing the claimed threshold; provides theoretical grounding for how performance scales with total bit-capacity and could help disentangle precision effects from model size.
- Zhang et al. (2023) *H2O* and Li et al. (2024) *SnapKV* — Important baseline context for the eviction vs. quantization comparison, as heavy-hitter and persistent-importance methods dominate current KV cache compression research.
- Liu et al. (2025a) *FlashRL / 8bit rollouts* — Directly relevant to the latent/throughput appendix, emphasizing how precision impacts sampling speed in reinforcement learning and test-time scaling pipelines.
*(Presented as suggestions for deeper contextualization rather than critiques of novelty.)*

## Suggestions
1. **Reconstruct Pareto frontiers with uncertainty quantification:** Overlay bootstrapped 95% confidence bands or shade regions of statistical indistinguishability. This will immediately clarify whether the ~4.2 GB crossover point is robust or lies within the noise margin of the sweeps.
2. **Clarify the threshold discrepancy across sections:** Explicitly reconcile why the inflection point shifts from ~8-bit 4B in serial scaling to ~8-bit 8B in KV compression. If the threshold is axis-dependent, reframe the contribution as a *multi-axis* scale dependency rather than a single fixed cutoff.
3. **Control for benchmark saturation in precision claims:** Add a difficulty-normalized analysis (e.g., evaluating on stratified subsets or plotting pass@k vs. effective bits) to isolate true task-type sensitivity from early saturation artifacts. Alternatively, soften the conclusion to acknowledge that observed precision preferences are entangled with benchmark difficulty ceilings.
4. **Integrate a lightweight serving benchmark:** Replace theoretical memory estimates for at least one parallel scaling/KV compression configuration with peak VRAM + wall-clock measurements from vLLM. Quantifying the gap between theoretical byte allocation and actual throughput will ground the "memory-optimal" claims in realistic deployment constraints.

---

## PFhrOUJZ5o

- GT: Reject (avg 5.0)
- Predicted: N/A (4.4/10)
- Match: N/A

### Final Review

## Summary
This paper introduces LAION-Comp, a large-scale dataset of 540K+ high-aesthetic images annotated with structured scene graphs, positioning explicit relational conditioning as the solution to compositional bottlenecks in text-to-image modeling. The authors integrate a lightweight GNN-based scene graph encoder into modern diffusion and flow-matching backbones, and establish CompSGen Bench to systematically evaluate complex scene generation. Empirical results show consistent gains in structural alignment metrics over prompt-only baselines and earlier scene-graph methods, complemented by a secondary demonstration of structure-guided image editing.

## Strengths
- **High-Value Data-Centric Resource:** The construction of a half-million-scale, open-vocabulary scene graph dataset addresses a well-documented gap between small, manually annotated SG corpora (VG, COCO) and unstructured web-scale text data. Rigorous statistical characterization and human verification (>95% accuracy across objects, attributes, and relations) establish the dataset as a credible foundation for compositional research.
- **Efficient Backwards-Compatible Integration & Scaling Evidence:** The proposed GNN encoder adds minimal overhead (~14.7M parameters, <3% latency increase) while consistently improving SG-IoU, Entity-IoU, and Relation-IoU across SDXL, SD3.5, and FLUX backbones. The dataset-scale ablation (Table 4) convincingly demonstrates monotonic performance gains as sample size increases, decoupling data quantity from architectural modifications.

## Weaknesses
- **Circular Evaluation Pipeline Undermines Accuracy Claims:** The primary compositional metrics rely on extracting triples from generated images using GPT-4/GPT-4o and computing textual IoU against ground truth—the same VLM family used to generate the initial dataset annotations. This creates a self-referential evaluation loop where the model may simply learn to mimic the annotation LLM's semantic priors rather than achieve true visual grounding. Without vision-based, deterministic grounding metrics (e.g., detector-level bounding box overlap or segmentation IoU), the reported compositional superiority remains statistically unverified and highly susceptible to VLM hallucination bias.
- **Incomplete Benchmarking Against Contemporary Compositional Controls:** The quantitative evaluation omits modern layout-controlled and attention-guidance methods (e.g., GLIGEN, BoxDiff, RealCompo, MIGC++) that are explicitly cited in the related work but absent from results tables. By comparing primarily against vanilla backbones and older SG2IM models, the paper fails to isolate whether gains stem from the novel dataset or from standard fine-tuning on any structured signal. This omission prevents validation of the core claim that explicit structural data fundamentally outperforms state-of-the-art architectural composition controls.
- **Unverified Architectural Inductive Bias & Noise Robustness:** The paper attributes performance gains to a "designed scene graph encoder" but provides no ablation comparing the GNN to simpler, permutation-invariant aggregation (e.g., mean/max pooling of CLIP-encoded triples). If flat pooling yields similar results, the graph architecture is an unnecessary bottleneck. Additionally, while the authors acknowledge ~4% automated annotation errors, there is no systematic analysis of how specific noise types (relation hallucinations, occluded objects, color misattributions) propagate into generation failures, leaving the dataset's robustness and the method's scaling behavior unproven.

## Nice-to-Haves
- Elevate the SG-based editing framework from the appendix to the main text with clearer methodology, locality ablation, and precise terminology (clarifying that it is training-free *only* with respect to the already fine-tuned backbone).
- Report training compute budgets (GPU hours/FLOPs) for structural fine-tuning versus standard text-only fine-tuning to properly contextualize scalability claims.
- Provide stratified performance breakdowns by scene graph complexity (node/edge count) to demonstrate whether the model gracefully degrades on denser scenes or hits a structural ceiling.

## Novel Insights
The work effectively pivots the compositional generation paradigm from purely architectural interventions (attention manipulation, bounding-box grounding) to a data-centric foundation, demonstrating that relation-dense, structurally explicit supervision can resolve ambiguities that sequential text prompts inherently fail to capture. The finding that modern flow-matching and diffusion backbones absorb structural embeddings without architectural bottlenecks suggests the primary constraint in complex synthesis is not model capacity, but the absence of high-fidelity, relation-aware training signals at web scale. This reinforces a broader shift in generative AI: scaling structured supervision may yield more robust compositional control than increasingly complex inference-time routing.

## Potentially Missed Related Work
- **LayoutGPT / LLM-Grounded Diffusion:** While cited in passing, recent frameworks that leverage LLMs for dynamic layout planning and box-level conditioning during inference offer a direct alternative to dataset-level structural annotation. Benchmarking against these methods would clarify the practical trade-offs between upfront annotation cost and inference-time computational overhead for achieving compositional fidelity.

## Suggestions
- **Decouple Evaluation from the Annotation VLM:** Replace or supplement the LLM-based IoU metrics with vision-centric grounding evaluations (e.g., GroundingDINO + SAM for per-object spatial IoU, or frozen detection model confidence scores) to objectively verify whether generated regions align with ground-truth structures independent of VLM semantic biases.
- **Run a GNN vs. Flat Aggregation Ablation:** Train variants where the GNN is replaced with simple concatenation and mean/max pooling of CLIP-encoded node/edge embeddings. This will definitively prove whether relational message-passing provides a measurable inductive advantage over flat sequence representations.
- **Include Direct Comparisons to Modern Compositional Baselines:** Add quantitative results for 2-3 contemporary structural control methods (e.g., GLIGEN, RealCompo) on CompSGen Bench to establish whether the LAION-Comp data advantage holds against current architectural SOTA, or if the gains are specific to the chosen baselines.

---

## cEXEmyW77N

- GT: Accept (Poster) (avg 5.0)
- Predicted: N/A (4.3/10)
- Match: N/A

### Final Review

## Summary
This paper investigates whether LLM-generated bibliographies are distinguishable from human-authored reference lists by contrasting citation graph topology with semantic text embeddings across ~10,000 focal papers. The authors demonstrate that structure-only features fail to separate LLM graphs from ground truth (~60% accuracy, near chance), whereas high-dimensional title/abstract embeddings enable robust detection via Random Forests (~83–84%) and GNNs (~93%). The study concludes that modern LLMs accurately replicate human citation topology but leave a detectable semantic fingerprint, arguing that integrity audits must prioritize content signals over structural diagnostics.

## Strengths
- **Rigorous, multi-tier baseline design:** The construction of field-matched, subfield-matched, and temporally constrained random baselines effectively controls for macro-level bibliometric statistics. This cleanly isolates the topological realism of LLM graphs from trivial degree or topic distributions, providing a highly credible null model.
- **Systematic elimination of trivial artifacts:** Extensive controls, including i.i.d. random vector replacements, PCA-k dimensionality ablations, and cross-generator/encoder generalization experiments, robustly rule out high-dimensionality bias and confirm that separability stems from structured semantic content rather than noise or dataset leakage artifacts.
- **Transparent evaluation protocol:** The authors avoid cherry-picking by reporting full accuracy distributions across 500 hyperparameter configurations, employing stratified splits with fixed seeds, and conducting saturation analysis. This methodological transparency sets a strong standard for graph ML evaluation in scholarly contexts.

## Weaknesses
- **Methodologically vacuous GNN node features and missing MLP baseline:** Section 6 assigns a constant global edge count to every node as a structural feature. This provides zero discriminative signal during message-passing and misrepresents what the GNNs are actually learning. Furthermore, the omission of a simple MLP baseline on the aggregated embeddings strongly suggests that the GNN’s accuracy gains derive entirely from the node embeddings, with structural message-passing contributing negligibly. This undermines the claimed utility of graph neural architectures for this task and overcomplicates the results.
- **Unjustified embedding aggregation strategy:** Summing 3072-dimensional node embeddings to graph-level vectors conflates semantic direction with list magnitude. Although the authors subsample graphs to match sizes, summation remains highly sensitive to minor cardinality variations and semantic density. Without ablation against standard pooling methods (mean, max, or attention), it remains unclear whether detection performance reflects genuine semantic divergence or aggregation-induced magnitude shifts.
- **Black-box "semantic fingerprint" limits mechanistic insight:** The paper convincingly establishes that embeddings drive separability but entirely fails to probe what the semantic fingerprint represents. Without feature attribution (e.g., SHAP on RF splits, probing classifiers, or correlation with metadata like recency, citation count, or methodological focus), the findings remain purely descriptive. This leaves the core scientific question—*how* LLM reference selection behavior actually deviates from human practices—unanswered, limiting the paper's impact beyond applied detection.
- **Overlooked data overlap and restricted temporal scope:** The train/test split groups focal papers with their matched random graphs but does not enforce node- or component-disjoint splits across the dataset. Given that citation networks share highly cited seminal references, semantic and structural overlap across splits likely inflates generalization estimates. Additionally, restricting focal papers to 1999–2021 omits modern publication dynamics and recent LLM training cutoffs, raising serious doubts about the temporal generalizability of the reported semantic fingerprint.

## Nice-to-Haves
- Report classification performance stratified by the 19 MAG top-level fields to verify whether the semantic fingerprint is consistent across disciplines or driven by domains with sparser training corpora.
- Include a simple cosine-similarity baseline (e.g., mean focal-reference alignment and reference-reference variance) to establish a stronger lower bound and contextualize the marginal utility of 3072-D embeddings over basic topical alignment metrics.
- Evaluate the detector's robustness when LLMs are explicitly instructed to mimic human citation patterns (e.g., enforcing recency/topic balance) to determine if the fingerprint is an inherent selection bias or merely an artifact of default prompting.

## Novel Insights
The paper delivers a sharp empirical dichotomy: contemporary LLMs have effectively internalized the macroscopic *geometry* of scientific citation networks, accurately reproducing degree distributions, local clustering, and global sparsity constraints, yet they consistently fail to replicate the latent *semantic cohesion* that organizes human bibliographies. This structural mimicry paired with semantic divergence indicates that LLMs operate as highly capable statistical graph generators that capture citation connectivity patterns but misalign on the underlying rationale for those connections. Consequently, the work fundamentally shifts the paradigm for research integrity: reliable auditing cannot rely on graph topology, as synthetic provenance is effectively invisible to structural diagnostics, and must instead target the subtle, embedding-discernible selection biases that govern how models populate scholarly discourse.

## Potentially Missed Related Work
- **Systematic studies on citation recommendation system biases** (e.g., demographic, geographic, and venue skews in algorithmic suggestion engines) — highly relevant for contextualizing how LLM-driven reference selection might computationally compound existing scholarly visibility inequalities.
- **Benchmarking works on Retrieval-Augmented Generation (RAG) for literature review** — essential for delineating the boundary between parametric knowledge biases (studied here) and retrieval-augmented citation behaviors that dominate real-world research workflows.

## Suggestions
- Replace the constant global edge-count feature with meaningful node-level structural attributes or remove structural node features entirely to transparently isolate the contribution of embeddings vs. topology. Add a standard MLP baseline trained on the same aggregated embeddings to quantitatively prove whether message-passing provides any measurable gain over simple feature aggregation.
- Replace sum pooling with mean/max/attention pooling and report a brief ablation. Justify the choice explicitly and demonstrate that separability persists regardless of aggregation method to eliminate magnitude confounding.
- Implement feature attribution or embedding probing to map the "semantic fingerprint" onto interpretable bibliometric axes (e.g., recency tilt, prestige weighting, methodological vs. theoretical framing). This will transform the detector from a black-box auditor into a diagnostic tool that reveals *how* LLMs systematically bias scientific discovery.

---

## hQZQVLJrH9

- GT: Withdrawn (treated as Reject) (avg 4.5)
- Predicted: N/A (3.7/10)
- Match: N/A

### Final Review

## Summary
This paper establishes a first-order mathematical equivalence between activation steering and training-data influence functions by framing both as dual projections within a shared sensitivity geometry. The authors introduce Influence-Aligned Steering (IAS), a computationally cheap alignment diagnostic $\omega(x)$ to predict steering feasibility, alongside spectral optimality criteria and low-rank generalization bounds. The framework is validated with first-order fidelity checks, a layer-alignment ablation on GPT-2 Medium, and a spectral direction test on ResNet-50.

## Strengths
- **Conceptual Unification:** The primal-dual projection framework elegantly bridges two historically disjoint interpretability paradigms (activation engineering and data attribution), providing a unified sensitivity lens that clarifies when and why steering can or cannot replicate data-level interventions.
- **Actionable Feasibility Diagnostic:** The alignment cosine $\omega(x)$ (Thms. 5.1 & 6.2) provides a theoretically grounded, low-compute pre-check (~2 JVP/VJP passes) that reliably predicts steering viability. The empirical demonstration that $\omega(x)$ increases monotonically with network depth directly supports practical layer-selection heuristics.
- **Empirical Confirmation of Linear Regime:** Figure 1 validates the core theoretical premise with a ~0.978 cosine similarity between predicted and realized logit shifts, confirming that the first-order approximation holds tightly for infinitesimal perturbations as claimed.

## Weaknesses
- **Non-Rigorous Proof of the Central Theorem:** Theorem 4.2 (Steering-Influence Equivalence) is the paper's keystone claim, yet it is supported only by a hand-wavy "Idea of the proof" that merely restates the construction rather than deriving it. For a theory-heavy venue like ICLR, the core equivalence theorem must include a complete, rigorous mathematical derivation (e.g., explicit construction of the mapping, norm bounds, and residual characterization), not just intuitive sketches.
- **Unrealistic Assumption & Ignored Influence Fragility:** Corollary 1 relies on the affine independence of influence vectors across the training set, which is practically false for modern, highly correlated datasets (e.g., web-scraped LLM corpora). The paper offers no regularization scheme or discussion of how collinearity destabilizes the $\ell_1$-minimal measure. Furthermore, while Basu et al. (2021) is cited, the paper completely ignores the well-documented fragility of influence functions in non-convex, over-parameterized regimes; if influence scores are noisy or uncorrelated with true retraining effects, the IAS vector inherits and amplifies this noise, a fundamental limitation left unaddressed.
- **Critical Empirical Omission Despite Bold Claims:** The abstract and introduction explicitly promise a workflow that maps "undesired behaviors back to causal training examples." However, there is **zero experimental validation** of this claim. The paper lacks any data-removal, tracing, or case-study experiment demonstrating that the constructed measure $\rho_{\mathbf{s}}$ actually identifies influential data better than baselines or yields meaningful debugging signals. Additionally, detoxification results (Table 1) lack error bars or statistical testing across seeds, and the perplexity values (~13k) appear to suffer from a reporting or unit mismatch that obscures true downstream impact.

## Nice-to-Haves
- Include a structured pseudocode block detailing the exact computation of IAS, $\omega(x)$ estimation, and the power-iteration spectral recipe to eliminate ambiguities in the computational recipe.
- Provide an ablation on the Tikhonov damping parameter $\phi$ and practical Hessian surrogates (e.g., KFAC, randomized SVD) to demonstrate how approximation errors affect alignment diagnostics and IAS stability in deep, ill-conditioned layers.
- Clarify the systematic slope deviation of ~1.50 (rather than the ideal 1.0) in Figure 1; while directionality aligns (cosine 0.978), the scaling mismatch hints at unmodeled second-order accumulation or Hessian-damping effects that warrant a brief discussion.

## Novel Insights
The paper successfully reframes activation steering and data influence not as competing interventions, but as restricted projections onto different subspaces of the same output sensitivity manifold. This geometric unification yields a principled, computationally trivial certificate ($\omega(x)$) that quantifies the intrinsic "steerability" of a model for a given input, fundamentally shifting model editing from heuristic vector tuning to a provably bounded optimization over alignment geometry.

## Potentially Missed Related Work
- **Basu et al. (2021) / Pruthi et al. (2020):** Cited but critically under-integrated; a dedicated discussion on how gradient correlation, Hessian misspecification, and dataset redundancy degrade influence approximations is necessary to ground IAS in realistic attribution bounds.
- **TrACe / RELATIF (Ghorbani & Zou, 2019; Barshan et al., 2020):** Relevant for contextualizing how existing methods handle correlated data and approximate leave-one-out effects, which directly informs the limitations of the affine independence assumption in Corollary 1.

## Suggestions
1. **Rigorize Theorem 4.2 and Relax Corollary 1:** Provide a formal, step-by-step proof of the steering-influence mapping in an appendix. Replace or weaken the affine independence requirement by analyzing the $\ell_1$-solution landscape under collinearity, or introduce a concrete regularization strategy (e.g., elastic net or thresholding) that stabilizes $\rho_{\mathbf{s}}$ on real-world correlated datasets.
2. **Validate the Data Attribution Pipeline:** Add a dedicated experiment where $\rho_{\mathbf{s}}$ is used to rank training examples for a specific emergent behavior, followed by targeted data reweighting or removal to measure causal impact. Compare against standard influence baselines to empirically substantiate the paper's core promise of "tracing provenance."
3. **Strengthen Empirical Robustness & Clarity:** Report statistical significance (e.g., 95% CIs across prompt batches/seeds) for Table 1, correct the perplexity metric reporting to align with standard log-probability conventions, and explicitly discuss the practical bounds of the $\omega \geq 0.7$ threshold across diverse edit types rather than a single layer-depth trend.

---

## Vgm77U4ojX

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (6.8/10)
- Match: N/A

### Final Review

## Summary
SIGMADOCK introduces a fragment-based SE(3) Riemannian diffusion model for molecular docking that replaces coupled torsional parameterizations with independent rigid-body fragment assembly. By leveraging a novel stochastic fragmentation scheme (FR3D), soft triangulation constraints, and an SO(3)-equivariant architecture, the method achieves 79.9% Top-1 PB-valid accuracy on the temporally split PoseBusters benchmark, strictly trained on PDBBind(v2020) without post-hoc energy minimization.

## Strengths
- **Rigorous Geometric Motivation and Theoretical Grounding:** The paper provides a mathematically sound critique of torsional diffusion models. Theorem 1 formally demonstrates that torsional updates induce highly entangled, non-product measures in Cartesian space due to Jacobian coupling, whereas disjoint rigid fragments yield a clean product of Haar measures on SE(3)^m. This directly justifies the architectural shift.
- **Strong Empirical Results Under Strict Protocols:** The evaluation protocol rigorously isolates methodological gains from data scale by restricting training to ~19k complexes and matching train-test splits with prior work. Achieving 79.9% Top-1 PB-valid accuracy surpasses both classical physics-based dockers and recent DL approaches under identical conditions, demonstrating genuine generalization across sequence similarity splits.
- **Efficient and Plausibility-Driven Design:** The integration of FR3D, soft triangulation conditioning (Lemma 1), and a coordinate-frame-invariant Newton-Euler prediction head effectively constrains the conformational manifold without over-parameterization. This design yields chemically valid generations at ~0.57s/seed, successfully bypassing the need for heavy post-hoc minimization or separately trained confidence networks.

## Weaknesses
- **Narrow Evaluation Protocol Undermines Practical Claims:** The method is exclusively validated on rigid-receptor redocking with known pocket centers. Real-world drug discovery relies on cross-docking, apo-structure handling, and induced fit. While acknowledged in limitations, the absence of these evaluations makes sweeping claims about "reliability and feasibility for HTVS" premature and unsupported for ICLR standards.
- **Chirality Violations Expose Generative Gaps:** The authors rightly avoid energy minimization, yet explicitly acknowledge that chiral centers can be altered during fragment linking and require post-hoc filtering to discard undesired stereoisomers. This reliance on non-differentiable, post-generation filtering contradicts the narrative of learning a chemically coherent manifold *by construction* and shifts computational burden to the sampling stage.
- **Confounded Evidence for Geometric Disentanglement:** The primary claim that SE(3)^m diffusion outperforms torsional models due to geometric decoupling is compared against DiffDock, which uses a fundamentally different backbone and training setup. Without a controlled ablation swapping only the parameterization (SE(3)^m vs. T^k × SE(3)) under the identical architecture, dataset, and schedule, it remains unclear how much of the performance gain stems from the manifold choice versus architectural capacity.
- **Lack of Virtual Screening Metrics Despite HTVS Focus:** Redocking RMSD and PB-validity measure structural recovery on single complexes, but they do not correlate strongly with enrichment in realistic screening campaigns. The paper heavily argues for HTVS applicability yet omits standard metrics like Enrichment Factor (EF), AUC, or BEDROC on benchmark sets like DUD-E or MUV, leaving a critical gap in validating downstream utility.

## Nice-to-Haves
- Report statistical confidence intervals or error bars across multiple random seeds for the primary Top-1 metrics to rigorously quantify variance.
- Provide a systematic analysis of performance degradation as the pocket center is misspecified (e.g., shifted 1–3Å from the true ligand centroid) to bound the method's robustness in blind screening scenarios.
- Include explicit rank correlation analysis between the proposed pseudo-energy/PB heuristic and ground-truth RMSD across all generated seeds to validate the efficacy of the lightweight re-ranking step.
- Stratify error analyses by ligand flexibility (rotatable bond count) and molecular weight to empirically verify whether fragmentation mitigates the scaling pathologies hypothesized for torsional models.

## Novel Insights
The paper's most impactful contribution is reframing molecular docking as the reassembly of chemically coherent rigid bodies rather than the optimization of coupled torsional angles. By proving that independent noise on SE(3)^m fragments maps linearly to Cartesian space without metric-induced entanglement, the authors sidestep the ill-conditioned Jacobians and gauge ambiguities that plague torsional diffusion. The introduction of soft triangulation constraints elegantly pseudo-reduces degrees of freedom by implicitly fixing bond angles while leaving dihedrals free, demonstrating that principled geometric inductive biases can yield state-of-the-art accuracy without the massive data and compute overhead characteristic of modern co-folding models.

## Potentially Missed Related Work
- **Fragment-linking & linker-design diffusion models** (e.g., Guan et al., LinkerNet, NeurIPS 2023) — relevant for contextualizing the triangulation prior and rigid-fragment assembly within broader 3D molecular generation.
- **Latent flexible-receptor docking approaches** (e.g., DiffDock-Pocket, Plainer et al., 2023) — useful for positioning SIGMADOCK's rigid-receptor assumption against recent efforts that jointly sample side-chain conformations within the pocket.

## Suggestions
1. **Conduct a Controlled Manifold Ablation:** Train and evaluate a direct torsional-diffusion baseline using the exact same EquiformerV2 backbone, training schedule, and dataset. This is essential to isolate the geometric disentanglement benefits from confounding architectural variables.
2. **Incorporate Differentiable Chirality Constraints:** Move chirality preservation from post-hoc filtering into the generative process, e.g., by adding chiral-aware edge features or a differentiable stereochemical penalty to the score matching objective, to align fully with the "no post-processing" narrative.
3. **Evaluate Virtual Screening Performance:** Benchmark the model on standard screening datasets (e.g., DUD-E, LIT-PCBA, or MUV) reporting EF@1%, EF@5%, and ROC-AUC to substantiate HTVS feasibility beyond single-complex redocking.
4. **Add Cross-Docking or Pocket Perturbation Experiments:** Include a focused evaluation on cross-docking benchmarks (apo-receptors) or systematically report performance drops when the pocket definition radius/center is artificially enlarged or shifted. This will ground the claimed robustness and clarify the method's boundaries in realistic deployment settings.

---

## iaoAKDRAJQ

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (6.3/10)
- Match: N/A

### Final Review

## Summary
This paper provides a unified theoretical comparison between adaptive optimizers (e.g., Adam, Shampoo) and Normalized Steepest Descent (NSD) by analyzing the distinct smoothness and noise geometries they implicitly exploit. The authors extend the notion of adaptive smoothness to the nonconvex regime, prove it governs the convergence of a broad class of preconditioned methods, and demonstrate that this stronger assumption enables $\tilde{O}(1/T^2)$ acceleration in convex settings and dimension-free stochastic rates for NSD under a novel adaptive variance metric, with matching lower bounds under standard assumptions.

## Strengths
- **Rigorous nonconvex extension for structured preconditioners:** The paper successfully lifts the adaptive smoothness framework from convex to nonconvex optimization, establishing $\tilde{O}(T^{-1/4})$ rates for a unified meta-algorithm covering diagonal (Adam/AdaGrad) and non-commutative (Shampoo/ASGO) preconditioners. This fills a notable gap in prior work restricted to convex or purely diagonal analyses.
- **Clean theoretical separation results:** The paper tightly answers whether "stronger" geometric assumptions yield algorithmic benefits. Theorem 4.3 provably separates convex acceleration ($O(T^{-2})$) under adaptive smoothness from the $\Omega(T^{-1})$ worst-case under standard $\ell_\infty$ smoothness. Theorem 4.5 similarly shows adaptive variance eliminates dimension dependence in stochastic NSD, with a matching lower bound (Theorem 4.7) confirming the necessity of this assumption.
- **Substantial technical contribution:** Lemma 3.3 introduces a novel matrix inequality that elegantly handles noncommutativity in general well-structured preconditioner sets, isolating a provable $\log d$ penalty that vanishes in the commutative/diagonal case. This tool bypasses the scalar telescoping limitations of prior analyses and is likely useful beyond this specific work.

## Weaknesses
- **Unverified practical tightness of the geometric dichotomy:** The entire argument hinges on adaptive smoothness/variance being meaningfully tighter than standard notions, yet Proposition 2.5 only guarantees $\Lambda_H(f) \leq d \cdot L_{\|\cdot\|_H}(f)$. If the ratio scales with $d$ in practice, the theoretical advantages (acceleration, dimension-free rates) are offset by a worse constant, rendering the separation mathematically valid but practically vacuous. The paper provides no analysis, heuristics, or empirical measurements of when $\Lambda_H(f) \ll d \cdot L_{\|\cdot\|_H}(f)$ actually holds.
- **Restrictive assumptions limit scope to realistic deep learning regimes:** The convex acceleration result (Theorem 4.3) requires prior knowledge of the domain radius $D$ or relies on a projection step (Algorithm 8), which is misaligned with the unconstrained, hyperparameter-tuned optimization used in practice. More critically, the unified nonconvex analysis depends on Assumption D.1 (uniform PSD-bounded gradient noise covariance), a strong condition that excludes heavy-tailed noise and state-dependent variance scaling commonly observed in large-batch or low-precision training.
- **Lack of empirical grounding weakens impact:** While purely theoretical papers are acceptable at ICLR, this work makes explicit claims about *why* adaptive and NSD-type optimizers behave differently in practice but offers zero empirical validation of its central quantities ($\Lambda_H$, $\sigma_H$). Without trajectory measurements showing these adaptive metrics behave as assumed, the paper remains an elegant but isolated mathematical exercise, limiting its influence on optimizer design.
- **Opaque technical exposition in main text:** The core mechanism enabling the nonconvex guarantees (Lemma 3.3 and the transition to noncommutative preconditioners) is deferred entirely to dense appendices. The main text lacks intuitive sketches of how matrix logarithmic bounds circumvent noncommutativity, making the paper inaccessible to the broader optimization community.

## Nice-to-Haves
- Include a 1-2 paragraph conceptual sketch of Lemma 3.3 in Section 3.3, explaining how the logarithmic transformation decouples matrix differences without relying on entry-wise commutativity.
- Discuss parameter-free or dual-averaging extensions for Algorithm 2/8 that could eliminate the explicit dependence on $D$, bringing the acceleration result closer to practical hyperparameter-free optimization.

## Novel Insights
The paper reframes adaptivity not merely as a heuristic learning-rate scaling mechanism, but as an implicit geometric reparameterization that changes the underlying smoothness and noise structure of the optimization landscape. By showing that adaptive preconditioners effectively "search" over a family of induced norms to find a tighter smoothness constant, the authors reveal why NSD and adaptive methods converge at different rates despite operating in the same dual norm space. The introduction of adaptive variance extends this idea to stochasticity, demonstrating that uniform geometric control over preconditioner-induced noise can break dimension dependency where standard variance assumptions fail, exposing a fundamental symmetry between smoothness adaptation and noise adaptation in non-Euclidean optimization.

## Potentially Missed Related Work
- None identified. The paper thoroughly covers recent foundational works on non-Euclidean NSD (Pethick et al., Kovalev 2025), adaptive preconditioner analysis (Xie et al., An et al.), and acceleration in non-Euclidean settings (Cutkosky, Allen-Zhu & Orecchia).

## Suggestions
1. Quantify or bound the ratio $\Lambda_H(f) / L_{\|\cdot\|_H}(f)$ for canonical non-convex landscapes (e.g., quadratic with anisotropic curvature, Rosenbrock, or simple two-layer networks) to demonstrate that the adaptive constant does not trivially scale with $d$. Even synthetic worst-case vs. average-case analysis would significantly strengthen the motivation for Q2.
2. Replace or supplement the deterministic $D$-dependence in the convex acceleration theorem with a discussion of online-to-batch or dual-averaging variants that adaptively estimate the domain size, clarifying how the theoretical rate translates to unconstrained deep learning pipelines.
3. Add a dedicated paragraph in the main text outlining the proof strategy of Lemma 3.3, specifically how the matrix log-derivative inequality (Lemma C.1) is leveraged to bound the noncommutative telescoping sum, improving readability without expanding the page count.

---

## rBj2iVyrhh

- GT: Reject (avg 2.0)
- Predicted: N/A (4.2/10)
- Match: N/A

### Final Review

## Summary
The paper proposes Classifier-Constrained Alternating Training (CCAT), a two-stage framework designed to mitigate modality imbalance by first pretraining a shared classifier with a modality-contribution regularization term, then freezing it as a stable anchor during alternating modality updates. Modality-specific LoRA adapters bridge the distribution gap between fused and unimodal features, while a sample-level secondary optimization step targets severely imbalanced instances. The approach yields consistent accuracy improvements across three multimodal benchmarks.

## Strengths
- **Accurate diagnosis of classifier structural bias:** The empirical tracking of modality contributions (Fig. 1) convincingly demonstrates that alternating encoder updates alone fail to prevent early-dominance bias in the shared decision boundary, correctly identifying a critical, underexplored gap in prior alternating training methods.
- **Cohesive algorithmic design with systematic ablation:** The integration of contribution-regularized initialization, classifier freezing, alternating optimization, and sample-level secondary updates is logically structured. Table 2 ablations clearly isolate the contribution of each component, validating that freezing prevents decision-boundary monopolization and secondary updates recalibrate weak modality representations.
- **Consistent empirical gains and reproducible setup:** The method achieves substantial multimodal accuracy improvements (+1.35% to +6.76%) over established gradient-modulation and early alternating baselines on CREMA-D, Kinetic-Sound, and MVSA. Implementation details, optimizer schedules, and hyperparameter grids are explicitly reported, supporting straightforward reproduction.

## Weaknesses
- **Critical omission of direct competitors in main comparisons:** Despite explicitly citing MLA, MMPareto, and LFM as recent state-of-the-art methods and claiming consistent superiority over them, Table 1 entirely excludes these baselines. This omission prevents independent verification of the central performance claims and undermines the paper's positioning against the most relevant contemporary work.
- **Unquantified computational overhead and scalability concerns:** The pipeline requires classifier pretraining, full alternating batch sweeps, and per-batch secondary forward/backward passes for samples below threshold $\beta$. Neither training wall-clock time nor FLOPs/memory overhead is reported relative to synchronous or single-pass alternating baselines, leaving a major practical limitation completely unaddressed.
- **Overstated theoretical grounding and imprecise terminology:** Section 3.1 presents a “mathematical analysis” and “proof” of gradient dynamics, but derives approximations assuming a linear fusion model ($\gamma_1 f^{(1)} + \gamma_2 f^{(2)}$) that does not mathematically extend to the bidirectional cross-attention architecture actually deployed in Section 3.2. Additionally, Equations 5–6 label a normalized cosine-similarity score as “estimated mutual information” without variational bounds or information-theoretic justification, misrepresenting a heuristic contrastive proxy as a formal MI estimator.
- **Lack of statistical rigor and variance reporting:** Results are averaged over three random seeds, but no standard deviations, confidence intervals, or significance tests are provided. Given the moderate benchmark sizes and the multi-stage, threshold-dependent training pipeline, this omission makes it impossible to distinguish genuine gains from standard training variance.

## Nice-to-Haves
- Sensitivity analysis for the pretraining regularization coefficient $\lambda$, as it directly dictates initial boundary quality and heavily influences frozen-stage performance.
- Direct ablation comparing LoRA adapters against full-rank linear projections to determine whether low-rank constraints are structurally necessary or merely a parameter-efficient convenience.
- Brief discussion or empirical probe on how the method handles completely missing modalities at inference, given the fixed-classifier + decision-level fusion design.

## Novel Insights
The paper’s most valuable contribution is reframing modality imbalance not merely as an encoder gradient conflict, but as a *decision boundary drift* problem amplified by alternating training paradigms. By recognizing that early-dominance bias structurally entrenches in the classifier—paralleling convergence dynamics observed in class-imbalanced learning—the authors propose a clean decoupling: stabilize the decision anchor first, then adapt representations around it via lightweight per-modality corrections. This shifts the optimization focus away from heuristic gradient modulation and toward representation alignment around a fixed objective, offering a conceptually distinct pathway for multimodal representation learning.

## Potentially Missed Related Work
- Recent advances in classifier calibration and decision-boundary anchoring for vision-language or large multimodal models (e.g., asymmetric logit adjustment, temperature-calibrated decision heads in parameter-efficient fine-tuning). While the paper grounds itself well in class-imbalance literature, connecting the frozen-anchor design to contemporary classifier stabilization techniques in foundation-model fine-tuning would strengthen broader relevance.

## Suggestions
1. **Include missing SOTA baselines in Table 1** (MLA, MMPareto, LFM) under identical training/evaluation protocols, or explicitly detail why architectural differences preclude direct comparison.
2. **Report efficiency metrics** (training time per epoch, peak GPU memory, effective FLOPs) relative to standard alternating and synchronous baselines to contextualize the two-stage + secondary update overhead.
3. **Reframe Section 3.1 as a motivating heuristic** rather than a formal proof, and correct the terminology in Equations 5–6 to accurately describe the metric as a normalized similarity/contribution score unless a rigorous mutual information bound is introduced.
4. **Add statistical variance** (standard deviations or confidence intervals) across the three random seeds to all main results and ablation tables to meet ICLR’s empirical reporting standards.
5. **Clarify the metric shift in Section 3.3**, where the contribution computation transitions from pretrained feature-level similarity to decision-level fusion scores, and justify why this change preserves the intended regularization effect during alternating training.

---

## C6WWMryELL

- GT: Reject (avg 5.5)
- Predicted: N/A (4.8/10)
- Match: N/A

### Final Review

## Summary
This paper investigates output volatility in long-form LLM generation, introducing VOLTBench to systematically quantify length and structural instability across multiple sampling runs. Through attention trace analysis, the authors identify "attention collapse" and "instability" as internal precursors to generation failures. To address these, they propose SELB, a training-free decoding intervention that enforces section transitions via logit boosting and suppresses premature termination tokens. Extensive evaluation across tasks scaling to ~500 sections demonstrates significant gains in length adherence, stability, and structural fidelity compared to strong baselines.

## Strengths
- **Rigorous Reframing of Evaluation Metrics:** VOLTBench successfully shifts the assessment paradigm from single-instance quality to multi-run distributional stability. The benchmark’s multi-dimensional design (language, complexity, structured vs. unstructured, and scales up to 100k tokens) effectively exposes brittle failure modes in both closed and open-weight models that prior single-shot benchmarks miss.
- **Compelling Empirical Performance at Extreme Scales:** SELB achieves substantial reductions in volatility (LVC) and near-perfect structural adherence (FAD/SCA) on the 100–500 section tasks where specialized long-text models and frontier APIs catastrophically fail. The inclusion of hidden-state similarity and lexical diversity metrics provides credible evidence that the method preserves semantic coherence rather than merely generating repetitive filler to satisfy length constraints.
- **Interpretable Probing of Long-Horizon Failure:** The attention trace analysis offers a clear, visual diagnostic linking observable generation breakdowns (premature termination, section skipping) to internal attention dynamics. This provides the community with a practical tool for analyzing why standard autoregressive decoding degrades over long sequences.

## Weaknesses
- **Incremental Algorithmic Novelty of SELB:** The core mechanism (applying strong positive bias to predefined structural tokens and masking EOS/filler logits with $-\infty$) is functionally equivalent to established constrained decoding, grammar forcing, and dynamic stopping techniques. The paper does not theoretically or empirically differentiate SELB from prior structure-enforcing decoders, leaving the contribution feeling like an effective engineering application of known primitives rather than a methodological advancement expected at ICLR.
- **Missing Component Ablations and Hyperparameter Justification:** The relative contribution of $M_{struct}$ (structural boosting) versus $M_{fail}$ (EOS/filler suppression) is never isolated. Without these ablations, it is impossible to determine whether stability gains stem from active structural guidance or merely from artificially preventing the model from stopping. Furthermore, critical hyperparameters ($\beta$, $\tau_{max}$, and hybrid grace periods) lack sensitivity analysis, severely limiting reproducibility and practical deployment guidance across diverse architectures or prompt formats.
- **Correlational Probing and Methodological Mislabeling:** While the attention traces align temporally with failure modes, the analysis remains strictly observational. The paper provides no causal interventions (e.g., constraint token masking post-collapse, attention head perturbation, or KV-cache saturation controls) to prove that attention dynamics *drive* volatility rather than merely co-occur with it. Additionally, Appendix H is explicitly titled "CKA Analysis" but exclusively computes and discusses Cosine Similarity; conflating these mathematically distinct representational similarity metrics undermines analytical rigor.
- **Restricted Generalization and Deployment Constraints:** SELB’s validation is heavily anchored to a single Qwen2.5-7B variant. The method’s reliance on logit-level access, explicit token targeting, and rigid section thresholds renders it inapplicable to black-box commercial APIs and raises unresolved questions about robustness on implicitly structured outputs (e.g., continuous essays, technical reports without headers) or fundamentally different architectures (e.g., MoE, state-space models).

## Nice-to-Haves
- Clarify how $V_{title}^{(p+1)}$ is dynamically constructed for prompts without fixed chapter/section templates, as real-world long-form tasks rarely use strict enumerative formatting.
- Expand sampling beyond $N=5$ seeds or report confidence intervals for distributional metrics (LSD/LVC), as high-variance baselines (e.g., LVC > 45%) make tail estimates with only 5 runs statistically fragile.
- Provide qualitative, side-by-side generation samples at forced transition points to assess whether SELB introduces semantic discontinuities, hallucinated headers, or low-information padding in narrative flows.

## Novel Insights
The paper’s most valuable contribution is the empirical demonstration that *generation scale* and *output stability* are fundamentally decoupled in current LLMs. By showing that long-text specialized models can achieve high token counts while suffering catastrophic structural volatility (LVC > 70%), the work challenges the prevailing assumption that domain-specific SFT inherently guarantees reliable controllability. The identification of periodic "attention summits" as necessary refocusing signals offers a compelling mechanistic hypothesis for why standard autoregressive sampling loses track of long-horizon structural constraints, framing volatility not as random noise but as a predictable decay of constraint-attention coupling.

## Potentially Missed Related Work
- **Constrained Decoding & Grammar Forcing (e.g., Outlines, XGrammar, Neurosymbolic decoding):** Highly relevant for contextualizing SELB’s logit manipulation against existing formal-constraint enforcement methods that could theoretically achieve similar structural adherence without custom biasing schedules.
- **Long-Context Decoding Interventions (e.g., KV-cache eviction-aware generation, positional interpolation mitigations):** These works directly address attention degradation and representational drift over extended sequences, providing alternative or complementary mechanisms for stabilizing long-horizon generation.

## Suggestions
1. **Run strict component ablations ($M_{struct}$ only, $M_{fail}$ only, combined)** and conduct a hyperparameter sensitivity sweep ($\beta$, $\tau_{max}$) to isolate the exact mechanism driving stability gains and establish practical tuning guidelines.
2. **Position SELB explicitly against modern constrained/template-guided decoding baselines** to empirically validate whether the method provides unique advantages beyond standard grammar forcing or structural prompting.
3. **Correct the CKA/Cosine Similarity conflation in Appendix H** and, if possible, add a minimal causal intervention (e.g., masking constraint tokens upon detected attention decay) to move the mechanistic analysis from correlational to demonstrative.
4. **Clarify the pipeline for constructing target title tokens** for non-templated prompts and discuss the method’s limitations and failure modes when applied to continuous, implicitly structured, or API-hosted models.

---

## GiaF5cFIpI

- GT: Reject (avg 3.5)
- Predicted: N/A (3.4/10)
- Match: N/A

### Final Review

## Summary
This paper introduces a real-time, streaming framework for adaptive neural stimulation, integrating online latent space construction, continuous dynamical modeling, a non-parametric kernel regressor for state-dependent stimulus-response mapping, and a constrained optimization routine to design high-dimensional stimulation patterns. The pipeline demonstrates sub-100ms computational latency, rapid convergence (~10–20 stimulations), and robust adaptation to simulated non-stationary response dynamics. While the engineering integration addresses a tangible workflow bottleneck in closed-loop neuroscience, the empirical validation relies heavily on synthetic stimulation injections applied to offline traces, significantly limiting the claims of biological applicability and algorithmic robustness.

## Strengths
- **Cohesive, deployable closed-loop architecture:** The framework successfully unifies streaming dimensionality reduction, dynamical state prediction, adaptive response modeling, and constrained stimulation design into a single Algorithm 1 loop. Demonstrated end-to-end latencies (averaging <10ms, worst-case <100ms) on standard hardware meet the stringent real-time requirements necessary for brain-machine interface and closed-loop optogenetic experiments.
- **Effective adaptation to non-stationary mappings:** The introduction of a temporal kernel component that discounts outdated stimulus-response observations is a practical and well-motivated solution to biological drift and experimental instability. Experiments in Figure 2e convincingly show that the regressor recovers predictive accuracy following abrupt discontinuities and continuous rotational drift, addressing a common failure mode in static neuro-perturbation models.

## Weaknesses
- **Circular validation via synthetic stimulations on real data:** The core experimental validation on calcium and electrophysiological recordings (Section 4.1) does not use actual biological stimulation responses. Instead, stimulations are injected via a simple linear autoregressive model (`at = 0.8·at-1 + ut`). This completely side-steps the complex, non-linear, network-mediated, and heterogeneous responses that non-parametric mapping is designed to capture, rendering the real-data results tautological. While real-stimulation results appear in Appendix C, relegating the most critical ecological validation to the supplement fundamentally undermines the paper's central claim.
- **Weak comparative baselines:** The method is benchmarked primarily against a "blind" dynamical model and random/target-shuffled stimulation strategies. It lacks direct comparison to established state-of-the-art stimulation design algorithms (e.g., Bayesian optimization, active learning frameworks, or linear input-output dynamical models) cited in the introduction. Without these comparisons, it is impossible to assess whether the proposed kernel+gradient-optimization pipeline offers measurable gains in alignment or sample efficiency over simpler or more sophisticated alternatives.
- **Mathematical mismatch in constrained optimization:** Equation 8 proposes using an $L_1$ penalty to approximate $L_0$ neuron-count sparsity but optimizes it using L-BFGS-B. L-BFGS-B is a quasi-Newton method designed for box-constrained, smooth objectives; it is ill-suited for non-differentiable $L_1$ penalties and typically fails to produce true zero-valued coefficients without explicit thresholding or proximal updates. This mismatch raises concerns about whether the method actually achieves the claimed sparse stimulation patterns or merely produces dense, low-amplitude vectors.
- **Unaddressed kernel scaling and lack of statistical rigor:** The stimulus-response estimator (Eq. 7) stores all historical observation triples, causing memory and inference costs to scale linearly with experiment duration. The reported complexity bounds do not reconcile this growth with the requirement for stable, long-term real-time deployment. Furthermore, several key results (Figures 4, 5) lack rigorous statistical reporting (e.g., confidence intervals, pairwise significance tests across stochastic runs), and no systematic hyperparameter ablation (kernel bandwidths, $\lambda_1$, latent dimensionality) is provided to demonstrate robustness.

## Nice-to-Haves
- Theoretical convergence rates or stability bounds for the proposed `sjPCA` streaming algorithm under observation noise.
- Extension of the framework to model temporally overlapping or sustained stimulations, as the current fixed-delay, single-pending assumption is restrictive for rapid experimental protocols.

## Novel Insights
The work correctly identifies that rigid, parametric assumptions often fracture when applied to biological circuit perturbation, and demonstrates that treating the stimulus-response landscape as a dynamically drifting function can be effectively managed with online kernel discounting. By running multiple latent space hypotheses and dynamical filters in parallel, the framework implicitly acknowledges that no single subspace universally captures neural computation; instead, adaptive subspace selection driven by real-time predictive error offers a pragmatic pathway to disentanglement. This shifts the paradigm from static manifold identification to continuous representational arbitration during active intervention.

## Potentially Missed Related Work
- PETRELS or Grassmannian online subspace tracking algorithms — highly relevant for establishing theoretical and empirical baselines against the novel `sjPCA` streaming formulation.
- Exact sparse stimulation selection methods (e.g., Mixed-Integer Programming or greedy forward/backward selection) — relevant for validating whether the $L_1$ relaxation adequately captures hardware-constrained neuron targeting.

## Suggestions
- Move the real-stimulation experiments from the appendix to the main text and benchmark them directly against a state-of-the-art stimulation optimizer to prove ecological validity and algorithmic superiority.
- Replace or augment the L-BFGS-B solver with a proximal gradient method or reweighted $L_1$ scheme to rigorously enforce sparsity, and include an ablation sweep over $\lambda_1$ and kernel bandwidths to demonstrate tuning robustness.
- Implement a kernel approximation strategy (e.g., sliding-window truncation, inducing points, or Nyström sampling) to bound memory/compute growth, and report latency stability across experiments involving thousands of stimulation events.
- Include proper statistical testing (e.g., bootstrapped confidence intervals, paired t-tests across random seeds/datasets) for all alignment and error reduction claims to confirm reliability beyond point estimates.

---

## Me0n0iESJY

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (5.6/10)
- Match: N/A

### Final Review

## Summary
The paper introduces OptMerge, a model merging benchmark and algorithm designed for Multimodal Large Language Models (MLLMs). It proposes a data-free optimization method that denoises task vectors via low-rank SVD approximation and stabilizes convergence through SGD and mean initialization, demonstrating that merging can match or exceed mixture training efficiency across multiple visual capabilities and modalities with significantly reduced computational cost.

## Strengths
- **Timely and Rigorous Benchmark Construction:** The authors address a clear gap by curating the first MLLM-specific merging benchmark with fine-grained task categorization (VQA, Geometry, Chart, OCR, Grounding) and >100k samples per task. Covering both full fine-tuning (InternVL2.5) and LoRA regimes (Qwen2-VL) ensures broad applicability, and the public release of checkpoints/code provides a valuable community resource.
- **Practical and Effective Algorithmic Design:** OptMerge successfully addresses known instabilities in data-free task vector optimization. The targeted interventions (SVD-based denoising for full-FT models, SGD with mean initialization for LoRA models) resolve the norm-inflation "shortcut" phenomenon, yielding consistent average gains and robust performance on real-world HuggingFace checkpoints and 32B-scale models.
- **Strong Empirical Validation & Efficiency Gains:** The paper convincingly demonstrates that data-free static merging can rival multi-task mixture training while requiring only a fraction of the compute time and memory. The modality merging experiments also effectively showcase cross-modal complementarity, advancing practical pathways toward decentralized Omni-model development.

## Weaknesses
- **Mischaracterization of the Specialization-Generalization Trade-off:** The paper claims that "merging individually specialized models outperforms expert MLLMs on their target tasks." This is factually inconsistent with Tables 2 and 3: merged models consistently degrade on peak specialist metrics (e.g., InternVL VQA/GQA drops from 60.91 to 57.13, Geometry drops from 55.20 to 54.48) in exchange for higher multi-task averages. Framing inevitable multi-task regularization losses as "outperforming" specialists obscures a critical limitation for domain-specific deployment and misleads readers regarding merging capabilities.
- **Decoupled Theoretical Contribution and Algorithm:** Theorem 3.1 provides standard PL/L-smoothness bounds to explain why excessive fine-tuning ($\eta T$) harms merging due to cross-task interference and curvature terms. However, the theory is purely descriptive and fails to mathematically connect to OptMerge's core mechanisms. It does not justify why SVD truncation ($k=\text{rank}/5$) reduces the interference term, nor does it provide guarantees or convergence analysis for the proposed data-free gradient updates. The theoretical and algorithmic contributions operate in isolation.
- **Insufficient Evaluation of Post-Merge Language Degradation & Unverified Mixture Claim:** The evaluation relies exclusively on domain-specific accuracy metrics and completely omits text-only benchmarks (e.g., MMLU, GSM8K, IFEval) and generative quality metrics (hallucination, instruction adherence), leaving it unknown whether merging visual specialists degrades core LLM capabilities. Furthermore, for Qwen2-VL, the "mixture training" upper bound is approximated by an off-the-shelf instructed model rather than a true mixed-data baseline trained under identical conditions, making the central claim that merging surpasses data mixing empirically unverified.
- **Heuristic Design Choices Lack Grounding:** OptMerge relies on several unexplained hyperparameters: the fixed rank ratio $k=\text{rank}/n_{tasks}$, a grid search for $\lambda$, and 300 fixed optimization iterations. While Table 8 shows some robustness to $k$, the paper provides no spectral analysis or interference-based criterion to determine why dropping specific singular values removes "noise" rather than valid task signals, nor does it offer an adaptive selection mechanism for heterogeneous task distributions.

## Nice-to-Haves
- Implementing an adaptive, spectrum-aware rank selection mechanism or an interference-driven stopping criterion would reduce heuristic tuning and improve robustness across varying task combinations.
- Expanding the modality merging evaluation to include broader video/audio understanding benchmarks (e.g., Video-MME) and adding qualitative case studies showing capability trade-offs would strengthen the "Omni-model" claims.

## Novel Insights
The paper’s most valuable observation is the empirical validation that controlled, lightweight optimization of task vectors can effectively balance multi-task interference in MLLMs without access to training data. By demonstrating that merging can match or exceed mixture training efficiency while maintaining strong average performance across heterogeneous visual capabilities, the work solidifies model merging as a viable, decentralized alternative to expensive joint fine-tuning. The explicit analysis of norm-inflation shortcuts in gradient-based task vector optimization provides a practical heuristic toolkit (SGD + mean initialization + targeted truncation) that addresses real stability bottlenecks in data-free merging pipelines.

## Potentially Missed Related Work
- **Task Arithmetic in the Tangent Space (Ortiz-Jimenez et al., NeurIPS 2023)** and **Fisher-weighted merging / Task-Specific Sparsification**: These works explore similar parameter interference dynamics and information geometry in weight space. Contrasting OptMerge's SVD heuristic against these could yield stronger theoretical grounding and clarify when low-rank truncation is optimal versus sparsification.

## Suggestions
- **Correct claims regarding specialist performance.** Replace statements implying merged models "outperform experts on target tasks" with precise language acknowledging the inherent specialization-generalization trade-off. Provide explicit delta tables (Merged − Best Individual) to transparently report gains in average performance against predictable losses in peak specialist capability.
- **Add core language/instruction benchmarks.** Evaluate merged models on text-only and instruction-following datasets to rigorously quantify any catastrophic forgetting or capability degradation. This is critical for validating that merging MLLMs does not compromise the underlying LLM.
- **Strengthen the theory-method connection.** Provide an analysis linking the SVD truncation step to the reduction of the $O(\delta \eta T)$ interference term in Theorem 3.1, or empirically demonstrate how singular value energy correlates with task interference. Include spectral decay plots to justify the fixed $k$ ratio.
- **Implement a true mixture training baseline for Qwen.** To sustain the claim that merging surpasses data mixing, conduct an actual SFT run using the combined task datasets on Qwen2-VL-Base under equivalent compute/epochs, rather than relying on a pre-trained model as a proxy.
- **Clarify LoRA implementation details.** Explicitly state whether LoRA adapters are materialized into full dense weights prior to merging or if the optimization operates directly on component matrices, including how potential rank inflation is handled. Add this to a reproducibility appendix.

---

## Ksvv8x00eo

- GT: Withdrawn (treated as Reject) (avg 3.5)
- Predicted: N/A (5.6/10)
- Match: N/A

### Final Review

## Summary
This paper introduces CaTS-Bench, a multimodal benchmark for context-aware time series captioning and diagnostic reasoning, constructed from 11 real-world datasets. The authors propose a scalable, oracle-based caption generation pipeline rigorously validated through factual verification, human indistinguishability studies, and paraphrase robustness checks. Extensive evaluations of leading VLMs reveal that while finetuning and program-augmented generation improve numeric fidelity, current models largely neglect visual plot cues and struggle with precise temporal statistical inference.

## Strengths
- **Rigorous validation of semi-synthetic ground truth:** The paper directly confronts skepticism around LLM-generated references with exceptional thoroughness. Manual verification confirms >98.6% factual accuracy across statistical and trend claims, human detectability sits near chance (41.1%), and paraphrase robustness experiments show high ranking stability (Spearman $\rho \approx 0.92$), demonstrating the benchmark measures semantic fidelity rather than oracle mimicry.
- **Actionable empirical discovery on visual modality neglect:** The modality ablation and attention analyses provide a robust, well-supported finding: state-of-the-art VLMs default to textual priors and fail to extract meaningful information from line charts. This is reinforced by tests on alternative encodings (GAF, RP) which do not recover performance, isolating the failure to current multimodal alignment rather than benchmark artifacts.
- **High reproducibility and transparent diagnostic design:** The release of exact prompts, detailed finetuning configurations, explicit metric formulations, and a public dataset meets strong ICLR reproducibility standards. The inclusion of perturbation-based Q&A tasks and a PAL baseline effectively isolates numeric reasoning capabilities from superficial linguistic generation.

## Weaknesses
- **Short sequence lengths fundamentally mismatch the "reasoning" claims:** The average sample window spans only ~26–29 time steps (Table 2). While adequate for local trend spotting, these short horizons are insufficient to evaluate genuine temporal reasoning, such as seasonality detection, structural break identification, or long-range dependency modeling. The paper repeatedly frames CaTS-Bench as a testbed for "time series reasoning," yet the evaluation scope is largely restricted to short-horizon descriptive summarization, overstating the benchmark's capacity to model complex temporal dynamics.
- **Proposed metrics lack validation against human judgment of model outputs:** The paper introduces a custom "Numeric Score" and "Statistical Inference Accuracy" but provides no correlation between these automated scores and human assessments of *model-generated* captions. While ground-truth captions are validated, a benchmark proposing novel evaluation criteria must demonstrate that its metrics align with human perception of caption quality and factual utility. The 0.7/0.3 recall/accuracy weighting and 5% tolerance threshold are presented as heuristics without sensitivity analysis or empirical grounding, leaving the core numeric evaluation framework methodologically unverified.
- **Single-oracle dependency with insufficient human-correction scale:** Despite robust validation, the reference corpus relies exclusively on Gemini 2.0 Flash. The human-revisited test subset comprises only 579 samples (~14% of the test split), which is too small to confidently assert that the benchmark captures diverse human descriptive conventions or fully mitigates subtle oracle biases. This risks penalizing models that produce factually correct but stylistically divergent captions, limiting the generalizability of the reported rankings.

## Nice-to-Haves
- Report mean ± standard deviation or confidence intervals directly in the main experimental tables to meet standard ICLR presentation expectations, rather than relegating variance analysis to the appendix.
- Provide stratified results by sequence length, volatility, and trend complexity to clarify whether reported failures reflect genuine reasoning deficits or trivial aggregation over short windows.
- Explicitly document the exact numeric formatting (decimal precision, delimiters) and maximum context bounds injected into prompts to rule out tokenization or context-window artifacts in the numeric fidelity scores.
- Consider including domain-expert participants in future human baselines to establish a more realistic performance ceiling for specialized verticals like healthcare or climate analytics.

## Novel Insights
The most impactful contribution of this work is the empirical demonstration that modern Vision-Language Models exhibit a systematic failure in multimodal alignment when tasked with time series interpretation, consistently ignoring explicit visual trend cues in favor of textual metadata. This breakdown persists across model scales, proprietary versus open-source architectures, and even when alternative mathematical visualizations replace standard line plots. The finding reframes current VLM "chart understanding" as largely superficial pattern recognition tied to axes and titles, rather than genuine geometric or trend grounding. This provides a clear, measurable architectural failure mode that signals the need for specialized numeric-visual fusion layers rather than relying on general-purpose vision encoders for structured temporal data.

## Potentially Missed Related Work
- The paper effectively contrasts itself against TADACap, TRUCE, and TACO. For completeness, the authors may wish to briefly situate CaTS-Bench against recent chart-centric LLMs (e.g., DePlot, CharterV) or specialized time-series VLMs to clarify how the proposed pipeline differs from end-to-end chart captioning or forecasting adaptations rather than penalizing the omission.

## Suggestions
- Conduct a formal correlation study (e.g., Spearman/Pearson, Kendall-Tau) between the proposed numeric/linguistic metrics and human Likert ratings of a stratified sample of VLM outputs. Use this to empirically justify the 0.7/0.3 weighting and 5% tolerance, proving the metrics capture human-aligned quality rather than heuristic surface overlap.
- Reframe the narrative and contribution claims to accurately reflect the benchmark's actual scope: CaTS-Bench is a strong foundation for *short-horizon, context-aware descriptive captioning*. Explicitly acknowledge that long-sequence temporal reasoning, anomaly detection windows, and structural change analysis require extended versions of the benchmark.
- Establish and publish a standardized evaluation protocol that scores models on the human-revisited subset separately from the semi-synthetic corpus. This will allow the community to directly measure the gap between oracle-optimized performance and human-aligned caption quality as the field advances.

---

## wgGJE6Z1B3

- GT: Accept (Poster) (avg 5.5)
- Predicted: N/A (5.4/10)
- Match: N/A

### Final Review

## Summary
This paper addresses the computational bottleneck in training draft models for speculative decoding by proposing a data-centric filtering strategy. The authors identify that tokens yielding flatter target-model distributions offer greater per-step reductions in the $L_1$ distance governing acceptance rates, introduce a cosine-similarity-to-uniform "flatness" metric to quantify this, and develop SFDD, a sample-level distillation pipeline. Experiments on the EAGLE-2 framework demonstrate that filtering to 50% of the data cuts training time by ~2× while preserving ~96% of the full-dataset inference speedup.

## Strengths
- **Clear, acceptance-centric theoretical framing:** The paper correctly identifies the mismatch between standard KL-divergence distillation and SD's $L_1$-driven acceptance objective, and provides an analytically tractable Gaussian toy model to motivate why flatter distributions yield higher optimization headroom.
- **Systematic and comprehensive empirical validation:** The evaluation spans multiple models, datasets, temperature regimes, and retention ratios (down to 5%). SFDD consistently outperforms standard uncertainty heuristics (entropy, top-1, margin, energy, PPL) and random baselines across all tested settings.
- **Strong training dynamics analysis:** The per-epoch tracking of gradient norms, loss saturation, and $\Delta L_1$ reductions (Fig. 2, Fig. 7, App. F.5) provides concrete empirical evidence that high-flatness tokens sustain meaningful gradient flow while low-flatness tokens saturate rapidly.
- **Practical deployability:** The pipeline requires only a single offline forward pass of the frozen target model, incurs modest overhead (~3.8%), and operates orthogonally to existing draft architecture or loss-function designs.

## Weaknesses
- **Lack of mechanistic differentiation from entropy:** While the paper empirically shows flatness outperforms entropy, Appendix F.2 explicitly links both to divergence from the uniform distribution. However, cosine similarity (L2-based) and Shannon entropy (log-prob-based) induce different rankings. The paper fails to explain *why* the L2 geometry better predicts $\Delta L_1$ reduction or acceptance-rate headroom than the log-space dispersion metric. Without this, flatness risks being perceived as an empirically tuned variant of entropy rather than a theoretically grounded innovation.
- **Missing downstream generation quality evaluation:** The evaluation relies exclusively on inference speedup and acceptance length. While SD is theoretically distribution-preserving, aggressive data filtering during KD training can induce subtle alignment degradation, mode collapse, or hallucination in practice. The absence of output quality metrics (e.g., exact match, ROUGE/BLEU, or LLM-as-judge scores on reasoning/summarization tasks) leaves the claim that SFDD "preserves inference capabilities" empirically incomplete.
- **Absence of a compute-matched baseline:** The paper compares filtering at fixed retention ratios but omits the critical control of training the *full dataset* for proportionally fewer epochs (or using early stopping). Without this, it remains ambiguous whether SFDD genuinely selects higher-quality data or merely serves as a coarse proxy for reduced compute/earlier convergence.
- **Unsupported hypothesis and missing statistical rigor:** Section 5.4 speculates that SFDD’s slight training-time advantage over random filtering stems from "enhanced batching efficiency," providing zero compute profiling (e.g., padding ratios, sequence-length distributions, or hardware utilization metrics) to substantiate this. Additionally, all main result tables report point estimates without variance across seeds or repeated filtering runs, making it impossible to assess the statistical robustness of the reported gaps under stochastic subset selection.

## Nice-to-Haves
- Visualize rank-ordered softmax distributions for high- vs. low-flatness tokens to confirm the metric captures meaningful structural uncertainty rather than selecting uniform noise or out-of-distribution artifacts.
- Provide a complexity/scaling analysis for pretraining-scale corpora (billions of tokens), discussing potential bottlenecks and mitigation strategies (e.g., stratified sampling, proxy scoring) for the offline filtering step.
- Explore dynamic or online scoring strategies that adapt to the evolving draft distribution $q$ across training epochs, rather than relying on static initial $p$-based importance scores.

## Novel Insights
The paper successfully reframes data importance for speculative decoding away from traditional confidence- or error-centric lenses and toward an acceptance-rate-driven perspective grounded in $L_1$ distance minimization. By demonstrating that flat target distributions inherently provide more per-step optimization headroom under constrained KD updates, the work establishes "flatness" as a static, target-model-only proxy for training value. This shifts the paradigm from uniform or heuristic-based data curation to a targeted distillation strategy that explicitly acknowledges the structural mismatch between standard language modeling objectives and the geometric requirements of speculative verification.

## Potentially Missed Related Work
- **DSIR** (Xie et al., 2023) & **IFD** (Li et al., 2024) — Modern gradient/likelihood-based data selection frameworks; contrasting SFDD against these would clarify whether SD-specific metrics outperform general-purpose importance sampling.
- **Token-scaled/adaptive distillation methods** (e.g., Kim et al., 2023; Zhou et al., 2025) — Recent works that dynamically weight tokens during KD based on loss or difficulty; discussing how flatness differs from adaptive reweighting would strengthen the methodological positioning.
- **Medusa/Lookahead SD architectures** (Cai et al., 2024; Zhang et al., 2024) — Non-EAGLE draft paradigms with distinct training dynamics; referencing these would contextualize the claim that SFDD is architecture-agnostic.

## Suggestions
- Replace or rigorously verify the "enhanced batching efficiency" claim in Section 5.4 with actual training profiling data, or remove it to maintain methodological precision.
- Add a compute-matched baseline training the full dataset for proportionally fewer epochs (or with validation-based early stopping) to isolate the data-quality signal from mere compute reduction.
- Report standard deviations or confidence intervals across multiple random seeds/repeated filtering runs in Tables 1–3 to establish statistical reliability.
- Provide generation quality metrics (e.g., exact match on GSM8K, MT-Bench judging scores) to verify that filtering does not degrade reasoning, factual accuracy, or instruction adherence.
- Include a controlled experiment or analysis explaining why cosine similarity to uniform outperforms Shannon entropy in predicting $\Delta L_1$ reduction, particularly under varying vocabulary sizes, temperature settings, or heavy-tailed logit distributions.

---

## cZFgsLq8Gs

- GT: Accept (Poster) (avg 4.0)
- Predicted: N/A (4.8/10)
- Match: N/A

### Final Review

## Summary
This paper introduces DeepScientist, an LLM-driven autonomous research system that formalizes scientific discovery as a goal-driven Bayesian optimization loop. By integrating a persistent `Findings Memory` with a three-stage iterative workflow (hypothesizing, implementing, and analyzing), the system conducts large-scale, parallelized exploration campaigns to redesign and validate AI methodologies. Empirically, it demonstrates the ability to surpass human-designed state-of-the-art methods on three frontier AI tasks while transparently quantifying exploration efficiency, scaling behavior, and systemic bottlenecks.

## Strengths
- **Ambitious empirical scale with transparent funnel reporting**: The work deploys ~20,000 GPU hours across real ML research problems rather than synthetic benchmarks, and explicitly publishes the exploration funnel (~5k ideas → ~1.1k validated → 21 SOTA-surpassing → 5 papers), providing crucial baseline statistics for the autonomous science community. (Evidence: Sec 4.3, Fig 4)
- **Iterative, memory-driven discovery architecture**: The system demonstrably learns from its own failures and successes, enabling progressive methodological refinement across discovery cycles (e.g., T-Detect → TDT → PA-TDT) rather than producing isolated, one-shot generations. (Evidence: Sec 3, Fig 5, Sec 4.1)
- **Rigorous multi-faceted output validation**: Generated papers are evaluated by both automated reviewers and a 3-person human program committee emulating ICLR standards. Achieving an average rating (~5.00) comparable to ICLR 2025 submissions, alongside fully detailed generated papers in the appendix, validates that the system produces scientifically coherent artifacts rather than engineering hacks. (Evidence: Table 3, Appendix F)

## Weaknesses
- **Lack of statistical rigor and variance reporting**: Main SOTA claims are presented as point estimates without error bars, confidence intervals, or multiple-seed runs, despite the high stochasticity inherent to LLM generation and implementation. Given ~1,100 experimental trials, the absence of false discovery rate control means marginal gains (e.g., the 1.9% token/sec improvement) cannot be reliably distinguished from benchmark noise or statistical overfitting. At ICLR, claims of "frontier-pushing" progress require rigorous variance quantification.
- **Superficial Bayesian optimization implementation**: The core BO loop relies on uncalibrated LLM surrogate scores and fixed UCB acquisition weights ($w_u=w_q=\kappa=1$) without sensitivity analysis, surrogate validation, or ablation. Critically, the framework updates the memory with empirical results from failed implementations, yet ~60% of failures stem from code generation/engineering errors rather than flawed scientific hypotheses. Failing to disentangle implementation noise from scientific invalidity corrupts the surrogate's feedback signal, undermining the theoretical optimization premise and reducing the method to heuristic prompt-guided search.
- **Reproducibility barriers and overstated autonomy claims**: The system's heavy reliance on proprietary, closed-model APIs (Gemini-2.5-Pro, Claude-4-Opus) and prohibitive compute costs (~$100k, 20k GPU hours) severely limit independent verification and academic accessibility. Furthermore, the methodology explicitly requires three human experts to supervise, filter hallucinations, and manually prepare baseline repositories, directly contradicting the "fully autonomous" framing in the abstract and complicating claims about closed-loop system capabilities.

## Nice-to-Haves
- Compare against a compute-matched random search baseline or standard cheap surrogates (e.g., Gaussian Process) under identical GPU budgets to empirically isolate the efficiency gains of the LLM-guided BO architecture.
- Release the complete, unfiltered `Findings Memory` (including all failed implementations, raw logs, and system prompts) to enable independent verification of the claimed "learning from failure" mechanism and facilitate community-driven ablation studies.
- Clarify and quantify the exact frequency, scope, and veto criteria of human supervision to accurately position the system as "human-supervised autonomous discovery" rather than claiming full closed-loop autonomy.

## Novel Insights
The paper transparently reveals that the primary bottleneck in current AI Scientist systems is not ideation capacity, but execution fidelity and verification overhead. With ~60% of computational budget consumed by implementation errors rather than scientifically invalid hypotheses, the work effectively shifts the research focus from "can AI generate ideas?" to "how do we efficiently filter, robustly implement, and scientifically validate them?" The near-linear scaling of discoveries with parallel compute, driven by shared memory synchronization, further suggests that future autonomous science will depend less on raw trial-and-error volume and more on knowledge-sharing architectures that accelerate collective exploration across asynchronous workers.

## Potentially Missed Related Work
- Work on robust surrogate modeling and failure-aware Bayesian optimization (e.g., handling noisy/censored observations in expensive black-box optimization) — highly relevant for designing feedback filters that distinguish engineering crashes from scientific dead ends.
- Neural Architecture Search (NAS) and AutoML literature on fidelity estimation and search space pruning — relevant for optimizing the exploration funnel and reducing the compute waste inherent in testing thousands of unverified hypotheses.

## Suggestions
- Report variance/error bars across multiple seeds for all main SOTA claims, and explicitly apply statistical correction for multiple hypothesis testing across the ~1,100 validated trials to establish which breakthroughs survive rigorous significance thresholds.
- Validate the surrogate calibration by reporting rank correlation (e.g., Spearman/Kendall) between the LLM-predicted utility/exploration scores and actual experimental outcomes, and conduct a sensitivity analysis on the UCB hyperparameters ($w_u, w_q, \kappa$) to demonstrate that the acquisition function meaningfully guides search rather than defaulting to arbitrary exploration.
- Add a compute-matched baseline comparison (e.g., random hypothesis sampling or simple heuristic search) under identical GPU-hour constraints to empirically prove that the BO-guided memory architecture, rather than brute-force scaling alone, drives the observed discovery efficiency.

---

## kMfVTka2WB

- GT: Reject (avg 2.0)
- Predicted: N/A (1.6/10)
- Match: N/A

### Final Review

## Summary
The paper proposes a Covariance-Adjusted SVM (CSVM) that uses per-class Cholesky decomposition to linearly transform data from the original feature space into a Euclidean space, arguing that standard max-margin principles fail in the original "statistical" space due to its covariance structure. It introduces the SM Algorithm, an iterative self-training procedure that pseudo-labels test data to refine sample covariance estimates. Experiments on five binary tabular datasets report performance gains over standard kernel SVMs and global whitening approaches.

## Strengths
- Clearly articulates a practical, geometrically grounded intuition: standard SVM margins treat all directions uniformly, whereas decorrelating and scaling each class separately aligns the optimization geometry with intra-class data dispersion.
- Provides a consistent, multi-metric empirical comparison (Accuracy, Precision, Recall, F1, AUC) across five diverse datasets, systematically benchmarking against linear kernels, common non-linear kernels, and PCA/ZCA whitening pipelines.

## Weaknesses
- **Fundamentally flawed theoretical framing and incorrect claims regarding KKT conditions.** The assertion that the input space is "Non-Euclidean" and that KKT boundary conditions are therefore invalid is mathematically incorrect. $\mathbb{R}^d$ equipped with a Mahalanobis metric is a Hilbert space isometric to Euclidean space, and the SVM optimization remains a convex QP. KKT conditions govern constraint optimization and are invariant to the metric terminology. The lemmas confuse a change of basis (whitening) with a breakdown of convex optimization theory, undermining the paper's theoretical foundation.
- **The SM Algorithm conflates metric transformation with transductive self-training, introducing severe data leakage and unquantified confirmation bias.** Step 2(g) merges pseudo-labeled test points into the training set to update covariance matrices. This is functionally equivalent to transductive self-training or EM refinement, not a novel population covariance estimator. It inherently leaks the unseen test distribution into the model, prevents true inductive deployment, lacks convergence guarantees, and introduces cascading error risk. No ablation separates the performance gains of the Cholesky transformation from the dataset augmentation effect.
- **Incomplete multi-class formulation and ambiguous decision rule.** Lemma 2.2 claims an $N$-class problem yields $N$ distinct classifiers in the input space, yet the paper never defines a decision rule to reconcile conflicting predictions or aggregate outputs across boundaries. The experimental section exclusively evaluates binary classification, leaving the practical application and mathematical coherence of the proposed $N$-classifier structure entirely unaddressed.
- **Limited empirical rigor and likely unfair baseline comparisons.** Evaluation relies on a single 80/20 split without cross-validation, multiple random seeds, variance reporting, or statistical significance testing. Baseline kernel SVMs (RBF, Sigmoid) show suspiciously degraded performance, strongly indicating inadequate hyperparameter tuning or missing feature standardization. Without rigorous tuning and variance metrics, the claimed improvements cannot be distinguished from methodological artifacts.
- **Numerical instability in ill-conditioned settings.** The method strictly requires invertible, class-wise Cholesky decomposition of sample covariance matrices. No regularization, diagonal loading, or pseudo-inverse fallback is provided. In realistic high-dimensional or collinear settings, sample covariances are frequently rank-deficient, causing the proposed transform to fail or produce unstable weight vectors.

## Nice-to-Haves
- Provide a formal derivation showing the mathematical equivalence (or lack thereof) between CSVM and standard linear SVM trained on Cholesky-whitened data with reversed weights.
- Visualize 2D decision boundaries and margin splits on synthetic data with controlled covariance structures to directly validate the "margin split ratio" claim.
- Include wall-clock training time and memory usage to quantify the computational trade-off introduced by iterative Cholesky decomposition and covariance updates.

## Novel Insights
The paper geometrically reframes standard data whitening as a necessary coordinate transformation to align SVM optimization with statistical dispersion, highlighting that per-class decorrelation implicitly shifts the decision boundary offset in the original space. While the theoretical packaging is unconventional and mathematically imprecise, the operational insight—separately aligning each class's covariance structure before margin maximization yields boundaries that better respect asymmetric class dispersion—offers a useful, if already implicitly known in whitening and elliptical classification literature, perspective on localized metric adaptation for binary classification.

## Potentially Missed Related Work
- Ellipsoidal/Quadratic SVM frameworks that natively incorporate class-specific scatter matrices into the primal/dual optimization without requiring pre-transformation.
- Label Propagation and Transductive SVM ($\psi$-SVM/TSVM) literature, which extensively formalizes iterative pseudo-labeling and self-training convergence.
- Regularized covariance estimators (e.g., Ledoit-Wolf shrinkage, graphical lasso) critical for stable Cholesky decomposition in finite-sample regimes.

## Suggestions
1. Abandon the "Non-Euclidean input space" and "KKT invalidity" narrative. Reframe the contribution as a Mahalanobis/pre-whitening transformation and rigorously derive how the decision boundary offset in the original space maps to the inverse-covariance ratio.
2. Formalize the SM algorithm within established transductive/semi-supervised learning theory. Compare it against standard TSVM or self-training baselines, implement a strict stopping criterion, and run an ablation isolating the transformation step from the pseudo-labeling loop.
3. Strengthen the experimental protocol: adopt 5-10 fold CV or $\ge$ 10 random seeds, report mean $\pm$ std, apply rigorous hyperparameter optimization to all baselines, and integrate covariance regularization (e.g., shrinkage) to ensure numerical stability across varying $n, d$ regimes.

---

## eETr3lrOQB

- GT: Accept (Poster) (avg 4.5)
- Predicted: N/A (4.8/10)
- Match: N/A

### Final Review

## Summary
The paper proposes **VQ-Transplant**, a two-stage framework that decouples vector quantization module development from costly full-tokenizer training by swapping new VQ algorithms into a frozen pre-trained encoder, followed by lightweight (5-epoch) decoder adaptation to align quantization-decoder distributions. Alongside this framework, the authors introduce **MMD-VQ**, leveraging Maximum Mean Discrepancy for non-parametric distribution alignment. The work claims near state-of-the-art reconstruction fidelity with up to 95% computational savings and extensive cross-dataset validation.

## Strengths
- **Addresses a genuine research bottleneck:** Tightly coupling VQ modules with encoder-decoder architectures in modern tokenizers imposes prohibitive compute and instability barriers. The proposed decoupled substitution + lightweight adaptation workflow directly lowers this barrier, enabling rapid, resource-efficient quantization research.
- **Systematic empirical validation across VQ variants and datasets:** The paper rigorously tests multiple VQ algorithms (Vanilla, EMA, Online, Wasserstein, MMD) under both multi-scale and fixed-scale configurations, consistently demonstrating that distribution-alignment methods preserve codebook utilization and minimize quantization error post-substitution. Cross-dataset evaluations (ImageNet, FFHQ, CelebA-HQ, LSUN) further validate generalization.
- **Effective and rapid mismatch mitigation:** The observation that simply substituting the VQ module causes reconstruction degradation due to decoder prior misalignment, and that this is reliably resolved within 5 epochs of decoder fine-tuning (with clear saturation curves), is a valuable practical insight for the community.
- **Sound theoretical grounding for MMD-VQ:** The motivation for replacing moment-matching (Wasserstein) with all-moment matching (MMD) is well-justified, and controlled synthetic experiments clearly demonstrate MMD-VQ's robustness under non-Gaussian latent distributions.

## Weaknesses
- **Inflated and methodologically misaligned efficiency claims:** The headline claim of 95% compute reduction/21.8× speedup compares baselines trained on OpenImages with 16× A100 GPUs against VQ-Transplant on ImageNet-1k with 2× A100s, making it an apples-to-oranges comparison. Table 6 shows that on the *same* setup, VQ-Transplant requires 22 hours versus 25–35 hours for a from-scratch baseline. Crucially, the from-scratch baselines are stopped at 5–7 epochs while discrete tokenizers typically require hundreds of epochs to converge. The paper does not quantify the true efficiency trade-off relative to fully converged baselines, significantly overstating the practical speedup.
- **Absence of downstream generative evaluation:** Despite framing the work around tokenizers for visual generation, the paper exclusively reports reconstruction metrics. Discrete tokenizers are evaluated primarily on the quality of images they enable downstream models (AR or diffusion) to generate. Without reporting generation FID/IS or performing downstream fine-tuning, it remains unclear whether the improved reconstruction fidelity translates to actual synthesis quality or if the adaptation introduces latent artifacts detrimental to sampling.
- **Marginal empirical advantage of MMD-VQ on real data:** The authors correctly note in Appendix B that real-world encoder features are approximately Gaussian due to the Central Limit Theorem, which neutralizes MMD-VQ's theoretical advantage over Wasserstein-VQ on standard benchmarks. The real-data performance gap is consistently marginal (<0.02–0.05 r-FID), raising questions about the practical necessity and computational overhead of MMD-VQ compared to simpler distribution-alignment alternatives.
- **Ambiguous Stage II adversarial formulation:** The paper states Stage II uses a "frozen DINO-S discriminator" while simultaneously claiming to incorporate DiffAug, LeCAM, and consistency regularization to "improve discriminator training." These regularizations are irrelevant for a frozen network, implying $L_{GAN}$ functions as a static feature-matching loss rather than a dynamic adversarial game. This contradiction obscures the actual optimization dynamics and complicates reproducibility.

## Nice-to-Haves
- Report standard deviations or multiple-run variance for r-FID and r-IS, given the sensitivity of adversarial fine-tuning to initialization and stochasticity.
- Provide direct quantitative evidence of latent distribution alignment (e.g., pre/post-adaptation MMD distances, PCA/t-SNE visualizations of codebook vs. feature embeddings) rather than inferring mismatch solely from rFID drops.
- Explicitly document dimensional/architectural constraints (e.g., the parallel quantization hack required to fit fixed-scale VQs into multi-scale frameworks) to set realistic expectations for "plug-and-play" compatibility with structurally divergent quantizers.
- Quantify the memory and wall-clock overhead of MMD kernel matrix computations at large codebook scales to guide practitioners on resource trade-offs.

## Novel Insights
The core contribution reframes tokenizer development from monolithic, expensive co-optimization to a modular pipeline where the bulk of adversarial alignment is shifted to the decoder. The systematic finding that only distribution-alignment VQ algorithms (Wasserstein, MMD) survive transplantation without codebook collapse establishes a clear design principle for future plug-and-play quantization research. By demonstrating that a converged tokenizer's decoder priors can be realigned to a new quantization space in a handful of epochs, the work opens a highly efficient iterative research loop: researchers can now benchmark novel VQ formulations at a fraction of the traditional cost, provided they pair them with appropriate distribution-matching objectives.

## Potentially Missed Related Work
None identified. The paper comprehensively cites recent advances in discrete tokenization, VQ algorithms, and adversarial adaptation up to 2024/2025.

## Suggestions
1. **Recalibrate efficiency benchmarks:** Replace the cross-dataset/hardware speedup claim with a controlled ablation plotting r-FID/IS vs. GPU-hours (or FLOPs) for VQ-Transplant vs. from-scratch training *until convergence*. Quantify the exact compute saved relative to a fully converged tokenizer to accurately represent the algorithmic efficiency gain.
2. **Add downstream generation experiments:** Fine-tune a standard autoregressive generator (e.g., VAR or a lightweight transformer) on the discrete tokens produced by the transplanted quantizers. Report generation metrics (gFID, gIS) to validate that reconstruction improvements meaningfully translate to synthesis quality, which is critical for ICLR-level impact.
3. **Clarify Stage II optimization dynamics:** Explicitly state whether the discriminator weights are entirely frozen or partially updated. If frozen, reframe the adversarial term as a fixed perceptual/distillation prior and remove references to discriminator-specific regularizations that are inapplicable. Provide training curves for Stage II to demonstrate adaptation stability.
4. **Contextualize MMD-VQ's practical utility:** Report the computational overhead (time/memory) of MMD distance calculations relative to Wasserstein-VQ, and provide a clear decision guideline for practitioners on when the theoretical robustness to non-Gaussianity justifies the added cost given the marginal gains on standard vision datasets.

---

## GRufFX1gAy

- GT: Accept (Poster) (avg 4.5)
- Predicted: N/A (5.1/10)
- Match: N/A

### Final Review

## Summary
InnoGym introduces a principled framework for evaluating AI agent innovation by formalizing two complementary metrics: Performance Gain ($G$) and Methodological Novelty ($N$). The paper presents iBench, a curated suite of 18 real-world "improvable" tasks, and iGym, a unified execution environment for long-horizon agent evaluation. Empirical results reveal that while modern agents can explore diverse methodological paths, implementation fragility prevents these novelties from translating into meaningful performance gains over human baselines.

## Strengths
- **Principled & Actionable Framework:** The decoupling of innovation into performance gain and methodological novelty, mapped against explicitly defined task regimes (breakthrough, performance, conceptual), directly addresses a critical blind spot in correctness-only benchmarks. The formalization $(P, S, V, D)$ provides a clear, extensible standard for future evaluation work.
- **Rigorous Benchmark Engineering:** The two-stage filtering pipeline and thorough standardization steps (validator construction, rank-to-absolute evaluator normalization with high correlation coefficients, reproducibility checks) demonstrate strong scholarly and engineering rigor. The curation focuses exclusively on "improvable" tasks where human SOTA is known but suboptimal, ensuring measurable headroom for innovation.
- **Compelling Empirical Diagnosis:** The systematic evaluation across three distinct agent scaffolds reveals a consistent and actionable finding: the primary bottleneck for current agents is not ideation capacity but execution robustness. This is well-supported by success rate breakdowns (Table 6) and bootstrap statistical analysis in the appendix, offering a sobering but highly useful reality check for the agentic community.

## Weaknesses
- **Novelty Metric Relies on Closed LLMs with Insufficient Validation:** The core novelty score $N$ hinges entirely on proprietary models (Codex for extraction, GPT-5 for judging). While the authors validate $D_{\text{AGENT}}$ against human judgments, the total sample size is only 11 triplets. This is statistically inadequate to guarantee reliability across highly diverse domains (from combinatorial optimization to temporal action localization) or to rule out judge biases (e.g., favoring verbose, deep learning-centric solutions over classical heuristics). Without broader validation or open-weight alternatives, the primary metric lacks community-level trust and reproducibility.
- **Main-Table Reporting Contradicts Robustness Claims:** Table 2 reports the *best* score across 3 runs, which inherently masks failure rates and performance variance. While Appendix E provides a rigorous pessimistic imputation and bootstrap analysis, presenting only peak performance in the main results directly undermines the paper's central thesis that agent *robustness* is the limiting factor. For ICLR, main-text tables must transparently reflect the stochastic reality of agentic systems.
- **Incomplete Main Evaluation Undermines Cross-Domain Claims:** Only 10 of the 18 curated tasks are evaluated in the primary experiments due to compute constraints. This leaves nearly half the benchmark untested and weakens the claim of broad, cross-domain applicability. Without results on the remaining 8 tasks (including Operations Research and classical optimization), the macro-average conclusions risk being domain-skewed.
- **Absence of Classical Baselines for Calibration:** The paper evaluates only LLM-based agent scaffolds without including traditional solvers (e.g., OR-Tools, CPLEX, or evolutionary algorithms). Without these baselines, it is impossible to calibrate whether the observed novelty scores reflect genuine methodological innovation or merely LLM-specific implementation artifacts, coding styles, or tool-calling quirks.

## Nice-to-Haves
- **Domain-Stratified Reporting:** Provide per-domain breakdowns (ML, OR, Math/Science) of $G$ and $N$ rather than a single macro-average. This would clarify whether the robustness-creativity gap is universal or domain-specific.
- **Qualitative Case Studies:** Include side-by-side comparisons of high-$N$/low-$G$ vs. high-$N$/high-$G$ solutions with their extracted `summary.md` outputs. This would ground the abstract novelty scores in tangible methodological differences and improve interpretability.
- **Open-Weight Novelty Pipeline:** Release a fully open-source evaluation script using publicly available LLMs to lower adoption barriers and enable community-led metric refinement.

## Novel Insights
InnoGym successfully reframes innovation not as a static property of correctness, but as a dynamic, frontier-shifting process quantified through the interplay of performance gain and methodological divergence. By explicitly defining "improvable" tasks—where human baselines exist but are suboptimal—the benchmark isolates the precise regime where AI agents should theoretically excel: iterative refinement and paradigm exploration. The empirical finding that current agents frequently traverse novel methodological paths but fail to execute them robustly establishes a critical paradigm shift for the field: the bottleneck in automated scientific and engineering discovery is no longer idea generation, but implementation reliability and long-horizon constraint satisfaction. This insight redirects community focus from prompting for originality toward engineering for execution stability.

## Potentially Missed Related Work
- Lehman & Stanley (2008) on Novelty Search in evolutionary computation — highly relevant for contextualizing the exploration-exploitation trade-off observed in Sec 4.3 and provides classical grounding for novelty-driven optimization without fitness pressure.
- DSPy-style optimization or automated code evolution frameworks — could strengthen the discussion on how current agent search dynamics compare to purely algorithmic code mutation strategies when optimizing for $G$ vs $N$.

## Suggestions
1. **Restructure Main Table 2:** Replace "best-of-3" scores with mean performance ± standard deviation alongside success rates. Move best-case results to an appendix if desired, ensuring the main text aligns with the paper's robustness thesis.
2. **Expand Validation Rigor for $D_{\text{AGENT}}$:** Conduct a bias sensitivity analysis by swapping the judge model (e.g., to open-weight 70B+ models) and testing on a broader, domain-stratified set of triplets (~50+). Report any score drift to quantify judge-induced variance.
3. **Clarify Scope in Main Text:** Explicitly frame the 10-task evaluation as a preliminary, compute-constrained cohort in Section 4, and provide statistical justification (e.g., domain coverage metrics) that this subset reliably proxies the full 18-task distribution.
4. **Add One Classical Baseline:** Include a high-temperature independent sampling baseline or a classical solver on 2-3 representative tasks to calibrate the novelty metric and demonstrate that high $N$ scores correlate with genuinely distinct algorithmic strategies, not LLM formatting artifacts.

---

## iIEEgI6WsF

- GT: Accept (Poster) (avg 5.5)
- Predicted: N/A (6.1/10)
- Match: N/A

### Final Review

## Summary
This paper proposes On-Demand Communication (ODC), a decentralized parameter server paradigm adapted for Fully Sharded Data Parallel (FSDP) training. By replacing per-layer collective `all-gather` and `reduce-scatter` operations with non-blocking point-to-point RDMA/CUDA-IPC transfers, ODC relaxes synchronization granularity from the layer level to the minibatch level, mitigating straggler effects caused by highly variable sequence lengths. The approach unlocks simplified minibatch-level load balancing and reports consistent throughput improvements across supervised fine-tuning and reinforcement learning workloads.

## Strengths
- **Precise problem identification:** The paper clearly articulates a fundamental mismatch in modern distributed training: per-layer collective barriers inherently assume balanced workloads, an assumption violated by the heavy-tailed sequence length distributions prevalent in LLM post-training.
- **Clean, practical system integration:** ODC successfully reframes FSDP as a decentralized parameter server while preserving the exact memory layout and synchronous optimization semantics. The implementation leverages native one-sided RDMA primitives and provides an open-source code patch, lowering the barrier to adoption.
- **Comprehensive empirical scope:** Evaluations span multiple model scales (1.5B–32B), diverse datasets, and distinct training paradigms (SFT, RL), supported by a rigorous parametric study and empirical convergence verification against the collective baseline.

## Weaknesses
- **Conflated attribution of performance gains** — The most significant speedups (up to 36%) are achieved using `LB-Mini`, a minibatch-level packing algorithm that is strictly enabled by ODC's relaxed synchronization. The paper does not disentangle how much improvement stems from the point-to-point communication decoupling versus the superior algorithmic scheduling flexibility, obscuring the true isolated systems contribution.
- **Unsubstantiated scalability at production scale** — Experiments cap at 32 GPUs, yet Figure 11 explicitly demonstrates that ODC's point-to-point RDMA transfers significantly underperform NCCL's hierarchical collectives across nodes. Without empirical validation at 64+ devices or analytical modeling, the claim that ODC is broadly superior for large-scale training remains speculative, as decentralized P2P traffic inherently forgoes topology-aware routing and exacerbates switch contention.
- **Reliance on proxy straggler metrics** — The reported "bubble rates" in Appendix G are algorithmically estimated by the packing solver rather than measured via hardware profilers (e.g., Nsight Systems GPU idle timestamps). This fails to rigorously prove that communication decoupling translates to actual reduced hardware stalls, weakening the core empirical motivation for replacing collectives.

## Nice-to-Haves
- Explicit discussion of floating-point non-associativity in arbitrary-order P2P gradient accumulation and its implications for mixed-precision training reproducibility.
- Inclusion of actor rollout time in RL throughput metrics to better reflect end-to-end pipeline efficiency rather than isolating only the training backward pass.

## Novel Insights
The paper successfully reframes a rigid systems constraint as a tunable design parameter: it demonstrates that fine-grained synchronization is not a mathematical requirement of data parallelism, but an artifact inherited from workload-balanced training regimes. By decoupling synchronization boundaries to the minibatch level, ODC reveals that communication topology and load balancing granularity are inextricably linked. Relaxing synchronization does not merely hide latency; it fundamentally expands the feasible search space for data partitioning, enabling coarser packing strategies that inherently resolve straggler effects more robustly than microbatch-level heuristics. This shifts the performance bottleneck from communication coordination to pure compute heterogeneity, offering a principled pathway for designing communication schemes tailored to skewed, real-world data distributions.

## Potentially Missed Related Work
- Topology-aware collective decomposition and flexible routing strategies (e.g., recent NCCL hierarchical aggregations, SHARP offloading, or dynamic bucketized reduces) — relevant for contextualizing ODC’s inter-node bandwidth trade-offs and exploring hybrid routing that could mitigate P2P contention at scale.
- Bounded-staleness and flexible data parallelism frameworks that explore asynchronous or partially synchronized gradient updates — relevant for positioning ODC’s strict minibatch boundary against broader straggler-tolerant DP paradigms and understanding convergence trade-offs.

## Suggestions
- **Strict communication-vs-algorithm ablation:** Compare ODC and Collective under *identical* load-balancing constraints (or adapt `LB-Mini` conceptually for the collective baseline) to isolate the pure bandwidth/latency benefit of point-to-point decoupling from the scheduling advantage.
- **Hardware-level profiling validation:** Replace algorithm-estimated bubble rates with Nsight Systems timelines or NCCL profiling traces. Directly measure GPU SM occupancy, idle/wait cycles, and compute-communication overlap ratios across devices to provide empirical, hardware-grounded proof of straggler elimination.
- **Scalability & topology modeling:** Conduct multi-node experiments at 64–128+ devices or provide a rigorous bandwidth model projecting ODC's crossover point against NCCL. Analyze how uncoordinated P2P RDMA bursts interact with modern fat-tree NIC topologies and propose mitigation strategies (e.g., hierarchical caching, rate-limiting, or adaptive routing) to ensure viability beyond single-node-adjacent clusters.

---

## 41JeFWdVFa

- GT: Reject (avg 4.7)
- Predicted: N/A (4.6/10)
- Match: N/A

### Final Review

## Summary
This paper proposes LDP, a lightweight denoising autoencoder plug-in that enforces low-resolution (LR) cyclic consistency to improve the generalization of single-image super-resolution (SISR) models to unseen degradations. By conditioning a patch-dependent noise schedule on LR high-frequency residuals, LDP operates in two modes: as a fine-tuning regularization loss or as a gradient guidance term for diffusion posterior sampling. The method is evaluated across CNN, Transformer, State-Space, and diffusion backbones, reporting consistent PSNR/SSIM gains on synthetic and real-world benchmarks with minimal parameter overhead.

## Strengths
- **Pragmatic, architecture-agnostic design validated across diverse backbones.** LDP seamlessly integrates into deterministic (SwinIR, MambaIR), GAN-based (FeMaSR), and diffusion models (StableSR) without architectural overhaul, demonstrating consistent LR-cyclic consistency improvements in both training regularization and inference guidance modes.
- **Rigorous LR prediction validation and computational efficiency.** Tables 1–2 convincingly demonstrate that LDP learns non-trivial degradation mappings rather than collapsing to bicubic downsampling, outperforming heavier alternatives (DRN, DualSR) while introducing only ~642K parameters and requiring substantially less training memory/time than competing plug-ins like Lway.
- **Comprehensive experimental scaffolding and reproducibility.** The paper covers five synthetic degradation types, three real-world datasets, and multiple reference/no-reference metrics. Detailed hyperparameters, dataset generation pipelines, and open-source code strongly support independent replication.

## Weaknesses
- **Lack of statistical rigor and missing standard baselines for posterior sampling.** All reported gains (e.g., +0.3 to +0.7 dB PSNR) are single-run without standard deviations or multi-seed averages, making it impossible to assess whether improvements exceed initialization variance. More critically, the diffusion posterior sampling experiments (Table 5) evaluate LDP in isolation without comparing against established LR-consistency gradients like standard DPS or DR2. Without these baselines, the marginal utility of LDP's specific guidance formulation remains empirically unsubstantiated.
- **Unverified theoretical premise and brittle conditioning dependency.** The core mechanism relies on the assumption that patch-wise high-noise schedules align HR and LR features, yet this is justified solely by citing an external work without verifying the alignment under the authors' specific noise regime (e.g., via feature-space similarity metrics). Furthermore, the method strictly requires paired LR inputs at inference to compute the high-frequency conditioning signal ($y_{hf}$) and the cyclic loss gradient. This fundamentally restricts LDP to supervised/semi-blind settings, directly contradicting the broader implication of zero-shot or purely blind generalization.
- **Inconsistent perceptual trade-offs and hyperparameter fragility.** Several no-reference metrics (CLIPIQA, MANIQA) consistently degrade after fine-tuning, particularly for GAN-based models. The authors dismiss this as "metric artifact" without quantitative distortion-perception analysis or user studies to confirm whether LDP genuinely improves perceptual fidelity. Additionally, the claimed "plug-and-play" nature is undermined by highly variable, hand-tuned loss weights ($\lambda \in [0.1, 10]$, $\tau \in [1, 100]$) across different backbones (Appendix D), suggesting the method's performance is tightly coupled to architecture-specific fine-tuning rather than universal applicability.

## Nice-to-Haves
- Visualize spatial gradient maps of $\nabla_{x_t} L_{sym}$ during diffusion sampling to demonstrate whether guidance provides localized structural corrections or uniform suppression.
- Report inference wall-clock times under standard acceleration (FP16, kernel fusion) to contextualize the ~47% overhead in Table 13 for practical deployment.
- Provide an explicit failure case analysis where cyclic regularization oversmooths textures or conflicts with strong generative priors, clarifying operational boundaries.

## Novel Insights
The paper reframes classical cyclic consistency from a rigid pixel-space constraint into a feature-aligned degradation process. By treating LR high-frequency residuals as a conditioning signal for patch-dependent noise scheduling, LDP effectively decouples degradation modeling from the primary SR backbone, demonstrating that lightweight, modular regularization objectives can suppress hallucination artifacts across deterministic and generative pipelines alike. This suggests a promising paradigm shift in SISR: rather than redesigning architectures for robustness, degradation-aware constraints can be injected as auxiliary optimization signals that actively narrow the solution space toward LR-consistent reconstructions.

## Potentially Missed Related Work
- **DPS (Chung et al., 2023) & DR2 (Wang et al., 2023b)** — Already cited but must serve as explicit empirical baselines in Table 5 to isolate LDP's marginal guidance gain over standard diffusion posterior sampling.
- **Cold Diffusion (Bansal et al., 2022)** — Relevant for contextualizing arbitrary transform inversion without strict reliance on high-noise alignment assumptions, which could strengthen the theoretical discussion of the DAE scheduling.

## Suggestions
- Report results averaged over at least 3 random seeds with standard deviations for Tables 3–5 to validate that modest metric improvements are statistically meaningful.
- Include standard DPS and DR2 as direct baselines in the diffusion posterior sampling experiments (Table 5) to rigorously isolate the effectiveness of LDP's gradient formulation.
- Conduct a controlled ablation of the $y_{hf}$ conditioning signal (e.g., zeroed, shuffled, or replaced with uniform noise) to rule out shortcut learning and verify that the DAE genuinely models spatially varying degradation rather than trivially regressing from the provided residuals.
- Standardize loss weights across backbones or introduce a principled, architecture-agnostic heuristic for $\lambda$ and $\tau$ to substantiate the "plug-and-play" claim.
- Supplement the qualitative dismissal of dropping no-reference metrics with a quantitative distortion-perception trade-off analysis (e.g., calibrated FR-IQA vs. NR-IQA correlation or pairwise preference scoring) to objectively validate perceptual improvements.

---

## 2EQPpEZtEK

- GT: Reject (avg 3.3)
- Predicted: N/A (5.4/10)
- Match: N/A

### Final Review

## Summary
DISTAR proposes a zero-shot text-to-speech framework that operates entirely in a discrete residual vector quantization (RVQ) code space, coupling a causal AR language model for inter-patch sketching with a bidirectional masked diffusion transformer for intra-patch infilling. The architecture eliminates explicit duration predictors, introduces RVQ-aware decoding heuristics to stabilize parallel sampling, and supports test-time bitrate/compute control via stochastic layer pruning. Evaluations on English-only benchmarks demonstrate competitive intelligibility and subjective naturalness against state-of-the-art continuous and discrete baselines.

## Strengths
- **Pragmatic hybrid discrete architecture:** By decomposing generation into AR planning and discrete masked diffusion, the model inherits the stability, explicit termination tokens, and interpretable decoding of discrete language models while recovering parallel intra-patch refinement. This directly mitigates the compounding exposure bias typical of purely autoregressive TTS.
- **Effective inference controllability engineering:** The stochastic layer truncation strategy successfully decouples synthesis quality from RVQ depth, enabling flexible test-time bitrate and compute scaling without retraining. Furthermore, the lightweight layer/time temperature shaping and hybrid sampling schedules directly address a observed confidence bias, yielding robust greedy decoding options.
- **Comprehensive empirical grounding:** The paper benchmarks against a strong, contemporary set of continuous and discrete systems using both objective (WER, SIM, UTMOS) and subjective (CMOS, SMOS) metrics. Systematic ablations on CFG configuration and patch size provide transparent evidence supporting core architectural choices.

## Weaknesses
- **Unsubstantiated computational efficiency claims:** The paper asserts "comparable or lower computational cost" and evaluates at NFE=24, yet directly compares against baselines running at NFE=10 (DiTAR) and NFE=32 (E2TTS/F5TTS) with zero reported wall-clock latency, Real-Time Factor (RTF), or throughput measurements. Without latency benchmarks normalized to hardware and audio duration, claims of parallel efficiency and computational parity are empirically unsupported.
- **Missing architectural decomposition ablations:** To validate the necessity of the hybrid design, the paper omits comparisons against a strong, capacity-matched pure AR codec-LM baseline and a standalone masked diffusion model trained on identical RVQ codes. Consequently, it is impossible to determine whether reported performance gains stem from the AR-diffusion coupling or simply from architectural scaling and increased parameter capacity.
- **Misaligned claims regarding consistency and generalization:** The abstract emphasizes superiority in "speaker/style consistency," yet objective similarity (SIM) on LibriSpeech (0.67 for the medium model) trails strong flow-matching baselines (0.70 for E2TTS), and the model is trained exclusively on ~50k hours of clean English speech. The paper lacks cross-lingual, accented, or highly emotional/expressive evaluations, failing to validate broad zero-shot robustness claims and ignoring the known ceiling of RVQ discretization for fine-grained prosodic variation.

## Nice-to-Haves
- Clarify the conditioning interface and optimization dynamics: Explicitly state whether the AR module outputs discrete tokens or continuous hidden states for the diffusion conditioner, and reconcile the differing learning rates with the "end-to-end optimization" narrative.
- Provide sequence-length scaling experiments (e.g., 10s vs. 30s+ utterances) to directly evidence the claim that patch-level diffusion mitigates long-form timbral drift and exposure bias accumulation.
- Visualize per-step diffusion confidence scores across temporal offsets and RVQ depths to empirically confirm that the masked refiner resolves dependencies bidirectionally rather than devolving into copy-from-context behavior.

## Novel Insights
The most compelling contribution of this work lies in its disciplined return to discrete token spaces for speech synthesis. By performing iterative masked demasking directly over RVQ codes, DISTAR sidesteps the gradient instability and mode collapse frequently encountered in high-dimensional continuous flow-matching, while naturally respecting the Cartesian product structure of intra-frame depth without resorting to flattening or delay patterns. The identification and mitigation of the "tail-first" decoding bias through structured temperature shaping reveals a previously underreported pathology in parallel discrete refinement, demonstrating that non-autoregressive infilling on speech codes requires explicit positional and depth-wise confidence calibration to maintain acoustic fidelity and prevent left-to-right shortcutting.

## Potentially Missed Related Work
- **VALL-E 2 (Chen et al., 2024) & SoundStorm (Google, 2023)** — Highly relevant discrete and hybrid TTS systems that employ parallel/non-autoregressive token prediction strategies. Discussing these would better contextualize DISTAR’s position within the discrete speech generation literature and clarify its architectural distinctions.
- **Any-Order Autoregressive / Structured Discrete Diffusion frameworks** — Recent theoretical extensions connecting masked diffusion to any-order autoregression (e.g., Ou et al., 2024; Shih et al., 2022) could strengthen the formal grounding of the iterative demasking process applied to the hierarchical RVQ domain.

## Suggestions
- Rigorously benchmark wall-clock latency, RTF, and peak memory usage against F5TTS and DiTAR, ideally running all systems at matched NFE steps, to substantiate computational efficiency claims and enable fair throughput comparisons.
- Introduce strong, matched-capacity baselines for a pure AR codec-LM and a pure masked diffusion refiner to isolate the marginal contribution of the hybrid drafting mechanism.
- Align the abstract's narrative strictly with empirical findings: acknowledge that while subjective naturalness (CMOS/SMOS) and intelligibility (WER) lead, objective speaker similarity (SIM) remains competitive rather than dominant, and explicitly narrow the scope to English TTS in the main text rather than deferring this limitation to an appendix.

---

## NFB4QGGS65

- GT: Accept (Poster) (avg 6.5)
- Predicted: N/A (6.9/10)
- Match: N/A

### Final Review

## Summary
This paper provides a rigorous theoretical reinterpretation of the GPTQ post-training quantization algorithm, proving that when executed in reverse dimensional order, it is mathematically identical to Babai’s nearest plane algorithm for the Closest Vector Problem (CVP) on a lattice defined by the layer's activation Hessian. This geometric equivalence yields a tight, worst-case error bound for a no-clipping quantization regime, which the authors leverage to design two practical variants (SSQR and HPTQ) and an optimized CUDA inference kernel for SSQR. Empirical evaluations across multiple Qwen3 and Llama model families demonstrate that these methods achieve strong perplexity and zero-shot accuracy, often outperforming standard GPTQ and RTN at comparable effective bitwidths.

## Strengths
- **Rigorous theoretical mapping:** The paper cleanly proves the algebraic and geometric equivalence between GPTQ’s error propagation and Babai’s orthogonal projections (Theorems 2 & 4), demystifying a widely-used heuristic and anchoring it in established lattice theory.
- **Direct theory-to-practice translation:** The derived worst-case error bound (Theorem 5) is explicitly tied to the trace of the LDL decomposition's diagonal matrix, leading to principled quantization order heuristics (`min-pivot`) and informing the design of no-clipping quantization schemes (SSQR, HPTQ).
- **Strong empirical validation & systems contribution:** The proposed methods are benchmarked comprehensively across model scales and datasets, competing with or surpassing strong baselines. The inclusion of a custom SSQR CUDA kernel (~2× speedup over BF16) and a public codebase demonstrates a commitment to reproducible, systems-aware research.

## Weaknesses
- **Deployment gap for the no-clipping regime and HPTQ:** The core theoretical guarantees strictly require Z† = Z (unconstrained integer grids). While SSQR and HPTQ circumvent clipping via outlier buffering and Huffman coding, the practical viability of HPTQ is critically undermined by the absence of a decoding kernel or latency analysis. Variable-length Huffman codes induce irregular memory access patterns and decoding overhead that fundamentally clash with high-throughput GPU tensor-core pipelines. Without quantifying this overhead, the reported compression-accuracy Pareto front is overly optimistic and insufficient to claim practical deployment readiness.
- **Unjustified computational cost of the principled ordering heuristic:** The `min-pivot` ordering minimizes the theoretical error bound but incurs $O(c^3)$ preprocessing overhead, compared to $O(c)$ for the standard `act-order`. The paper acknowledges the marginal empirical gains but provides no wall-clock timing comparisons or cost-benefit analysis. For billion-parameter models with large layer dimensions, this cubic preprocessing cost lacks practical justification and risks making the theoretical ordering strategy computationally prohibitive.
- **Under-specified positioning relative to concurrent work:** Footnote 1 mentions a concurrent preprint (Birnick, 2025) establishing a similar equivalence. The manuscript does not explicitly delineate how its complete algebraic proofs, derived error bounds, novel algorithmic designs, and system implementations extend or differ from this independent finding. Clearer scoping is required to robustly establish priority and novelty for an ICLR audience.

## Nice-to-Haves
- Include an empirical scatter plot comparing the theoretical worst-case layer-wise error bound against actually observed quantization errors across transformer blocks to validate bound tightness in practice.
- Provide a brief sensitivity analysis showing how the derived bounds and proposed methods degrade under limited calibration data or out-of-distribution activation shifts.
- Extend the theoretical discussion to explicitly address how standard per-group clipping constraints or scale-aware basis reductions could be formally integrated into the framework for fixed-width hardware formats.

## Novel Insights
The paper successfully bridges classical lattice algorithms with modern LLM compression, revealing that GPTQ’s greedy error correction is not merely an algebraic convenience but a sequential orthogonal projection through a Hessian-defined activation lattice. This reframing demystifies why GPTQ’s local updates yield globally robust approximations and provides a formal geometric lens to analyze quantization order, error propagation, and worst-case guarantees. By casting weight quantization as an instance of CVP, the work opens a principled, two-way channel: decades of lattice optimization heuristics can systematically refine future quantizers, while the statistical structure of LLM weights may inspire new, data-informed variants of closest-vector algorithms.

## Potentially Missed Related Work
- None identified. The literature review appropriately covers foundational PTQ methods, lattice/CVP theory, and recent low-bit baselines (AQLM, QuIP#, QTIP), with adequate citation of independent concurrent findings.

## Suggestions
- Develop or simulate an HPTQ decoding latency and memory bandwidth profile to quantify the real-world inference overhead of variable-bitwidth Huffman codes during matrix multiplication, ensuring accuracy-compression gains translate to actual hardware efficiency.
- Add a concise comparative analysis explicitly contrasting this work’s scope, proofs, and contributions with Birnick (2025) in the related work or introduction to firmly establish the paper’s distinct novelty and priority.
- Benchmark and report the wall-clock calibration overhead for `min-pivot` versus `act-order` on standard LLM layer shapes, or propose/evaluate a sub-cubic approximation to justify its adoption in large-scale quantization pipelines.

---

## crKJJ4Ej60

- GT: Accept (Poster) (avg 5.3)
- Predicted: N/A (3.9/10)
- Match: N/A

### Final Review

## Summary
This paper introduces the **Copy-Paste** paradigm, a two-stage framework designed to mitigate contextual faithfulness hallucinations in Retrieval-Augmented Generation (RAG). Stage 1 employs constrained prompting methods to generate high-copying responses, while Stage 2 leverages Direct Preference Optimization (DPO) to internalize this preference, producing `CopyPasteLLM`. Trained on only 365 samples, the method reports substantial accuracy gains (12.2%–24.5%) on counterfactual benchmarks like FaithEval. The authors additionally propose the **Context-Parameter Copying Capturing** algorithm to mechanistically analyze knowledge reliance, concluding that effectiveness stems from suppressing parametric confidence rather than enhancing contextual representations.

## Strengths
- **Highly Efficient Two-Stage Pipeline with Strong Empirical Gains:** The transition from heuristic copying constraints to DPO alignment is logically structured and empirically validated. Achieving significant accuracy improvements on challenging counterfactual benchmarks (FaithEval, ConFiQA-MC) with only 365 training samples is a compelling demonstration of data-efficient contextual alignment, especially when compared to baselines requiring 10k–32k samples.
- **Comprehensive Component Validation:** The ablation studies (Appendix G) effectively isolate the contributions of the high-copying preference signal and the "answer stamping" procedure. Showing that performance collapses to baseline levels without either component strongly supports the core architectural design choices.

## Weaknesses
- **Unsubstantiated Data Efficiency Claims Due to Missing Calibrated Baselines:** The central claim that `CopyPasteLLM` is uniquely data-efficient rests entirely on comparing it against models trained on 18,000 to 32,580 samples (Table 1). DPO and SFT methods are known to follow different learning curves; without demonstrating that `Context-DPO`, `Canoe`, or standard SFT baselines plateau or fail when trained on the identical 365 high-copying samples, the purported efficiency advantage remains unverified. It is possible that DPO alone, or any preference method, would converge rapidly on such a constrained, high-signal subset.
- **Ground-Truth "Answer Stamping" Undermines Real-World Applicability:** The automated preference construction pipeline (Algorithm 2) explicitly relies on appending gold answers to chosen responses and incorrect answers to rejected ones. The ablation in Figure 12 confirms that removing this stamping step causes performance to collapse. This heavy reliance on labeled data fundamentally contradicts the promise of a fully automated, unsupervised RAG pipeline, as gold answers are rarely available in real-world retrieval scenarios.
- **Methodological Artifact in Mechanistic Analysis Weakens Core Interpretability Claims:** The `Context-Parameter Copying Capturing` algorithm (Algorithm 4, Line 10) intentionally discards tokens that appear in both context-conditioned and context-free generations (`S_com`). This design choice systematically biases the analysis toward domain-unique terminology, artificially inflating the apparent separation between "contextual" and "parametric" knowledge representations in Figure 4 (UMAP) and Figure 3 (logits). The conclusion that the model "recalibrates internal confidence" is therefore based on a filtered proxy that may simply be masking shared linguistic priors, making the mechanistic insight correlational rather than rigorously causal.
- **Lack of Statistical Variance on Primary Results:** Given the extremely small training footprint ($N=365$) and the known sensitivity of DPO to preference noise and initialization, reporting single-point accuracies in Tables 1 and 3 is insufficient for ICLR standards. The absence of mean ± standard deviation or confidence intervals across multiple seeds makes it impossible to assess the reliability and reproducibility of the highlighted 12–24% performance gains.

## Nice-to-Haves
- **Causal Mechanistic Probing:** While the logit power and UMAP analyses are descriptive, incorporating causal interventions such as activation patching, logit lens attributions, or attention head ablation would rigorously validate the hypothesis that performance gains stem from parametric suppression rather than mere attention reallocation.
- **Human Evaluation for Fluency-Coherence Trade-off:** As the method inherently optimizes for lexical overlap, a human evaluation focusing on discourse coherence, readability, and semantic faithfulness (rather than string-match metrics) would clarify the practical limits of the framework, particularly on multi-hop or synthesis-heavy queries.

## Novel Insights
The paper effectively reframes "contextual faithfulness" from an abstract semantic alignment problem into an explicit, tractable optimization of lexical reuse. The mechanistic hypothesis that enforcing high-copying behavior functions as an "anchor" to reduce generative entropy and suppress competing parametric pathways offers a compelling theoretical lens for why extractive constraints outperform unconstrained abstractive generation in knowledge-conflict scenarios. This shifts the narrative of hallucination mitigation from purely architectural changes to explicit behavioral conditioning during the preference learning phase.

## Potentially Missed Related Work
None identified. The literature review effectively covers recent faithfulness tuning methods (Context-DPO, Canoe, ParamMute, CoCoLex), citation-based generation, and mechanistic analyses of knowledge conflicts (KTC, Retrieval Heads).

## Suggestions
- **Add Calibrated Efficiency Comparisons:** Train strong baselines (Context-DPO, SFT, Canoe) on the exact same 365 high-copying samples used for `CopyPasteLLM`. This will isolate whether the observed gains are due to the copying signal itself or simply an artifact of DPO's convergence dynamics on highly filtered data.
- **Address the Label-Dependency Bottleneck:** Propose and evaluate a fully unsupervised variant of the preference pipeline. For example, replace "answer stamping" with self-consistency scoring, judge-based confidence thresholds, or retrieval-based relevance verification to demonstrate utility in label-scarce environments.
- **Validate or Revise the Mechanistic Proxy:** Conduct a control experiment using the Capturing algorithm *without* the `S_com` discard rule, or explicitly quantify how filtering overlapping tokens affects the UMAP separation metrics. If the separation vanishes without filtering, the mechanistic claims must be substantially toned down.
- **Report Statistical Significance:** Rerun primary evaluations (Tables 1 and 3) across multiple data sampling seeds (e.g., 5–8 runs) and report mean accuracy with standard deviations. This is essential to confirm that the gains from 365 samples are robust and not stochastic outliers.

---


# Summary

Papers: 40 | Accuracy: N/A
