=== CALIBRATION EXAMPLE 65 ===

# Harsh Critic Review
## Section-by-Section Critical Review

**Title & Abstract:** The title accurately reflects the paper's core contribution: inductive reasoning for TKGs with emerging entities. The abstract clearly states the problem (closed-world assumption fails for emerging entities), the key insight (semantically similar entities have transferable patterns), and the proposed solution (TRANSFIR). The claimed improvement (28.6% avg. MRR) is supported in the results. The abstract is well-written and meets ICLR standards.

**Introduction & Motivation:** The introduction effectively motivates the problem using real-world examples (e.g., new users, Barack Obama). It clearly differentiates the task from prior work on static KG induction and extrapolative TKG reasoning. The formal problem definition in Section 2 is precise. The three-perspective empirical study (Sec. 3) is a major strength, providing concrete, data-driven evidence for the problem's prevalence (Q1: ~25% emerging entities), its cause (Q2: representation collapse), and the feasibility of a solution (Q3: transferable patterns). This section is excellent and frames the paper's contribution convincingly.

**Methodology (Sec. 4):** The three-stage pipeline (Classification–Representation–Generalization) is logically structured. However, several aspects require deeper scrutiny:
1.  **Codebook Mapping & Frozen Textual Embeddings:** The method's core premise is that entities with similar *semantic types* share patterns, and these types are inferred from frozen textual embeddings (e.g., from BERT). This is a strong assumption. The paper notes in Sec. 5.4 that performance on GDELT can drop without textual encoding due to noisy titles, and a failure case in Appendix F.1 is attributed to insufficient semantic information. This is a **significant limitation** that should be discussed more prominently. The approach may struggle with entities having uninformative, ambiguous, or highly context-dependent names (common in real-world TKGs). The choice of a *frozen* embedding is justified for stability but precludes fine-tuning for the task.
2.  **Interaction Chain (IC) Construction:** The TopK selection based on query-relation similarity (Eq. 6) is pragmatic to focus the context but is an architectural choice that could filter out chronologically important but relationally dissimilar events. An ablation on this selection mechanism (vs. using all events in the window) would strengthen the argument.
3.  **Pattern Transfer Mechanism:** The transfer operation `h̃_e = h_e + ω_e · c^{dyn}_{π(e)}` (Eq. 11) is essentially a cluster-conditioned residual update. While simple, the rationale for this specific form is not deeply explored. Why is additive modulation the right inductive bias? A brief discussion or comparison to alternatives (e.g., concatenation + non-linear projection) would be helpful.
4.  **Training Dynamics:** The model trains `L_lp` and `L_codebook` simultaneously. It is unclear if there is a risk of the codebook learning to cluster entities based on temporal interaction patterns (the target task) rather than pure semantic similarity from text, which could lead to circular reasoning. Some analysis of the learned codewords (beyond the three examples in Fig. 4b) would be insightful.

**Experiments & Results (Sec. 5):**
- **RQ1 (Performance):** The experimental setup is rigorous. Using a 5:2:3 chronological split to induce more emerging entities is appropriate. The comparison across three categories of baselines (graph-based, path-based, inductive) is comprehensive. The results are impressive and clearly demonstrate TRANSFIR's superiority on the specific task of reasoning for *emerging entities*. The consistent gains across datasets are convincing.
- **RQ2 (Analysis):** The visualization and collapse ratio analysis effectively show that TRANSFIR mitigates representation collapse. The case study is illustrative. However, the claim that clusters are "semantically coherent" is based on a few hand-picked examples (Fig. 4b). A more systematic analysis (e.g., clustering purity metrics w.r.t. some external entity type taxonomy, if available) would provide stronger evidence.
- **RQ3 (Ablation):** The ablation study is thorough and shows each component's contribution. The note about GDELT and textual encoding is honest and important.
- **RQ4 (Generalization):** The experiments on the "Unknown" setting (Fig. 10) and varying temporal splits (Fig. 11) are excellent additions that convincingly show the method's robustness. The hyperparameter sensitivity analysis is adequate. The efficiency analysis (Fig. 7) is a plus.

**Writing & Clarity:** The paper is generally well-written and easy to follow. Figures are informative. Some minor points: The reference to "LogCL" in Fig. 2(c) caption appears before the model is introduced. The pseudocode (Algorithm 1) is helpful. The flow from empirical study to method to experiments is logical.

**Limitations & Broader Impact:** The limitations are partly acknowledged in the failure case analysis and the GDELT textual encoding observation. However, the discussion should be expanded in the main text or a dedicated limitations section. Key limitations include: (1) Heavy dependence on the quality and informativeness of entity names/textual descriptions. (2) Assumption that semantic similarity (from names) correlates with temporal interaction patterns—this may not always hold. (3) The method does not address emerging *relations*, as noted in the conclusion. The broader impact statement is standard and appropriate. The reproducibility statement is strong with public code.

### Overall Assessment
This is a strong paper that addresses a well-motivated and underexplored problem in TKG reasoning. The empirical study is compelling, the proposed method is novel, and the experimental evaluation is extensive and demonstrates clear state-of-the-art performance on the specific task. The main concerns revolve around the core assumption of using frozen textual embeddings for semantic clustering and some design choices in the Interaction Chain and pattern transfer that could benefit from more justification or analysis. These are not fatal flaws but are important points for the authors to address. The contribution is significant and likely meets the bar for ICLR, provided the limitations and assumptions are discussed more transparently.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces TRANSFIR, an inductive reasoning framework for Temporal Knowledge Graphs (TKGs) designed to handle the challenge of *emerging entities*—entities that appear during inference without any historical interactions in the training data. The core idea is to transfer reasoning patterns from semantically similar known entities to these emerging ones. This is achieved via a three-stage pipeline: (1) a learnable vector-quantized codebook classifies entities into latent semantic clusters using frozen textual embeddings, (2) Interaction Chains (ICs) encode entity-specific historical interaction sequences, and (3) a pattern transfer mechanism propagates cluster-level temporal dynamics to generate informative representations for emerging entities.

### Strengths
1.  **Well-Motivated and Defined Problem:** The paper provides a clear, formal definition of the "inductive reasoning for emerging entities" task, a practical but underexplored setting. The motivating empirical study is compelling, showing that ~25% of entities are emerging in standard TKG datasets and that existing SOTA models suffer severe performance drops (~28.6% MRR degradation) on queries involving them, which convincingly establishes the problem's significance.
2.  **Novel and Coherent Methodological Design:** The proposed TRANSFIR framework is a novel synthesis of ideas. The use of a *learnable* VQ codebook on *frozen* textual embeddings is a clever design that provides stable, interaction-aware semantic clustering without risking representation collapse for new entities. The Interaction Chain formulation and the subsequent cluster-level pattern transfer mechanism are directly motivated by the empirical observation of transferable patterns and address the core challenge effectively.
3.  **Extensive and Convincing Empirical Validation:** The experiments are thorough. TRANSFIR is evaluated against a wide array of strong baselines (13 methods across three categories) on four standard TKG benchmarks under a strict chronological split. The reported performance gains are substantial and consistent (average 28.6% MRR improvement). The ablation studies, sensitivity analyses, and investigations into different settings (e.g., "Unknown" vs. "Emerging") robustly validate the contribution of each component and the model's generalization ability. The public code release aids reproducibility.

### Weaknesses
1.  **Dependence on Textual Entity Descriptions:** The framework's first stage relies on pre-computed, frozen textual embeddings (e.g., from BERT). While this is a practical choice, it introduces a dependency on the quality and availability of descriptive entity names. The paper notes performance degradation on GDELT where entity "titles" are noisy (e.g., "EGYPT (EGY@ OPP REF LEG SPY...)"), and the ablation shows removing textual encoding can sometimes help on this dataset. This limits applicability to domains with purely symbolic or poorly described entities.
2.  **Limited Discussion on Computational Overhead:** Although the paper includes a runtime/memory analysis showing favorable comparison to some baselines, the overall complexity of the pipeline—involving a Transformer encoder for ICs, a codebook lookup, and a pattern transfer MLP—is non-trivial. A more detailed discussion on the trade-off between this added complexity and the performance gain, especially for very large-scale graphs, would strengthen the practical evaluation.
3.  **Incomplete Baseline Adaptation Discussion:** The paper compares against static inductive KG methods (e.g., InGram, MorsE) by adapting them to the temporal setting (e.g., merging time windows). While necessary, this adaptation may not represent their optimal performance in a native temporal inductive setting. A more detailed justification of the adaptation protocol or a discussion of the inherent limitations of these baselines for the *temporal* emerging entity task would provide a fairer context.

### Novelty & Significance
**Novelty:** The work is highly novel. While inductive learning on static KGs is established, formalizing and tackling the specific problem of *temporal* inductive reasoning for entities with *zero historical interactions* is a distinct and significant contribution. The core technical ideas—the interaction-aware VQ codebook for clustering emerging entities and the chain-based pattern transfer within clusters—are new to the TKG reasoning literature.

**Significance:** The significance is high. The identified problem is pervasive in real-world dynamic systems (social networks, recommendation systems, etc.). Providing a functional solution, which demonstrably outperforms a suite of strong baselines, advances the state of the art towards more realistic, open-world TKG reasoning. The performance improvements are substantial, and the framework's design is principled and well-explained.

### Suggestions for Improvement
1.  **Robustness to Noisy or Missing Text:** To mitigate the first weakness, the authors could explore or propose a fallback mechanism for when textual information is poor or absent. For instance, an initial clustering could be based on a minimal symbolic identifier, or a learnable module could refine initial embeddings using the first few observed interactions of an emerging entity.
2.  **Deeper Analysis of Learned Clusters:** The case study in Figure 4(b) is excellent but limited. A more systematic quantitative analysis of the semantic coherence of the learned clusters (e.g., using external type ontologies if available for the datasets, or via human evaluation of sampled clusters) would provide stronger evidence that the codebook discovers meaningful, transferable categories.
3.  **Explicit Comparison to a "Fine-Tune Embeddings" Baseline:** A strong and simple baseline for handling new entities is to continue training (fine-tuning) their randomly initialized embeddings during inference. While the representation collapse argument suggests this would fail, explicitly including and comparing against such a baseline would make the empirical results even more compelling and directly justify the need for the proposed transfer mechanism.
4.  **Clarify the "First Appearance" Constraint:** The "Emerging" setting is defined strictly for an entity's *first* appearance (tq = te(e)). The experiments in RQ4 (Unknown vs. Emerging) show performance improves with even a little history. It would be helpful to clarify how TRANSFIR would be deployed in practice: only at the precise moment of first appearance, or also for subsequent predictions? The framework seems applicable to both, but the core novelty is most critical for the strict, harder setting.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Ablation comparing learnable codebook to fixed semantic clustering.** Use a fixed clustering (e.g., k-means on BERT embeddings or known entity types) to show whether the adaptive learning of the VQ codebook is critical, or if any clustering suffices. This directly tests the claimed benefit of the codebook.
2. **Comparison with meta-learning/few-shot baselines for emerging entities.** Methods like MAML or prototypical networks adapted for TKGs are natural competitors for handling unseen entities. Their absence weakens the claim of state-of-the-art inductive reasoning.
3. **Performance evaluation under standard temporal splits (e.g., 8:1:1).** The paper uses a custom 5:2:3 split to increase emerging entities. Results on standard splits are necessary to show the method's general applicability and that improvements are not an artifact of the split.
4. **Experiments on datasets with known entity types.** Using a dataset with ground-truth entity categories (e.g., YAGO) would allow quantitative validation that the learned clusters capture real semantics, rather than relying on anecdotal examples.

### Deeper Analysis Needed (top 3-5 only)
1. **Quantitative analysis of cluster coherence.** Metrics like silhouette score or purity (if external labels exist) are needed to substantiate the claim that the codebook discovers meaningful semantic categories. Without this, the semantic transfer premise is weakly supported.
2. **Analytical justification and correlation of the proposed Collapse Ratio with performance.** The paper introduces a new metric but does not show how it correlates with MRR/Hits across models or training steps. This is critical to validate it as a useful diagnostic tool for representation collapse.
3. **Fine-grained analysis of component contributions.** The ablation shows performance drops, but a deeper analysis (e.g., measuring how much the pattern transfer changes embedding directions/norms for emerging vs. known entities) is needed to understand the mechanism behind the improvement.

### Visualizations & Case Studies
1. **t-SNE visualizations for all major baselines.** Only LogCL is shown. Visualizing embeddings from other strong baselines (e.g., REGCN, HisRes) would demonstrate whether representation collapse is a universal problem and highlight TRANSFIR's advantage.
2. **Systematic failure case analysis.** The appendix mentions one failure due to poor textual information. Presenting several failure cases with root causes (e.g., misclustering, noisy interaction chains) would clearly expose the method's limitations and boundary conditions.
3. **Visualization of constructed Interaction Chains for diverse queries.** Showing the selected interactions for a few queries (with relevance scores) would illustrate whether the TopK relation similarity effectively retrieves transferable patterns or introduces noise.

### Obvious Next Steps
1. **Fine-tuning the textual encoder instead of using frozen embeddings.** The reliance on frozen, generic BERT embeddings is a clear limitation. Fine-tuning the encoder on the task or using entity descriptions should have been explored as a baseline or variant.
2. **Pre-training or initializing the codebook with semantic knowledge.** The codebook is learned from scratch jointly with the task. Initializing it using external knowledge (e.g., entity types) could improve convergence and cluster quality, and is a natural step.
3. **Discussion and preliminary experiments on emerging relations.** The problem is framed around emerging entities, but emerging relations are equally important. A discussion of the method's limitations and potential adaptations for relations is missing and expected.

# Final Consolidated Review
## Summary
This paper introduces TRANSFIR, an inductive reasoning framework for Temporal Knowledge Graphs (TKGs) designed to handle emerging entities—those that appear during inference without any historical interactions. The core idea is to transfer temporal reasoning patterns from semantically similar known entities using a three-stage pipeline: a learnable vector-quantized codebook for latent semantic clustering, Interaction Chains to encode historical sequences, and a cluster-level pattern transfer mechanism.

## Strengths
- **Compelling problem formulation and motivation.** The paper provides a clear, formal definition of inductive reasoning for emerging entities with zero history, a practical but underexplored setting. A thorough empirical study demonstrates the problem's prevalence (~25% of entities are emerging) and the severe performance drop of existing methods, establishing a strong need for a solution.
- **Novel and effective methodology.** The proposed framework synthesizes several ideas into a coherent pipeline. The interaction-aware VQ codebook operating on frozen textual embeddings enables stable semantic clustering for emerging entities, while the Interaction Chain encoding and cluster-level pattern transfer directly implement the insight that semantically similar entities share transferable temporal patterns.
- **Extensive and convincing experimental validation.** TRANSFIR is evaluated against 13 strong baselines across four TKG benchmarks under a strict chronological split. It achieves consistent and substantial improvements (average 28.6% MRR gain). Ablation studies, robustness checks (e.g., varying temporal splits, "Unknown" vs. "Emerging" settings), and efficiency analyses provide comprehensive evidence for the method's effectiveness and generalizability.

## Weaknesses
- **Dependence on informative textual entity descriptions.** The first stage relies on frozen textual embeddings (e.g., from BERT) to establish semantic clusters. The paper acknowledges (in ablation and failure case analysis) that performance can degrade when entity names are noisy or uninformative (e.g., in GDELT), limiting applicability in domains with purely symbolic or poorly described entities.
- **Limited quantitative validation of cluster semantics.** The claim that the codebook discovers "semantically coherent" clusters is supported only by a few hand-picked examples (Figure 4b). A more systematic, quantitative analysis of cluster coherence (e.g., using external type ontologies if available, or cluster purity metrics) would strengthen the foundational premise that semantic similarity enables pattern transfer.

## Nice-to-Haves
- Exploring a fallback mechanism or more robust text encoding for cases with noisy or missing entity descriptions.
- A deeper analytical probe into the pattern transfer mechanism (e.g., how the transfer vector modulates embeddings) to better understand its operation.
- Inclusion of a simple "fine-tune embeddings" baseline during inference to directly contrast with the proposed transfer approach.

## Novel Insights
The paper's key insight is that temporal reasoning patterns are often transferable across entities of similar semantic types, even in the complete absence of historical interactions for a new entity. This is empirically demonstrated (Observation 3) and directly operationalized through the novel combination of a task-aware semantic codebook and cluster-level pattern propagation. This provides a principled path to overcome representation collapse for emerging entities, a significant limitation of prior transductive and inductive methods.

## Suggestions
- In the main text, consolidate and expand the discussion of limitations, particularly the dependency on text quality and the current focus on entities (not relations), to provide a clearer view of the method's boundaries.
- Consider adding a quantitative analysis of cluster coherence in the appendix, even if based on a simple metric like silhouette score on the textual embeddings, to better substantiate the semantic clustering claim.

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 6.0, 6.0]
Average score: 6.0
Binary outcome: Accept
