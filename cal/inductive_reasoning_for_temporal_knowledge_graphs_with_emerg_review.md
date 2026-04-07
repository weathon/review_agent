=== CALIBRATION EXAMPLE 66 ===

# Harsh Critic Review
Now I have enough to write a thorough review. Let me produce it.

---

## Section-by-Section Critical Review

### Title & Abstract

The title accurately reflects the contribution. The abstract's headline claim—28.6% average MRR improvement—is specific and matched in Table 1. The "roughly 25% of entities are emerging" statistic, however, depends critically on the non-standard 5:2:3 chronological split (instead of the widely used 8:1:1). Under the standard split the proportion would be meaningfully smaller, so presenting this number without qualification slightly overstates the ubiquity of the phenomenon. The code link is included, which is positive for reproducibility.

---

### Introduction & Motivation

The problem is well-framed and genuinely important. The motivating example (Barack Obama's first state visit) is intuitive. Contributions are enumerated clearly, including both the technical components and the empirical study.

One framing concern: the paper characterizes prior inductive methods as failing because "new entities already have known interactions" in static KGs but not in TKGs. This is a valid distinction, but the contrast with methods like GenTKG and ICL (which use LLMs and therefore need no historical interactions at all) is never directly addressed here. The introduction could be clearer about why text-based LLM approaches are insufficient.

---

### Preliminaries & Problem Formalization (Section 2)

The formal definition of emerging entities and the evaluation protocol (queries restricted to `tq = te(e)`, i.e., the strict zero-history condition) are clearly stated. This is the most challenging version of the task and a legitimate focus.

**Concern:** The "Unknown" setting (introduced only in Appendix F.3) allows the entity to have some test-time history before `tq`. The paper treats this as supplementary, but it is arguably the more practically relevant scenario in many real systems (where a newly observed entity quickly accumulates a few interactions). Burying this setting in the appendix under-represents a setting where TRANSFIR's relative advantage may also be important to understand.

---

### Empirical Investigation (Section 3)

The three-angle investigation (Data, Representation, Feasibility) is a genuine strength of the paper. The "Collapse Ratio" metric is carefully defined in Appendix C.2 and grounded in generalized variance theory. The t-SNE visuals combined with the quantitative collapse ratio (1.02 → 0.0055 in LogCL) provide convincing evidence of Observation 2.

**Concern on Observation 3:** The feasibility claim—that semantically similar entities share transferable temporal patterns—is illustrated by a single example (president → government visit patterns). This is the most important empirical claim motivating the entire method, yet it receives only anecdotal support. No quantitative analysis is provided of how frequently such pattern matches occur across entity types, or what fraction of test queries for emerging entities are actually covered by a cluster with meaningful IC history. Without this, it is unclear how often the key assumption holds in practice.

**Concern on the 25% statistic:** This figure is measured under the paper's non-standard 5:2:3 split. Under the standard 8:1:1 split (used by most prior TKG papers), the proportion of emerging entities would be substantially lower. The paper does not report what this number would be under the standard split, making it hard to assess whether the problem is as widespread as claimed in general settings.

---

### Methodology (Section 4)

**Codebook Mapping (Section 4.1):** The VQ codebook design is technically sound and leverages a well-understood mechanism (VQ-VAE). However, there is a notable inconsistency in the commitment loss (Equation 3):

> `L_commit = ||h_e − sg[c_{π(e)}]||²`

The stop-gradient is applied to the codeword, which means gradients would flow through `h_e`. But the paper explicitly states "entity embeddings remain **fixed** during training." If `h_e` is frozen, this loss has zero gradient with respect to any learnable parameter and is a no-op. This is not explained or justified. In the original VQ-VAE, the commitment loss encourages the encoder (whose weights are trainable) to commit to its assigned codeword. Here, the "encoder" is a fixed BERT model, making this loss vacuous. The authors should clarify whether `h_e` is truly frozen during training or whether there is a trainable projection layer receiving the commitment gradient.

**Interaction Chain Encoding (Section 4.2):** The construction (Equation 5–6) is clear. The query-relation-guided TopK filtering is a sensible design for reducing irrelevant context.

**Critical gap:** By definition, an emerging entity at `tq = te(e)` has **no historical interactions**—so its IC `C_q` is empty. The paper never explicitly addresses this case. The benefit presumably flows entirely through the cluster-level dynamic prototype `c_k^{dyn}` in the Pattern Transfer step. This should be stated explicitly, and an ablation or analysis of how often the query entity's IC is empty vs. non-empty would clarify the relative contributions of Sections 4.2 and 4.3.

**Chain Pattern Transfer (Section 4.3):** The cluster pooling (Equation 9) aggregates IC representations from known entities in the same cluster, then the pattern transfer (Equation 11) modulates each entity's static embedding using the cluster prototype. This is a clean design. One potential weakness: the cluster prototype is a simple arithmetic mean of IC embeddings, which may fail when a cluster is semantically heterogeneous or when the active entities at a given timestamp are not representative of the cluster as a whole.

---

### Experiments & Results (Section 5)

**Non-standard train/test split:** The paper uses a 5:2:3 chronological split rather than the widely adopted 8:1:1 split. The justification given is that this "reveals more emerging entities." However, this means:
1. All baseline numbers in Table 1 are not directly comparable to any published result in the TKG literature, making it impossible for readers to verify baseline implementations against reported numbers.
2. Baselines trained on only 50% of the data are likely severely disadvantaged, which inflates TRANSFIR's apparent improvement margins. For instance, REGCN achieves MRR 0.1175 on ICEWS14 in this evaluation, far below its published performance with the standard split. It is not clear whether any hyperparameter tuning was performed for baselines under this new split.

This is the most significant methodological concern in the paper. The authors should either (a) also run experiments under the standard split (even with fewer emerging entities), or (b) provide a very thorough justification, with evidence that the baselines are fairly tuned under the new split.

**Baseline modification for static inductive methods:** For CompGCN, MorsE, and InGram, the paper "merge[s] a small window of timestamps (e.g., 7) into a subgraph to run." This is a significant non-trivial adaptation. Since these methods were not designed for the TKG setting at all, their poor performance may reflect this mismatch rather than an intrinsic inability to handle emerging entities. The comparison is not entirely fair, and the paper should acknowledge this limitation more prominently.

**Missing Hits@1:** The paper reports MRR, Hits@3, and Hits@10. Hits@1 is a more stringent metric and is standard in KG link prediction. Its omission is unexplained and potentially selective.

**No variance in main table:** Table 1 reports single values without standard deviations. Appendix F.2 includes error bars for ablations, so the infrastructure exists. For the main comparison, variance over 3 seeds should be shown, especially given that improvements over the best baseline can be relatively small on individual datasets (e.g., +15% MRR on ICEWS05-15).

**Textual encoder comparison (Table 2):** BERT (2019) outperforms Qwen3-Embedding (2025) across all three reported datasets, with a notable gap on ICEWS14 (0.3246 vs. 0.2567). Qwen3-Embedding is a state-of-the-art dense retrieval model specifically designed for embedding tasks. This counterintuitive result is neither explained nor ablated. Possible explanations (e.g., BERT's tokenization aligning better with short entity names, codebook training being tuned to BERT's embedding geometry) should be investigated and discussed. As it stands, this result raises questions about whether the codebook was effectively over-fit to BERT representations.

**GPU efficiency claim (Figure 7):** The paper claims "significantly lower peak GPU memory usage." However, Figure 7 is difficult to read and the precise values for LogCL and HisRes are not clearly stated in the text. A clearer table or more explicit comparison would strengthen this claim.

---

### Ablation Study (Section 5.4)

The four ablation variants are well-chosen and the conclusions are consistent across metrics (confirmed in Appendix F.2). The observation that removing the textual encoder can *improve* performance on GDELT is an important honest acknowledgment; the explanation (GDELT entity names contain abbreviations and symbolic elements) is plausible but insufficiently analyzed. It would be useful to know whether the VQ clustering degrades specifically on GDELT, and whether a simpler fallback (e.g., random BERT embeddings or entity frequency features) helps.

**Concern:** The ablation variant "-Codebook: removing the codebook mapping and using static clustering features only" is described ambiguously. What exactly are "static clustering features"? How are entities clustered without the codebook? This variant's design needs to be spelled out more clearly, since it is directly testing the paper's core contribution.

---

### Limitations & Broader Impact

The failure case (Section F.1 / Appendix) on entity "Bala Ngilari" is a useful concrete illustration. The limitation that sparse/noisy entity names impair the text-based clustering is acknowledged.

**Unacknowledged limitations:**
1. The method implicitly assumes that a single cluster assignment adequately captures an entity's type. Many real-world entities are cross-domain (e.g., a president who is also an economist), and a hard VQ assignment may fail in such cases.
2. The approach depends on the quality of cluster membership: if a cluster contains too few known entities (or known entities with sparse ICs), the cluster prototype will be uninformative. No analysis of cluster sizes or their effect on prediction quality is provided.
3. The assumption that semantic name similarity (BERT embedding proximity) implies temporal pattern similarity is the method's core bet—and while Observation 3 provides anecdotal support, it could break down in domains where entity names are less informative (as already observed with GDELT).
4. The method requires a frozen pre-trained text encoder at inference time for any new entity. This may raise latency or dependency concerns in streaming/online deployment, which is the primary use case for emerging-entity reasoning.

---

### Overall Assessment

TRANSFIR addresses a real and underappreciated problem—reasoning over entities that emerge without any prior interaction history in temporal knowledge graphs—and makes a technically coherent contribution via VQ-codebook-driven semantic clustering and cluster-level pattern transfer. The empirical study in Section 3 is a genuine strength, providing principled motivation. Experimental results in Table 1 are impressive in magnitude. However, the paper has several issues that need to be resolved before acceptance at ICLR. Most critically: the non-standard 5:2:3 split makes all comparisons to published baselines impossible and likely inflates improvements; the commitment loss (Eq. 3) is technically inconsistent with frozen entity embeddings; the central assumption of transferable patterns lacks quantitative grounding beyond a single anecdote; and the surprising underperformance of Qwen3-Embedding vs. BERT goes unexplained. Taken together, these issues do not invalidate the core contribution, but they do undermine confidence in the reported performance margins and in the correctness of some design choices. Addressing the split concern alone would significantly strengthen the paper's credibility.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes TRANSFIR, a novel inductive framework for reasoning on Temporal Knowledge Graphs (TKGs) that specifically targets emerging entities without historical interactions. The method leverages a codebook-based classifier to cluster entities semantically and propagates interaction patterns from similar known entities to those without history. Experimental results across four datasets demonstrate a significant performance improvement over existing baselines in this challenging zero-history inductive setting, effectively mitigating representation collapse.

### Strengths
1.  **Clear Problem Formulation and Empirical Evidence:** The authors rigorously define the problem of "emerging entities" in TKGs, providing compelling empirical evidence (Section 3) that ~25% of entities appear in the test set and cause "representation collapse" in standard models. The introduction of a "Collapse Ratio" metric (Appendix C) to quantify this degradation adds scientific rigor to the motivation.
2.  **Novel Integration of Text and Structure:** The approach creatively combines frozen textual embeddings (for semantic clustering of unseen entities) with temporal structural patterns (Interaction Chains). The "Classification–Representation–Generalization" pipeline offers a structured solution to the cold-start problem that avoids the reliance on entity-specific training embeddings.
3.  **Robust Empirical Validation:** The evaluation is extensive, covering four diverse datasets (ICEWS14, 18, 05-15, GDELT). The model demonstrates consistent superiority across MRR and Hits@k metrics, with an average MRR improvement of 28.6% over the strongest baseline (LogCL). Ablation studies (Fig. 5) clearly validate the contribution of each component (Codebook, IC, Pattern Transfer).

### Weaknesses
1.  **Heavy Reliance on Entity Textual Metadata:** The method critically depends on high-quality entity titles for text embedding (BERT). In true inductive settings where entities might lack distinct semantic labels or have noisily formatted titles (as noted in the ablation discussion for GDELT), the semantic clustering fails. This limits the method's applicability compared to purely structure-based inductive approaches like InGram or ULTRA.
2.  **Baseline Comparison Nuances:** While the paper compares against strong baselines, many (e.g., LogCL, REGCN) are primarily transductive models adapted for this inductive task. While this shows robustness, a deeper comparison with existing *inductive* TKG methods (e.g., ALRE-IR, though limited) would better contextualize the novelty within the specific inductive TKG subfield.
3.  **Computational Complexity vs. Real-World Scaling:** While the efficiency analysis (Appendix D.3) claims linear complexity, the use of a Transformer encoder for Interaction Chains and the VQ codebook search for *all* entities per timestamp adds overhead. The paper claims lower GPU memory usage but does not fully isolate the cost of the VQ clustering step compared to standard embedding lookups in large graphs.
4.  **Minor Issues with Citations:** Some references cite future conference years (e.g., "Zhang et al., 2025a", "Hadipour et al., 2025"). While this likely refers to arXiv versions or accepted pre-prints, it creates potential confusion regarding the timeline of prior art and needs clarification for reproducibility.

### Novelty & Significance
*   **Novelty (High):** There is limited work on inductive reasoning in TKGs specifically for entities with **zero historical interactions**. Existing inductive KG work (InGram, ULTRA) assumes some structural context. TRANSFIR's specific focus on bridging the semantic gap via VQ codebooks in a temporal context is a meaningful contribution to the ICLR scope.
*   **Significance (High):** The problem of emerging entities is prevalent in dynamic systems (social media, event forecasting). Solving "representation collapse" for new entities is a fundamental step toward open-world temporal reasoning.
*   **Clarity (Good):** The methodology is described with clear equations and pipeline diagrams (Fig. 3). The distinction between "Emerging" (zero history) and "Unknown" (some history) is carefully made in Appendix F.
*   **Reproducibility (Good):** Code and configuration are made available (Section 8), and datasets are standard. However, the reliance on external BERT models requires the authors to ensure version consistency in the public code.

### Suggestions for Improvement
1.  **Analyze Structure-Only Fallback:** To strengthen the claim of inductive reasoning, conduct an experiment where entity text is removed or replaced with random IDs to evaluate how much the model relies on text vs. structural patterns. This would clarify the "textual dependency" weakness.
2.  **Refine Baseline Selection:** Ensure the comparison includes more recent "Inductive TKG" specific models if available. If the goal is to highlight the "Cold Start" aspect, explicitly compare with a baseline that uses standard embedding initialization (e.g., Random Initialization per entity) to show the specific gain of the *transfer* strategy.
3.  **Clarify Citation Dates:** Replace future-dated citations with the arXiv version number or the specific conference where they are accepted (e.g., "NeurIPS 2024 (under review)" or "arXiv:2501.xxxxx") to maintain academic rigor.
4.  **Detailed Complexity Breakdown:** Provide a more concrete breakdown of inference time per entity vs. per cluster in practical deployments (e.g., on a single GPU), rather than just memory usage, to address scalability concerns more transparently.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Text-Free Variant:** Replace BERT embeddings with random initialization to prove the graph reasoning mechanism works independently of external semantic knowledge. Without this, the performance gains may simply reflect superior entity initialization via BERT rather than the proposed Interaction Chain framework.
2. **Semantic-Only Baseline:** Evaluate a model using only frozen BERT embeddings with a simple scorer (no Codebook or Interaction Chains). This isolates whether the complex reasoning pipeline adds value over strong pre-trained semantic priors alone.
3. **Split Sensitivity:** Evaluate performance under standard 8:1:1 chronological splits to ensure results are not artifacts of the 5:2:3 split designed to maximize emerging entities. Claims of widespread emergence must hold under conventional evaluation protocols.
4. **True Inductive Baselines:** Compare against dedicated inductive TKG methods rather than transductive models forced into an inductive setting. Current baselines may be unfairly penalized by lack of adaptation, inflating TRANSFIR's relative improvement.

### Deeper Analysis Needed (top 3-5 only)
1. **Codebook Necessity:** Analyze why VQ clustering outperforms direct cosine similarity on text embeddings. If the gain is marginal, the VQ complexity is unjustified overhead compared to simpler semantic matching.
2. **Collapse Metric Validation:** Demonstrate a strong correlation between the proposed "Collapse Ratio" and downstream MRR. Without this, the metric is merely a visualization aid rather than evidence of the mechanism's success.
3. **Cluster Semantic Validity:** Provide quantitative metrics (e.g., silhouette score or ontology alignment) confirming clusters align with semantic types. Claiming clusters represent 'Countries' requires statistical validation beyond cherry-picked t-SNE examples.
4. **Error Analysis:** Systematically categorize failure cases where cluster assignment was incorrect due to ambiguous text. This reveals the upper bound of performance imposed by the textual encoder quality.

### Visualizations & Case Studies
1. **Attention Weights:** Visualize attention weights over the Interaction Chain for emerging entities to verify the model focuses on transferable patterns. Uniform attention would contradict the claim of selective pattern transfer.
2. **Embedding Trajectories:** Show how emerging entity embeddings evolve during training compared to known entities using dynamic plots. Static t-SNE is insufficient to prove collapse is prevented throughout the optimization process.
3. **Pattern Transfer Flow:** Diagram specific examples of patterns transferred from known to emerging entities within a cluster. This validates whether the 'Generalization' step actually propagates useful temporal dynamics.
4. **Cluster Boundary Cases:** Visualize entities near cluster boundaries that were misclassified. This exposes the robustness of the VQ codebook to semantic ambiguity.

### Obvious Next Steps
1. **Decouple Text and Graph:** Redesign the framework to function without pre-trained language models to ensure applicability in text-sparse domains. Reliance on BERT limits the method to datasets with high-quality entity descriptions.
2. **Standardize Inductive Protocol:** Adopt a community-agreed inductive TKG benchmark protocol to ensure fair comparison across future works. Current adaptations of transductive baselines introduce variability in evaluation rigor.
3. **Handle Emerging Relations:** Extend the Codebook mechanism to cluster and transfer patterns for unseen relations. Real-world graph evolution involves both new entities and new relation types.
4. **Reduce Computational Overhead:** Optimize the VQ search and Transformer encoding to match the efficiency of simpler graph baselines. Current efficiency claims rely on specific hardware settings that may not scale.

# Final Consolidated Review
## Summary

This paper proposes TRANSFIR, an inductive reasoning framework for temporal knowledge graphs (TKGs) that addresses reasoning on "emerging entities"—entities that appear during inference without any historical interactions. The method uses a VQ codebook to cluster entities by semantic similarity (via frozen BERT embeddings), encodes Interaction Chains to capture temporal patterns, and transfers cluster-level patterns to emerging entities. The paper demonstrates that emerging entities constitute approximately 25% of entities under the chosen split, cause representation collapse in existing models, and that TRANSFIR achieves significant MRR improvements (28.6% average) across four benchmarks.

## Strengths

- **Well-motivated problem with rigorous empirical grounding.** The paper identifies a real gap: existing TKG methods fail on entities with zero interaction history. Section 3 provides a three-perspective empirical study (data, representation, feasibility) with the Collapse Ratio metric (Appendix C.2) quantifying representation degradation. The finding that emerging entities comprise ~25% of test entities motivates the work concretely.

- **Principled technical design for the zero-history setting.** The Classification–Representation–Generalization pipeline is coherent: frozen text embeddings provide type-level priors for emerging entities (bypassing the need for entity-specific training), while cluster-level prototypes aggregate Interaction Chains from similar known entities. The VQ codebook is a reasonable mechanism for enabling pattern transfer without requiring per-entity optimization.

- **Strong empirical performance with consistent gains.** Table 1 shows TRANSFIR outperforming all baselines across four datasets, with MRR improvements of 24.6%, 24.3%, 15.0%, and 50.5% respectively. The ablation study (Figure 5) clearly validates the contribution of each component (Codebook, IC, Pattern Transfer, Textual Encoding).

- **Clear formalization of the task.** The definition of emerging entities (first appearance time `te(e)` and queries restricted to `tq = te(e)`) precisely specifies the zero-history condition, distinguishing it from the "Unknown" setting where entities gain some test-time history.

## Weaknesses

- **Non-standard train/test split complicates reproducibility and comparison.** The paper uses a 5:2:3 chronological split rather than the widely adopted 8:1:1. While the paper justifies this as revealing more emerging entities, it means baseline numbers cannot be verified against published results, and baselines trained on only 50% of data may be disadvantaged. The paper does not clarify whether baselines received hyperparameter tuning under the new split. This does not invalidate the results but limits direct comparison to prior literature.

- **Commitment loss formulation appears inconsistent with frozen embeddings.** Equation 3 defines `L_commit = ||h_e − sg[c_{π(e)}]||²`, with stop-gradient on the codeword. The paper states "entity embeddings remain **fixed** during training." If `h_e` is truly frozen, this loss has zero gradient with respect to any learnable parameter and is vacuous. This should be clarified—either `h_e` is projected through a trainable layer, or the commitment loss serves a different purpose than stated.

- **Key assumption of transferable patterns lacks quantitative support.** Observation 3 (Section 3) claims that semantically similar entities share transferable temporal patterns, motivating the entire method. However, this is supported only by a single anecdotal example (president → government visit patterns). No quantitative analysis is provided of how frequently pattern matches occur across entity types, or what fraction of emerging-entity queries fall into clusters with informative IC histories.

- **Surprising underperformance of Qwen3-Embedding vs. BERT is unexplained.** Table 2 shows BERT (2019) outperforming Qwen3-Embedding (2025) across all three datasets, with a substantial gap on ICEWS14 (0.3246 vs. 0.2567). This is unexpected for a state-of-the-art embedding model. The paper does not investigate whether the codebook was tuned to BERT's geometry or whether this reflects a genuine advantage of BERT for short entity names.

- **Ablation variant descriptions are ambiguous.** The "-Codebook" variant is described as "removing the codebook mapping and using static clustering features only" (Section 5.4). What "static clustering features" means is unclear—how are entities clustered without the codebook? This should be specified since this variant directly tests the core contribution.

- **Main results table lacks variance estimates.** While Appendix F.2 includes error bars for ablations, Table 1 reports single values without standard deviations across the three seeds mentioned in Appendix E.3. This is needed for assessing the reliability of improvement margins, particularly where gains over the second-best baseline are modest (e.g., +6.7% Hits@3 on ICEWS05-15).

## Nice-to-Haves

- **Correlation between Collapse Ratio and MRR.** While the paper shows improved Collapse Ratio (0.0055 → 0.8677), demonstrating a quantitative correlation between this metric and downstream MRR would strengthen the claim that preventing collapse causes better predictions.

- **Cluster semantic validation.** The case study in Figure 4(b) shows clusters appear semantically coherent (Country, Civic & Parties, Citizen), but quantitative metrics (e.g., silhouette score or alignment with known entity types if available) would validate the VQ codebook's clustering quality more rigorously.

- **Hits@1 reporting.** This is a more stringent metric than Hits@3 or Hits@10 and is standard in link prediction. Including it would strengthen the evaluation.

- **Analysis of codebook size vs. cluster informativeness.** When a cluster contains few known entities or entities with sparse ICs, the dynamic prototype may be uninformative. An analysis of how cluster size affects prediction quality would clarify failure modes.

## Removed Points

*These points are flagged to be removed, treat them with caution:*

- **Citations with future years (e.g., "Zhang et al., 2025a").** This is a common formatting issue for papers under review and reflects arXiv preprints or accepted works. Not a substantive criticism of the paper's content.

- **Text-free variant as missing experiment.** This IS addressed in the ablation study (Figure 5, "-Textual encoding" variant), which shows performance drops when BERT embeddings are replaced with random initialization.

- **GPU efficiency figures hard to read.** Figure 7 presents the data clearly; this is a minor presentation preference, not a missing evaluation.

- **Heavy reliance on entity textual metadata.** This is acknowledged in the paper (Section 5.4 and F.1 discuss GDELT's noisy entity names) and addressed via the ablation showing textual encoding's contribution. The dependency is explicit, not hidden.

## Novel Insights

The paper makes a genuine contribution by formally defining and empirically validating the "emerging entity" problem in TKGs. The observation that representation collapse occurs specifically for emerging entities—and that it can be mitigated via semantic clustering and pattern transfer—is the key insight. The Collapse Ratio metric provides a useful diagnostic for future work on inductive TKG reasoning. The finding that frozen text embeddings can serve as type-level priors for zero-shot entity handling is not entirely new (similar ideas exist in static KG work), but the combination with temporal pattern transfer via cluster-level prototypes is novel.

## Suggestions

1. **Clarify the commitment loss:** Either state that `h_e` is projected through a trainable layer before codebook assignment, or explain what role the commitment loss plays if embeddings are frozen.

2. **Provide standard split results:** Even if 5:2:3 better reveals emerging entities, including results under 8:1:1 (or reporting the proportion of emerging entities under both splits) would allow readers to calibrate the problem's scale against prior work.

3. **Quantify Observation 3:** Provide statistics on how often semantically similar entities share temporal patterns—for example, what fraction of test queries have at least one known entity in their cluster with a non-empty Interaction Chain.

4. **Add variance to Table 1:** Include standard deviations across seeds to establish the reliability of improvements.

5. **Explain the BERT vs. Qwen3 result:** Investigate whether this reflects BERT's tokenization being better suited for short entity names, or whether hyperparameter tuning favored BERT. This helps users choose encoders for new domains.

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 6.0, 6.0]
Average score: 6.0
Binary outcome: Accept
