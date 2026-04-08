=== CALIBRATION EXAMPLE 35 ===

# Harsh Critic Review
Now I have sufficient material for a thorough review. Let me produce it.

---

## Section-by-Section Critical Review

### Title & Abstract

The title is accurate and clearly signals the scope. The abstract's headline claim—"average improvement of 28.6% in MRR"—is computed as the arithmetic mean of four per-dataset relative improvements ({24.6%, 24.3%, 15.0%, 50.5%}), which is dominated by a single outlier result on GDELT. Averaging relative improvements across datasets with very different absolute magnitudes obscures more than it reveals. The abstract does not mention the non-standard chronological split (5:2:3 instead of the conventional 8:1:1) that was specifically constructed to enlarge the emerging-entity fraction; this split choice should be disclosed upfront as it governs what the numbers actually mean.

---

### Introduction & Motivation

The problem is genuinely well-motivated. The observation that ~25% of entities are unseen during training is important, but this statistic is obtained under the authors' custom 5:2:3 split—an 8:1:1 split would expose far fewer unseen entities. The introduction should qualify this. The "representation collapse" framing is conceptually sharp and the three-question empirical decomposition (Data / Representation / Feasibility) is a clear organizing structure. The opening example (Obama's first state visit) is illustrative, though the analogy to molecular networks (Hadipour et al., 2025) is peripheral and can be cut.

---

### Preliminaries and Problem Formalization (Section 2)

The formal definition of emerging entities as those satisfying $e \in E_{1:t} \setminus E_{1:t-1}$ is clean and unambiguous. The constraint $t_q = t_e(e)$ (strict zero-history) is a strong but principled choice and clearly distinguishes this task from the more relaxed "Unknown" setting of Appendix F.3. One gap: the paper never specifies how entity titles are defined across datasets. For GDELT, entity names are abbreviations (e.g., "EGYPT (EGY@ OPP REF LEG SPY...)"), which directly affects the BERT-based classification step. The formalism implicitly assumes entity titles exist and are meaningful, which should be flagged as an assumption.

---

### Empirical Investigation (Section 3)

**Q1 (Prevalence):** Figures 2(a)/(b) convincingly show entity emergence and performance degradation across four datasets. However, re-emphasizing that the 25% figure is split-dependent is important. A brief sensitivity analysis ("how does the fraction change at 8:1:1?") would strengthen this finding.

**Q2 (Representation Collapse):** The "Collapse Ratio" (CR) is a novel metric defined as the ratio of generalized variances of emerging vs. known entity embeddings (Appendix C.2). The metric is rotation-invariant and well-grounded in Anderson (1958) and Zbontar et al. (2021). However, CR = 0.0055 vs. 1.0201 after training in LogCL is presented without error bars or statistical significance across seeds, and the reference set used to normalize (known-entity embeddings) can itself shift during training. If the known entities also collapse (which is plausible under contrastive loss), the ratio can be inflated. The paper should verify this by also reporting GS(X_ref) before and after training.

**Q3 (Transferability):** Observation 3 is qualitative—a cherry-picked example of president entities sharing visit–negotiation patterns. This is motivational but not statistical evidence. The claim that "semantically similar entities exhibit comparable interaction histories" is the core bet of the entire framework, yet it is never quantified (e.g., as intra-cluster interaction entropy or pattern overlap). This is a meaningful hole given how central this assumption is.

---

### Methodology (Section 4)

**Codebook Mapping (Section 4.1):** The VQ codebook with stop-gradient is standard (VQ-VAE). However, there is a technical inconsistency: the commitment loss is

$$\mathcal{L}_{\text{commit}} = \|\mathbf{h}_e - \text{sg}[\mathbf{c}_{\pi(e)}]\|_2^2$$

whose gradient w.r.t. $\mathbf{h}_e$ would normally push entity embeddings toward their prototype—but $\mathbf{h}_e$ is declared *frozen* throughout (they are fixed BERT outputs). This means $\mathcal{L}_{\text{commit}}$ is zero-gradient with respect to any learnable parameter and contributes nothing to training. The commitment loss is designed to be used when the encoder is learnable; applying it to frozen embeddings is a design error (or at minimum a dead loss term). The paper needs to clarify what $\mathcal{L}_{\text{commit}}$ actually optimizes in this specific setup.

**Interaction Chain Encoding (Section 4.2):** The TopK filtering by cosine similarity between query and interaction relation embeddings (Eq. 6) is a reasonable design choice but raises a subtle concern: at inference time for an emerging entity, there are no prior interactions, so $\mathcal{C}_q = \emptyset$ and $\mathbf{h}_{\text{eq}}^{\text{IC}}$ is undefined or a zero vector. The paper never addresses this case explicitly. The IC embedding for emerging entities must come entirely from the Pattern Transfer module, but this dependency is not stated clearly—readers cannot tell whether there is a fallback path or whether the model silently produces a zero/null IC embedding for emerging entities.

**Chain Pattern Transfer (Section 4.3):** Equation (11): $\tilde{\mathbf{h}}_e = \mathbf{h}_e + \omega_e \cdot \mathbf{c}_{\pi(e)}^{\text{dyn}}$. Here $\omega_e = \Psi(z_e)$ where $z_e = [\mathbf{h}_e \| \mathbf{c}_{\pi(e)}^{\text{dyn}}]$. The dimensionality of $\omega_e$ is never specified. If $\omega_e \in \mathbb{R}^d$, the "·" is element-wise multiplication; if $\omega_e \in \mathbb{R}$, it is a scalar gate. These are architecturally very different choices. Appendix D does not clarify this. Related: if a semantic cluster $k$ has no query entities at timestamp $t$ (i.e., $Q_k = \emptyset$), the dynamic prototype $\mathbf{c}_k^{\text{dyn}}$ is the average of an empty set, which is undefined. The paper does not specify a fallback (e.g., using the static codeword $\mathbf{c}_k$). This edge case is non-trivial because sparse datasets will frequently have clusters with no active queries.

The overall pipeline flow (Classification → Representation → Generalization) is logical, but the paper's description of what happens for an emerging entity at query time in a single coherent walkthrough is missing. The reader must reconstruct this from scattered equations.

---

### Experiments (Section 5)

**Dataset Split:** The authors explicitly depart from the standard 8:1:1 chronological split and use 5:2:3, justifying this as exposing more emerging entities. This creates a fundamental comparability problem: all reported numbers are incomparable to every prior published result on ICEWS14, ICEWS18, ICEWS05-15, and GDELT. The paper presents no results under the standard split, so it is impossible to know whether TRANSFIR's baseline performance is competitive with published numbers, or whether baselines are simply weakened by having less training data. This is a serious issue for reproducibility and fair benchmarking within the ICLR community.

**Baseline Evaluation Fairness:** Several baseline adaptations are non-trivial:
- Static inductive methods (InGram, CompGCN, MorsE) "merge a small window of timestamps (e.g., 7) into a subgraph." This is a coarse approximation that may poorly represent these methods' strengths—they are inherently designed for static graphs and are being evaluated in a setting they were never designed for.
- TILP's rule length is reduced (from 5 to 3/2) for computational reasons. This fundamentally changes the expressiveness of the model and should be reported as a caveat.
- The paper reports MRR as unavailable for GenTKG "due to it's reliance on multiple generations for each query." However, an MRR estimate could be obtained from a single greedy generation. Omitting this metric for one specific baseline makes the comparison in Table 1 uneven.

**Missing Hits@1:** All prior TKG reasoning papers report Hits@1, Hits@3, and Hits@10. This paper only reports Hits@3 and Hits@10. For link prediction, Hits@1 is arguably the most informative metric (exact top-1 accuracy). Its absence is conspicuous and weakens the comparison to any method in the literature.

**GDELT Outlier:** TRANSFIR's Hits@10 improvement on GDELT is reported as 101.4%—a doubling of the best baseline. Given that the best baseline on GDELT achieves only 0.0932 Hits@10 (very low absolute performance), a doubling may reflect a baseline failure mode in the new split rather than a genuine algorithmic advantage. The paper does not investigate this critically.

**Table 2 Metric Ambiguity:** Table 2 (Textual Encoder comparison) reports a value of 0.3246 for BERT on ICEWS14, which matches the Hits@10 value from Table 1, not MRR (0.1687). The table header does not specify which metric is being reported. This needs to be clarified.

**Statistical Reporting:** Only TRANSFIR's results are averaged over three random seeds. Baselines are reported without variance estimates. The paper should at minimum report standard deviations for TRANSFIR and acknowledge that baseline variance is unknown.

**Ablation descriptions are vague:** The "-Codebook" variant uses "static clustering features only" and "-Pattern Transfer" uses "static representations." These descriptions are insufficient to understand what the ablated model actually does. How are entities assigned to clusters without the learned codebook? What "static representation" replaces the transfer? Clearer operational definitions would make the ablations interpretable.

---

### Representation Analysis (Section 5.3 / Figure 4)

The t-SNE visualization improvement is visually compelling. The CR improvement from 0.0055 to 0.8677 is dramatic. However, one should note that t-SNE visualizations are sensitive to perplexity and iteration count, and the selection of which datasets/seeds to display in the main text is not explicitly described. The Collapse Ratio alone as a proxy for "informative embeddings" needs additional validation: a uniformly random embedding space would have high CR but be useless for prediction. The paper would benefit from showing that high CR correlates with downstream prediction accuracy, not just that TRANSFIR has high CR.

---

### Writing & Clarity

Section 5.5 (Extended Experiments) has a broken paragraph that reads "ciency in GPU memory and computational time" and "sults and experimental details are available in Appendix F.3" mid-section, suggesting text was truncated or misplaced during compilation. Equation (8) spans multiple lines but the aggregation is not rendered clearly in the parsed text. The case study in Section 5.3(c) is useful but the notation mixing graph arrows and prose is hard to parse. The algorithmic description in Appendix D.2 is the clearest part of the paper.

---

### Limitations & Broader Impact

The authors honestly acknowledge GDELT's noisy entity titles and the failure case in Appendix F.1. They also note future work on emerging *relations*. However, several limitations are unaddressed:

1. **Scalability at inference time:** The cluster-pooling step (Eq. 9) computes dynamic prototypes per timestamp based on all query entities. For large graphs with many queries per timestamp, this adds an extra forward pass cost not fully accounted for in the complexity analysis.

2. **The entity-title availability assumption** is strong. In many real-world TKG settings, entities may have numerical IDs, non-English names, or no associated text at all. The approach degrades gracefully only when BERT produces meaningful embeddings.

3. **The "25% emerging entities" framing** as a general property of TKGs is stated without enough qualification. This fraction depends heavily on the time split, which the authors control. A practitioner applying the standard split would see very different fractions.

4. **Transductive performance not reported:** The paper never shows how TRANSFIR performs on the *standard* Vanilla setting (all test triples). TRANSFIR could hypothetically sacrifice transductive performance to gain inductive performance. Table 1 only shows the Emerging subset.

---

### Overall Assessment

TRANSFIR addresses a genuine and under-studied problem—reasoning for entities that appear without any historical interactions in TKGs—and the three-stage Classification–Representation–Generalization pipeline is a clean, well-motivated design. The empirical investigation (Section 3) is a genuine contribution that frames the problem quantitatively. The codebook-based approach is creative and the VQ mechanism is an appropriate fit. However, the paper has several concerns that collectively prevent confident acceptance. The most critical are: (1) the non-standard 5:2:3 data split renders every number incomparable to prior work and potentially inflates reported improvements by weakening baselines through reduced training data; (2) the commitment loss is applied to frozen embeddings and thus contributes nothing, which is either a technical error or a mismatch between the method's description and its implementation; (3) the core transferability assumption is only qualitatively validated; (4) edge cases in the pipeline—no IC for emerging entities, empty clusters—are never addressed; and (5) Hits@1 is absent. ICLR expects both methodological soundness and experimental rigor; addressed, the contribution could stand, but in its current form the paper requires substantial revision before it meets that bar.

# Neutral Reviewer
## Balanced Review

### Summary
This paper addresses a critical gap in Temporal Knowledge Graph (TKG) reasoning: inductive forecasting for emerging entities that appear with zero historical interactions. The authors propose TRANSFIR, a Classification–Representation–Generalization framework that maps entities into latent semantic clusters via a frozen-text VQ codebook, encodes relation-conditioned Interaction Chains to capture temporal patterns, and propagates learned chain prototypes across clusters to generate informative embeddings for unseen nodes. Extensive experiments across four benchmarks demonstrate that TRANSFIR consistently outperforms strong graph-based, path-based, and inductive baselines while mitigating the representation collapse typically observed for zero-history entities.

### Strengths
1. **Well-Motivated & Empirically Grounded Problem Definition.** The paper rigorously justifies the need for emerging entity reasoning, quantifying that ~25% of entities in standard TKGs are unseen during training (Sec. 3, Obs. 1) and introducing a geometric Collapse Ratio metric to substantiate the degradation in embedding quality (Sec. 3, Obs. 2). This empirical investigation directly supports ICLR's expectation for clear, evidence-driven motivation.
2. **Cohesive Methodological Pipeline.** The Classification–Representation–Generalization design directly tackles the zero-history constraint. The use of an interaction-aware VQ codebook to assign categorical priors without updating entity embeddings (Sec. 4.1) is a principled choice that stabilizes training, while the cluster-level prototype pooling (Eq. 9) and transfer mapping (Eq. 11) offer a clear mechanism for knowledge propagation.
3. **Comprehensive & Rigorous Evaluation.** The evaluation spans four established TKG datasets with a custom chronological 5:2:3 split designed to stress-test inductive generalization (Sec. 5.1). Results show consistent gains (avg. +28.6% MRR), supported by thorough ablations (Sec. 5.4), hyperparameter sensitivity (Sec. 5.5), multiple PLM backbones (Table 2), efficiency analysis (Fig. 7), and extended robustness tests under varying temporal splits and the "Unknown" setting (Sec. 5.5).
4. **Reproducibility & Open Science.** The paper provides a clear reproducibility statement, pseudocode (App. D.2), complexity analysis (App. D.3), and commits to public code release, aligning well with ICLR's standards for transparency.

### Weaknesses
1. **Ambiguity in Handling Strictly Zero-History Entities in the IC Module.** Sec. 4.2 defines the Interaction Chain $C_q$ by collecting *past interactions of the query entity* within window $T$. For an emerging entity queried exactly at its emergence time ($t_q = t_e(e)$), this set is vacuous. The paper does not explicitly describe how $h^{IC}_{e_q}$ is computed in this zero-history case, nor how an empty chain flows through the Transformer and relation-guided attention (Eq. 8). This creates a critical gap between the problem formulation and the architectural pipeline.
2. **Over-Reliance on Pretrained Textual Embeddings for Clustering.** The VQ codebook depends entirely on frozen PLM title embeddings. The authors acknowledge in Sec. 5.4 that GDELT entity titles are highly noisy/abbreviated, causing textual encoding to sometimes hurt performance. While multiple PLMs are tested, the framework lacks a robust fallback (e.g., structural fallback, random-walk priors, or learnable text augmentation) for entities with poor or missing textual metadata, limiting real-world applicability.
3. **Insufficient Detail on Baseline Adaptation for the Inductive Setting.** While 13 baselines are evaluated, the paper only briefly mentions implementation adjustments (App. E.2). Inductive baselines like InGram and static methods like CompGCN require non-trivial adaptations to respect the strict chronological split and the "no historical interaction" constraint for emerging entities. Without explicit adaptation protocols, it is difficult to verify whether performance gaps stem from architectural limitations or suboptimal baseline tuning under the novel evaluation protocol.
4. **Non-Standard Representation Collapse Metric.** The proposed Collapse Ratio (App. C.2) uses log-determinant covariance to measure geometric spread. While mathematically sound, it is sensitive to dimensionality and sample size ($n < d$), and the paper does not compare it against established metrics in self-supervised/contrastive literature (e.g., effective rank, alignment/uniformity, or singular value decay). This makes cross-paper comparisons and quantitative claims slightly less robust.

### Novelty & Significance
The paper addresses a highly significant and under-explored problem in dynamic graph learning: open-world forecasting for entities with zero interaction history. Methodologically, TRANSFIR introduces a novel synthesis of vector-quantized semantic clustering and temporal interaction chain transfer, which effectively sidesteps representation collapse without relying on entity-specific gradient updates. While individual components (VQ clustering, sequence Transformers, prototype pooling) are established, their integration into a unified pipeline explicitly designed for zero-history temporal induction represents solid novelty. The work meets ICLR's acceptance bar in terms of empirical rigor, problem relevance, and architectural soundness, provided the methodological ambiguities are clarified.

### Suggestions for Improvement
1. **Explicitly Formulate Zero-History Handling.** Add a clear subsection or pseudocode branch detailing how the framework processes queries where $|C_q| = 0$. For example, specify whether a learned null token, cluster prototype alone, or neighbor chain substitution is used, and update Eq. 8 to handle empty sequences gracefully.
2. **Provide Detailed Baseline Adaptation Protocols.** Include a table or appendix section outlining precisely how each baseline (especially InGram, ULTRA, and temporal GNNs) was modified to comply with the chronological split and the strict no-past-interaction constraint. This ensures fair comparison and strengthens reproducibility.
3. **Augment or Diversify the Clustering Prior.** To mitigate sensitivity to noisy/missing PLM text, consider incorporating lightweight structural priors (e.g., relation-type distributions or degree profiles) into the VQ input, or experiment with a text dropout/augmentation strategy during training to improve robustness on datasets like GDELT.
4. **Complement Collapse Ratio with Standard Metrics.** Report effective rank or alignment/uniformity metrics alongside the Collapse Ratio. This will ground the representation quality claims in widely accepted standards and facilitate direct comparison with recent contrastive/self-supervised KG literature.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Fair Input Comparison:** Baselines likely use random/init embeddings while TRANSFIR uses BERT; add experiments where baselines receive identical textual embeddings or TRANSFIR is tested without them to isolate the architectural contribution from the input advantage.
2. **Text Robustness Stress Test:** Systematically inject noise or mask entity titles to quantify performance degradation, as the method admits failure on GDELT when text is ambiguous (Sec 5.4).
3. **Strict Zero-History Isolation:** Disaggregate results for entities with *absolute* zero history versus those with test-window interactions (Appendix F.3), as the core claim specifically addresses the former "no historical interactions" scenario.
4. **Baseline Text Augmentation:** Re-run top baselines (e.g., LogCL, REGCN) with equivalent textual features to ensure the performance gain is not solely due to auxiliary semantic information.

### Deeper Analysis Needed (top 3-5 only)
1. **Cluster-Pattern Correlation:** Quantify the correlation between codebook cluster membership and actual interaction pattern similarity, proving clusters capture temporal dynamics rather than just textual similarity.
2. **Codebook Stability Analysis:** Measure cluster assignment volatility over training epochs to ensure the transfer targets are stable and not shifting arbitrarily during optimization.
3. **Systematic Failure Taxonomy:** Expand the single failure case (Sec F.1) into a categorized analysis of when transfer fails (e.g., ambiguous text, unique roles, sparse clusters) to define method boundaries.
4. **Semantic Drift Measurement:** Evaluate performance decay as the time gap between entity emergence and query increases to test the limits of static textual embeddings over long horizons.

### Visualizations & Case Studies
1. **Concrete Transfer Trace:** Visualize a specific instance where a known entity's interaction chain directly influences an emerging entity's prediction score to demystify the "Pattern Transfer" mechanism.
2. **Attention Heatmaps:** Display relation-guided attention weights over the Interaction Chain to verify the model focuses on temporally relevant events rather than noise.
3. **Cluster Evolution Plot:** Show t-SNE of entity embeddings and codebook prototypes across training epochs to demonstrate convergence and separation of emerging entities into clusters.

### Obvious Next Steps
1. **Text-Independent Variant:** Develop and evaluate a version using only structural information to ensure applicability to text-sparse knowledge graphs, which are common in real-world scenarios.
2. **Cross-Dataset Generalization:** Train on one dataset (e.g., ICEWS) and test on another (e.g., GDELT) to verify true inductive capability beyond chronological splits within the same domain.
3. **Dynamic Codebook Mechanism:** Implement time-evolving prototypes to handle entity semantic drift rather than relying on static textual embeddings that cannot adapt to role changes.

# Final Consolidated Review
## Summary

The paper addresses inductive reasoning for emerging entities in Temporal Knowledge Graphs (TKGs)—entities that appear at inference time without any historical interactions. The authors propose TRANSFIR, a framework that maps entities to latent semantic clusters via a VQ codebook (using frozen BERT embeddings), encodes Interaction Chains to capture temporal patterns, and propagates cluster-level prototypes to provide informative representations for zero-history entities. Experiments across four TKG benchmarks show consistent improvements over graph-based, path-based, and static inductive baselines.

## Strengths

- **Rigorous problem formulation with empirical grounding.** The paper formally defines emerging entities (first appearance at $t_e(e)$ with zero history) and provides quantitative evidence that ~25% of entities are unseen during training, with existing methods showing significant performance degradation. The Collapse Ratio metric (log-det covariance ratio between emerging and known entity embeddings) provides a principled measure of representation quality degradation.

- **Well-motivated three-stage pipeline.** The Classification–Representation–Generalization design directly addresses the zero-history constraint: the VQ codebook provides category priors without requiring interaction history, Interaction Chains capture sequential patterns, and Pattern Transfer propagates learned dynamics to emerging entities. The ablation study (Fig. 5) confirms each component contributes meaningfully.

- **Comprehensive empirical evaluation.** The paper evaluates across four datasets (ICEWS14, ICEWS18, ICEWS05-15, GDELT) with multiple baselines (13 methods), ablations, hyperparameter sensitivity, multiple PLM backbones (Table 2), efficiency analysis (Fig. 7), and robustness tests under varying temporal splits (Fig. F.4) and the "Unknown" setting with test-time history (Fig. 10).

## Weaknesses

- **Non-standard chronological split (5:2:3) limits comparability.** The paper explicitly departs from the conventional 8:1:1 split to increase the proportion of emerging entities. While this serves the paper's research question, it renders all reported numbers incomparable to prior published results. The paper does not provide baseline results under the standard split, making it difficult to assess whether baselines are weakened by reduced training data. The "25% emerging entities" statistic is split-dependent and should be qualified accordingly.

- **Technical inconsistency: commitment loss on frozen embeddings.** Equation (3) defines $\mathcal{L}_{\text{commit}} = \|\mathbf{h}_e - \text{sg}[\mathbf{c}_{\pi(e)}]\|_2^2$, which would normally push entity embeddings toward their prototypes. However, $\mathbf{h}_e$ is explicitly frozen (Sec. 4.1: "These embeddings remain fixed during training"). With stop-gradient on $\mathbf{c}_{\pi(e)}$, this loss term has no gradient path to any learnable parameter—it is a dead loss. The paper should clarify whether this is intentional or a design error.

- **Missing explicit handling for zero-history entities in the IC module.** For an emerging entity queried at its emergence time, the Interaction Chain $\mathcal{C}_q$ is empty (no past interactions). Equation (8) aggregates over the chain, but the paper does not specify what happens when $|\mathcal{C}_q| = 0$. Is $\mathbf{h}^{\text{IC}}_{e_q}$ a zero vector? A learned null embedding? This edge case is central to the proposed task and requires explicit treatment.

- **Empty cluster edge case undefined.** Equation (9) defines the dynamic prototype $\mathbf{c}^{\text{dyn}}_k$ as an average over query entities in cluster $k$ at timestamp $t$. If $Q_k = \emptyset$ (no query entities in that cluster), the prototype is undefined. Sparse datasets or small codebooks may frequently encounter this situation. The paper does not specify a fallback mechanism.

- **Heavy reliance on textual entity embeddings.** The VQ codebook depends entirely on frozen BERT embeddings. Section 5.4 acknowledges that GDELT's noisy entity titles (abbreviations like "EGYPT (EGY@ OPP REF LEG SPY...)") cause textual encoding to sometimes hurt performance. The framework lacks a robust fallback for entities with poor or missing textual metadata—a realistic scenario in many TKG applications.

- **Missing Hits@1 and Table 2 metric ambiguity.** Standard TKG reasoning papers report Hits@1, Hits@3, and Hits@10. This paper omits Hits@1, which is the most informative metric for link prediction (exact top-1 accuracy). Table 2 (Textual Encoder comparison) reports a value of 0.3246 for BERT on ICEWS14, which matches Hits@10 from Table 1, but the table header does not specify which metric is being reported, creating confusion.

- **Transferability assumption not quantitatively validated.** Observation 3 claims "semantically similar entities exhibit comparable interaction histories" based on a single qualitative example (Fig. 2d). This assumption underpins the entire framework—entities are clustered by textual similarity and expected to share temporal patterns—yet no quantitative validation (e.g., intra-cluster pattern entropy, cross-cluster pattern divergence) is provided.

## Nice-to-Haves

- Detailed baseline adaptation protocols in the appendix, specifying exactly how each transductive baseline was modified to respect the chronological split and zero-history constraint. This would strengthen reproducibility.

- An experiment isolating the contribution of textual embeddings: running TRANSFIR with randomly initialized (rather than BERT) entity embeddings, and/or providing baselines with equivalent textual features to ensure the performance gain is architectural rather than data-related.

- Hits@1 results to complete the standard evaluation suite.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Critic's claim that the 28.6% improvement is "misleading" or "dominated by GDELT."** The arithmetic mean of relative improvements is a standard reporting practice, and all four datasets show substantial gains (15–50%). The GDELT result is not an outlier invalidating the overall claim.

- **Critic's speculation that GDELT's 101.4% improvement "may reflect baseline failure mode."** Without evidence that baselines are broken rather than simply weak on emerging entities, this is conjecture. The paper discusses GDELT's challenging entity titles, which affects both TRANSFIR and baselines.

- **Critic's demand for standard-split results to assess "baseline weakening."** The paper's research question is specifically about emerging entities; the 5:2:3 split is a deliberate design choice to stress-test this setting. While comparability concerns are valid, demanding standard-split results is outside the paper's stated scope.

- **Critic's complaint about "cherry-picked" Observation 3.** The observation is motivational, not claimed as statistical proof. The ablation study validates that pattern transfer helps empirically.

- **Neutral reviewer's concern about "Collapse Ratio sensitivity to dimensionality."** The metric is mathematically well-grounded (rotation-invariant, based on established statistics) and the paper provides both quantitative CR values and t-SNE visualizations as complementary evidence.

## Novel Insights

The Collapse Ratio metric provides a novel quantitative lens on representation collapse in temporal graph learning—a phenomenon where entities without training supervision drift toward degenerate embeddings. The ratio of generalized variances (log-det covariance) between emerging and known entity embeddings offers a rotation-invariant measure that could be adopted by future work on open-world graph learning. Additionally, the Observation that entities of similar semantic type share temporal interaction patterns (e.g., presidents follow visit–negotiation sequences across countries) suggests an inductive bias direction for temporal reasoning that goes beyond the paper's specific architectural contributions.

## Suggestions

- Add explicit handling for empty Interaction Chains: define a null token or fallback to static cluster prototypes when $|\mathcal{C}_q| = 0$, and document this in the method section and pseudocode.

- Clarify the purpose of the commitment loss (or remove it if it is non-functional), and verify in the code whether it contributes to training.

- For empty clusters at timestamp $t$, use the static codeword $\mathbf{c}_k$ as a fallback for $\mathbf{c}^{\text{dyn}}_k$, and state this clearly.

- Report Hits@1 for completeness and clarify which metric Table 2 reports.

- Add quantitative validation of the transferability assumption: compute intra-cluster pattern similarity or train a classifier to predict cluster membership from interaction sequences, providing evidence that textual clusters correlate with temporal dynamics.

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 6.0, 6.0]
Average score: 6.0
Binary outcome: Accept
