=== CALIBRATION EXAMPLE 17 ===

# Final Consolidated Review
## Summary
This paper proposes a scalar evaluation metric for continuous-time dynamic graph generative models based on two-stage random projections inspired by the Johnson–Lindenstrauss (JL) lemma. The method embeds variable-length per-node event histories and then aggregates node embeddings into a fixed-size graph representation, enabling cosine-distance comparison between whole dynamic graphs; the paper also introduces a perturbation-based benchmark for assessing CTDG metrics along fidelity, diversity, sample efficiency, and runtime axes.

## Strengths
- **The paper isolates a real blind spot in current CTDG evaluation: sensitivity to joint topology/feature/temporal perturbations.** In particular, the event-permutation experiment is specific and compelling: topological baselines are insensitive because topology is preserved, and feature-marginal baselines are insensitive because the feature multiset is preserved, whereas the proposed JL-metric achieves a median Spearman correlation of **0.988** (Table 1). This is concrete evidence that the proposed representation captures some interaction between event content and temporal/topological context that the baselines miss.
- **The paper contributes a reasonably coherent evaluation protocol for CTDG metrics rather than just a new score.** The fidelity/diversity/sample-efficiency/runtime framework is adapted carefully to the single-graph-many-events CTDG setting, and the perturbation suite is more thoughtful than a simple “add random noise” stress test: edge rewiring, time perturbation, event permutation, mode dropping, and mode collapse each target a different failure mode.
- **The method is practically efficient relative to snapshot-based baselines, and the efficiency claim is supported by the design.** The use of structured random matrices instead of explicit dense projections is not just generic engineering: it directly supports the paper’s goal of avoiding explicit snapshot instantiation, and the reported runtime gap versus snapshot-based graph statistics is substantial (Table 1).
- **The paper’s central practical goal—a unified scalar metric for CTDG comparison—is well matched to the proposed construction.** Existing baselines in the paper indeed fragment evaluation across multiple handcrafted statistics and separate feature-vs-topology measures, whereas the proposed metric yields one score from a single graph-level representation.

## Weaknesses

###: Fatal
None.

### Major:
- **The paper does not validate its central claim as a metric for evaluating generative models, only as a metric sensitive to synthetic perturbations.** The title, abstract, and introduction repeatedly position this as a “quality metric for evaluating generative models of dynamic graphs,” but Section 4 evaluates only perturbed copies of real graphs, not outputs from actual CTDG generative models. This is a meaningful gap: monotonicity under controlled corruptions is useful, but it does not establish that the metric ranks real DGGMs sensibly, distinguishes stronger from weaker generators, or aligns with any downstream notion of sample quality. For a metric paper at ICLR, this missing validation materially limits significance.
- **The representation discards important adjacency information, weakening the claim of jointly modeling graph topology and features in a general sense.** In Section 3, each node history is built from the time-ordered concatenation of events the node participates in, but the event representation is explicitly simplified to  
  \[
  \tilde c(t_i)=(t_i,\mathbf e_{\text{src,dst}}(t_i)),
  \]
  after “dropping the node identifier.” As written, this means a node embedding retains timestamp and edge-feature information for its incident events, but not the identity of the counterpart node in each interaction. That omission matters because partner assignment is central topological information. The experiments do show sensitivity to some topological perturbations (e.g., rewiring), so the method is not topology-blind; however, the paper’s stronger claim that it generally captures topology and feature-topology dependencies is overstated given that explicit neighborhood identity is removed from the formal representation.
- **The JL-based conceptual justification is substantially weaker than the paper’s framing suggests.** Section 3 is explicit that the connection is a hypothesis (“we argue,” “we posit”), not a theorem, but the paper still leans heavily on JL as the conceptual reason the metric should work. The mismatch is that standard JL guarantees apply to a finite set of points in a common Euclidean space under one random linear map, whereas the method uses variable-length node histories, “ignores unused rows” of \(W_1\), and applies a second projection \(W_2\) across variable-size node sets. No target dynamic-graph distance is defined and no preservation theorem is proved for the actual two-stage construction. This does not make the empirical method invalid, but it does mean the main theoretical narrative is more speculative than established.
- **The method’s handling of variable-length inputs raises unresolved theoretical and methodological questions.** The paper claims that “JL embedding quality is agnostic to vector length,” then uses a projection matrix \(W_1^{M\times n}\) where shorter vectors simply use fewer rows. That is not a standard statement of the JL lemma; more importantly, the paper does not analyze what similarity notion is preserved when different nodes/graphs effectively experience different truncated submaps. The same issue reappears in the second projection over varying numbers of nodes. Since this variable-size handling is core to the method, the lack of analysis is a substantive weakness rather than a minor missing proof.
- **Permutation invariance over node ordering is not specified, even though the second projection depends on arranging node embeddings into a graph-level object.** Section 3 says each graph is transformed into \(\tilde{\mathcal G}=\{\tilde{\mathbf x}_1,\dots,\tilde{\mathbf x}_o\}\) and then compared by Frobenius cosine distance, but the paper does not clearly state how nodes are ordered before applying \(W_2\), nor why the resulting metric is invariant or robust to relabeling. For graph data, this is not a cosmetic detail: if ordering is arbitrary, the graph-level representation may depend on indexing conventions rather than structure.

### Minor
- **The formal CTDG definition is narrower than the surrounding prose.** Section 2.1 says CTDG events can include node or edge creation/deletion and feature changes, but Equation (1) instantiates events only as timestamped directed edge events with edge features. This does not invalidate the experiments, but it weakens the paper’s generality claims about dynamic graphs more broadly.
- **Some critiques of baseline metrics are stated too categorically.** For example, the paper says classical estimators such as KS/MMD “assume an i.i.d. relationship” and are therefore challenged in this setting. That concern is directionally reasonable, but the wording overstates the case: even if their formal interpretation is imperfect for temporally dependent descriptors, such distances can still be useful heuristics. The paper would be stronger if it distinguished “not statistically ideal” from “empirically ineffective.”
- **The computational comparison is somewhat tilted toward the proposed method by evaluating snapshot baselines only at the Nyquist rate.** The paper does justify this choice as the lossless resolution, so this is not an unfair comparison in the usual sense, but it should be presented more carefully as a strong-form baseline regime rather than the only practically relevant one.
- **The diversity benchmark depends on TGN embeddings to define modes, which introduces model dependence into an otherwise “application-agnostic” evaluation pipeline.** This is not unreasonable as a benchmark construction, but it should be acknowledged more explicitly.

### Trivial
- **Cosine distance on the final graph embeddings is only lightly justified.** It is a reasonable default, but the paper provides little argument for why this is the right comparison operator beyond convenience and familiarity.
- **Hyperparameter robustness for the embedding dimensions \(n\) and \(o\) is not analyzed in the main paper.** The paper mentions grid search in Appendix D, but the main text would benefit from at least a brief robustness summary given that these dimensions are central to the method.

## Nice-to-Haves
- Evaluate the metric on outputs from at least two actual CTDG generative models and show whether rankings align with expected model quality or with a reasonable consensus of existing metrics.
- Add an ablation probing **temporal-order sensitivity** directly, e.g., shuffle event order within node histories while preserving event multisets.
- Compare the second random projection \(W_2\) against simpler graph-level aggregations such as mean/sum pooling over node embeddings.
- Provide per-dataset score-vs-perturbation curves in addition to aggregated violin plots, to expose dataset-specific failures or brittleness.
- Clarify whether and how the method can incorporate richer event types, node features, deletions, or explicit counterpart-node identity without losing its efficiency advantages.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **Claims imported from unrelated papers/reviews about node2vec-style walks, Transformers, temporal random-walk bias, or the paper “not even being about temporal generation.”** These do not correspond to the submitted paper and are factually inapplicable.
- **Pure reproducibility complaints about omitted low-level implementation details or release status.** The paper provides a repository link and cites all referenced artifacts; such criticisms are not appropriate here.
- **Generic strength claims such as “the paper is well-written” or “the topic is important.”** These are too generic to be meaningful strengths.
- **Requests for statistical significance testing as a core flaw.** Confidence intervals or hypothesis tests would be useful, but for this style of empirical metric benchmark they are a nice-to-have rather than a substantive defect.
- **Complaints that the comparison is unfair because baselines are expensive at the chosen snapshot resolution.** The paper explicitly evaluates static metrics at the Nyquist rate to avoid information loss; while this accentuates the efficiency advantage, the asymmetry does not favor the authors’ method in a way that invalidates the comparison.

## Novel Insights
The most important synthesis across the reviews is that the paper is strongest not as a theoretically grounded JL result, but as an empirical demonstration that **random-projection summaries of event histories can detect perturbations involving feature-topology coupling that standard CTDG metrics miss**. At the same time, the exact construction reveals a key conceptual limitation: by removing counterpart-node identity from each event, the method behaves more like a metric over **incident event streams per node** than a full dynamic-graph metric in the strict graph-theoretic sense. This suggests the paper’s true contribution is narrower and more precise than its framing: it offers an efficient scalar similarity measure for dynamic interaction traces, with promising sensitivity properties, but not yet a fully justified general-purpose quality metric for CTDG generative models.

## Suggestions
- **Narrow or recalibrate the main claim** unless you can add stronger validation: present the method as an empirically effective CTDG similarity metric under perturbation-based evaluation, rather than fully established DGGM evaluation metric.
- **Add experiments on actual generated graphs** from existing CTDG generators; even a small model-ranking study would substantially strengthen practical significance.
- **Address the representation gap directly**: either incorporate counterpart-node identity into \(\tilde c(t_i)\) or explicitly discuss the resulting limitation and test adversarial rewiring cases that preserve per-node timestamps/features while changing partners.
- **Clarify permutation handling and invariance** for the second-stage graph embedding. State the node ordering convention and test sensitivity to relabeling.
- **Strengthen the theory section by downgrading claims or adding a formal proposition.** If a full theorem is out of reach, a careful statement of what the two-stage random map can and cannot be expected to preserve would still improve technical soundness.
- **Include temporal-order ablations and \(W_2\) ablations** to demonstrate that the proposed architecture, not just random compression in general, is responsible for the reported gains.
- **Be more precise about scope** in the CTDG formalism: specify whether the method currently handles only timestamped edge events with edge features, or general dynamic graphs with node features and deletion events.

# Actual Human Scores
Individual reviewer scores: [8.0, 6.0, 8.0, 8.0]
Average score: 7.5
Binary outcome: Accept
