# Progressive Memory Transformers: Memory-Aware Attention for Time Series

- Avg Score: 5.33
- Decision: Reject
- Scores: 4, 6, 6

## Abstract
Self-supervised learning has become the de‑facto strategy for time‑series domains where labeled data are scarce, yet most existing objectives emphasize \emph{either} local continuity \emph{or} global shape, seldom both.  We introduce \textbf{Progressive Memory Transformer} (PMT), a memory-augmented transformer backbone that maintains a writeable memory bank across overlapping windows, allowing representations to accumulate evidence from short, medium, and long horizons without re‑reading the entire sequence.  On top of our proposed memory-aware attention, we formulate a hierarchical contrastive protocol that aligns embeddings at three complementary granularities---tokens, windows, and full sequences---through a token-window Gaussian loss, a memory‑state loss, and a global \texttt{[CLS]} loss.  Together, PMT and these multi‑scale objectives yield a task‑agnostic model for time‑series data, providing strong features even when only $1$--$5\%$ of labels are available.  We validate the approach on seven UCR/UEA/UCI benchmarks on classification tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes Progressive Memory Transformer (PMT), a sliding-window Transformer architecture that maintains a writable memory bank for each window and layer. Memory states are updated via reset/carry gates and propagated across both time (window-to-window) and depth (layer-to-layer). The model is trained using three contrastive losses at different temporal scales: a hierarchical Gaussian contrastive loss (HGCL) for local structure, a progressive contrastive loss (PCL) for memory supervision, and an instance-level contrastive loss (ICL) for global alignment. Experiments on several small-scale time-series classification datasets (UCR/UEA/UCI) under low-label settings (1–5%) show moderate improvements over baselines.

### Strengths
- The paper is clearly written, with well-explained figures and masking diagrams.
- The proposed memory mechanism is conceptually sound: instead of read-only caches (e.g., Transformer-XL), it introduces writable, gate-controlled slots aligned to each sliding window.
- The design of contrastive losses targets local, intermediate, and global temporal structures.
- Visualization of memory activations and ablations of loss weights are helpful to understand how each component behaves.
- Computational analysis is provided.

### Weaknesses
- The novelty needs to be clarified. The paper mainly combines writable memory (as in Transformer-XL, Compressive Transformer, and Titans) with hierarchical contrastive objectives.

- Evaluation covers only seven small classification datasets with short sequences.

- Transfer learning under in- and cross-domain scenarios is not tested.

### Questions
- Could you show ablations replacing writable memory with a read-only cache (Transformer-XL style) to isolate the benefit of “writability”?

- How sensitive is the model to patch size or window length?

- Do the authors think any experiments on broader tasks are reasonable? i.e., softCLT is tested on forecasting.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces Progressive Memory Transformer (PMT), a lightweight transformer architecture for self-supervised time-series representation learning. Unlike conventional stateless transformers, PMT maintains a writable, hierarchical memory across overlapping temporal windows, allowing representations to accumulate information progressively rather than repeatedly re-encoding past segments. In particular, the core of the proposed PMT is Progressive Memory Attention (PMA), which combines causal attention with adaptive gating and reset mechanisms, enabling selective retention, refinement, or overwriting of context. Although this design is reminiscent of an LSTM, the authors do not appear to draw any explicit connection to it. On top of this architecture, the authors propose a three-level contrastive learning framework to capture local nuance, mid-range motifs, and global semantics: i) Hierarchical Gaussian Contrastive Loss – enforces smoothness among nearby tokens and overlapping windows; ii) PMA Contrastive Loss (PCL) – supervises writable memory tokens to capture mid-range temporal motifs; iii) Instance Contrastive Loss (ICL) – aligns sequence-level `[CLS]` representations for global semantics.

### Strengths
- The proposed memory-augmented transformer for time-series SSL is a timely and well-motivated innovation. The proposed PMT ticks a few boxes of the memory-augmented transformers by maintaining a writable, hierarchical memory across overlapping temporal windows. The authors further propose a multi-scale contrastive loss to capture local nuance, mid-range motifs, and global semantics in one framework.
- Empirical evaluations on UCR/UEA/UCI benchmarks show promising performance, especially in the few-label regime. 
- The paper is, in general, well-structured. 
- The code is available.

### Weaknesses
- The design of PMA bears a strong resemblance to LSTM architectures. However, the authors do not appear to explicitly appreciate or elaborate this connection. Similar to LSTM’s hidden and cell states, PMT maintains a persistent memory that is progressively updated across temporal windows, enabling earlier segments to influence subsequent ones. It also incorporates learnable gating mechanisms that determine how much prior memory is retained, refined, or reset—analogous to the forget, input, and output gates in LSTMs. However, unlike LSTM, PMT appears less effective at capturing long-range dependencies, (as evidenced by its weaker performance on the Ford A/B datasets) due to its dependency on patch size. 

- Following my previous point, it might also be interesting to see the ablations on different patch sizes, at least for Ford A/B datasets. 
- Comparisons to prior arts, e.g., Titans, Transformer-XL, on the same setups would consolidate the paper more.
- The evaluations are limited to the classification task. What about other TS tasks, such as forecasting and anomaly detection ?
- Although “lightweight,” the hierarchical PMA still appears to be heavy. It might be more interesting to see comparisons to lightweight CNN-based SSL baselines.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes the Progressive Memory Transformer (PMT), a self-supervised architecture for time-series representation learning.
PMT introduces writable, hierarchical memory units that propagate contextual information across overlapping windows, enabling progressive context aggregation without re-encoding the entire sequence.
Three complementary contrastive objectives are defined:
(1) HGCL (Hierarchical Gaussian Contrastive Loss) – encourages local smoothness within windows.
(2) PCL (PMA Contrastive Loss) – aligns intermediate-level memory semantics.
(3) ICL (Instance Contrastive Loss) – preserves global sequence consistency.
Experiments on seven benchmarks (HAR, Epilepsy, Wafer, FordA/B, POC, ElectricDevices, etc.) show strong results, especially under low-label (1–5%) regimes.

### Strengths
1. Architectural novelty:
The proposed progressive memory attention (PMA) extends Transformer-XL with writable memory propagation across both temporal windows and model layers. The idea of dynamically updating memory states is original and technically sound.

2. Hierarchical learning objective:
The combination of HGCL, PCL, and ICL aligns well with the model hierarchy (token → window → sequence), achieving coherent multi-scale supervision.

3. Comprehensive evaluation:
The model is validated on seven diverse benchmarks and includes reasonable ablations on the loss components and qualitative memory visualizations.

4. Interpretability attempt:
Visualizations of middle-layer memory embeddings illustrate the progressive clustering of classes, partially supporting the claimed representational hierarchy.

### Weaknesses
(1). Limited and inconsistent empirical superiority.
PMT is not consistently better than strong baselines such as TS2Vec and SoftCLT.
Performance drops on datasets such as FordA, FordB, and ElectricDevices are attributed to "patchification loss," yet this claim is speculative. No patch-size or stride sensitivity analysis supports it.

(2). Relation to recent global-token approaches.
Recent work, for example "Sequence Complementor: Complementing Transformers for Time Series Forecasting with Learnable Sequences" (AAAI 2025), introduces learnable global tokens that complement local context and achieve similar goals of long-range dependency modeling.PMT’s progressive memory resembles such global or complement tokens, yet the paper does not clearly distinguish whether writable memory offers capabilities beyond static learnable tokens or Perceiver-style latent arrays.

(3). The proposed writable memories, overlap pooling, and gating introduce extra computation, but there is no quantitative comparison of FLOPs, memory footprint, or runtime versus other baseline methods (Transformer-XL or PatchTST).

### Questions
see weakness above

### Soundness
3

### Presentation
3

### Contribution
3
