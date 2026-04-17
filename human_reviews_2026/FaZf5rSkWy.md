# Dynamic Relational Priming Improves Transformer in Multivariate Time Series

- Decision: Reject
- Scores: 4, 4, 6

## Abstract
Standard attention mechanisms in transformers employ static token representations that remain unchanged across all pair-wise computations in each layer. This limits their representational alignment with the potentially diverse relational dynamics of each token-pair interaction. While they excel in domains with relatively homogeneous relationships, standard attention's static relational learning struggles to capture the heterogeneous inter-channel dependencies of multivariate time series (MTS) data where different channel-pair interactions within a single system may be governed by entirely different physical laws or temporal dynamics. To better align the attention mechanism for such domain phenomena, we propose attention with dynamic relational priming (prime attention). Unlike standard attention where each token presents an identical representation across all of its pair-wise interactions, prime attention tailors each token dynamically (or per interaction) through learnable modulations to best capture the unique relational dynamics of each token pair, optimizing each pair-wise interaction for that specific relationship. This representational plasticity of prime attention enables effective extraction of relationship-specific information in MTS while maintaining the same asymptotic computational complexity as standard attention. Our results demonstrate that prime attention consistently outperforms standard attention across benchmarks, achieving up to 6.5% improvement in forecasting accuracy. In addition, we find that prime attention achieves comparable or superior performance using up to 40% less sequence length compared to standard attention, further demonstrating its superior relational modeling capabilities.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Prime Attention, a modified attention mechanism for multivariate time series (MTS) forecasting. Standard attention uses fixed token representations for all pairwise interactions, termed static relational learning by the authors. Prime Attention introduces dynamic relational priming through learnable, pair-specific filters. The authors formalize this distinction between static and dynamic approaches in a theoretical framework.

### Strengths
- The conceptual distinction between static and dynamic relational learning is clear and useful, providing good motivation for the method.

- The method shows consistent but modest improvements on heterogeneous MTS datasets like Weather and Solar-Energy.

- The module can replace standard attention in multiple backbone architectures, demonstrating modularity and broad applicability.

### Weaknesses
- Scalability is a major concern. The $O(N^2 \times d)$ memory overhead for pairwise filters ($F_{i,j}$) makes the method impractical for high-dimensional MTS. Claims about maintaining asymptotic complexity are misleading. 

- The paper mentions a GNN-based sparsification strategy in the appendix to address scalability  but provides no empirical evaluation. Its viability and performance trade-offs remain unknown.

- Implementation details for the filters ($F_{i,j}$) are unclear. How are filters parameterized24? Was the FFT-based initialization actually used in experiments25252525?

### Questions
- While authors argue its improvement, but results lack statistical rigor. Are the reported performance improvements statistically significant? Please report results averaged over multiple random seeds with variance or confidence intervals.

- Can you provide a quantitative evaluation of the GNN-based sparsification strategy from Appendix? What is the resulting performance versus memory trade-off?

- Was the FFT-based lead-lag initialization (Equation 13) used in the main experiments? If so, for which datasets, and what is its impact compared to random initialization? 

- How are the $F_{i,j}$ filters parameterized for high-dimensional datasets? Is the generating MLP shared across all pairs, or are parameters learned for each pair independently?

- Why does the method fail to produce significant gains on homogeneous datasets like ECL and Traffic? What properties of these datasets limit the method's effectiveness?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper aims to address the "domain mismatch" problem when applying Transformers to MTS forecasting. The authors argue that standard attention's "static relational learning" (i.e., fixed K/V representations) fails to capture the inherent "heterogeneous" dependencies in MTS data. To this end, the paper proposes "dynamic relational priming attention" (prime attention), the core of which is to introduce a learnable, pair-specific $(i, j)$ "primer" $\mathcal{F}_{i,j}$. This primer dynamically modulates the K/V vectors via element-wise multiplication ($\odot$) *before* the attention calculation. Experiments show that this mechanism improves performance when replacing SOTA models, especially on datasets with high heterogeneity.

### Strengths
* The paper's diagnosis of the conflict between "static K/V representations" and the "heterogeneity" of MTS data is both sharp and reasonable. Linking the limitations of the attention mechanism to the specific properties of the domain (heterogeneity) provides a solid foundation for the paper.

* A key strength of this paper is that its experimental design is tightly coupled with its core hypothesis. The results strongly support this hypothesis: the method shows significant improvements on datasets considered "heterogeneous" (like Weather and Solar) but "marginal" improvements on those considered "homogeneous" (like ECL and Traffic). This validates that the method is indeed solving the specific problem it claims to address.

* **Insightful Efficiency and Mechanism Analysis**:
    * The sequence length efficiency analysis (Fig. 1) is a highlight, suggesting that Prime Attention, as an effective inductive bias, may allow the model to learn relationships more efficiently.
    * The attention map analysis in Appendix G.3 (Fig. 6, 7) is particularly convincing. It demonstrates the model's ability to adapt based on data characteristics: on the Solar dataset, it learns a "channel-independent"-like behavior, while on the Exchange dataset, it learns to strengthen cross-channel interactions. This powerfully demonstrates the "dynamic" and adaptive nature of the mechanism.

### Weaknesses
* **Over-reliance on the Appendix for Structural Integrity**:
    This is a notable structural issue. The main paper is not fully self-contained. Some of the most critical evidence supporting the core arguments is placed entirely in the appendix. For example, the **attention map analysis (Appendix G.3)**—which should be a central figure in the main body—as well as key ablation studies (G.1) and empirical complexity analysis (G.2), are not in the main text. This makes it difficult for a reader to coherently evaluate the paper's core claims within the main body and weakens its persuasiveness.

* **Insufficient Justification for the Novelty of the Core Mechanism**:
    1.  **Gating Mechanism**: The core mechanism ($\odot \mathcal{F}_{i,j}$) is a form of gating. However, the paper fails to sufficiently argue why this specific pair-wise gating (which occurs *inside* the attention calculation and carries an $\mathcal{O}(N^2)$ cost) is a superior design choice compared to other "global" gating strategies in the field (e.g., fusing "interactive" and "non-interactive" representations *outside* the attention calculation).
    2.  **Comparison with MHA**: The paper categorizes standard MHA as "static" in Section 4. This argument may be an oversimplification, as it overlooks that MHA itself is a mechanism for capturing multiple, parallel dynamic patterns. The paper **lacks a direct experimental comparison** to demonstrate that its explicit $\mathcal{F}_{i,j}$ parameterization (with $\mathcal{O}(N^2)$ cost) has a fundamental advantage over a standard MHA with an equivalent parameter count (and its *implicit* modeling capabilities).

### Questions
1. **Pair-Specific vs. Global Gating**: Can you provide experimental or theoretical justification for why your pair-specific gating mechanism $\mathcal{F}_{i,j}$ outperforms global gating strategies, such as those in [1]?

2. **Comparison with Multi-Head Attention**: Could you conduct a fair comparison between Prime Attention and standard Multi-Head Attention (MHA) with an equivalent total parameter count (e.g., by adjusting $d_{model}$ or the number of heads $h$) to demonstrate the necessity of $\mathcal{F}_{i,j}$?

3. **Marginal Improvement on Homogeneous Datasets**: Given the marginal improvement on homogeneous datasets like ECL, do the learned $\mathcal{F}_{i,j}$ primers approach an identity vector, effectively making Prime Attention degenerate into standard attention? An analysis of the $\mathcal{F}$ matrix in this case would be valuable.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper identifies a fundamental mismatch between the static relational learning of standard attention mechanisms and the heterogeneous nature of multivariate time series (MTS) data. The authors argue that in domains like NLP, token relationships are relatively homogeneous (e.g., semantic), whereas in MTS, different pairs of channels (e.g., temperature-pressure vs. temperature-flow) can be governed by entirely different physical laws and temporal dynamics (e.g., lead-lag vs. instantaneous correlation). Standard attention, which uses a single, fixed token representation (K, V) for all pairwise interactions, is ill-suited for this heterogeneity.
To address this, the paper introduces a novel theoretical framework to formalize this problem, defining "Static Relational Learning" (where a token's representation is independent of its interaction partner) and "Dynamic Relational Learning" (where a token's representation is conditioned on its partner). The paper provides strong empirical validation, showing that simply swapping standard attention for prime attention in SOTA Transformer backbones (iTransformer, Timer-XL, FreDF) yields consistent and significant forecasting improvements (up to 6.5%).

### Strengths
S1. This paper is well-motivated and solves the timely and important problem exist in the time-series forecasting domain

S2. The paper is well-written and easy to follow

S3. The experimental results show that proposed methods significantly improved the performance, which empirically validate their hypothesis.

### Weaknesses
W1. The most significant practical drawback is the memory complexity of the primer $\mathcal{F}$, which is $\mathcal{O}(N^2 \times d_{\text{model}})$ where $N$ is the number of channels. This scales quadratically increase with the number of variables and is a major limitation for high-dimensional datasets (e.g., Traffic N=862). The paper defers the solution (a GNN-based sparsification) entirely to Appendix D.2. This is a critical methodological detail for practical application and should be at least summarized in the main paper.


W2. The main paper (Sec 5) presents random initialization and lead-lag (FFT-based) initialization as options for $\mathcal{F}$. However, the ablation study in Appendix G.1 (Fig 3) strongly suggests that "Full" initialization (using lead-lag and other features) is significantly better than "Random." This dependency on pre-calculated features (similar to LIFT) somewhat undermines the "end-to-end" learning narrative and should be discussed more transparently.

### Questions
Q1. For the high-dimensional datasets (Traffic N=862, ECL N=321), was the full $\mathcal{O}(N^2)$ primer $\mathcal{F}$ instantiated, or was the GNN-based sparsification (Appendix D.2) used? If the sparse version was used, this is a critical experimental detail that should be in the main paper, as it changes the method being evaluated.

Q2. Could you please clarify the performance of prime attention with random initialization in the main results tables (Table 1 & 2)? Figure 3 (Appendix) suggests its performance is notably worse than the "Full" initialization. How much of the performance gain is attributable to the dynamic priming mechanism itself, versus the injection of pre-calculated FFT features at initialization?

Q3. The primer $\mathcal{F}_{i,j}$ is applied to both the Key (K) and Value (V) vectors. Did you ablate the impact of applying it only to K (modulating attention scores) versus only to V (modulating aggregated values)? Is the dual application necessary?

### Soundness
3

### Presentation
3

### Contribution
3
