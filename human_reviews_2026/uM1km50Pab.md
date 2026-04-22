# Leto: Modeling Multivariate Time Series with Memorizing at Test Time

- Avg Score: 3.33
- Decision: Reject
- Scores: 4, 2, 4

## Abstract
Modeling multivariate time series data has been at the forefront of machine learning research efforts across diverse domains. However, effectively capturing dependencies across both time and variate dimensions, as well as temporal dynamics, have made this problem extremely challenging under realistic settings. The recent success of sequence models, such as Transformers, Convolutions, and Recurrent Neural Networks, in language modeling and computer vision tasks, has motivated various studies to adopt them for time series data. These models, however, are either: (1) natively designed for a univariate setup thus missing the the rich information that comes from the inter-dependencies of time and variate dimensions; (2) inefficient for long-range time series; and/or (3) propagating the prediction error over time. In this work, we present Leto, a native 2-dimensional memory module that takes the advantage of temporal inductive bias across time while maintaining the permutation equivariance of variates. Leto uses meta in-context memory modules to learn and memorize patterns across the time dimension, and simultaneously, incorporates information from other correlated variates, if needed. Our experimental evaluation shows the effectiveness of Leto on extensive and diverse benchmarks, including time series forecasting (short, long, and ultra-long), classification, and anomaly detection.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper introduces LETO, a 2D meta in-context memory architecture for multivariate time series modeling. Unlike traditional sequence models that process each variable independently or ignore inter-variate dependencies, LETO maintains temporal inductive bias while being permutation equivariant across variates. The core idea is to use interconnected memory modules that capture both temporal dynamics and cross-variable interactions through a shared meta-memory mechanism. Theoretical analysis (Theorem 1) suggests that such interconnected modules can represent higher-rank kernels more efficiently than independent ones. Extensive experiments on forecasting, classification, and anomaly detection benchmarks show that LETO achieves state-of-the-art performance across short-, long-, and ultra-long-term horizons.

### Strengths
The paper is generally well-organized and easy to follow, and it provides comprehensive experimental validation demonstrating the effectiveness of the proposed 2D meta-memory architecture.

### Weaknesses
1) The motivation for LETO is unclear. The paper asserts that existing multivariate time series models lack temporal inductive bias and inter-variate dependency modeling, yet prior architectures (e.g., TCNs, Transformers with positional encodings, TimesNet, Graph-based models) already incorporate both in different ways. The authors should clarify (1) what exact deficiency remains in these approaches, and (2) how LETO’s 2D memory introduces a distinct, structural bias rather than another learned mechanism.

2) Theorem 1 provides only a linear-algebraic argument suggesting that interconnected memory modules can represent higher-rank kernels with fewer parameters. However, this result holds only under linear recurrent assumptions and does not reflect the nonlinear, attention-based nature of LETO’s actual implementation. Moreover, the proof does not present an explicit constructive mapping or quantify expressivity beyond rank. Consequently, while the theorem offers an intuition for parameter efficiency, it does not constitute a rigorous theoretical justification of the proposed architecture’s behavior.

3) LETO’s meta-memory module may implicitly retain information beyond the fixed input window used for competing models. While the paper claims identical input–output settings across baselines, the presence of a persistent memory state effectively extends the model’s historical context. This raises fairness concerns, as LETO may have access to longer-term information than baselines restricted to feedforward input windows. The authors should clarify whether the meta-memory is reset between batches and whether the effective receptive field is comparable across models.

4) Although the paper benchmarks against a wide range of prior models, it does not provide diagnostic experiments or analyses that substantiate the claimed limitations of existing methods. The argument that previous architectures lack temporal inductive bias or inter-variate dependency modeling remains qualitative. There is no evidence demonstrating that such deficiencies are the cause of baseline underperformance. As a result, the experiments validate LETO’s performance but not its motivation.

5) The paper repeatedly refers to temporal “dynamics” and “state evolution,” suggesting an underlying dynamical-systems perspective. However, the theoretical section provides only a static, linear-rank argument and does not model or analyze any stochastic or continuous-time process. Is LETO intended to approximate an underlying dynamical system (e.g., through an implicit SDE/ODE formulation), or is the term “dynamics” purely metaphorical? If the former, could the authors formalize how the meta-memory update corresponds to a discrete approximation of such dynamics, and discuss its stability or stochastic robustness properties?


6) The paper contains several minor grammatical and formatting inconsistencies (e.g., duplicated words such as “the the,”in the abstract section, inconsistent use of “includes,” subject–verb mismatches, and irregular equation referencing). Terminology is occasionally inconsistent (“meta in-context memory,” “meta-memory,” etc.), and some sentences are overly long or rhetorically strong. A careful language and formatting revision would improve readability and professionalism.

### Questions
1) Existing multivariate time series models such as TCNs, TimesNet, and Transformer variants already incorporate both temporal inductive bias (via convolutions or positional encoding) and inter-variate modeling (via attention or graph structures). What specific structural deficiency do these methods still have that LETO’s 2D memory addresses? Please clarify how LETO’s “meta in-context memory” introduces a genuinely new inductive bias rather than another learned attention mechanism.

2) Theorem 1 provides a linear-rank argument under linear recurrence assumptions, but LETO is nonlinear and attention-based. How does this result theoretically connect to the actual architecture? Could the authors formalize or empirically validate the claimed expressivity or parameter-efficiency advantage beyond this simplified setting?

3) If the model indeed embodies implicit dynamics, could the authors clarify how its update rule relates to a discrete or stochastic dynamical formulation and whether any stability guarantees hold?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents an architecture, LETO, for multivariate time series analysis. The authors frame this as a "native 2D" model that is derived from the principles of meta-learning and memorizing at test time (TTM). The model attempts to properly handle the distinct properties of the time dimension (requiring temporal inductive bias) and the variate dimension (requiring permutation equivariance). While the initial perspective is interesting, the paper's central claims (regarding its "native 2D" nature, TTM/meta-learning foundation, and novelty) do not sufficiently align with its actual contributions.

### Strengths
1. **Interesting Conceptualization:** The paper correctly identifies a key challenge in multivariate time series: the temporal dimension requires a causal, sequential model (like an RNN), while the variate dimension requires a permutation-equivariant model (like an Attention mechanism). The proposal to build a hybrid "native 2D" architecture to address this is an interesting perspective.
2. **Novel Technical Component:** The technical solution to merge the RNN-like state with the attention-like mechanism, by using a kernelized attention approximated by its Taylor series, is a clever technical contribution.
3. **Extensive Experiments:** The model is evaluated across a wide range of tasks, including long-term and short-term forecasting, classification, and anomaly detection, demonstrating extensive empirical effort.

### Weaknesses
The paper suffers from significant weaknesses in its theoretical framing and the completeness of its comparisons.
1. **Hard to Follow:** 
   * Overall, the paper is difficult to read, lacks a clear structural diagram, and Figure 1 does not effectively and clearly illustrate the structure of the proposed model.
   * Figure 2 is too small, making it very difficult to read. While we understand this may be due to page limits, we strongly suggest replacing it with a clearer, larger figure in any revision.
   * The Appendix numbering is flawed: it jumps from Table 5 to Table 7, with Table 6 missing.
   * Table 7 is present in the Appendix but appears to be unreferenced in the text, leaving its context and purpose unclear.
2. **Mismatched Theoretical Framing (TTM/Meta-Learning):** The paper's claim on "meta-learning" and "memorizing at test time" as motivation for its architecture is unconvincing. By the paper's definition, any recurrent model with a state update (RNN, LSTM, or even modern SSMs) could be reframed as a "test-time learner" that adapts its internal "memory" to a new input. This is likely why the community does not generally apply this terminology to state-based sequence models, as it does not provide functional novelty or a clear distinction from standard recurrent processing. Unless the authors can justify the necessity of using the TTM/Meta-learning terminology and framework, or refactor the paper's title and narrative to de-emphasize TTM/Meta-learning and focus the contribution on a native 2D multi-variate temporal attention mechanism, a recommendation for acceptance will not be given. This is because the TTM/Meta-learning framing is not just mismatched but potentially misleading.
3. **Lack of Analytical Comparison (vs. iTransformer)**: The paper includes iTransformer in its experiments, acknowledging it as a baseline. However, the core idea of LETO's variate-handling mechanism (using an attention-like module for permutation equivariance across channels) is conceptually very similar to iTransformer's core idea (inverting the Transformer to apply attention across variates to model correlations). Given this strong conceptual overlap, the paper is missing a crucial analytical or theoretical discussion differentiating LETO's hybrid approach from iTransformer's pure-attention approach.
4. **Conditional Omission of SSM Baselines (Mamba):** The paper omits all modern SSMs (like Mamba) from its tables. We note the justification in Appendix D, which states that other literature (ModernTCN) has shown superiority over SSMs, and thus comparing to ModernTCN is sufficient. This reasoning is acceptable only if the authors confirm that the experimental setup (hyperparameters, training, testing, etc.) for ModernTCN used in this paper is generally identical to the setup in the paper of ModernTCN. Without this confirmation, the comparison is not valid. Even so, given that LETO is a recurrent-style model, a direct comparison to at least one modern SSM baseline is still encouraged for a more complete and convincing evaluation.
5. **Lack of Critical Ablation Studies:** The paper's core technical "trick" is the use of a Taylor series to approximate the softmax kernel. The choice of a 3rd-order approximation appears arbitrary. There is no ablation study on the order of this approximation (e.g., 1st vs. 2nd vs. 3rd order) to justify this key design choice. This makes the "principled derivation" from the TTM framework seem more like an ad-hoc engineering decision.

### Questions
See the Weaknesses.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces LETO, a native 2-dimensional memory-based architecture for modeling multivariate time series. LETO fuses two meta in-context memory modules: a recurrent temporal memory for capturing time dynamics and a permutation-equivariant attention-based memory for mixing information across variates. The paper provides a theoretically motivated formulation, implementation details for efficient parallelizable training, and comprehensive empirical evaluations across datasets for forecasting, classification, and anomaly detection. Ablation studies are presented to justify design choices.

### Strengths
1、The paper tackles a significant and longstanding challenge in time series modeling: capturing both long-range temporal and cross-variate dependencies without sacrificing efficiency or robustness.

2、 There are quite a few nice illustrations.

3、 This work focuses on an important problem that could have real-world applications.

4、 The figures and tables used in this work are clear and easy to read.

### Weaknesses
1、As a paper submitted to ICLR 2026, the comparison baselines chosen by the authors are relatively outdated, lacking evaluations against the latest methods from 2025. This limitation undermines the credibility of the reported performance gains and raises concerns about the reliability of the experimental conclusions.

2、The Related Work section does not adequately cover recent advances in multivariate modeling. In particular, it lacks discussion of several representative studies published in 2024–2025 that have significantly advanced this area. As a result, the literature review appears incomplete and fails to clearly position the proposed work within the current state of research.

3、While high-level architectural design is summarized (see Figure 1 on Page 5), there is a lack of granular description regarding data flow, batching, and precise ordering of operations—especially important for readers seeking reproducibility or trying to adopt the approach in their own pipelines, especially when parallelism is highlighted as a main practical advantage.

4、While the reported results are competitive, the ablation study (Table 4, Page 8) is limited in terms of scope. The architectural variants removed (“cross-variate attention”, “linear attention”, “weighted gating”) are natural ablations, but ablations for Taylor expansion order, chunk size in parallelization, or the extent of cross-memory coupling are missing. This weakens the argument for the chosen design.

5、In the NIPS 2024 workshop[1], some researchers pointed out that current methods sometimes use the "drop-last" trick [2] to improve performance. Therefore, It is recommended that you clarify whether the "drop - last" operation [2] was used in your paper in the implementation details section of your paper for transparency.

[1] Fundamental limitations of foundational forecasting models: The need for multimodality and rigorous evaluation

[2] TFB: Towards Comprehensive and Fair Benchmarking of Time Series Forecasting Methods

If my problem is solved, I will improve my score!

### Questions
Please see the weaknesses！

### Soundness
2

### Presentation
2

### Contribution
2
