# CGTFra: General Graph Transformer Framework for Consistent Inter-series Dependency Modeling in Multivariate Time Series

- Decision: Reject
- Scores: 2, 4, 6, 4

## Abstract
Transformers have emerged as dominant predictors in multivariate time series forecasting (MTSF), prompting an in-depth investigation into their limitations within this application. Firstly, the conventional temporal information for timestamp in MTSF suffers from the unavailability of future timestamps and the diversity of timestamp formats across real-world datasets, which poses a significant practical challenge and necessitates cumbersome adjustments for a unified forecasting model. Secondly, existing Variate Transformers, such as iTransformer, typically model inter-variate dependencies (IVD) predominantly within shallow self-attention layers, neglecting the critical requirement for deep-layer IVD modeling, thereby causing dependency information loss and difficulties in model optimization. We refer to this phenomenon as inconsistent IVD modeling. To address these limitations, CGTFra, a Graph Transformer framework, is designed to promote consistent IVD modeling. Specifically, we introduce a frequency-domain masking and resampling method for feature enhancement that preserves periodic characteristics in the frequency domain. Additionally, by comprehensive analysis of the distinctions and connections between self-attention mechanisms of Variate Transformers and Graph Neural Networks (GNNs) in capturing IVD, a dynamic graph learning framework is integrated into the Transformer to explicitly model IVD in deep network layer. Crucially, we then propose a consistency-constrained alignment to strengthen the network to learn more robust IVD and temporal feature representations.  The core design philosophy of CGTFra can be integrated into any existing Variate Transformer-based framework and CGTFra demonstrates superior predictive performance across 13 long- and short-term datasets with high computational efficiency. Code is available at https://anonymous.4open.science/r/CGTFra.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes CGTFra, a General Graph Transformer Framework for multivariate time series forecasting. The core idea is to model both the temporal dependencies and inter-variable correlations using a unified graph-transformer paradigm. Experiments on several real-world datasets demonstrate that CGTFra achieves consistent improvements over state-of-the-art baselines.

### Strengths
1. This paper offers a new perspective on modeling inter-variable dependencies (IVD), providing fresh insights that could inspire future research in multivariate time series forecasting.

2. The experimental section compares CGTFra with a wide range of strong baselines, and the results consistently show performance gains on multiple benchmark datasets.

### Weaknesses
1. The paper incorrectly states that iTransformer employs positional encoding. In fact, iTransformer does not use positional encodings, as it models temporal order implicitly within feature embeddings rather than through token positions.

2. The paper also misinterprets the role of Feed-Forward Networks (FFNs) in the Transformer architecture. FFNs do not explicitly model intra-series dependencies; instead, they act as nonlinear mappings that refine representations after the attention operation, capturing temporal relationships between the input and predicted future values.

### Questions
1. In some datasets (e.g., ETT), the variables appear to be largely independent. Why is the Inter-Variable Dependency (IVD) module necessary in such cases?

2. What are the specific experimental settings used to generate the results in Figure 2?

3. Can the FMR be combined with architectures other than Transformers? In other words, is FMR a model-agnostic module or specifically tailored to Transformer-based frameworks?

### Soundness
2

### Presentation
2

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
This paper proposes a Graph Transformer framework CGTFra with adaptive frequency masking and resampling method and dynamic graph learning framework, which can diminish the importance of timestamps and promote consistent inter-variate dependencies (IVD) modeling. Experiments demonstrate the effectiveness of CGTFra.

### Strengths
1.Integrating the dynamic graph learning and consistency alignment loss to promote the modeling of consistent IVD is interesting.

2.The experiments are presented in considerable detail.

### Weaknesses
1.The dynamic graph learning lacks innovation and appears to be a standard adaptive graph construction method, which is widely explored in prior works. The authors should clarify the relationship between CGTFra and these works to further emphasize their own contribution.

2.The claim that IVD are modeled exclusively in shallow layers (on line 104) is unconvincing, as stacking multiple attention layers is a straightforward way to model IVD in deep layers. This oversight weakens the motivation for promoting consistent IVD modeling. The authors should clarify the oversight further to avoid misunderstanding.

3.Some visual comparisons should be quantified, as the claimed improvements are often subtle and difficult to assess from the plots alone, e.g., the claims on lines 143, 170, and 375. These claims should be backed by quantitative data, e.g., percentage gains, to make the comparisons clear.

4.The paper has some weaknesses in the experiments, which are not convincing enough:

(1)Considering CGTFra is a graph Transformer framework, more GNN-based models and even hypergraph-based models should be compared to further validate the effectiveness of CGTFra, e.g., Ada-MSHyper [1] and MTSF-DG [2].

(2)There are some overstatements and factual errors in the experimental analysis. For example, the claim that CGTFra consistently exhibits enhanced performance on ETT and solar datasets (on line 360) seems to be an overstatement. According to Table 1, CGTFra is actually outperformed by FilterNet [3] on both ETTm1 and ETTm2 datasets in terms of MSE. The claim that introducing DGL results in performance degradation on ECL (on line 438) seems to conflict with the results of Table 4. The authors should thoroughly review the analysis.

(3)There seem to be several inconsistencies in the bolding of results in Tables 2 and 3. For example, “iInformer + Solar” in Table 3, the best results of MAE are not bolded. The authors should carefully check all tables to avoid these mistakes.

[1]Shang Z, Chen L, Wu B, et al. Ada-MSHyper: Adaptive multi-scale hypergraph Transformer for time series forecasting. NIPS 2024.
[2]Zhao K, Guo C, Cheng Y, et al. Multiple time series forecasting with dynamic graph modeling. VLDB 2024.
[3]Yi K, Fei J, Zhang Q, et al. FilterNet: Harnessing frequency filters for time series forecasting. NIPS 2024.

### Questions
1.Some notations and formulas are confusing. For example, in the definition of MTS on line 263, f(t) often represents values at time t, why define f(t) as a 2D matrix? On Formula 1, the l in the summation is missing. On Formulas 5 and 6, are these trainable parameters the same? If not, why use the same notations? The Formula 7 seems incorrect for producing the described MCM on line 319, as the concat operation is missing. On Formula 8, the mathematical definition of KL divergence should be used instead of the engineering-style. Why calculate the KL divergence between P and Q instead of Q and P?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper addresses two limitations of Transformers in multivariate time series forecasting by proposing the CGTFra framework: (1) it introduces frequency-domain masking and resampling methods to replace positional encoding, thereby reducing dependence on timestamp information; (2) it incorporates a dynamic graph learning framework to explicitly model inter-variable dependencies in deeper network layers, addressing the limitation that existing methods only capture dependencies in shallow self-attention layers. Additionally, this paper is the first to propose a consistency alignment loss to constrain the dependency structures learned in both shallow and deep layers. The authors validate the effectiveness of their approach across 13 datasets, though some theoretical justifications and technical details require further elaboration.

### Strengths
1. The problem motivation is clear and well-supported by empirical evidence.
2. The authors are the first to propose modeling the dependency relationship between shallow and deep representations from a "consistency modeling" perspective, explicitly constraining their alignment using KL divergence.
3. The authors conduct comprehensive comparisons with 13 state-of-the-art methods across 13 datasets and validate the contribution of each module through ablation studies. The experimental design is thorough.

### Weaknesses
1. The theoretical justification for the equivalence between self-attention and GNNs (Appendix A.6) is intuitive. For the proposed consistency alignment loss, there is a lack of theoretical or mathematical guarantees—no formal bounds or convergence guarantees are provided to justify the rationality of this constraint.
2. The paper explains that the traffic dataset exhibits fixed periodic patterns, which accounts for why FMR underperforms the original iTransformer in Table 6. This suggests that FMR may excessively suppress periodicity, a characteristic that is very common in time series data. Have the authors considered adaptively adjusting the masking intensity based on the periodicity characteristics of the dataset to mitigate this issue?
3. Regarding performance degradation on large-variable datasets such as solar and traffic: the discussion of when DGL or CAL might lead to performance decline is somewhat superficial (limitations in Appendix A.16). The impact of these modules on large-variable datasets is mixed, sometimes degrading performance, but the paper lacks deeper analysis beyond speculation about "alignment challenges."
4. The authors need to clarify some issues in the methodology section; see the questions below.

### Questions
1. In Equations 5-6, how are the node embeddings $\Theta$ initialized? What is the purpose of $Concat(X^sa, \Theta)$?
2. Section 3.1 proposes learning independent frequency masks for each variable, but the paper lacks analysis of the learned masks: How much do the masks differ across variables? Do the masks tend to preserve low-frequency or high-frequency components? Is there a relationship with the periodicity of the data? Such analysis would enhance the interpretability of FRM.
3. Regarding the consistency alignment design, Figure 4 shows that the two mechanisms exhibit similarities but also some differences. Could the forced alignment of these two mechanisms through Equation 8 lead to information loss?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The author defines the problem of inter-variate dependencies (IVD), referring to the loss of dependency information among variables in deep self-attention layers. To address this issue, the authors propose CGTFra, a framework designed to promote consistent modeling of inter-variate dependencies.

### Strengths
S1. The author makes a noteworthy observation that time-based positional encodings do not necessarily improve performance in multivariate time series forecasting.

S2. The author enhances forecasting accuracy by addressing the inconsistency between shallow and deep attention scores.

### Weaknesses
W1. It remains unclear whether directly integrating the adaptive adjacency matrix $A$ into the MCM would offer a more structurally concise design, thereby eliminating the need to optimize two separate objectives, i.e., $L_{align}$ and $L_{mae}$.

W2. The improvement in forecasting accuracy attributed to maintaining consistency appears to be based solely on empirical results. The author should discuss whether there is any theoretical foundation supporting this effect.

W3. The coordinates mentioned in the caption of Figure 4 do not align with the values shown in the figure, and the overall presentation appears inconsistent. The author should verify or clarify this issue, and it may be more effective to use multiple figures to illustrate the inconsistency across attention layers.

### Questions
See W1-W3.

### Soundness
3

### Presentation
2

### Contribution
3
