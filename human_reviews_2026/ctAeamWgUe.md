# CrGSTA: Cross-domain Root causal Graph Spatial-Temporal Attention Network

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 4, 2, 4

## Abstract
Modern monitoring systems generate massive, high-dimensional time series
where failures rarely remain isolated but cascade across interdependent components.
Identifying their true origins requires more than anomaly detection; it requires
interpretable models that disentangle causal structure from noisy signals.
While Granger causality has gained traction for root cause analysis (RCA), existing
neural methods often rely on multilayer perceptrons applied independently at
each time step, which increases parameter counts, struggles with long-range dependencies,
and overlooks seasonal and periodic patterns. We introduce CrGSTA
(Cross-domain Root causal Graph Spatial-Temporal Attention Network), a scalable
and interpretable framework that unifies time- and frequency-domain representations
through cross-domain attention. CrGSTA employs graph-based spatiotemporal
attention to capture directional dependencies, while frequency-aware
features recover periodic structure. A lightweight self-attention decoder reconstructs
dynamics, ensuring deviations are attributed to true root causes rather than
propagated effects. We conduct experiments along three dimensions: temporal
scalability, spatial scalability, and ablations on domain contributions and fusion
strategies. On multiple synthetic and real-world datasets, CrGSTA new state of
the art achieving up to 13% Avg@10 improvement by leveraging wider temporal
windows with only 8.5M parameters compared to (200M+) of other baselines.
By explicitly coupling temporal and frequency cues, CrGSTA balances accuracy,
interpretability, and efficiency for RCA in complex monitoring environments, providing
a foundation for more resilient and transparent analysis in real-world systems.
https://github.com/crgsta2025/CrGSTA

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper tackles root cause analysis (RCA) for high-dimensional multivariate time series, arguing that existing neural Granger approaches miss long-range dependencies, seasonal/periodic structure, and scale poorly. It proposes CrGSTA—a Cross-domain Root causal Graph Spatio-Temporal Attention network. CrGSTA unifies time- and frequency-domain views with a shared spatial graph attention backbone, temporal attention on lags/bins, and bi-directional cross-attention to fuse domains. The encoder yields interpretable, Granger-style coefficients and latent exogenous variables; a lightweight self-attention decoder reconstructs dynamics. Emperical experiments demonstrate that CrGSTA is effective and helps improve performance compared with other models.

Contributions:
1) An interpretable, parameter-efficient RCA framework that couples spatial graph attention with time–frequency modeling.
2) A multi-path encoder–decoder with cross-domain fusion that captures periodicity/seasonality while preserving causal interpretability.
3) Extensive experiments shows SOTA performance.

### Strengths
1) The cross-attention mechanism between time and frequency domains is novel for RCA tasks, and targeted important and practical problem of this domain.
2) Results and analysis indicate practical value for large-scale industrial monitoring where both interpretability and compute budget matter. Also the model design directly target the theoretical deficit of previous works, which helps explain the performance improvement with experiment demonstration.
3) Detailed reproduce informations and settings.

### Weaknesses
1) The discussion on related work for root cause analysis can be improved to be more thorough and coherent. In particular, incorporating discussion and comparison with more works utilizing causal graphs, such as Wang, D., Chen, Z., Fu, Y., Liu, Y., & Chen, H. (2023, August). Incremental causal graph learning for online root cause analysis. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (pp. 2269–2278), will help improve the quality.
2) The space utilization in the main plot could be improved. For example, the title text appears relatively small while noticeable blank areas remain unused. Adding descriptive captions would also help readers follow the figures more easily. In addition, the presentation of the experimental section could be more clearly structured, with explicit motivation, experimental settings, result interpretation, and concluding remarks.
3) The evaluation currently only compares against AERCA among causal methods. Including additional recent RCA approaches would provide a more comprehensive and convincing comparison.
4) The clarity of the ablation table may be enhanced by clearly highlighting the ablated components, which would help readers better interpret the contribution of each module.
5) Table-1 in Appendix is somehow misleading, “Comparison of Root Cause Analysis (RCA) and Anomaly Detection Approaches”, since there are large portion of RCA and Anomaly detection approaches, the table content does not justify the title and intention well.

### Questions
1) Could you discuss additional causal graph–based RCA methods and compare them with this paper? Including these works in the related-work section would further improve the coherence and thoroughness of the analysis.
2) If adaptable, incorporating comparisons with more causal models in the experimental section would further substantiate the effectiveness claims of the proposed method.
3) Improving space utilization and adding a clearer caption for Figure 1 may enhance readability. For the ablation study plots, explicitly tagging the ablated components would also improve clarity.
4) The material presented in “Table 1: Comparison of Root Cause Analysis (RCA) and Anomaly Detection Approaches” reads more like a feature comparison than a conceptual justification of the methods. It would be helpful to justify the table name or adjust it to ensure that the title and the presented content are aligned.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents CrGSTA (Cross-domain Root causal Graph Spatial-Temporal Attention Network), a framework for interpretable root cause analysis (RCA) in multivariate time series. CrGSTA integrates temporal and frequency-domain representations via cross-domain attention and employs a graph-based encoder-decoder to capture both spatial and temporal dependencies. The method builds upon Granger causality for interpretability, uses graph attention networks to model variable relationships, and applies self-attention for efficient temporal modeling. Extensive experiments on the Lotka–Volterra synthetic dataset and the SWaT industrial dataset demonstrate that CrGSTA achieves competitive or superior results to statistical, non-causal, and causal baselines (e.g., AERCA), with fewer parameters and improved interpretability.

### Strengths
+ Well-motivated integration of temporal, frequency, and causal reasoning.
+ Strong empirical performance on both synthetic and real-world benchmarks.
+ Comprehensive ablation studies validating design choices.

### Weaknesses
- This paper is primarily an integration of existing mechanisms (attention, GNN, Granger causality).
- The evaluation scope is limit. It only consider two datasets, both within standard RCA benchmark. There exist many public multivariate time series (MTS) benchmarks such as SMD, SMAP, MSL.Testing on these would better demonstrate the generalization ability.
- Some dense mathematical sections could benefit from summarizing intuition alongside formulas.
- The paper could show more qualitative evidence (e.g., visualized causal graphs or case studies) to confirm interpretability in practice.

### Questions
1. How well does the model generalize across domains (e.g., trained on SWaT, tested on other industrial datasets)?
2. Does the frequency-domain branch improve interpretability, performance, or both?
3. What is the computational complexity compared with AERCA for long sequences?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces CrGSTA, a scalable and interpretable deep learning framework for root cause analysis in multivariate time series. CrGSTA addresses the parameter inefficiency and limited pattern-capturing capacity of prior MLP-based approaches by adopting a multi-path encoder–decoder architecture. The model jointly leverages time-domain and frequency-domain representations through cross-attention and incorporates graph-based spatio-temporal attention to capture complex inter-variable relationships. Experimental results on the Lotka–Volterra and SWaT benchmarks demonstrate that CrGSTA achieves state-of-the-art performance while maintaining significantly higher parameter efficiency, highlighting its practicality for large-scale, high-dimensional monitoring scenarios.

### Strengths
- The paper is clearly written and easy to follow, with well-structured explanations that support reader comprehension.
-  The proposed multi-path encoder—integrating parallel time- and frequency-domain representations with cross-domain attention fusion—is well-motivated and technically sound.
-  The model achieves strong empirical performance, matching or surpassing state-of-the-art baselines while using up to 20× fewer parameters, demonstrating notable efficiency and a well-validated architectural design through thorough ablation studies.

### Weaknesses
- The experimental evaluation is limited to only two datasets (Lotka–Volterra and SWaT). Incorporating additional benchmarks—as done in AERCA [1]—would help strengthen the generalizability and external validity of the findings.
-  Although the framework is positioned as interpretable, no human-in-the-loop evaluation or rule-based validation is provided. A qualitative case study that links attention weights or learned coefficients to known process topology would substantially reinforce the interpretability claim.
-  The magnitude–phase component appears to yield only marginal gains, as it does not consistently improve performance across the ablation studies, raising questions about its practical benefit to the overall architecture.

[1] Han, X., Absar, S., Zhang, L., & Yuan, S. (2025). Root Cause Analysis of Anomalies in Multivariate Time Series through Granger Causal Discovery. The Thirteenth International Conference on Learning Representations. 
- The main contribution and claim of the paper are not clearly stated. The introduction initially suggests a focus on causality, but the technical sections and experiments emphasize parameter efficiency. It remains unclear how the proposed method supports or interprets causality. Providing a qualitative analysis—such as illustrating how the model identifies or explains causal relationships—would strengthen this claim.

-  The writing and methodological explanations require further elaboration.
1.	Several architectural components are introduced without sufficient justification. The paper should explain why each module is needed, how they differ from components in existing models, and how the equations reflect these design intentions. For instance, how are the weights and QKV mappings computed, and what motivates the bidirectional interaction between temporal (time → frequency) and spectral (frequency → time) contexts?
2.	The evaluation is limited in dataset diversity. Prior work such as AERCA evaluates on six datasets (Linear, Nonlinear, Lotka–Volterra, Lorenz-96, SWaT, MSDS), and similar breadth would strengthen claims of generalizability.
3.	The experimental comparison set could be expanded. Additional RCA-based or GNN-based approaches should be included to provide a more comprehensive benchmarking against current methods.

### Questions
1. In Section 4.3, the authors state that “Results are presented in Fig. 2 and summarized in Tables 8 and 9.” However, the content in these tables does not correspond to Figure 2 Did the authors intend to refer to Tables 5 and 6 instead?
2.  In the SWaT dataset results, the reported performance of AERCA is shown as 0 for window sizes 5, 7, 10, and 12. Could the authors clarify whether this indicates a failure mode, an implementation issue, or missing results?
3.  In the Sparsity & Smoothness loss formulation, the symbol Ωis introduced but not defined. What does Ωrepresent in this context?
4.  On the Lotka–Volterra benchmark, CrGSTA’s performance shows large variation as the window size increases, while other baseline models remain comparatively stable. Could the authors provide an explanation for this sensitivity?
4. AERCA seems to be higher performance in many places, e.g., Figure 2(a) 0.791 and Table 5
5. Why there are 0.000s in the result (e.g., Table 6). This may indicate the evaluation is not fair.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents CrGSTA, a novel deep learning framework for Root Cause Analysis (RCA) in high-dimensional, complex multivariate time series (MTS). The work is motivated by a key limitation in existing neural Granger causality methods (e.g., AERCA), which often rely on per-time-step MLPs. The authors argue this approach scales poorly, increasing parameters dramatically with longer temporal windows, and fails to capture periodic or seasonal patterns critical for RCA.

To solve this, CrGSTA proposes a unified encoder-decoder architecture grounded in Granger causality (where the "root cause" is the unexplained exogenous residual, $z_t$). Experiments on a synthetic (Lotka-Volterra) and a real-world (SWaT) dataset show that CrGSTA achieves new state-of-the-art results (e.g., +13% Avg@10 on Lotka-Volterra).

### Strengths
S1. The paper addresses a critical and practical problem: scalable, interpretable RCA. It correctly identifies the trade-off between accuracy and parameter/computational cost in prior SOTA causal models as a major bottleneck.

S2. The experimental results and their in-depth analysis are insightful.

S3. The results for RQ2 (Fig 3) demonstrate that CrGSTA also scales more efficiently with the number of variables (spatial dimension), using significantly fewer parameters (1.7M) than AERCA (4.0M) to achieve comparable performance at 60 variables.

### Weaknesses
W1. One of the concerns is the reported performance of the primary causal baseline, AERCA, on the SWaT dataset. In Table 6 and Figure 2b, AERCA scores 0.000 on all metrics (AC@1, Avg@10, etc.) for all window sizes. This is highly ridiculous. It implies the model failed completely (worse than random chance). Did it fail to run, suffer from OOM, or was there an implementation bug? This is a critical comparison, and the 0.000 result is not adequately explained, undermining the perceived gap between CrGSTA and the prior SOTA.

W2. The paper states AERCA's parameter count "grows to over 100M" and "200M+" (Sec 4.3). Figure 5b shows it reaching 350M. This is a key selling point, but the implementation details for AERCA in Table 4 mention "8 layers (1000 nodes) per lag." This seems to be the authors' own implementation choice, and it's unclear if this is a necessary or fair configuration for the AERCA baseline. A brief justification for why AERCA was configured to be so large is needed.

W3. The model is exceptionally complex. While the ablation justifies this, it makes the paper very dense and challenging to parse. Figure 1, in particular, is a difficult "boxes and arrows" diagram that obscures the elegance of the parallel-then-fused design.

W4. The paper's evaluation is deep but narrow. All claims are supported by strong results on only two datasets: one synthetic (Lotka-Volterra) and one real-world (SWaT). While these are good benchmarks, the strong claims of the paper would be far more generalizable if validated on at least one other real-world domain (e.g., cloud microservice, finance, or a different CPS dataset).

### Questions
Q1.  The 0.000 Avg@10 score for AERCA on SWaT (Table 6, Fig 2b) is a major concern. Please clarify this result. Did the baseline implementation fail to run, OOM, or did it genuinely produce zero correct results? This is the most critical point to address.

Q2. The model's validation is strong but limited to Lotka-Volterra and SWaT. How do you expect CrGSTA's cross-domain (Time+Freq) approach to perform on systems that are less governed by physical laws or clear periodicity, such as a stochastic cloud microservice environment?

Q3. Could you elaborate on the parameter configuration for the AERCA baseline? Is the "8 layers (1000 nodes) per lag" setting (Table 4) derived from the original AERCA paper, or was this a choice made for this paper? 

Q4. The ablation (Fig 4) shows that for SWaT, the Frequency-only model (F (Mag)) outperforms the Time-only model (T) and even naive T+F fusion. This is a very interesting finding. Does this imply that for some real-world CPS systems, frequency-domain analysis is more important than time-domain for RCA?

### Soundness
2

### Presentation
3

### Contribution
3
