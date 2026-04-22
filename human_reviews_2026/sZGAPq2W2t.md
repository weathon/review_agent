# TS-TPR: Tensor Product Representation for Multivariate Time Series Forecasting

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 2, 6

## Abstract
Real-world multivariate time series exhibit nonstationary inter-variable dependencies, which evolve dynamically due to external environmental shifts. While capturing these intricate dynamics is crucial for accurate forecasting, many existing methods still struggle to explicitly model these complex relationships. This motivates the need for compositional learning, which explicitly separates relational and temporal components and flexibly recombines them. Such a design allows models to adapt to time-varying inter-variable relationships and generalize to unseen patterns. To address this, we introduce TS-TPR, a novel framework that employs tensor product representations for compositional learning. Specifically, context-aware role generation identifies the most salient relationships at each time, while hierarchical filler extraction summarizes the corresponding temporal patterns. By combining these dynamically generated roles and fillers via tensor products, TS-TPR creates an explicit, structured representation that naturally scales to many variables and adapts as dependencies shift. Through experiments on diverse real-world benchmarks, we show that TS-TPR not only outperforms state-of-the-art baselines but also provides interpretable, time-varying insights into inter-series interactions.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors address a key challenge in multivariate time series forecasting (MTSF): nonstationary inter-variable dependencies, where relationships between variables evolve dynamically. They argue that existing models often entangle these relational dynamics with temporal patterns, leading to poor performance under distribution shifts. To solve this, the authors propose TS-TPR, a novel framework based on Tensor Product Representation (TPR) for compositional learning. The core idea is to explicitly disentangle relational "roles" from temporal "fillers". To manage the $N \times N$ complexity of variable relations, the framework learns a "codebook" of $K$ generalized "relation prototypes". At each time step, the model dynamically selects the $M$ most relevant prototypes (roles) based on the current context and uses a hierarchical attention mechanism to extract the corresponding temporal patterns (fillers). A Linear Transformer is then used to perform the TPR binding and unbinding operations to generate the final forecast. Experiments on long-term, short-term, and zero-shot forecasting benchmarks demonstrate that TS-TPR achieves state-of-the-art or competitive performance.

### Strengths
* The motivation is clear and targets a critical, well-argued problem in time series: modeling nonstationary inter-variable dependencies.
* The idea of using a learnable **codebook** to quantize the $N \times N$ relationship space into $K$ generalized prototypes is a novel and effective method to address the combinatorial complexity problem.
* The experimental evaluation is comprehensive, covering long-term, short-term, and zero-shot settings, and shows strong performance against robust SOTA baselines.

### Weaknesses
* The framework's **architectural design is relatively complex**, integrating multiple sophisticated modules. This design results in significant computational overhead: empirical data (Table 7) shows that TS-TPR uses **3-5x more memory and training time** than baselines. However, there is a **cost-benefit mismatch**, as the performance gain (e.g., on the Weather dataset) is **marginal** for such a high cost, questioning its practical value.
* The **"zero-shot" generalization claim is weak**. All zero-shot experiments (Table 3, 9) are conducted *within* the ETT dataset family (e.g., ETTh1 $\rightarrow$ ETTh2). As all ETT datasets originate from the same physical source, this does not sufficiently prove cross-domain generalization.
* The **"interpretability" claim (Figure 3) is overstated**. The figure effectively demonstrates *adaptability* (i.e., the model switches from codes #5, #13 to code #2 when the data pattern shifts) but not *interpretability*, as the semantic meaning of what codes $c_2, c_5, c_{13}$ actually represent is never analyzed.

### Questions
1. Regarding the computational cost (Table 7): How do the authors justify the 3-5x increase in memory/time for the marginal (e.g., ~2.5%) MSE improvement on datasets like Weather? Is this trade-off practical for real-world deployment?
2. The zero-shot claims (Table 3) are limited to the ETT family. Was true cross-domain generalization (e.g., training on ETT, testing on Weather) evaluated? If not, can the authors provide further justification for why the learned codebook prototypes are "generalized" and not just specific to the ETT domain?
3. Regarding Figure 3, can the authors provide any semantic analysis of the learned codebook vectors (e.g., $c_2, c_5, c_{13}$)? What do these learned "relation types" actually *mean*? Without this, the claim is adaptability, not interpretability.

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
This work introduces TS-TPR, a framework for multivariate time series forecasting that leverages tensor product representations (TPRs) to explicitly disentangle relational structures from temporal features. Specifically, the role is generated from the relational attention map and the filler is constructed from a combination of relational and temporal features. The role and filler are bind and unbind to make the final prediction. Experiments on real-world datasets demonstrate that TS-TPR outperforms state-of-the-art models in both accuracy and interpretability.

### Strengths
The proposed framework introduces a model to separately extract inter-variable dependencies and temporal patterns, enhancing the interpretability.

### Weaknesses
1. **Unclear motivation for introducing the Tensor Product Representation (TPR) framework.**
   The rationale for adopting the TPR framework is confusing. In prior works cited by the authors, TPR is typically used to decompose fillers and their corresponding roles from mixture representations. However, in this paper, both roles and fillers are *predefined*. It remains unclear why the model first binds them to form mixture representations and then unbinds them to recover the fillers.

2. **Limited connection between the proposed method and the TPR framework.**
   The resulting approach does not appear to be genuinely TPR-based; instead, it merely combines relational and temporal features through several attention layers. For example, in standard TPR theory, the unbinding vector corresponds directly to the role, whereas in the proposed model, these two vectors are derived from two sources.

3. **Inappropriate experimental setup.**
   Most main experiments are conducted with a lookback window of (T = 96), which is insufficient, as some baseline models require longer lookback windows to achieve their optimal performance. The evaluation should include at least (T = 336) to ensure fairness and comprehensiveness.

4. **Incorrect complexity analysis.**
   The complexity calculation is inaccurate. Equation (3) has a complexity of $O(CL^2)$, and Equation (6) has $O(C^2L)$; therefore, the overall complexity should be at least $O(CL^2 + C^2L)$. The authors are encouraged to verify this through empirical runtime comparisons on synthetic datasets with *varying numbers of channels and input lengths*.

5. **Insufficient ablation studies.**
   More ablations are needed to validate the design choices, including:
   (1) directly using $e^{attr}$ followed by a projection head for forecasting;
   (2) testing alternative ways to combine $e^{attr}$ and $e^{rel}$.

### Questions
1. What exactly does the term "context" refer to in the phrase "Context-aware Role Generation"?
2. How does the proposed model achieve dynamic relation modeling as claimed? Based on my understanding, the relations are fixed within each input window. If the claim simply means that different inputs lead to different relation patterns, this property is not unique—many existing models can achieve the same behavior.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces TS-TPR, a multivariate time-series forecasting approach that leverages Tensor Product Representations. The core idea is to model inter-variable relations as roles and temporal attributes as fillers. The authors design a two-stage filler computation process:  
1. Relation-aware fillers: Uses attention to query source variables with relation embeddings.  
2. Role-aware aggregation: Employs attention to query relation instances using target-specific role embeddings, generating role-aligned fillers for each target variable.  

The forecasting mechanism relies on binding/unbinding operations implemented through linear attention, featuring a context-aware unbinding operator that considers the target's temporal context. The Role Selector utilizes a VQ codebook of relation prototypes to identify the top-M roles for each target variable. The authors demonstrate through ablation studies that both the Role Selector and the two-stage filler extraction are essential components.

### Strengths
1. Achieves state-of-the-art or near-state-of-the-art results on multiple benchmarks, with consistent gains across long-term, short-term, and zero-shot settings.  
2. Decoupling relations (roles) from temporal attributes (fillers) and recombining them via a TPR-style binding/unbinding mechanism provides a principled way to model time-varying dependencies and explain which roles matter.  
3. The hierarchical pipeline makes alignment between concrete relations and abstract roles explicit; ablation studies show consistent gains for the full combination.  
4. Evaluation spans ETT, ECL, Weather, Traffic, and EPF datasets, including model complexity, training-time, and memory comparisons, strengthening the practical engineering case.

### Weaknesses
1. Equation 5 computes \(R_i\) based on overall distances \(D_{i,k}\) without time resolution. It is unclear whether roles adapt per timestep, per window, or remain static per series. Missing time-resolved role trajectories and stability metrics.  
2. Claims of natural scaling to many variables are only demonstrated up to \(C=862\). 
3. Most zero-shot transfers remain within the ETT-family; heterogeneous transfers would provide a stronger test of compositional generalization.
4. All benchmark datasets use data collected between 2011-2020 (too old).

### Questions
1. Are roles selected per window or per timestep, and can time-resolved role assignment trajectories be shown?  
2. Can you provide training/inference curves versus \(C\) and evaluate sparse relation approximations?  
3. How does hard VQ-based role selection compare to soft attention over the codebook or temperature-annealed VQ?  
4. Can heterogeneous cross-domain transfers be added with leakage-safe normalization?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper addresses nonstationary inter-variable dependencies in multivariate forecasting by advocating a compositional approach that explicitly factorizes relational and temporal structure. It introduces TS-TPR, which uses tensor product representations: a context-aware role generator selects salient inter-series relationships at each time step, while a hierarchical filler extractor summarizes the corresponding temporal patterns; their tensor products yield explicit, structured, and scalable representations that adapt as dependencies shift. The authors claim state-of-the-art results across diverse benchmarks and highlight interpretability—via time-varying role/filler components—as a key advantage over methods that entangle relation and dynamics.

### Strengths
1. The target problem - multivariate time series forecasting is important, and it is interesting to see the proposed solution coming from the view of tensor product representation.  
2. It's good to see that some empirical results on efficiency are provided. At the first glance, I have some doubts on how this method would affect the efficiency for both training and inference. Based on their results, I am convinced that the overhead is not significant.  
3. The experiments include the zero-shot setting. I appreciate the results on the zero-shot setting as I believe that it can demonstrate the generalizability of the proposed method.

### Weaknesses
1. I think it might be better to include some experiments on synthetic datasets to directly support the claim "the framework not only adapts to new relational contexts but also provides interpretability by revealing which relationships guide each prediction". It would greatly improve the credibility. 
2. The authors do not provide some experiments on how to balance the prediction accuracy and cookbook regularization. In Sec. 3.5, some values for $\alpha$ and $\beta$ are provided. However, I think it would be better to have some empirical results on the effect of different values. 
3. The datasets used in the experiments, especially for zero-shot setting. There are several more comprehensive benchmarks proposed since 2025, e.g., fev-benchmark [1] and Gift-Eval [2]. I would suggest to have some results on those benchmarks and investigate how it compare with other baselines. 

References:

[1] fev-bench: A Realistic Benchmark for Time Series Forecasting

[2] GIFT-Eval: A Benchmark For General Time Series Forecasting Model Evaluation

### Questions
1. Some typos in the manuscript. Please revise it. For example, ". urthermore," in Page 5. 
2. I wonder if the authors can provide some case studies on Weather dataset to show that how the proposed method can resolve the distribution shift issue. As in Figure 1, the authors show that the relationship between two vars can vary across time. I believe that might be better to show some results directly on this.

### Soundness
3

### Presentation
2

### Contribution
3
