# Fighter: Unveiling the Graph Convolutional Nature of Transformers in Time Series Modeling

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 4, 4

## Abstract
Transformers have achieved remarkable success in time series modeling, yet their internal mechanisms remain opaque. This work demystifies the Transformer encoder by establishing its fundamental equivalence to a Graph Convolutional Network (GCN). We show that in the forward pass, the attention distribution matrix serves as a dynamic adjacency matrix, and its composition with subsequent transformations performs computations analogous to graph convolution. Moreover, we demonstrate that in the backward pass, the update dynamics of value and feed-forward projections mirror those of GCN parameters. Building on this unified theoretical reinterpretation, we propose **Fighter** (Flexible Graph Convolutional Transformer), a streamlined architecture that removes redundant linear projections and incorporates multi-hop graph aggregation. This perspective yields an explicit and interpretable representation of temporal dependencies across different scales, naturally expressed as graph edges. Experiments on standard forecasting benchmarks confirm that Fighter achieves competitive performance while providing clearer mechanistic interpretability of its predictions.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper explores analogies between Transformers and GNNs in the context of time series forecasting, drawing parallels between the attention mechanism and the message-passing framework. However, the paper has very limited technical novelty, some claims are overstated, and the empirical analysis has significant issues.

### Strengths
* Understanding attention-based architectures in the context of time series forecasting is an important and timely research problem.

### Weaknesses
* **Limited technical novelty.**  The technical novelty of the paper is very limited.
    - The paper’s main premise , i.e., that Transformers can be seen as a specific implementation of a GNN operating on a fully connected graph with data-dependent edge weights, is an obvious and well-known fact, which has been known and discussed in the community for years (see, e.g., [1] and the popular 2020 blog post [2]).  
    - The MLP layer used between attention blocks in Transformer architectures increases model capacity and transforms the representations extracted by the attention block. The paper argues that this transformation is redundant since it is not used in vanilla GNN architectures (see, e.g., lines 199–201). However, there is no logical connection between the fact that a layer is not used in vanilla GNNs and it being generally unnecessary.  
    - Given that the information propagation mechanism is the same, the equivalence observed when analyzing derivatives (Section 4.2) appears obvious and does not provide any novel insight.  
    - The paper claims that, inspired by the analogy between GNNs and Transformers, one can use powers of the learned adjacency matrix to propagate information. However, since the graph is fully connected and the weighting mechanism data-adaptive, it is unclear what advantage this offers over standard multi-head attention.

* **Poor empirical evaluation.**  There are several issues in the empirical evaluation
    - The proposed approach is evaluated only against outdated baselines, on a small number of datasets, and with no indication of variability (e.g., standard deviations across runs). Moreover, the paper does not clearly describe how baselines were trained and tuned. This, combined with the absence of released code, makes the results difficult to reproduce.  
    - The reported performance gains appear unreasonable (e.g., nearly 100× on some datasets) given the minor architectural modifications introduced. Furthermore, based on the sensitivity analysis, one would expect Fighter with a single hop to perform similarly to a Transformer, as the architectures are essentially equivalent. However, the reported performance on Weather is ~100× better even for a single hop, compared to the Transformer results in the table - while the Transformer performance shown in the plot (red line) appears more in line with Fighter. What is happening here? Are you sure the results were reported on the correct scale?  In addition, forecast accuracy on the Electricity dataset seems to improve with longer forecasting horizons, which is quite implausible since the task becomes more difficult.

Even disregarding the empirical issues, the very limited technical novelty alone would prevent me from recommending acceptance.


References

[1] Dwivedi et al., "A Generalization of Transformer Networks to Graphs", arxiv 2020  

[2] Chaitanya Joshi, "Transformers are Graph Neural Networks*", 2020 blog post: https://graphdeeplearning.github.io/post/transformers-are-gnns/ -- (arxiv 2025)

### Questions
See weaknesses.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The work focuses on proving the equivalence between a Transformer encoder and a flexible graph convolution network. 
The authors also propose a flexible graph convolution transformer called Fighter, which removes redundant linear projections to improve efficiency and claims to have higher expressivity.

### Strengths
The paper is written clearly, and theoretical results are presented briefly in the main paper with further details deferred to the appendix. I also like the illustrations to aid the readers.

### Weaknesses
I find the paper interesting and well-written overall. However, the experimental results section could be strengthened. I have included several questions and suggestions in that regard. I'm happy to improve the rating if my questions are answered.

### Questions
1. What is the difference in the attention matrices of the Transformer and the GCN? Could you please report the norm of the difference? Basically, Table 5 shown in numbers.
2. In Table 1, why is there a huge drop in MSE/MAE in the forecasting tasks, but comparatively a minimal improvement in the classification accuracy?
3. Can the authors please provide a table comparing the number of parameters of different baseline models with Fighter, and also their per-epoch training time?
4. Can the authors please provide confidence intervals for all the numerical results?
5. Commenting on the better performance of Fighter over longer sequences just by comparing two datasets is not wise. To make such a claim, the trend must be observed over a larger number of data points (in this case, datasets and prediction length).
6. Fig 4(b) is not clear. Could the authors please zoom in or adjust the scale so that the order over different $\kappa$ values can be observed?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors focus on time series forecasting and introduce a connection between the transformer encoder and the graph convolutional network, by mapping the attention distribution matrix to a dynamic adjacency matrix, the layer transformations to graph convolution, and the updated attention values and projection coefficients to the GCN parameters. Finally, based on this analogy, the Flexible Graph Convolutional Transformer, so-called Fighter, is proposed, excluding unnecessary linear projections and considering multi-hop graph aggregation, representing multi-scale temporal dependencies by graph edges.

### Strengths
1. **Originality:** The authors make an analogy between the graph convolution and the transformer encoders in time series forecasting, and mathematically prove the connection under specific circumstances, which is a novel approach bringing two hot modeling aspects in modern time series frameworks.
2. **Quality and Clarity:** The methodological section (and additional details in the appendix) is easy to follow, despite being heavy in formulas.
3. **Significance:** On the few showcased datasets, the forecasting performance of the proposed Fighter model significantly outperforms transformer-based baselines.

### Weaknesses
- **W1-Significance (Motivation of Proposed Contribution):** The motivation behind the proposed idea is not clearly showcased. The authors aim to connect graph convolution representations with transformer attention for time series forecasting, yet it is not clearly explained why existing approaches are inadequate. In particular, several transformer variants addressing optimization issues in time series forecasting have already been proposed, but the paper does not clarify whether these methods are conceptually related to the proposed approach or whether the limitations they face are effectively addressed by the new method.
- **W2-Clarity (Positioning against Related Works):** The authors do not discuss related work in graph representation learning for time series, where methods often distinguish between sparse [1,3] vs. fully-connected adjacencies [4] and static vs. dynamic dependencies [4,5]. The Introduction and Related Work sections should be updated to position the proposed Fighter model in the context of existing graph-based time series approaches. Importantly, the work seems conceptually similar to StemGNN [5], where the adjacency is learned based on a latent correlation attention layer on the whole sequence.
- **W3-Quality and Significance (Experimental Evaluation):** The set of baselines considered is limited to older transformer-based approaches and does not include more recent improvements, alternative architectures, or graph-based methods (see references and TimeMixer, TimeXer, iTransformer, and PatchTST from https://github.com/thuml/Time-Series-Library). Additionally, the datasets used do not cover the full benchmark in the forecasting community (e.g., several standard datasets are not evaluated, and few horizon lengths are considered). The inclusion of the text classification dataset is unclear, as it is not standard in the time series community and does not directly relate to the forecasting task. If additional tasks are to be considered, they should be standard in the time series field, such as classification and anomaly detection, to ensure comparability with existing literature.
- **W4-Clarity and Significance (Computational Complexity):** Although the authors refer to computational improvements enabled by their proposed method, they do not provide a computational analysis or experimental comparison for time cost and memory complexity compared to baselines or variants of the model. The fully-connected design of the adjacency, combined with message passing, is in general computationally expensive and should be compared to baselines to justify the claims for computational improvement.
W5-Significance (Reproducibility): Although a substantial explanation of the experimental setup is given, the code implementation of the proposed method is not available in the submission; therefore, direct reproducibility of the experimental results is not possible.

**References:**
1. Shang C, Chen J, Bi J. Discrete graph structure learning for forecasting multiple time series. arXiv preprint arXiv:2101.06861. 2021 Jan 18.
2. Bai L, Yao L, Li C, Wang X, Wang C. Adaptive graph convolutional recurrent network for traffic forecasting. Advances in neural information processing systems. 2020;33:17804-15.
3. Wu Z, Pan S, Long G, Jiang J, Chang X, Zhang C. Connecting the dots: Multivariate time series forecasting with graph neural networks. InProceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining 2020 Aug 23 (pp. 753-763).
4. Yi K, Zhang Q, Fan W, He H, Hu L, Wang P, An N, Cao L, Niu Z. FourierGNN: Rethinking multivariate time series forecasting from a pure graph perspective. Advances in neural information processing systems. 2023 Dec 15;36:69638-60.
5. Cao D, Wang Y, Duan J, Zhang C, Zhu X, Huang C, Tong Y, Xu B, Bai J, Tong J, Zhang Q. Spectral temporal graph neural network for multivariate time-series forecasting. Advances in neural information processing systems. 2020;33:17766-78.

### Questions
1. Can the authors showcase with examples/illustrations how the proposed design solves long-term problems in the time series community, e.g., correlations, dynamic dependencies, distribution shift, or other?
2. Can the authors position clearly the proposed method against related works in temporal modeling and graph-based modeling, including justifications for the architectural choices with respect to common challenges (see Q1)?
3. The experimental comparisons should be extended to more relevant baselines and datasets or more time series tasks to improve the significance of the contribution.
4. What are the computational aspects of the proposed architecture and how do these compare to baselines?

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
4

### Summary
This paper presents FIGHTER (Flexible Graph Convolutional Transformer), a novel framework that reinterprets Transformers as Graph Convolutional Networks (GCNs) for time series modeling. The authors provide a unified theoretical analysis showing that the Transformer’s attention distribution acts as a dynamic adjacency matrix, while its value and feed-forward updates mimic GCN feature propagation. Building on this equivalence, they propose FIGHTER, a simplified Transformer variant that removes redundant projections and introduces multi-hop graph aggregation, enabling explicit, interpretable temporal dependencies. Experiments on standard forecasting datasets (Electricity, Weather) and text classification (AG News) show that FIGHTER achieves state-of-the-art or superior results with dramatically lower errors and improved interpretability through graph-based attention visualizations.

### Strengths
The paper makes an important conceptual contribution by rigorously connecting the Transformer and GCN formulations, offering both theoretical insight and architectural innovation. The idea of treating attention as a learnable adjacency matrix provides a fresh and unifying lens for understanding sequence modeling. The derivations are detailed and mathematically grounded, bridging the forward and backward pass analysis of both architectures. Empirically, FIGHTER achieves substantial performance gains and offers clear interpretability via graph-based attention visualization. The design is also elegant—removing redundant projections and adding multi-hop aggregation leads to improved efficiency and better long-range dependency capture.

### Weaknesses
The experiments, though diverse, are somewhat limited in dataset variety and task complexity; 

Additional real-world datasets or ablations on larger Transformer variants would strengthen the empirical foundation. 

Moreover, the mathematical exposition is dense and sometimes overly formal, which might obscure intuition for non-theoretical readers. 

The practical computational trade-offs of the multi-hop attention (in memory and speed) are not deeply discussed. 

Finally, interpretability is primarily visual and qualitative; a more quantitative assessment of interpretive fidelity would improve credibility.

### Questions
How does FIGHTER scale computationally with increasing hop parameter κ, especially in long sequences?

Can the proposed equivalence framework generalize to multi-head or cross-attention Transformers (e.g., encoder-decoder setups)?

Are there scenarios where the dynamic adjacency interpretation breaks down—for instance, in sparse attention or masked modeling settings?

How stable are the gradients and convergence behavior compared to standard Transformers during long training runs?

### Soundness
3

### Presentation
2

### Contribution
2
