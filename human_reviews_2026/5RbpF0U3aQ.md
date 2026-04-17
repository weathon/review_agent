# When Do Distant Dependencies Matter? Diagnostics for Long-Range Propagation in GNNs

- Decision: Reject
- Scores: 4, 2, 6, 4

## Abstract
Graph Neural Networks (GNNs) propagate information locally through message passing. While local propagation is often sufficient for short-range tasks, performance can degrade when distant interactions are required. In this paper, we introduce a diagnostic metric that quantifies the role of long-range propagation. The metric is derived from margin-aligned sensitivities, providing an interpretable measure of the dominance of one-hop neighbors in margin-relevant influence. Using this diagnostic, we show that the need for long-range propagation is dataset- and architecture-dependent, rather than universal. We further demonstrate that this diagnostic metric is predictable from well-studied graph-theoretic measures, aligning with the assumptions of rewiring-based approaches. Finally, we show how the diagnostic can be leveraged during training: we design an additional layer that selectively incorporates sensitivity to long-range dependencies and can be applied to any standard GNN backbone. Experiments on both node- and graph-level benchmarks demonstrate consistent gains over rewiring-based methods, without altering the original graph topology.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
- This paper introduces a measure of range for trained GNNs: the proportion of accurate classification confidence ('margin') sensitivity (Jacobian) that is confined to the one-hop neighbourhood; for node- and graph-classification tasks.
- The authors show that an estimate of this 'long-range capture index' can be obtained from a lasso regression over several 'structural indicators' based purely on graph topology and commonly used in graph rewiring methods — and that there is a correlation between their LR capture index and this structure-only estimate
- The authors introduce an alternative graph readout, FLAN, that re-weights and shifts output layer node features by their node-level estimated (i.e. structure-only) LR indices, without requiring rewiring, architecture changes or additional (significant) compute/time
- Experiments on several classification datasets show a minimal benefit for node classification tasks (for which the estimated index correlates weakly) and a modest but consistent benefit for graph classification tasks (for which the estimated index correlates more strongly)

### Strengths
- [S1] The paper is reasonably well-written
- [S2] The long-range capture index is intuitive
- [S3] There are existing long-range sensitivity measures using the Jacobian, but baking in model performance with classification margin is a neat idea
- [S4] The FLAN layer provides a modest performance boost on TUD graph classification tasks with minimal compute requirements compared to rewiring methods

### Weaknesses
- [W1] A core contribution of this paper is the introduction of a 'diagnostic of long-range sensitivity'. A few points on this:
	- [W1.1] The long-range capture index is a "one-hop dominance score" for a given node, and the authors argue that a low index indicates that there are more long-range interactions. However, there is no distinction between a node being dominated by, e.g., nodes at the 2-hop distance and a node being dominated by nodes at greater distances, or a variety of distances. I would argue that a 2-hop dominated node is no more 'long-range' than one dominated by the 1-hop
	- [W1.2] What is this index actually for? Is it intended as a measure of how 1-hop-dominated a model is, or how 1-hop-dominated a task is? It seems to me that if we find that a model is highly 1-hop dominated, we don't know if that is because the model has failed to capture long-range interactions or if it is because the task itself is 1-hop dominated
	- [W1.3] A Jacobian-based long-range measure is not novel; I would like to see a discussion of [1, 4]
	- [W 1.4] Later in the paper it is argued that a structural proxy index correlates strongly enough with the long-range index to be used as a substitute. Does this not somewhat undermine the long-range index as a contribution?

- [W2] I am surprised that a paper that purports to be primarily about long-range interactions does not make use of datasets explicitly described as long-range, or mention graph Transformers, or multi-hop MPNNs, or any other architectures that explicitly address long-range interactions, other than rewiring methods
	- [W2.1] The paper seems focused only on rewiring methods and over-squashing; indeed, it seems to be more a paper about over-squashing in general than it is about long-range interactions
	- [W2.2] Though it is a flawed benchmark [2], experiments on the Long Range Graph Benchmark [3], or at minimum, a discussion, would seem essential for a LR-focused paper
	- [W2.3] Other long-range datasets have been introduced recently [4, 5] that should also be discussed
	- [W2.4] Line 397: you claim that the graph classification benchmarks of TUD 'require the propagation of LR dependencies', and indeed this is a quote from Karhadkar et al. 2023, but that paper only cites the original Morris et al. 2020, and AFAIK that paper does not make this claim of long-rangedness. Could you justify what makes these tasks long-range? IMDB, for example, appears to have an average diameter of <2 hops.  "Their structures are tightly coupled to the downstream task" — what do you mean by this? Is this not the case for all graph tasks?

- [W3] Given that much of the paper, including FLAN, is built on the idea that the structural proxy index correlates strongly with the long-range capture index, I was a little underwhelmed by Table 1 — many of the datasets correlate quite weakly, and many are not robust over backbones as you claim. 

- [W4] I would have liked some more details in the Appendix about the datasets used here — their characteristics (size, diameter, average node pair distance, etc). I would suggest including a table in the Appendix showing such characteristics. In tables and figures I would also group related datasets under a header explaining their type/significance, e.g. heterophilic, small homophilic, 'long-range', graph- and node-level, etc
- [W5] I attempted to run the code and it worked for the MUTAG dataset only. I did not spend much time on this but it appears non-trivial to reproduce all experiments for other datasets. Even for MUTAG the results from the code provided do not reproduce those in the paper exactly. There is no README or documentation.
- [W6] Line 246: "On heterophilic graphs, GCN and GAT exhibit consistently negative correlations". RomanEmpire and AmazonRatings are both heterophilic and the correlation in Figure 2 is negligible
- [W7] You make no mention of over-smoothing; 'computational bottlenecks' described at line 143 would appear to correspond to over-smoothing, but there is no discussion of it or of over-smoothing literature
---
[1] Bamberger, Jacob, et al. "On Measuring Long-Range Interactions in Graph Neural Networks." arXiv preprint arXiv:2506.05971 (2025).

[2] Tönshoff, Jan, et al. "Where did the gap go? reassessing the long-range graph benchmark." arXiv preprint arXiv:2309.00367 (2023).

[3] Dwivedi, Vijay Prakash, et al. "Long range graph benchmark." Advances in Neural Information Processing Systems 35 (2022): 22326-22340.

[4] Liang, Huidong, et al. "Towards Quantifying Long-Range Interactions in Graph Machine Learning: a Large Graph Dataset and a Measurement." arXiv preprint arXiv:2503.09008 (2025).

[5] Zhou, Dongzhuoran, Evgeny Kharlamov, and Egor V. Kostylev. "GLoRa: A Benchmark to Evaluate the Ability to Learn Long-Range Dependencies in Graphs." The Thirteenth International Conference on Learning Representations. 2025.

### Questions
- [Q1] Line 387: "The sensitivity index compresses long-range demand into a single axis that is highly predictive of where the baseline fails. Because the dominant error varies monotonically with $p_u$, this rank-1 translation plus diagonal reweighting is a minimal intervention that corrects the under-performing p regime."
	- I don't understand what this means. Could you elaborate?
- [Q2] In Figure 1, is each scatter point a different node? What is $\bar{\rho}_u$? This appears to be the only use of bar notation. 
	- Notation is a bit inconsistent; there are several places where $\rho$ is used for the LR capture index, but Section 3 leads me to believe that the index should be represented by either $p_u$ for a single node, or $\rho_G$ for a graph (on this point, why mix $p$ and $\rho$ at all?)
	- You also use $S$ for both the node-level structural indicator in (7) and for source/aggregated sensitivity in (3)
- [Q3] It is an interesting result that the structural proxy index recovers $p_u$ so closely for some datasets, e.g. MUTAG. Can you elaborate on why you think this might be?

### Soundness
2

### Presentation
2

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
This work revisits the problem of _over-squashing_ in GNNs, which limits the ability of distant nodes to effectively exchange information. The authors argue that the importance of long-range information is task- and node-dependent, so interventions should be adaptive rather than universal. To quantify this adaptivity, they introduce a Jacobian-based sensitivity index that measures how a node’s classification margin changes with local versus non-local perturbations. Furthermore, they find that this sensitivity can be predicted from graph structure alone. Finally, they propose FLAN, a _rewiring-free_ and _lightweight conditioning layer_ that adjusts the readout stage based on the structure-predicted sensitivity proxy.

### Strengths
- The motivation and presentation of this work are clear and intuitive. The idea seems novel and potentially useful. 
- Various setups are used (different tasks, GNN backbones, and dataset sizes) to support the effectiveness of the method.

### Weaknesses
- The biggest weakness is the limited empirical evaluation of the proposed FLAN method. I appreciate that the authors are considering multiple architectures (GCN, GIN) for their experiments, but some of the datasets typically used for assessing over-squashing alleviation/long-range capabilities are not being used. For instance, the TUDataset benchmark is common, but the graphs contained in it do not have large diameters that would require long-range information propagation. A more standard benchmark for this is the Long-Range Graph Benchmark [[6], [7]]. Synthetic datasets such as Trees-NeighborsMatch [[8]] are also commonly used. Moreover, datasets such as Chameleon from WebKB have previously been criticized, and alternative datasets have been proposed for evaluation under heterophily [[9]].
- Another weakness of this work is that FLAN is a rewiring-free method, but there is no comparison against other rewiring-free baselines like PANDA [[1]], GESNs [[2]], Graph ViT/MLP-Mixer [[3]].
- If I understand correctly, the linear model is an estimator for p_u, but there is no theoretical justification or derivation explaining why this estimator is expected to be a good approximation for p_u.
- As I understand, the points in Figure 1 show different data items (nodes for Cornell, graphs for MUTAG) and their calculated p_u/m_u. Why is there so little of them, though? E.g., for Cornell, I'd expect ~100 points, and if not all are plotted, which ones exactly are plotted?
- Why do we want the estimator to be sparse? Is sparsity a way to interpret which of the 4 measures matter most?
- (Minor) Within the rewiring approaches, there are some baselines missing, such as approaches using virtual nodes or multi-hop neighborhoods [[4], [5]].
- (Very minor) Please consider saving the figures in the Appendix as PDFs/SVGs instead of images so that they will render better when zoomed.

[1]: https://arxiv.org/pdf/2406.03671  
[2]: https://arxiv.org/pdf/2212.06538  
[3]: https://proceedings.mlr.press/v202/he23a.html  
[4]: https://arxiv.org/pdf/2201.12674  
[5]: https://proceedings.mlr.press/v202/shirzad23a/shirzad23a.pdf
[6]: https://arxiv.org/abs/2206.08164
[7]: https://arxiv.org/abs/2309.00367
[8]: https://arxiv.org/abs/2006.05205
[9]: https://arxiv.org/pdf/2302.11640
[10]: https://arxiv.org/pdf/2405.19121?

### Questions
Please see weaknesses above.

I believe that the overall idea is interesting, and from the results shown until this moment, it does have potential. The paper is also relatively well-written. However, the experimental evaluation is very limited, and it is unclear to me if FLAN can truly alleviate over-squashing/smoothing or help with long-range dependencies in practice. 

I would highly recommend that the authors expand their empirical evaluation and add at least experiments on the LRGB and some synthetic datasets, such as Trees-NeighborsMatch. Another very interesting option would be something akin to the associative recall tasks from [[10]]. I am willing to reassess my decision depending on new experiments provided in the rebuttal period. However, right now there is very little evidence that the proposed method and techniques help in practice.

[10]: https://arxiv.org/pdf/2405.19121?

### Soundness
3

### Presentation
3

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
This paper introduces a task-aligned diagnostic, the long-range capture index $p_u$, which quantifies how much margin-relevant signal is captured by one-hop information in GNN decisions. Derived from margin-aligned Jacobian sensitivities, the index reveals that reliance on distant dependencies is dataset and architecture-dependent. Crucially, $p_u$ is predictable from topology alone , with substantially higher $R^2$ than predicting margins. Building on this, the authors propose FLAN, a rewiring-free readout that conditions the classifier on the predicted $\hat{p}_u$ via per-node rescaling and bias translation. This lightweight head adaptively compensates for long-range pressure and delivers consistent gains over rewiring-based methods on both node- and graph-level benchmarks.

### Strengths
- Well written: Introduces a novel, task-aligned, and interpretable diagnostic, the long-range capture index, derived from margin-aligned sensitivities. This is a significant contribution to understanding the computational bottleneck in GNNs.
- The proposed FLAN layer is a minimal, parameter-efficient, and effective intervention that improves performance without altering the graph topology or increasing the model's depth.
- Strong empirical results and efficiency

### Weaknesses
- Computing the ground-truth sensitivity indices $p_u$ and $\rho_G$ relies on a fully trained GNN, access to labels (for margin computation), and node features (for Jacobian sensitivities), which limits the diagnostic’s applicability in label-scarce or feature-limited settings.
- On node classification, especially small heterophilous datasets, the structure-only proxy for $\hat p_u$ exhibits high variance in $R^2$, and the corresponding gains from FLAN are less stable across seeds and backbones.
- The Jacobian-based diagnostic may be computationally costly and sensitive to training noise or feature scaling on very large graphs; scalability and numerical stability are not thoroughly characterized.
- Predicting $\hat p_u$ from topology depends on a specific set of indicators (e.g., PageRank, curvature, commute time); robustness to alternative features and distribution shift is underexplored.
- FLAN modifies only the readout (per-node rescaling and bias translation) while freezing the encoder; if oversquashing stems from insufficient message-passing capacity, benefits may plateau without encoder or topology changes.
- The aggregation from node-level $p_u$ to graph-level $\rho_G$ is not deeply ablated; results could depend on choices of normalization or weighting across nodes/graphs.
- Parity with rewiring baselines (compute budget, hyperparameter tuning, and training schedules) is not fully detailed, leaving room for comparison bias.

### Questions
Please see the weaknesses. Also, please report FLAN performance when conditioned on (i) the oracle index $p_u$ computed post-training with labels and Jacobians, versus (ii) the structure-only proxy $\hat p_u$ computed from topology. Include mean $\pm$ std across seeds and the absolute/relative delta between the two conditions on both node- and graph-level benchmarks.

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
3

### Summary
This paper investigates the role of long-range interactions in graph neural networks (GNNs) for both node-level and graph-level tasks. It introduces a Jacobian-based metric that quantifies the contribution of one-hop interactions to the decision margin, relative to interactions spanning two or more hops. The authors show that this measure can largely be predicted from the graph structure alone. Finally, the paper proposes a rewiring-free layer designed to enhance long-range information propagation.

### Strengths
1. **Clarity** The paper is well written.
2. **Significance of the problem** The problem of evaluating the importance of long range interactions is timeline and of interest.
3. **Novelty** The results that topology alone can be used to predict long-rangedness is novel and interesting.
4. **Scope** The paper has a large scope, covering both diagnostic tool and solution.

### Weaknesses
1. **Limited coverage of broader literature**
The Background and Related Work section provides an extensive overview of over-squashing and rewiring but does not sufficiently discuss related research on long-range interactions beyond over-squashing (e.g., [1, 2]). In particular, a comparison clarifying how the use of the Jacobian in this paper differs from [2] would help better situate the work within the broader literature.

2. **Weaknesses of the long-range capture index**
See Questions 1–3 below. A deeper discussion of the design choices underlying the long-range capture index—and the potential limitations or assumptions they entail—would further strengthen the paper.

3. **Lack of evaluation on explicitly long-range tasks**
The experiments focus on tasks that are not known to require long-range reasoning. Including an evaluation on a task with known long-range dependencies, even if synthetic, would provide stronger evidence supporting the proposed method.

_________

**References:**

[1] Dwivedi, V. P., Rampášek, L., Galkin, M., Parviz, A., Wolf, G., Luu, A. T., & Beaini, D. (2022). Long range graph benchmark. Advances in Neural Information Processing Systems, 35, 22326-22340.


[2] Bamberger, J., Gutteridge, B., le Roux, S., Bronstein, M. M., & Dong, X. (2025). On Measuring Long-Range Interactions in Graph Neural Networks. In Proceedings of the 42nd International Conference on Machine Learning.

### Questions
1. **Definition of the graph-level margin.**
How exactly is the graph-level margin ​defined for a node u (Equation 6)? The Jacobian of the graph-level output is a vector indexed solely by the nodes, so I do not understand how it can be binned by distances as in Equation 3.

2. **Exclusion of self-node influence.**
The long-range capture index ignores the self-node influence. Why are the nodes themselves excluded in Equation 4?

3. **Definition of short- vs. long-range interactions.**
By defining the long-range capture index as the fraction captured by the one-hop neighborhood—and considering a margin “long-range” when this number is low—you implicitly assume that a two-hop interaction is long-range. However, one could argue that a two-hop interaction might still be considered short-range. Why do you only consider one-hop neighbors as short-range interactions?

4. **Experimental setup** 
For node-level tasks, GNNs of only two layers were considered to evaluate the correlation between the long-range capture index (Fig 1 and Fig 2). One may argue that these networks are too shallow to process long-range information. Do you expect the results to hold for deeper networks?

5. **Error bars in Figure 2.**
Could you please report error bars for the correlations in Figure 2?

6. **Interpretability of Table 1.**
In Table 1, it is difficult to interpret what the reported numbers represent. Could you include a simple baseline for comparison, perhaps a random baseline or one based on node degrees?

7. In Table 1 and Figure 3, are the coefficients shown for a single run, or are they the mean coefficients computed over the 20 runs (as in Figure 2)?

8. **Attributing FLAN’s performance.** Would it be possible to attribute FLAN’s improved performance to its ability to better model long-range interactions? For instance, could you compute the one-hop index of the trained FLAN model and compare it with that of the backbone model?

### Soundness
2

### Presentation
3

### Contribution
3
