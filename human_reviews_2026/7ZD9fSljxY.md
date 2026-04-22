# Robust Forecasting of Network Systems Subject to Topology Perturbation

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 4, 4

## Abstract
Many real-world dynamical systems, such as epidemic, traffic, and logistics networks, consist of sparsely interacting components and thus naturally exhibit an underlying graph structure. Forecasting their evolution is computationally challenging due to high dimensionality and is further complicated by measurement noise and uncertainty in the network topology. We address this problem by studying the predictability of graph time series under random topology perturbations, a problem with major implications that has remained largely unexplored. In the limit of large networks, we uncover distinct noise regimes: systems that are predictable with arbitrary accuracy, systems predictable only up to limited accuracy, and systems that become entirely unpredictable. Motivated by this characterization, we propose a time series forecasting framework based on a probabilistic representation of network dynamics, which leverages Bayesian coreset approximations for scalable and robust dimentionality reduction. Numerical experiments on both synthetic and real-world networks demonstrate that our approach achieves competitive accuracy and robustness under topology uncertainty, while significantly reducing computational costs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses the critical and challenging problem of forecasting network dynamical systems, such as traffic or epidemic networks, in the presence of topology perturbations. The authors make two primary contributions. First, they provide a theoretical analysis that characterizes the predictability of graph time series under random topology perturbations. Second, motivated by this analysis, the paper proposes a novel and robust forecasting framework named Network Coreset Forecasting (NCF). NCF utilizes a Graph Convolutional Network (GCN) encoder to generate node embeddings, applies Bayesian coresets for scalable and robust dimensionality reduction by selecting a representative subset of node embeddings, models the latent temporal dynamics using an RNN, and finally reconstructs the full graph state via a GCN decoder. Empirical results on real-world traffic datasets demonstrate that NCF significantly outperforms baselines in accuracy and robustness under topology uncertainty, while also being more computationally efficient.

### Strengths
1. The paper tackles the highly practical and under-explored problem of robust forecasting for dynamic graph systems where the topology is uncertain or noisy, a common scenario in real-world applications.

2. The analysis on predictability regimes under noise provides a solid theoretical motivation for the necessity and design of a robust forecasting model.

3. The authors commendably discuss the method's limitations, using synthetic Kuramoto data to highlight the accuracy-robustness trade-off, where NCF underperforms on clean, smooth, noise-free trajectories.

### Weaknesses
1. The model relies on a two-stage training approach rather than a full end-to-end optimization. This may lead to sub-optimal representations, as the GCN encoder and the RNN are not jointly optimized. The rationale for this design choice over an end-to-end approach is not fully justified.

2. The theoretical analysis and experiments primarily focus on i.i.d. random noise (Bernoulli or Gaussian). This does not capture more structured or adversarial perturbations (e.g., targeted node/edge removal, regional sensor failures) that are common in reality, thus narrowing the scope of the "robustness" claim.

3. The paper lacks a qualitative analysis of the nodes selected by the Bayesian coreset. It is unclear what properties these nodes possess (e.g., high centrality, high dynamic variance) and how this selection mechanism practically contributes to robustness.

4. Lack of baselines. For example, [1-3] discusses predicting network dynamics (also on partially observed networks), however, the authors did not include them for comparison.

[1] Predicting long-term dynamics of complex networks via identifying skeleton in hyperbolic space. KDD 2024.
[2] Learning Continuous System Dynamics from Irregularly-Sampled Partial Observations. NeurIPS 2020.
[3] HOPE: High-order Graph ODE For Modeling Interacting Dynamics. ICML 2023.

5. The experiments are confined to medium-sized graphs (e.g., PEMS datasets, 500-node Kuramoto), leaving scalability as an open question. More different synthetic graph structures (such as Erdos-Renyi network, Barabasi-Albert network), and other real-world structures such as social networks, epidemic networks, etc., should be considered and discussed.

### Questions
Please see Weaknesses for details.

Moreover, I think the paper is in the wrong format for ICLR, which may conflict with the submission guidelines.

### Soundness
2

### Presentation
2

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
The paper studies the problem of forecasting network systems when the underlying topology is perturbed. It provides a mathematical analysis characterizing the predictability of such systems under random topology noise, distinguishing fully predictable, limited predictable, and unpredictable regimes. Motivated by this analysis, the authors propose Network Coreset Forecasting (NCF), a framework combining Bayesian coreset selection, GCN encoder-decoder and LSTM to learn temporal dynamics in a reduced latent space.

### Strengths
1. The topic is timely and practically meaningful. Robust forecasting under perturbed topologies is a central problem in applications. 
2. The paper provides a rigorous and original mathematical analysis of network predictability on noisy topologies, deriving clear error bounds and interpretive regimes.

### Weaknesses
1. The proposed NCF framework mainly reuses existing components (Bayesian coreset selection, GCN and LSTM) within a new theoretical interpretation. The model essentially performs latent-space forecasting with probabilistic justification, rather than introducing a new learning mechanism or architecture. 
2. The theoretical results assume predictability becomes worse with the topology perturbations going larger, yet the coreset-based abstraction explicitly removes nodes and edges, potentially amplifying perturbations. Moreover, GCN encoder-decoder may still rely on the (possibly perturbed) original adjacency matrix in order to estimate all the nodes, so the claimed robustness is partial and bounded by the sensitivity of these components to the matrix. 
3. The RNN module is tied to a specific coreset node set. When the topology or node count changes, the coreset must be recomputed and the trained RNN becomes incompatible with new size of the coreset, preventing inference on unseen graphs. In addition, coreset selection requires full-graph statistical computation and must be repeated whenever the topology changes, limiting scalability and online applicability. 
4. Baselines included in experiments are not the latest nor enough.

### Questions
1. The paper does not specify how the coreset size is determined, is it fixed or adaptive? Corresponding analysis should be included. 
2. Besides training time, what is the inference time?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The authors propose a time series forecasting framework robust to topology perturbation. The GCN embeddings undergo dimensionality reduction through a Bayesian coreset, making them robust to topology perturbations. Intuitively, this seems similar to the idea of shrinking an image, which in turn reduces the impact of noisy pixels on the overall features. The framework is then tested on synthetic as well as real temporal networks.

### Strengths
1. The idea is novel in the sense that it is applied to temporal graphs
2. The prior art is covered well
3. The paper is written clearly and follows a logical order
4. The main content is well supported by proofs and implementation details in the appendix
5. The source code is provided

### Weaknesses
1. Issues with citation style in the text. Please use `\citet{}` and `\citep{}` appropriately
2. In the main text, confidence intervals are not reported
3. There are more traffic datasets, such as METRLA and PEMSBAY. Many algorithms that perform well on PEMS0* do not always perform well on them.
4. The figures 3,4,6,7, and 9 take more space than necessary, and can be presented more efficiently.
5. The SoTA was not reported for all datasets; for example, please check **[R1]** where the algorithm `mspace` reports an MAE of 8.7 on PEMS04, and 6.33 on PEMS08. It would be interesting to see the impact of noisy data on the performance achieved by mspace.

> [R1] Rahman, A. U., & Coon, J. Node Feature Forecasting in Temporal Graphs: an Interpretable Online Algorithm. Transactions on Machine Learning Research.

### Questions
1. How does the work relate to state-space models of temporal graph forecasting?
2. What are the 95% confidence intervals for the numerical results? Does it make the performance improvement of the proposed model statistically significant, i.e., is there no overlap of the confidence intervals of the proposed model with the baselines?
3. How does the performance gain of the proposed model change with the size of the dataset and the number of training samples available?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper studies how well we can forecast graph/network dynamical systems when the topology is perturbed. In the large-scale limit, it claims three noise regimes: with small noise you can reach arbitrarily high accuracy; with moderate noise you can only reach limited accuracy; with larger noise the system becomes unpredictable. Based on this view, the paper proposes Network Coreset Forecasting (NCF): first use a GCN to get node embeddings, then use a Bayesian coreset to pick a small set of representative nodes, then run an RNN/LSTM in this lower-dimensional space and decode the results back to the full graph.

### Strengths
1. The problem is important and realistic: real systems (traffic/epidemics/power grids) often have uncertain topology, so studying robust forecasting is valuable.
2. The idea of using noise regimes to describe predictability, and then designing an algorithm around that view, is interesting.
3. The method is modular and easy to extend: coreset selection, GCN encoder/decoder, and RNN for time dynamics (the paper also shows pseudo-code).
4. Experiments are fairly thorough, include real traffic data, and are set up to show pros/cons and how results relate to the stated bounds.

### Weaknesses
1. Gaussian noise threshold has the wrong scale. A node sums noise from many neighbors; variances add, so the total standard deviation is about $\sqrt{\text{degree}}\times\sigma$. To keep the total perturbation $O(1)$, you need $\sigma=O(1/\sqrt{n})$, not $O(1/n)$.

2. Spatial modes $y_k$ are not defined. Then \max_{k} ||y_k||_{\infty}  in the thresholds is not knowable. If y_k is a flat eigenvector, ||y_k||_\infty \sim 1/\sqrt{n}; if y_k is very sparse (near one-hot), ||y_k||_\infty\sim 1. This changes thresholds by orders of magnitude, not just a constant factor.
3. Missing basic stability/Lipschitz conditions. Without global Lipschitz/contractivity/ISS-type conditions, you cannot guarantee a bound on \sup_t of the error. In some parameter ranges the system may amplify small noise, so the error can blow up.
4. Coreset bound does not imply forecasting error bound. The cited coreset result controls posterior/log-likelihood error in a functional norm $||\cdot ||_{\pi,L}$, which is not the same as end-to-end state forecasting error (MAE/MSE) of the RNN+decoder. A bridging theorem is missing.

### Questions
1. See Weaknesses

2. If the topology noise uses edge flips or a mixture of edge additions and deletions, do the conclusions still hold, and how do the thresholds change?

3. Where does the decoder’s supervision signal come from? During training, do you observe the ground truth of the dropped nodes? If so, does the test-time distribution match the training distribution?

### Soundness
2

### Presentation
2

### Contribution
2
