# EdgeMask-HGNN: Learning to Sparsify Hypergraphs for Scalable Node Classification in Hypergraph Neural Networks

- Avg Score: 3.00
- Decision: Reject
- Scores: 4, 2, 2, 4

## Abstract
Hypergraph Neural Networks (HGNNs) have achieved remarkable performance in various learning tasks involving hypergraphs— a data model for higher-order relationships across diverse domains and applications. However, the scalability of HGNNs is limited by the computational and memory demands incurred by dense hypergraph structures. Existing unsupervised sparsifiers address the scalability issue but sacrifice downstream predictive performance. To address this, we propose **EdgeMask-HGNN**, a novel framework that introduces a learnable, task-aware sparsification mechanism to reduce the hypergraph size while preserving predictive performance. EdgeMask-HGNN offers two distinct masking: a fine-grained node-hyperedge masking and a coarse-grained hyperedge-level masking, both trained end-to-end using supervision from the downstream task. We provide theoretical analysis showing that our approach (i) yields stable model outputs under stochastic masking, and (ii) ensures convergence of retention probabilities under gradient descent. 
Extensive experiments on multiple node classification benchmarks demonstrate that EdgeMask-HGNN reduces or maintains memory usage on both small- and large-scale hypergraphs without sacrificing accuracy, and in some cases outperforms HGNNs trained on full hypergraphs. Moreover, EdgeMask-HGNN consistently surpasses unsupervised sparsification baselines such as random, degree-based, and spectral sparsification.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose a hypergraph sparsification framework, which removes certain hyperedge membership to achieve scalability as well as downstream task performance improvement.

To this end, the authors propose EHGNN, which learns a masking matrix for a given incidence matrix. Specifically, the matrix is optimized from the targeted downstream task loss, such as node classification.

The authors verify the performance of EHGNN in several benchmark hypergraph datasets.

### Strengths
S1. Given that real-world group interactions occur on a scale, the research topic is very important.

S2. The authors demonstrate the effectiveness of the proposed method under diverse backbone HNNs.

### Weaknesses
**W1. [Goal of sparsification]** The title and presentation of the work focus on **scalability**. However, it seems that the proposed method often requires more training time and GPU memory consumption (Tables 3 and 4). Given these results, I think the scalability of the proposed method is not experimentally supported. Could the authors further clarify these results?

**W2. [Training]** It seems that authors are using sampling strategies in their methods. However, the sampling process often cuts the gradient, making the training infeasible. It seems that authors incorporate graph sparsification techniques (Lines 268 - 270), such details should be self-contained in the manuscript. Could the authors elaborate on this?

**W3. [Datasets]** I think the used datasets are not that large-scale, as which are fewer than 100K nodes. Given that graph sparsification works are evaluated in a million-scale graphs, the authors are expected to evaluate their methods in larger hypergraphs. I think the authors can refer to the work: (Datasets, tasks, and training methods for large-scale hypergraph learning, DAMI 2023).

### Questions
See Weakness.

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
This paper introduces EdgeMask-HGNN, a learnable task-aware sparsification framework for Hypergraph Neural Networks (HGNNs) to address scalability challenges. The key contributions include: (1) two masking strategies - fine-grained incidence-level (EHGNN-F) and coarse-grained hyperedge-level (EHGNN-C) masking, both trained end-to-end using downstream task supervision; (2) theoretical analysis proving stability under stochastic masking and O(1/ε) convergence of retention probabilities; and (3) extensive experiments on 15 datasets demonstrating that the method maintains or reduces memory usage while preserving (or sometimes improving) predictive performance compared to full HGNN training.

### Strengths
**Originality:**
- First work to propose learnable, task-aware sparsification for HGNNs, addressing a gap in the literature where only unsupervised methods existed
- The dual-granularity framework (incidence-level vs. edge-level) is a sensible design choice that provides flexibility
- Feature-conditioned variants using permutation-invariant pooling show creativity in adapting to semi-supervised settings

**Quality:**
- Extensive experimental evaluation across 15 datasets with multiple metrics (accuracy, memory, runtime)
- Proper ablation studies investigating design choices (Table 8, Figure 4)
- Theoretical analysis attempts to provide formal guarantees (Theorems 5.1, 5.2)
- Code release promised for reproducibility
- Model-agnostic framework demonstrated on multiple HGNN architectures (Table 5)

**Clarity:**
- Well-structured paper with clear motivation illustrated in Figure 1
- Comprehensive appendix with detailed proofs and additional experiments
- Generally clear writing, though notation becomes dense in places

**Significance:**
- Addresses the important problem of HGNN scalability
- Demonstrates that selective sparsification can sometimes improve accuracy (8/15 datasets), suggesting noise reduction benefits
- However, significance is substantially diminished by inconsistent results and failure on the exact scenarios (very large, dense hypergraphs) where scalability matters most

### Weaknesses
### 1. **Fundamental Contradiction in Core Contribution**

The paper's primary motivation is memory reduction for scalability on large-scale hypergraphs. However, **Table 3 reveals that on Yelp** (the largest and densest dataset at 679,302 edges), several variants consume **MORE memory than full training**:
- EHGNN-F (cond.): 138.1 GB vs. 111.2 GB (Full)
- EHGNN-C (cond.): 139.5 GB vs. 111.2 GB (Full)

This directly contradicts the paper's main claim. The explanation provided (lines 401-405) about "pooling and scorer MLPs introduce extra activations" is insufficient. **This needs rigorous analysis:**
- Provide exact memory breakdown: parameters vs. activations vs. optimizer states
- Why does the theoretical space complexity O(m + kd) not hold in practice?
- Under what density/size thresholds does memory actually increase?
- Is this a fundamental limitation or an implementation issue?

---

### 2. **Inconsistent Performance Without Predictive Framework**

Results show high variance across datasets with no clear pattern:
- **Improvements:** ModelNet40 (+0.37%), DBLP-CA (+0.19%), several others
- **Degradations:** 20news (-0.30%), NTU2012 (-0.31%), Actor (+12-14% but highly variable across variants)
- **Equivalent:** Multiple datasets within error bars of full training

For a methods paper at ICLR, this inconsistency is problematic. **The paper lacks:**
- A principled framework predicting when sparsification helps vs. hurts
- Analysis of hypergraph properties (density, edge size distribution, homophily) correlated with success
- Clear decision criteria for practitioners choosing between variants

---

### 3. **Weak Theoretical Contributions **

**Theorem 5.1 (Stability):**
- Assumes HGNN is L-Lipschitz w.r.t. Frobenius norm but provides **no validation**
- Are standard HGNNs (HGNN, HyperGCN, etc.) actually Lipschitz? What is L?
- The bound shows variance decreases as p→0 or p→1, which is obvious—doesn't explain empirical behavior

**Theorem 5.2 (Convergence):**
- Assumes gradient signs remain fixed throughout training—**highly unrealistic**
- O(1/ε) rate is standard for many optimization problems, not a novel contribution
- No empirical validation showing this rate actually occurs (Figure 6 only shows final distributions)

**Missing theory:**
- When does task-aware sparsification improve over unsupervised methods?
- What properties of the loss landscape explain why removing edges helps in some cases?
- Connection between sparsification and generalization bounds

---

### 4. **Insufficient Baseline Comparisons **

**Critical missing baselines:**
- **Spectral methods OOM** on exactly the datasets where comparison matters (Walmart, Yelp, DBLP-CA). This makes claims about outperforming spectral methods unsubstantiated where it counts
- **No other learnable/task-aware methods:** The paper positions itself against only unsupervised baselines. Are there no graph sparsification methods (e.g., L0-based, DropEdge variants) that could be adapted?
- **No graph-to-hypergraph adaptations:** What if you apply graph sparsification methods after clique expansion?

---

### 5. **Heterophilic Graphs: Acknowledged but Unresolved **

- Appendix G shows **poor performance on heterophilic benchmarks** (Actor, Pokec, Twitch)
- Section E.1 shows only marginal differences between EHGNN-F and EHGNN-C on synthetic heterophilic data
- The explanation (line 1023-1026) attributes this to Laplacian averaging, but this affects **full training too**—not specific to sparsification

**This limits applicability** given growing interest in heterophilic graph learning.

---

### 6. **Parameter Overhead Not Properly Addressed **

Table 9 shows EHGNN-F has **massive parameter overhead:**
- Yelp: 5,482,067 vs. 958,473 (Full) = **5.7× more parameters**
- Walmart: 472,417 vs. 11,787 (Full) = **40× more parameters**

Yet the paper focuses on activation memory. Questions:
- How does this affect training time and convergence?
- What about optimizer states (Adam stores 2× parameters)?
- Does this create overfitting risk, especially given inconsistent results?
---

### 7. **Missing Critical Experimental Details **

- **No convergence analysis:** How many epochs? What stopping criteria?
- **Sparsity budget selection:** How is k or κ chosen? Different per dataset? Sensitivity analysis missing beyond one dataset (Figure 3a)
- **Statistical testing:** With overlapping error bars (e.g., Cora: 77.52±0.28 vs. 78.35±0.34), are differences significant?
- **Hyperparameter selection:** How sensitive are results to MLP hidden dimensions (Table 7)?

---

### 8. **Runtime Claims Need Clarification **

- Table 4 shows EHGNN methods often **slower** than Full on small datasets
- The explanation (mask-learning overhead) contradicts efficiency claims
- Figure 3c shows gains only on largest datasets, but these are the same ones with memory issues

---

### Questions
1. **Memory paradox on Yelp:** Can you provide a detailed memory breakdown (parameters, activations, gradients, optimizer states) explaining why memory increases on dense hypergraphs? Is this fundamental to your approach or fixable? At what hypergraph density/size does this crossover occur?

2. **When does your method work?** Can you provide a quantitative analysis correlating dataset properties (from Table 1: density = |E|×avg_edge_size / |V|, edge size distribution, homophily, etc.) with performance gains/losses? This is essential for practitioners.

3. **Spectral baseline on large graphs:** Can you implement an approximate spectral method (e.g., using random projections or incomplete Cholesky) to enable comparison on Walmart and Yelp? Without this, your claims about outperforming spectral methods are incomplete.

4. **Statistical significance:** Are the reported improvements statistically significant? For example, on Cora (77.52±0.28 vs 78.35±0.34), the confidence intervals overlap substantially.

5. **Theoretical assumptions validation:** 
   - What is the Lipschitz constant L for standard HGNNs empirically?
   - How often do gradient signs flip during training? Can you show this empirically?
   - If assumptions are violated, do the theorems still provide useful insights?

6. **Feature-conditioned variants:** Why don't EHGNN-F(cond.) and EHGNN-C(cond.) consistently outperform non-conditioned versions despite added complexity? When should they be preferred?

7. **Sparsity budget selection:** How should practitioners choose k or κ? Is there a principled approach (e.g., based on validation performance, theoretical bounds, computational budget)?

8. **Heterophilic failure mode:** You mention this is due to Laplacian averaging (line 1023), but Full training uses the same mechanism yet performs better. What specifically about sparsification makes heterophilic graphs harder?

9. **Comparison with graph methods:** Have you tried adapting graph sparsification methods (e.g., via clique expansion then applying graph sparsifiers)? Why/why not?

10. **Overfitting analysis:** With 40× more parameters on some datasets (Table 9), do you observe overfitting? How does train vs. validation vs. test performance compare?

### Soundness
2

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
This work proposes EdgeMask-HGNN with two distinct masking strategies, i.e., fine-grained node–hyperedge masking and coarse-grained hyperedge-level masking, to address the scalability challenges of HGNNs. Theoretical analyses establish the perturbation stability and the convergence of the method’s mask parameters. Furthermore, the authors conduct comprehensive experiments on fifteen benchmark datasets, demonstrating that EdgeMask-HGNN reduces or maintains memory usage on both small- and large-scale hypergraphs without sacrificing accuracy.

### Strengths
1. The manuscript provides theoretical analysis on the proposed method to demonstrate the perturbation stability and convergence of mask parameters.
2. The overall presentation is clear and well-structured. Moreover, the authors provide source code to ensure reproducibility.

### Weaknesses
1. The use of edge masking for selective message passing is not novel. Prior studies in GNNs and HGNNs for semi-supervised learning have explored similar ideas, including CO-GNN [1] and HeteHG-VAE [2].
2. Key backbone methods are omitted, such as ED-HNN [3] and SheafHGNN [4].
3. The benchmark datasets are relatively small; the largest include Walmart (88,860 nodes) and Yelp (679,302 hyperedges).
4. For the datasets described as large-scale, the proposed method requires more GPU memory for training than using full hypergraphs, making it difficult to justify the necessity of EdgeMask-HGNN.
5. Furthermore, existing work [5] reports that these datasets can be trained as full hypergraphs on two NVIDIA Tesla P100 GPUs with 12 GB memory each, raising concerns about the necessity of the proposed method.
6. Although the authors provide theoretical analysis on perturbation stability and the convergence of mask parameters, additional experiments are needed to validate the edge masking strategy.
7. Based on the experimental results, the proposed method is more suitable to tackle the heterophilic issue in hypergraphs. 

[1] Cooperative Graph Neural Networks. ICML'24.

[2] Heterogeneous Hypergraph Variational Autoencoder for Link Prediction. T-PAMI'21.

[3] Equivariant hypergraph diffusion neural operators. ICLR'23.

[4] Sheaf hypergraph networks. NeurIPS'23.

[5] You are AllSet: A Multiset Function Framework for Hypergraph Neural Networks. ICLR'22.

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
In this paper, the authors propose EdgeMask-HGNN, a framework that sparsifies hypergraphs for more efficient hypergraph neural network learning. Specifically, it introduces a fine-grained node–hyperedge masking and a coarse-grained hyperedge-level masking mechanism to reduce hypergraph complexity while preserving task performance. Theoretical analysis demonstrates the stability of model outputs under stochastic masking and the convergence of retention probabilities during optimization. Extensive experiments on multiple node classification benchmarks show that EdgeMask-HGNN achieves comparable or superior accuracy to full HGNNs while significantly improving efficiency.

### Strengths
1. The paper is clearly written and easy to follow.

2. The proposed method is thoroughly validated across a wide range of datasets, results in Table 2 robustly demonstrate superiority over baselines (degree, random, spectral).

### Weaknesses
1. Several works focus on hypergraph structure learning, such as [1]. A comparison with these methods in the experiments would be valuable.

2. Some studies, such as [2–3], on sparse GNNs perform joint sparsification of both the graph structure and the neural network parameters; these works are related to this study and should be discussed.

3. The proposed masking method has already been well studied in existing works.
 
[1] Cai, Derun, et al. "Hypergraph Structure Learning for Hypergraph Neural Networks." IJCAI. 2022.

[2] Chen, Tianlong, et al. "A unified lottery ticket hypothesis for graph neural networks." International conference on machine learning. PMLR, 2021.

[3] Liu, Chuang, et al. "Comprehensive graph gradual pruning for sparse training in graph neural networks." IEEE Transactions on Neural Networks and Learning Systems 35.10 (2023): 14903-14917.

### Questions
See weaknesses

### Soundness
2

### Presentation
2

### Contribution
2
