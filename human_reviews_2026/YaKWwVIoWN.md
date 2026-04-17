# Graph Unlearning via Reconstruction --- A Range-Null Space Decomposition Approach

- Decision: Reject
- Scores: 6, 4, 4

## Abstract
Graph unlearning is a machine unlearning technique tailored to graph neural networks (GNNs) to remove nodes or edges from the training graph. Conventional methods such as retraining is highly inefficient, while influence function-based approaches merely work on minor removal, like 10\% or less of the graph edges. To resolve the problems, we reverse the aggregation process in GNN training by modeling the interaction between unlearned nodes and their neighbors. Given one unlearned node, its embedding is roughly disassembled and assigned to its neighbours by reconstruction, and then removed from its neighbours by embedding modification. We also introduce range-null space decomposition to rectify the raw estimation of the interaction with theoretical support. Experimental results on multiple representative datasets and GNN models demonstrate the efficiency of at least $40\times$ acceleration compared with retraining and superior unlearning utility, efficacy, and privacy of our proposed approach compared with other method.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a novel graph unlearning method that reverses GNN aggregation by learning node-wise interactions through embedding reconstruction. Rather than retraining the entire model or modifying all parameters, the method learns two MLPs to model interactions between unlearned and retained nodes, applies range-null space decomposition to rectify estimation errors with theoretical guarantees (Theorem 1), and modifies only the output embeddings to eliminate the influence of unlearned nodes. Experiments on three datasets (Cora, Citeseer, CS) with four GNN backbones (GCN, GAT, SGC, GIN) demonstrate 40 to 88× speedup compared to retraining while achieving comparable F1 scores, strong unlearning efficacy on poisoned data, and privacy preservation under membership inference attacks.

### Strengths
1. **Novel approach with theoretical guarantees**. The core insight of reversing GNN message passing through reconstruction is elegant, and the range-null space decomposition (Equation 7, Theorem 1) provides formal guarantees that f_2^{RND} reduces L2-norm error compared to the raw f2 estimation. This addresses a fundamental challenge in graph unlearning where the entanglement of node features makes influence estimation difficult. The method's design naturally handles the under-determined reconstruction problem from lower to higher dimensional embeddings.

2. **Strong empirical performance with good generalizability**. The method achieves substantial efficiency gains (Table 2 shows 0.14-2.02 seconds vs 6-128 seconds for retraining on various models) while maintaining utility close to retraining across different unlearning ratios (Figure 6 shows consistent trends from 10% to 40%). The approach works uniformly across multiple GNN architectures without architecture-specific modifications, and demonstrates superior robustness compared to GIF whose performance degrades rapidly with increasing unlearning ratios (Figure 6 shows GIF dropping from 0.75 to 0.5 F1 score while the proposed method maintains 0.72-0.82).

3. **Effective visualization demonstrates embedding distribution equivalence**. The 2D kernel density estimation visualizations (Figures 7 and 8) provide intuitive evidence that the method produces embedding distributions matching retraining. Figure 7(b) shows the unlearned distribution closely overlaps with retraining at 30% ratio, while Figure 8(b) demonstrates the modified approach maintains this property at 40% ratio.

### Weaknesses
1. **Insufficient theoretical justification for key design choices**. While Theorem 1 proves f_2^{RND} has smaller error than f2, the paper lacks analysis of when and why reconstructing specifically the (k-1)-th layer embeddings is optimal. The justification on page 4 (lines 209) merely states it "balances the amount of information to be learned and that to be forgotten" without formal analysis.

2. **The method requires case-specific formulations that undermine generality**. The core approach fails at different unlearning scenarios without ad-hoc modifications: L^Inter extends to L^Inter+ for low ratios (Equation 14, Appendix C.1), embeddings switch from h^k to h̄^k with inference on modified graph G' for ratios above 30% (Equations 9 to 15, Appendix C.2), and edge unlearning requires separate handling (Equation 17, Appendix C.3).

### Questions
1. Could you provide theoretical or empirical evidence showing that (k-1)-th layer reconstruction is optimal compared to other layers?

2. Recent work has shown that unlearning methods can be vulnerable to sophisticated unlearning inversion attacks that exploit the differences between original and unlearned models to reconstruct unlearned data [1,2,3]. Since your method explicitly models node interactions and stores them in learned MLPs, is your approach robust to these inversion attacks? Can you evaluate attacks like those in Hu et al. [1] for general machine unlearning, Zhang et al. [2] specifically for GNN unlearning inversion, and Wu and Wang [3] for attacking incomplete unlearning? 

3. Input encryption techniques [4,5,6] are widely used to protect sensitive data during model training. Could you briefly compare how graph unlearning differs from input encryption in terms of privacy guarantees and practical deployment?

### References

[1] Hongsheng Hu, Shuo Wang, Tian Dong, Minhui Xue. “Learn what you want to unlearn: Unlearning inversion attacks against machine unlearning”. IEEE S&P 2024. 

[2] Jiahao Zhang, Yilong Wang, Zhiwei Zhang, Xiaorui Liu, Suhang Wang. “Unlearning Inversion Attacks for Graph Neural Networks”. WSDM 2026.

[3] Kun Wu, Wendy Hui Wang. “Verification of Incomplete Graph Unlearning through Adversarial Perturbations”. KDD 2025.

[4] Yangsibo Huang, Zhao Song, Kai Li, Sanjeev Arora. “InstaHide: Instance-hiding Schemes for Private Distributed Learning”. ICML 2020.

[5] Yangsibo Huang, Zhao Song, Danqi Chen, Kai Li, Sanjeev Arora. “TextHide: Tackling Data Privacy in Language Understanding Tasks”. EMNLP 2020.

[6] Sitan Chen, Xiaoxiao Li, Zhao Song, Danyang Zhuo. "On InstaHide, Phase Retrieval, and Sparse Matrix Factorization". ICLR 2021.

### Soundness
3

### Presentation
3

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
This paper proposes an efficient graph unlearning method that avoids expensive retraining. Instead of updating full GNN parameters, the authors reverse GNN message passing: they decompose the influence of an unlearned node, reconstruct its embedding from neighbors, and subtract this influence from retained nodes. A range–null space decomposition is introduced to correct reconstruction errors with theoretical guarantees. The framework achieves up to 40× faster unlearning than retraining while maintaining comparable model utility and privacy across various GNN architectures and datasets.

### Strengths
1. The approach does not depend on a specific GNN architecture.
2. The method significantly reduces computational cost.
3. Experimental results show that the method achieves effective forgetting of target nodes while keeping performance on retained nodes close to the retrained-from-scratch baseline.

### Weaknesses
1. Although key methods like GraphEraser, GIF, and GNNDelete are included, the paper does not compare against very recent graph unlearning methods. This makes it unclear how competitive the method is versus the latest developments.
2. The paper emphasizes low FLOPs overhead (~0.3%), but it only reports theoretical FLOPs, not actual runtime, or wall-clock training time on large graphs or large GNNs. No analysis is provided on how the method scales to industrial graph sizes or deeper GNNs.
3. This paper suffers from inconsistent citation and formatting styles that do not fully follow standard English academic writing norms—for example, improper spacing after mathematical symbols and duplicated content such as the repetition around line 104.

### Questions
Figure 4 does not include comparisons with any baseline methods, making it difficult to evaluate the unlearning efficacy of the proposed approach relative to existing techniques. In addition, the framework introduces several new components (interaction modeling, reconstruction, RND rectification, local refinement), but no pseudocode or algorithm is provided. This makes reproducibility unclear, and the added complexity raises concerns about computational overhead—what is the actual time/memory cost, and how do you justify that this added burden is worthwhile in practice?
Moreover, the challenges stated in the Introduction do not seem to be fully aligned with what the method actually solves, and the analysis is somewhat imprecise. In the multiple unlearning request scenario, how does your method perform compared to baselines? What are the empirical results for your method under multiple sequential unlearning requests, and how stable or scalable is it in that setting?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper addresses the problem of node-unlearning in GNNs. The authors propose to reverse the aggregation process by modelling node-wise interactions. To handle the fact that embedding transformations are often irreversible, they employ a Range-null space decomposition (RND) framework. Their approach modifies only embeddings and keep embedding distributions similar before and after unlearning. Experimental results on several real-world graph datasets demonstrate significant speed-ups and comparable model utility and privacy robustness.

### Strengths
S1. The experimental evaluation is thorough and comprehensive, covering multiple benchmark datasets and GNN architectures to demonstrate both efficiency and utility. 

S2. The method design is intuitive and interpretable, the authors provide a clear workflow for node unlearning. 

S3. Transferring RND decomposition into graph-unlearning context is also intuitive for addressing challenges of inverse reconstruction. 

S4. Theoretical analysis is also provided with framework design.

### Weaknesses
(Most Important) W1. Lack the section of “The Use of Large Language Models (LLMs)”. As the organizer stated: “Not disclosing significant LLM usage can lead to desk rejection of the paper.” 

W2. The method’s reliance on the pseudo-inverse of the layer Jacobian matrix 
𝐻
H
 
assumes numerical stability and invertibility, but the paper does not sufficiently analyse or mitigate potential ill-conditioning of 
𝐻
H
 
. 

W3. The learnable component f_2 is presumed to capture the null-space residuals of the range–null decomposition, yet the paper does not provide rigorous guarantees or regularization to ensure that f_2  does not overfit noise. 

W4. The approach focuses on embedding modification, which raises the concern that if the removed node had a significant gradient influence on the model parameters (e.g., centrality node), which may not sufficiently erase its impact at the parameter level. 

W5. While the RND technique is novel in the context of graph unlearning, the underlying mathematical tool is established in other ML/signal tasks, especially in recent two years [1],[2],[3], so the novelty of method design is somewhat incremental. 

W6. Many spelling, grammar, and citation typos are found: 

Line 32-33: “exhorbitant computational overhead”. 

Line 275-276: “Although we does not…, but …”. 

Line 882-883: “Then, we prove that … is projecton matrix”. 

Line 275-276: “Although …, but …” 

In addition, the majority of wrong citations usage, i.e. ~\cite{} rather than ~\citep{}, this constitutes a format inconsistency. 

--- 

References 

[1] Wang, Yinhuai, et al. "Gan prior based null-space learning for consistent super-resolution." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 37. No. 3. 2023. 

[2] Chen, Jiacheng, et al. "Null space matters: range-null decomposition for consistent multi-contrast MRI reconstruction." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 38. No. 2. 2024. 

[3] Li, Andong, et al. "Learning Neural Vocoder from Range-Null Space Decomposition." arXiv preprint arXiv:2507.20731 (2025).

### Questions
Please check the weaknesses part

### Soundness
3

### Presentation
2

### Contribution
3
