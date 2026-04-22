# MLOT: Generalizing the Bipartite Structure to a Multi-Layered Framework for Optimal Transport

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 4, 4, 8

## Abstract
Although optimal transport (OT) has achieved significant success and widespread application in various fields, its structure remains relatively simple, relying on bipartite graphs with only two layers of nodes for transportation. In this paper, we propose a multi-layer optimal transport (MLOT) method that extends the original two-layer structure to handle transportation problems across multiple hierarchical levels, making it more adaptable to the complex structures found in deep learning tasks. In this framework, the source distribution flows through intermediate layers before reaching the target distribution, where estimating the intermediate distributions becomes crucial for solving the MLOT. Under entropic regularization, we further propose the MLOT-Sinkhorn algorithm to solve the multi-layer OT problem, where intermediate distributions can be estimated through the transportation calculations between adjacent layers.  This algorithm can be accelerated using GPUs and significantly outperforms general solvers such as Gurobi. We also present theoretical results for the entropic MLOT, demonstrating its efficiency advantages and convergence properties. Furthermore, we find that our MLOT is well-suited for machine learning tasks based on data augmentation. As a result, we apply the MLOT-Sinkhorn algorithm to tasks such as text-image retrieval and visual graph matching. Experimental results show that reformulating these problems within the MLOT framework leads to significant improvements in performance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a multi-layer optimal transport (MLOT) method that extends the original optimal transport formulation. The theoretical properties of MLOT are discussed. Two theoretically guaranteed algorithms are designed to solve MLOT. The authors further conduct experiments on text-image retrieval and visual graph matching.

### Strengths
The MLOT generalizes both OT and low-rank OT and looks interesting.

### Weaknesses
1. Line 190 "The optimization described above is essentially a convex optimization problem" is not very obvious. The authors may consider formalizing this using a proposition and provide formal proofs.

2. Proposition 1 looks not very clean. For example, can $\epsilon_0$ be 0?

3. In Line 211, why is the Bregman iterative algorithm a Sinkhorn-based algorithm?

4. It would be better to directly provide a computational complexity at Line 239 - Line 241.

5. For Line 314 - Line 317, what's the advantage of using augmented data? Are there any disadvantages of using augmented data?

6. In Figure 6, the generated interpolations images are not of high quality. Many of them look like simple weighted sum of the source image and the target image at the pixel level.

7. Equation (34) seems wrong.

8. It is still not very clear why MLOT is more adaptable to complex structures found in deep learning tasks.

### Questions
See above.

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
This paper introduces Multi-Layered Optimal Transport (MLOT), a novel framework that generalizes traditional optimal transport from a bipartite structure to a multi-layer graph. The authors develop an entropically regularized formulation and derive efficient Sinkhorn-style algorithms.

### Strengths
1) The core idea of generalizing OT from a bipartite to a multi-layered graph is both interesting and important. It elegantly addresses a key limitation of vanilla OT in deep learning, where data often has a more complex, hierarchical, or dynamic structure. 

2) MLOT-Sinkhorn is shown to be highly efficient and accurate on synthetic data, scaling to problems where baseline solvers like Gurobi fail.

3) Theoretical justification of the  global convergence.

### Weaknesses
1) The authors correctly identify the conceptual proximity of their multi-layered framework to Dynamic Optimal Transport and the Schrödinger Bridge Problem (SBP), viewing MLOT as a discrete-state analogue. However, this connection remains under-explored.

2) The propagation chain on MLOT-Sinkhorn algorithm creates feedback loops where the output of one layer's update is immediately used as input for another's within the same iteration. The proof of Proposition 4 in Appendix D may be incorrect, because it fails to account for this temporal and spatial coupling, treating the layers as if they could be analyzed in isolation and then combined with a simple worst-case bound. The updates are not independent across layers, creating a propagation chain that's not captured:

$u_k^{l+1} = a_k^l * (S_k v_k^l)$

$v_k^{l+1} = a_{k+1}^l * (S_k^⊤ u_k^{l+1})  $

$a_k^{l+1} = ((S_{k-1}^⊤ u_{k-1}^{l+1}) * (S_k v_k^{l+1}))^{1/2}$

3) The stated limitation ("our algorithm cannot obtain the exact solution or deal with more general graphs") is too brief and can be expanded.

### Questions
Given the stated connection to Generalized Schrödinger Bridges (GSB), is it possible to provide a quantitative comparison against specialized GSB solvers (for example HOTA [1]) on a standard 2D benchmark (Stunnel, Vneck, and GMM ref. [1])? 

[1] Buzun et. al.  HOTA: Hamiltonian framework for Optimal Transport Advection. https://arxiv.org/abs/2507.17513

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
3

### Summary
This paper proposes Multi-Layered Optimal Transport (MLOT), which extends standard Kantorovich optimal transport from a two-layer bipartite structure to a multi-layer framework. 
The authors develop the MLOT-Sinkhorn algorithm that iteratively computes both couplings and intermediate distributions under entropic regularization. 
In their empirical study, they demonstrate computational advantages over Gurobi and GraphOT-based methods on synthetic data and apply MLOT to data augmentation tasks including CLIP-based retrieval and visual graph matching.

Claim:I don't have much expertise in OT theory and entropic OT, hence I will focus more on the evaluation of applications of MLOT. I might also miss something when evaluating the comparison against existing literature in the field.

### Strengths
1. The formulation of MLOT seems novel to me.
2. They provide convergence analysis for their MLOT-Sinkhorn algorithm.
3. They demonstrate the efficiency and effectiveness of their method on synthetic datasets.
4. The writing of the theory part is overall clear and the paper is easy to follow.

### Weaknesses
1. I am somewhat confused about the setup of the original MLOT (Eq. 2). As far as I understand, the problem does not have a unique solution. Could the authors clarify this? Is this a misunderstanding? If not, would this lead to an ill-defined problem?
2. The application section is confusing. In Section 3.3, the authors start with "the application of MLOT to address tasks that involve augmented data." However, the authors didn't explain well what task it is and what role data augmentation plays in it. This leads to difficulty in understanding how MLOT could help and be a potentially better method.
3. Regarding the application of CLIP-based text-image retrieval, what is the intuition of setting text as the intermediate distribution? Why would solving for the unique solution to this MLOT-Sinkhorn problem lead to a better retrieval?
4. I am not convinced about the benefit of introducing the distribution of augmented data and the formulation of MLOT. Could the authors provide any insight on why this is fundamentally a better method? Regarding the empirical advantage of MLOT with Random Augmentation in Table 3, I feel like this is not a fair comparison since MLOT essentially gains more information from data. For example, for Softmax or Independent Sinkhorn, what if we compute scores using original data and augmented data separately, then aggregate them?

Overall, I find the theory part of this paper sound, but the application section and following experiments design are confusing.

### Questions
1. Could the authors elaborate on the relationship between MLOT and multi-marginal OT?
2. I am confused about Figure 4. I cannot tell why MLOT is better than Barycenter from the qualitative result.
3. I suggest moving Figure 2 to where it is first referred to.
4. The authors should use separate notation for their K and the metric R@k.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This work introduces multi-layer optimal transport (MLOT), extending classical entropic OT so that mass flows through $K-2$ intermediate layers that the marginal distributions are inferred jointly with the couplings between adjacent layers. Authors proposed two algoroithms to  compute the MLOT, a Bregman projection scheme and a Sinkhorn-like fixed-point iterations, which supports GPU-friendly matrix scaling while estimating the intermediate distributions. The authors provided convergence arguments and demonstrate the potential applications on synthetic layered transport, CLIP-based zero-shot retrieval with data augmentation, visual graph matching, and image interpolation. I reviewed this work from the previous conference cycle; the current draft shows clear progress with better exposition and richer experiments, so I inclined to support acceptance.

### Strengths
- The algorithm implementations are straightforward to implement on GPUs and appear to scale to reasonably large synthetic instances.

- The empirical comparison with generic LP solvers and shortest-path reductions highlights practical efficiency gains when $K>2$ and problem sizes exceed the comfort zone of exact solvers.

- This work showcased multiple applications of ML task including retrieval, graph matching, and interpolation indicate potential for applying MLOT.

### Weaknesses
- The formulation in Eq.\ (2) is a standard multi-stage min-cost flow with entropic regularization, which related formulations exist under the names layered OT, and network-flow OT. The paper does not clearly delineate how MLOT differs from these treatments beyond recomputing intermediate marginals.
-  The downstream sections would benefit from extra sentences on how CLIP features are normalized into histograms and how $(P_1,P_2)$ is converted into a retrieval score, plus a short comment on the fine-tuning schedule.
- I would appreciate a short sensitivity study on $K$ or $(\epsilon,\tau)$, and the reference list should include the GPU-friendly, graph-based OT solver of [1]. These adjustments seem feasible post-acceptance.

[1] Lahn et al.. “A Combinatorial Algorithm for Approximating the Optimal Transport in the Parallel and MPC Settings,” NeurIPS 2023.

### Questions
- Eq.~(8) features $(u_k\odot v_{k-1})^{-\epsilon/\tau}$ with little explanation can you elaborate more on this?
- How does MLOT fundamentally differ from multi-layer OT with entropic regularization beyond introducing unknown intermediate marginals? 
-  For the CLIP retrieval task, how are continuous features mapped to probability vectors $a_k$ for each layer, and how is the final ranking extracted from $(P_1,P_2)$? 
-  Could you report GPU memory details for the large synthetic problems as well?

### Soundness
3

### Presentation
3

### Contribution
3
