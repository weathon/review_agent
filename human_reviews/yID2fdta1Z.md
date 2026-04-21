# Robust Graph Neural Networks via Unbiased Aggregation

- Avg Score: 5.00
- Decision: Reject
- Scores: 5, 5, 5, 5, 5

## Abstract
The adversarial robustness of Graph Neural Networks (GNNs) has been questioned due to the false sense of security uncovered by strong adaptive attacks despite the existence of numerous defenses.
In this work, we delve into the robustness analysis of representative robust GNNs and provide a unified robust estimation point of view to understand their robustness and limitations.
Our novel analysis of estimation bias motivates the design of a robust and unbiased graph signal estimator.
We then develop an efficient Quasi-Newton iterative reweighted least squares algorithm to solve the estimation problem, which unfolds as robust unbiased aggregation layers in GNNs with a theoretical convergence guarantee. 
Our comprehensive experiments confirm the strong robustness of our proposed model, and the ablation study provides a deep understanding of its advantages.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper reviews the estimation bias of several representative GNNs. It proposes a quasi-Newton iterative reweighted LS algorithm to optimize a GNN model, whose loss is based on the minimax concave penalty. This penalty alleviates the bias in typical GNN models and is also more robust.

### Strengths
The paper proposes a quasi-Newton iterative reweighted LS algorithm to optimize a GNN model, whose loss is based on the minimax concave penalty. This penalty alleviates the bias in typical GNN models and is also more robust.

### Weaknesses
1. The paper is not well-written, with notations that are not always defined. E.g., the superscript (0) first appears in eq. (1) but has never been defined. In addition, the paper uses a strange definition of $\ell_1$ norm to mean either the usual 1-norm or the 2-norm. The $\ell_2$ "norm" in this paper is defined to be the square of the usual 2-norm. This is technically not a norm!

1. The contributions are weak as the formulation is essentially based on TWIRLS with the MCP penalty by Zhang, 2010. Robustness issues in Lasso regression have also been well-studied.

1. Most of the cited works are from 2021 or before. The authors are missing recent GNN robustness works like “On the robustness of graph neural diffusion to topology perturbations", NeurIPS 2022 and “Graph-coupled oscillator networks", ICLR 2022.

1. The experiments are limited with tests only on Cora and CiteSeer. Attacks like TDGIA and MetaGIA are not considered.

### Questions
1. A quasi-Newton optimization approach is proposed. Is this related to standard quasi-Newton optimization procedures like BFGS, Broyden, etc.?

1. RUGE is claimed to be unbiased. Is this true and is it proven? An analytical robustness measure is also not provided.

### Soundness
4 excellent

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a robust model (RUNG) based on unbiased aggregation. Specifically, the authors first unify a few robust GNNs through the lens of $l_1$-based graph smoothing, while they still suffer from accuracy degradation when subjected to large attack budgets. To build a robust model against attacks with a large budget, authors introduce a robust and unbiased estimator by minimizing an objective function with a minimax concave penalty. Subsequently, an efficient Quasi-Newton iterative algorithm is used to optimize the objective function, from which RUNG is derived. Experimental results demonstrate that RUNG outperforms the baseline (defense) GNNs on CoraML and Citeseer, under most attack settings.

### Strengths
- Overall, the paper is well-written.
- The unified view of robust estimation on prior GNNs is interesting.
- Authors have considered adaptive attacks.

### Weaknesses
- The proposed method relies on a strong assumption of graph homophily, raising concerns about the performance of RUNG on heterophilic datasets. Thus, conducting additional experiments on heterophilic graphs is necessary. Otherwise, authors should explicitly discuss this limitation in the manuscript.
- Authors leverage QN-IRLS to approximate the inverse Hessian matrix to address the scalability issue. However, the experiments are only conducted on two small graphs, making it unclear whether the proposed approach would still work on large graphs.
- The accuracy improvement under small attack budgets is less convincing. For instance, the accuracy gap between RUNG and the runner-up model is often smaller than the standard deviation, as observed under local attacks with a 20% budget on CoraML. My concern is that RUNG might underperform the baselines with a new random seed, considering that the authors obtained averaged accuracy using only 5 different random splits.
- Authors have not performed sensitivity analyses on critical hyperparameters such as $\lambda$ and $\gamma$. It's unclear how these hyperparameters affect the performance of RUNG.
- Some recent defense models (e.g., [1, 2, 3]) are not compared in this work.
- Typo: the runner-up model is SoftMedian rather than RUNG-$l_1$ under the 5% attack budget in Table 2.

[1]: Li et al., "Reliable Representations Make A Stronger Defender: Unsupervised Structure Refinement for Robust GNN", KDD'22. \
[2]: Deng et al., “GARNET: Reduced-Rank Topology Learning for Robust and Scalable Graph Neural Networks”, LoG'22. \
[3]: Lei et al., "EvenNet: Ignoring Odd-Hop Neighbors Improves Robustness of Graph Neural Networks", NeurIPS'22.

### Questions
- What is $\beta$ in Section 2.3? I would suggest authors to use another notation, since $\beta$ has been used for SoftMedian.
- How do authors perform adaptive PGD attack on Jaccard-GCN, which preprocesses the adjacency matrix?
- Authors mention SVD-GCN as one of the baselines. Where are the results of SVD-GCN?
- It's unclear to me whether the defense under large attack budgets is practical. Is there any realistic application/scenario where an attacker can largely perturb the graph structure?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper analyzes the robust GNN models including SoftMedian, TWIRLS, and  ElasticGNN from a unified view of robust estimation. Building on this analysis, the authors introduce a robust unbiased aggregation method, further developing an efficient Quasi-Newton iterative reweighted least squares algorithm. However, the empirical validation seems somewhat limited, confined to two small graphs and a single attack setting.

### Strengths
The paper provides a unified view of $l_1$-based robust graph signal smoothing for three robust GNNs.

It introduces an unbiased graph signal estimator, which is unfolded into feature aggregation layers, aiming to enhance the robustness of GNNs.

### Weaknesses
1. The definition of $l_1$-based graph smoothing is not clear. The MCP function $\rho_\gamma$ used in Equation 4 and Equation 7 is not defined.

2. The paper does not provide an analysis of the computational complexity of RUNG, leaving its efficiency and applicability to larger datasets, such as Ogbn[1], unclear. Additionally, numerical experiments are limited, being only applied to Cora-ML and Citeseer datasets. It would be beneficial to see how RUNG performs on larger-scale datasets.

3. The description of the attack setting in Section 4.1 lacks clarity. Could you provide more details on how the adaptive evasion attack is designed based [2] you referenced? Specifically, what type of perturbations are performed during the attack - are they feature perturbations, graph topology perturbations, or both?

4. The robustness validation of RUNG in the paper is notably insufficient, solely relying on the PGD attack.
A comprehensive assessment using various attacks, as detailed in references [3][4][5][6], is necessary for a credible demonstration of RUNG's robustness.

5. Implementation code was not provided.

[1]. Hu W, Fey M, Zitnik M, et al. Open graph benchmark: Datasets for machine learning on graphs[J]. Advances in neural information processing systems, 2020, 33: 22118-22133.

[2]. Mujkanovic F, Geisler S, Günnemann S, et al. Are Defenses for Graph Neural Networks Robust?[J]. Advances in Neural Information Processing Systems, 2022, 35: 8954-8968.

[3]. Zheng Q, Zou X, Dong Y, et al. Graph robustness benchmark: Benchmarking the adversarial robustness of graph machine learning[J]. arXiv preprint arXiv:2111.04314, 2021.

[4]. Chen Y, Yang H, Zhang Y, et al. Understanding and improving graph injection attack by promoting unnoticeability[J]. arXiv preprint arXiv:2202.08057, 2022.

[5].D. Zügner and S. Günnemann, “Adversarial attacks on graph neural networks via meta learning,” in Proc. Int. Conf. Learn.Representations, 2019.

[6].D. Zügner, A. Akbarnejad, and S. Günnemann, “Adversarial attacks on neural networks for graph data,” in Proc. Int. Conf. Knowl. Discovery Data Mining, 2018.

### Questions
How to calculate the derivative in the $W_{ij}^{(k)}$ of Equation (7)?

The hyperparameter of $\lambda$ and $\gamma$  appears crucial in the model. Could you provide an ablation study to demonstrate the impact of varying these values? It would be insightful to understand how sensitive the performance of RUNG is to the selection of  $\lambda$ and $\gamma$.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors propose a unifying perspective on three "successful" graph defenses against adversarial structure perturbations. They argue that all models are instances of the so-called ElasticGNN. Moreover, the authors propose a practical optimization algorithm for the devised non-smooth objective that can approximate an L1 estimator. The resulting GNN-layer is called RUNG. The authors provide empirical evidence regarding the efficacy of their method - especially for large perturbation budgets.

### Strengths
1. Unified perspective on effective and established defenses
2. New optimization approach for an ElasticGNN derivative
3. Method shows strong empirical performance, especially for large perturbation budgets
4. The authors evaluate their defense using adaptive attacks and study the transferability of the perturbations between models

### Weaknesses
1. Empirical evaluation only using Cora ML and Citeseer is insufficient. Consider larger graphs like ogbn-arxiv as well.
1. Computational complexity and cost are neither discussed nor evaluated.
1. The authors should provide an ablation study on how the design choices affect the performance.
1. The authors should show that their method breaks. For example, complement Figure 2 with a setting where the method fails.
1. The authors state "The simulation in Figure 2 verifies that our proposed estimator (η(x) := ργ(∥x∥2)) recovers the true mean regardless of the increasing outlier ratio." Which is somewhat misleading since the asymptotically optimal breakdown point of a location estimator is 50% (under certain assumptions, e.g., without constraining the value range of the estimation).

Minor:
1. It would be interesting to see how the perturbations of RUNG transfer to the other defenses as well
1. Section 2.2 could benefit from some notes on the composition of multiple GNN layers
1. The dimension-wise median, is differentiable almost everywhere (similarly to sorting). There are only differentiability issues if the "center element" changes.

### Questions
1. Can the authors elaborate more on how the model behaves if the perturbation strength is very high? I.e., does the model then effectively become an MLP?
1. How does RUNG defy the breakdown point?
1. 10 layers of "graph smoothing" seems to be a lot! How does RUNG perform with fewer steps (e.g. as low as 2-3 as commonly used on Cora/Citeseer)?

I am willing to raise the score if the questions are being resolved and the empirical evaluation is improved (see above).

### Soundness
3 good

### Presentation
3 good

### Contribution
4 excellent

---

## Human Reviewer 5

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper provides some analysis of the robustness of various GNNs and present a unified perspective to understand their strengths and limitations.  The paper identifies an issue with estimation bias in $\ell_1$-based robust graph smoothing and proposes a robust and unbiased graph signal estimator to address this bias. The Quasi-Newton IRLS algorithm is introduced, which can be integrated into GNNs as feature aggregation layers.

### Strengths
1. The paper provides some analysis of the robustness of various GNNs, offering valuable insights into their performance under adversarial attacks.

2.  The introduction of the Quasi-Newton IRLS algorithm, which can be integrated into GNNs as feature aggregation layers, is both innovative and practical.

### Weaknesses
1. The paper posits that $\ell_1$-based robust estimation introduces an estimation bias, which purportedly deteriorates model performance, especially when an attacker adds heterophilic edges. This foundational claim lacks clarity and remains unproven. The assertion that the unveiled estimation bias explains the significant performance degradation under large attack budgets is inadequately substantiated.

In the numerical simulations, "clean samples" and "outlier samples" are generated using a Gaussian distribution. This illustration raises concerns.

- The designation of red dots as outliers is perplexing. There's no discernible reason to not categorize the red dots as clean samples. If we were to accept this alternate categorization, the entire example would be questionable.  Furthermore, when the third plot indicates that 80\% of the data are outliers, the persistence of the estimation at the center of the blue dots raises concerns. Such behavior might be influenced by factors not elaborated upon in the paper. Consequently, it's premature to commend this as a positive outcome of the proposed method.


- The proposed method offers only a rudimentary approximation of one step in GNN aggregation (7). Even if the example were valid, it doesn't convincingly demonstrate robustness in GNNs. Given that $W^k$ fundamentally varies for each $k$, there's no assurance that the single-step gradient descent aligns with the theorem.

2. The computation of $W$ in eq(7) remains ambiguous. How exactly is it derived?

3. The statement "which not only provides clear interpretability but also covers many classic GNNs as special cases" is misleading. The term "covers" is inappropriate. The paper doesn't truly encompass classic GNNs as special cases but rather offers an approximate perspective on them through the lens of Graph Signal processing.

4. The paper's limitations are not adequately addressed. Like other defense papers aiming to prune the influence of heterophilic edges, this method likely has restricted applicability, perhaps being best suited for datasets that inherently assume homophily.

5. A glaring omission is the lack of reported computational costs. This oversight leaves readers questioning the practical feasibility of the proposed approach.

6. The experimental section is inadequate. The range of attacks considered is also severely limited. It is imperative to incorporate evaluations against poison and evasion attacks, as well as both white-box and black-box scenarios, and to consider both injection and modification types.

### Questions
1. Can you address and provide clarity on the concerns highlighted in the weaknesses section?
2. The citation style throughout the manuscript is inconsistent. Can you rectify this?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
