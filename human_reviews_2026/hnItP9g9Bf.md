# A Strategy-Agnostic Framework for Partial Participation in Federated Learning

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 2, 6

## Abstract
Partial participation (PP) is a fundamental paradigm in federated learning, where only a fraction of clients can be involved in each communication round. In recent years, a wide range of mechanisms for partial participation have been proposed. However, the effectiveness of a particular technique strongly depends on problem-specific characteristics, e.g. local data distributions. Consequently, achieving better performance requires a comprehensive search across a number of strategies. This observation highlights the necessity of a unified framework. In this paper, we address this challenge by introducing a general scheme that can be combined with almost any client selection strategy. We provide a unified theoretical analysis of our approach without relying on properties specific to individual heuristics. Furthermore, we extend it to settings with unstable client-server connections, thereby covering real-world scenarios in federated learning. We present empirical validation of our framework across a range of PP strategies on image classification tasks, employing modern architectures, such as FasterViT.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes PPBC, a strategy-agnostic framework for federated learning with partial client participation, and its variant PPBC+, which handles unavailable or offline clients. The framework introduces a surrogate-gradient accumulation mechanism to compensate for biased or stochastic client sampling and provides convergence guarantees under both non-convex and strongly convex objectives. It also provides theoretical results of the proposed PPBC under nonconvex and strongly convex conditions. Experiments on CIFAR-10 and Food101 with ResNet-18 and FasterViT show that PPBC and PPBC+ outperform baselines across various sampling strategies.

### Strengths
1. This paper aims to design a strategy-agnostic PPBC, which integrates different client sampling rules and participation strategies. 

2. The authors provided the theoretical analysis under nonconvexity and strong convexity. 

3. Experimentally, PPBC shows obvious advantages compared to all baselines. (Need to be verified if the authors could provide more evidence, such as source code. )

### Weaknesses
W1. According to Line 20 of Algorithm 1, the proposed PPBC requires all clients to send $g_{m}^{k,H^k}$ to the server.  

W2. The Assumption 2.3 does not specify the ranges of the constants $\delta_1$ and $\delta_2$, even though this assumption is commonly used.  

W3. In the description of Algorithm 1 (lines 251-253), it says that the number of participating clients can vary. However, according to the definition of $C$ in Corollaries 3.3, 3.5, 3.6 and 3.7, the number of participating clients in each epoch in Algorithms 1 and 2 is fixed. It leads to an inconsistency between the algorithm and theory, which is my big concern about this paper. The case in the theory is just a special case of their proposed algorithm. 

W4. In Sections 3.2-3.4, the discussion on theoretical results could be deeper and enriched. For example, it will be helpful for readers to understand the theoretical connection if the authors can specifically compare the theoretical results of partial participation with and without unavailable clients. 

W5. The use of a Top-$C$ client selection in Section 4.1 conflicts with the definition of full client participation. Full participation means all clients are sampled in each round. Selecting only the Top-$C$ clients results in partial participation, and therefore, the experimental terminology and settings should be clarified or corrected.  

W6. For the experiment results in Tables 1 and 2, the proposed PPBC achieved a significant improvement in test accuracy, compared to the baselines. Based on my personal experience, it is uncommon. Moreover, I did not find any source code in the supplementary materials, which makes it difficult for me to verify the reproducibility of the reported results.

**Minor:** 

W7. Line 261: "an ablation studies" -> "ablation studies"

w8. Line 335: "we refuse using" -> "we refuse to use"

### Questions
Q1. Can the authors explain the motivation to use the Geometric distribution to generate different numbers of iterations? Can it bring any advantage?

Q2. Can the authors explain how the unavailable clients affect the algorithm in theory?

Q3. In Algorithm 1, the server performs client sampling twice (in lines 5 and 10). Could the authors clarify why two separate sampling steps are necessary? Or what is the motivation behind that? 

Q4. Can the authors provide a communication cost analysis or comparison, theoretically or experimentally, of PPBC, PPBC+, and baselines? Since PPBC and PPBC+ need to communicate more than FedAvg, the analysis or comparison will enhance the understanding of the proposed algorithm. 

I would consider updating my rate if my concerns can be addressed properly.

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
This paper studies the problem of partial client participation in Federated Learning, where only a subset of clients participates in each communication round. Existing works often assume unbiased sampling or analyze specific strategies (e.g., FedAvg, FOLB, PoC). The authors propose a strategy-agnostic theoretical framework (PPBC) and an extension *PPBC+* for cases with unstable or unavailable clients. The key idea is to introduce gradient surrogates that accumulate information from inactive clients and correct bias caused by partial participation. The paper provides convergence proofs under both convex and non-convex assumptions, without relying on bounded gradient norms. Theoretical rates are claimed to improve over existing works by removing non-vanishing terms. Empirical studies on CIFAR-10 (ResNet18) and Food101 (FasterViT) show better accuracy compared to several baselines, e.g., FedAvg, SCAFFOLD, FedDyn, MOON, and F3AST.

### Strengths
1. The paper targets biased partial participation, which is a long-standing issue in federated optimization. This paper provides a unified theoretical analysis that relaxes some restrictive assumptions in prior works.
2. The idea of accumulating “gradient memory” from inactive clients to correct bias resembles error feedback mechanisms, but it is interestingly reinterpreted in the context of non-contractive sampling operators.
3. The derivations are detailed and mathematically consistent, with convergence rates derived for both convex and non-convex settings.

### Weaknesses
Despite its rigorous math and structure, the paper has serious issues regarding novelty, clarity, and experimental validation.

### 1. Theoretical novelty is overstated.

- The proposed framework (PPBC) is essentially an adaptation of error-feedback (EF) and compressed SGD ideas (see Stich & Karimireddy 2020; Richtárik et al. 2021) applied to partial participation.
- The "strategy-agnostic" claim is misleading: the algorithm still requires specifying a client sampling rule $R$ and weighting scheme $π$, hence it is not truly agnostic, but merely compatible with multiple heuristics.
- The main theoretical contributions (Theorem 3.2 and 3.4) resemble classical results from distributed optimization and show only incremental extensions to known bounds (O(1/ε²) for nonconvex, O(κ log(1/ε)) for convex).
- Algorithm 1 (PPBC) and Algorithm 2 (PPBC+) largely mimic FedAvg with gradient memory buffers and stochastic masking; there is no fundamentally new optimization idea.

### 2. Lack of meaningful baselines and comparisons.

- The paper compares PPBC mainly to generic optimizers (FedAvg, SCAFFOLD, FedDyn), not to modern partial participation frameworks such as:
  - FedNova, FedProx, FedPAQ, or adaptive sampling methods (Oort, F3AST, FedCS+).
- The fairness of the comparison is questionable: all methods use the same sampling rule $R_k = Top_C(π_k)$, which may favor the proposed algorithm by design.

### 3. Overly dense and verbose presentation.

- The paper reads more like a mathematical report than a conference paper.
  - The Introduction (Section 1) is excessively long (over 3 pages).
  - Many definitions (e.g., Assumptions 2.1–2.4) are standard and could be condensed.
- Overall readability and intuition are limited; it is difficult to see *why* the new framework helps beyond mathematical manipulation.

### Questions
I have no question for this paper. Some suggestions are provided for improvement. 

1. Clarify what is new compared to error-feedback, EF21, and FedDyn frameworks.
2. Add communication cost and time-to-convergence metrics to demonstrate efficiency.
3. Reduce redundancy in the introduction and add intuitive explanations and diagrams.
4. Include theoretical intuition: why does bias correction help convergence in non-IID FL?

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
3

### Summary
The paper addresses the fundamental challenge of partial participation (PP) in federated learning, where only a subset of clients participate in each communication round. The authors propose PPBC (Algorithm 1) and its extension (Algorithm 2), two strategy-agnostic optimization frameworks can work with arbitrary client weighting and sampling strategies. The authors provide unified convergence guarantees for both non-convex and strongly-convex objectives, with bounds that avoid previously observed non-vanishing bias terms.

### Strengths
It is a general framework that supports a wide range of weighting and sampling rules, biased or unbiased.

### Weaknesses
1. It is unclear to me why we need a general framework for partial participation. We know that client participation not only depends on server but also has close relation with system and client itself. 

2. The unified theoretical analysis is claimed as one of the major contributions, but I do not see how this generate existing results. I expect a table to show all the comparisons. 

3. There are so many flexible client participation works, including [1-5] and more. I am curious about the comparison between the proposed framework and analysis and these existing works.

[1] Wang, Shiqiang, and Mingyue Ji. "A unified analysis of federated learning with arbitrary client participation." Advances in neural information processing systems 35 (2022): 19124-19137.
[2] Cho, Yae Jee, Jianyu Wang, and Gauri Joshi. "Towards understanding biased client selection in federated learning." International Conference on Artificial Intelligence and Statistics. PMLR, 2022.
[3] Wang, Shiqiang, and Mingyue Ji. "A lightweight method for tackling unknown participation statistics in federated averaging." arXiv preprint arXiv:2306.03401 (2023).
[4] Li, Zhe, et al. "FAST: A Lightweight Mechanism Unleashing Arbitrary Client Participation in Federated Learning." IJCAI (2025)
[5] Yang, Haibo, et al. "Anarchic federated learning." International Conference on Machine Learning. PMLR, 2022.

### Questions
Please include the theoretical and experimental comparions with existing flexible client participation works in federated learning.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a new theoretical and algorithmic framework for federated learning (FL) under partial participation (PP). Unlike previous works, which rely heavily on specific strategies like FedAvg and often require restrictive assumptions (e.g., gradient boundedness or unbiased sampling), the authors propose a strategy-agnostic and theoretically sound framework that supports a wide range of biased sampling and client weighting strategies. They introduce two main algorithms: 1) PPBC: Partial Participation with Bias Correction, which maintains gradient surrogates on inactive clients and introduces a bias-corrected aggregation; and 2) PPBC+: An extension to handle unavailable clients, modeling dropout with Bernoulli trials. Theoretical results show that the framework achieves convergence in both non-convex and strongly convex settings without requiring non-vanishing error terms in theory. Experimental results on CIFAR-10 and FOOD101 show superior performance of the framework over existing baseline algorithms.

### Strengths
1. The paper develops tight convergence rates for both non-convex and strongly convex objectives without relying on unnatural assumptions (e.g., bounded gradients or bounded differences), thereby overcoming limitations of prior work.
2. The proposed framework decouples the optimization procedure from client selection and weighting strategies, enabling flexible, plug-and-play adaptation across a wide range of partial participation settings.
2. The algorithm PPBC+ explicitly accounts for unavailable clients, making it more practical and robust for real-world federated learning scenarios.

### Weaknesses
1. The experiments report results in terms of communication rounds but do not discuss communication overhead, memory consumption, or wall-clock time, all of which are critical factors in practical federated learning (FL) deployments.
2. All baselines adopt the authors’ fixed Top-C sampling strategy; however, methods like FedDyn and Moon were originally designed with their own adaptive client selection mechanisms. Enforcing a uniform sampling rule may underestimate their true performance, making the comparison potentially unfair.
3. The algorithms PPBC+ and F3AST are omitted from the CIFAR-10 experiments, while PPBC and several other baselines are omitted from the FOOD101 experiments, resulting in an incomplete experimental evaluation that limits cross-setting insights.
4. The terms "full client participation" and "partial client participation" used in Section 4 are potentially confusing. A more precise terminology would be: "partial client participation without unavailable clients/devices" and "partial client participation with unavailable clients/devices."

### Questions
In the proposed framework, even clients that are not selected for communication are required to compute and update their gradient surrogates, which may lead to a significant increase in computation cost. Furthermore, beyond the communication at each epoch $k$, additional communication is required at every iteration $h$ within each epoch, introducing additional communication overhead. Could the authors quantitatively analyze these extra computation and communication costs?

### Soundness
3

### Presentation
2

### Contribution
3
