# A Sublinear Adversarial Training Algorithm

- Decision: Accept (poster)
- Scores: 3, 6, 6, 8

## Abstract
Adversarial training is a widely used strategy for making neural networks resistant to adversarial perturbations. For a neural network of width $m$, $n$ input training data in $d$ dimension, it takes $\Omega(mnd)$ time cost per training iteration for the forward and backward computation. In this paper we analyze the convergence guarantee of adversarial training procedure on a two-layer neural network with shifted ReLU activation, and shows that only $o(m)$ neurons will be activated for each input data per iteration. Furthermore, we develop an algorithm for adversarial training with time cost $o(m n d)$ per iteration by applying half-space reporting data structure.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors consider fully connected one hidden layer neural networks with shifted ReLU activations and develop an adversarial training algorithm with cost $o(mnd)$ per iteration by applying half-space reporting data structure. To do that, the authors rely on three main techniques:
1. approximation via pseudo-network sufficiently precise with high probability
2. using shifted ReLU functions instead of standard ones to attain $o(mnd)$ time complexity per training iteration
3. use a tree data structure to find the neurons that need updating more quickly (Half-space reporting)

The algorithm consists of a loop that first calculates the adversarial training set and the indices of the activated neurons for each element of the training set, then doing a gradient update for all active neurons. A series of theorems and lemmata are presented, analyzing the proposed algorithm, which is followed by a time complexity analysis. The reviewer did not check the proofs in the appendix.

### Strengths
- Adversarial training algorithms and their analysis is an important problem in the field of adversarial robustness. 
- The assumptions seem reasonable.

### Weaknesses
- The relevance of the paper remains unclear. Is it theoretically relevant or also practically?
- The restriction to 1 hidden layer networks is strong. 
- The Paper is hard to read, the reasoning jumps back and forth. The reviewer recommends a thorough rewrite. It would be good if the authors first introduce concepts and then use them. An example is the "query, insert, remove" from the data structure in Algorithm 1 (Page 6) that gets introduced in Section 5 (Page 8). Similarly, Theorem 1.1 (Page 2) has a reference to Equation 2 (Page 4). The paper seems to be rushed, also reads that way. 

Typos:
	- 3.1 the typesetting of $\ell_p$ 
	- The inequality in Theorem 4.1 seems to be the wrong way round. 
	- In theorem 4.3: "$T = \Theta(\epsilon^{-2} K^2)$. in Algorithm"

### Questions
- Are the contributions of the paper mainly of theoretical or also practical interest? Can the algorithm be efficiently parallelized? Are efficient vectorized/GPU implementations possible? What are the memory requirements of the algorithm? 
- Page 3: what is the definition for an $\epsilon$-net?
- What about deeper networks? On an intuitive level, the reviewer does not see a reason that the complexity should worsen there.

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes an adversarial training algorithm for 2-layer neural networks with ReLU activation that has sublinear o(mnd) per-iteration time complexity, compared to standard Ω(mnd). The result mainly relies on the insight that only o(m) neurons will be activated for each input data per iteration.

### Strengths
1. The paper presents some interesting result on analyzing per-iteration complexity of adversarial training of a special setting of 2-layer neural networks. Theoretical analyses are solid and insightful.
2. Complexity of adversarial training is less analyzed compared with standard training, the work would be interesting to some researchers.

### Weaknesses
1. Just like in many other works on 2 layer neural nets, it is unclear if the insight can generalize to the more common setting of deeper neural nets.
2. It seems only optimal adversaries are analyzed (e.g., in theorem 4.4), but most cases it is impractical to have an optimal adversary.

### Questions
Could authors provide some discussions on and thoughts on if the activation sparsity could generalize to deeper nets?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors propose an adversarial training algorithm designed for two-layer ReLU networks. Combining multiple techniques, they manage to prove that their algorithm runs in time sublinear in the size of the network.

### Strengths
- The authors propose an interesting algorithm and manage to show that per iteration cost is sublinear in the size of the network for a two-layer ReLU network. The paper is very clearly written with detailed proofs in the appendix.  

- The use of the techniques such as half-space reporting data structures and shifted ReLU in the design of the algorithm are interesting and instructive.

### Weaknesses
- The current result seems to be more of a theoretical interest than a practical algorithm. The limitation of two-layer network, well-separation for adversarial loss are both severe limitations in practice. 

- There are also some unclear or missing details that need to be addressed in the Questions section below.

### Questions
- Is there any assumption on the adversarial example oracle A in Algorithm 1? I don't seem be able to find any reference to its complexity analysis in the main paper. Usually adversarial examples are found by iterative projected gradient descent (on the input x) and their runtime should also be taken into account. 

- If we take away the adversarial example generation in Algorithm 1 and just use the input x_i as training example, we should then obtain a sublinear training algorithm for a two-layer ReLU network. I am wondering why the authors are focusing on adversarial training, or if there are any parts of the algorithm that are specific to the problem of adversarial examples. The 3 techniques listed in Section 2 are not specific to adversarial training.

### Soundness
4 excellent

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper tries to address the computational complexity of adversarial training in neural networks, which is an active research area.
The work addresses training efficiency by trading off extra memory for computation and incorporating an additional half-space data structure, efficiently identifying only the active neurons for computation. The key insight is that not all neurons are active for every input data point, enabling the computation of the dot product exclusively with the active neurons. While the method of initially identifying active neurons is not novel, the application of efficient geometric data structures is both new and promising. The research introduces an algorithm designed for this purpose; although it does not include experimental results, it does offer convergence proofs for two-layer networks.

### Strengths
The approach is novel where adversarial training that leverages geometric data structures at the expense of extra space. It extends the direction of first identifying the active neurons and then only taking dot product with those neurons. 

The paper is well written and however, without empirical results or illustrative examples, the clarity of the paper's practical implications could be enhanced.  

The work is significant as it has a potential to further improve the direction of efficient adversarial training of robust neural networks.

I have not conducted an exhaustive review of the convergence proofs; however, upon initial examination, they appear to be correct.

### Weaknesses
The study lacks experiments, even on toy datasets, to determine whether its findings are applicable in practical scenarios.

Furthermore, it has not been compared with existing works. Although it is primarily theoretical, clarity on how it measures up against established methods is missing.

The paper does not discuss how reducing training time might impact the robustness performance of the model. This trade-off between efficiency and robustness is crucial for practical use cases.

Likewise, the trade-off between memory usage and efficiency warrants discussion. While the paper could comment on memory requirements, it may be that for only two-layer networks, the difference is minimal. However, it would still be beneficial to understand this aspect, perhaps by employing very wide neural networks to confirm it in a practical scenario.

It is also unclear whether the proposed methods can be effectively translated into actual GPU computations. If the approach results in underutilized GPUs, it might still be valuable but may not reduce the overall wall time. Some commentary on this feasibility would be constructive.

### Questions
To translate the theoretical advantages into practical scenarios, the paper should present at least some toy experiments.

It is important to explore how reducing training time affects the model's robustness performance, as the trade-off between efficiency and robustness is crucial for practical use cases.

The same consideration should be given to the trade-off between memory usage and efficiency. It would be beneficial for the paper to comment on memory requirements. While the impact on only a two-layer network might be minor, understanding this trade-off is essential, and could be tested using neural networks with extremely high widths in practical scenarios.

In short, a toy experiment featuring a two-layer network that showcases practical, minimal use cases by comparing memory usage, computational efficiency, and robustness performance would be a valuable addition to the work.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
