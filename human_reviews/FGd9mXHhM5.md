# Achieving Optimal Breakdown for Byzantine-Robust Gossip

- Avg Score: 5.00
- Decision: Reject
- Scores: 3, 6, 6

## Abstract
Distributed approaches have many computational benefits, but they are vulnerable to attacks from a subset of devices transmitting incorrect information. This paper investigates  Byzantine-resilient algorithms in a decentralized setting, where devices communicate directly with one another. 
We investigate the notion of breakdown point and show an upper bound on the number of adversaries that decentralized algorithms can tolerate. We introduce an algorithmic framework that recovers ClippedGossip and NNA, two popular approaches for robust decentralized learning, as special cases. This framework allows us to generalize NNA to sparse graph, and introduce CG+, which is at the intersection of the two. Our unified analysis framework gives near-optimal guarantees for CG+ (and other approaches with additional assumptions). Experimental evidence validates the effectiveness of CG+ and the gap with NNA, in particular against a novel attack tailored to sparse graphs that we introduce.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper analyzes the breakdown point of robust algorithms in decentralized optimization and proposes the CG Plus method, which achieves the optimal breakdown point. Convergence guarantees and experimental results are provided to support the proposed method.

### Strengths
1. The analysis of the breakdown point for robust algorithms in decentralized optimization is novel and has not been previously explored. 


2. The proposed CG Plus is well-motivated, addressing the impractical clipping threshold issue in ClippedGossip.

### Weaknesses
1. The experimental evaluation is insufficient. The authors only conducted experiments on the MNIST dataset, and additional results from other datasets are recommended to strengthen the conclusions.

2. The experimental results do not clearly demonstrate the advantage of CG Plus. In fact, ClippedGossip seems to outperforms the proposed CG Plus overall. The authors' claim that "ClippedGossip might fail against other attacks" is unconvincing. It is recommended that further results be provided to demonstrate CG Plus's advantage over ClippedGossip; otherwise, the development of CG Plus seems unjustified.

### Questions
1. The authors utilize the Laplacian matrix of the graph as the gossip matrix. However, in decentralized optimization, the gossip matrix is typically required to be doubly stochastic (He et al., 2022). Moreover, in Section 4.1, the weighted averaging step with $W_{ij}$ in ClippedGossip  (He et al., 2022) is replaced by a "communication step-size" $\eta$. The condition $\eta \leq 1/u_{max}$ seems unable to recover traditional averaging when there are no Byzantine agents. The authors should provide more explanation regarding $\eta$ and clarify why they chose not to use the conventional doubly stochastic gossip matrix. 

2. The paper investigates the breakdown point in terms of the number of Byzantine agents. However, in traditional Byzantine-robust decentralized optimization, the focus often lies on the gossip weights assigned to Byzantine agents, rather than merely the number of Byzantine agents (He et al., 2022). For instance, if more than half of the neighbors are Byzantine, but the total gossip weight assigned to them is minimal, the algorithm can still remain robust. Does this perspective conflict with Theorem 1? It is recommended that the breakdown point analysis consider the gossip matrix in more detail.

### Soundness
1

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
4

### Summary
This paper analyzes the optimal breakdown point of robust aggregators in the Byzantine-robust decentralized average consensus problem and proves that the breakdown point of the proposed CG$^+$ method almost aligns with the optimal values. To further validate the effectiveness of the CG$^+$ method in general Byzantine-robust decentralized stochastic problems, this paper examines its theoretical convergence and demonstrates its practical performance compared to existing methods in experiments.

### Strengths
The analysis of the optimal breakdown point of robust aggregators is both novel and significant in the field of Byzantine-robust decentralized learning.

### Weaknesses
1. The lower bound of the breakdown point indicated in Theorem 1 appears to be not tight, as the breakdown point of CG$^+$ presented in Theorem 2 ($b < \frac{\mu_{\min}}{2} - 1$) does not align with the lower bound ($b \geq \frac{\mu_{\min}}{2}$). This leaves a gap when $b \in$ { $\lceil \frac{\mu_{\min}}{2} - 1 \rceil, \lfloor \frac{\mu_{\min}}{2} \rfloor$}. Therefore, there may be better methods available that can match the lower bound and tolerate more Byzantine neighbors than CG$^+$. I suggest the authors discuss whether this gap is fundamental or if CG$^+$ could potentially be improved to match the lower bound exactly. 
		
2. In Corollary 2, the authors only demonstrate the optimality of the proposed CG$^+$ method. I would like to know the theoretical consensus rate of the honest models when using this method. Can the honest models achieve consensus by the end of the training? Could the authors provide theoretical guarantees on how quickly or to what degree consensus is achieved among honest nodes in the training process?
		
3. The proposed CG$^+$ method does not demonstrate any performance improvement over the existing ClippedGossip in the experiments, which raises doubts about the practical effectiveness of CG$^+$. 

4. Equation (CG) appears to be not correct. The update rule of ClippedGossip involves a doubly-stochastic mixing matrix to aggregate messages from neighbors, while (CG) does not include such a mixing matrix. Please refer to (He et al., 2022) and correct this equation.

### Questions
My detailed questions are listed in the above section; please refer to it.

### Soundness
2

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
The authors prove a theoretical upper bound on the breakdown point for Byzantine-robust distributed machine learning in decentralized frameworks, and then propose a novel method called $CG^+$, which can achieve the proven upper bound. The theoretical convergence guarantee of $CG^+$ is provided together with empirical evaluation in this paper.

### Strengths
+ The problem of obtaining Byzantine robustness in distributed learning on a decentralized framework is challenging and meaningful. 
+ This paper is generally well organized. 
+ This paper proves a new upper bound of the breakdown point and proposes a novel method, which can reach the optimal breakdown point.

### Weaknesses
- The proposed method $CG^+$ is like a combination of ClippedGossip and NNA, and thus the novelty of the proposed method is a little bit limited (I understand that $CG^+$ has a better performance and guarantee than each of the two methods). 
- Could the authors briefly explain why the clipping scheme in $CG^+$ can achieve a better theoretical guarantee than ClippedGossip?
- There are replicated references (lines 577-583).

### Questions
Please see the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
