# Accelerating Optimization and Machine Learning via Decentralization

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 0, 4

## Abstract
Decentralized optimization enables multiple devices to learn a global machine learning model while each individual device only has access to its local dataset. By avoiding the need for training data to leave individual users’ devices, it enhances privacy and scalability compared to conventional centralized learning where all data have to be aggregated to a central server.  However, decentralized optimization has traditionally been viewed as a necessary compromise, used only when centralized processing is impractical due to communication constraints or data privacy concerns. In this study, we show that decentralization can paradoxically accelerate convergence, outperforming centralized methods in the number of iterations needed to reach optimal solutions. Through examples in logistic regression and neural network training, we demonstrate that distributing data and computation across multiple agents can lead to faster learning than centralized approaches—even when each iteration is assumed to take the same amount of time, whether performed centrally on the full dataset or decentrally on local subsets. This finding challenges longstanding assumptions and reveals decentralization as a strategic advantage, offering new opportunities for more efficient optimization and machine learning.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors study distributed optimization, with local functions $f_i$ having different smoothness constants $L_i$, and discuss choosing different stepsizes $\gamma_i=1/L_i$ in the initial phase of the algorithm.

### Strengths
The paper is well written.

### Weaknesses
1) First on the term "decentralized". In decentralized settings, there is no server. This is the way "decentralized" is used in the whole literature. "server-assisted decentralized gradient methods" is an oxymoron and is misleading. You should just refer to distributed optimization, in the client-server model.

2) The contribution is weak. You consider choosing different stepsizes at the beginning, we can call this warm-up, then after a while you reset the stepsizes to $\gamma=1/L$. You claim that there are theoretical contributions, but the theoretical part is the one based on PEP. What is shows is that the worst case exhibited by the PEP is better with different stepsizes. Since the algorithm has more parameters, this is expected. 

3) The idea of using different stepsizes is not novel. One way to correct the incurred deviation is to take a weighted average at the server, see for instance the method i-Scaffnew in Yi et al. "Explicit Personalization and Local Training: Double Communication Acceleration in Federated Learning", 2025. In case of partial participation or sampling, one can do importance sampling by choosing probabilities proportional to $L_i$. 

4) The idea of adapting to the local datasets by choosing different stepsizes can be implemented by using even better matrix stepsizes, see for instance Safaryan et al. "Smoothness matrices beat smoothness constants: Better communication compression techniques for distributed optimization".

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper discusses the convergence time of gradience methods in centralized vs distributed optimization scenarios. Most notably, the paper argues that in data-heterogenous distributed set-ups where (possibly) the smoothness parameters of different agents vary, distributed/decentralized scenarios can greatly benefit from using local step-sizes (which is set as $\eta_i=1/L_i$ according to local smoothness $L_i$). The main contribution of the paper is proposing an algorithm for distributed optimization based on the local step-sizes which also has a switching threshold for fixing the step-size to $1/L$ (where L is the global smoothness) after some initial iterations.

### Strengths
The paper is well-written and studies an important problem since data heterogeneity is an important bottleneck for distributed/federated learning which can slow down the GD's convergence. The paper also presents experiments on real-world data to show its claims.

### Weaknesses
1- The idea of using local smoothness and adaptive step-sizes based on these local smoothness parameters for each client is not new, for example see the reference [*] which studies a similar idea with several local SGD step-sizes (i.e., the federated learning setup) and also includes a method for approximating the local smoothness. Can the authors please clarify how their algorithm improves upon the previous algorithms in literature? 

2- The paper does not provide substantial theoretical evidence to back its main claim i.e., that their proposed algorithm can provably lead to speedups compared to ordinary GD. Can the authors clarify whether their approach provably leads to better convergence results and/or superior test error performance? 



[*] Adaptive federated learning with auto-tuned clients, Kim et al, ICLR 2024.

### Questions
Please see the comments above.

typo:
 $x^*$ is defined twice in line 221.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
This paper considers the decentralized optimization problem with strongly convex and L-smooth objectives. They challenge the belief that centralized optimization is more efficient and argue that decentralized optimization can lead to faster convergence. They propose an algorithm which adjust the step size based on local smoothness and formulate a performance estimation problem (PEP) which characterize the worst-case performance of optimization algorithms. Solving the PEP with SDP solving yields solutions with improved worst-case performance. Empirical evaluations on W8A, MNIST, CIFAR-10 are given.

### Strengths
Adopting PEP from literature to analyze the worst case analysis look interesting.

### Weaknesses
- The notations are rather messy. The paper in the current form is not ready for publication. To name a few,
    - In Algorithm 1, S,F,T are used without definition and not explained. The F seems to be defined in Section 3.2 but is already far away from the algorithm.
    - The vectors are traditionally annotated in bold. S in algorithm 1 use \mathcal while F, T does not.
    - Between 120 and 127, $g_i^k$ is used to denote both gradient computed on local copies and shared copies which makes it a bit ambiguous which $g_i^k$ in algorithm refers to.
    - Line 238, $f(x^*)$ is an example of abuse of notation as $x^*$ is denoted as a list of vector in line 221.
- Line 241: not sure what does ".... are interpolated for each local function class" mean. 
- Line 242: "... are generated recursively by algorithm 1" yet algorithm has no $x_i^k$.
- Claims are not properly reference, e.g. line 107 "it is well known..."
- Line 180 says the algorithm 1 "converges to an exact optimal solution as conformed by theoretical evaluations" and refer to Fig. 1 which is clearly an empirical evaluation on a simple problem.

More importantly, there are already abundant works decentralized optimization (with or without heterogeneity) yet their convergence rates and empirical performances are not compared in the paper.

### Questions
See weakness.

### Soundness
2

### Presentation
1

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
This paper claims that decentralization, where data is distributed across devices, can accelerate convergence compared to centralized training. Specifically, the authors propose Algorithm 1, a server-assisted method that sets a local learning rate based on the local smoothness constant ($L_i$). This mechanism is intended to leverage data heterogeneity, as the data distribution impacts the local loss landscape and thus $L_i$. While the paper's central claim is about the benefit of *decentralization*, the core mechanism relies specifically on *heterogeneous local learning rates*. The experimental validation for the proposed algorithm is somewhat limited.

### Strengths
The paper's primary strength is its core finding: decentralized optimization can, under certain conditions, outperform its centralized algorithms in convergence speed (i.e., iteration count). This result is very impressive and challenges the conventional wisdom that decentralization is merely a necessary compromise for privacy or communication constraints.

### Weaknesses
1.  **Algorithmic Clarity:** The practical implementation of setting local learning rates could be clarified. The method relies on the local smoothness constant $L_i$, and a more detailed discussion on the strategy for estimating this value would be beneficial. For instance, it would be helpful to understand the practical trade-offs of this estimation, such as the computational overhead required (e.g., how much time this adds) and the algorithm's sensitivity to estimation error (e.g., how accurately $L_i$ must be determined for the method to remain effective).

2.  **Novelty and Related Work:** In terms of novelty, the idea of client-level or local learning rates has been explored in related Federated Learning literature (e.g., FedSPS [1]), though not in direct comparison to centralized methods. The paper would be strengthened by a more thorough discussion of relevant works on FL with local learning rates to better contextualize its own contribution.

3.  **Experimental Setup:** The main weakness of the paper lies in its experimental setup. The empirical validation is weak, as it is limited to logistic regression and simple CNNs on MNIST and CIFAR-10. Furthermore, the proposed algorithm is only compared against standard Gradient Descent (GD), in both full-batch and mini-batch settings. A more convincing validation would require comparison against other modern optimizers.

### Questions
1.  The paper states, "we estimate the smoothness constant using 1,000 data entries, following its formal definition." Could the authors please provide the precise details of this estimation procedure? How sensitive is the algorithm's performance to the accuracy of this estimation?

2.  The related work section appears to be missing several highly relevant citations that also discuss the benefits of decentralization. A discussion of these works is essential to position the paper's contribution correctly.
    * For example, [2] suggests FedAvg can outperform (centralized) mini-batch SGD even when gradient divergence is large.
    * [3] found that local SGD's steps introduce an implicit bias that aids the drift towards flat minima.
    * [4] found that the consensus distance in decentralized SGD can act as a benign noise, serving as a sharpness regularizer.

References
---
[1] Locally Adaptive Federated Learning. TMLR.

[2] A New Theoretical Perspective on Data Heterogeneity in Federated Optimization. ICML, 2024.

[3] Why (and When) does Local SGD Generalize Better than SGD? ICLR, 2023.

[4] Decentralized SGD and Average-direction SAM are Asymptotically Equivalent. ICML, 2023.

### Soundness
2

### Presentation
3

### Contribution
2
