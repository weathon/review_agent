# Robust and Efficient Collaborative Learning

- Decision: Reject
- Scores: 4, 4, 2, 4

## Abstract
Collaborative machine learning is challenged by training-time adversarial behaviors. Existing approaches to tolerate such behaviors either rely on a central server or induce high communication costs. We propose *Robust Pull-based Epidemic Learning (RPEL)*, a novel, scalable collaborative approach to ensure robust learning despite adversaries. RPEL does not rely on any central server and, unlike traditional methods, where communication costs grow in $\mathcal{O}(n^2)$ with the number of nodes $n$, RPEL employs a pull-based epidemic-based communication strategy that scales in $\mathcal{O}(n \log n)$. By pulling model parameters from small random subsets of nodes, RPEL significantly lowers the number of required messages without compromising convergence guarantees, which hold with high probability. Empirical results demonstrate that RPEL maintains robustness in adversarial settings, competes with all-to-all communication accuracy, and scales efficiently across large networks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Robust Pull-based Epidemic Learning (RPEL), a decentralized Byzantine-robust collaborative learning algorithm.
Unlike traditional robust federated or decentralized learning methods that require all-to-all communication (O(n²) cost) or a trusted central server, RPEL uses a randomized pull-based epidemic protocol, where each node periodically pulls model parameters from a small, random subset of peers (O(log n) neighbors). Theoretical analysis shows convergence in nonconvex settings with data heterogeneity and omniscient adversaries. Empirical results on MNIST and CIFAR-10 show that RPEL achieves accuracy comparable to all-to-all robust methods but with significantly reduced communication.

### Strengths
- An interesting idea: Using pull-based epidemic communication for robust decentralized learning is neat. It meaningfully reduces communication overhead while preserving robustness.

- Solid theoretical grounding: The convergence proofs are rigorous and build upon established robust aggregation theory (e.g., Allouah et al. 2023). The notion of an Effective Adversarial Fraction provides an intuitive probabilistic understanding of robustness under random sampling.

- Empirical competitiveness: Despite reduced communication, RPEL maintains similar accuracy to all-to-all robust methods under several common Byzantine attacks (SF, FOE, ALIE).

### Weaknesses
- Limited empirical validation: Experiments are restricted to small datasets (MNIST, CIFAR-10) and small clusters (100 clients). The claims of large-scale scalability remain untested in actual large or high-dimensional systems.

- Attack model realism: Tested adversaries are restricted (sign-flipping, ALIE, FOE). Many powerful and classic attacks (Krum, Bulyan, Fang attack) are ignored, not to mention analysis against coordinated or sampling-aware adversaries that could exploit the pull mechanism.

- Parameter sensitivity: Robustness depends critically on the correct choice of sampling size s and effective adversary bound. These hyperparameters are tuned heuristically without sensitivity analysis.

### Questions
- How sensitive is RPEL to the sampling parameter s? Does small mis-specification significantly reduce robustness?

- Could adaptive adversaries bias the pull selection process or predict which nodes will be queried?

- How would RPEL perform when the network topology is inherently sparse and not fully connected?

- Can the concept of the Effective Adversarial Fraction be empirically validated beyond simulation (e.g., real-world distributed setting)?

### Soundness
3

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
This paper introduces the Robust Pull-based Epidemic Learning (RPEL) protocol, a novel and scalable collaborative approach designed to ensure robust learning despite omniscient Byzantine adversaries. By employing a pull-based epidemic communication strategy, RPEL significantly reduces the communication complexity. The work establishes rigorous finite-time convergence guarantees under non-convex settings and introduces the concept of the Effective Adversarial Fraction as a key robustness metric. Empirical results validate effectiveness of the proposed method against state-of-the-art attacks.

However, its practical impact is currently limited by two key areas needing improvement. First, the strict reliance on a synchronous communication model is a major drawback, so it’s necessary to address this by relaxing the assumption and evaluating performance under asynchronous conditions. Second, the baseline selection lacks clear justification, and the fairness of the comparison is questionable, given that baselines operated under sub-optimal conditions.

### Strengths
1. The paper proposes RPEL, representing the first scheme in a decentralized, serverless setting to simultaneously achieve communication efficiency and robustness against Byzantine attacks. This innovation in mitigating high communication costs is paramount for large-scale decentralized learning.
2. The theoretical analysis is highly rigorous, providing convergence guarantees under a general non-convex and data heterogeneous setting. Notably, it proves robustness against omniscient attacks, a threat model that is significantly more powerful than those typically assumed.

### Weaknesses
--The experimental evaluation lacks a clear justification for its specific choice of baselines. While the paper compares RPEL against CS+, ClippedGossip, and GTS in Appendix C.2, it fails to explicitly state the rationale for selecting these methods over others. Furthermore, the other relevant methods discussed in the related work, such as Remove-Then-Clip (RTC), were notably omitted from the comparison without any explanation, which leaves the comparative analysis feeling incomplete. It’s also questionable whether the comparison is fair, given that the baseline methods were operating under sub-optimal conditions.

--The paper should include more discussion on RPEL under asynchronous settings.. The entire theoretical framework, including the convergence and robustness guarantees, presumes that all participating nodes possess a global clock and respond promptly to model requests. This assumption is often untenable in large-scale, geographically dispersed decentralized systems, where asynchronous settings are virtually a necessity. The authors concede this limitation in Appendix D, noting that while the proposed pull-based approach effectively thwarts Byzantine flooding attacks, it remains vulnerable to DoS attacks in a more general asynchronous model. Therefore, the paper should endeavor to relax this stringent assumption and includes further experiments and discussion on the method's performance and robustness under more realistic asynchronous conditions.

### Questions
Please see weaknesses.

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
4

### Summary
This paper introduces a distributed collaborative learning framework called Robust Pull-based Epidemic Learning (RPEL). Instead of exchanging model parameters with all nodes in the network, RPEL adopts a pull-based strategy, where each node retrieves updates from only a small, randomly selected subset of peers. This design dramatically reduces communication overhead while maintaining learning performance. The key contribution lies in the theoretical analysis showing that such random sampling does not compromise robustness with high probability and still guarantees convergence of the overall learning process. The approach is notable because, despite its simplicity, it provides strong theoretical assurances for a randomized communication protocol: in every round, each honest node pulls updates from a random subset of peers. However, while the expected fraction of corrupted neighbors remains the same under random sampling, the actual fraction in any given round can fluctuate significantly, potentially deviating the training from benign behavior. Therefore, establishing worst-case bounds on the effective adversarial fraction is critical to ensuring the robustness of the proposed algorithm. The paper attempts to show these worst-case bounds.

### Strengths
1. It discusses a reasonable problem where there is a need for scalable and robust learning algorithms.
2. The paper attempts to rigorously analyze the algorithms and provides bounds for their claims.
3. The paper is very well written and easy to follow.

### Weaknesses
**Major Weaknesses**
1. The theoretical claims are not carefully analyzed which becomes an extreme downside as the main contribution of the paper is the theoretical guarantees of the proposed algorithm. 

(a) Consider Lemma 4.1, it provides a lower bound for $s$ such that $\frac{\hat{b}}{s + 1} \in O(\frac{b}{n})$ with at least probability $p$. This essentially gives the lower bound on $s$ to limit the effective adversarial fraction by $O(\frac{b}{n})$ for every honest node throughout the training. This lemma becomes trivial if the lower bound is more than $n$ and does not provide meaningful guarantee. 

For parameters in experimental results in Fig. 1, $T = 200, n = 100, b = 10, \mathcal{H} = 90$, if we consider the probability of failure $1-p = 0.2$, the actual lower bound on $s$ comes out to be:
$$s \geq 386,$$
It results in similar lower bounds for other experimental settings as well. 

(b) Similarly, for eq. $7$ in Lemma A.4, the actual lower bound for $s$ is far more than used in the experimental settings for any reasonable probability of failure. As mentioned section 6.1, Eq 7 requires $s$ to be very large and hence $s$ is chosen through simulation which reflects that performed experiments don't enjoy the theoretical guarantees presented in earlier sections. 

2. The notion of robustness in Definition 5.1 is not sufficient. The upper bound on bias $\Sigma ||v_i - \bar{v}_{\mathcal{U}}||^2$ may seem small and constant but it is proportional to $d$ (dimension of vectors). This can still result in very high bias in applications like collaborative learning where the model size is reasonably large. 

The problem of robust mean estimation is very well studied at this point. The optimal bias guarantees are independent of the dimension proposed in [1]. Robust Aggregators like coordinate-wise median and geometric mean fail to give bias independent and are shown to provide robustness even in the centralized federated learning setup [2]. Even the theoretically optimal aggregators proposed in [1] are prone to attacks due to their computational bottleneck in centralized collaborative learning [3]. 

This highlights that aggregators and attacks used in this paper are not state-of-the-art. There has been a significant advancement in the literature of robust mean estimation in last 5-6 years which has been neglected in the analysis. 

[1] I. Diakonikolas, G. Kamath, D. Kane, J. Li, A. Moitra, and A. Stewart, “Robust estimators in high-dimensions without the computational intractability,” in SIAM Journal on Computing, 2019. 

[2] B. Zhu, L. Wang, Q. Pang, S. Wang, J. Jiao, D. Song, and M. I. Jordan, “Byzantine-robust federated learning with optimal statistical rates,” in AISTATS, 2023.

[3] Choudhary, S., Kolluri, A. and Saxena, P., 2024, May. "Attacking byzantine robust aggregation in high dimensions." in IEEE Symposium on Security and Privacy (SP) 2024.

### Questions
1.  Can authors provide the exact values of their effective adversarial fraction along with the probability of failure for the experimental results (using their theoretical results) ?
2. Why is there a uniform sampling in line 12-14 of Algorithm 1?

### Soundness
1

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
3

### Summary
This paper proposes a way to make distributed, collaborative learning, robust to adversarial/byzantine attacks. The idea is that each agent/node can pull updates from a subset s of its neighbors. Among these, there are going to be some expected number of adversaries, but provided there are enough honest nodes in the neighborhood (something captured by a parameter defined as the "effective adversarial threshold") learning can remain robust to adversaries. The paper proposes a randomized algorithm called RPEL for client selection and aggregation, which comes with theoretical guarantees on single iteration expected error reduction and convergence guarantees, and its performance is validated empirically (particularly against all-communication byzantine robust baselines).

### Strengths
The paper is very well presented and motivated. It studies an important and challenging problem. 

The algorithm comes with both theoretical guarantees and empirical support. 

The presented algorithm requires less restrictive bounds on the communication overhead and lower bounds on connectivity with honest nodes than existing methods.

### Weaknesses
Not necessarily weaknesses but clarifications about some points might be helpful: 

1. I understand they are conceptually different, and agree with the description of why it is important to do pulls (not pushes, which would allow for flooding by adversaries) in this paper’s setting. But how about technically/theoretically? Meaning, can you elaborate more, from a technical perspective, why pull-based and push-based lead to differing analysis? 

2. Similarly, I did a brief review of the proofs, and I was finding references to similar proofs in the all-to-all communications setting of Farhadkhani et al., or to the work of Allouah et al. 2024a on robust learning. What are the key technical contributions of this paper? In other words, how does the setup of this paper add to the analytical complexity of establishing error bounds or convergence guarantees? This was not clear to me throughout, and having some clarification of the technical contributions might help. 

3. Equation (7) seems to be key for Theorem 5.6., but it only appears in the appendix, which affected the flow of the paper, and including the condition in the main paper would be helpful. 

4. While not crucial/important, I was curious about the code, and the anonymized link was appreciated, but other than the readme file, I encountered the error “The requested file is not found.” for other files. 

5. Minor edit suggestion: I believe the sentence in 248-249 should read the other way around. Meaning: "We call the ratio \frac{b}{s+1} the Effective adversarial fraction."

### Questions
While this paper was overall clear and interesting to me, I would appreciate clarification on the following two points during the rebuttal period, to help better inform my final assessment. 

1. What are the technical differences between push-based and pull-based epidemics? (more details above in the weakness section). 

2. What are the technical contributions/differences of this paper's theoretical results on convergence or per-round error reduction compared to existing works such as Farhadkhani et al. and Allouah et al., which are referenced often in the appendix, too? (more details above in the weakness section).

### Soundness
3

### Presentation
4

### Contribution
3
