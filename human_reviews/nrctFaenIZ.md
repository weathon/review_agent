# GradSkip: Communication-Accelerated Local Gradient Methods with Better Computational Complexity

- Decision: Reject
- Scores: 8, 5, 5, 5

## Abstract
We study a class of distributed optimization algorithms that aim to alleviate high communication costs by allowing clients to perform multiple local gradient-type training steps prior to communication. While methods of this type have been studied for about a decade, the empirically observed acceleration properties of local training have eluded all attempts at theoretical understanding. In a recent breakthrough, Mishchenko et al. (ICML 2022) proved that local training, when properly executed, leads to provable communication acceleration, and this holds in the strongly convex regime without relying on any data similarity assumptions. However, their ProxSkip method requires all clients to take the same number of local training steps in each communication round. Inspired by a common sense intuition, we start our investigation by conjecturing that clients with ``less important'' data should be able to get away with fewer local training steps without this impacting the overall communication complexity of the method. It turns out that this intuition is correct: we managed to redesign the original ProxSkip method to achieve this. In particular, we prove that our modified method, for which we coined the name GradSkip, converges linearly under the same assumptions and has the same accelerated communication complexity, while the number of local gradient steps can be reduced relative to a local condition number. We further generalize our method by extending the randomness of probabilistic alternations to arbitrary unbiased compression operators and by considering a generic proximable regularizer. This generalization, which we call GradSkip+, recovers several related methods in the literature as special cases. Finally, we present an empirical study on carefully designed toy problems that confirm our theoretical claims.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper addresses an intriguing issue in the domain of distributed optimization algorithms. Conventionally, in these algorithms, clients need to have periodic communication and each client performs an equal number of local training steps per communication round. The authors question this norm, pointing out that some clients might face more complex data or difficult problems, potentially necessitating more local training. 

The paper introduces a novel algorithm, GradSkip, which realizes this intuition. The authors also provides a clear mathematical analysis and proof. The paper demonstrates that the number of local gradient steps can be reduced relative to the local condition number without undermining the communication complexity. Furthermore, the paper extends its discussion to include other scenarios like variance reduction and gradient compression, leading to the development of GradSkip+.

### Strengths
1. The paper uncovers a notable conclusion that clients with simpler data or problems might require fewer local training steps, a concept not widely addressed in current literature.

2. The authors support their findings with stringent and well-articulated mathematical proofs, enhancing the credibility and academic rigor of their work.

3. The analysis provided is detailed and easy to follow, making the complex concepts accessible to readers.

4. Introduction of unbiased compression operators is a significant technical innovation. This concept broadens the scope for a range of new algorithms, marking a substantial contribution to the field.

5. The paper succeeds in providing a comprehensive framework that not only encompasses many known algorithms (ProxGD, ProxSkip, RandProx-FB) but also suggests the potential for several unknown algorithms through its unbiased compression operator.

### Weaknesses
One minor critique is that the paper's theoretical bounds are not tight in constant terms.

### Questions
Even though I acknowledges the theoretical contributions of this work, I have a question regarding its practical relevance. Specifically, how severe the issue of statistical heterogeneity is in machine learning? How large is the divergence of curvatures among clients? This question is related to the significance of GradSkip algorithm (and potentially any following works) in real-world scenarios.

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work proposes a new local gradient-type method for distributed optimization with communication and computation constraints. The proposed method inherits the same accelerated communication complexity from ProxSkip while further improving computational complexity. And two variants of the proposed method, i.e., GradSkip+ and VR-GradSkip+ are proposed.

### Strengths
1. A new local gradient-type method for distributed optimization with communication and computation constraints is proposed in this work, which is the extension of the ProxSkip method. The proposed method inherits the same accelerated communication complexity from ProxSkip while further improving computational complexity.

2. And two variants of the proposed method, i.e., GradSkip+ and VR-GradSkip+ are proposed.

### Weaknesses
1. The assumption that functions $f_i(x)$ are strongly convex is too strong since many functions will not satisfy this assumption when utilizing neural networks.

2. Lack of theoretical analysis of the communication complexity of the proposed method. In distributed optimization, communication complexity is crucial for minimizing inter-node communication to enhance system efficiency and reduce communication costs.

3. The experimental results are limited, the authors should conduct more experiments to verify the performance of the proposed method.

4. The writing of this work is poor. I can't find the Conclusion section. And the summary of contributions is excessively lengthy.

5. There are lots of mistakes in this work, for example, 

``Appendix ??'',

 ``see Algorithm ?? in the Appendix'', 

`` (see Appendix)''

### Questions
Please see the weakness above.

### Soundness
1 poor

### Presentation
1 poor

### Contribution
1 poor

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes Gradskip for solving federated optimization problems with smooth strongly convex objective. Gradskip improves local gradient computation complexity and achieves the optimal communication complexity. The paper further extends the idea of Gradskip to propose Gradskip+ and VR-Gradskip+, which covers a wider range of application.

### Strengths
1. The proposed Gradskip method and its extensions modify Scaffnew by allowing skipping local gradient computation and improve the local gradient computation complexity to $O(\min(\sqrt{\kappa_{\max}},\kappa_i)\log(1/\epsilon))$ from $O(\sqrt{\kappa_{\max}}\log(1/\epsilon))$, while still achieving the optimal communication complexity $\sqrt{\kappa}\log(1/\epsilon)$. I suggest the authors summarize their results and existing work in table.
2. Allowing skipping gradient computation is helpful to address system heterogeneity as slow clients can compute less in a communication round.

### Weaknesses
1. The novelty of this paper looks somewhat limited. The novelty and main contribution is that Gradskip doesn't always compute local gradient and thus requires $O(\min(\sqrt{\kappa_{\max}},\kappa_i)\log(1/\epsilon))$ proposes Gradskip, instead of $O(\sqrt{\kappa_{\max}}\log(1/\epsilon))$. However, the framework and analysis of proposed Gradskip is similar to Scaffnew.
2. The improvement on computational cost heavily depends on the values of $q_i$, which rely on $\kappa_i$. However, Remark 3.3 says GradSkip addresses heterogeneity by assigning $q_i$ to clients in accordance with their local computational resources. It is unclear how to connect $\kappa_i$ to the local computational resources.
3. Can Gradskip also make improvement on computation time over Scaffnew? What is the time cost for computing gradient in each iteration?

### Questions
see the section of weaknesses

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
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
Built upon ProxSkip, authors proposed GradSkip (and variants GradSkip+) algorithms by incorporating new randomness with each client. The proposed algorithm attains better computation complexity compared to existing works.

### Strengths
1. The key novelty lies in the newly introduced client-wise randomness, which induces fake local steps and less local steps (Lemma 3.1 and 3.2), the idea is elegant.
2. Better computation complexity.

### Weaknesses
1. Compared to ProxSkip (Mishchenko et al. (2022)), the algorithm here requires finer structure information from the devices, i.e., individualized function smoothness parameters, while ProxSkip only requires a global smoothness parameter. And all clients are required to coordinate in advance to know the global information $\kappa_{\max}$, which may be a bit unrealistic.
2. According to Theorem 3.6, the client gradient query number is improved from $\sqrt{\kappa_{\max}}$ to $\min(\kappa_i, \sqrt{\kappa_{\max}})$, while the iteration and communication complexity does not change, the claimed $O(n)$ superiority only appears in scenarios where the devices are very unbalanced (most of them have small $\kappa$, while few of them attain very large $\kappa$. As mentioned in your experiments, only one ill-conditioned device), I may view such scenarios to be relatively rare in real world (or it is better if authors can rationalize it). If so the derived improvement seems to be a little bit weak.
3. As far as I understand, the proof heavily relies on the proof of ProxSkip, which restricts the significance of the contribution a bit.

To summarize, I think the algorithm is an interesting extension of ProxSkip with an elegant modification, while I concern that the improvement may be a bit marginal to cross the bar. Please definitely indicate if I misunderstood any points. Thank you very much for your efforts.

### Questions
1. In Assumption 3.4, why not extend each $f_i$ to attain a personalized strong convexity parameter $\mu_i$? I think it should be expected.
2. As a separate question, compared to communication complexity, whether improving individual computation complexity is an important question to the FL community, I expect that such improvement should be attractive to marginalized devices.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
