# Estimating Markov Chain Transition Probabilities for Steady Aging Models from $n$-step Data

- Decision: Reject
- Scores: 4, 0, 4, 8

## Abstract
Usage-dependent aging processes of products such as batteries have become increasingly important. However, in general, revealing aging dynamics of discrete time Markov Decision Processes (MDP) from sparse $n$-step data is challenging as multi-step transition probabilities become complex and make it impossible to efficiently optimize the likelihood function for larger $n$. In this paper, we consider classes of steady-aging processes, which can be characterized by band matrices. Based on novel explicit diagonalizations of such matrices, we are able to optimally solve the likelihood in an efficient manner. In contrast to existing benchmark approaches our approach scales well and remains applicable for large $n$. Further, our evaluations for synthetic as well as real-world data verify a high estimation accuracy (about 98\%) with comparably few data (about 10$\times$ more observations as states).

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
1

### Summary
The paper aims to estimate aging dynamics of discrete-time Markov chains (DTMCs) from sparse n-step data, which is often intractable to optimize the likelihood function for large n.
The paper only considers a class of stead-aging processes captured by band matrices.
The paper is able to efficiently solve the likelihood through explicit diagonalization of band-structured transition matrices.
The proposed method scales well, and is applicable for large n.
The experiments on both synthetic and real data, demonstrate its high estimation accuracy.

Overall,  the targeted problems, the proposed solution and the new technical contributions are easy to follow. 
The targeted problems might be only interested for a relatively small community in ICLR. Although the proposed diagonalization technique of band-structured transition matrices seems to be new, the authors fails to provide theoretical guarantees like approximation error analysis for such methods which could be more important for the others to use this approach. 
Based on these concerns, I think this submissions is not ready to be published, and thus gave the weak rejection.

### Strengths
- Overall, the targeted problems, the proposed solution and the new technical contributions are easy to follow.

- It might be novel to consider the diagonalization technique of band-structured transition matrices for aging processes, while i am not sure how novel is such math technology for the other communities like probabilities and applied mathematics. The authors fail to clarify this in the submission.

### Weaknesses
- Notation clarity: The authors should improve the Notation clarity. For instance, in Eqs(1) and (3), n degree power is different!

- The studied problems might be important for industries and a small community in ICLR. I'm not sure how important the studied problems to ICLR audiences. It may be not that important for such conference.

### Questions
- Except the experimental results, can you provide any theoretical guarantees such as approximation error analysis, for the proposed diagonal technique of band matrices?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
3

### Summary
This work suggests the use of an explicit diagonalisation formula for banded (stochastic) matrices for efficiently calculating the likelihood of certain discrete-time Markov models.

### Strengths
Improving the estimation of models for battery lifetimes certainly seems important for moving away from fossil fuels.

### Weaknesses
**Clarity:**

I find this paper quite difficult to read due to the models not being very clearly defined. This starts with Definition 1 where I am puzzled by the terms "pre-state" and "post-state". The same issue affects Definition 2. From the later equations, I think I understand the meaning but I don't think this is standard terminology. More generally, in the introduction the manuscript would benefit from a more precise and formal descriptions of what the claimed contributions are, e.g.:
1. What precisely is the computational complexity of existing methods?
2. What precisely is the computational complexity of the newly proposed method?
3. What precisely is the novelty (is Theorem 1 novel or only its application in this context)?

And the entire class of models to be treated should be stated first before discussing the computational aspects involved in the estimation. The manuscript explains the models by stating what the shape of the observations is and then assumes that the reader infers the model from this information. This is a particular problem in the case of the "generalisation" stated on Page 5.

**Novelty:**

I am no expert in discrete-time--discrete-state Markov chain models. However, it would surprise me if the decomposition from Theorem 1 (and its subsequent use for calculating $n$-step transition probabilities) was novel. That said, from reading the paper, it is not clear to me if the authors would claim that this result is novel or merely that its use in the specific ageing-model estimation context is novel.

**Related approaches:**

There are some alternative modelling approaches such as semi-Markov models / discrete-time change-point models as well as continuous-time Markov chains. If the main claimed contribution of the present paper is the more efficient calculation of the likelihood, then It might not be necessary go discuss these. However, if the main claimed contribution is in the modelling, then it would be good if the authors could compare with such approaches.

The paper also entirely misses any discussion of hidden Markov and semi-hidden Markov models (or their continuous state or continuous time analogues such as state-space models). These are widely used in the literature on remaining-useful-life prediction including battery-life estimation. In settings in which the states are only partially or noisily observed, such models seem much more principled than the "generalisation" proposed by the authors on Page 5 (for which they also do not seem to give real-world examples).

### Questions
1. Since the work focuses on "bi-diagonal" transition matrices, can we not simply and cheaply evaluate the likelihood using geometric-distribution probabilities of the kind mentioned at the bottom of Page 4 (Line 215)? Can you prove that the diagonalisation-based likelihood evaluation has lower complexity?

2. Is Theorem 1 claimed to be novel?

3. Wouldn't it make more sense to fit continuous-time Markov chains here if the regime switches occur only rarely?

### Soundness
3

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper focuses on an understanding of product aging and the modelling of corresponding processes. The idea is to reconstruct the aging trajectory from a small number observations while not relying on domain expertise. The aging process itself is modelled as a Markov chain with a particular transition matrix P intended to capture the aging dynamics. To facilitate estimation of P based on observed data, the authors introduce a procedure that is based on a diagonalization of P in an explicit way. The proposed methodology is tested on several examples and is shown to perform well.

### Strengths
The analytical, or at least partially analytical, approach is welcomed. The resulting scalability is likely to make the method useful in at least some scenarios that are encountered in practice. The experiments are well-chosen and illustrate the results of the paper.

### Weaknesses
I would say that the flow of the paper is at time unclear and the main contribution is not as clearly stated as it should be for such a conference paper. The range of applications of such methodology, beyond a very specific task of estimating certain aging models, is not clear. The authors should carefully try to see where other such scenarios can be encountered.

### Questions
How does the model behave if the number of observations is really large? To what extent can the method be extended to other structures, beyond the band matrix?

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
2

### Summary
The paper addresses the challenge of estimating complex usage-dependent aging processes, such as battery degradation, from sparse multi-step (`n`-step) transition data using standard Markov Chain methods. These methods become computationally infeasible as `n` grows large. The authors propose a novel approach for "steady-aging" processes modeled by discrete-time Markov Chains with 
band matrices (specifically bi-diagonal). The key innovation is the explicit diagonalization of the band transition matrix, enabling efficient and exact computation of multi-step probabilities (`n`-step) and allowing optimal likelihood function optimization even for large `n`.

### Strengths
**Efficiency gains over standard approaches** 

**Evaluation uses both synthetic data (controlled environment) and real-world battery data (practical relevance).**

**High accuracy (~98% estimation accuracy).**

**Data efficiency ("comparably few data" ~10x observations per state).**
 
**Scalability (`n >= 1000` periods solved in seconds). This directly supports the key claim of handling large `n`.**

**Limitations and scope are clearly presented**

### Weaknesses
The text does not provide enough details about the exact nature and size of real-world datasets and detailed experimental setup parameters beyond `n` sizes (though Appendix A is mentioned).

The claim "near-optimal prediction results" is not explicitly quantified or compared to a theoretical optimum in the provided text.

### Questions
Can you be more precise about the  claim on "near-optimal prediction results" ?

### Soundness
3

### Presentation
4

### Contribution
4
