# Distributed Algorithms for Euclidean Clustering

- Avg Score: 7.00
- Decision: Accept (Poster)
- Scores: 8, 6, 6, 8

## Abstract
We study the problem of constructing $(1+\varepsilon)$-coresets for Euclidean $(k,z)$-clustering in the distributed setting, where $n$ data points are partitioned across $s$ sites. We focus on two prominent communication models: the coordinator model and the blackboard model. In the coordinator model, we design a protocol that achieves a $(1+\varepsilon)$-strong coreset with total communication complexity $\tilde{O}\left(sk + \frac{dk}{\min(\varepsilon^4,\varepsilon^{2+z})} + dk\log(n\Delta)\right)$ bits, improving upon prior work (Chen et al., NeurIPS 2016) by eliminating the need to communicate explicit point coordinates in-the-clear across all servers. In the blackboard model, we further reduce the communication complexity to $\tilde{O}\left(s\log(n\Delta) + dk\log(n\Delta) + \frac{dk}{\min(\varepsilon^4,\varepsilon^{2+z})}\right)$ bits, achieving better bounds than previous approaches while upgrading from constant-factor to $(1+\varepsilon)$-approximation guarantees. Our techniques combine new strategies for constant-factor approximation with efficient coreset constructions and compact encoding schemes, leading to optimal protocols that match both the communication costs of the best-known offline coreset constructions and existing lower bounds (Chen et al., NeurIPS 2016, Huang et. al., STOC 2024), up to polylogarithmic factors.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
1

### Summary
The paper addresses the problem of efficiently constructing $(1+\varepsilon)$-coresets for Euclidean (k,z)-clustering in a distributed environment, focusing on two prominent and practically relevant communication models: the coordinator and the blackboard models. The authors propose novel distributed protocols that provide strong $(1+\varepsilon)$-approximation guarantees while simultaneously significantly reducing the overall communication complexity compared to prior best known results.

### Strengths
- The protocols presented almost match the lower bounds ie up to polylogarithmic for distributed $(1+\varepsilon)$ coreset construction for Euclidean (k,z)-clustering in both the coordinator and blackboard models. Further, 

- The work introduces compact encoding schemes that enable coordinators/servers to exchange essential summary information rather than raw point coordinates.

### Weaknesses
- The protocols likely introduce implementation complexity relating to encoding/decoding, and requirements for synchronization among nodes.

### Questions
It will be helpful for readers if the authors can elaborate on the computational overhead incurred from the compact encoding and decoding schemes.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies the construction of coresets for Euclidean $(k,z)$-clustering in distributed settings, where the $n$ input points are partitioned among $s$ sites. Two models are considered: the coordinator model, where all sites communicate only with a central coordinator over private channels using private randomness, and the blackboard model, where each site can write on a public blackboard visible to all, but still uses private randomness.

In the coordinator model, the authors design a $(1+\epsilon)$-coreset using roughly $O(sk + dk + dk \log(n\Delta))$ bits of communication (hiding dependencies on $z$ and $\epsilon$). In the blackboard model, they further reduce the communication complexity to $O(s \log(n\Delta) + dk \log(n\Delta))$ bits.

To achieve these results, the paper first presents a randomized bicriteria algorithm that uses only $O(s \log n + kd \log n)$ bits of communication. This is achieved via a lazy sampling procedure based on the adaptive sampling technique from prior literature. The key insight is to sample from a distribution close to the adaptive one, but requiring fewer communication bits. The resulting bicriteria solution is then used to construct the coresets through sensitivity sampling. Finally, the authors present experimental results supporting their theoretical claims. However, the paper lacks technical proofs, making it difficult to verify the correctness of the results (though they appear plausible).

### Strengths
1. The results are interesting and elegant — unlike prior approaches that required a communication term of $sdk \cdot \log(n\Delta)$ (due to each site transmitting point coordinates), the proposed approach eliminates this per-site dependence.

2. The techniques appear to involve clever ideas and nontrivial technical depth.

### Weaknesses
1. The main concern is presentation quality. The paper does not clearly explain the technical innovations. Moreover, the absence of proofs prevents the reader from understanding or validating the key arguments. This issue is aggravated by theorem statements (e.g., Lemmas 2.1 and 2.2) referring to algorithms that are not actually described in the main text. While the results seem promising, the lack of detail makes it impossible to assess the technical contributions on their merit.

2. Related to the above, several algorithmic components are mentioned (e.g., in Lemmas 2.1 and 2.2) without any accompanying description or pseudocode, making it hard to follow the logic or reconstruct the methodology.

### Questions
1. Could the authors elaborate on the main technical ideas that enable the reduction in communication complexity?

2. What is the size of the constructed coresets? Is there a particular reason why the coreset size is not explicitly stated in the technical results?

### Soundness
1

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the problem of constructing coresets for Euclidean (k,z)-clustering in distributed settings, where n data points are partitioned across s sites. The authors study two communication models: the coordinator model and the blackboard model. For the coordinator model, they present a protocol that achieves a strong coreset with total communication complexity $\tilde{O}(sk+dk/\min${$\epsilon^4, \epsilon^{2+z}$}$+dklog(n\Delta))$ bits, which improves upon prior work (Chen et al., 2016) by avoiding directly to transmit all coordinates. In the blackboard model, they further achieve better bounds of communication cost than previous algorithms. The main contribution of this work is to provide the compact encoding schemes, and sampling operations, and efficient algorithm in message-passing.

### Strengths
1.The paper improves the communication complexity for distributed (k,z)-clustering coresets in both the coordinator and blackboard models as illustrated in Figure 1, eliminating the need to send raw coordinates.

2.The paper presents a solid theoretical contribution to the study of distributed coreset construction for Euclidean (k,z)-clustering. 

3.The paper is well-written and well-structured, making the technical content accessible.

4.The combination of new coreset constructions, compact encoding strategies, and communication-efficient protocols results in practical algorithms that are easy to implement in real-world distributed systems.

### Weaknesses
1.While the paper proposes distributed algorithms for both the coordinator and blackboard models, the experimental evaluation only includes results for the blackboard model. No empirical comparison or validation is provided for the coordinator-based algorithm. 

2.There is no evaluation of how the algorithm scales with the number of distributed machines, which is critical for understanding its practical applicability in real-world distributed systems.

3.The paper provides only communication complexity bounds, without reporting the actual running time and memory on the coordinator or client machines. 

4.The paper lacks a clear and formal definition of clustering under general topologies, which is briefly mentioned but not elaborated upon.

### Questions
1.The authors should provide experimental comparisons with existing algorithms in the coordinator model to clearly demonstrate the advantages of their approach.

2.Could the authors include more details and comparative results on how the algorithm performs with varying numbers of machines, in order to assess its scalability in distributed settings?

3.In practical distributed applications, how should the parameters k and s be selected to balance clustering quality and communication efficiency in both the coordinator and blackboard models?

4.Regarding the blackboard model, is the per-machine computational complexity small enough to allow application on resource-constrained edge devices?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper constructs coresets for the $(k,z)$ clustering problem in distributed setting for coordinator model where a coordinator facilitates the communication between nodes in rounds and also for the blackboard model where communication is using a shared 'blackboard'. For both models, by using the coresets, the paper improves the existing communication complexity for the $(k,z)$ clustering problem. The paper also demonstrates the effectiveness of the coresets with empirical evaluations on real and synthetic data.

### Strengths
1) The problem is an interesting one and will be of interest to the community.
2) To the best I could check, the claims in the paper appear sound. The paper is overall written well. 
3) The paper improves on existing communication complexity bounds for both the coordinator and blackboard models.
4) The way, the usual tools and tricks of coreset literature like JL transform, bicriteria approximation, sensitivity sampling etc, have been modified to fit the requirements of the distributed setting will be of interest to the coreset community.

### Weaknesses
The two minor weaknesses that I see in the paper are:
1) The structure of the paper makes it a little hard to parse. Also, many ideas like JL transform, bicriteria approximation, sensitivity sampling which are well known are used in a clever way to get the results. However, that technical novelty and challenges are not sufficiently evident from the main body of the paper. The authors should try to highlight them. It may be a good idea to bring the discussion on why the usual techniques are not directly applicable from appendix to main body and also why coordinate wise sampling is required can be elaborated.

2) The experiments are of a proof-of-concept nature and not extensive. Detailed experiments and comparisons on some more datasets and different settings will strengthen the paper.

### Questions
See Weaknesses

### Soundness
3

### Presentation
2

### Contribution
2
