# Differential Privacy of Cross-Attention with Provable Guarantee

- Decision: Reject
- Scores: 6, 6, 6, 5

## Abstract
Cross-attention has become a fundamental module nowadays in many important artificial intelligence applications, e.g.,
retrieval-augmented generation (RAG), system prompt, guided stable diffusion, and many more. 
Ensuring cross-attention privacy is crucial and urgently needed because its key and value matrices may contain sensitive information about model providers and their users.
In this work, we design a novel differential privacy (DP) data structure to address the privacy security of cross-attention with a theoretical guarantee.
In detail, let $n$ be the input token length of system prompt/RAG data, $d$ be the feature dimension, 
$R$ be the maximum value of the query and key matrices, $R_w$ be the maximum value of the value matrix, and $r,s,\epsilon_s$ be parameters of polynomial kernel methods. 
Then, our data structure requires $\widetilde{O}(ndr^2)$ memory consumption with $\widetilde{O}(ndr^2)$ initialization time complexity and $\widetilde{O}(d r^2)$ query time complexity for a single token query.
In addition, our data structure can guarantee that the process of answering user query satisfies $(\epsilon, \delta)$-DP with 
$\widetilde{O}((1-\epsilon_s)^{-1} n^{-1} \epsilon^{-1} R^{2s} R_w r^2)$ additive error and $2\epsilon_s/(1-\epsilon_s)$ relative errorbetween our output and the true answer.
Furthermore, our result is robust to adaptive queries in which users can intentionally attack the cross-attention system. 
To our knowledge, this is the first work to provide DP for cross-attention and is promising to inspire more privacy algorithm design in large generative models (LGMs).

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies the problem of privately computing cross-attention. The authors propose a data structure that can answer cross-attention queries within certain relative and absolute errors while satisfying $(\varepsilon,\delta)$-DP. The main idea is to approximate the exponential inner product using a polynomial kernel function, which converts the original problem into one of answering distance queries.

### Strengths
The paper is the first to propose a DP algorithm for cross-attention computation. The algorithm is both time- and space-efficient for initialization and can quickly answer adaptive user queries. The idea of approximating the exponential inner product by a polynomial kernel function (Lemma I.5) and the distance query algorithms (Algorithms 5 and 6) are interesting.

### Weaknesses
1. It is unclear what information is protected in this paper.  It is necessary to clearly state which parts of the input are private ($K$ and $V$?) and which parts of the output are public ($Q$?). Moreover, since each row of $K$ is derived from one word in the input, the proposed algorithm seems to only protect word-level privacy.
2. The paper does not contain any experiments, making it difficult to evaluate whether the obtained errors fall within a reasonable range. Since the problem arises from a real-world application, it is strongly recommended to provide some illustrative experiments, such as standard NLP tasks on real-world datasets.
3. The proposed algorithm has a relative error, which could be unsatisfactory. In Section 5, it is mentioned that the parameter $\alpha$ can be removed. Does this imply that the algorithm can be modified to involve only additive errors, or does the $n^{-1}\varepsilon_{S}$ term still exist?

### Questions
1. The definition of differential privacy in the article is unclear. It is defined for datasets represented by vectors in $\mathbb{R}^n$. However, in cross-attention, the key and value matrices have dimensions of $n\times d$. I recommend clarifying the definition so that it fits the cross-attention setting.
2. In line 269, it is stated that "D is only a normalization factor and does not contain sensitive information". However, the normalization factors depend on $A$, which is calculated from $K$ and $V$. Thus, $D$ contains private information and should be protected. I do not understand why we can directly discard $D$ in this context.
3. In Algorithm 3, the variables $P_{wx}$ and $s_{wx}$ are used without adding any noise. Can you explain why the resulting algorithm still preserves differential privacy?
4. Has the problem of private distance queries been studied in the literature? If so, it would be helpful to include some related work.
5. The initialization time complexity and query time complexity in the abstract do not include $d$. Is this a typo?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a differentially private algorithm for implementing Cross Attention. It reformulates Cross Attention as a private summation problem and employs a classic binary tree algorithm to ensure privacy. The paper provides a comprehensive analysis of the proposed algorithm, though it lacks empirical evaluation.

### Strengths
1. The novel idea of reformulating Cross Attention as a weighted distance problem allows the use of classic differential privacy algorithms, such as the binary tree mechanism, as building blocks to ensure the privacy of Cross Attention.

2. A comprehensive analysis of the algorithm is provided, covering aspects such as utility, computational complexity, and privacy properties. I believe this work provides a solid foundation for developing differentially private mechanisms for core components in large language models.

3. Figure 1 in the appendix is very helpful for understanding the connection between cross attention and the dp binary tree.

### Weaknesses
1. The comparison with DP-SGD is limited. Currently, DP-SGD is widely used in various fine-tuning tasks and can be relatively easily integrated into existing systems. In contrast, the proposed DP Cross-Attention requires complex algorithmic design, making adaptation to existing systems challenging. It requires significant modifications, which reduces its accessibility.

2. For linear queries, the matrix mechanism [1] generally outperforms the binary tree. I would be interested in hearing more discussion on the matrix mechanism. Could you also provide an estimate of the width of the binary tree (i.e., the number of leaves) in your algorithm?

[1] Li, Chao, Gerome Miklau, Michael Hay, Andrew McGregor, and Vibhor Rastogi. "The matrix mechanism: optimizing linear counting queries under differential privacy." The VLDB journal 24 (2015): 757-781.

### Questions
In Theorem 1.2, you mentioned that with probability 1 - $p_f$, the mechanism is $(\epsilon, \delta)$-DP. It gives me a feeling that the mechanism is $(\epsilon, \delta + p_f) $- DP. What's the value of $p_f$ in practice? Given a fixed $\epsilon$, the larger is $ \delta + p_f$ , the less private is the mechanism.

### Soundness
3

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
2

### Summary
This paper introduces a modified cross attention algorithm with differential privacy guarantees. The authors modify the cross-attention algorithm by approximating the exponential inner product which allows them to cast the modified algorithm into a weighted distance problem. The paper introduces a several novel tree based data structures to efficiently compute a private the private weighted distance. They produce detailed proofs of privacy guarantees as well as error analysis.

### Strengths
- Given that cross-attention modules are fundamental in numerous applications involving sensitive data such as RAG, this work addresses an important gap. The transformation of cross-attention computations into a "weighted distance problem" is a creative solution, making it amenable to DP techniques in a novel way.
- The paper does a good job of grounding its methodology in formal mathematical guarantees. By presenting detailed proofs of differential privacy for the cross-attention mechanism as well as an error and complexity analysis, the authors provide a robust theoretical backing.

### Weaknesses
- The appeal of cross attention is its superior empirical performance. The paper adapts the algorithm in several ways without evidence how the algorithm's performance is changed.
- While I understand that the contributions of this paper are of theoretical nature, the motivation of the paper is the popularity of the cross-attention mechanism which is due to its practicality. As such, an empirical evaluation of the utility-privacy trade-off would strengthen the paper especially considering the main audience of this conference. However, the paper is already in the current state fairly dense and I'm not sure whether adding even more content will help the readability in its current format.

### Questions
- How do you guarantee that $K_{i,j} \in [0,R]$ and $V_{i,j} \in [-R,R]$? Is this done by clipping? If so, what effect does that have on the performance of the algorithm?
- What is the scope of the adjacency relation? Is it leaving out individual tokens? If so, in practice secrets may reach across many tokens and the DP guarantees would not provide meaningful privacy.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper introduces a novel differential privacy (DP) mechanism, DPTREE, for preserving privacy in cross-attention mechanisms in large generative models (LGMs). The structure provides privacy guarantees while supporting adaptive queries with theoretically bounded errors and optimized memory and computation costs.

### Strengths
1.Novel Application of DP in Cross-Attention: The paper proposes the first DP solution specifically for cross-attention, a fundamental mechanism in LGMs, which is innovative and addresses privacy in an underexplored area.
2.heoretical Rigor: The paper is backed by solid theoretical proofs of privacy and accuracy guarantees, adding depth and credibility to the approach.
3.Adaptive Query Handling: The mechanism is robust against adaptive queries, making it suitable for real-world applications where inputs can vary significantly.
4.Memory and Computation Efficiency: Through the DPTREE structure, the proposed method optimizes memory usage and computational complexity, enhancing scalability.

### Weaknesses
1.Lack of Innovation in Core Techniques: The DPTREE mechanism heavily relies on polynomial kernel approximations and basic DP noise mechanisms, which are established techniques in DP literature. The limited novelty in these technical foundations weakens the originality of the proposed approach.
2.Limited Generalization and Practicality: DPTREE is specifically tailored to cross-attention, and its reliance on polynomial kernel approximations may not generalize well across other types of attention mechanisms or data structures in LGMs. This specificity reduces the model's adaptability, which could hinder broader applicability and limit its impact in the field.
3.Computational and Initialization Costs: Although optimized for memory during inference, the initialization and computational costs for high-dimensional data remain high. This could make the solution impractical for larger LGMs or applications requiring frequent real-time updates.

### Questions
1.Given the DPTREE's reliance on polynomial kernel approximations and its focus on cross-attention, how well does it generalize to other attention mechanisms or model architectures, especially in high-dimensional or heterogeneous data scenarios?
2.What strategies does the DPTREE mechanism have in place to manage potential approximation errors that may accumulate during adaptive querying?
3.Are there specific types of datasets or applications where DPTREE demonstrates significantly better or worse performance?

### Soundness
3

### Presentation
3

### Contribution
3
