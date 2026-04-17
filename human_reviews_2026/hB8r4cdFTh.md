# Cohort Squeeze: Beyond a Single Communication Round per Cohort in Cross-Device Federated Learning

- Decision: Reject
- Scores: 2, 6, 4

## Abstract
Virtually all federated learning (FL) methods, including FedAvg, operate in the following manner: i) an orchestrating server sends the current model parameters to a cohort of clients selected via certain rule, ii) these clients then independently perform a local training procedure (e.g., via SGD or Adam) using their own training data, and iii) the resulting models are shipped to the server for aggregation. This process is repeated until a model of suitable quality is found. A notable feature of these methods is that each cohort is involved in a single communication round with the server only. In this work we challenge this algorithmic design primitive and investigate whether it is possible to ``squeeze more juice" out of each cohort than what is possible in a single communication round. Surprisingly, we find that this is indeed the case, and our approach leads to up to 74% reduction in the total communication cost needed to train a FL model in the cross-device setting. Our method is based on a novel variant of the stochastic proximal point method (SPPM-AS) which supports a large collection of client sampling procedures some of which lead to further gains when compared to classical client selection approaches.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
Most FL methods (e.g., FedAvg) use a single communication round per client cohort: the server sends the model, clients train locally, and updates are aggregated. This work challenges that design, showing that allowing multiple interactions per cohort can significantly improve efficiency. Using a new stochastic proximal point variant (SPPM-AS) that supports diverse client sampling strategies, the approach achieves up to 74% reduction in total communication cost in cross-device federated learning.

### Strengths
* The interaction of three main components (proximal methods, same cohort involved in multiple local rounds, sampling strategy) seems promising.

* The authors provide an extensive theoretical framework to support their work.

### Weaknesses
* Readability and Structure.
The paper is difficult to follow due to poor organization. It lacks a dedicated background and related work section, and the method section does not provide a high-level overview or intuitive explanation of the proposed approach. As a result, the reader struggles to grasp the motivation and flow of ideas.

* Clarity of Contributions.
The technical contributions are not clearly articulated. The paper seems to combine two main ideas, (i) the introduction of SPPM variants, and (ii) the use of multiple local communication rounds per selected cohort to reduce overall communication, but their relationship is not clearly explained. It remains unclear how these two elements interact or reinforce each other.

* Novelty and Positioning.
The novelty related to the proximal point optimization framework is not well established. The authors should better distinguish what is new in their method versus what is already known from existing proximal point or stochastic optimization literature.

* Practical Feasibility.
While the empirical observation that reusing the same client cohort across multiple local rounds reduces communication is interesting, its practical applicability is questionable in realistic cross-device FL settings, where maintaining the same cohort across rounds is rarely guaranteed. The paper should discuss this limitation or provide strategies to address it.

### Questions
1. Clarity and Structure.
One of the main problems of the paper is its clarity.
(a) A dedicated background section is needed.
(b) The paper assumes readers are already familiar with proximal point methods. A concise background explaining their principles and how they are used in centralized and distributed machine learning would greatly improve accessibility.
(c) The method section lacks a high-level or intuitive explanation. Please include a short intuitive summary (one or two sentences) of how the proposed method works, and consider adding an overview figure or a detailed algorithmic diagram illustrating the method in a federated setting, with more context than the current Algorithm 1.

2. Novelty of the Contribution.
What is the novel contribution of the work with respect to existing proximal point methods? Please clarify which aspects of your approach are original and how they differ from known stochastic or distributed proximal point variants.

3. Related Work.
The paper currently lacks a related work section. Please discuss prior work on proximal point methods and communication-efficient federated learning, and explicitly position your contribution relative to these studies.

4. Practical Feasibility of Cohort Reuse.
The paper claims that increasing the number of local communication rounds within the selected cohort reduces overall communication. While this may hold empirically, how can one guarantee the availability of the same cohort of clients in realistic cross-device environments? Please clarify the assumptions or mechanisms that make this feasible in practice.

5. Experimental Details in the Introduction.
In the Introduction, the experimental setting for the results presented in Figure 1 is missing. Even a minimal description (model, dataset, data distribution, number of clients) would help the reader interpret the figure. Please also provide a reference to where full experimental details appear later in the paper.

6. Readability of Figure 1.
Figure 1 is not straightforward to interpret. Please consider redesigning it to make the message clearer or move it to a later section where more context is available.

7. Experimental Scope and Validation.
Although the Introduction mentions experiments on non-convex neural networks in cross-device settings, two concerns arise:
(i) The neural network used appears to be a simple CNN.
(ii) The scale of the experiment involves only 100 clients.
Please consider validating the results on more complex neural networks, larger and more realistic datasets, and with different numbers of clients to strengthen the empirical claims.

8. Reproducibility.
The reproducibility of the experiments is limited. Please improve the shared code and documentation to allow straightforward replication of the reported results.

9. Notation Clarification.
The prox expression used at the end of Assumption 2.3 and in Algorithm 1 is never introduced. Please define it clearly and explain its role in the proposed method.

10. Communication Cost Reduction.
The abstract reports a 74 percent reduction in total communication. Does this result refer to the hierarchical federated learning setting? If so, please clarify this in the text, since the result may be specific to that configuration and less general than in standard FedAvg settings.

### Soundness
3

### Presentation
1

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
This paper introduces "Cohort Squeeze," a framework for cross-device federated learning designed to enhance communication efficiency. The proposed method, SPPM-AS, allows a selected cohort of clients to perform multiple local communication rounds before synchronizing with the central server. The authors provide a theoretical convergence analysis for strongly convex problems and empirically demonstrate the method's effectiveness in reducing total communication rounds on both convex and non-convex tasks.

### Strengths
Strengths:
1.The paper provides a comprehensive investigation and comparison of three different client sampling strategies (NICE, Block, and Stratified Sampling), offering valuable insights.
2.The empirical evaluation is thorough, covering both convex and non-convex settings across multiple datasets, which demonstrates the practical applicability of the approach.
3.The experimental results are highly promising, showing a potential reduction in communication costs by up to 74% compared to baseline methods.

### Weaknesses
Weaknesses:
1.The novelty of the approach is not clearly distinguished from Hierarchical Federated Learning (HFL). The paper lacks a necessary discussion and comparison with the existing HFL literature, making the contribution's positioning unclear.
2.The description of the algorithm is ambiguous. It is not clear how the "K local communication rounds" are actually implemented, i.e., whether clients within a cohort communicate with each other to solve the proximal operator, and if so, how.
3.The communication cost model (defined as TK) is oversimplified and potentially unfair. It assumes that the cost of a global communication round is identical to that of a local, intra-cohort communication round, which leads to a questionable comparison with baselines like FedAvg.
4.The theoretical convergence guarantee is provided only for the strongly convex case. This leaves a significant gap, as many of the successful experiments are conducted on non-convex neural networks.

### Questions
Questions: 
1.Could you elaborate on the relationship between your proposed framework and Hierarchical Federated Learning (HFL)? What are the key distinctions and novel contributions of your method when compared to existing hierarchical approaches?
2.Could you provide a more detailed description of the intra-cohort process during the K local communication rounds? Specifically, do clients communicate with each other to solve the proximal operator, and if so, what is the protocol for this communication and aggregation?
3.The total communication cost is defined as TK, which assumes local and global rounds have the same cost. Could you justify this modeling choice? How might the comparative results change with a more nuanced cost model that distinguishes between the two (e.g., Cost = T × C_global + T × K × C_local)?
4.Given the strong empirical results in non-convex settings, could you provide any theoretical insights or discussion on the convergence properties of SPPM-AS for non-convex objectives?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces SPPM-AS, a stochastic proximal point variant that supports different sampling strategies. When applied to federated learning, the method naturally supports various client-sampling protocols. The authors prove convergence to an $\epsilon$-approximate solution under strong convexity assumptions and validate the approach through experiments demonstrating its performance.

### Strengths
**1.** The paper proposes a new federated learning algorithm designed to improve communication efficiency by allowing flexible client-sampling strategies.

**2.** The work is supported by both theoretical analysis and empirical evaluations, demonstrating the effectiveness of SPPM-AS in several settings.

### Weaknesses
**1. Motivation vs. focus mismatch.** The abstract emphasizes *cohort squeeze* and introduces the work as addressing a new communication bottleneck. However, most of the paper instead centers on client-sampling strategies within standard federated learning communication rounds. Although Sections 3.3 and 3.6 present communication-related experiments, the paper does not provide a clear and rigorous communication-cost analysis of SPPM-AS, which would be expected given the motivation presented in the abstract and title.

**2. Practicality of iteration-complexity result.** The convergence guarantee relies on selecting a specific stepsize $\gamma$ that depends on $\sigma_{\star,AS}$, which itself is defined in terms of $x_\star$. Thus, the theoretically optimal stepsize depends on unknown quantities and cannot be directly used in practice, meaning the provable accuracy bound does not strictly apply to the implementable algorithm.

**3. Strong convexity assumption.** The theoretical analysis requires strong convexity (Assumption 2.2). While this leads to cleaner guarantees, it limits applicability to real federated learning scenarios, where models are typically non-convex (e.g., deep networks). The gap between theory and practice is acknowledged but remains significant.

**4. Presentation and clarity.** The paper would benefit from a clearer and more consistent connection between the communication problem highlighted in the introduction/abstract and the theoretical development in the main body.
Algorithm 1 (SPPM-AS) and Table 1 could be explained more thoroughly in the main text to improve readability.

**Minor issues.**

Line 123: $n = \tilde{b} 1,2,\cdots,n$ should likely be corrected to $[n] = \lbrace1,2,\cdots,n \rbrace$. 

Line 161: stray [] bracket.

### Questions
Can your analysis be extended to non-convex objectives? If not, what part of the proof critically depends on strong convexity? Understanding this would help clarify whether the framework could be adapted beyond the strongly convex setting.

### Soundness
2

### Presentation
2

### Contribution
2
