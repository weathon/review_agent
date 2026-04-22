# Learning Local Search with Theoretical Indicators for Job Shop Scheduling

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Job shop scheduling problem (JSSP), where job sequences must be assigned across multiple machines to minimize makespan under fixed routes and varying processing times, is one of the most challenging combinatorial optimization problems. To improve search efficiency, we propose LSI, Local Search with Indicators, a learning-based local search method for JSSP. LSI integrates scheduling-theoretic conditions as indicators into the action evaluation, enabling the policy to focus on swaps that guarantee makespan reduction. By incorporating theoretically proven conditions into the action evaluation, LSI prioritizes promising swaps rather than treating all moves equally, representing a principled improvement of makespan. Despite relying only on a lightweight multilayer perceptron (MLP) policy network, LSI achieves competitive or superior performance compared to strong state-of-the-art approaches on diverse JSSP benchmarks, offering faster inference and robust scalability without retraining. These results demonstrate the effectiveness of embedding problem-structured theoretical principles into learning-based combinatorial optimization.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces LSI, a method that integrates three theoretically derived necessary conditions as binary indicators into a policy network. This design allows an MLP-based policy network to predict actions within the N5 neighborhood and iteratively refine an initial solution for the Job Shop Scheduling Problem (JSSP).

### Strengths
1. The paper provides comprehensive descriptions of the model design and experimental parameters, ensuring that the research can be reproduced.

2. The theoretical groundwork is robust, with detailed proofs of the necessary conditions supplied in the appendix to support the proposed method.

### Weaknesses
1. The contribution of the paper is relatively limited, as it primarily introduces three theoretical conditions without demonstrating their clear applicability to other combinatorial optimization problems.

2. LSI shifts the learning paradigm from a data-driven approach to one reliant on heuristic methods by incorporating expert knowledge. This approach contradicts the field's broader goal of reducing manual intervention. The motivation and necessity of this paradigm shift require further evaluation.

3. The experimental validation of the three theoretical conditions is insufficient (see Questions), which weakens their credibility and impact.

4. The improvements of LSI over NeuroLS are marginal, making it difficult to justify the unique role of the three proposed conditions. It would be more convincing to apply these conditions to other learning-based methods and measure the resulting gains.

5. The definition of the Markov Decision Process (MDP) state appears flawed. The embedding structure's parameters should be treated as the true state, rather than being conflated with the state representation.

6. The paper claims that MLP was chosen to balance lightweight implementation with high performance. However, the ablation study does not compare the runtime between MLP and bi-GAT, nor does it demonstrate that MLP outperforms bi-GAT.

### Questions
1. Appendix H is empty, yet it is critical for understanding the actual impact of the three theoretical conditions. For example, do these conditions genuinely influence the predicted actions to satisfy the proposed criteria? How do the rates of condition satisfaction evolve across iterations?

2. Why were these three specific conditions selected? Could there be additional conditions that might be equally or more effective? What would happen if the three conditions were replaced with random numbers? Have experiments been conducted using only one or two of the conditions to test their individual contributions?

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
This paper proposes a learning-based local search framework that integrates three theoretically derived conditions for makespan reduction as binary indicators to learn to solve the job shop scheduling problem. Despite relying only on a lightweight multilayer perception policy network, their method achieves competitive performance across JSSP benchmarks.

### Strengths
1. The idea of incorporating theoretically derived conditions for JSSP to improve learning seems quite neat and seems to be a novel direction as previous research typically relies on hand-crafted (heuristic) features.
 
2. Empirical results on a variety of JSSP benchmarks seem promising, further validating the idea.

### Weaknesses
1. I find the presentation of the theoretical conditions a bit hard to understand. Limited intuitions were provided to explain why each condition (prop 1,2,3) holds. 

2. (I understand this is acknowledged by the authors, so I appreciate them being upfront, but) it is actually a bit concerning that the conditions seem to be tailored to JSSP only. While I understand that extending to other COPs require significant empirical setup cost, but to strengthen the work, the authors should consider providing experiments on different JSSP variants (e.g. different objectives, flexible job shop scheduling instead of just JSSP) and discuss how similar conditions may be able to applicable to other COPs (discussions, pointers, references is sufficient for other COPs). 

3. To my understanding, the benchmark instances that the authors test on are quite standard and may not reflect real world complexities and scale.

### Questions
My questions are directly related to the weaknesses:

1. Can the authors provide illustration of the three propositions to help readers visualize the conditions more easily, and can the authors explain the intuition of why each condition works?

2. Can the authors provide experiments on other JSSP variants, and discuss how similar theoretical conditions may help the learning for other COPs?

3. Can the authors test on real JSSP benchmarks with more complex problem distributions, and further extend the evaluation to even larger scales?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces a new learning-driven local search technique for job shop scheduling. Its main innovation is using three theoretically-grounded, necessary conditions for improving the schedule makespan, which are converted into simple binary indicators. This approach allows a very simple MLP-based policy network to surpass more complex state-of-the-art methods that use sophisticated graph neural networks, proving the power of embedding theoretical knowledge into machine learning for combinatorial optimization.

### Strengths
1. The paper is well-organised and well-written in general. 
2. The authors demonstrate their method's robustness via a detailed empirical evaluation performed on a range of standard JSP benchmarks.
3. The research is well-motivated and justified with empirical evidence that echoes their claims.
4. The performance is superior to existing learning-based methods.

### Weaknesses
1. The paper's novelty is constrained, as its core framework is heavily derived from prior works, L2S and TBGAT. It adopts their Markov Decision Process formulation and a subset of the N5 neighbourhood for the local search process. The sole apparent contribution—the application of theoretical indicators to select operation pairs—is itself adapted from existing operations research literature, further limiting the originality of the proposed method.
2. Leading neural combinatorial optimization (NCO) methods, such as L2S and TBGAT, rigorously demonstrate that their computational complexity scales linearly with problem size. Given the fundamental importance of efficiency in combinatorial optimization, the absence of a similar analysis for the proposed method diminishes the completeness of its evaluation and, consequently, its practical significance.
3. The method's limited novelty, combined with its exclusive focus on the Job Shop Problem, restricts its overall impact and broader relevance to the field.

### Questions
1. The paper would be strengthened by a detailed computational complexity analysis of the proposed algorithm.
2. What is the computational overhead associated with calculating the theoretical indicators in each step of the local search?
3. For large-scale problem instances (e.g., exceeding 2000 operations), could the process of computing these indicators become the primary computational bottleneck, potentially undermining the method's efficiency?
4. Given that the theoretical foundations are adapted from prior Operations Research literature, to what extent does this work contribute new theoretical discoveries, as opposed to the application of existing ones?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes LSI, a learning-based local search method for the JSS problem. The core contribution is the integration of a theoretically-derived conditions for makespan reduction directly into the policy network as binary indicators. This approach allows the use of a simple MLP-based architecture as compared to complex GNN encoders. The paper demonstrate through extensive experiments that LSI achieves better performance on standard JSS benchmarks.

### Strengths
The paper derives and uses necessary conditions to guide the search. The quality of the work is good with rigorous experimental evaluation against baselines. The method is scalable to larger instances by being trained only on 10x10.

### Weaknesses
- The primary contribution that is the hand-derived theoretical indicators based on domain knowledge.  The process of deriving such indicators for other variants of JSS or for different objectives for standard JSS (e.g., total weighted tardiness) appears non-trivial and can limit the broader applicability of this specific method without significant additional theoretical work.
- Currently the method is evaluated only on JSP. Experiments on other variants of JSP (e.g., Flexible JSP) or on other objectives would strengthen the work.

### Questions
- The proposed indicators are specific to JSP for makespan minimization. Could you comment on the feasibility and potential challenges of deriving a similar set of theoretical indicators for other scheduling objectives, and to other JSP variants?
- The initial solution is based on FDD/MWKR. What is the impact on performance if the initial solution is randomly initialized?
- The abstract states that LSI offers "faster inference"; however, the runtime reported in Table 1 is mostly comparable to learning-based methods like SN and IRD, and this is given that they use complex GNNs while LSI uses MILP.

### Soundness
3

### Presentation
3

### Contribution
2
