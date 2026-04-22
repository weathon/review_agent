# Circuit HMM: A Deterministic Hidden Markov Model for Automated Sequential Circuit Design

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 6, 4

## Abstract
Designing logic circuits requires significant time for manual programming, which hinders the rapid iteration of product development. To alleviate this extensive manual effort, researchers have investigated machine learning methods for automatically programming the hardware description language (HDL) code, and have achieved success in designing combinational circuits. However, due to the complexity of internal state transitions, the design accuracy for sequential circuits remains insufficient for practical applications, whereas the design accuracy of combinational circuits can achieve 99.99999999999%.

This paper proposes a novel machine learning model, Circuit HMM, a deterministic Hidden Markov Model (HMM), for accurately designing sequential circuits. Our key insight is that the input-output relationship of the sequential circuits can be formalized as a Markov Process, significantly reducing the design space. With this computationally efficient model, we prove that the design accuracy of the sequential circuit converges to 100% with linear complexity. Circuit HMM (1) first learns the hidden states by constructing an effective finite state machine (FSM) by heuristic **state mining**, which ensures the error rate caused by the inaccurate states converges to zero; (2) accurately transforms the sequential circuit design problem into a series of combinational circuit design problems by efficient **state encoding**; and (3) then learns the combinational circuit implementation from the input-output relations with a state-of-the-art logic regression tool, i.e. the BSD Learner, which ensures the combinational error rate converges to zero. Experimental results demonstrate that the proposed method can accurately design real-world circuit modules comprising up to 5,000 logic gates, significantly outperforming the state-of-the-art. In 41 out of 43 cases, the design accuracy converges to 100% within 5 minutes.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
To address the insufficient accuracy of existing methods in automated sequential circuit design, this paper introduces Circuit HMM, which formalizes the input-output relationship of sequential circuits as a Markov process to reduce the design space. The design is realized through a three-step procedure: state mining, state encoding, and circuit generation. Experiments demonstrate that the method can accurately design practical circuit modules containing up to 5,000 logic gates. In 41 out of 43 cases, the design accuracy converged to 100% within 5 minutes, outperforming existing approaches.

### Strengths
1. This paper addresses the important problem of automated sequential circuit design, offering well-defined practical value. 
2. The experimental results significantly outperform existing baseline methods, demonstrating clear advantages in terms of accuracy, scale, and efficiency. 
3. The theoretical derivations are logically rigorous, with proofs provided for the relevant theorems.

### Weaknesses
1. The paper has a deficiency in terms of innovation. Since HMM itself is a classical sequential modeling framework, the primary contribution lies in its deterministic adaptation and engineering workflow optimization. Moreover, the state mining (reliant on BFS and Monte Carlo) and the encoding optimization (dependent on Simulated Annealing) constitute applications or improvements of well-established algorithms, and thus the work does not present fundamental algorithmic originality.
2. The paper does not explicitly state whether its assumptions are valid for all sequential circuit types, such as asynchronous or circuits with complex feedback. The validation is solely based on experiments with synchronous circuits.
3. The paper lacks ablation studies for some modules. It fails to verify the necessity of the BSD Learner module. How would accuracy and efficiency change if BSD were replaced with other combinational circuit learning tools (e.g., a BDD Learner)? Alternatively, if the SA-based encoding optimization were removed, how much would the circuit size increase? Including such experiments would strengthen the validation of the method's effectiveness.
4. The study does not test the performance on extreme-scale circuits (e.g., those with 10,000+ gates), nor does it analyze the efficiency degradation of state mining when the number of states becomes very large (e.g., exceeding 10⁴). Consequently, it fails to verify the scalability limits of the proposed method.
5. The paper does not specify the training parameters (e.g., learning rate, number of iterations) for the Transformer/LSTM models.

### Questions
1. Please respond to the concerns I have raised in the 'weaknesses' section. If the revision can adequately address most of the critical issues, I would consider raising the score.
2. In my personal opinion, the contribution of this paper lies more in its solid experimentation and significant performance improvements, rather than in its theoretical advancement in AI. Considering this, EDA-focused conferences (such as DAC or ISCA, with an upcoming November deadline) would be a more suitable venue for it.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents Circuit HMM, a deterministic Hidden Markov Model designed to automate the design of sequential circuits with high accuracy. ​

Circuit design automation reduces manual programming time, enhancing product development speed.
Previous machine learning methods excelled in combinational circuits but struggled with sequential circuits due to complex state transitions. ​
Circuit HMM achieves 100% design accuracy with linear complexity by modeling sequential circuits as Markov Processes. ​
The model learns hidden states through heuristic state mining, transforming the design problem into combinational circuit problems. ​
Experimental results show that Circuit HMM can design circuits with up to 5,000 logic gates, achieving 100% accuracy in 41 out of 43 cases within 5 minutes. ​


The methodology of Circuit HMM involves three main stages: State Mining, State Encoding, and Circuit Generation. ​

State Mining identifies necessary internal states using a Breadth-First Search (BFS) approach, ensuring the error rate converges to zero. ​
State Encoding transforms the sequential design problem into combinational problems, optimizing state representation to reduce circuit size. ​
Circuit Generation uses the BSD Learner to create combinational circuits and outputs HDL code for the sequential circuit. ​
The process guarantees that the design accuracy converges to 100% with sufficient input-output examples.

### Strengths
The empirical results seem strong. The proposed method is evaluated against established benchmarks, demonstrating superior performance in circuit design accuracy and size. ​

--Evaluated on VerilogEval v2 and DesignWare IP datasets, showcasing its effectiveness in real-world applications.

--Achieved an average accuracy of 60.03% across various circuits, with some achieving 100% accuracy.

--The method significantly outperforms state-of-the-art techniques, particularly in larger circuits with complex state transitions. ​

--The results indicate that Circuit HMM can handle circuits with up to 5,000 gates efficiently, maintaining a small circuit size.

### Weaknesses
The paper is very poorly written. The claims made are strong, but this reviewer has struggled to understand the technical basis for the claims, and to understand the empirical results.

I don't understand the first main claim: "It formalizes the input-output relationships as a Markov Process, which reduces the complexity of the design space." Where is this claim (a) formally stated and (b) proven theoretically or empirically? Section 4.2 seems to be where this is done, but this is entirely inadequate. I am looking for direct, clear evidence that you reduce the complexity of the design space. By what factor is this reduction? Guaranteed reduction, or expected reduction?

Your explanation of "complexity" is poor. I understand the complexity of combinational circuits; now the complexity of sequential circuits is different. Estimating the complexity of the design space for sequential circuits is a challenging problem because it involves both combinatorial and temporal dimensions. This article needs a clear exposition of this. Total Complexity ~ Combinational Topologies×State Machines, and if the number of distinct state machines with m states and i inputs is roughly:
FSM Complexity ~ (2^m)^{(2^i 2^m)}
This grows double-exponentially with the number of inputs and memory elements.

How precisely does your approach reduce this complexity? What are the precise new bounds?

Claim 2: "we prove that the design accuracy of the sequential circuit converges to 100% with linear complexity" Theorem 2 seems to cover this claim, so I am happy with this.

Claim 3: "the proposed method can accurately design realworld circuit modules comprising up to 5,000 logic gates". Here the writing is so poor that it is difficult for me to verify that this is true.

Table 1 is the key empirical artefact of the paper and needs much better explanation. Reviewers are NOT obliged to refer to Appendices, and a proper explanation is needed here. Even going to the Appendices I still struggle to understand your results.

How does this approach scale? I see you clear outputs that help me understand this crucial question.

Overall, the poor writing leaves me wondering what actually has been achieved. I don't see clear evidence to back up your claims.

### Questions
1. What is the reduction is state-space complexity of the approach?
2. Show formally of empirically how the approach scales.

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
2

### Summary
The authors propose a deterministic hidden Markov model for automatically generating a sequential circuit. Their method performs better than the deterministic method and the probabilistic hidden Markov method.

### Strengths
1. The performance of this work is much better than the baseline methods.
2. The modeling method of sequential behavior is insightful.
3. The complexity of this method is better than probabilistic methods.

### Weaknesses
1. It seems that only one level of latches will be considered in this method, which is not general enough for all types of sequential circuits.

### Questions
1. I am a little bit confused about the "deterministic HMM" that is claimed by the authors in the paper. In \textbf{State Mining} and \textbf{State Encoding}, "randomly generate a set of PIs" and "optimize the state encoding using a Simulated Annealing algorithm" could add randomness in the algorithm. Maybe the authors should explain the "deterministic" with more words.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper is an extension of Binary Speculation Diagram Learner (Cheng, 2024a), from combinatorial to sequential circuit design. Given a black-box sequential circuit (oracle), the proposed algorithm aims to implement it. The algorithm firstly uses breadth-first search to find all the internal states of the sequential circuit, which iteratively finds new reachable states from current known states (by randomly generated PIs). Then all the $N$ states are encoded as $⌈log_2 N⌉$ bits. In this way the problem reduces to $N$ combinatorial circuit design problems which can be solved by (Cheng, 2024a). The state encoding can be optimized by a simulated annealing algorithm which minimizes the BSD node count. Experimental result shows that it can successfully generate sequential circuits up to thousands of states and gates.

### Strengths
- This paper focuses on generating sequential circuits, which is more challenging and less explored in literatures than combinatorial ones
- The empirical experimental result is good
- Many details in the appendix

### Weaknesses
- The idea of regarding sequential logic circuits as Markov processes is not very new (if it is not a common sense). Sequential logic circuits can be modelled as finite state machines, so Figure 1 (c) is actually more aligned with my knowledge on sequential circuits, and the perspective shown in Figure 1 (b) may not be so common. 
- The problem definition (section 2.1) confused me. While it is directly referenced from (Cheng, 2024a), I feel that the actual problem in this paper is different. The input is not a fixed dataset of $N$ input-output pairs, but rather a black-box sequential circuit, which can be interacted by any inputs during the algorithmic process. I think only in such a way can the random generation of PIs (line 252) make sense. 
- The impact of the paper is not clearly presented. Given that the target sequential circuit does exist and can be interacted, the paper is more likely a "reverse engineering" of an existing sequential circuit design, rather than learning a design from scratch based on examples of input-output pairs. It is not clear to me how the proposed approach can be integrated into existing EDA tools (line 1014).
- The comparison with baselines methods (Transformer, LSTM, BSD) is not very fair. The proposed approach can interact with the oracle with arbitrary inputs (online), while the other approaches learn on a fixed dataset (offline). I think the main contribution of this paper is the state mining algorithm, so a stronger baseline would be to replace the proposed state mining algorithm with other approaches (including those mentioned in appendix A.7) and compare the final accuracy and circuit size.

### Questions
- I wonder whether a sequential logic circuit can be defined as a function $\phi:\\{0,1\\}^n\rightarrow\\{0,1\\}^m$ (line 140). Due to the existence of states, the output of a circuit can be different even if the same inputs are fed into the circuit.

### Soundness
2

### Presentation
2

### Contribution
2
