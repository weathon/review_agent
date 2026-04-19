# An Exact Solver for Satisfiability Modulo Counting with Probabilistic Circuits

- Decision: Reject
- Scores: 6, 3, 8, 5, 3

## Abstract
Satisfiability Modulo Counting (SMC) is a general language to reason about problems integrating statistical and symbolic artificial intelligence. An SMC formula is an SAT formula in which the truth values of a few Boolean predicates are determined by model counting, or equivalently, probabilistic inference. Existing solvers optimize surrogate objectives and hence provide no formal guarantee. Hence, an exact solver is desperately in need. However, the direct integration of satisfiability and probabilistic inference solvers results in slow SMC solving because of many back-and-forth invocations of both solvers. We develop KOCO-SMC, a fast exact SMC solver, exploiting the fact that many similar probabilistic inferences are needed throughout SMC solving. We compile the probabilistic inference part of SMC solving into probabilistic circuits, supporting efficient lower and upper-bound computation. Experiment results in several real-world applications demonstrate that our approach provides exact solutions, much better than those from approximate solvers, while is more efficient than direct integration with the current exact solvers.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Satisfiability Modulo Counting (SMC) is an extension of SAT that incorporates constraints that involve model counting. Since probabilistic inference and model counting are two closely related concepts, SMC adaptively captures the satisfiability problem in scenarios involving uncertainty. 

In this paper, the authors introduce an exact SMC solver (KOKO-SMC). The core idea is to track the upper and lower bounds of the probability inside probabilistic constraints during variable assignments. If these bounds violate the satisfaction condition the conflicts are recorded as learned Boolean clauses and appended to the Boolean part. This prevents the same conflict from occurring in future interactions.

Another feature of the approach is that knowledge compilation from discrete functions are integrated into probabilistic circuits. This speeds up the updates of bounds.

### Strengths
The paper presents an interesting approach for Satisfiability Modulo Counting. The authors also implement their approach in a new exact solver for SMC. The experimental results give some evidence that their solver performs better than other solvers in some relevant benchmarks.

### Weaknesses
The approach is strongly based on the conflict driven clause learning approach (CDCL) paradigm. So from a technical perspective, the main contribution seems to be the Upper-Lower Watch algorithm, which is an algorithm for monitoring the Marginal MAP of a probabilistic constraint using probabilistic circuits. I'm not sure how novel this approach is.

### Questions
Could you discuss related work on the Upper-Lower Watch approach to keep track of the values in the Marginal MAP? What is the novel step in this algorithm, how does it compare with existing approaches, etc?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper focuses on the design of a solver for  satisfiability modulo counting (SMC). SMC was introduced in 2014 by fredrikson and jha but have since received renewed interest over the past two years. Essentially, SMC allows counting predicates. The paper discusses a counting solver that combines CDCL methodology with lower and upper bounds via probabilistic circuits. The high-level description in the paper is not very different compared to   fredrikson and jha except for the usage of lower and upper bounds via probabilistic circuits.

### Strengths
On a high-level the paper is promising but my concerns are with empirical evaluation.

### Weaknesses
The empirical evaluation is limited to one specific setting which does not capture the class of SMC, in fact, the formulas can be encoded as stochastic SAT and therefore, one can rely on state of the art SSAT solvers such as [1]. 
Here is how we can encode the studied formulas into SSAT. 

We can simply ask to solve Exists (X) Random (Y) \phi(X) \wedge f(X,Y).  -- note that this would provide a X for which the model count of f(X,Y) is maximized. The SSAT solvers are indeed aware of the exact count, and this can be retrieved from their logs. Otherwise, one can simply substitute the assignment of X and just run any of the state of the art model counters and compute counts. 

There is one possibility that the aforementioned approach will not work but this requires authors to demonstrate empirically. I hope authors would do so during rebuttal phase and in such a case, I would be happy to revise my score to Accept. 

[1] https://github.com/NTU-ALComLab/ssatABC

### Questions
Please see above.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper studies an extended satisfiability problem involving both propositional and probabilistic inference and proposes an exact solver, KOCO-SMC, to this problem. The main idea is to integrate a Boolean SAT solver with a probabilistic circuit to detect the conflict in a probabilistic constraint early by monitoring the lower and upper bounds of its probability value. Experimental results over synthetic and real-world problem instances demonstrate the superior runtime performance of KOCO-SMC compared to the exact and approximate baselines.

### Strengths
1. The satisfiability modulo counting (SMC) problem is an emerging formulation that can capture symbolic constraints and probabilistic uncertainty simultaneously. The authors developed an efficient solver for the SMC problem which significantly outperformed the baselines. 
2. The authors identified that the main weakness of the existing solvers is unable to detect the conflict in probabilistic constraints timely. They addressed it by using a probabilistic circuit to monitor the lower and upper bounds of the probability value for early conflict detection.
3. The paper presents the application of the SMC problem in real-world scenarios including supply chain design and package delivery problems, and shows the superior performance of applying their solver compared to the baselines.

### Weaknesses
1. The abused notation for $x_i$ in Section 3.1 confused me. For example, the sum over $x_3$ and $x_4$ at line 226 looks weird since they are already assigned to False at line 224.  I recommend using different variables to indicate the road selection and clearness.
2. The variable-watching scheme for probabilistic constraints misses some immediate conflicts. For example, we can not detect the conflict at line 233 timely if $x_1$ is assigned True but not watched.

### Questions
1. For the trivial combination of a SAT solver and a model counter, do you extract a conflict clause from a probabilistic constraint if the model counter finds it UNSAT?
2. In Figure 3, a problem instance is considered solved if one correct solution is found within five runs. Can you elaborate on how to define “solved” for an UNSAT instance?
3. What is the benchmark used in Figure 5? How did you calculate the running time of the left figure considering the unsolved instances? Did you consider the solved instances only or add a penalty for the unsolved instances?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper introduces KOCO-SMC, an exact solver for Satisfiability Modulo Counting (SMC) problems, integrating probabilistic circuits for efficient conflict detection. Unlike existing solvers, KOCO-SMC provides exact solutions efficiently by pre-compiling probabilistic inference. Key contributions include (1) KOCO-SMC as an efficient exact solver, (2) superior performance in experiments compared to other baselines, and (3) a case study showing its real-world applicability.

### Strengths
The paper’s strengths include KOCO-SMC’s innovative conflict detection for early pruning, which significantly speeds up SMC solving, especially in unsatisfiable cases. It effectively overcomes limitations of prior solvers, making it highly practical for real-world applications.

### Weaknesses
The paper's weaknesses include a limited problem scope, which restricts the broader applicability and generalizability of the results. Expanding the range of problem types could strengthen its impact. Additionally, the writing lacks readability.

### Questions
1. What is the complexity of KOCO-SMC?
   A complexity analysis would clarify its efficiency compared to existing SMC solvers.

2. How does KOCO-SMC differ from existing SAT-exact solvers? 
   A detailed comparison could highlight KOCO-SMC’s unique conflict detection and pruning methods.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper proposes a method for solving satisfiability modulo counting (SMC) problems. SMC is a problem that consists of a combination of a Boolean SAT problem and a model counting problem. Current exact solvers combine a SAT solver with a model counting solver. However, this approach requires an excessive number of invocations of SAT and model counters and could be slow. The proposed KOCO-SMC solves SMC problems by using a probabilistic circuit and a SAT solver. By first compiling a probabilistic distribution into a circuit, it accelerates to answer probabilistic queries on solving an SMC problem. Moreover, using PCs enables the computation of upper and lower bounds of probabilities, which results in the efficient discovery of conflicts and contributes to efficient search. The paper evaluates the proposed approach extensively with real and synthetic datasets, showing that the proposed exact approach is more efficient than baseline methods.

### Strengths
1. The proposed method uses PCs effectively. PCs shine in situations where we need to answer different probabilistic queries for a distribution extensively. Speeding up SMC with a PC is a clever idea.
2. Extensive experimental results show the superiority of the proposed over multiple baseline methods on both synthetic and real data. Reading to significant improvements.
3. The presentation of the paper is clear and easy to understand the contributions.

### Weaknesses
Assumption 1 is not true. As shown by (Choi et al., 2022), smoothness, decomposability, and determinism do not enable tractable computation of MMAP query. Choi et al. (2022)  have shown that PCs satisfying Q-determinism (Definition 2) support tractable MMAP queries, but otherwise, it is difficult to answer MMAP in polytime. Since Q-determinism is the property specific for a given query set $Q$, answering MMAP for any query $Q$ is intractable. This was the motivation why Choi et al. (2022) proposed an iterative pruning method for MMAP. 

If my understanding of the MMAP query is correct, Lemma 1 does not hold as is since it depends on Assumption 1. The paper should show why the proposed method can efficiently compute upper and lower bounds.

(Minor) Line 258: Unit propagation will not force $x_2 = False$ for this case.

### Questions
I'd be happy if the authors addressed the above concern about the MMAP query.

### Soundness
2

### Presentation
3

### Contribution
3
