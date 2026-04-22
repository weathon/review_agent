# From Sequential to Parallel: Reformulating Dynamic Programming as GPU Kernels for Large-Scale Stochastic Combinatorial Optimization

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 2

## Abstract
Dynamic programming (DP) is central to combinatorial optimization, optimal control, and reinforcement learning, yet its perceived sequentiality has long hindered scalability. We introduce a general-purpose GPU framework that reformulates broad classes of forward DP recursions as batched min--plus matrix--vector products over layered DAGs, collapsing actions into masked state-to-state transitions that map directly to GPU kernels. This approach removes a major bottleneck in scenario-based stochastic programming (SP), where the use of DP has traditionally restricted the number of scenarios due to excessive computational cost.
Our framework exposes massive parallelism across scenarios, transition layers, and, when applicable, route or action options, via self-designed GPU kernels that implement Bellman updates with warp-/block-level reductions and numerically safe masking. In a single GPU pass, these kernels can process over $10^6$ uncertainty realizations, far beyond the capacity of prior scenario-based methods. We demonstrate the approach in two canonical SP applications: (i) a vectorized split operator for the capacitated vehicle routing problem with stochastic demand, exploiting **2D** parallelism (scenarios $\times$ transitions); and (ii) a forward inventory reinsertion DP under an order-up-to policy, exploiting **3D** parallelism (scenarios $\times$ inventory transitions $\times$ route options). Across benchmarks, the implementation scales nearly linearly in the number of scenarios and achieves one to three orders of magnitude speedups over multithreaded CPU baselines, yielding tighter SAA estimates and consistently stronger first-stage decisions under identical wall-clock budgets. Viewed as hardware-aware software primitives, our min--plus DP kernels offer a drop-in path to scalable, GPU-accelerated stochastic discrete optimization.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents a novel and impactful GPU-based reformulation of dynamic programming (DP) for large-scale stochastic combinatorial optimization. By expressing DP recursions as batched min–plus matrix–vector products over layered DAGs, the authors unlock massive scenario-level and transition-level parallelism. The approach demonstrates impressive speedups and enables practical evaluation of millions of uncertainty realizations, leading to improved SAA solution quality under realistic time budgets. However, there are several minor issues and open questions that need clarification or further validation. If the authors can address these points during the rebuttal, I would consider raising my score. The contribution is solid, and the direction is promising.

### Strengths
1. The idea is novel: By formulating stochastic programming (SP) as a specially DP, the paper proposes a novel approach which leverages GPU to solve SP. While the traditional sampling-based SP only includes thousands of scenarios, the new approach allows millions of scenarios. It also speeds up the computation. 

2. The examples shown in the draft are simple but very clear: The 2D and 3D examples shown in the paper are very classical DP problems and the paper make it very clear about how to take advantage of GPU to solve the issues.

3. The experiment results are solid and sound. The results clearly illustrates the advantages of solving the problem using GPU vs CPU.

4. The paper also provides open-source CUDA implementation for reproducibility.

### Weaknesses
1. Limitation of the applicable cases: the authors in the conclusion show that the method is only applicable to some specific type of issues. 
Further explain in the Questions sections.

2. Some of the notations are not very clear. Further explain in the Questions sections.

3. Minor typos and grammar issues. Further explain in the Questions sections.

### Questions
1. The authors talk about the limitation of the approach. A related question: will the approach still be applicable if the action or the state is continuous? 
2. Could you explain what will happen in the edge case, like if there’s some irregular state spaces, how does the algorithm perform.
3. From the 2D and 3D example, it seems like the algorithm design is a bit different in the appendix. Is it possible to write a unified algorithm and somehow put it in the main body instead of the appendix? Since the specific algorithm steps are very important for the paper. 
4. The paper uses DAG several times, and I suppose it refer to Directed Acyclic Graph, but I don’t think it is formally defined (Please ignore this if I am wrong). Also, some intuitions about why DAG matters here can make the paper easy to follow.
5. Some terminology inconsistency. E.g., two-dimensional vs 2D.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper accelerates finite-horizon forward dynamic programs by formulating each Bellman update as a batched min-plus matrix–vector operation and is engineered to run in parallel on GPUs. Two applications are demonstrated: (1) a VRP split problem with stochastic demands (choosing route cuts along a fixed giant tour), exploiting 2D parallelism across scenarios and transitions, and (2) a dynamic stochastic inventory routing reinsertion operator, exploiting 3D parallelism across scenarios, inventory transitions, and route options. The results show higher scenario throughput and improved expected-cost decisions under the same wall-clock budget.

### Strengths
1. The paper neatly reformulates forward DP over layered DAGs as masked min–plus matrix–vector products, making the recursion uniform, parallel, and easy to adapt.

2. The use of masking with warp- and block-level reductions, and structured 2D/3D parallelism is technically solid and makes effective use of GPU hardware.

3. Experiments demonstrates how large-scale scenario batching achieves high GPU throughput, clearly linking faster computation to more stable estimates and better decisions under fixed time budgets.

4. The VRPSD and DSIRP examples show that the approach applies to different DP structures, and the toy examples and algorithmic sketches improve clarity.

### Weaknesses
1. Overstated novelty and missing literature. The paper treats the “batched min–plus matrix–vector” view as a key contribution, but this idea is well known from tropical/semiring DP and existing GPU semiring frameworks [4, 2, 5]. It should also acknowledge prior GPU-DP work in [1] and [3]. In my mind, the true novelty lies in the scenario-batched kernels and problem-specific engineering, not in the algebraic formulation itself.

2. The GPU is compared to a different CPU algorithm, mixing hardware and algorithmic effects. A multi-threaded CPU version of the same method would make the comparison more fair.

3. The paper notes GPU memory limits but gives no quantitative analysis. With 106 scenarios on an 11 GB GPU, state grids per stage must be very small if transitions are stored densely. This suggests that large scenario batching trades off against state-space size, a limitation that should be discussed.

4. The VRPSD and DSIRP examples are solid but narrow; extending to more complex DP forms (non-layered, constrained, or
risk-sensitive) remains unclear. I feel that the work is better seen as a systems contribution for forward DP rather than a general GPU-DP framework.

References

[1] M. A. Boschetti, V. Maniezzo, and F. Strappaveccia. Route relaxations on GPU for vehicle routing problems. European Journal of Operational Research, 258(2):456–466, 2017.

[2] T. A. Davis. Algorithm 1000: Suitesparse: Graphblas: Graph algorithms in the language of sparse linear algebra. ACM Transactions on Mathematical Software, 45(4):44:1–44:25, 2019.

[3] J. Farrington, K. Li, W. K. Wong, and M. Utley. Going faster to see further: GPU-accelerated value iteration and simulation for perishable inventory control using JAX. arXiv preprint arXiv:2303.10672, 2023.

[4] S. Maleki, M. Musuvathi, and T. Mytkowicz. Parallelizing dynamic programming through rank convergence. In Proceedings of the 19th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming (PPoPP’14), pages 219–232. ACM, 2014.

[5] C. Yang, A. Buluc, and J. D. Owens. Graphblast: A high-performance linear algebra-based graph framework on the GPU. ACM Transactions on Mathematical Software, 2022.

### Questions
1. You present the min-plus matrix–vector product as a core contribution. Please clarify how this differs from prior work such as [4] (DP as a tropical matvec) and existing semiring frameworks like [5] (references refer to the ones mentioned in the "Weaknesses" box

2. How does this work differ from earlier GPU-based dynamic programming methods?
(a) VRP: comparison to [1], who used GPUs for DP-based q-route and ng-route relaxations.
(b) Inventory: comparison to [3], who used GPU-based value iteration (parallel Bellman updates) for large state spaces.

3. The DSIRP speedup (9.3 × 104) looks extremely high. please clarify how the CPU baseline was implemented, in particular:
(a) What algorithm was it implementing?
(b) Why didn’t you include a CPU implementation of your own min–plus algorithm for a fair comparison?
(c) Could you also compare with a semiring library baseline (e.g., [5]) to separate hardware from algorithmic effects?

4. The framework seems to require, in the dense worst case, batched transitions of size O(|Omega|m2) with |Ω| = 106.
(a) How was this handled on an 11GB GPU? Please report the per-stage state sizes mt used in CVRPSD and DSIRP.
(b) Please include a memory-complexity analysis showing the trade-off between |Omega| and m. If large scenario counts are only possible for very small m (e.g., m < 50–60), how does that affect the claimed generality of the framework?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the scalability bottleneck of Dynamic Programming (DP) in large-scale Stochastic Programming (SP) problems. A general-purpose GPU framework is proposed, which reformulates forward DP recursions as "batched min-plus matrix-vector products" over layered Directed Acyclic Graphs (DAGs). This approach allows the computation to be mapped directly to GPU kernels, enabling massively parallel computation. Validated on two classic SP applications, the framework demonstrates the capability to process large-scale scenarios, achieving a speedup compared to traditional multi-threaded CPU baselines.

### Strengths
1. This paper addresses a critical gap by extending GPU acceleration from its mature application in continuous optimization solvers to the computationally intensive domain of SP.
2. It introduces a general-purpose GPU framework that reformulates broad classes of forward DP recursions into batched min-plus matrix-vector products. 
3.  The framework achieves a speedup compared to traditional multi-threaded CPU baselines.

### Weaknesses
1. The experimental comparisons in this paper are conducted at the operator level rather than the problem level. The core performance graph compares the authors' GPU DP kernel against their own CPU DP implementation. This constitutes a relatively weak baseline. While it demonstrates the operator's throughput, it fails to prove that the end-to-end performance is superior to a SOTA solver when this operator is integrated into a complete solution framework.
2. The paper lacks external benchmarking for solution quality. Although Figures 5 and 6 effectively illustrate internal trends, the study never compares the final solution quality found by its method against publicly available "best-known solutions" in the literature. We learn that the operator is fast, but it remains unclear whether the solutions it finds are competitive.
3. The paper lacks a dedicated "Related Work" section, making it difficult for reviewers to clearly position and contrast this work's contributions against other SOTA methods in the field.

### Questions
The acceleration at the operator level is significant. To evaluate the practical impact of this work, could the authors integrate this GPU operator into a complete solving framework and compare its end-to-end performance (in terms of total runtime vs. solution quality) against a SOTA solver on the same CVRPSD/DSIRP instances?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a novel approach to stochastic programming problems (with DP representable recourse) that is highly parallelizable on GPUs.   Computationally, the authors demonstrate the efficacy of their approach on large-scale inventory and routing problems.

### Strengths
- **Motivation**: Overall, the evaluation of candidate solutions over many scenarios in stochastic programming is computationally challenging, and the authors propose a unique new perspective that may have relatively strong applicability within the framework proposed, but also in many other settings.  
- **Efficiency**: The method is very efficient, especially for problems with large-scale scenario sets and complex second-stage problems.

### Weaknesses
- **Scope & Generality**: One major drawback of this approach is the requirement of being able to represent the second-stage problem on a GPU through a DP.  While this does capture some stochastic programming problems, it may not be as applicable as other classical/ML-based methods.  Another weakness of the process is the requirement to implement a DP formulation for each new problem.  Compared to existing stochastic programming methods that formulate second-stage optimization problems, the barrier to implementation would be significantly higher. That said, there are certainly use cases where investment in implementation would be worthwhile.   
- **Baselines**: This paper explores relatively large-scale benchmarks and demonstrates the efficiency of their approach.  However, their approach essentially acts as an evaluation on a set of candidate first-stage decisions, rather than truly optimizing over the entire decision space (what most methods do in stochastic programming). For that reason, it isn't easy to assess the quality of the solution in this case.  Comparing against established exact methods (integer L-shaped, Benders' dual decomposition) or heuristics (progressive hedging via MPI-SPPY) would help assess decision quality.  It may only be possible to get these methods working on smaller-scale instances, but it would still be highly informative.  
- **Clarity**: Some details in this work are a bit unclear.  For example, based on my understanding of the paper, they obtain a set of candidate first-stage decisions and then use GPU parallelization to evaluate them; however, it is unclear how these decisions are received and how many are evaluated.  Moreover, there is no exploration into the decision quality over a varied number of first-stage candidates, which is necessary to assess the viability of this approach.  
- **ICLR Fit**: While this paper explores GPUs, which are widely used in ML, it doesn't actually do any ML, only exact evaluations through parallel DP operations.  For that reason, there may be more suitable venues, such as CPAIOR and INFORMS Journals, etc.  That said, there is a growing interest at the intersection of stochastic programming and ML, so there is some relevance.

### Questions
- Is there any way to extend this to problems that are not representable as DPs?
- How many candidate first-stage solutions are used?
- How does the performance vary with the number of first-stage decisions?
- The word 'training' is somewhat misleading, as it implies a notion of training that is not present.  In stochastic programming, it is essentially solving over different-sized scenario sets.  
- I believe the authors may be using the ICLR 2025 template.
- The code is not anonymized.

### Soundness
2

### Presentation
2

### Contribution
2
