# Sampling On Metric Graphs

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 8, 4, 4

## Abstract
Metric graphs are structures obtained by associating edges in a standard graph with segments of the real line and gluing these segments at the vertices of the graph.
  The resulting structure has a natural metric that allows for the study of differential operators and stochastic processes on the graph.
  Brownian motions in these domains have been extensively studied theoretically using their generators.
  However, less work has been done on practical algorithms for simulating these processes.
  We introduce the first algorithm for simulating Brownian motions on metric graphs through a timestep splitting Euler-Maruyama-based discretization of their corresponding stochastic differential equation.
  By applying this scheme to Langevin diffusions on metric graphs, we also obtain the first algorithm for sampling on metric graphs.
  We provide theoretical guarantees on the number of timestep splittings required for the algorithm to converge to the underlying stochastic process.
  We also show that the exit probabilities of the simulated particle converge to the vertex-edge jump probabilities of the underlying stochastic differential equation as the timestep goes to zero.
  Finally, since this method is highly parallelizable, we provide fast, memory-aware implementations of our algorithm in the form of custom CUDA kernels that are up to ~8000x faster than a GPU implementation using PyTorch on simple star metric graphs.
  Beyond simple star graphs, we benchmark our algorithm on a real cortical vascular network extracted from a DuMuX tissue-perfusion model for tracer transport.
  Our algorithm is able to run stable simulations with timesteps significantly larger than the stable limit of the finite volume method used in DuMuX while also achieving speedups of up to ~1500x.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper touches on a very interesting problem with solid research dealing with differential equations posed on metric graphs. Nevertheless, I feel that ICLR by its core definition is the wrong venue for this research and it is better suited for numerical analysis journal/UQ journal.

### Strengths
I think the paper is well written and deals with an interesting problem but I do not see it as being particularly focused on learning algorithms and doubt its suitability for ICLR.

### Weaknesses
I feel that the poor performance of the FVM needs to be discussed and these methods usually perform very well on metric graphs but have, of course, other weaknesses. The formatting of all the references is terrible please fix the bib-entries.

### Questions
- Please explain how changing the orientation of an edge does not change the inward derivative?
- In (1) and (2) the subscript e is not explained. It is clearly the restriction onto the edge but please be precise.
- I am a bit suspicious about the FVM scheme as it seems to completely fail for the Fokker Planck equation, any reasons why? There are existing FVM schemes for quantum graphs to be found on Github that seem to do well for other PDEs.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper tackles sampling on metric graphs by introducing a time-step–splitting Euler–Maruyama (EM) scheme to simulate Brownian motion, and hence Langevin diffusions, on metric graphs. It proves that the number of vertex crossings is finite with high probability and that exit probabilities of the simulation converge to the SDE's vertex-edge jump probabilities as the step size tends to zero. The authors also present highly parallel, memory-aware implementations, and experiments show the method outperforms a finite-volume baseline in accuracy and speed.

### Strengths
1. The paper introduces a novel and interesting time-step–splitting EM scheme for Brownian/Langevin on metric graphs, with guarantees of finite splits (w.h.p.) and exit-probability convergence as step size goes to 0. 

2. This method is efficient and scalable with memory-aware, highly parallel GPU implementation that showing strong speedups and accuracy gains over a finite-volume baseline with empirical validation

### Weaknesses
Results are interesting but restricted to star graphs and applicability to general metric graphs is neither analyzed nor empirically validated.

Beyond finite splits and exit-probability consistency, there are no non-asymptotic weak/strong error bounds or sampling error rates.

Minors:

- "Sampling On Metric Graphs" should be "Sampling on Metric Graphs"

- Line 405 the normalizing constant B if given by -> the normalizing constant B is given by

### Questions
Q1: What's the non-asymptotic convergence rate?

Q2: What are the challenges for non-star graphs, and can you show experiments on non-star graphs?

### Soundness
2

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
3

### Summary
This paper investigated the practical implementation of sampling in metric graphs. It seems that the authors generalise the Euler-Maruyama discretization of Langevin diffusion for continuous distribution. They provide a theoretical guarantee that the jump distribution generated by their proposed algorithm asymptotically converges to the target distribution. Then they provide a parallelizable version of the proposed algorithm to get fast and memory-saving implementation.

### Strengths
1. The problem investigated in this paper seems to be novel, having theoretical and practical value.

2. This paper has a certain mathematical depth.

### Weaknesses
1. There are some claims that they did not explain clearly. For example, in Line 133-134, they claimed that the results of the star graphs researched in this paper can extend to general graphs. They did not explain how to extend.

2. The presentation should be improved. The key section “Brownian Motion on Metric Graphs” should be more detailed, especially on the boundary conditions, which is not friendly to the readers.

### Questions
The theoretical results seems to be valid for all target distribution $b_v$, which means they did not need to make any assumption on $b_v$. This is different from Unadjusted Langevin Algorithm for sampling in continuous distribution, which usually make assumptions on smoothness and isoperimetry properties of potential function.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper focuses on the numerical simulation problem of Brownian motion and Langevin diffusions on metric graphs and proposes a timestep splitting Euler-Maruyama-based discretization method.
It handles vertex crossings by recursively splitting the simulation step, moving the particle to the vertex, and sampling a new edge. Theoretical analysis prove the algorithm terminates finitely and converges to correct vertex-edge jump probabilities as the timestep decreases.
A custom, memory-aware CUDA kernel is implemented for fast, parallelized execution on GPUs. 
Numerical experiments on a 5-edge star graph with linear and quadratic potentials demonstrated the effectiveness of the proposed method. 
The reported results show significant speedups and higher accuracy in recovering steady-state densities compared to FVM.

### Strengths
1. Though the theory of function space and Brownian motion on metric graphs is well-established, practical simulation algorithm is non-existent. This paper provides a concrete and practical method to simulate this process, which is a novel contribution to the field and will be beneficial for future work in this field. 

2. Theoretical analysis is thorough and insightful. Theorem 2 & 3 addresses the concern of an infinite loop due to repeated vertex crossings within a single timestep, which is crucial for the practicality of the algorithm.  Corollary 1 links the algorithm's behavior to the underlying SDE, proving that the simulated jump probabilities converge to the correct theoretical values $b_v$. These theoretical analysis provide a solid ground for the proposed method.

3. The CUDA kernel implementation achieves a massive speedup over a simple Pytorch implementation. This engineering effort significantly elevates the paper's utility for practitioners and researchers needing large-scale simulations.

### Weaknesses
1. Experimental Evaluation is limited critically. The entire numerical evaluation is conducted on a synthetic star graph with only 5 edges. Metric graphs are powerful precisely for modeling networks with complex cycles, multiple vertices, and varied edge lengths. Demonstrating performance only on a star graph provides almost no evidence that the algorithm works on metric graphs in general. Besides, this paper does not demonstrate the effectiveness of algorithm on a real-world problem or dataset, which undermines its potential impact. The performance gap versus the FVM baseline, while impressive, is less meaningful without a real-world context.

2. While Theorem 2 guarantees finite runtime, empirical analysis of the computation cost introduced by vertex crossings is missed. How does the average number of splits M scale with $\delta t$, the drift magnitude, and the graph complexity? This is an important practical consideration that is left unexplored.

### Questions
1. Can you provide theoretical or experimental analysis to show that Algorithm 1 works effectively on a non-star metric graph, for instance, a graph containing a cycle or multiple interconnected vertices?

2. Can you provide experimental results on a concrete real-world dataset?

3. This paper focuses exclusively on standard boundary conditions, which can be further improved by discussing limitations of the current algorithm regarding these more general conditions or outlined a path for future extension.

### Soundness
3

### Presentation
2

### Contribution
2
