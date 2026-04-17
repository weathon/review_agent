# BoGrape: Bayesian optimization over graphs with shortest-path encoded

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Graph-structured data are central to many scientific and industrial applications where the goal is to optimize expensive black-box objectives defined over graph structures or node configurations---as seen in molecular design, supply chains, and sensor placement. Bayesian optimization offers a principled approach for such settings, but existing methods largely focus on functions defined over nodes of a fixed graph. Moreover, graph optimization is often approached heuristically, and it remains unclear how to systematically incorporate structural constraints into BO. To address these gaps, we build on shortest-path graph kernels to develop a principled framework for acquisition optimization over unseen graph structures and associated node attributes. Through a novel formulation based on mixed-integer programming, we enable global exploration of the combinatorial domain over graph structures and explicit embedding of problem-specific constraints. We demonstrate that our method, BoGrape, is competitive both on general synthetic benchmarks and representative molecular design case studies with application-specific constraints.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
- This paper proposes a Bayesian optimization (BO) framework for optimizing black-box functions whose inputs are graphs, which commonly arise in domains such as molecular design, supply chains, and sensor placement.
- The authors study four variants of the shortest-path graph kernel (SP, SSP, ESP, and ESSP) and integrate them within a Gaussian process surrogate model.
- The main methodological contribution is the formulation of the acquisition function optimization as a mixed-integer programming (MIP) problem, which enables global optimization using off-the-shelf MIP solvers.
- The MIP has the acquisition function (in this paper, the authors consider the lower confidence bound, LCB) as its objective and includes constraints that ensure the predictive mean and variance from the GP model are consistent with the graph kernel, as well as additional constraints enforcing valid shortest-path relationships within the graph.
- The authors theoretically prove that each feasible MIP solution corresponds bijectively to a valid connected graph in the search space.
- The proposed framework, named BoGrape, is evaluated on both synthetic graph benchmarks and real-world molecular design tasks, demonstrating competitive or superior performance compared to baseline methods.

### Strengths
- This paper proposes a method to optimize an acquisition function in BO for a graph-input framework. This area of research is under-explored and has only a limited amount of literature.
- The method proposed here, i.e., formulating the acquisition function optimization problem as a mixed-integer programming (MIP) problem, is novel in the BO literature for graph functions.
- Formulating the acquisition function as an MIP also generalizes the capability of BO for graph functions, extending it from optimizing functions defined over graph nodes to functions defined over graph structures.
- The authors also provide a fruitful discussion on how to generalize the framework to other scenarios, such as graph structures with varying but bounded sizes and other nonlinear acquisition functions.

### Weaknesses
- The presentation of the paper needs improvement. Many typos and a wrong reference in the main part cause some confusion. For example, there is no Eq. (5) in the paper, and it seems that the authors were trying to refer to Eq. (MIP-SP).
- The main limitation of this paper is the high computational cost of solving the mixed-integer programming (MIP) formulation, which may restrict scalability to large graph sizes or complex kernels. More discussion on this limitation is required. This includes considering larger problems and presenting the comparison in a more informative way, for example by showing the Pareto front between acquisition optimization runtime and BO performance.
- There is a lack of explanation on the choices of parameters used in the competitor methods. For example, for methods that sample random graphs, why did the authors consider only 20 candidate graphs in each iteration? Since random sampling is cheap, it would be nice to see an ablation study with higher numbers.
- It is unclear how feasibility constraints of valid molecules are encoded in the method; more explanation on this part is needed.
- The authors considered several kernels but did not discuss the pros and cons of each of them. This discussion would help readers choose kernels that encode the properties relevant to their applications.

### Questions
- In the introduction, it is stated that there are two scenarios of graph optimization problems: optimizing over nodes and optimizing over graph structure, and this paper considers only the second case. If the algorithm is not applicable to the first case, this point should be specifically mentioned in the abstract to avoid overclaiming the ability of the framework. If the framework can be generalized to the first scenario, a brief idea on how to do it should be discussed.
- It is difficult to distinguish between node labels and node features. An explanation through an example would be helpful.
- The notation $\beta$ is used twice: in the objective of the MIP in Eq. (1a) and in the graph kernel weights in Eq. (3). I suggest changing the notation for clarity.
- In the last two constraints of (MIP-SP), what indices are the summations over? $w$?
- Should the $\alpha$ and $\beta$ in Eq. (3) be positive?
- Line 290: superscript $I$.
- Line 411: What is NAS? Network Architecture Search?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents BoGrape, a new Bayesian optimization framework designed for black-box functions defined over graphs. The authors introduce four shortest-path-based graph kernels: SP, SSP, ESP, and ESSP; and then use them in BO with a global acquisition optimization scheme formulated as a mixed-integer optimization problem. Experiments are carried out on both synthetic and real-world scenarios, where the authors validate their proposed kernels and use them for BO on QM7 and QM9 datasets.

### Strengths
- This paper is well-written with clear logic and is very easy to follow.

- Unlike previous works on graph BO, which use evolutionary algorithms to explore the search space, this paper proposes a global optimization framework over the feasible graph space based on mixed Integer programming. This idea sounds very novel to me.

- The proposed graph kernels based on shortest-path look solid to me.

- I am happy to see that the authors compared the performance of their proposed kernels to previous graph kernels (but why are the results in the appendix?)

- I am also glad to see that the authors use baselines based on graph BO rather than simply comparing to random sampling. 

- The authors also provide further details on the computational complexity, scalability, and limitations of their proposed framework, which help the audience better understand the pros and cons of this new method.

### Weaknesses
While the authors said their method can be easily applied to other common acquisition functions, they only consider LCB in their methodology and experiments. Is it possible to extend BoGrape to UCB (this one should be straightforward) or EI?

Another limitation, as mentioned by the authors in the appendix, is the scalability issue when the underlying graph is large. The maximum size of the graphs used in the experiments is 25, but in many real-world applications (e.g., social networks and infrastructure networks), the network size may go up to $10^5$. In these cases, I am unsure whether using a global optimization based on MIP is more efficient than adopting evolutionary algorithms. 

But I don't think this is necessarily a "weakness" of the current work, since most of the graph-level optimization algorithms suffer from the scalability problem. Overall, I have a positive evaluation of this work.

### Questions
- On Figure 2, why is there no error bar on the first plot for the SSP kernel?

- Instead of comparing among your proposed kernels in Figure 2, I think it will be better to show the performance of other graph kernels in the main text (I notice that they are in Appendix C) by making each subplot a bit smaller. The reason is that the audience will be more interested in how your proposed kernels compare to previous graph kernels, rather than solely comparing among the new ones.

- A minor suggestion: it's better to briefly mention the testing accuracy of the GNNs that are used as ground-truth.

- I am a bit curious about the learned values of $\alpha$ and $\beta$ in your kernels, e.g., results in Section 4.1 Model Performance. (I think they should reflect the "importance" of graph structure and node feature in the underlying function?)

- Another minor suggestion: please consider using a larger font size for x and y labels, ticks on x and y axes, titles, and legends in Figures 2, 3, and 4.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes a graph-based Bayesian optimization that efficiently solves graph-level optimization problems using shortest-path encoding and mixed-integer programming (MIP). It establishes a bijective mapping between shortest-path MIP encoding and the set of connected graphs, thereby addressing the challenge of modeling graph structural constraints. Additionally, it introduces four variants of shortest-path kernels and a mixed kernel formulation to balance structural and feature similarity, and achieves constrained global acquisition optimization over mixed discrete–continuous spaces, with validated effectiveness on both synthetic and molecular design tasks.

### Strengths
1. This paper proposes a shortest-path-based MIP encoding strategy that establishes a strict bijection between connected graphs and the optimization constraint space.
2. It designs four positive-definite graph kernel variants and a mixed kernel formulation to adapt to different graph scales and nonlinear characteristics, integrating structural and attribute similarities.
3. It achieves global acquisition optimization within the MIP framework, explicitly modeling problem constraints and avoiding the local optima issues of traditional heuristic methods.

### Weaknesses
1. Since the computational complexity of MIP solving increases exponentially with the number of nodes, the current method only supports graphs with ≤30 nodes, making it challenging to handle large-scale graph scenarios.
2. The experiments mainly focus on undirected connected graphs, with insufficient validation of adaptability to directed and disconnected graphs, and a lack of related experiments, as well as detailed discussion and verification.

### Questions
1. The current MIP encoding only supports undirected connected graphs with no more than 30 nodes, and its scalability to large graphs has not been fully verified. Additionally, the proposed extensions for directed and disconnected graphs lack empirical support. It is recommended that the authors include corresponding validation experiments to enhance the method's generality and practicality.
2. The current experimental benchmarks do not include domain-specific Bayesian optimization methods, such as those for molecular design. The authors are encouraged to incorporate domain-relevant baselines to more comprehensively evaluate the performance and practical applicability of the proposed approach.

### Soundness
3

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
This paper introduces BoGrape, a novel framework for Bayesian optimization (BO) over graph-structured domains. The key idea is to encode shortest-path relations in a mixed-integer programming (MIP) formulation, enabling global optimization of acquisition functions over combinatorial graph spaces while incorporating explicit structural constraints. The method is evaluated on synthetic benchmarks and molecular design tasks, showing superior performance to several baselines including random search, evolutionary algorithms, and existing graph BO methods.

### Strengths
1.The proposed approach is novel. The paper presents the first unified framework that formulates BO directly over graph structures. The use of MIP for global acquisition optimization is technically innovative and allows explicit incorporation of hard structural constraints, which is critical for applications such as molecular design.

2.The authors provide formal proofs ensuring the feasible domain of the formulation is equivalent to the graph space consisting of all connected graphs. This gives BoGrape a solid theoretical grounding compared to heuristic approaches.

3.The paper is generally well-written, with a clear motivation and logical flow from problem definition to methodology and results.

### Weaknesses
Overall, the main issues of the paper lie in two aspects.

Scalability concerns:
The proposed MIP formulation scales poorly with the number of nodes O(n3) variables due to shortest-path encoding. The largest experiments involve small graphs (n < 20), limiting its practical applicability. See questions 1 and 2.

Limited baselines:
Comparisons mainly include random, evolutionary, and one BO-based baseline.
Missing comparisons with recent neural graph optimization methods. see question 3.

### Questions
1. How would BoGrape perform on larger graphs (e.g., n>50)?

2. In line 52, the paper states that existing methods are “incapable of efficiently exploring the search domain.”However, based on later sections, the proposed method does not appear to demonstrate clear improvements in efficiency. 
Could the authors provide quantitative comparisons of training and inference time costs for all methods across different tasks to substantiate this claim?

3. Why were recent neural graph optimization methods not included for a more comprehensive and up-to-date comparison? For instance, in the molecular experiments (Section 4.3), it is recommended that the authors extend the comparison beyond Bayesian optimization and evolutionary algorithms to include recent domain-specific neural graph optimization methods. Incorporating these approaches would provide a more comprehensive and up-to-date evaluation and better demonstrate the practical potential of the proposed method in real-world applications.

### Soundness
2

### Presentation
3

### Contribution
3
