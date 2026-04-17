# Adversarial examples for heuristics in combinatorial optimization: An LLM based approach

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 6, 2

## Abstract
This work employs LLMs to generate adversarial examples for heuristics in combinatorial optimization.
The problem, given a heuristic for an optimization problem, is to generate a problem instance where the heuristic performs poorly. We find improved adversarial constructions for well-known heuristics for k-median clustering, bin packing, the knapsack problem, and a generalization of Lov\'asz's gasoline problem. Specifically, we adapt the FunSearch framework [Romera-Paredes et al., Nature 2023] to obtain adversarial constructions for these problems. We note that using FunSearch is crucial to our improved constructions --- local search does not give comparable results. 
The advantage of FunSearch is that it produces structured instances that yield theoretical insights which are post-processed and generalized by a human researcher while other metaheuristics usually produce only unstructured instances that are harder to generalize.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper applies FunSearch (LLM-based evolutionary program search) to generate adversarial instances for combinatorial optimization heuristics. The authors investigate four problems: Nemhauser-Ullmann knapsack heuristic, Best-Fit bin packing, k-median hierarchical clustering, and iterative rounding for the gasoline problem. They propose "Co-FunSearch" combining automated generation with manual refinement.

### Strengths
1. Rigorous mathematical follow-up: Unlike pure black-box approaches, the authors provide formal proofs for most claims (Theorems 3.1-3.4).

2. Honest reporting: The limitations section (5.4) honestly reports failures.

3. Diverse problem selection: Four different combinatorial optimization domains demonstrate some generality.

### Weaknesses
1. The paper doesn't establish why finding adversarial instances via LLM is an important research problem. The paper doesn't demonstrate that these adversarial instances have any practical implications. Are these contrived constructions or do they reveal fundamental algorithmic weaknesses?

2. Results may already be known or trivial: The bin packing instance is suspiciously simple - the authors should check whether this construction appears implicitly in prior work.

3. Gasoline extrapolation from Table 2 to general claims lacks justification. Need either formal proof of scaling or results for larger instances.

4. "Co-" prefix admits heavy expert refinement needed, contradicting automated discovery narrative. Contribution breakdown between LLM and human unclear.

### Questions
Given that (1) heavy human expert refinement is required ("Co-"), (2) success rate is ~50% or lower, (3) several results have questionable novelty/correctness, and (4) practical impact is unclear - can you articulate a clear value proposition for why the community should pursue LLM-based adversarial instance generation over traditional mathematical analysis? What specific advantages does this approach offer that justify its costs and limitations?

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
2

### Summary
This paper considers the problem of finding hard instances for a fixed algorithm for a fixed problem. The studied problems are all NP-hard, namely Knapsack, Bin packing, k-median in hierarchical clustering, and Lovasz's gazoline problem. Specific approximation algorithms are considered for each of these problems. The goal is to use artificial intelligence to generate instances which maximize the approximation ratio of the considered algorithms. Instead of generating the instance itself, a compact description is generated using a large language model artificial intelligence. The prompt asks the AI to come up with a worse instance, and provides a first python program generating an instance.

I find this approach quite crazy, in my own humble experience, lower bound instances have to be constructed by hand, and computers turned out to be bad in generating hard instances. But investigating the power of LLMs for this task is an interesting approach.

### Strengths
The paper succeeds to find an instance which breaks a conjecture from 2024 about a specific algorithm for the gazoline problem.  For other problems the LLM gave the authors ideas to improve the generated instance. So this work shows that LLM could help in algorithm design. I don't know much about LLMs, so I don't have the right background to evaluate the paper. But I am into algorithm design, and hence any new method that could help is welcomed.

### Weaknesses
Overall I am still hungry after reading the paper. From what I have seen in the algorithm community, is that most hard instances consists of some fractions, to which some epsilon values have been added or removed. Also I think the approach could benefit from the design of a grammar describing compactly instances, rather than generating a Python program, because the search space is more targeted to a form of aimed instances.

### Questions
You don't give the evaluation function to the LLM. Explain this choice.

Line 372, explain what each dimension means. I guess first is weight, and second is profit.
Several times you mention a choice of temperature, which I only know in the context of simulated annealing. Maybe explain.

### Soundness
2

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
3

### Summary
The paper proposes a method for finding adversarial instances for combinatorial optimization heuristics based on FunSearch. They use LLMs for creating problem instances in a form of Python programs where given heuristic does not perform well. The authors argue that this representation is more interpretable than less structured instance vectors obtained for example by local search. The paper focuses on heuristics for four combinatorial optimization problems including the knapsack problem, bin packing, hierarchical clustering and a variant of the gasoline problem. The authors find tighter lower bounds than the ones that were known before for these problems using their method of Co-FunSearch.

### Strengths
1. Originality 

The paper is based upon existing FunSearch technique but it introduces novel elements such as including experts in the search pipeline in their Co-FunSearch variant. They also apply FunSearch to the combinatorial optimization domain. The authors also find novel theoretical results for the considered problems which seem to be not have been known before this work. 

2. Quality

The paper presents numerous experiments and tests proposed methodology for four important problems thus providing a solid experimental base for their results.

3. Clarity

The paper is well-written and easy to follow.

4. Significance

Combinatorial optimization heuristics have important practical and theoretical meaning. Constructing adversarial examples for them provides additional insights into the nature of the problems for which they were created. Using an interpretable representation of such adversarial examples such as Python-code can be useful for further analysis of these examples and constructing novel heuristics which would be more robust.

### Weaknesses
1. (Significance) Co-FunSearch relies on collaboration with human experts which might be a bottleneck for large-scale applications of the method. As the authors state (lines 076-077), expert modifications were essential for generating meaningful insights. Thus, the methodology can be used as a supporting tool but not as a standalone process for generating adversarial examples. 

2. (Clarity) Even though local search is known in the field of combinatorial optimization, could you specify in more detail how exactly you perform it since it is one of the baselines in your experiments.

### Questions
1. Can Co-FunSearch be made fully autonomous by using agents for refining found programs (lines 076-077, Figure 1c, 3c)?

2. The authors argue that solutions found by their method are more interpretable and symmetric when compared to solutions found by local search (lines 066-067). It would be interesting to see whether they are also more robust to noise. For example, if we take a vector provided by local search such as the one in the footnote 1 (lines 106-107) and add some small noise to its element, will it maintain it's properties as an adversarial example for a given heuristic? What if we do the same to solutions provided by your method?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper extends FunSearch to generate adversarial examples for heuristics for combinatorial optimization. Specifically, the authors look at knapsack, bin packing, k-median clustering, and the gasoline problem. The value of the task is generating practical samples that are hard to approximate for a certain solver, closing the gap to the worst-case upper bound.

### Strengths
* Using LLM to tackle math and optimization problems is a trending research topic. 
* The proposed Co-FunSearch framework seems to outperform FunSearch in the experimental evaluations.
* An ablation study is provided in the experiment section.

### Weaknesses
* While the idea of identifying worst-case instances for existing heuristics is conceptually interesting, it is unclear whether this direction achieves the same level of impact or technical novelty as improving the heuristics themselves — as demonstrated, for example, in FunSearch and its case studies.  

* Significant revisions in writing and presentation are necessary before the technical contributions of this work can be properly evaluated.  
    * The introductory paragraph discussing AI advancements in biology, chemistry, and mathematics is tangential to the main topic and should be removed to improve focus.  
    * A substantial portion of the paper is devoted to describing problem definitions, heuristics, and code snippets, yet key implementation details of the proposed method are missing. It remains unclear how Co-FunSearch differs from FunSearch, or how it can be concretely instantiated with a given heuristic. The authors should include algorithmic diagrams or pseudo-code in a dedicated “Method” section to clarify these aspects.  
    * If the authors wish to elaborate on problem formulations, additional context and explanations are needed for readers unfamiliar with the examples. For instance, the so-called “famous gasoline problem” attributed to Lovász is insufficiently explained; as presented, it is not interpretable and unlikely to be “famous” to the broader audience.  
    * In line 214, the authors state:  
      > “The main goal in all these problems is to search for a vector v which optimizes the given objective.”  
      
      This statement lacks precision. It is unclear  
        1. what the “vector v” represents and how it fits within the different search paradigms described, and  
        2. what the “given objective” is, i.e., which function or metric is actually being optimized.

### Questions
* In the footnote on page 2, 
    > For instance, one of the local-search-generated lists outperforming FunSearch was: [0.003031, 0.005466,
0.006098, 0.007283, 0.021158, 0.068030, 0.073417, 0.170490, 0.202092, 0.219287, 0.306771, 0.375912,
0.540358].
   
    It is unclear how this is relevant to a "discernible pattern" and please provide explanations. 
* To compute the approximate ratio, the optimal value is required. How do you know the optimal value if the problem parameter is always changing? Is the scalability of this framework bottlenecked by solving for the optimal value?
* How many problem instances are used to calculate Table 1? How are they collected? How do you determine the size of the problem instance?
* In Table 1, what does "Previous Best Lower Bound" mean? 
* In Table 1, what is the "Best-Fit" problem?

### Soundness
2

### Presentation
2

### Contribution
2
