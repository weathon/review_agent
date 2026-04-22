# LLM-Guided Evolutionary Program Synthesis for Quasi-Monte Carlo Design

- Avg Score: 4.80
- Decision: Accept (Poster)
- Scores: 2, 4, 8, 2, 8

## Abstract
Low-discrepancy point sets and digital sequences underpin quasi-Monte Carlo (QMC) methods for high-dimensional integration. We cast two long-standing QMC design problems as program synthesis and solve them with an LLM-guided evolutionary loop that mutates and selects code under task-specific fitness: (i) constructing finite 2D/3D point sets with low star discrepancy, and (ii) choosing Sobol’ direction numbers that minimize randomized quasi-Monte Carlo (rQMC) error on downstream integrands. Our two-phase procedure combines constructive code proposals with iterative numerical refinement. On finite sets, we rediscover known optima in small 2D cases and set new best-known 2D benchmarks for N ≥ 40, while matching known 3D optima up to the proven frontier (N ≤ 8) and reporting improved 3D benchmarks beyond. On digital sequences, evolving Sobol' parameters yields consistent reductions in rQMC mean-squared error for several 32-dimensional option-pricing tasks relative to widely used Joe–Kuo parameters, while preserving extensibility to any sample size and compatibility with standard randomizations. Taken together, the results demonstrate that LLM-driven evolutionary program synthesis can automate the discovery of high-quality QMC constructions, recovering classical designs where they are optimal and improving them where finite-N structure matters. Data and code are available at https://github.com/hockeyguy123/openevolve-star-discrepancy.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a new methodology to find low-discrepancy point sets, by using an evolutionary algorithm where the mutation phase is done by an LLM. Two settings are inspected: in the first setting, the goal is to minimize the star discrepancy of a point set in 2D and 3D. Here the generation space is the space of Python programs. In the second setting, the goal is to minimize the integration error for a 32D option-pricing task. Here the generation space is the set of parameters (so-called direction numbers) for Sobol digital nets. In both cases, the solution found by the proposed method outperforms baselines.

### Strengths
Guiding program synthesis through LLMs is a thriving avenue of research, which has shown progress in various fields. I appreciate that this paper brings this idea to the problem of low-discrepancy point set generation. The experiments show an improvement over the best known star-discrepancy sets in 2D and 3D, and improve over Joe and Kuo direction numbers in the Asian option pricing experiment.

### Weaknesses
My main concern is the lack of scientific evaluation, especially given that the proposed method is highly resource-intensive, involving thousands of LLM calls, each requiring computing the fitness of the point set. Such evaluation is critical to assess the proposed approach. Examples of lacking scientific evaluation include:
- comparing with simpler methods for exploring the space of low-discrepancy sequences. I discuss this point below for each experimental setting.
- reporting the improvement in terms of fitness during the run of the evolutionary algorithm.
- comparing with simple (single- or multi-turn) LLM prompting without population-based evolution.

For the star-discrepancy experiment, a natural baseline is local optimization from an existing high-quality construction. Since the two-phase prompting strategy already instructs the LLM to “use scipy optimization routines such as scipy.optimize.minimize,” it would be appropriate to compare against standard optimizers (e.g., L-BFGS-B, SLSQP) initialized from Clément et al. or other RQMC sequences, using a comparable computational budget (e.g., 2000 randomizations).

For the Asian call experiment, the LLM modifies direction numbers in dimensions 4–6 relative to the Joe and Kuo numbers. Since the search space over these dimensions has size 2^12, comparable to the 2000 LLM calls reported, it is unclear if the proposed algorithm provides an advantage over random search. Including such comparison (and comparing to exhaustive search to get an indication of how far away the solution found by the LLM is from the optimum) would strengthen the analysis. Another baseline is the LatNetBuilder software, which is specifically designed for identifying good low-discrepancy point sets, including Sobol sequences. An interesting approach would be to rank direction numbers in terms of some fitness measure (e.g., star-discrepancy t-value) using the software, then to evaluate them in this order on the Asian call experiment, and to compare the performance with the LLM-guided evolutionary approach, as a function of the number of tested direction numbers.

### Questions
How many different direction numbers were evaluated when running the evolutionary algorithm on the option pricing problem? Is it 2000 as reported in Appendix B.3?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces an LLM-guided evolutionary framework to automate the discovery of quasi-Monte Carlo (QMC) constructions. The authors treat the design of low-discrepancy point sets and Sobol’ sequence direction numbers as a program synthesis problem. Within the proposed OpenEvolve framework, an LLM iteratively mutates Python programs that generate candidate point sets or Sobol’ parameters, guided by fitness scores (e.g., inverse star discrepancy or rQMC mean squared error).

### Strengths
1. Casting QMC design as program synthesis is conceptually elegant and connects symbolic LLM reasoning with continuous numerical optimization.

2. The paper carefully combines constructive heuristics with iterative optimization, and the experimental evaluation uses statistically robust paired tests

### Weaknesses
1. The “LLM-guided” aspect is somewhat opaque. How much improvement stems from the LLM’s structured code editing versus brute-force evolutionary search or the built-in optimizer (SLSQP)?

2. This problem setting seems too narrow, and i am not an expert in this domain so i do not understand the significance of this improvement. 

3. Are these compared baselines the best baseline in this field? Is there some other learning-based baselines? '


4. In my view, it seems the authors only carefully design a prompting procedure to solve a specfic problem. Not sure these design strategies can be used for other problems. 

5. The scale N/d is kind of limited, and we do not know whether this method can scale. In my understanding, LLM can be quite hard to scale for high-dimensional numbers.

### Questions
See Weakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper applies LLM to perform numerical integration with an evolutionary computation angle to improve the integration results.
Overall a nice paper.

### Strengths
Well written paper, 
The application of LLM with an evolutionary loop to generate populations is interesting
The method has an evolving loop of introducing mutations to provide better solutions.

### Weaknesses
It appears that the approach provides better solution with N becomes larger and larger. However, the improvement is marginal 
Over SOBOL and clement et. Al in the sense of MSE. While this marginal improvement is great, the computational complexity of 
Running an evolutionary loop is something unknown right now.

### Questions
- P values to the order of 10^-14 indicate a  population with very little variance, is there any reason for such lack of variance?

- I think, this is a very interesting application, However, one needs to understand and study the computational complexity of getting such solutions. Moreover, I get that there are about 3 applications studied in this paper, I would like to know that generality of this approach, do you have to fine tune the LLM in some way.

- However, about the mutation, does the LLM involve some kind of RL based training to generate better mutations.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper applies an LLM-guided evolutionary program synthesis framework (OpenEvolve) to two problems in Quasi-Monte Carlo (QMC) design: 

**1**. constructing finite 2D/3D point sets with low star discrepancy 


**2**. optimizing Sobol' direction numbers to reduce integration error for high-dimensional financial models. 

The authors report finding new state-of-the-art 2D point sets for $N \ge 40$ and discovering new Sobol' parameters that outperform the standard Joe-Kuo parameters on a suite of financial option pricing tasks.

### Strengths
**1**. The paper successfully refined the Joe&Kuo's result, finding new Sobol's direction numbers that result in a statistically significant reduction in integration error (MSE) for a suite of 32-dimensional financial integration tasks;

**2**. The paper introduces the current LLM tools for solving a long-standing discrete optimization problem;

**3**. The paper is well-written, clearly organized and does a good job of introducing the complex technical details of QMC.

### Weaknesses
**Major Weaknesses:**


**1**.Minimal Contribution in Point Set Discovery:  The paper's "two-phase" strategy for point sets is a significant weakness. As shown in Figure 1, the LLM's "direct construction" in Phase 1 provides almost no improvement (0.0962 $\rightarrow$ 0.0924). The entire significant gain (0.0924 $\rightarrow$ 0.0744) comes from Phase 2, which is just the LLM generating code to call a standard classical optimizer (scipy.optimize.minimize). If this is the case, then it is not a novel discovery by the LLM; it's the automation of a standard workflow any human researcher would perform.

**2**. Lack of Methodological Novelty: The core evolutionary framework is a direct application of the pre-existing OpenEvolve. The paper does not propose any new methodological innovations to this framework. Its contribution is limited to applying this existing tool to the QMC domain.

**3**. Highly Incremental Sobol' Results: While statistically significant, the improvement in the Sobol' task is extremely small in absolute terms (e.g., an MSE of 4.52e-05 vs. 4.10e-05). Crucially, the search was initialized with the strong Joe & Kuo (2008) parameters. This frames the discovery not as a major breakthrough, but as a minor, local refinement of an existing solution.


**Minor Weakness:**

**1**. The authors acknowledge that they performed only one evolutionary run per problem. Since evolutionary algorithms are stochastic, this single run provides good new parameters but tells us nothing about the robustness or reliability of the search method.



If there's any misunderstanding, I would be more than happy to correct my opinions.

### Questions
I have two main questions, which are based on the weaknesses above:

**1.** Corresponding to Major Weakness 1, how could the authors justify that the improvement in the point-set problem comes from the LLM-guided approach, rather than just the application of scipy.optimize?

**2.** For the Sobol' discovery, how much credit belongs to the LLM's "intelligent search" versus the fact that it started with the Joe & Kuo parameters as its initialization? Have the authors tried running the search from a random initialization to see if it can discover good parameters from scratch?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper applies OpenEvolve, a Large Language Model (LLM)-guided evolutionary program synthesis based on AlphaEvolve, to improve low-discrepancy sequences. Specifically, they apply this methodology to (1) construct finite 2D and 3D point sets with minimal star discrepancy (leading to small integration error); (2) optimising the direction numbers of Sobol’s sequence to reduce high-dimensional integration error 
The LLM-guided evolutionary approach generates and mutates code that produces candidate point sets or sequence parameters, guided by a fitness function that measures, respectively, star discrepancy or integration error. 
Specifically, in 2D, for a fixed number of points $N \leq 10$ the method recovered previously known optimal point sets and found new point sets with lower star discrepancy than previously discovered for $N > 30$ (0.0150 at $N=100$ vs the prior 0.0188).
For 3D sequences, the method matched all known optima up to N=7, and found new record-low discrepancies for N>8, for which optima are unknown. 
Finally, for Sobol sequences in 32 dimensions, the evolved direction parameters reduced the mean squared error of randomised QMC integration on an Asian option pricing task compared to the standard Joe-Kuo parameters.

### Strengths
* The paper applies a powerful emerging method (LLM-guided evolutionary program synthesis) to a well-suited, long-standing problem in a creative manner.  This is very significant in two ways. Firstly, it is an interesting application of LLM-guided evolutionary methods, which are establishing themselves as an important tool in scientific discovery. Secondly, discovering new low-discrepancy point sets and sequence parameters is significant for the QMC and numerical methods community.  

* The paper is clearly organised, provides an adequate amount of background, and contains extensive experimental results to back up its claims. The authors compare their results against a thorough set of baselines, including traditional sequences (Halton, Sobol, etc), simple lattice heuristics, and even more recent GNN-based MPMC methods. 
While the Sobol sequence parameters were optimised with respect to the integration error of Asian option pricing, the authors evaluate their performance on multiple other exotic option types, showing that the solution generalises beyond this specific problem.

### Weaknesses
* Presentation:  line 267 has a missing reference. In addition, line 097 would benefit from a citation when referring to the “total variation of $f$ in the sense of Hardy and Krause”. 

* The paper would benefit from more details on the LLM settings (which LLM is used?) and, perhaps, comparisons across multiple LLMs or LLM ensembles. I would expect that different LLMs would lead to different programs and solutions of different quality/optimality.

* A comparison or discussion justifying the choice of OpenEvolve over other LLM-guided evolutionary methods (such as ShinkaEvolve) would be interesting. 

* Finally, one limitation compared to the previously established baseline Sobol parameters by Joe-Kuo is the solution’s potential specialisation for the options pricing problem.

### Questions
* Justify the choice of OpenEvolve instead of other LLM-guided evolutionary algorithms

*  Which LLM was used, what inference settings, and why?

### Soundness
4

### Presentation
3

### Contribution
3
