# Near-Optimal Sample Complexity Bounds for Constrained Average-Reward MDPs

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Recent advances have significantly improved our understanding of the sample complexity of learning in average-reward Markov decision processes (AMDPs) under the generative model. However, much less is known about the constrained average-reward MDP (CAMDP), where policies must satisfy long-run average constraints. In this work, we address this gap by studying the sample complexity of learning an $\epsilon$-optimal policy in CAMDPs under a generative model. We propose a model-based algorithm that operates under two settings: (i) relaxed feasibility, which allows small constraint violations, and (ii) strict feasibility, where the output policy satisfies the constraint. 
We show that our algorithm achieves sample complexities of $\tilde{O}\left(\frac{S A (B+H)}{ \epsilon^2}\right)$ and $\tilde{O} \left(\frac{S A (B+H)}{\epsilon^2 \zeta^2} \right)$ under the relaxed and strict feasibility settings, respectively. Here, $\zeta$ is the Slater constant indicating the size of the feasible region, $H$ is the span bound of the bias function, and $B$ is the transient time bound. Moreover, a matching lower bound of $\tilde{\Omega}\left(\frac{S A (B+H)}{ \epsilon^2\zeta^2}\right)$ for the strict feasibility case is established, thus providing the first minimax-optimal bounds for CAMDPs. Our results close the theoretical gap in understanding the complexity of constrained average-reward MDPs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduce a minimax-optimal primal-dual method for constrained AMDPs under the generative setting.

### Strengths
The analysis is solid and close the gap in the current literature of constrained average-reward MDPs. The approach used makes sense, converting to a MDP and update the dual variable, following the standard primal-dual setup. it is impressive to see that the proposed algorithm match the lower bound, which is also presented in this paper. Overall I find this work substantial and the paper is nicely written.

### Weaknesses
1. As this paper proposes to use a generative model, a natural question would be, how feasible is it to extend to the markovian model setting? Oftentimes, e.g. in robotics, we cannot sample arbitrary state-action pairs even given a simulator. Thus the generative model assumption does not hold here. 
2. Although the approach of reducing the AMDP to DMDP with Blackwell-optimality appears in other works, I wonder if one can solve the AMDP directly. Experience shows that large discount factor pose instability to the algorithm, limiting the applicability of the proposed framework. As such, the contribution of the work is limited to algorithm analysis. 
3. I am not familiar with the field of constrained MDPs so I encourage the authors to present related work's results, even in discounted setting (as this work use some sort of discounted setup) to give a fair comparison.
4. I find the use of model-based approach failing to attract my interest, as real life problems are unlikely be handled through world model learning. In addition, for large state or action spaces, function approximation seems to be unavoidable, in which this work does not address either. 
Nevertheless, I think this is solid work, for which I lean positively.

### Questions
See above.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper provides sample complexity bounds on learning constrained average-reward MDPs under the generative model setting. The authors propose a primal-dual framework that works for the relaxed feasibility and strict feasibility regimes. The algorithm achieves near-optimal sample complexity in both settings. Furthermore, the paper characterizes a matching lower bound for the strict feasibility setting that incorporates the Slater parameter. This shows that the sample complexity upper bound for the setting is (nearly) minimax optimal.

### Strengths
- The theoretical contributions, if correct, are clearly significant. Learning infinite-horizon average-reward constrained MDPs seems a harder problem than learning CMDPs in the finite-horizon setting or the discounted-reward setting. The lower bound that incorporates Slater's constant is novel, and it is very interesting that the lower bound matches the complexity upper bound.
- The proofs are based on novel ideas, not repeating existing analysis techniques. Their arguments are carefully prepared based on the theory of average-reward MDPs.

### Weaknesses
I have two concerns that, I hope, can be addressed or clarified by the authors. Let us discuss them as below.
- In each iteration, the algorithm requires solving an unconstrained average-reward MDP. Theoretically, as claimed in the paper, the part can be dealt with by a black-box MDP solver such as a linear programming-based method. However, it seems to important to incorporate errors incurred from the black-box MDP solver and to discuss its complexity. 
- I think that the most subtle part is about arguing that the average-reward of approximate MDPs is bounded. However, I had hard time understanding the proof of Lemma 17. It is not clear why the bias function of an approximate MDP is bounded. Moreover, the average-reward of an approximate MDP remains constant for all states? I also have the following more specific questions about the proof. 

&ensp; Why is it that (Lemma 8.6.4, Puterman 1994) implies $|\rho^\star - \hat \rho^\star|_\infty\leq O(\varepsilon)$?

&ensp; Why does $|| \hat h||_\infty\leq H$ hold?

### Questions
I have listed my questions in the weakness part above.

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
4

### Summary
This paper establishes the first minimax-optimal sample complexity bounds for learning in Constrained Average-Reward MDPs (CAMDPs) under a generative model. The authors propose a model-based primal-dual algorithm that operates under both relaxed feasibility (allowing $\varepsilon$ constraint violation) and strict feasibility (zero violation) settings. The main results show sample complexities of $\tilde{O}(SA(B+H)/\varepsilon^2)$ for relaxed feasibility and $\tilde{O}(SA(B+H)/(\varepsilon^2\zeta^2))$ for strict feasibility, where $\zeta$ is the Slater constant. The paper also provides matching lower bounds, establishing the first minimax-optimal characterization for CAMDPs.

The paper is technically sound with rigorous theoretical analysis:
Strengths:
⦁	The primal-dual algorithm design is well-motivated and properly analyzed
⦁	The proof technique cleverly reduces AMDP concentration to DMDP concentration via Lemma 11
⦁	Both upper and lower bounds are carefully constructed with matching dependencies on key parameters
⦁	The treatment of relaxed vs. strict feasibility is thorough and reveals fundamental differences ($1/\zeta^2$ factor)
Minor concerns:
⦁	The assumption that reward functions $r$ and $c$ are known (lines 184-186) simplifies the problem, though the authors correctly note this doesn't affect leading-order complexity
⦁	Some proof steps rely heavily on prior work (e.g., Vaswani et al. 2022 for discounted concentration), which is acceptable but limits novelty in techniques

Finally, I am familiar with the relevant literature on constrained MDPs and average-reward RL. I carefully checked the main proof techniques and they appear sound, though I did not verify every detail in the appendix. There is a possibility I may have missed some subtle technical issues in the lengthy appendix proofs.

### Strengths
Presentation: The paper is generally well-written and organized:
Strengths:
⦁	Clear problem formulation with helpful notation
⦁	Good use of proof sketches in main text with details deferred to appendix
⦁	Figures 1-3 effectively illustrate the hard instance constructions
⦁	The progression from methodology to upper bounds to lower bounds is logical
Areas for improvement:
⦁	The paper is dense with heavy notation that could benefit from a notation table
⦁	Algorithm 1 could be more clearly presented with clearer separation of initialization vs. iteration
⦁	The connection between discount factor $\gamma$ and the transient time/bias span could be explained more intuitively before diving into technical details
⦁	Some key quantities (e.g., the perturbation $\omega$) appear in Algorithm 1 before being properly motivated

Contributions: This is a solid theoretical contribution to constrained RL:
Originality:
⦁	First work to establish sample complexity for CAMDPs in the average-reward setting
⦁	Novel lower bound constructions specific to the constrained setting
⦁	The separation between relaxed and strict feasibility via the $\zeta^2$ factor is new and insightful
Significance:
⦁	Closes an important theoretical gap between unconstrained AMDPs and constrained settings
⦁	The Slater constant $\zeta$ appears naturally and its role is well-characterized
⦁	Results unify understanding across different MDP formulations (finite-horizon, discounted, average-reward)
Limitations:
⦁	Primarily theoretical with no empirical validation
⦁	The gap between relaxed and strict feasibility (factor of $\zeta^2$) is proven tight, but practical implications are unclear
⦁	Limited discussion of when the bounded transient time assumption holds in practice

Strength:
1.	Theoretical completeness: First work to provide matching upper and lower bounds for CAMDPs, establishing minimax optimality with respect to all key parameters ($S$, $A$, $B$, $H$, $\varepsilon$, $\zeta$)
2.	Technical rigor: The proofs are careful and detailed, with novel techniques for handling constraints in the average-reward setting
3.	**Clear problem formulation**: The paper precisely defines two distinct feasibility settings and shows fundamental computational differences between them
4.	Comprehensive treatment: Both weakly communicating and general (multichain) MDPs are addressed with appropriate lower bounds for each class

### Weaknesses
1.	No empirical validation: The paper is purely theoretical with no experiments demonstrating the algorithm's practical performance or validating the constants hidden in $\tilde{O}$ notation
2.	Known rewards assumption: While not affecting asymptotic rates, assuming known $r$ and $c$ is somewhat restrictive and simplifies the learning problem
3.	Limited practical guidance:
⦁	No discussion of how to estimate unknown parameters ($B$, $H$, $\zeta$) in practice
⦁	No computational complexity analysis of the algorithm
⦁	Unclear how tight the constants are in practice
4.	Presentation density: The paper packs substantial technical content that may be challenging for readers not deeply familiar with average-reward MDPs and concentration inequalities
5.	Scope limitations:
⦁	Only single constraint case studied (extension to multiple constraints not discussed)
⦁	No discussion of function approximation or large state spaces
⦁	The gap between the generative model setting and online/model-free learning remains unaddressed. It would be valuable to discuss how these theoretical insights might extend to practical applications involving resource-constrained online decision-making, such as LLM inference scheduling under memory constraints [1] or dynamic pricing with capacity limitations [2]
6.	Minor technical gaps:
⦁	The assumption $\varepsilon \in (0, 1/(1-\gamma)]$ appears in theorems without clear justification of why this range is natural
⦁	The relationship between the empirical bias span $\hat{H}$ and true $H$ could be made more explicit.

References
[1] Ao, R., Luo, G., Simchi-Levi, D., & Wang, X. Optimizing LLM Inference: Fluid-Guided Online Scheduling with Memory Constraints. arXiv preprint arXiv:2504.11320.
[2] Ao, R., Jiang, J., & Simchi-Levi, D. Learning to Price with Resource Constraints: From Full Information to Machine-Learned Prices. arXiv preprint arXiv:2501.14155.

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies the sample complexity of learning in constrained average-reward Markov decision processes (CAMDPs) under the generative model setting. The goal is to determine how many samples are required to compute an \varepsilon-optimal policy that satisfies a long-run average constraint.

Formally, each policy \pi has a steady-state average reward \rho^\pi_r(s) and a constraint value \rho^\pi_c(s), and the learner aims to solve
\max_{\pi}\;\rho^\pi_r(s)\quad \text{s.t.}\quad \rho^\pi_c(s)\ge b.
The analysis distinguishes two regimes: relaxed feasibility (where the constraint can be violated by at most \varepsilon) and strict feasibility (where it must be satisfied exactly).

The authors propose a model-based primal–dual algorithm that iteratively solves empirical unconstrained MDPs of the form r + \lambda c while updating the dual variable \lambda using projected stochastic gradient descent. This construction allows them to derive both upper and lower sample-complexity bounds.

The main results can be summarized as follows:
	•	Relaxed feasibility:
N = \tilde{O}\!\left(\frac{SA(B+H)}{\varepsilon^2}\right)
samples suffice to compute an \varepsilon-optimal policy.
	•	Strict feasibility:
N = \tilde{O}\!\left(\frac{SA(B+H)}{\varepsilon^2\zeta^2}\right),
where \zeta is the Slater constant, measuring the size of the feasible region.
	•	Matching lower bound:
The authors prove \tilde{\Omega}(SA(B+H)/(\varepsilon^2\zeta^2)) for strict feasibility, giving the first minimax-optimal characterization of sample complexity in CAMDPs.

Here S and A denote the number of states and actions, H the span of the bias function, and B a bound on transient time. The results extend prior analyses for (i) unconstrained average-reward MDPs (Zurek & Chen, '24) and (ii) discounted constrained MDPs (Vaswani et al., '22).

The proof builds on Lagrangian duality, strong duality for average-reward MDPs, and confidence bounds for the bias function estimated from the generative model. A Fano-style construction is used to obtain the lower bound. No experiments are provided; this is a purely theoretical paper.

### Strengths
1.	Technically solid.
The analysis is mathematically careful and connects several strands of recent work — unconstrained average-reward sample complexity, constrained discounted MDPs, and primal–dual analysis — into one coherent framework.
	2.	Closes a theoretical gap.
The paper provides, for the first time, tight upper and lower bounds for CAMDPs. This completes the theoretical landscape for sample complexity under the generative model.
	3.	Clear separation between regimes.
The dependence on the Slater constant \zeta elegantly captures how strict feasibility increases sample complexity, mirroring known results in convex optimization and constrained RL.
	4.	Methodologically sound.
The primal–dual approach is principled and aligns well with the structure of constrained reinforcement learning.

### Weaknesses
1.	Difficult presentation.
The exposition is unnecessarily dense. Key symbols are introduced before being defined (\tilde{c}, M’, \hat{M}), and the algorithmic steps (e.g., the reason for reward perturbations or the \epsilon-net projection for \lambda) are not well motivated. The reader has to reconstruct much of the reasoning from context.
	2.	Incremental novelty.
The core ideas extend known results from discounted CMDPs and unconstrained AMDPs rather than introducing new analytical techniques. The adaptation to the average-reward case is nontrivial but conceptually straightforward.
	3.	Limited intuition.
The paper would benefit from more discussion of how the constants H, B, and \zeta influence learning difficulty, or how they compare to analogous quantities in the discounted setting. At present, the results are formal but not deeply interpretable.
	4.	Limited to no empirical or illustrative validation.
Even a small numerical experiment illustrating the scaling with \zeta or comparing relaxed and strict feasibility would have improved readability and intuition.
	5.	Accessibility.
The technical writing assumes significant familiarity with the literature on bias-span bounds and average-reward MDPs. Without this background, it’s very hard to follow.

### Questions
1.	Novelty relative to Vaswani et al. (2022) and Zurek & Chen (2024).
	•	What is genuinely new in the analysis beyond extending those frameworks to the average-reward case?
	•	Are there any technical hurdles specific to the average-reward setting (e.g., lack of contraction) that required new ideas?
	2.	Understanding the constants.
	•	Can you provide intuition for how the bias-span H, transient time B, and Slater constant \zeta control the complexity?
	•	In practice, how might one estimate or bound these quantities?
	3.	Algorithm design.
	•	Why is reward perturbation needed in the primal–dual algorithm?
	•	How sensitive is the performance or convergence to the step size and to the discretization of the dual variable?
	4.	Broader applicability.
	•	Can your results be extended to settings without a generative model — e.g., online learning or policy-gradient approaches?
	•	Would similar rates hold if only sample trajectories were available?
	5.	Clarity and structure.
	•	Consider reorganizing the exposition so that the notation is introduced before use, and include a brief intuitive overview of the proof strategy. This would substantially improve readability.

### Soundness
4

### Presentation
2

### Contribution
3
