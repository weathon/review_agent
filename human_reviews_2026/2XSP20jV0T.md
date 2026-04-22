# Solving General-Utility Markov Decision Processes in the Single-Trial Regime with Online Planning

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 8, 4, 4

## Abstract
In this work, we contribute the first approach to solve infinite-horizon discounted general-utility Markov decision processes (GUMDPs) in the single-trial regime, i.e., when the agent's performance is evaluated based on a single trajectory. First, we provide some fundamental results regarding policy optimization in the single-trial regime, investigating which class of policies suffices for optimality, casting our problem as a particular MDP that is equivalent to our original problem, as well as studying the computational hardness of policy optimization in the single-trial regime. Second, we show how we can leverage online planning techniques, in particular a Monte-Carlo tree search algorithm, to solve GUMDPs in the single-trial regime. Third, we provide experimental results showcasing the superior performance of our approach in comparison to relevant baselines.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper investigates the setting of discounted infinite-horizon general-utility MDPs under the single-trial regime, and proposes a set of theoretical results on the policy class needed to represent the optimal solution, the complexity of achieving such optimum and a methodology to cast one such problem into an equivalent, finite horizon, standard MDP. Finally, an online planning method based on MCTS is proposed, achieving better performances than existing solutions on GUMDP formulation of existing problems.

### Strengths
The investigated problem is interesting: standard MDPs are known for being limited in what class of utilities they can represent, and the GUMDP framework allows us to overcome such limitation. However, this setting is much less investigated, and new solution techniques are welcomed. The paper is very clear and rigorous in how it guides the reader through its contributions, always being clear in drawing relevant connections with previously defined concepts and results. As a non-expert in general-utility MDPs, I found it very easy and clear to read. Also, the derived results are sound, and can serve as a good basis for future research in the direction of the single-trial regime. Theory builds up the fundamental pieces until the practical methodology based on MCTS is provided in a clear way.

### Weaknesses
I do not have any particular weakness to highlight. The only criticism that can be moved to the current version of the paper is the limited size of the experiment (an aspect the authors have been earnest upfront): the current small experiments are not really telling us much on how the proposed framework would scale to larger setting which are now common in RL (even though the results on the Gym environments are going in this direction a bit). However, I do not feel this is an invalidating weakness, as I appreciate that the nature of the work is mainly theoretical, and the proposed empirical results are still convincing enough to draw interest on this methodology anyway.

### Questions
- I think that, at the end of page 2, the symbolism for non-deterministic stationary policies $\Pi^D_S$ and deterministic stationary policies $\Pi_S$ are swapped?

- On line 112, what is $\mathcal{F}$? It is not defined anywhere in the following.

- "*However, the compressed representation used by the occupancy MDP is more amenable to practical implementations since the running occupancy can be incrementally updated as the agent interacts with its environment.*" What do you mean precisely here? That, in practical implementations, you can maintain the histories and incrementally update it with the new "pieces"? Why this could not be done with full-length histories as well in pretty much the same way?

- You say that discounting plays a significant role in your proposed formulation, and reference Proposition 2 as one of those. However, while I do agree that indeed your formulation relies on the fact that you are addressing the infinite-horizon discounted setting, I do not see where discounting is playing a role in such proposition. While, conversely, this is very clear in Proposition 1, that allows you to resort to truncated histories at the price of a constant term (depending on the discount factor $\gamma$) in terms of regret.

- Line 428: GUMDPS $\rightarrow$ GUMDPs

- Results of $\pi_{\text{Solver}}^*$ are a bit surprising: this is using an exact solver in the infinite-trial regime, and as such I would have expected to find a better solution than the method based on MCTS under the single-trial regime (in principle, the solver should be able to always achieve the optimal solution, given enough computational time and resources). Is there any limitation imposed on such solver, like a limited computational time? Could you please discuss on the gap in performance between the optimal solver and your proposed method a bit more?

### Soundness
4

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors propose a novel method for solving infinite-horizon discounted general utility MDPs in the single-trial regime. This includes some analysis showing how online planning techniques can be used to solve general utility MDPs, as well as empirical results which show that the proposed method yields superior performance when compared to prior methods.

### Strengths
The writing quality and presentation of this paper is outstanding. In particular, the analysis performed is well-motivated and presented in a clear and easy-to-read manner. The empirical results are encouraging.

### Weaknesses
I have no major concerns with this paper, and it is more or less in a publishable state.

Here are some minor concerns/suggestions for the authors:

- The notation related to the occupancy measure is a bit confusing. In particular, the notation of d_pi in equation 1 vs d_^pi in equation 3 onwards makes it unclear how these two terms are related. 
- Lines 173-174: the claim that 'practical applications often require identifying a policy that performs optimally when evaluated based on a single trajectory' is not supported. In particular, the authors need to formally prove/motivate this, or cite a prior work that does so.
- Theorem 1: The use of cost instead of reward may confuse the reader here. In particular, most readers may incorrectly interpret F_1(pi_s) > F_1(pi_m) as indicating that F_1(pi_s) is better than F_1(pi_m) (since most readers are familiar with the reward-based formulation). I would suggest adding some form of clarification in the theorem to avoid this potential issue (or perhaps removing the > altogether and instead only use words).
- I strongly encourage that the authors use the extra page of content allowed during/after the reviews to include more plots related to the empirical results (such as the ones in Appendix C).

### Questions
- Did the authors consider whether there are any assumptions needed to ensure that the occupancy measure is well-defined (i.e, that it exists and is unique)? For instance, some MDP-based methods require that the induced Markov chain / MDP has ergodic properties.

### Soundness
4

### Presentation
4

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
The paper aims to solve infinite-horizon discounted general-utility Markov decision processes (GUMDPs) in the single-trial (trajectory) evaluation setting. In particular this means that they consider a (non-linear) function $f$ of the empirical discounted occupancy of one trajectory $\mathbb{E}[f(d^{\pi})]$, compared to previous work that considered the expected discounted occupancy $f(\mathbb{E}[d^{\pi}])$. The intuition is that if we consider a non-linear objective and apply standard RL techniques, there is a gap between training and evaluation, since the objective it is being trained on is an upper bound of the true objective.  

They achieve this by proving three theorems: (i) non-Markovian policies are necessary for optimality in the single-trial objective; and (ii) reduction of the single-trial problem and solution to a finite-horizon occupancy MDP; and finally (iii) they show the NP-hardness of the policy optimization in the specified regime.

The authors propose to solve the occupancy MDP with MCTS, which guarantees convergence in the infinite iteration regime. Finally the method is tested on illustrative toy and OpenAI Gym environments, showing improvement over baseline random and infinite-trial solver policies. All the proofs and experiments are conducted for discrete state and action spaces.

### Strengths
The paper clearly delineates the single-trial from the infinite-trial formulation and establishes a solid method to reformulate the GUMDP as an occupancy MDP that can be solved with well established planning methods. The paper shows clear improvement over previous methods, albeit on simple domains. Overall, the authors identify a subtle problem when considering alternate MDPs with non-linear objectives.

### Weaknesses
It is not entirely clear what the main theoretical contribution of the work is, since the single-trial setting is a special case of the multiple-trial setting from Santos et, al. ICML 2025. 

Empirically, the method was only shown for very small state and action spaces where even a random policy achieves reasonable performance.  It should ideally be further tested on more complicated and larger environments, and particularly one a specific use case where single-trial evaluation is required by its nature (e.g. stock market predictions, robotics). 

The use cases of single-trial evaluation would exclude the existence of a (reliable) simulation. This is a serious limitation of the proposed method that relies on a planner.

### Questions
- What is the core difference between GUMDPs considered in this work and Convex MDP? It seems there is no difference if theoretical results follow a convex $f$. 
- The random policy already shows decent performance for some of the environments. Could you include harder environments to show the strength of your method?
- Explain the size of the uncertainty in table 1. They seem to vary significantly between environments and policy, the $pi_{MCTS}$ Taxi has even 0 uncertainty, indicating that the uncertainty is most likely underestimated.
- Consider rephrasing lines192-206 on the “Single-trial formulation for GUMDPs” to a list for easier readability. 
- Consider using a “hat” the empirical observables to distinguish them clearly from the expected values.
- Mention that you work with discrete state action spaces in the introduction already.
- Introduce examples of use cases of single-trial evaluations.

### Soundness
2

### Presentation
1

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
This paper tackles single-trial optimization in infinite-horizon discounted general-utility MDPs (GUMDPs), where the objective $f$ is possibly a non-linear function of the discounted state-action occupancies. The paper establishes some fundamental results for the problem. It formally proves that non-Markovian policies can strictly outperform stationary and Markovian policies for the single-trial regime. Then the paper introduces the occupancy MDP (OMDP), which is a finite-horizon MDP whose states augment environment states with the running discounted occupancy $o$. The authors establish a one-to-one mapping between histories and OMDP states and characterize equivalence. Based on this connection, they show that solving the OMDP reformulation with stationary policies corresponds to solving the original problem with non-Markovian policies. Despite this, it turns out that policy optimization with smooth convex utility functions is NP-hard.

### Strengths
The paper provides some fundamental theoretical results comparing stationary, Markovian, and non-Markovian policies for the single-trial regime. Moreover, the paper characterizes the connection between the original GUMDPs and their truncation to finite-horizon MDPs in terms of regret. The paper also proves that the GUMDPs with non-Markovian policies are equivalent to occupancy MDPs with stationary policies. These theoretical findings are novel and provide interesting insights for the single-trial regime of general utility-based MDP formulations.

### Weaknesses
Although the paper empirically tests the computational performance of the MCTS-based algorithmic framework presented in this paper, it lacks its theoretical analysis. The NP-hardness of computing an optimal policy should not prevent us from deriving a finite convergence guarantee to an optimal policy ($1/\epsilon$ versus $\mathrm{log}(1/\epsilon)$). However, there is neither a regret bound nor a convergence guarantee.

### Questions
As the authors delineated the connection to occupancy MDPS with stationary policies, would it be possible to design an algorithm with a provable performance guarantee, such as regret bounds and convergence to an optimal policy?

### Soundness
3

### Presentation
3

### Contribution
2
