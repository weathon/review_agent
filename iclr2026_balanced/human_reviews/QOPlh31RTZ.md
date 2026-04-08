## Human Reviewer 1

### Summary
This paper proposes CB-RLHF, a bi-level optimization framework for solving constrained RLHF problems.  
1. The upper level learns reward and cost functions from human feedback, while the lower level solves a dual-convex reformulation of the constrained RL problem using the Lagrangian dual.  
2. The method employs Clarke subdifferential and gradient approximation to handle non-differentiable hypergradients and proves a convergence rate of O(1/√K) and an approximation error of O(1/√N).  
3. The algorithm is compared against PEBBLE, Safe-RLHF, and PARL on four MuJoCo tasks, showing that CB-RLHF improves constraint satisfaction and return performance.

### Strengths
1. The theoretical formulation appears rigorous and well-grounded, even though I didn't fully verify all the proof details.  
2. The paper provides a complete methodological pipeline—from problem formulation to convergence theory and experiments—with a coherent narrative.  
3. The combination of theory and empirical validation makes the proposed framework conceptually convincing.

### Weaknesses
1. The method critically relies on strong duality to convert a non-convex constrained problem into a convex one, yet the paper does not discuss or verify the duality gap.  
   In practical RLHF settings involving approximation and sampling, zero duality gap is unlikely to hold. The authors should analyze or empirically estimate the duality gap’s impact on convergence and correctness, or at least clarify conditions under which strong duality is justified.  

2. The bi-level structure is theoretically elegant but computationally heavy. Frequent estimation of λ*(ϕ, ψ) and inner-loop optimization make the algorithm expensive.  
   Its feasibility for large-scale RLHF—especially in LLM fine-tuning scenarios—remains untested. The paper would benefit from demonstrating CB-RLHF on a small-scale language model (e.g., 1.5B parameters) to establish practical viability.  

3. The experimental design mixes reward and cost by defining return = reward − cost, which is inconsistent with the problem formulation that treats reward and cost separately.  
   Because reward and cost are in different units and the dual algorithm is scale-insensitive (replacing λ, c with αλ, c/α does not dramatically change results), this modification obscures interpretability. It would be more principled to report cumulative reward and constraint violation separately.  

4. Experimental fairness is questionable. The total training steps vary across the four environments, and results seem sensitive to stopping criteria.  
   For instance, in Walker2D, using 1e5 steps (like HalfCheetah and Swimmer) may allow PARL to outperform CB-RLHF; conversely, reducing HalfCheetah to 0.6e5 steps (like Hopper) makes CB-RLHF weaker in both return and constraint violation. This suggests possible cherry-picking of stopping points.  

5. The proposed framework effectively assumes a Bradley–Terry preference model. As such, its applicability is limited to preference-based feedback rather than general RLHF (which can involve scalar rewards, rankings, or textual feedback).  
   The title should more accurately read “A Constrained Bi-level Framework for Preference-based Human Feedback” rather than “RLHF,” which implies broader generality.  

6. Minor issue: in line 212, z₀ and z₁ should be s₀ and s₁.

### Questions
1. How is the number of inner-loop dual updates tₖ determined? Is it adaptive, fixed, or tuned per environment?  
2. Can the authors provide theoretical guarantees or empirical analysis under small (non-zero) duality gaps? What happens if strong duality fails slightly—does convergence degrade gracefully?

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
2

### Confidence
3

---

## Human Reviewer 2

### Summary
This paper studies the problem of jointly learning a reward function, a cost function, and a policy from human feedback. The authors formulate the problem as a constrained bi-level optimization problem, where the upper level infers the reward and cost functions from feedback, while the lower level optimizes a policy to best align with that feedback. To solve this problem, the authors propose a double-loop algorithm, Constrained Bi-level Optimization for Reinforcement Learning from Human Feedback (CB-RLHF), which solves the lower-level optimization problem in the inner loop and the upper-level optimization problem in the outer loop. The authors establish a theoretical guarantee that CB-RLHF converges at a rate of $O(1/\sqrt{K})$, and demonstrate its effectiveness across multiple simulation environments.

### Strengths
1. The studied problem, constrained reinforcement learning from human feedback (RLHF), is well-motivated and finds important applications in large language model alignment.
2. The idea of formulating the constrained RLHF problem as a constrained bi-level optimization problem is interesting.
3. The authors propose a double-loop algorithm and provide a theoretical guarantee on convergence.

### Weaknesses
1. The writing and readability of this paper needs to be improved.
2. In Eq. (1), what is the motivation of using $H(\pi)$ as a regularization term, instead of the KL divergence with the reference policy?
3. It is hard to understand the theoretical results provided in this paper. (i) The abstract of this paper mentioned that they prove that algorithm CB-RLHF converges at a rate of $O(1/\sqrt{K})$. However, Theorem 1 only states that the gradient is bounded by $O(1/\sqrt{K})$. How does this result imply the convergence rate to the globally optimal policy?  
(ii) In Theorem 2, what are the definitions of SubOptR and SubOptC, in particular, what is the definition of human policy $\pi_h$? Does Theorem 2 provide the performance gap between the optimal policy and the output policy of algorithm CB-RLHF? If it is, why is it not dependent on $K$?  
4. The constrained LLM alignment problem has been studied in several prior works. The authors should discuss more on the advantages of the proposed constrained bi-level optimization approach compared to the existing primal-dual approaches, e.g., Safe RLHF [Dai et al., 2023].  
5. This paper only provides experiments on MuJoCo. It would enhance this paper if the authors can provide experiments on LLMs.

### Questions
Please see the weaknesses above.

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
4

### Confidence
3

---

## Human Reviewer 3

### Summary
This paper proposes a bi-level optimization framework for RLHF. The upper-level optimization learns both a reward and a cost function from preference feedback, while the lower-level optimization solves the dual of a constrained RL problem. To handle the non-smoothness of the objective, the authors employ the Clarke subdifferential framework to approximate the hypergradient and provide convergence and sample complexity analyses. Empirical results demonstrate that the proposed algorithm outperforms three baselines across four environments.

### Strengths
- The integration of a bi-level optimization framework into the constrained RLHF setting is well-motivated and appears novel.
- Although I did not check the proofs in full detail, the proposed method for addressing the non-convex lower-level optimization problem seems mathematically sound and appropriate.

### Weaknesses
- Some notations, particularly those related to the cost function, are unclear (see questions below). I also recommend the authors carefully proofread the Appendix. Several proofs are difficult to follow and contain typos, which hinder readability.
- The experimental results are not sufficiently comprehensive to support the main claims. Including more environments or additional ablation studies would strengthen the empirical evidence.

### Questions
- My understanding is that a high value of $J_c (\tau)$ indicates a trajectory with a high cumulative cost (i.e., a worse trajectory). Then, In line 197, why does the BT model seem to prefer trajectories with higher costs? Furthermore, in Theorem 2, why is the negative cost of the learned policy upper-bounded?
- In lines 441–444, when generating the synthetic feedback data, is a greedy model used instead of the BT model? If so, could the authors clarify the rationale?
- Minor typos: (i) In line 401, $\epsilon$ should be $\epsilon_k$. (ii) In line 416, $(\pi_)$ should be $(\pi_h)$.

### Soundness
2

### Presentation
2

### Contribution
3

### Rating
4

### Confidence
3

---

## Human Reviewer 4

### Summary
This paper addresses the problem of learning from human feedback in reinforcement learning (RL) scenarios that involve constraints. The authors formulate this problem as a constrained bi-level optimization task. In this formulation, the upper-level problem learns reward and cost functions from human feedback, while the lower-level problem solves for the optimal policy under those given functions.
To solve this, the paper introduces an algorithm called CB-RLHF. The algorithm handles the non-convexity of the lower-level problem by using its dual formulation and addresses potential non-differentiability using a Clarke subdifferential approximation. The authors provide a theoretical convergence guarantee and present pseudo experiments on MuJoCo environments to show that the method can address both misalignment and constraint inference limitations.

### Strengths
- Important Problem Formulation: The paper tackles the important and practical problem of learning from human feedback while simultaneously respecting safety or ethical constraints. The formulation of constrained bi-level optimization problem might be a notable contribution.
- Theoretical Grounding: The authors make a serious attempt to provide a theoretical foundation for their algorithm. They provide proofs for the convergence of their method, which adds rigor to their claims.

### Weaknesses
Insufficient Experimental Validation: The experiments are not sufficient to support the paper's claims about RLHF.
- Unrealistic Oracle: The experiments use a synthetic oracle based on ground-truth functions, which bypasses the core challenges of noisy, ambiguous, and costly feedback from real humans.
- Missing Key Baseline: The paper fails to compare against the most obvious baseline: a standard, iterative RLHF approach where the update cycle is simply run much more frequently. It is unclear if the proposed complex optimization is better than just iterating faster.
- Lack of Ablation Study: The method introduces two key components (a cost function and a bi-level framework) but provides no ablation study to disentangle their effects. We cannot know what is truly responsible for the performance changes.
- Lack of Analysis: The paper presents results but offers almost no analysis. It claims success even when the method underperforms baselines on some metrics (e.g., Hopper return), with no discussion as to why.

Lack of Clarity in Theoretical Justification: The lack of clarity makes the core justification for the method confusing.

### Questions
Could the authors clarify how strong duality (Lemma 1) is guaranteed to hold for the problems?

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
4

### Confidence
3