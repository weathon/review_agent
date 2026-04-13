## Human Reviewer 1

### Summary
This paper proposes a new way to solve restless bandit: putting both the state information and the action information into a continuous vector space, and do linear programming to maximize the reward in the next $\tau$ rounds. When we look forward $\tau$ steps with a large enough $\tau$, this algorithm can find near optimal solutions. The authors show that normally the suboptimality gap is $O(\sqrt{1/N})$, but with certain conditions, it can be reduced to be $O(\exp(-cN))$, where $N$ is the number of arms and $c$ is some constant. As for the gap corresponding to $\tau$, they also show it is about $O({1\over \tau})$. In experiments, it is shown that choosing $\tau$ as a small value (e.g., 10) leads to very good performance, indicating the efficiency of model predictive control.

### Strengths
- The analysis shows a good suboptimality gap.

- The algorithm is quite efficient to implement.

- The experiment results are also convincing.

### Weaknesses
- The author assumes that all the arms are statistically identical. Is this a common assumption in restless bandits? I think there are many cases that the arms are not identical. It seems that your algorithm cannot be adapted to this setting easily (e.g., if all the arms have distinct transitions and rewards)?

### Questions
- Are there any comparison on the running time of different algorithms, and your algorithm with different $\tau$? 

- Are there any experiment on real-data?

- In which part of the proof shows that $\tau(\epsilon) = O({1\over \epsilon})$?


=========After Rebuttal=========

Thanks for your reply, I do not have other questions.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
3

---

## Human Reviewer 2

### Summary
This paper addresses the infinite-horizon average reward RMAB problem. It proposed a model predictive control (MPC) policy using a rolling computational horizon of length $\tau$, which achieves a suboptimality gap with order $\mathcal{O}(1/\sqrt{N})$ with $N$ being the number of arms.

### Strengths
The paper presents a novel use of dissipativity for analyzing RMABs, which offers fresh insights into this field. The suboptimality bounds are rigorously derived, showcasing that the MPC-based policy approaches optimality as the number of arms, 𝑁, increases.

### Weaknesses
Though this paper conveys a solid algorithm design and theoretical proof, we must admit that the current submission has significant weaknesses.  

1. My main concern is that the order of $\mathcal{O}(1/\sqrt{N})$ gap has been a well-known result for a long time, which various types of algorithms can achieve. In particular, the LP-based algorithms in many related works cited by this submission without an additional MPC layer can also achieve this optimality gap.  Though the reviewer admits that the proposed MPC layer can bring some advantages, the key reason that this reviewer does not champion this paper is that there is a new work (https://arxiv.org/pdf/2410.15003) recently has pushed the gap to the order of $\mathcal{O}(1/N)$ by using a diffusion approximation technique. I understand that the authors submitted their work earlier than this recent work and may not be aware of this new work, the main concern always holds as the proposed MPC-based algorithm does not improve the well-known optimality gap.

2.  Resolving the LP at each time step is not really a novel idea, which can be found in multiple works in related work cited by this submission. Though controlling $\tau$ is new, resolving LP at each time step causes a significant amount of computational complexity.  Aligned with my first concern, why do we need such a complex algorithm that does not even improve the theoretical guarantee, which is claimed as a main contribution in this work?

3. In many real-world applications that can be modeled as an RMAB framework, the underlying MDP for each arm may not be known in advance. My question is how can we leverage the proposed MPC-based algorithm for such a setting? 

4. This paper considers a homogeneous arm setting. The reviewer quite doubts the scalability of the proposed algorithm when arms are heterogeneous and the number of arms is very large.

### Questions
Please refer to **Weaknesses**.

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
5

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper studied the discrete time infinite horizon average reward restless markovian bandit (RMAB) problem and focused on the asymptotic optimality in this setting. In particular, a model predictive control (MPC) based non-stationary policy with a rolling computational horizon $\tau$ is proposed and its sub-optimality gap is presented. The performance of this policy is also evaluated via simulations.

### Strengths
- RMAB has been extensively studied in recent years, from both the offline and online settings. This paper investigates the fundamental property of RMAB (e.g., the asymptotic optimality, optimality gap) in the offline setting. It is a challenging and interesting problem. 
- A model predictive control (MPC) based non-stationary policy has been developed and theoretically analyzed, i.e., its optimality gap is characterized in terms of the number of arms $N$. 
- Some experimental results were presented to validate the performance of this MPC based policy and its comparison with baselines.

### Weaknesses
- The asymptotic optimality performance of RMAB has raised many attentions in recent years. Despite that this paper proposed a policy by leveraging the MPC idea, it is hard to identify the technical novelty from the perspectives of algorithm design and technical proofs and results (See questions below). This paper heavily relies on the previous works such as Gast et al. 2023 a,b. 
- The simulations are rather weak in both the settings and the baseline methods considered. 
- This paper in general is poorly written with many typos, broken sentences and abused notations.

### Questions
- On one hand, the LP based relaxation has been extensively used in the RMAB literature, such as Verloop 2016, Zhang and Frazier 2022. On the other hand, the randomized rounding procedure is almost the same as that in Gast et al. 2023a and a finite-horizon MPC algorithm (LP-update policy) was proposed in Gast et al. 2023a,b. It is more like a straightforward extension. From the algorithmic perspective, can you more explicitly state what you see as the key novel aspects of the MPC based algorithm compared to prior work, particularly Gast et al. 2023a,b.? 
- The first result in Section 4.1. This is not surprising and it is commonly known in the RMAB literature that LP-based method for RMAB is provably asymptotically optimal. Indeed, the LP based method has been leveraged to design index policy for RMAB problem without the hard-to-verify indexability condition as required by the Whittle index policy, and such a LP-based method to design index policy is provably asymptotically optimal as shown in the literature, e.g., Verloop 2016 is one of the first works in this domain.  Can you discuss how your result in Section 4.1 advances the state-of-the-art beyond what was already known from works like Verloop 2016? Are there any aspects of your analysis or bounds that are novel or improved? 
- The second result in Section 4.2. Likewise, the proof is directly from Gast et al. 2023a and Hong et al. 2024a. For both results, can your clarify exactly how the use of dissipativity differs from or improves upon previous approaches? can you discuss any limitations of previous methods that your approach overcomes? 
- In practice, how to determine how many arms to pull at each time, given that $\alpha N$ may not be an integer? If it is not an integer (may consider as an average constraint, there is no need to design an index policy). 
-Equation (3) itself is not a RMAB problem. It should be properly defined with the budget constraint to be satisfied at teach time. It may be better to rigorously define (3)  
- Many typos in the paper, just to name a few here: line 120, "the budget constraint, $\alpha$", line 146, $\boldsymbol{x}_s$ is not defined. line 144, since $u(s,a)$ is denoted as the empirical distribution of the state-action pairs $(s,1)$, why not just express it as $u(s,1)$ since the action is fixed.
- In Section 3.1 after introducing the optimization problem in (8), the authors claimed that this problem is computationally easy to solve. On one hand, indeed this is a linear problem and it is "relatively" easy to solve when the state space and the number of arms is small, given many LP solvers. On the other hand, this claim is not precise, since solving a LP with large-scale parameters/spaces can still be very computationally expensive and take a lot of time in practice. This may be one of the limitations of LP based method for RMAB compared to the Whittle index policy although LP based methods do not require the indexability condition. 
- Can you elaborate why the definition of (8) imposes the constraint to be satisfied for each time $t$ as claimed in lines 198-199?

### Soundness
2

### Presentation
1

### Contribution
2

### Rating
3

### Confidence
4

---

## Human Reviewer 4

### Summary
The paper addresses the discrete time infinite horizon average reward Restless Markovian Bandit (RMAB) problem with a Model Predictive Control (MPC) approach. The proposed MPC algorithm achieve the suboptimality gap $O(1/\sqrt{N})$ with a minimal set of assumptions, and can achieve exponential convergence under local stability conditions. Moreover, the MPC algorithm works well in practice with SOTA computational complexity.

### Strengths
1.	The paper introduces a novel application of MPC to the RMAB problem which achieve suboptimal gap $O(1/\sqrt{N})$ with a minimal set of assumptions and exponential convergence rate under local stability conditions.
2.	The proposed MPC approach reduces the computational burden associated with solving RMAB problems and perform well in numerical experiments.
3.	This paper presents an interesting framework based on dissipativity, and provides theoretical analysis in this paper.

### Weaknesses
1.	In Section 6, the algorithm LP-priority is not formally introduced. It is confusing to distinguish between the LP-update and LP-priority algorithms due to the lack of a clear definition.
2.	What are the main technical contributions of the theoretical analysis compared to existing works? I suggest highlighting the novelty and primary contributions of the theoretical analysis more clearly.
3.	In Line 88, the paper states, "It performs well both in terms of the number of arms N as well as the computational time horizon T, beating state-of-the-art algorithms in our benchmarks." However, in the numerical experiments section, the authors did not compare the computational efficiency of the proposed MPC approach with existing algorithms. Moreover, it would be beneficial to provide a more rigorous discussion on why the MPC approach reduces the computational burden.

### Questions
1.	In Section 6, the algorithm LP-priority is not formally introduced. It is confusing to distinguish between the LP-update and LP-priority algorithms due to the lack of a clear definition.
2.	What are the main technical contributions of the theoretical analysis compared to existing works? I suggest highlighting the novelty and primary contributions of the theoretical analysis more clearly.
3.	In Line 88, the paper states, "It performs well both in terms of the number of arms N as well as the computational time horizon T, beating state-of-the-art algorithms in our benchmarks." However, in the numerical experiments section, the authors did not compare the computational efficiency of the proposed MPC approach with existing algorithms. Moreover, it would be beneficial to provide a more rigorous discussion on why the MPC approach reduces the computational burden.

### Soundness
4

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
3