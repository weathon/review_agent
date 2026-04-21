# Sample Efficient Multiple-policy Evaluation in Reinforcement Learning

- Avg Score: 4.00
- Decision: Reject
- Scores: 3, 5, 5, 3

## Abstract
We study the multiple-policy evaluation problem where we are given a set of $K$ policies and the goal is to evaluate their performance (expected total reward over a fixed horizon) to an accuracy $\epsilon$ with probability at least $1-\delta$. We propose a sample-efficient algorithm named \CAESAR for this problem. Our approach is based on computing an approximate optimal offline sampling distribution and using the data sampled from it to perform the simultaneous estimation of the policy values. \CAESAR has two phases. In the first we produce coarse estimates of the visitation distributions of the target policies at a low order sample complexity rate that scales with $\tilde{O}(\frac{1}{\epsilon})$. In the second phase, we approximate the optimal offline sampling distribution and compute the importance weighting ratios for all target policies by minimizing a step-wise quadratic loss function inspired by the DualDICE \citep{nachum2019dualdice} objective. Up to low order and logarithmic terms \CAESAR achieves a sample complexity $\tilde{O}\left(\frac{H^4}{\epsilon^2}\sum_{h=1}^H\min_{\mu_h}\max_{k\in[K]}\sum_{s,a}\frac{(d_h^{\pi^k}(s,a))^2}{\mu_h(s,a)}\right)$, where $d^{\pi}$ is the visitation distribution of policy $\pi$, $\mu$ is the sampling distribution, and $H$ is the horizon.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
The paper studies the off-policy evaluation problem in offline settings. The authors introduce an important sampling-based estimation procedure for policy evaluation. In general, the estimation process consists of two parts: the first part is to estimate the visitation of the target polices and in the second part, the density ratio and the offline distribution are estimated via the data. The sample complexity over the performance error is established.

### Strengths
1. The paper studies a finite sample guarantee for the performance error. 
2. The underlying problem, i.e., off-policy evaluation, is important in offline RL. 
3. The structure of the paper is clear.

### Weaknesses
1. The paper misses a lot of the important results in the OPE literature: 
     a. Uehara, Masatoshi, Jiawei Huang, and Nan Jiang. "Minimax weight and q-function learning for off-policy evaluation." ICML
     b. Kallus, Nathan, et al. "Doubly robust distributionally robust off-policy evaluation and learning." ICML
     c. Liu, Yao, Pierre-Luc Bacon, and Emma Brunskill. "Understanding the curse of horizon in off-policy evaluation via conditional 
          importance sampling." ICML 
     d. Kallus, Nathan, Yuta Saito, and Masatoshi Uehara. "Optimal off-policy evaluation from multiple logging policies." ICML
The above works (but not limited to) are highly related to the current paper, from the marginalized important sampling estimator to doubly robust estimator, and then include the evaluation of multiple logging policies. The lack of discussions on these works limits the novelty of the current work. 

2. The scenarios and applications for the simultaneous OPE on multiple policies are not convincing. OPE for single/mixture/logging policy is very important in RL literature, however, the current framework in this work lacks evidence of the real-life application. 

3. In the offline RL setting, the distributional shift is one of the most important issues, in both OPE and OPL problems. Unfortunately, in this work, the study and explicit treatment of this issue does not appear. See the review: Levine, Sergey, et al. "Offline reinforcement learning: Tutorial, review, and perspectives on open problems." arXiv preprint arXiv:2005.01643 (2020).

4. In recent years, there has been significant progress in relaxing the coverage condition in offline RL literature. In many works, the single policy concentrability condition is enough to learn a policy with a sample efficient guarantee in the policy learning setting. Therefore, the full coverage assumption, which is implicitly assumed in this work, is very stringent. This greatly limits the potential application scope in practice.  See the one of the related papers: Jin, Ying, Zhuoran Yang, and Zhaoran Wang. "Is pessimism provably efficient for offline rl?." ICML

5. The calculation of the importance weighting ratios is very difficult in practice. The challenges come from the Realizability and Stabablity. The choices of the function class are not well discussed in this function approximation setting. Also, the practical optimization stability problem is not discussed. 

6. Theorem 4.8: The sample complexity in this work is almost not improved in terms of the intuitive $K$-policy extension of the existing work in Uehara, Masatoshi, et al. "Finite sample analysis of minimax offline reinforcement learning: Completeness, fast rates and first-order efficiency." arXiv. 

7. One of the major concerns in this work is the lack of empirical evaluation of the benchmark. One cannot position this work in the existing literature without evidence from comprehensive empirical studies. Especially, considering the complexity of the estimation procedure and the optimization stability issue behind it, this practical application of this algorithm is unknown.

### Questions
See weakness.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper explores the multiple-policy evaluation problem and introduces the CASEAR algorithm as a solution. 

The algorithm begins by creating estimators for the policy visitation distributions, followed by constructing mean reward estimators for all K policies.

### Strengths
This paper addresses the multiple-policy evaluation problem, an area that has received relatively limited attention in the field. It is a valuable contribution to the community. 

The proposed method appears to be novel.

### Weaknesses
The paper is difficult to follow, and my primary concern is the correctness/meaningfulness of the results.

There are numerous typos, and both the notation and overall writing could be improved.

Additional relevant references could be incorporated into this work.

### Questions
1. Theorem 4.8: The theorem claims that we can evaluate all K policies within an eps error margin using n trajectories. However, the paper previously only mentioned that the trajectories are drawn from some distribution mu. Without any assumptions about mu, it's easy to construct a scenario where evaluating a particular policy pi is infeasible (e.g., if mu is completely unaligned with pi). In such a case, the sample complexity upper bound would be infinite, rendering it uninformative. Clarifying when this upper bound is finite and meaningful is essential.

Can you explicitly state any assumptions made about mu and how these assumptions ensure the sample complexity bound is finite and meaningful in practice? This would help clarify an important aspect of the theoretical guarantees.

2. Comparison with Dann et al. (2023): From the discussion, it is unclear whether the bound in this paper or the one in Dann et al. (2023) is tighter. 

Can you provide a more detailed comparison between the two bounds, including specific scenarios where each bound might be tighter? Can discuss the conditions under which their bound remains finite would be helpful for readers to understand the practical implications of their results?

3. While Dann’s bound is guaranteed to be finite (as d ^ max pertains to states that can be reached by some target policy), the bound in this paper can be infinite (since d pi k / mu h can be infinity).

3. Purpose of MOM: Could you clarify why the MOM method is proposed in this paper? It seems to appear abruptly. This method is typically applied to handle unbounded random variables. Is this the reason for its use here?

Can you provide a brief explanation of why MOM is necessary in their context, and how it relates to the specific challenges in their problem setting? This would help readers better understand the motivation behind using this technique.

5. The term pi det refers to all deterministic policies. The algorithm also attempts to estimate the visitation distributions for all of them (see line 381). It’s unclear why estimating only the visitation of the target policies is insufficient. Additionally, estimating the visitation distributions for all deterministic policies seems like a strong requirement.

Could you clarify why this broader estimation is necessary, and whether there are any computational or theoretical implications of this approach? Can you discuss potential alternatives or approximations that might be less computationally intensive?

6. The paper includes many terms in the form of ratios. It would be helpful to add more explanations on why these ratios are well-defined. For example. Why is hat w / hat mu bounded? Why is tilde mu / hat mu bounded?

Can you provide a brief explanation or proof for why each of these key ratios is well-defined and bounded. This would help readers better understand the technical details of the approach.

7. Is the optimization program in equation (6) well-defined? Does a solution hat mu* always exist? If the solution has an infinite value, is it still meaningful?

8. Gap-Dependent Related Works:

"Offline Reinforcement Learning Under Value and Density-Ratio Realizability: The Power of Gaps"

"Revisiting the Linear-Programming Framework for Offline RL with General Function Approximation"
Additional Related Works:

9. Some other related work:

"Offline Reinforcement Learning with Realizability and Single-Policy Concentrability"
Additional Clarifications:

"Reinforcement Learning in Low-Rank MDPs with Density Features"

10. What does w in Lemma 4.5 attempt to approximate? I guess it is d pi?

11. Why g(w h) is referred to as a gradient in the IDES algorithm?

12. Use double quotation marks (`` '') in the first paragraph.

13. Line 287 has a typo: “sum i”?

14. In Algorithm 1, D_h(s,a) = [xxx]?

15. In Lemma 4.7, the notation hat mu_1, hat mu_N  conflicts with the use of mu h in other sections.

16. mu * is unclear upon first reading in Section 3.1; a forward pointer would be helpful.

Providing a brief discussion on the existence and uniqueness of solutions to this optimization problem, as well as the implications of potentially infinite values would be great.

17. Remark 4.1 is missing "some citations".

18. Line 267 is unclear. Trajectory data is mentioned, but it states that they are drawn from mu h. Is the dataset drawn at the trajectory level or at the (state, action, next state) tuple level?

19. Can you clarify the constant C in Proposition 4.2 and explain how C h is chosen in Algorithm 1?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
The paper addresses the problem of multiple-policy evaluation in reinforcement learning, where the goal is to estimate the performance of a set of $K$ target policies up to a specified accuracy with high probability. The authors propose a novel, sample-efficient algorithm called **CAESAR** (Coarse and Adaptive EStimation with Approximate Reweighing) to tackle this problem. CAESAR operates in two phases: first, it coarsely estimates the visitation distributions of the target policies using the **MARCH** (Multi-policy Approximation via Ratio-based Coarse Handling) algorithm; second, it approximates the optimal offline sampling distribution and computes importance weighting ratios using the **IDES** (Importance Density Estimation) algorithm. The authors provide theoretical guarantees for their algorithm, demonstrating a non-asymptotic, instance-dependent sample complexity of $\tilde{O}\left( \frac{H^4}{\epsilon^2} \sum_{h=1}^H \max_{k \in [K]} \sum_{s,a} \frac{(d_h^{\pi_k}(s,a))^2}{\mu^*_h(s,a)} \right)$, where $H$ is the horizon, $\epsilon$ is the desired accuracy, $d_h^{\pi_k}$ is the visitation distribution of policy $\pi_k$, and $\mu^*_h$ is the optimal sampling distribution. The paper also discusses how CAESAR compares to existing methods and its applicability to policy identification.

### Strengths
- **Novel Approach**: The paper introduces the CAESAR algorithm, which combines coarse estimation and importance weighting to achieve sample-efficient multiple-policy evaluation. This novel approach addresses a gap in existing reinforcement learning literature.

- **Theoretical Guarantees**: The authors provide rigorous theoretical analysis, offering non-asymptotic, instance-dependent sample complexity bounds for their algorithm. This strengthens the validity of their contributions.

- **Innovative Sub-algorithms**: The MARCH algorithm for coarse estimation of visitation distributions and the IDES algorithm for importance density estimation are innovative and could have applications beyond this work. They effectively adapt and extend existing techniques to the finite-horizon setting.

- **Discussion and Comparisons**: The paper includes a thorough discussion section that compares CAESAR with existing methods, highlighting its advantages and situating it within the broader research landscape.

### Weaknesses
- **Clarity and Readability**: The presentation lacks clarity in several sections. The theoretical analysis is particularly dense, with complex notation and insufficient intuitive explanations, which may impede understanding for readers not deeply familiar with the topic.

- **Practical Implementation**: The paper assumes access to an oracle for solving convex optimization problems, which may not be practical in real-world applications. The computational complexity and practical feasibility of implementing CAESAR are not thoroughly discussed.

- **Lack of Empirical Evaluation**: There is a significant lack of empirical experiments or simulations to validate the theoretical findings. Including empirical results would strengthen the paper by demonstrating the practical effectiveness of the proposed algorithms.

- **Scalability Concerns**: While the MARCH algorithm aims to estimate visitation distributions efficiently, the scalability of the method to environments with large state and action spaces is not fully addressed.

- **Comparison with Prior Work**: Although the paper discusses existing methods, it lacks comprehensive empirical comparisons. Demonstrating how CAESAR performs relative to prior algorithms in practice would enhance the contribution.

### Questions
1. **Intuitive Explanation of CAESAR**: Could the authors provide more intuitive explanations or illustrative examples to clarify the key steps of the CAESAR algorithm, particularly how coarse estimation contributes to sample efficiency?

2. **Computational Complexity**: How does the computational complexity of CAESAR compare with existing methods, especially considering the need to solve convex optimization problems and estimate importance weights? Are there practical algorithms that can implement CAESAR efficiently without reliance on an oracle?

3. **Empirical Validation**: Have the authors considered conducting empirical evaluations of CAESAR on benchmark reinforcement learning tasks to demonstrate its practical performance and verify the theoretical claims?

4. **Scalability to Large Spaces**: How does the proposed method scale to environments with large or continuous state and action spaces? Are there strategies to mitigate potential computational bottlenecks in such scenarios?

5. **Assumptions and Limitations**: Can the authors discuss the practicality of the assumptions made in the theoretical analysis and how they might impact real-world applications? Are there ways to relax these assumptions while maintaining the algorithm's performance guarantees?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper introduces CAESAR, a sample-efficient algorithm for multiple-policy evaluation in reinforcement learning. The approach involves computing an approximately optimal sampling distribution and using data drawn from this distribution to evaluate multiple policies simultaneously. The authors also provide a theoretical analysis of the algorithm’s sample complexity.

### Strengths
1. The work is specifically designed to handle multiple-policy evaluation efficiently, avoiding the linear scaling in sample complexity that would occur if each policy were evaluated independently.

### Weaknesses
1. Although the authors say that their work is practical, they have not shown any experimental results of their algorithm. From a theoretical perspective, they also didn't show whether their sample complexity bound is tight enough, compared with existing works.

2. This paper has some funny typos and is not well-organized. (1) In line 313, they write "(some citations)" without really citing anything. (2) In line 279, the $\mu$ is not defined in their algorithm's input. (3) $\alpha^*_\pi$ in line 383 is not clearly defined, etc. Besides these typos, so many subroutines and abbreviations also make this paper very hard to understand. The authors put too many trivial lemmas in their main text, making their main contribution unclear.

3. In this work, the authors propose two algorithms, MARCH and IDES, as subroutines of their main algorithm, CAESAR. MARCH provides a coarse estimation of the visitation distribution for all deterministic policies. And IDES rely on the estimated visitation distribution from MARCH. Thus, their main algorithm CAESAR works only on deterministic policies. Confusely, in the preliminary section, the authors define policies as stochastic policies. However, throughout the paper, they didn't make justifications on why their method shifts to be only applicable to deterministic policies.

4. As discussed above, the MARCH algorithm provides coarsely estimated visitation distributions of the target policies, which introduces errors. Likewise, the IDES algorithm relies on a sampling distribution derived from an approximate optimization problem (6), which minimizes only an upper bound of the estimation variance rather than the exact variance. These approximations result in a biased estimator, which is not preferable.

5. In Session 5, when comparing their sample complexity with related works, the authors apply the CR-lower bound (line 419) to derive a lower bound of their method CAESAR. However, as the authors themselves point out, the CR-lower bound is applicable only to unbiased estimators. However, as far as I understand, the estimator from CAESAR is not guaranteed to be unbiased. This discrepancy makes the lower-bound discussion less reliable.

6. The authors introduce the MARCH algorithm to estimate the visitation distributions of target policies; however, they didn't compare it to any existing algorithms designed for the same purpose.

### Questions
1. The authors claim that their method is fully offline. However, I am not sure if their IDES algorithm can work without online interaction. Namely, how is the $s^{i'} _h$ generated without interacting with the environment?

2. In Lemma 4.5, it is clear that the errors propagate step-by-step, with the error of each time step twice the error in the previous step. How can this eventually lead to an accurate estimation?

### Soundness
2

### Presentation
2

### Contribution
2
