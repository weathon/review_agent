# Enabling Pareto-Stationarity Exploration in Multi-Objective Reinforcement Learning: A Weighted-Chebyshev Multi-Objective Actor-Critic Approach

- Decision: Reject
- Scores: 6, 3, 6, 6, 1

## Abstract
In many multi-objective reinforcement learning (MORL) applications, being able to systematically explore the Pareto-stationary solutions under multiple non-convex reward objectives with theoretical finite-time sample complexity guarantee is an important and yet under-explored problem.
This motivates us to take the first step and fill the important gap in MORL. 
Specifically, in this paper, we propose a weighted-Chebyshev multi-objective actor-critic (\policyns) algorithm for MORL, which uses multi-temporal-difference (TD) learning in the critic step and judiciously integrates the weighted-Chebychev (WC) and multi-gradient descent techniques in the actor step to enable systematic Pareto-stationarity exploration with finite-time sample complexity guarantee.
Our proposed \policy algorithm achieves a sample complexity of $\tilde{\mathcal{O}}(\epsilon^{-2}p^{-2}\_{\min})$ in finding an $\epsilon$-Pareto-stationary solution, where $p_{\min}$ denotes the minimum entry of a given weight vector $p$ in the WC-scarlarization.
This result not only implies a state-of-the-art sample complexity that is independent of objective number $M$, but also brand-new dependence result in terms of the preference vector $p$. 
Furthermore, simulation studies on a large KuaiRand offline dataset, show that the performance of our \policy algorithm significantly outperforms other baseline MORL approaches.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies multi-objective reinforcement learning by extending existing multi-objective optimization techniques within the framework of actor-critic methodologies. Central to the discussion is the concept of (local) (weak) Pareto optimality, which serves as the foundational solution criterion for balancing multiple objectives. The proposed training architecture employs a single actor coupled with multiple critics, where each critic is specifically trained to approximate the value function corresponding to a distinct objective. During the training process, the actor is updated using a weighted gradient approach that integrates the estimated values from each critic. These weights are supposed to balance the influence of each objective, guiding the actor towards policies that achieve a desirable trade-off among the competing objectives. The authors rigorously demonstrate that their method is capable of converging to locally Pareto optimal policies. This convergence is guaranteed up to an error bound, which is linked to the accuracy of the critic's value function estimations.

### Strengths
1. The manuscript is exceptionally well-written, presenting complex concepts in a clear and accessible manner. It seems that the authors have meticulously structured the paper, which significantly enhances readability. Additionally, the discussion of related works is thorough, providing a comprehensive overview of existing approaches in multi-objective reinforcement learning and actor-critic methodologies. 

The experimental component of the study appears to be conducted on a large-scale real-world short video platform. (However, the manuscript lacks detailed descriptions of the experimental setup, making it challenging to fully assess the validity the results. Clarification on the this concern is detailed in the section below.)

### Weaknesses
1. The major weakness might lie in the novelty of the proposed method. To the knowledge of the reviewer, Although the theoretical analysis partially answers the challenges mentioned in the introduction, the proposed method seems to largely based on (Désidéri, 2012) and  (Momma et al., 2022), with the multi-objective optimization gradients replaced with policy gradients.

2. See questions below.

### Questions
1. In Theorem 4, what is the different between |w*-w| and zeta? Intuitively, both these two terms are related to the error of value functions. The reviewer is confused because the definition of w* also needs clarification. The optimality of the critic network depends on both phi(s) and w. Different state representation phi might correspond to different optimal w. Is phi function fixed?

2. In terms of the proposed method, what is the major novelty compared to previous methods in MOO literature? The reviewer understands the discussions already presented in the paper, but is just wondering whether the proposed method is replacing the gradients in MOO with policy gradients given by different critics.

3. It seems that the detailed experiment description is not specified even in the appendix. What is the horizon of this game? What is the size of the state set?

4. Is the weight vector p predefined or learned?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper tackles the multiobjective reinforcement learning problem. It builds on top of the actor-critic algorithm using multi-gradient
descent algorithm (MGDA), named multiobjective actor-critic (MOAC), designed by Zhou et al. (2024) and leverages the framework introduced by Momma et al. (2022) to optimize a multiobjective weighted by a preference vector, denoted by p. The difference from Zhou et al. (2024) is the introduction of preference weighting, which allows exploration of the Pareto Stationary front solutions. The proposed WC-MOAC algorithm maintains the sample complexity $\tilde{O}(\epsilon^{-2})$, shown in Zhou et al. (2024), after the addition of the preference weighting. Meanwhile, the algorithm is tested on a large-scale real-world dataset from the recommendation logs of the short video streaming mobile app Kuaisho.

Zhou, T., Hairi, F. N. U., Yang, H., Liu, J., Tong, T., Yang, F., ... & Gao, Y. (2024). Finite-time convergence and sample complexity of actor-critic multi-objective reinforcement learning. ICML.

Momma, M., Dong, C., & Liu, J. (2022, June). A multi-objective/multi-task learning framework induced by pareto stationarity. In International Conference on Machine Learning (pp. 15895-15907). PMLR.

### Strengths
The methodology and the algorithm are clearly stated. The assumptions and the main theorem are well presented.

### Weaknesses
The contribution can be stated more directly. For example, the abstract mentions that the algorithm fills the gap "to systematically explore the Pareto-stationary solutions". It would be more apparent if the paper explained why it is needed to explore these solutions and how the exploration is achieved. 

The paper overstates the contribution without a clear acknowledgement of previous work. The paper should introduce the work by Zhou et al. (2024) with more details, and it is also necessary to make a comparison to clarify the novelty of the work. Some examples of overstated contributions examples are listed below. (1) It is claimed in the abstract that "the performance of our WC-MOAC algorithm significantly outperforms other baseline MORL approaches." However, the performance drops from the behaviour clone baseline for two objectives.  (2) Line 100 states, "Collectively, our results provide the first building block toward a theoretical foundation for MORL." Zhou et al. (2024) gave a sample complexity bound of the same order.  (3) Line 113 states, "To mitigate the cumulative systematic bias injected from the WC-scalarization weight direction and finite-length state-action trajectories, we propose a momentum-based mechanism in WC-MOAC." What is the difference in the mechanism from Zhou et al. (2024)? 

The clarity and correctness of writing can also be improved. Please refer to the question section for details. 

The experiment design is not fully convincing, and the result is hard to understand. Please refer to the question section for details. More experiments can be designed to show the advantages of using the preference.

### Questions
1. Line 113 states, “ To mitigate the cumulative systematic bias injected from the WC-scalarization weight direction and finite-length state-action trajectories, we propose a momentum-based mechanism.” What is this systematic bias? Also, why does this momentum mechanism remove the dependence on the number of objectives M?

2. Where is Lemma 1 cited from? In Qiu et al. (2024) Proposition 4.2, they state that a stochastic policy has to maximize linearized scalarization for all weight vectors p to be a weakly Pareto optimal policy. However, Lemma 1 of the paper only requires maximizing the infinity norm of the scalarization for some weight vector.

3. For the experiment in Table 1, how is the preference vector chosen? 

4. For the experiment in Figure 1, would TSCA work better by switching the main objective? More experiment designs to show the necessity of preference would be better.

5. The paper focuses on the finite-time convergence results for multi-objective actor-critic. What is the advantage of using actor-critic algorithms?

Qiu, S., Zhang, D., Yang, R., Lyu, B., & Zhang, T. (2024). Traversing pareto optimal policies: Provably efficient multi-objective reinforcement learning. arXiv preprint arXiv:2407.17466.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents the weighted-Chebyshev multi-objective actor-critic (WC-MOAC) algorithm to address the challenge of systematically exploring Pareto-stationary solutions in multi-objective reinforcement learning under multiple non-convex reward objectives. The WC-MOAC algorithm is designed to combine multi-temporal-difference (TD) learning in the critic phase with weightedChebyshev scalarization and multi-gradient descent in the actor phase, effectively managing the complexities of non-convexity and scalability in multi-objective problems. The primary theoretical result of this work is a finite-time sample complexity bound, $O\left(\epsilon^{-2} p_{\min }^{-2}\right)$, which is independent of the number of objectives $M$, where $p_{\min }$ denotes the minimum entry of the weight vector $p$ in the scalarization. This independence from $M$ is a notable theoretical advance. The empirical evaluation, conducted on the KuaiRand offline dataset, indicates that the WC-MOAC algorithm surpasses baseline methods in performance, highlighting its robustness and potential for realworld applications in multi-objective reinforcement learning.

### Strengths
I feel the paper's originality comes from its novel application of the weighted-Chebyshev scalarization technique within the framework of multi-objective reinforcement learning, accompanied by theoretical guarantees on finite-time sample complexity. This is particularly noteworthy given the limited prior work addressing non-convex reward objectives in this field. The technical rigor is evident, as the authors present a well-founded theoretical analysis that establishes finite-time sample complexity guarantees. This analysis is both rigorous and practically relevant, especially in environments that require systematic exploration of Pareto-stationary solutions. The empirical validation further strengthens the contribution, with experimental results on the KuaiRand dataset showing that the proposed algorithm consistently outperforms baseline approaches. This demonstrates the algorithm's robustness and effectiveness in practical scenarios. Additionally, the research fills an important gap in multi-objective reinforcement learning, with potential applications across various domains requiring multi-objective optimization. Its relevance to real-world problems enhances the significance of the work.

### Weaknesses
While the paper provides a comprehensive theoretical foundation, certain methodological aspects could be clarified further. For example, the integration of the multi-gradient descent update, which computes a dynamic weighting vector $\lambda_t$ that balances exploration with convergence, could benefit from a more detailed discussion on its rationale and practical implementation steps. Additionally, the empirical evaluation is limited to a single dataset, the KuaiRand offline dataset, which raises questions about the algorithm's generalizability. Expanding the experimental analysis to include diverse datasets or multi-objective environments would provide deeper insights into the algorithm's robustness across varied applications.

### Questions
The paper would benefit from further clarification on several key points. First, could the authors discuss the impact of the preference vector $p$ on the algorithm's performance, especially in environments with highly non-convex reward landscapes? As the convergence rate depends on $p_{\min }$, understanding the selection of $p$ would be useful for real-world applications. Additionally, what limitations, if any, might arise in adapting the proposed algorithm to online settings, given that the study's empirical evaluation is restricted to offline data? The paper also raises questions regarding the sensitivity of the algorithm's performance to variations in $p$ and whether there are recommended heuristics for selecting optimal values. Lastly, in relation to multi-temporaldifference learning, were there specific stability challenges encountered, particularly when combining it with weighted-Chebyshev updates? If so, what mechanisms or parameters were introduced to maintain convergence stability?

### Soundness
2

### Presentation
2

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
This paper investigates multi-objective reinforcement learning (MORL) using a weighted-Chebyshev multi-objective actor-critic (WC-MOAC) framework. The authors implement a multi-temporal-difference (TD) learning method for the critic and apply a weighted-Chebyshev multi-gradient-descent algorithm (MGDA) for the policy gradient in the actor. The algorithm is shown to achieve a sample complexity of $\mathcal{O}(\epsilon^{-2})$ to reach an $\epsilon$-Pareto stationary point. The authors also provide empirical validation using a real-world dataset.

### Strengths
The paper is well written with clear motivation and explanation. The authors’ theoretical analysis provides an important contribution to MORL. I would suggest the paper be accepted after minor revision.

### Weaknesses
Some of the details are not clear enough and can be improved. I have put them in Questions.

### Questions
1.	Page 5 eq (2). On the left of “:=” is a function of x, while on the right you have a scalar. Please make the definition consistent.
2.	Page 5 Lemma 1. You do not provide a proof for Lemma 1. Please reference the exact theorem or proposition from the cited work that supports Lemma 1.
3.	Lemma 1 states existence of the vector p, which is directly related to a weak Pareto-optimal. However, it is not clear how $p$ is determined in practice within the algorithm. Does Theorem 4 hold for arbitrary p or for the specific p provided in Lemma 1? It would be helpful to include a more detailed explanation of p’s role in the algorithm and whether it influences the theoretical guarantees.
4.	Page 7. It will be clearer if you specify what arguments you are optimizing. In eq (7), could you clarify whether the optimization is over $\lambda$, or both $\lambda$ and $\theta$? It appears from my understanding that eq (9) optimizes over both $\lambda$ and $\theta$, whereas eq (10) involves only $\lambda$.
5.	Page 7 line 340, “gradient” should be “Jacobian”.
6.	Page 8 Def 3 line 419. $\lambda$ is already given. Why do you have $\min_{\lambda}$?
7.	Page 9 Theorem 4. Some of the quantities, such as $\lambda_A$, $L_J$, $R_w$, $\bf{\gamma}$ are from the appendix, it is better to add some explanation for these quantities in the statement of the theorem.
8.	Page 9 line 467. You mention that state-of-the-art sample complexity for single-objective RL is $\mathcal{O}(\epsilon^{-2})$ bu Xu et al. And your Corollary 5 achieves the same complexity for MORL, which is a great achievement. Actually, under some special structure like linear quadratic problem, the one can show a sample complexity of $\mathcal{O}(\epsilon^{-1})$, as proved by Zhou and Lu in “Single Timescale Actor-Critic Method to Solve the Linear Quadratic Regulator with Convergence Guarantees”. Please consider adding it as a remark to enhance completeness of the paper.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 5

### Rating
1

### Rating Number
1

### Confidence
5

### Summary
This paper proposes weighted-Chebyshev multi-objective actor-critic (WC-MOAC) to find Pareto-stationary policies for MORL with sample complexity guarantees. The proposed WC-MOAC combines two techniques, namely the (stochastic) multiple gradient approach and the momentum-based updates of the dual variables. The WC-MOAC has a sample complexity of $\tilde{O}(\epsilon^{-2})$ in finding an epsilon Pareto-stationary policy. Finally, simulation results on an offline dataset is provided.

### Strengths
- The obtained sample complexity is independent of the number of objective functions and has a nice scaling with $M$ (However, there is a severe concern stated below).
- This paper handles two formulations, namely, discounted-reward and average-reward MDPs, simultaneously (despite that the algorithm and the analysis are agnostic to the reward setting in principle).
- Overall the paper is well-organized and easy to follow, with the concepts, definitions, and theoretical results clearly explained.

### Weaknesses
The major issue with this paper is that it is very similar to the paper “Finite-Time Convergence and Sample Complexity of Actor-Critic Multi-Objective Reinforcement Learning” by (Zhou et al., 2024) published in ICML 2024, in terms of both the algorithm design, paper writing, and the claimed contributions. The flow of the paper exactly follows (Zhou et al., 2024), and most of the paragraphs are just paraphrased versions of (Zhou et al., 2024). For example:

- The pseudo code of WC-MOAC are almost the same (almost verbatim) as those of the MOAC algorithm (cf. Algorithms 1 and 2 in (Zhou et al., 2024)). 
- The theoretical result of WC-MOAC in Theorem 4 and Corollary 5 appear almost the same as the Theorem 5 and Corollary 6 in (Zhou et al., 2024).
- The 1st  paragraph of Introduction (Lines 34-46) appears to be paraphrased from the first two paragraphs of the Introduction of (Zhou et al., 2024).
- The 2nd paragraph of Introduction (Lines 47-64) appears to be paraphrased from the third paragraph of the Introduction of (Zhou et al., 2024).
- The paragraphs about the “Key Contributions” (Lines 97-124) appear to directly follow the “Main Contributions” of the Introduction of (Zhou et al., 2024).
- In Section 3, the problem formulation about MOMDP (Lines 181-187) appears very similar to the second paragraph of Section 3.1 of (Zhou et al., 2024).
- The part on “Learning Goal and Optimality in MORL” (Lines 200-215) appears almost the same as the “Problem Statement” in Section 3.1 of (Zhou et al., 2024). Moreover, even the footnote #3 in this paper almost goes verbatim compared to the footnote #1 in (Zhou et al., 2024).
- The preliminaries about the policy gradient for MORAL (Lines 274-296) also largely resembles Section 3.2. Specifically, several sentences about Lemma 2 and Assumption 2 of this paper are exactly the same as those in Lemma 1 and Assumption 2 of (Zhou et al., 2024).
- The description about Assumption 3 and Lemma 3 in this paper (Lines 424-437) appears to exactly follow the Assumption 3 and Lemma of (Zhou et al., 2024).

### Questions
In addition to the above, here are some further technical questions:
- Technically: One of my main technical concerns is the motivation for finding a Pareto-stationary policy (under the assumption that the state and action spaces are finite) in the specific context of MORL. Specifically, while it is indeed difficult to find the whole Pareto front in MORL, it is actually not hard to find one or some Pareto-optimal policies by adapting reducing MORL to single-objective RL and finding the convex coverage set (e.g., (Yang et al., 2019)). For example, based on (Chen and Magulari, 2022), one can use a policy-based method with off-policy TD learning (under linear function approximation) to find an epsilon-optimal solution for single-objective RL with a sample complexity of $\tilde{O}(1/\epsilon^2)$. There are also several other recent works like (Lan 2021; Fatkhullin et al., 2023; Liu et al., 2020; Chen et al., 2022) that can find an epsilon-optimal policy with sample complexity guarantees. To adapt these results to MORL, one can use linear scalarization and thereby find one Pareto-optimal policy (specific to some preference vector). As a result, it remains not totally clear why it is theoretically appealing to design an algorithm for finding only a Pareto-stationary policy if we can already find Pareto-optimal policies (despite that Pareto-stationarity is indeed a widely adopted concept in the MOO literature).

- Another concern is the novelty in terms of algorithm and convergence analysis. Specifically, the WC-MOAC algorithm appears to be a direct application of the MOAC algorithm (Zhou et al. 2024) and also similar to the MOO algorithm CR-MOGM of (Zhou et al. 2022), which is the enhanced (stochastic) MGDA method (e.g., (Desideri, 2012; Liu and Vicente, 2021)) with the momentum update of the dual variable vector, to the setting of MORL (with the multi-objective critic learned by standard TD updates). Under a properly learned critic, then the stochastic multiple gradients can have a sufficiently low bias such that it enables similar convergence guarantees as in the general MOO. This is also shown in Theorem 11 (as a direct result of Lemma 10). As a result, the sample complexity and the convergence analysis of WC-MOAC essentially resemble those of CR-MOGM in (Zhou et al. 2022) for the general non-convex case, cf. Theorem 3 and Appendix E of (Zhou et al. 2022).

References:
- (Yang et al., 2019) Runzhe Yang, Xingyuan Sun, and Karthik Narasimhan, “A Generalized Algorithm for Multi-Objective Reinforcement Learning and Policy Adaptation,” NeurIPS 2019.
- (Zhou et al., 2022) Shiji Zhou, Wenpeng Zhang, Jiyan Jiang, Wenliang Zhong, Jinjie Gu, Wenwu Zhu, “On the Convergence of Stochastic Multi-Objective Gradient Manipulation and Beyond,” NeurIPS 2022.
- (Chen and Magulari, 2022) Zaiwei Chen and Siva Theja Maguluri, “Sample Complexity of Policy-Based Methods under Off-Policy Sampling and Linear Function Approximation,” AISTATS 2022.
- (Lan 2021) Guanghui Lan, “Policy Mirror Descent for Reinforcement Learning: Linear Convergence, New Sampling Complexity, and Generalized Problem Classes,” Mathematical programming, 2021.
- (Fatkhullin et al., 2023) Ilyas Fatkhullin, Anas Barakat, Anastasia Kireeva, Niao He, “Stochastic Policy Gradient Methods: Improved Sample Complexity for Fisher-non-degenerate Policies,” ICML 2023.
- (Liu et al., 2020) Yanli Liu, Kaiqing Zhang, Tamer Basar, Wotao Yin, “An Improved Analysis of (Variance-Reduced) Policy Gradient and Natural Policy Gradient Methods,” NeurIPS 2020.
- (Chen et al., 2022) Zaiwei Chen, Sajad Khodadadian, and Siva Theja Maguluri, "Finite-sample analysis of off-policy natural actor–critic with linear function approximation," IEEE Control Systems Letters, 2022.

### Soundness
2

### Presentation
3

### Contribution
1
