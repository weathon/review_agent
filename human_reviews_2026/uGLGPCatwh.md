# Confounding Robust Meta-Reinforcement Learning: A Causal Approach

- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Meta-Reinforcement Learning (Meta-RL) focuses on training policies using data collected from a variety of diverse environments. This approach enables the policy to adapt to new settings with only a few training steps. While many Meta-RL methods have demonstrated success, they often rely on the assumption that unobserved confounders can be excluded \emph{a priori}. This paper investigates robust Meta-RL in sequential decision-making, given confounded observational data collected across multiple heterogeneous environments. We introduce a novel augmentation procedure, called Causal MAML, which employs partial identification methods to generate posterior counterfactual trajectories from candidate environments that align with the confounded observations. These counterfactual trajectories are then used to find a policy initialization that produces strong generalization performance in the target domain. Theoretical analysis reveals that our causal Meta-RL approach is guaranteed to yield a solution that minimizes generalization loss.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper tackles unobserved confounders in Meta-RL via partial identification methods to generate counterfactual trajectories from candidate environments that align with the confounded observations.

### Strengths
This paper addresses an important challenge in reinforcement learning: the presence of unobserved confounders, approached from a causal inference perspective. Numerical experiments are conducted to demonstrate the effectiveness of the proposed method.

### Weaknesses
1. The confounding environment appears to be overly simplified. In many sequential settings, the transition dynamics are modeled as a function  $f: \mathcal{S} \times \mathcal{X} \times \mathcal{U} \rightarrow \mathcal{S} \times \mathcal{U}$, whereas the paper assumes that the current unobserved state is not influenced by the history and it somehow does not reflect the challenge in the sequential setting. This assumption restricts the applicability of the proposed method. I am wondering whether the methods extend to this more general setting, and if so, what additional assumptions or modifications (e.g., on the evolution of $\mathcal{U}$, identifiability, or estimation strategy) are required.

2.  There appear to be important missing components in the paper. In particular, no formal identification results are established for the counterfactual trajectories. It remains unclear what additional assumptions and regularity conditions are required to ensure identification, and what the theoretical guarantees are regarding the quality of the estimated counterfactual trajectories. Furthermore, in practical applications, it is not evident how to determine the dimension of the unobserved states. 

3. The paper claims that the solution minimizes the generalization error. However, the theoretical results only show for the first-order stationary point. More argument and discussion are needed for this statement.

### Questions
See above.

### Soundness
3

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
4

### Summary
This work aims to address the problem of unobserved confounders in meta-reinforcement learning environments. By leveraging the method of partial counterfactual identification, the authors propose a causal MAML framework, which utilizes counterfactual trajectories to find a policy initialization that exhibits strong generalization performance in target domains.

### Strengths
1.	The paper addresses an important issue in meta-reinforcement learning, confounding bias, and introduces a causal perspective to tackle it.
2.	The proposed algorithm is rigorously derived through formal theoretical development, including the definitions of CMDPs and canonical causal models, as well as a convergence proof under bounded-gradient assumptions.

### Weaknesses
1.	The entire study is conducted under discrete and finite CMDP settings. Consequently, both theoretical formulation and empirical validation are confined to simplified, low-dimensional environments. While the framework is theoretically sound, its applicability and scalability to high-dimensional continuous control tasks remain unverified.
2.	The literature review could be more comprehensive. While the paper states that research on handling unobserved confounders in meta-reinforcement learning is still missing, several existing works have already investigated this direction using causal approaches.

### Questions
1.	Not disclosing significant LLM usage.
2.	In lines 086–090, the paper identifies the lack of “a systematic approach for performing meta-learning across heterogeneous domains with the presence of unmeasured confounding.”
However, several recent studies [1,2] have already explored causal approaches to address unobserved confounders in meta-reinforcement learning.
Therefore, the gap and motivation would be stronger if the authors explicitly acknowledge these prior works and clarify how their method fundamentally differs from, or advances beyond, existing causal meta-RL approaches.

1.	‘The shorter purple route’ should be ‘The … orange …’ in line 209.
2.	The paper does not include comparisons with existing meta reinforcement learning based on casual methods.

[1] Dasgupta I, Wang J, Chiappa S, Mitrovic J, Ortega P, Raposo D, Hughes E, Battaglia P, Botvinick M, Kurth-Nelson Z. Causal reasoning from meta-reinforcement learning. arXiv preprint arXiv:1901.08162. 2019 Jan 23.

[2] Dasgupta I, Kurth-Nelson Z, Chiappa S, Mitrovic J, Ortega P, Hughes E, Botvinick M, Wang J. Meta-reinforcement learning of causal strategies. InThe Meta-Learning Workshop at the Neural Information Processing Systems (NeurIPS) 2019.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper aims to address a critical and under-explored issue in meta-reinforcement learning: how to learn a policy that can robustly and quickly adapt to new tasks when the expert data used for meta-learning is contaminated by unobserved confounding variables. The authors propose a framework called Causal MAML. 

This framework employs variational inference to learn a causal generative model for inferring the posterior distribution of confounding variables from biased observational data. Then, this distribution is used to generate counterfactual trajectories to "purify" the data to help the training of MAML. 

The paper provides theoretical analysis to support the unbiasedness of its meta-gradient estimation and validates the effectiveness of the method through experiments in two custom-designed confounding environments (Windy Gridworld and Causal Pendulum).

### Strengths
1. The paper's primary strength is its rigorous formulation of the problem. It moves beyond heuristics to prove that the core issue is a biased meta-gradient resulting from confounding. The central theoretical result—that a gradient computed on ideal counterfactual data is an unbiased estimator of the true, unconfounded gradient  (Theorem 4.1)— provides a strong guarantee on the correctness of the optimization objective. This establishes a clear and principled target for the algorithm.
2. This paper focuses on the challenge of confounding robustness in Meta-RL, a highly relevant yet often overlooked issue in real-world applications. The introduction of causal inference into Meta-RL sounds interesting.

### Weaknesses
1. The proposed practical algorithm (Algorithm 1) is built on a foundation that is not scalable. The core "Counterfactual Bootstrap" step requires MCMC sampling from the posterior over all possible MDPs (ρb(M | Di_obs)). This is computationally infeasible for any non-trivial environment. The method's success in the paper is an artifact of using toy domains where this step is barely possible. This reliance on an unscalable sampling procedure makes the proposed algorithm impractical for real-world application.
2. The Windy Gridworld and Causal Pendulum used in the paper are essentially "toy problems", characterized by low-dimensional state spaces and simple dynamics models. While these environments help illustrate the concept of "confounding" intuitively, they are far removed from the complexity of real-world problems. 
3. The most concerning shortcoming of the paper is the complete absence of any direct evidence demonstrating that its causal inference module actually learns meaningful information about the confounder. The paper's core claim is that it "infers and utilizes the confounding variable $U$," yet the experimental section only reports final task rewards without any analysis or visualization of the learned latent variable $U$.
In a controlled environment where the ground truth of the confounding variable is known, such validation is straightforward and necessary. For instance, the authors should visualize the relationship between $U$ and the true confounder. In the absence of such validation, the causal inference module becomes an uninterpretable "black box." We cannot determine whether the performance improvement stems from successful causal inference or merely because the VAE structure happened to learn a useful—but causally irrelevant—abstract representation in these simple tasks. This significantly undermines confidence in the methodological core contribution.
4. The comparisons in the paper are limited to standard MAML and a simple pre-training baseline. This overlooks a substantial body of related work in unsupervised/self-supervised RL, which focuses on learning latent representations or skills from reward-free interactions (e.g., SMM, DIAYN). These methods are typically evaluated on more complex and widely adopted benchmarks (e.g., DeepMind Control Suite, MuJoCo). The failure to compare Causal MAML against such methods, or at least test it in environments of comparable complexity, makes its contribution appear somewhat isolated and raises serious doubts about its scalability.

### Questions
1. Could you provide a qualitative analysis to demonstrate that the learned latent variable $U$ indeed captures the true confounding information? For instance, by visualizing the relationship between $U$ and the actual wind force/spring stiffness.
2. How would your method handle confounding variables that dynamically change (non-stationary) or affect system dynamics in non-additive ways? Is the current framework sufficient to address these, or would it require significant modifications?
3. Have you considered evaluating your method on more challenging benchmarks (such as task sets with introduced confounding variables in MuJoCo) to demonstrate its scalability? Compared to self-supervised RL methods designed to learn latent environmental factors, what advantages does your approach offer?

### Soundness
2

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
4

### Summary
The authors propose a robust meta learning method that can perform effectively in environments with unmeasurable confounding factors that affect the environment dynamics. The key idea is to use causal inference and partial identification of confounding variables to overcome this. By augmenting counterfactual trajectories from an environmental model consistent with the observed data repeatedly, the proposed method un-biases the meta learner from effects of confounding variables. The authors present in depth theoretical proofs and empirical experiments in “Windy Gridworld” environment (unobserved wind patterns as confounding factor) show that the proposed method outperforms vanilla MAML and other RL-based methods.

### Strengths
The paper is well structured and easy to read. The authors motivate the problem well, by clearly identifying gaps in existing meta-RL algorithms. Bringing ideas from causal inference into meta-learning applications is quite novel and the results are promising, compared to vanilla meta learning methods. The detailed theoretical analysis, well outlined pseudo algorithms, description of the experiments conducted in the grid world and the performance achieved is very interesting.

### Weaknesses
A key concern is the choice of baselines - while the proposed method clearly is better than vanilla MAML, I think a comparison to other state of the art distributional robust [1] or bayesian [2] meta learning methods would put the strength of the proposed algorithm in better perspective. Quantitative comparison against these more relevant baselines would have made the contributions much more impactful. 

The fact that Causal PPO, In appendix C, matches or beats the proposed approach in all the tasks supports the need for stronger baselines. 

Another approach to tackle confounding variables could be to formulate it as partially observed markov decision processes (POMDPs) and leverage methods like RL^2 [3]. What are some advantages of using a causal inference approach over this? 

Minor typo : In appendix B.2 “log” appears twice in the equation. This is most probably a typo. 

Minor : The plot colors in the main paper are not consistent with the appendix, it would be great if they are consistent. PPO is “green” in the appendix but “orange” in the main paper.

[1] A Simple Yet Effective Strategy to Robustify the Meta Learning Paradigm https://arxiv.org/abs/2310.00708
[2] https://arxiv.org/abs/1806.03836 Bayesian model agnostic meta learning
[3] RL^2 https://arxiv.org/abs/1611.02779

### Questions
Causal PPO outperforming the proposed causal MAML approach brings forward the question of why we need meta-learning at all? The key seems to be having counterfactual data augmentation. Do the authors have some thoughts on certain tasks where a causal MAML would hold advantage over Causal PPO?

### Soundness
3

### Presentation
3

### Contribution
3
