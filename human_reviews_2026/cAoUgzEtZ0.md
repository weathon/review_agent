# Representation Convergence: Mutual Distillation is Secretly a Form of Regularization

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 2, 6, 6

## Abstract
In this paper, we argue that mutual distillation between reinforcement learning policies serves as an implicit regularization, preventing them from overfitting to irrelevant features. We highlight two separate contributions: (i) Theoretically, for the first time, we provide an end-to-end theoretical proof that enhancing the policy robustness to irrelevant features leads to improved generalization performance. (ii) Empirically, we demonstrate that mutual distillation between policies contributes to such robustness, enabling the spontaneous emergence of invariant representations over pixel inputs. Ultimately, we do not claim to achieve state-of-the-art performance but rather focus on uncovering the underlying principles of generalization and deepening our understanding of its mechanisms.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies generalization in reinforcement learning by examining the robustness of learned features. It proposes (1) a theoretical framework showing how policies sensitive to “rendering” features that do not generalizable suffer from worse performance in expected returns and (2) the conjecture that mutual distillation between two agents regularizes representation learning toward such robust, invariant features and thus improves generalization.

The paper derives bounds relating expected test returns to terms measuring representation robustness and train/test differences, and empirically investigates whether mutual distillation indeed yields more robust policies on Procgen environments. The empirical results exhibit strong performance gains of the mutual distillation variant over PPO and several controls.

### Strengths
- The paper is easy to follow for the most part and has a clear motivated given a widespread interest in generalization and representation learning in RL.
- exploring generalization in RL through distillation is well-motivated and interesting. As far as I know, this technique is not exhaustively researched / understood in the context of RL.
- The formalization of rendering families and the decomposition of generalization error into robustness and train-test divergence components is intuitive and provides a neat mathematical framing. I believe this framing ends being equivalent to classical POMDP definitions, so it may be worth to align language (e.g., rendering function - emission function / observation function). 
- MDPO performs consistently better than PPO and other baselines on Procgen tasks, even when controlling for model size or training budget (at least partly).

### Weaknesses
**Theoretical novelty and completeness**:

I found that the novelty of the theoretical exposition is overstated in serveral places. One of the main theorems (Theorem 3.3) mirrors the first-order performance difference bound from Schulman et al. (2015, TRPO) so closely that I believe it should be cited as a known existing bound. 
The paper furthermore claims to be the first to “ first to provide a rigorous proof of {the} intuition {that robustness to irrelevant features enhances generalization performance}", which is an overstatement in this phrasing in my view. Several prior works in the literature provide generalization (some as part of sample-complexity) bounds explicitly dependent on representation adequacy. A non-exhaustive list: 

- kernel RL and kernel complexity as a measure of representation capacity: Yeh et al. (2023)
- from a causality perspective: Kallus et al. (2020), Suau et al. (2023)
- from a symmetry and invariance perspective: Weltevrede et al. (2025)

The latter of the above even considers distillation for deep RL policies and should most certaintly be discussed in this work. 

**Disconnect between theory and practice**
The biggest weakness of this paper in my view is the strong disparity between the derived theoretical claims and the conjectured effect of mutual distillation. The argument or intuition for how mutual distillation learns more robust features is very handwavy. For example it is not obvious at all that policies from different intialization or different batch orders or other factors will learn significantly different features. The work above indeed does show a related effect theoretically using neural tangent kernel theory (Weltevrede et al.). Knowing that feature learning is difficult to treat theoretically in deep learning I would suggest an alternative path would be to support the argument more empirically. For example with experiments showing indeed that different initialization / different exploration seeds / different batch orders etc. lead to feature learning of distinct spurious features which can then be eliminated through mutual distillation. 

**Presentation**
The paper has several issues with presentation and clariity. Apart from several writing / language issues, the are also a number of ambiguities: 
- Inconsistency in robustness metric. The robustness term is defined as a maximum total variation distance max TV distance, yet Table 1 lists values larger than 1.
- Algorithm 1 implies two agents collecting data separately; it is unclear whether the 50M total steps refer to per agent or shared. The appendix does not explicitly resolve this. Without this clarification, comparisons to PPO may reflect more total experience rather than representational effects.
- Adversarial/robustness experiments underspecified. The procedure for generating “adversarial renderings” (random CNNs) lacks some detail, and, importantly, motivation. I find this a very perculiar way of generating adversarial examples, could the authors elaborate this choice?
- some quantities (e.g., $L_\pi$) is not defined in such a way that the paper is self-sufficient. One should not need to refer to TRPO to recall its definition.
- Despite being in the title, the term "representation convergence" is never formally defined. I'm not following what this term aims to imply, what converges to what? 


- Yeh, Sing-Yuan, et al. "Sample complexity of kernel-based q-learning." International Conference on Artificial Intelligence and Statistics. PMLR, 2023.
- Kallus, Nathan, and Angela Zhou. "Confounding-robust policy evaluation in infinite-horizon reinforcement learning." Advances in neural information processing systems 33 (2020): 22293-22304.
- Suau, Miguel, Matthijs TJ Spaan, and Frans A. Oliehoek. "Bad habits: Policy confounding and out-of-trajectory generalization in RL." arXiv preprint arXiv:2306.02419 (2023).
- Weltevrede, Max, et al. "How Ensembles of Distilled Policies Improve Generalisation in Reinforcement Learning." arXiv preprint arXiv:2505.16581 (2025).

### Questions
- can you provide a theoretical statement (e.g., in linearized form) showing that mutual distillation dynamics reduce the $R$-robustness term? 
- how do you define the “representation convergence” concept?
- do the mutual distillation learners together use the same amount of samples as PPO baselines? If not, how is data shared or counted? 
- could you describe the random CNNs used to generate alternate renderings and the motivation behind it?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper presents a theoretical and empirical study arguing that mutual distillation between reinforcement learning policies acts as an implicit regularization to prevent overfitting to irrelevant features, thereby improving generalization. It claims to be the first to theoretically prove that enhancing policy robustness to irrelevant features leads to better generalization. The paper introduces Mutual Distillation Policy Optimization (MDPO), which uses Deep Mutual Learning (DML) via a KL divergence loss between to (otherwise independently) trained PPO agents. Empirically, MDPO shows improved generalization performance over a PPO baseline on the hard level configuration of the Procgen benchmark and demonstrates enhanced robustness to visual disturbances like random convolutions or changes in brightness, contrast, saturation or hue.  

**Recommendation:**\
I recommend to reject the paper because it is lacking in certain fundamental areas: insufficient positioning with respect to related work,  issues with clarity regarding the significance of theoretical results, empirical results, and the novelty of the ideas/approach.

### Strengths
- The central idea of using mutual distillation to induce robustness to spurious correlations in the training data is interesting. 
- The empirical results for MDPO look promising.
- The analytical experiments of the robustness of the MDPO policy to visual disturbances and the quality of the learned representations in Sections 5.3 and 5.4 are very insightful.

### Weaknesses
- The positioning of this work is severely lacking, especially concerning previous literature.
	-  The paper seems to miss several related works on topics such as:
		- Representation learning in RL (for example, [1,2]) 
		- Policy distillation for generalization (for example, [3,4]) 
		- Mutual distillation (for example, [5])
		- Overfitting to training data in RL (for example, [6,7,8])
		- The above are not necessarily exhaustive. 
	- The claims regarding theoretical novelty are strong, but difficult for me to verify due to insufficient discussion of existing work, and my own lack of knowledge on this specific topic.  
	- The novelty of the motivation in Section 4 is difficult to judge, since it lacks comparison with related ideas in the existing literature (for example, see [7,8]). 
- The theoretical contribution has some issues:
	- The core theoretical claim is formulated too strongly. Corollary 3.8 only shows an increase in the _lower bound_ on generalization performance as robustness increases. However, increasing a lower bound does not guarantee an increase in actualized performance, which is what is promised in the abstract and introduction. 
	- Clarity of the theoretical framework is lacking. For example, the proofs depend on two policies $\pi$ and $\tilde{\pi}$, but there is no mention of what the significance of these two policies are or how they are related to the overall narrative of the paper. 
	- A discussion on the significance of the theoretical results is missing. Whether increasing a lower bound has any bearing on the realized performance depends on whether the bound is vacuous or not. This means the bounds derived in Section 3 would benefit from analysis or discussion on how tight they are. For example, the bound's dependence on the relative performance of two arbitrary policies $\pi$ and $\tilde{\pi}$ complicates the interpretation of its significance. 
- The motivation for mutual distillation seems conceptually incomplete. The logic presented in Sections 4.1 and 4.2 hinges on the policies encountering different spurious correlations due to collecting different trajectories as a result of different initializations. However, the DML mechanism's goal is to regularize the two policies to converge to the same policy. This seems to introduce a paradox: in order to benefit from the two policies collecting different training data, they need to be regularized to be the same. 
- Experimental details are missing, making it difficult to judge the validity and significance of the results. For example, the paper claims significant improvements, but there is no mention of number of seeds used, what the shaded areas in the figures denote, or how significance was tested.

### Questions
- Does it matter how $\pi$ and $\tilde{\pi}$ are related? Can we view them through the lens of the two policies used in mutual distillation somehow? 
- The policy $\pi$, $\tilde{\pi}$ and the linear approximation $L_\pi(\tilde{\pi})$ (the only positive term in the lower bound in Corollary 3.8), seem to originate from [9], which assumes $\pi$ and $\tilde{\pi}$ are close to each other. Does this mean the lower bound will only be non-vacuous if $\pi$ and $\tilde{\pi}$ are similar? How does this impact the significance of the derived bounds?
- Line 200: "During the training process, we can only empirically bound $\mathfrak{D}_{train}$" Why? 
- Figure 2: "(Right) Through mutual distillation via DML, two policies regularize each other to converge toward a more robust hypothesis space,..." I agree that they are regularized towards each other, but why would that be toward a more robust hypothesis space, and not just any other part of the non-robust hypothesis space?
- Why is an algorithm that regularizes two policies towards each other, the solution for the problem identified in Figure 1 that requires the two policies to collect substantially different data?
- For the experiments, how many seeds are used and what do the shaded regions indicate? Also, how is significance of the results determined?


**Things to improve that did not impact decision:**
- There many missing words, making sentences incomplete (for example, line 58 and 65 of the introduction). 
- Section 2.1: Stating that a function $f: X \to [0,1]$ does not sufficiently define it as a probability distribution (it is missing the constraint that all the probabilities sum up to 1). 
- Section 2.1: The authors seem to introduce a new MDP framework with the rendering function in the background section. Either this is new, which means it should not be in the background section, or it is an existing framework from somewhere else, which means it is missing a citation. 
	- The framework also seems reminiscent of a specific type of contextual MDP [10], perhaps it could be useful to frame this work within the CMDP framework. 
- Table 2: It would be interesting to also include the training performance for this experiment.
- Figure 3: Does the x-axis for MDPO include the timesteps of both agents (since they collect data independently)? In other words, at timestep 50 million in the figure, the individual MDPO policies will only have trained on 25 million steps each? If not, I feel this figure is slightly misleading. 

**References:**\
[1] Learning Invariant Representations for Reinforcement Learning Without Reconstruction. Zhang et al. 2021\
[2] Cross-Trajectory Representation Learning for Zero-Shot Generalization in RL. Mazoure et al. 2022\
[3] Learning Dynamics and Generalization in Reinforcement Learning. Lyle et al. 2022\
[4] How Ensembles of Distilled Policies Improve Generalisation in Reinforcement Learning. Weltevrede et al. 2025\
[5] Dual Policy Distillation. Lai et al. 2020\
[6] On the Importance of Exploration for Generalization in Reinforcement Learning. Jiang et al. 2023\
[7] Policy Confounding and Out-of-Trajectory Generalization in RL. Suau et al. 2024\
[8] Exploration Implies Data Augmentation: Reachability and Generalisation in Contextual MDPs. Weltevrede et al. 2025\
[9] Trust Region Policy Optimization. Schulman et al. 2015\
[10] A survey of zero-shot generalisation in deep reinforcement learning. Kirk et al. 2023

### Soundness
2

### Presentation
1

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
This paper attempts to prove the conjecture that the policy which is robust to irrelevant features would lead to improved generalization performance. In addition, they propose mutual distillation of policies to achieve such robustness and presents the intuition behind that. While they do not show state-of-the-arts performance, they present proof-of-concept for their approach with basic regularization baseline on all environments of Procgen benchmark.

### Strengths
### Strengths:

1. This work presents a formal proof of a long standing assumption that robustness of policy against irrelevant features improves generalization. In particular, they derive a lower bound for generalization performance that includes minimization of a robustness term, which is defined how a policy is influenced by two different rendering (perturbation) functions.

2. The paper is presented in a clear and well-organized way. Especially, Fig. 1 and 2 helps the readers to better understand the intuition and impact of DML in the discussed setting. 

3. Legitimate ablations are conducted and the results validate the claims.

### Weaknesses
### Weaknesses:

1. The proposed method has been validated only on the ProcGen benchmark. Experiments on more diverse set up is needed to show the applicability of such methods. 

2. While I understand that the target is not to outperform the state-of-the arts, but how DML stands against other data augmentation based approaches such as [1] are not evident. While the authors present result with SPO, it seems SPO performance itself is not upto the current standard. 

3. The proposed method relies on multiple policies for distillation. However, the computational overhead compared to single policy methods is not discussed.

[1] Raileanu, Roberta, et al. "Automatic data augmentation for generalization in reinforcement learning." Advances in Neural Information Processing Systems 34 (2021): 5402-5415.

### Questions
I am wondering how MDPO will scale to large number of policies beyond only two policies as discussed. Can you share some insights on that?

### Soundness
2

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
3

### Summary
This paper presents a theoretical framework to demonstrate that improving the policy robustness to irrelevant features enhances its generalisation performance. The paper further shows that deep mutual learning forms an implicit regularisation and prevents policy from overfitting to irrelevant features. Empirical results are given on the ProcGen benchmark, designed to test generalisability under controlled environments, as well as for toy examples. The proposed method, mutual distillation policy optimisation, demonstrates benefits as compared to the selected baseline approaches.

### Strengths
The paper presents a new theoretical framework to investigate generalisation issues in deep RL. Generalisation in RL is a major and actively researched topic. The insights provided by the paper will have far reaching impact.

### Weaknesses
Although impactful, the experimental evaluation is limited. In the sense that, it doesn't demonstrate the phenomenon exists, beyond testing on the ProcGen benchmark and presenting performance. Also, apart from the toy example. 

There are other methods focusing on distillation (mutual or peer). However, these papers seem not to be mentioned in the paper. It would be good to see a comparison, for example, 

* Periodic Intra-Ensemble Knowledge Distillation for Reinforcement Learning, https://arxiv.org/pdf/2002.00149

* Robust Domain Randomised Reinforcement Learning through Peer-to-Peer Distillation, https://arxiv.org/pdf/2012.04839 (this has been cited in the paper) 

* Online Policy Distillation with Decision-Attention, https://arxiv.org/pdf/2406.05488

While the theoretical framework seems to be strong, the paper lacks interpretability. More experiments would be helpful to understand the impact of mutual distillation on the representation space. 

Running multi distillation may increase the computational overhead, which has not beed discussed in the paper. 

Mutual distillation offers regularisation by reducing the reliance on irrelevant features. However, the generalisability of the approach hasn't been investigated.

### Questions
1) Have the authors analysed or discussed the additional cost introduced by mutual distillation? 

2) Has the method been tested across tasks with different sources of irrelevant variation, or is it specific to the evaluated benchmarks?

3) Can the representational changes be visualised or quantified to support the theoretical claims?

4) Could the authors comment on how their approach differs from or relates to the works mentioned above?

### Soundness
2

### Presentation
2

### Contribution
3
