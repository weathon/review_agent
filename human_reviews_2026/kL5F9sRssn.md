# Landmark-Guided Policy Optimization for Multi-Objective Language Model Selection

- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Selecting a pretrained large language model (LLM) to fine-tune for a task-specific dataset can be time-consuming and costly. With several candidate models available to choose from, varying in size, architecture, and pretraining data, finding the best often involves extensive trial and error. In addition, the "best" model may not necessarily be the one with the lowest test loss, as practical considerations such as deployment costs, inference throughput, and limited search budgets might also play crucial roles. To address this, we introduce LAMPS (LAnguage Model Pareto Selection), a novel and open-source multi-objective AutoML framework that quickly identifies near-Pareto-optimal pretrained LLMs for a task-specific dataset. It is based on two key ideas: (1) landmark fine-tuning, which generates early performance indicators of the candidate models, and (2) meta-learning via reinforcement learning, which learns an effective selection policy from historical performance data (a meta-dataset). Our results show that, on held-out datasets, LAMPS reduces search time by an average of 71% compared to exhaustive search, while still covering more than 98% of the optimal target space hypervolume.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
LAMPS is a language model selection framework that aims to choose the best model for a particular task. The authors claim that the best model is not necessarily the one with the lowest test loss but could be influenced by many other factors (resources, deployment costs). Hence, in the context of AutoML, finding the best model is expensive. They formulate the problem as an RL problem (trained with PPO) with an objective function of finding a Pareto optimal set over the models, by minimizing the joint cost for all factors (model size, inference throughput, deployment cost) (defined as the hypervolume indicator).

### Strengths
- The paper has well formatted definitions and theorems
- The figures are well made
- The problem they choose to solve is interesting.

### Weaknesses
- The related works section could be strengthened. From what I remember, there are works that try to automate neural architecture search, which seems like a similar problem setting. I understand that experiments would be too much to run, but at least including a discussion on whether these methods could be adapted to this task/setting would make the claims stronger.
- It can be expensive to run all the configurations before deciding what model to select.

### Questions
1. Is my understanding correct: the multi-objective RL comes in from the fact that each factor (test loss, model size, inference throughput, deployment cost) is a dimension in the hypervolume indicator ($i=0$ for test, $i=1$ for model size, etc.), and these factors are all aggregated with the Lebesgue measure?
2. Is there any analysis on how the conflicting factors affect the performance of LAMPS? What if there were no conflicting factors? What if there were only conflicting factors?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents LAMPS, a novel multi-objective AutoML framework for selecting pretrained large language models (LLMs). The authors treat the practical challenge as a multi-objective problem, balancing the trade off between model performance, training cost, etd. LAMPS combines landmark fine-tuning, meta-learning via reinforcement learning, which trains a policy on historical model performance. In the experimental results, LAMPS reduces search time by 71% while maintaining coverage of over 98% of the optimal target-space hypervolume.

### Strengths
The paper Introduces a novel method, LAMPS within a multi-objective AutoML framework for efficiently selecting and fine-tuning models along a Pareto front. It combines multi-objective optimization with invalid action masking in RL, which is a novel way to improve exploration efficiency and reduce wasted computation. LAMPS consistently identifies near-Pareto-optimal models faster than baselines

### Weaknesses
The termination condition assumes the agent can detect when all Pareto-optimal models have been fully fine-tuned, which is not practical for the real scenario. 

The RL agent training rely on fully fine tuning trajectories of 70 pretrained models, which limits its applicability to scenarios involving a large number of pretrained models.

The experiments use only nine datasets and two objectives, more experiments would strengthen the generality and demonstrate its performance on more objective trade-offs. It’s unclear how LAMPS scales when more objectives are added.

### Questions
The paper does not clearly define what constitutes a *fully fine-tuned* mode, could the authors specify the exact convergence condition?  

The paper uses a sparse reward defined only at the end of each episode (t = T). Have the authors considered using intermediate rewards at each timestep t, for example, based on the hypervolume of the partially fine-tuned model’s current performance?

### Soundness
2

### Presentation
2

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper addresses the problem of selecting and fine-tuning a subset of pretrained LLMs to approximate the Pareto front across multiple, potentially conflicting objectives, such as test loss and fine-tuning time. The authors formulate this as a hypervolume maximization problem with a cardinality regularization term, and establish a sufficient condition under which the optimizer recovers the true Pareto set.
Their proposed method, LAMPS, uses early segments of model learning curves, termed "landmarks," to predict final performance. These landmarks guide a reinforcement learning allocation policy, trained with Distral-regularized PPO, which decides which model to allocate further fine-tuning resources to. The policy receives a sparse reward that encourages efficient recovery of the full Pareto set, and incorporates mechanisms like invalid action masking to avoid unnecessary computation.
The deployment procedure begins with a zero-shot evaluation of all candidate models, followed by an iterative process of fine-tuning and evaluation under a budget constraint. The final output consists of non-dominated solutions. The method is meta-trained on a large dataset of full fine-tuning trajectories covering 70 models and nine classification datasets. Experiments show that LAMPS reaches 98 percent of the optimal hypervolume more quickly than baseline methods, achieving performance comparable to an oracle on most tasks while reducing wall-clock search time by 71 percent on average.

Using hypervolume to evaluate Pareto sets is standard in multi-objective optimization (MOO) and maximizing it aligns with Pareto optimality, but the paper’s contribution is to meta-learn a policy that allocates scarce fine-tuning budget across models to rapidly approach a good Pareto set in LLM selection.

### Strengths
- The paper introduces a novel framing of model selection and fine-tuning as a policy learning problem that directly optimizes hypervolume. The sparse reward is well-designed and aligned with the set-level objective.

- The formulation combining hypervolume and ℓ₀-style regularization is mathematically sound, with a clear condition on the regularization parameter. The reinforcement learning setup is thoughtful, using Distral for transfer, PPO for stability, and invalid-action masking to improve efficiency.

- The experiments cover a large and realistic search space with 70 pretrained LLMs across nine datasets. Results are reported with respect to wall-clock time, showing a 71 percent average reduction in search time to reach 98 percent of optimal hypervolume. Comparative performance is clearly illustrated in Figure 3.

Deployment details in Algorithm 1 and the inclusion of example commands and a policy checkpoint improve reproducibility and ease of adoption.

- The focus on hypervolume as a set-level objective better reflects practical needs in multi-objective LLM selection and offers a promising path toward reducing compute costs in real-world settings.

### Weaknesses
1.	The initial state and per-step evaluation use D_test, which risks peeking at the test set while adaptively selecting and training models, potentially biasing results. A pure validation set (or cross-validation) should drive decisions; the test set should be reserved for final evaluation only. 
2.	Baselines are underspecified relative to the literature. The paper compares against Blind/ZigZag/Oracle but omits strong, well-known multi-fidelity HPO and early-stopping methods (e.g., Hyperband/ASHA, BOHB [7]) and learning-curve extrapolation (Domhan et al.) [2], which directly trade resource vs. accuracy and are relevant to the same compute-constrained setting. It also omits multi-objective HPO baselines (e.g., scalarization approaches like ParEGO [3], hypervolume-based MOBO, or MO-ASHA [4]). Without these, the magnitude of LAMPS’ advantage is harder to assess [1]. 
3.	Experiments optimize only test cross-entropy and training time for classification tasks. Important deployment objectives (e.g., inference throughput/latency, VRAM, energy, dollar cost, fairness/robustness metrics, factuality metrics, or generation-quality metrics for seq2seq) are untested. The claim of objective-agnosticism would be stronger with such evaluations. 
4.	Ablations / design analyses are missing or light:
o	Effect of landmark schedule (number and spacing) on prediction fidelity and policy quality.
o	Reward shaping alternatives and sensitivity to the terminal-only reward.
o	Role of Distral vs. single-task policies.
o	Effect of invalid-action masking removal.
o	Sensitivity to the reference point (r) for hypervolume. (The literature notes hypervolume can be reference-point sensitive.) [5]
5.	Training uses 8× A100-40GB and advises ≥2 TB disk; this may be a barrier for many labs. Reporting total policy training cost and amortization analysis vs. per-dataset gains would help. 
6.	Relation to hypervolume-maximizing selection. Since LAMPS ultimately maximizes hypervolume of a set, comparisons or discussion against hypervolume-driven selection algorithms (e.g., SMS-EMOA [6]) would contextualize design choices and computational efficiency.

References:
[1] https://arxiv.org/abs/1603.06560
[2] https://www.ijcai.org/Proceedings/15/Papers/487.pdf
[3] https://ieeexplore.ieee.org/document/1583627
[4] https://arxiv.org/pdf/2106.12639
[5] https://arxiv.org/pdf/2005.00515
[6] https://www.sciencedirect.com/science/article/pii/S0377221706005443
[7] https://arxiv.org/pdf/1807.01774

### Questions
1.	Can you re-run the main results with decisions driven exclusively by a validation split (no access to test during policy rollouts) and report test-only results at the end? This would address potential test leakage. 
2.	It could be instructive to include:
o	Hyperband/ASHA (single-objective time-aware early-stopping) configured for either joint scalarization (e.g., weighted sum of objectives) or treating training time as the resource.
o	BOHB (combines BO with Hyperband).
o	Learning-curve extrapolation (Domhan et al.) to decide early continuation/termination.
o	Multi-objective HPO like ParEGO (scalarization BO) or MO-ASHA. How does LAMPS compare in wall-clock to 98% hypervolume? 
3.	Could you add experiments with (i) generation tasks (e.g., summarization), (ii) inference-time/VRAM/energy objectives, and (iii) robustness/fairness/factuality metrics to demonstrate objective-agnosticism?
4.	It may be useful to provide: (a) landmark schedule sensitivity; (b) reward variants (e.g., per-step dense rewards); (c) with/without Distral; (d) invalid-action masking off; (e) reference-point (r) sensitivity.
5.	What reference point (r) is used and how chosen per dataset? Any normalization across objectives?
6.	What is the total compute to train the 45M-step policy, and how many target datasets does it take to amortize that cost compared to (say) BOHB?
7.	Since some Hugging Face models require license acceptance, how is this handled in automated runs (e.g., CI or cluster) to ensure compliance?

### Soundness
2

### Presentation
3

### Contribution
3
