# Intention-Conditioned Flow Occupancy Models

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 6, 8, 2

## Abstract
Large-scale pre-training has fundamentally changed how machine learning research is done today: large foundation models are trained once, and then can be used by anyone in the community (including those without data or compute resources to train a model from scratch) to adapt and fine-tune to specific tasks. Applying this same framework to reinforcement learning (RL) is appealing because it offers compelling avenues for addressing core challenges in RL, including sample efficiency and robustness. However, there remains a fundamental challenge to pre-train large models in the context of RL: actions have long-term dependencies, so training a foundation model that reasons across *time* is important. Recent advances in generative AI have provided new tools for modeling highly complex distributions. In this paper, we build a probabilistic model to predict which states an agent will visit in the temporally distant future (i.e., an occupancy measure) using flow matching. As large datasets are often constructed by many distinct users performing distinct tasks, we include in our model a latent variable capturing the user intention. This intention increases the expressivity of our model, and enables adaptation with generalized policy improvement. We call our proposed method **intention-conditioned flow occupancy models (InFOM)**. Comparing with alternative methods for pre-training, our experiments on $36$ state-based and $4$ image-based benchmark tasks demonstrate that the proposed method achieves $1.8 \times$ median improvement in returns and increases success rates by $36\\%$.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The work provides a way to pre-train a general RL occupancy model using an unlabeled dataset, using a technique close to the newly introduced Temporal-Difference Flows combined with variational inference over the "intention" latent variable, which helps learn the final occupancy model. Given the pretrained flow occupancy model, the authors proposed a way to train a new policy and its Q-value using generalized policy improvement and a Q-value estimate from a reward model and an occupancy measure model. The authors provided extensive experimental validation of their pre- and post-training pipeline across various robotics benchmarks, as well as various ablation studies on the algorithmic choices.

### Strengths
- The paper is very well-written and describes the whole training pipeline in great detail.
- The approach bridges the well-developed flow-matching literature with an RL pretraining and subsequent fine-tuning, which distinguishes this approach from prior work on TD-flows.
- The approach of modeling the policy intentions as latent variables seems to be fresh and interesting, since it doesn't require any additional supervision, and distinguishes this approach from Multi-Task RL.
- Strong performance on many benchmarks;

### Weaknesses
- The final algorithm combines four (4) neural networks and, at first glance, looks extremely complicated, which can be prone to the accumulation of errors;

### Questions
- In Appendix C.1., in the derivation of the ELBO loss, does an inequality (c) (line 1155) require $\lambda \geq 1$ for this derivation? 
- Did the newly trained reward, Q-value, and policy models use the features learnt by the occupancy model?
- The improved effect of using the behavior cloning suggests that the dataset consists of high-quality data that is worth being "cloned". What is the issue if the dataset consists only of highly suboptimal but diverse rewards? And how strongly does the diversity of the dataset influence the pre-training performance?
- What are the results on the occupancy measure pretraining performance separately, compared to a standard TD-flows without intention decoding?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes InFOM, a method for unsupervised pre-training in offline reinforcement learning that learns a latent-variable generative model of long-horizon state occupancy conditioned on inferred user intentions. By combining variational inference with flow matching, the authors model the discounted occupancy measure. During fine-tuning, they use implicit generalized policy improvement (GPI) via expectile distillation. The method shows strong empirical gains over prior pre-training approaches.

### Strengths
1. Novel integration of flow matching with intention-aware occupancy modeling.

2. Strong empirical results: consistent gains across diverse domains, including challenging sparse-reward and vision-based tasks.

3. Well-motivated design choices, especially the use of SARSA-style bootstrapping and expectile-based implicit GPI to stabilize training.

### Weaknesses
1. Choice of Prior over Intentions Lacks Justification. The paper assumes a standard Gaussian prior $p(z)=N(0,I)$ for the latent intention $z$ . While common in VAEs, this choice may be suboptimal for modeling user intentions, which are often discrete or categorical (e.g., “pick”, “place”, “navigate to A”).

Suggestion: The authors should consider discrete latent variables (e.g., via Gumbel-Softmax) to improve interpretability and align better with the semantics of “intentions” in multi-task or goal-directed settings. Ablations comparing continuous vs. discrete priors would strengthen the modeling claim. The current Gaussian prior may encourage over-smoothed representations, potentially conflating distinct behavioral modes (as hinted in Fig. 4, where InFOM already shows better clustering—but could it be sharper with discrete codes?).

2.  Incorrect ELBO Formulation: The parameter $\lambda$ should be at least one, not arbitrary. In equation (3), the coefficient $\lambda$ must satisfy $\lambda \geq 1$ to guarantee that the derived expression is the lower bound.

### Questions
Q1: How well does the flow occupancy model generalize to out-of-distribution intentions? 

Q2: Is there a risk that the generative occupancy model produces unrealistic futures for novel $z$, harming policy learning?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This work leverages recent advances in generative models in the direction of flow-matching algorithms in order to tackle the paradigm of pre-training and finetuning RL models. This is done by learning occupancy models over the state space using flow matching (as done in Farebrother et al. (2025)) while  taking into account the fact that large pretraining datasets usually contain a mixture of intentions since they are collected by different users. The proposed approach InFOM explicitly models the intention as a latent variable using a VAE which is used to condition the occupancy model. While fine-tuning, the method generates intention-conditioned Monte-Carlo estimates of  the crictic from sampled future states and then distills them into a single critic via an upper-expectile loss. The authors show that across 36 state-based and 4 image-based tasks ), InFOM matches or outperforms strong pre-train-and-fine-tune baselines, reporting a 1.8× median return gain and a 36% success-rate increase, with particularly large gains on harder manipulation domains.

### Strengths
* The paper is well written and easy to follow
* The experimental results showing the proposed approach outperforms the baselines in most of the case with the gap widening on more difficult tasks
* A large number of baselines have been included and sufficient experimental detail is provided.
* I really the like the qualitative analysis of the learnt latent intention model.

### Weaknesses
The proposed approach builds on top of Farebrother et al. (2025) which uses flow matching to learn occupancy models, by explicitly modeling intention as a latent variable. Including an ablation / baseline comparing the downstream performance with intention conditioning of the occupancy model vs without conditioning it seems to be missing.

### Questions
Is there prior work one using other generative modeling approaches, specifically diffusion models for learning occupancy models? If yes, could the authors provide some intuition on comparison between using diffusion versus flow matching for learning occupancy models?

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
2

### Summary
The present work leverages flow-matching models to predict trajectories of future states for RL tasks to help the actor finding better policies.
Extensive experiments shows on gym and robotic manipulation benchmarks  shows promissing results.

### Strengths
- The idea of leveraging "intent" for multi-step generation especially in the multi-agent setting or iterative refininement with human input. 
- Extensive experiments show practicability of the method.

### Weaknesses
Major

 After several reading, I fail to understand how the algorithm works. It should not be that hard to understand: 
- flow-matching to predict future states is straightforward given trajectories either conditionned or guided on observed states.
- the predicted latent variable has to be used somehow by the policy, either by boosting an pre-trained one or by training from scratch.  

 However, the main body of the paper discusses Q-functions which in the end are conditioned on the intention. This leads the reader to infer that the intention variable is infered and fed to the agent along actor trajectories. 

My wild guess: since the policy is mainly trained via a behaviorial cloning loss (see algorithm 2 in appendix) which is barely mentioned in the main body,  using the predicted state actions, the flow-matching is merely doing data-augmentation on the off-line dataset. 

I thus think that the abstract is misleading, sections 2 and 3 should be rewritten entirely to focus method rather than the intention (ironically). Algorithm 2 is rather involved and should be explained fully in the main body of the paper, all losses should be introduced clearly. 


Minor 

1- Diffentiating through an ODE solver is associated to a citation to Park et al. 2025b.  However, this is a problem already accounted for in the Normalizing flow litterature, see FFJORD https://arxiv.org/abs/1810.01367 or even Neural ODE https://arxiv.org/abs/1806.07366. Considering that Ricky Chen is a major contributor to both flow matching and Normalizing flow literature, it is odd to cite Park et al for this aspect.

2- "The deterministic nature of ODEs equips flow-matching methods with simpler learning objectives and faster inference speed than denoising diffusion models (Lipman et al., 2023; 2024; Park et al., 2025b)" is problematic for two reasons. I doubt that the training objective is simpler for flow-matching compared to Diffusion model if by that the authors mean the target function to learn or even the numerical stability of the loss. It is true however that the numerical stability of the *inference* is better for FM compared to DDPM. It is also true that the MSE-based loss of FM is more numerically stable and lighter than the KL-based loss of ODE-based Normalizing flows   such as FFJORD, see for instance https://arxiv.org/abs/2107.07232. The second reasin is that Park et al. 2025b has little to do with the statement.

### Questions
1- How does the inference work ? No algorithm is provided. 

2- How do you intend to use this method for pre-trained policies ?

### Soundness
3

### Presentation
1

### Contribution
3
