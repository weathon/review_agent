# CDE: Curiosity-Driven Exploration for Efficient Reinforcement Learning in Large Language Models

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 0

## Abstract
Reinforcement Learning with Verifiable Rewards (RLVR) is a powerful paradigm for enhancing the reasoning ability of Large Language Models (LLMs). Yet current RLVR methods often explore poorly, leading to premature convergence and entropy collapse. Moreover, they tend to produce poorly calibrated policies that remain confident in their generations regardless of correctness. To address this challenge, we introduce **Curiosity-Driven Exploration (CDE)**, a framework that leverages the model's intrinsic sense of curiosity to guide exploration. We formalize curiosity with signals from both the actor and the critic: for the actor, we use perplexity over its generated response, and for the critic, we use the variance of value estimates from a multi-head critic architecture. Both signals serve as an exploration bonus within the RLVR framework to guide the model. Our theoretical analysis shows that the actor-wise bonus inherently penalizes overconfident errors and promotes diversity among correct responses; moreover, we connect the critic-wise bonus to the well-established count-based exploration bonus in RL. Empirically, our method achieves an approximate **+3** point improvement over standard RLVR using GRPO/PPO on AIME benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes Curiosity-Driven Exploration (CDE) to improve exploration and calibration in RL-based fine-tuning of large language models. The method adds two intrinsic signals: a per-response perplexity bonus for the actor and a variance-based uncertainty bonus across bootstrapped value heads for the critic. Both are clipped and gradually reduced. Experiments on math-reasoning datasets with a Qwen-3-4B model show better accuracy and more stable confidence behavior. The contribution is mainly algorithmic and empirical, with a small theoretical part that connects the critic variance to UCB-style exploration.

### Strengths
- Idea that could have a large impact on the LLM community, which is easy to implement on top of existing PPO/GRPO pipelines.
- Sample-specific actor bonus helps control over-confidence and keeps diversity among correct answers.
- Critic variance is simple but works well; the UCB connection gives intuition.
- Results are consistent across multiple datasets; training curves and calibration plots are convincing.
- The paper is clear and well structured; contains ablation studies.

### Weaknesses
- Limited scope: all tasks are math; no larger model or other domains such as code or factual QA. How useful is it on other tasks?
- No compute/cost analysis: multi-head critics increase memory and time, but this is not measured.
- No failure cases or examples showing what happens if exploration weight is too high.
- Reproducibility: no public code/configs mentioned.
- Lacks references to traditional curious-driven approaches employed in RL community. See references for examples

@InProceedings{pmlr-v260-bougie25a,
  title     = {Exploring Beyond Curiosity Rewards: Language-Driven Exploration in RL},
  author    = {Bougie, Nicolas and Watanabe, Narimasa},
  booktitle = {Proceedings of the 16th Asian Conference on Machine Learning},
  pages     = {127--142},
  year      = {2025},
  volume    = {260},
  series    = {Proceedings of Machine Learning Research},
  month     = {December},
  publisher = {PMLR},
  url       = {https://proceedings.mlr.press/v260/bougie25a.html}
}

@inproceedings{burda2019rnd,
  title     = {Exploration by Random Network Distillation},
  author    = {Burda, Yuri and Edwards, Harrison and Pathak, Deepak and Storkey, Amos and Darrell, Trevor and Efros, Alexei A.},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2019},
  url       = {https://openreview.net/forum?id=H1lJJnR5Ym}
}

@inproceedings{osband2016bootstrapped,
  title     = {Deep Exploration via Bootstrapped {DQN}},
  author    = {Osband, Ian and Blundell, Charles and Pritzel, Alexander and Van Roy, Benjamin},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2016},
  url       = {https://proceedings.neurips.cc/paper_files/paper/2016/file/8d8818c8e140c4643106b7c59d6a9b2c-Paper.pdf}
}

### Questions
- Are the value targets trained with extrinsic-only rewards? 
- How are the bootstrap heads decorrelated (data resampling unit, optimizer states, seeds, dropout)?
- Did you try actor-only and critic-only? Are the gains additive?
- Could you add multiple seed averages / confidence intervals on the main tables?
- Could you compare the approach with existing prior work in curiosity-driven RL?

### Soundness
2

### Presentation
2

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
This paper introduces 'Curiosity-Driven Exploration' (CDE) to improve RLVR exploration. It proposes two bonuses: perplexity (PPL) for GRPO and multi-head critic variance for PPO. The method is shown to fix a "calibration collapse" where baselines become overconfident in their errors.

### Strengths
* The proposed PPL bonus is a simple, intuitive, and well-motivated method for exploration. The paper provides compelling evidence that this simple bonus directly fixes the identified calibration collapse.
* The paper is well-written and easy to follow.

### Weaknesses
* The experiment setting is a little vague. It is not clear in Table 1 whether the experiments on PPO with description `w/ x Heads` are combined with PPL. 
    * If yes, why `w/ 2 Heads` is worse than `w/ PPL`?
    * If not, had the authors tried to combine PPL with a multi-head critic, and what are the results? 
* The discovery that a "Staircase" decay (i.e., a hard stop) for the PPL bonus is optimal (Table 2) is counterintuitive and feels ad hoc. The paper would be stronger if it investigated why a hard cutoff is so much better than a smooth anneal.
* The paper's argument for dismissing count-based exploration seems contradictory. Section 3.1 argues that hash-based counting fails due to the "poor expressiveness" of LLM embeddings. However, the proposed multi-head critic method (Section 3.3) fundamentally relies on these same LLM hidden-state representations to feed its value heads. The paper does not resolve why these representations are considered too poor for hashing but sufficient for learning complex value functions.

### Questions
Please refer to the weaknesses.

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper first conducts preliminary experiments showing that counting-based exploration methods using response embeddings are ineffective for LLM RL and then proposes Curiosity-Driven Exploration (CDE) to guide exploration during LLM RLVR.
The framework contains two parts:
   - exploration guided by actor curiosity: uses the negative seq logprob for reward shaping, with clipping and scaling coefficient to avoid reward hacking.
   - exploration guided by critic curiosity: uses a critic model with multiple heads, each trained on a subset of data, and uses the ensemble value for GAE while taking the standard deviation across heads as an exploration bonus.

Experiments mainly on Qwen3-4B-Base and math demonstrate the effectiveness of CDE, and additional experiments show that actor curiosity (negative logprob) can improve calibration.

### Strengths
- The paper is easy to follow, with clear motivation and structure. The SimHash attempt is also informative.
- Experiments show nontrivial improvements over vanilla GRPO and PPO baselines, supported by ablations on specific hyperparameters.
- The calibration analysis is also a bonus.

### Weaknesses
- Using the negative log-probability for reward shaping is not entirely new. The same formulation has been explored in related work [1], and it is also closely related to entropy shaping, which is also widely explored in related works [2, 3]. The clipping mechanism involving $\lvert A \rvert / \kappa$ is also conceptually similar to that in [3]. While direct comparisons are not strictly required by policy, discussing these related approaches would help position the contribution of this paper more clearly.
- The proposed framework introduces several new hyperparameters, yet only the ablations for coefficient scheduling for GRPO and sub-sampling fraction for critic update are provided. Given the number of hyperparameters, it would be worthwhile to conduct a more comprehensive sensitivity analysis.
- All experiments are conducted on Qwen3-4B-Base and math datasets. While I understand that it might be computationally expensive to scale up, it would be worthwhile to at least conduct experiments on other model families. To be honest, I won't be fully convinced by "Qwen + math" combination. Also considering the current field with various GRPO variants, in my opinion, vanilla GRPO is a weak baseline.

[1] Know When to Explore: Difficulty-Aware Certainty as a Guide for LLM Reinforcement Learning, arxiv 2025.

[2] GTPO and GRPO-S: Token and Sequence-Level Reward Shaping with Policy Entropy, arxiv, 2025. 

[3] Reasoning with Exploration: An Entropy Perspective on Reinforcement Learning for LLMs, arxiv 2025.

### Questions
Please refer to the weakness section for main questions.
 - I am a bit confused about the training setup in this paper. Are you doing on-policy or off-policy training? The entropy curve of GRPO w/o PPL in Figure 11 looks somewhat abnormal to me, I would expect a higher entropy level if using clip-higher in off-policy setting. And clip-higher should be a stronger baseline. So I checked the hyperparameters table, and based on the bs and mini_bs, it appears to be on-policy, but clip_ratio is also listed, which typicality has no effect in on-policy training. Could you clarify the exact training setting used in these experiments? If it is indeed on-policy, have you also validated your methods under an off-policy setting?
- The footnote explaining PPL bonus should probably be moved to the first page, since you start using this notation in the introduction, but the clarification appears a little late.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
+ Summary & Contributions
	- The authors focus their attention on the exploration challenge in RL fine-tuning of LLMs.
	- Following suit with curiosity-driven exploration strategies in deep RL, the authors propose their own curiosity-driven exploration methods for RL fine-tuning. 
	- Following an actor-critic setup for RL fine-tuning, the authors propose using the perplexity of the actor's response distribution as an intrinsic reward for guiding exploration. Meanwhile, the authors employ a critic ensemble with bootstrap sampling to approximate the true Bayesian posterior over the value function induced by the current actor policy and use the corresponding ensemble variance as an intrinsic reward to further guide exploration.
	- A theoretical result is provided to show that the use of the ensemble critic standard deviation to guide exploration is an asymptotically consistent estimator of the elliptical potential bonus used by LSVI-UCB (a statistically-efficient exploration algorithm) under a linear MDP assumption.
	- Empirical results show that the proposed curiosity-based reward for exploration yield improved performance relative to standard GRPO and PPO.

### Strengths
+ Quality
	- Strengths
		- The authors display a nice insight in trying to leverage ideas for exploration strategies from deep RL to improve LLM exploration during RL fine-tuning.
	- Weaknesses
		* Major
			- There is absolutely no scientific evidence offered in this paper to support the assertion that LLM reasoning aligns with models of childhood exploration in developmental psychology (L69-72, L151-155). This kind of hyperbole is also rather pointless as the authors could just introduce their method without unsubtantiated nonsense to give their method the veneer of being grounded in cognitive science. As the authors already point out, there is much deep RL work leaning on ideas of curiosity which already serves as a sufficient foundation to warrant their study in the context of LLMs and RL-based fine-tuning.
			- In light of the authors' proposed curiosity signals as intrinsic rewards, they proceed to augment the computation of advantages (L273). This runs counter to how intrinsic motivation is normally applied in (deep) RL, where the underlying policy optimization algorithm (PPO, GRPO, etc.) remains entirely unchanged and is simply run on the augmented MDP whose reward function has been modified to include the original environment reward plus a (potentially) weighted intrinsic reward term. Why did the authors feel compelled to deviate from this recipe? Why is it necessary or even sensible to modify the definition of the advantage function based on curiosity?
			- In motivating their proposed actor curiosity signal based on surprisal (not perplexity -- see minor comment below), the authors claim that a response with low probability under the current policy "resides in an underexplored region of its learned distribution." An alternative but equally (if not more) plausible possibility is that this response, either during the initial pre-training stage or in earlier rounds of fine-tuning, was determined to not yield correct responses, lead to negative advantages, and has accordingly had its policy of being generated reduce through successive policy-gradient updates. How are the authors able to take this surprisal curiosity signal and disambiguate between inexperience with a particular response from the implausibility of that response being optimal/correct? The fact that they can't seem to distinguish between those two critical scenarios is perhaps what warrants the subsequent clipping scheme (Equation 2).
			- The proof of Theorem 3.2 isn't actually a complete proof. There are numerous points where the authors simply claim a particular equation or inequality without any kind of explicit, step-by-step justification. Where are the steps to establish how Assumption A2 leads to $||\Lambda^{-1}_{n,t}||_{\text{op}} = O_p(\frac{1}{n})$? Where did the expansion in L895 come from? It seems like a consequence of the Sherman-Morrison-Woodbury matrix identity, but the whole point is to give a complete proof detailing those steps. Equation 4 is just stated as a matter of fact without any concrete steps as is the subsequent inequality. The authors vaguely gesture towards "finite-population sampling theory" without so much as a single citation of an existing result being utilized to then state an equation for "Var*" (which denotes some entirely unspecified variance -- why *?). Overall, the proof is entirely incomplete and, more worryingly, seems to be an amalgamation of equations and inequalities likely taken from another proper RL theory paper (the LSVI-UCB paper perhaps) just to inject math into the paper for the sake of justifying the authors heuristic approach (see "Mathiness" in [2]).
		* Minor
			- What the authors continuously refer to as the perplexity of the actor is, in reality, just the surprisal [1]. The authors even acknowledge that what they refer to isn't actually the perplexity in a footnote (L215). What is the point of erroneously labeling the surprisal as the perplexity?
			- While being able to give names to each of the injected hyperparameters in Equation 2 seems useful, it results in an unnecessarily convoluted equation. Either there is an $\frac{\omega}{\kappa}$ term which can be consolidated into a single hyperparameter or a $\omega \cdot \alpha$ term which can also be consolidated. Why can't Equation 2 follow the standard form of $r(q,o) + \lambda \cdot B_{\text{actor}}(q,o)$?
			- The authors claim that incorrect responses recieve a "larger penalty" by virtue of a smaller PPL bonus. Notice that a larger penalty is not the same thing as smaller reward bonus; relative to other actions, this might still end up inducing a positive advantage an increasing probability mass for incorrect responses. Also, it seems like the convolute structure of Equation 2 means this could be undone anyways through the $\kappa$ parameter?
			- Calling Theorem 3.1 a theorem is rather generous. The authors have stepped through the possible cases for the signs of log-probability differences.
			- The authors mention a "posterior distribution" when discussing actor-critic methods. What is this posterior distribution over? Where did it come from? Actor-critic methods do not engage with any kind of Bayesian posterior distribution by default.
			- There is some exposition concluding Section 3.2 where the authors distinguish their incorporation of a surprisal exploration bonus as distinct from traditional entropy regularization. I agree with this, however what they do end up convincing me of is that their proposed surprisal bonus may be recovering (either entirely or partially) the objective tackled by maximum entropy (maxent) RL approaches. In this case, a more appropriate baseline to compare to would be Soft Actor Critic (SAC) [3], rather than PPO.
			- It's not clear why averaging is the correct thing to do for deriving an alternative GAE for the ensemble critic. If each ensemble member represents a distinct hypothesis about the underlying $V^\pi$ then its not clear why averaging would be sensible. If anything, a posterior distribution over $V^\pi$ implies a posterior over $Q^\pi$ and, consequently, the advantage function itself.

+ Clarity
	- Strengths
		- Overall, the paper is reasonably written and reasonably structured.
	- Weaknesses
		* Major
			- N/A
		* Minor
			- The presented results for a hasing based technique seem rather out of place for this work. At the very least, it seems like the space could be better utilized for some other purpose more central to the main contributions of this work and the presented result could be relegated to the appendix with little impact on the paper.
		

+ Originality
	- Strengths
		- The incorporation of the specific proposed curiosity signals to enhance exploration in RL fine-tuning of LLMs is, to the best of my knowledge, novel.
	- Weaknesses
		* Major
			- It would be surprising to me if the authors are in fact the first to propose intrinsic motivation as a remedy for the well-known entropy loss/response distribution collapse issues known to plague RL fine-tuning of LLMs. Have the authors done a proper literature review for other intrinsic rewards aimed at remedying the same problem. Some of these seem to have been identified in Section 5 and yet not compared against as baselines in the reported experiments.
			- While curiosity-based exploration and intrinsic motivation are two remedies for facilitating better exploration in RL, they are by no means the only mechanisms for doing so. Indeed, existing work has already explored other avenues for addressing the same exploration challenge using different techniques [4,5]. The authors have done the bare minimum in their experiments, offering standard PPO/GRPO as baseline comparisons. A more thorough empirical evaluation would properly compare against other competing, alternative exploration strategies that go outside the avenue of curiosity/intrinsic motivation advocated for in this work.
		* Minor
			- N/A

+ Significance
	- Strengths
		- The reported results would seem to confirm that, up to suitable hyperparameter tuning, the proposed curiosity signals do enhance exploration for RL fine-tuning to result in improved performance beyond GRPO/PPO.
	- Weaknesses
		* Major
			- Across the tasks, the gains in performance seem quite small numerically. Moreover, there is no reporting of how many trials/random seeds are used to obtain these results leaving ambiguous the questions of whether the proposed curiosity metrics consistently enchance exploration in a way that yields statistically-significant performance improvements. Also, given the small size of improvements, whether the proposed curiosity signals are worthwhile given the potentially laborious grid search needed to find suitable hyperparameter values.
		* Minor
			- N/A
			
		
+ Final Remarks
	- Overall, I have identified critical issues with this paper on the axes of quality, significance, and originality. While the initial idea has some potential, there is much more work needed on the technical details and presentation as well as the empirical support for the method before publication.

### Weaknesses
Please see above.

### Questions
Please see above.

### Soundness
1

### Presentation
2

### Contribution
1
