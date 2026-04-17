# One-Token Rollout: Guiding Supervised Fine-Tuning of LLMs with Policy Gradient

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 4, 4

## Abstract
Supervised fine-tuning (SFT) is the predominant method for adapting large language models (LLMs), 
yet it often struggles with generalization compared to reinforcement learning (RL). 
In this work, we posit that this performance disparity stems not just from the loss function, but from a more fundamental difference:
SFT learns from a fixed, pre-collected dataset, whereas RL utilizes on-policy data sampled from the current policy. 
Building on this hypothesis, we introduce one-token rollout (OTR), a novel fine-tuning algorithm that guides SFT with the policy gradient method. 
OTR reframes the autoregressive learning process by treating each token generation as a single-step reinforcement learning trajectory. 
At each step, it performs a Monte Carlo ``rollout'' by sampling multiple candidate tokens from the current policy's distribution. 
The ground-truth token from the supervised data is then used to provide a reward signal to these samples. 
Guided by policy gradient, our algorithm repurposes static, off-policy supervised data into a dynamic, on-policy signal at the token level,
capturing the generalization benefits of on-policy learning while bypassing the costly overhead of full sentence generation.
Through extensive experiments on a diverse suite of challenging benchmarks spanning mathematical reasoning, code generation, and general domain reasoning,
we demonstrate that OTR consistently outperforms standard SFT. 
Our findings establish OTR as a powerful and practical alternative for fine-tuning LLMs and provide compelling evidence that the on-policy nature of data is a critical driver of generalization,
offering a promising new direction for fine-tuning LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces OTR, a training algorithm based on supervised data inspired in RL that treats each token generation as a single-step rollout of the policy. It then uses a reward function that takes the value 1 if the correct token was sampled and a hyperparameter $\beta$ otherwise.

### Strengths
S1) The paper is extremely well-written and clear
S2) The proposed algorithm is shown to improve results on a range of models and reasoning datasets.

### Weaknesses
The main weakness of this paper is that the claim that OTR is a generalization of DFT is false. This method is a Monte-Carlo approximation of the DFT objective which can be computed without the additional variance. The reason is the following. Take the OTR objective, replacing the empirical estimate of the expectation with the expectation is approximating:
1) $$\frac{1}{T}\sum_{t=1}^T E_{a_t\sim \pi_\theta(\cdot|s_t)} {R(a_t, x_t)\nabla_\theta  \log \pi_\theta(a_t|s_t)}$$

Now, consider (a constant baseline):
2) $$\frac{1}{T}\sum_{t=1}^T E_{a_t\sim \pi_\theta(\cdot|s_t)} {\beta \nabla_\theta  \log \pi_\theta(a_t|s_t)} = 0$$

This is because $E_{a_t\sim \pi_\theta(\cdot|s_t)} {\beta \nabla_\theta  \log \pi_\theta(a_t|s_t)} = E_{a_t\sim \pi_\theta(\cdot|s_t)} {\beta \frac{\nabla_\theta \pi_\theta(a_t|s_t)}{ \pi_\theta(a_t|s_t)}} = \beta \sum_a{\nabla_\theta \pi_\theta(a|s_t)} = \beta \nabla_\theta(1) = 0$

Thus, subtracting 2 from 1, we get:
3) $$\frac{1}{T}\sum_{t=1}^T E_{a_t\sim \pi_\theta(\cdot|s_t)} {(R(a_t, x_t) - \beta) \nabla_\theta  \log \pi_\theta(a_t|s_t)}$$

Now, because now all actions $a_t \neq x_t$ have reward 0, we are left with:
4)$$ = \frac{1}{T}\sum_{t=1}^T (1-\beta) \pi_\theta(x_t|s_t) \nabla_\theta  \log \pi_\theta(x_t|s_t)$$

Discarding the proportionality constant we are left with 
5) $$ \propto \frac{1}{T}\sum_{t=1}^T \pi_\theta(x_t|s_t) \nabla_\theta  \log \pi_\theta(x_t|s_t)$$

which is the DFT loss multiplied by an additional factor $\frac{1}{T}$. This factor, however, is biasing the objective, see: Liu, Zichen, et al. "Understanding r1-zero-like training: A critical perspective." arXiv preprint arXiv:2503.20783 (2025).

The above analysis also shows that whatever effect seen by setting different values of $\beta$ should disappear by dividing the learning rate by $1 - \beta$.

Now, I am not saying that because this paper boils down to a high-variance version of DFT it shouldn't be published. DFT is concurrent work and in the current landscape of Machine Learning with new papers pouring out with breakneck speed every week, it is fair to recognize two contributions coming out simultaneously. However, if I haven't made a mistake in my analysis, and please correct me if I did, the paper should be thoroughly restructured in order to acknowledge these points.

### Questions
I don't need to ask any further clarifying questions

### Soundness
1

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes One-Token Rollout (OTR), a novel algorithm for supervised fine-tuning of LLMs that incorporates principles from policy gradient methods in RL. The authors argue that the generalization gap between SFT and RL stems primarily from SFT's use of static, off-policy data, as opposed to RL's on-policy sampling. OTR addresses this by reframing each token prediction as a single-step RL trajectory, performing Monte Carlo rollouts to sample candidate tokens from the model's current distribution, and using the ground-truth token to assign rewards (1 for matches, beta < 1 for mismatches). This transforms SFT data into a token-level on-policy signal, leading to a loss that dynamically weights the ground-truth and penalizes sampled incorrect tokens. Experiments on mathematical reasoning, code generation, and general benchmarks using Qwen-series models show OTR outperforming standard SFT, with better generalization and reduced catastrophic forgetting.

### Strengths
- OTR demonstrates consistent improvements over SFT baselines across diverse benchmarks, including in-domain math tasks (GSM8K, MATH) and out-of-domain code/general reasoning (HumanEval+, MMLU-Pro). Notably, it mitigates performance degradation on non-target capabilities (fewer instances of drops below base model levels, aligning with the goal of preserving pre-trained knowledge.
- The paper's perspective on the SFT-RL gap is relevant and timely. The derivation from policy gradient to token-level loss is well explained and connects to concurrent work like Dynamic Fine-Tuning.

### Weaknesses
- OTR bears strong similarities to established regularization methods in LLMs, such as label smoothing, which softens one-hot targets to prevent overconfidence by redistributing probability mass to non-ground-truth classes. More importantly, the penalty term for sampled incorrect tokens (via beta < 0) closely resembles unlikelihood training (Welleck et al, 2019, which is not cited in the paper), where the model is trained to assign lower probabilities to negative samples drawn from its own distribution to combat degeneration; while framed as an RL innovation, OTR could be viewed as a rediscovery or minor extension of these ideas, and the paper lacks direct comparisons to such priors, potentially overstating novelty.
- Despite positioning OTR as bridging the SFT-RL gap through "on-policy" simulation, the single-token rollout and direct penalization of non-ground-truth samples make the "onlineness" aspect vanish in practice. In full RL (e.g., policy gradients for LLMs), on-policy learning involves multi-step trajectories and exploration of novel behaviors; here, rewards are rigidly tied to fixed ground-truth tokens, reducing it to supervised contrastive learning rather than true online adaptation. This limits the method's ability to discover behaviors beyond the initial data distribution, as noted in related work on policy gradients.
- Relatedly, the one-step nature of OTR makes MC rollouts redundant. Given that SFT already provides access to the full model distribution (logits) and target (ground-truth token) at each step, the expectation in Eq. 4 could be computed exactly using the temperature-scaled softmax (\Pi'), e.g., as \Pi'(xt) log \Pi(xt) + \beta ∑_{a ≠ xt} \Pi'(a) log \Pi(a), which is feasible for typical vocab sizes (~100k) on GPUs. This would eliminate sampling variance and cost (K=256), potentially making OTR more efficient and stable. Why use monte-carlo estimation when exact, efficient, closed-form solution is readily available? The paper does not explore or justify why an approximate MC approach is preferred over the closed-form alternative, especially since DFT achieves similar weighting without sampling.
- While DFT is discussed, broader baselines like unlikelihood training or entropy-regularized SFT are absent, weakening claims of superiority.

### Questions
- Given the single-token framing, how does OTR handle long-range dependencies in sequences, and does the "on-policy" simulation truly enable exploration akin to multi-step RL, or is it effectively off-policy with model-sampled negatives?
- Why use Monte Carlo sampling instead of exact computation over the distribution? Have the authors experimented with a closed-form version of the loss, and if so, how does it perform in terms of stability and efficiency?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Proposes a new language model finetuning method, called one-token rollout (OTR), which repurposes supervised finetuning (SFT) data for on-policy RL. The main idea is to use the next token in an output as a reward signal. For each token, instead of applying the standard cross-entropy loss, OTR samples multiple candidate tokens from the current policy and performs a policy gradient update, where the reward of the ground truth token is set to 1 and the reward of other tokens is set to a lower (negative) value. OTR is demonstrated to generalize better than standard SFT across several benchmarks and base models.

### Strengths
1. Relatively well-written and easy to follow.

2. The idea behind OTR is conceptually simple. I also find that it is sufficiently motivated through existing observations on the benefits of on-policy data.

3. I appreciate the comparison to the concurrent dynamic finetuning (DFT) method in Section 3.4. It helps clarify the distinction of OTR (despite not being strictly necessary due to concurrent work guidelines of ICLR).

4. The empirical evaluation considers several different benchmarks, including in-distribution and out-of-distribution ones.

### Weaknesses
1. It seems that OTR incurs an additional computational cost over SFT due to the need of generating multiple candidate tokens for each token in a response. It would be beneficial to:
    - Report for the existing results what the compute difference (e.g., FLOPS or runtime) was between OTR and SFT.
    - If it is non-negligible, run experiments where both OTR and SFT are allocated the same amount of compute. For example, can let SFT run for more optimization steps until it reaches the same compute that OTR used (based on the same data).
    - Straightforwardly discuss this limitation, specifying the exact computational overhead that OTR incurs compared to SFT.

2. The empirical evaluation compares OTR only to SFT. There are a plethora of methods to finetune language models. Thus, it is difficult to assess the significance of OTR, the sole contribution of this paper. In particular, how does it compare to RL with verifiable rewards (RLVR)? While RLVR employs a different type of supervision (reward on answer vs a demonstration of a reasoning chain and answer), I believe this is still a necessary comparison given that verifiable rewards in the domains considered can be easier to obtain than high quality demonstrations.

3. The gains of OTR over SFT are in many cases quite small. The paper does not report standard deviations, or any other measure of statistical significance. So it is unclear in how many of these cases the difference is just due to noise.



Review Summary and Recommendation
---

Overall, I find this paper to be on the borderline. On the one hand, it is relatively well-written and proposes a simple (in a good sense) method for converting SFT into an on-policy method. However, the empirical evaluation is somewhat lacking. I would be willing to reconsider my score if the authors are able to treat the comments above regarding the empirical evaluation and additional compute required by OTR.


Additional (More Minor) Comments
---

1. In the SFT definition (Equation (1)), conditioning on the prompt is missing. Also, as far as I am aware, the standard formulation does not normalize across the length of the sequence, rather across the number of sequences.

2. The explanation starting from line 151 on the use of temperature for sampling is rather standard. It is a subjective matter, but in my opinion it can be significantly shortened (e.g., there is no need to define the alternative sampling policy and explaining the generation process, rather it is possible to just say that a temperature greater than 1 is used to encourage exploration).

3. I do not believe it is necessary to write a stop gradient around the reward in Equation (8). At least, it is not standard to do so.

### Questions
--

### Soundness
3

### Presentation
3

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
This paper describes a method to improve on supervised fine-tuning (SFT) of large language models (LLMs) by doing one-token rollouts at every subsequence of an SFT dataset. The method samples K token completions of at every step of every sequence in the training dataset, and defines a REINFORCE-like loss where the reward is the fraction of K tokens that match the next ground truth token. 

Qwen models are fine-tuned using both the one-token rollout (OTR) method and SFT on a math training dataset, and evaluated on a wide suite of math and code generation benchmarks. Results show that OTR generally improves over SFT.

Ablation experiments show the importance of adding a slightly negative reward on tokens that are not equal to the ground-truth next token helps with training stability and improves performance.

### Strengths
1. The paper presents a simple way of implementing on-policy REINFORCE-like improvement of LLMs without doing full reinforcement learning, such as using GRPO. The method is quite simple to implement given an SFT training dataset.

2. Results show improvements over SFT fine-tuning for Qwen models, across a wide set of math and code generation benchmarks. The paper reports out-of-domain generalization improvements with OTR training as well. 

3. Ablations are included to show the necessity of using a small negative reward for incorrectly generated tokens.

### Weaknesses
1. One key missing aspect of this paper is a measure of complexity of the proposed method. In the introduction the paper claims that it's method "improves model generalization without incurring the computational cost of full sentence generation." However, as presented in the paper, for every training set sequence of length T tokens, OTR samples K tokens, which equals a total of T*K sampled tokens. This is the same as sampling K length-T completions for the prompt corresponding to the training set sequence. Note that K=256 is quite high in the paper experiments.

It is difficult to evaluate the claims of this work without any analysis of its additional complexity, which is not in the current paper.

2. Related to Weakness (1), SFT is not the correct baseline for OTR. Since the method takes on-policy samples (and from the above analysis requires TK samples per training dataset sequence), an online RL baseline is more appropriate. For example, an RL run which samples 1 full trajectory per training dataset sequence (requiring T samples), performs an agent update iteration, then repeating this K times, would match the compute used. It would be much more helpful to compare the performance improvement achieved by OTR with this baseline to evaluate the effectiveness of this method.

3. The paper only trained using Qwen models. This limits the evaluation of the generality of the proposed method across model types.

### Questions
In "training details" it is mentioned that OTR and SFT were trained using identical settings. Why is this? Shouldn't each setting be optimized in order to obtain the highest improvement? 

Why are only Qwen models used? Why not other open-source models with similar number of parameters?

In equation (6), why does the policy \pi have a ' mark?

### Soundness
3

### Presentation
3

### Contribution
2
