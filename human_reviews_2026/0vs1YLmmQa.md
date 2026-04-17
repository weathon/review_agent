# Principled and Tractable RL for Reasoning with Diffusion Language Models

- Decision: Reject
- Scores: 2, 4, 2, 0

## Abstract
Diffusion large language models (dLLMs) are a new paradigm of non-autoregressive language models that are trained to predict multiple tokens in parallel and generate text via iterative unmasking. Recent works have successfully pretrained dLLMs to parity with autoregressive LLMs at the 8B scale, but dLLMs have yet to benefit from modern post-training techniques, e.g. reinforcement learning (RL), that have proven effective for autoregressive models. Crucially, current algorithms aren't directly compatible with diffusion models due to their lack of left-to-right sequence likelihood factorization. Moreover, existing attempts at dLLM post-training with RL rely on unprincipled heuristics such as mean-field approximations. In this work, we present Amortized Group Relative Policy Optimization (AGRPO), an on-policy RL algorithm designed specifically for dLLMs. Our key insight is that by casting the denoising process as a multi-step Markov decision process, we can use Monte Carlo sampling to compute an unbiased policy gradient estimate, making AGRPO the first tractable yet faithful adaptation of policy gradient methods for dLLMs. We demonstrate AGRPO's effectiveness on different math/reasoning tasks, achieving up to +10.0\% absolute gain on GSM8K, 3.8x performance on the Countdown task over the baseline LLaDA model, and 3.4x performance gains over comparable RL methods such as diffu-GRPO. Furthermore, these gains persist across different numbers of sampling steps at inference time, achieving better tradeoffs between compute and performance. Our results establish that online RL algorithms can be extended to diffusion LLMs in principled ways, maintaining both theoretical soundness and practical effectiveness.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper studies the problem of on-policy reinforcement learning for diffusion language models. In dLLMs, computing marginal token probabilities is intractable, making autoregressive LLM RL objectives not directly applicable. The authors point out the approximations and bias in existing diffu-GRPO and UniGRPO objectives. To tackle the issues, the authors proposes Amortized GRPO (AGRPO), which rewrites GRPO’s tokenwise inner sum as an expectation over timesteps and estimates it with Monte-Carlo samples, resulting in an unbiased policy-gradient estimator. The objective also keeps the KL term inside the expectation approximated with the Schulman estimator. The method is paired with practical tricks (caching partially masked states, LoRA) for efficiency. AGRPO is tested with LLaDA-8B-Instruct across GSM8K, MATH, and Countdown, improving over the base model and outperforming diffu-GRPO and UniGRPO.

### Strengths
* The paper outlines the problem fairly clearly, outlining issues with existing methods and proposing a relatively straightforward algorithms to address them. 
* The authors also provide some details about the practical considerations which are useful. 
* While I have various concerns over the empirical setup detailed below, the results seem promising.

### Weaknesses
* The paper misses some closely related prior work on RL fine-tuning of diffusion language models [1, 2, 3]. I believe a comparison to these baselines would be critical. (There are several other recent papers studying the same problem but they count as concurrent work, so I do not expect the authors to compare with them) 
* There are several explanations and references in the paper that are incorrect:
    * L184: For early work on RL with LLMs the authors refer to the WizardLM paper as a reference for the use of PPO. However, there were quite a few papers [e.g. 4] prior to that.
    * Initial work on RL with LLMs was also focused on preference learning rather than reasoning ability in math or code.
    * L190: Learning reward models is still done for preference tasks and it is usually decoupled from the size of the model being trained so it is not as affected by the growing model sizes.
* In terms of baselines, simple test-time scaling methods like majority voting on the base-model are missing.
*  A lot of details around the actual runtime (e.g. how many steps of training, comparison of training time vs baselines) are missing. Additionally it seems that the numbers in Table 1 are the result of a single seed of training and single seed for evaluation. Please clarify if this is incorrect. I understand training with multiple seeds can be expensive but I atleast expect multiple seeds for eval.
*  There are no ablations or analysis to investigate factors affecting how the method works. For instance, analysis on the effect of k, m, n (during training), effect of low confidence remasking. 
*  Finally, the results are limited to a single model with 3 math-based tasks which is hard to get general insights from. 

[1] Venkatraman et al., 2024. Amortizing intractable inference in diffusion models for vision, language, and control. 

[2] Huang et al. 2025. Reinforcing the Diffusion Chain of Lateral Thought with Diffusion Language Models.

[3] Zekri and Boullé, 2025. Fine-Tuning Discrete Diffusion Models with Policy Gradient Methods.

[4] Ziegler et al., 2019. Fine-Tuning Language Models from Human Preferences.

### Questions
* Please clarify if the numbers are averaged over multiple seeds.
* In Fig 3, why are the numbers for the base model for different response lengths missing?

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work presents Amortized Group Relative Policy Optimization (AGRPO), a novel on-policy reinforcement learning algorithm tailored to advance the reasoning abilities of Diffusion Large Language Models (dLLMs).  This paper identify a key limitation in current RL approaches: while proven effective for autoregressive LLMs, they become computationally prohibitive for dLLMs, and existing dLLM-specific methods often resort to heuristic approximations that introduce bias into policy gradients and lack theoretical grounding. To address this, AGRPO reformulates the token-wise summation in the policy gradient objective as an expectation, enabling unbiased and efficient estimation through Monte Carlo sampling over generation timesteps. Empirical evaluations on mathematical reasoning tasks, including GSM8K, show that AGRPO substantially outperforms both baseline models and prior RL techniques, offering a theoretically rigorous framework that not only elevates dLLM performance but also rebalances the interplay between computational efficiency and output quality.

### Strengths
1. The experimental results are impressive, achieving high performance on the GSM8K benchmark (86.3) and matching the performance of models such as DeepSeekMath-Base 7B.
2. The core idea is novel and well-motivated, with several practical considerations incorporated into the methodology. I found the approach insightful and instructive.

### Weaknesses
1. While the idea is interesting, the writing requires improvement in several areas. Descriptions in many parts remain unclear, and the experimental section appears somewhat brief. The paper would benefit from more comprehensive ablation studies—for instance, to validate the impact of the practical considerations mentioned. A restructuring of the paper is also important to better align with academic writing standards.
2. The version of paper currently lacks a dedicated "Related Work" section, which is essential for contextualizing the contribution within the existing literature.
3. Related to the first point, the current version reads more like a technical report than a fully developed academic paper.

### Questions
1. Can this method be effectively adapted to other diffusion-based language models, such as Dream 7B?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents AGRPO (Amortized Group Relative Policy Optimization), a rl algorithm designed to post-train diffusion language models on reasoning tasks by reformulating the GRPO objective as an expectation over denosing timesteps that can be estimated via Monte Carlo sampling. Empirically, AGRPO has accuracy gains on math/logic benchmarks over prior policy optimization methods for dLLMs, trading extra compute for per-timestep estimates in a way that aims to reduce bias/variance in log prob estimation.

### Strengths
1. this paper discusses caching partially masked states, gradient accumulation per MC sample, and EOS/timestep handling, which are useful for training diffusion LLMs.
2. this paper shows their proposed method achieves better performance than baselinse UniGRPO and diffu-GRPO across 3 tasks.
3. This paper trades off compute (number of MC estiimates) with approximation accuracy and achieves better results than fewer MC sample baselines.

### Weaknesses
The paper’s writing is not rigorous:

1. Equation 2 is AR-specific and assumes a causal left-to-right generation; mapping its per-token inner sum to dLLM timesteps needs clearer justification. As written, the swap (|o_i|\to m) can mislead readers about where compute actually occurs in dLLMs.

2. Exact sequence likelihood in dLLMs is intractable because it would require marginalizing over all denoising orders/masking patterns, but AGRPO doesn’t do that: it optimizes using conditionals on partially-masked states and estimates an ELBO-based surrogate via MC. MC is unbiased for the ELBO, but the ELBO itself is a biased lower bound to the true log-likelihood. The paper should avoid calling such MC "exact". Additionally, optimizing the particular sampling order might not be optimal, because if a response is good, we ideally want to increase the probability of generating this response with all possible orderings to generalize better. UniGRPO and diffu-GRPO resample new masking patterns (randomly or full mask) can somehow encourage this. 

3. The notation $$(\pi_\theta(o_t \mid q, o_{<t}))$$ overloads AR conventions; in dLLMs (o_{<t}) should be defined as the partially masked state at timestep t. The paper uses this shorthand broadly without a crisp definition, which hurts clarity.

4. this paper also lacks a related work section

Limited novelty and lack of experiments:

1. AGRPO replaces one-step heuristics with an unbiased per-timestep estimator—a principled step—but the paper lacks ablations on k (MC samples), estimator variance/bias, and how k drives compute–quality trade-offs. Without these, the incremental benefit over prior heuristics is harder to quantify. The experiments are lacking rigorous ablations and for the main results, the tasks they choose are a subset of prior works.

2. Increasing the number of MC samples introduces additional compute, the paper doesn't analyze the efficiency and performance gain trade off.

### Questions
see above.

### Soundness
2

### Presentation
2

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
The paper identifies a key issue in post-training diffusion large language models (dLLMs), that the existing RL methods like GRPO, which work well for autoregressive (AR) models, are incompatible with dLLMs because dLLMs cannot compute probabilities in a single forward pass. It introduces Amortized Group Relative Policy Optimization (AGRPO), a reinforcement learning (RL) algorithm designed for fixing this, by reformulating the GRPO objective by treating the sum over timesteps as an expectation, which is then estimated using Monte Carlo sampling over timesteps. They claim this provides an unbiased policy gradient. The paper demonstrates performance gains on mathematical reasoning tasks (GSM8K, MATH, Countdown) over prior dLLM RL methods.

### Strengths
The core idea of trying to reformulate GRPO  to enable Monte Carlo estimation is new. If correct,  it can directly addresses the basic incompatibility between standard policy gradient methods and the dLLM architecture. 
The paper is generally well-written and clearly explains the theoretical shortcomings of existing approaches. There are additional tricks describe for efficient computation which can be valuable in other contexts as well.

### Weaknesses
I might have misunderstood the key idea — but it seems to be based on a wrong interpretation of what the actual incompatibility between PGRL methods and dLLMs. They use the AR-incorporated PPO/GRPO loss where the propensity ratio has been simplified because of autoregressive per token likelihood decomposition. This itself becomes invalid for dLLMs. While some form of regressive factorization is still true, the model likelihood will still not match the true likelihood. This makes AGRPO another approximation (which is fine, but the paper does not claim that). Furthermore even if this term is unbiased (which I do not believe to be the case), the KL term still requires true log-likelihoods, which is not what the AGRPO  trick allows to compute. Overall the description sidesteps the fundamental issue that dLLMs do not have a tractable sequence-level probability.

Additionally, the results compare only a specific length case. While compute resource constraints make sense for training, inference is done with the same model for different lengths. So that does not seem to a big issue. Additionally they use LoRA based fine-tuning instead of full model tuning, which makes the comparison a bit apples-to-oranges. The gains can be from LoRA, and then the baselines need to be tuned with LoRA. Moreover the paper does not compare against more recent baselines such as [1,2,3], which also attempt to deal with the likelihood mismatch issue.

[1] wd1: Weighted Policy Optimization for Reasoning in Diffusion Language Models

[2] PADRE: Pseudo-Likelihood based Alignment of Diffusion Language Models

[3] DiffuCoder: Understanding and Improving Masked Diffusion Models for Code Generation

### Questions
Experiments:
I do not see the contribution of low-discrepancy sampling being ablated. How much does it actually help over naive i.i.d. sampling?
Similarly, usage of sampling t, produces variance, there should be an ablation of performance against t/m .


The paper claims the new loss is more efficient scalable, but I do not see the compute-cost vs gain tradeoff. Using the m-sample likelihood is more compute expensive then the 1-step approximation used in diffu-GRPO/d1's likelihood approximation. The cost/tractability is not just from samples but from overall FLOPS, which is not compared.

Relatedly, the argument that rollout generation cost "dwarfs" the loss computation is highly dependent and needs some empirical measures

Theory:
The paper uses the diffusion time t, as also a generated sequence length measure (by conditioning on $$o^i_{<t}$$ . How do you then generate that with independently sampled time $t \in [1,m]$. This is not an issue in AR models, as they generate tokens sequentially for t=1 to t=m; but that is not the case here. Is this multiple one-step approximations? Or something different? 
What sequence/step likelihood is used for the KL computation in the objective.
Furthermore, the paper cites a blog-post for an unbiased KL estimate, but that post itself has multiple unbiased estimates. I think I know the version used, but this is not clear from the paper itself. 
Since the code is not with the paper, I also cannot compare the expression used compared to the one in the paper to see if I misunderstood the expression; and the paper does not give enough details to know exactly what the computations are ( for both the importance ratio and KL likelihood)

### Soundness
1

### Presentation
2

### Contribution
1
