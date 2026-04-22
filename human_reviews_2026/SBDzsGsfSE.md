# Step-Aware Policy Optimization for Reasoning in Diffusion Large Language Models

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 4, 4, 2

## Abstract
Diffusion language models (dLLMs) offer a promising non-autoregressive paradigm for text generation, but training them for complex reasoning remains challenging. Current reinforcement learning approaches typically rely on sparse, outcome-based rewards, which can lead to inefficient exploration and ``unstructured refinement'', where the model's iterative denoising steps fail to contribute meaningfully to the solution. While Process Reward Models (PRMs) effectively mitigate similar issues in autoregressive models, they often require expensive human annotation or external verifiers. In this work, we propose Step-Aware Policy Optimization (SAPO), a method to derive automatic process rewards for dLLMs without external supervision. By leveraging the diffusion model's natural operation, we design a reward function that incentivizes distributing problem complexity evenly across the denoising trajectory. This intrinsic process supervision guides the model to learn structured, robust reasoning paths, reducing the risk of derailing from correct traces. Our empirical results demonstrate that SAPO significantly improves performance on challenging reasoning benchmarks and enhances the interpretability of the generation process.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work proposes an extension to diffuGRPO to encourage the model to make progress towards the correct answer throughout the sampling process. This work essentially develops an algorithm to automatically obtain process rewards. The process reward is computed as the difference between the average reward of rollouts from some intermediate step and rollouts from the initial masked sequence. This enables developing a process reward model solely from the verifiable outcome supervision. They validate their proposed algorithm across math benchmarks and synthetic reasoning puzzles and observe that it outperforms diffu-GRPO which only rewards the final rollout state.

### Strengths
This paper focuses on an area with a lot of interest. Strong DLLMs are being developed for the first time and it’s therefore of interest to figure out how to best replicate AR post-training procedures for DLLMs.

This is a natural, well-motivated way to develop process reward models for discrete diffusion models. The one-step probability estimation in diffu-GRPO likely fails to appropriately reward sampling trajectories in the same way as AR models and this might help address that.

There are some details that need to be figured out for the efficient application of their method. Namely fixing the baseline to be the full masked sequence and applying the indicator to the process reward are nice additions to make the algorithm tractable and effective.

The experimental evaluation is sound and their results are generally quite strong compared to diffu-GRPO across settings. The plots of the reward curves provide nice additional validation. I also appreciate the sampling step sweep in Table 3 to display the results across inference budgets.

### Weaknesses
The motivation talking about the “hierarchical decomposition of complex reasoning” feels quite spurious. At best, this may provide a conceptual motivation for the approach. While the theoretical framework may be interesting, it does not derive the algorithm. Claims like "this framework provides a principled foundation for algorithm design" and "motivated by this theory" overstate the connection.

In general, I find the theoretical framework to significantly detract from clarity. Process reward models are widely studied to provide intermediate supervision to LLMs (and are discussed in the related works section). This work proposes an algorithm to get automatic process rewards for discrete diffusion models. This is an interesting contribution in and of itself and the motivation for such work is very clear. For instance, the method is very similar to Math-Shepherd (which is cited) for AR methods.
This work inherits the mean-field assumption from diffu-GRPO which likely limits the effectiveness of the RL supervision.

The authors mention “Although MdLLMs offer faster inference compared to ARMs” (L 287). Although this is widely stated in the literature, this is typically not true. DLLMs with typical decoding settings are significantly slower than similarly sized ARMs due to the lack of kv-caching. For instance, see Fast-dLLM (Wu et al. 2025) for an inference-time comparison.

My understanding is that SAPO requires additional rollouts from the intermediate state. However, There is limited discussion/reporting of the computational cost compared to diffu-GRPO.

### Questions
How does the computational cost compare between SAPO and diffu-GRPO?

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes Step-Aware Policy Optimization (SAPO), an RL algorithm for diffusion LLMs that uses process-based rewards aligned with a theorized latent reasoning hierarchy. The authors formalize complex problem solving as a hierarchical selection process and argue that standard methods suffer from unstructured refinement—iterations that do not add meaningful progress. SAPO aims to guide denoising steps toward structured, coherent reasoning and reports gains on challenging reasoning benchmarks with improved interpretability.

### Strengths
1. Introduces a process-based reward tailored to dLLMs’ iterative denoising, rather than only outcome-based signals.

2. Provides a theoretical framework (hierarchical selection with identifiability insights) that motivates the algorithmic design.

### Weaknesses
1. The unstructured refinement claim ("identifies unstructured refinement as a key failure mode in standard diffusion language models, where the iterative denoising process fails to align with this latent reasoning hierarchy") is primarily theoretical and shown via case study, but lacks strong experimental validation. Prior MDM results (e.g., easy-first decoding) suggest hierarchical behaviors already exist to some extent [1][2]; the paper does not numerically define or analyze the latent hierarchy to test this claim.

2. The proposed process reward differs from AR LLM process supervision which splits the reasoning process sequentially in cot chain. Here, an intermediate sample $x_{t1}$​ is a full sequence. Thus, rewarding accuracy at $t_1$ may mainly encourage early stopping (that is to say when model can generate $x_{t1}$​ correctly before using all steps, this means the model is quite confident in this generation and can do early exit), and the model is rewarded based on the confidence and probably not from learning a multi-step latent hierarchy. This weakens the connection between the theory and the method design.

[1] https://arxiv.org/pdf/2308.12219v2
[2] https://arxiv.org/abs/2510.08632

### Questions
The citation format is incorrect somewhere e.g., lines 118–119.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes step-aware policy optimization (SAPO) for training diffusion language models with RL. Their motivation is from the idea that complex reasoning should follow hierarchical decomposition rather than relying solely on outcome-based rewards. They introduce a Monte Carlo-based process reward that measures the contribution of intermediate denoising steps by comparing the final fully denoised completion's accuracy from different timesteps, providing richer learning signals than standard methods. The approach demonstrates improvements across six benchmarks with ablations on sampling parameters, generalization capability, and intermediate accuracy.

### Strengths
1. this paper presents motivation of why we should care about intermediate reasoning steps than out-come based learning in RLVR. and it proposes a MC estimate based method to quantify the advantage brought by sampling an intermediate step from t_2 to t_1. this brings more learning signal than merely just learn from the final fully denosied sequence.
2. this paper show better results in general as compared to previous methods across 6 benchmarks. 
3. this paper has nice ablation studies on the number of N for the MC estimates, and generalization ability testing and intermediate answer comparison.
4. the paper's writing is clear and related work is comprehensive.

### Weaknesses
regarding the motivation of this paper:
1. The "we, we..." example is presented as evidence that the model lacks structured reasoning, but this is more like a coherence or quality issue. a model could generate coherent text while still having poor underlying reasoning structure. The paper assumes all reasoning problems require explicit hierarchical decomposition into discrete steps, but this doesn't account for simpler problems where reasoning might happen implicitly in the latent denoising process. Forcing explicit step-by-step reasoning on easy problems could be wasteful when the model might efficiently solve them through smooth constraint satisfaction in latent space. Their theory doesn't distinguish between problem complexity levels.

    notation wise: their model suggests R is generated from the final S_L but R could more naturally being a composition of all intermediate reasoning steps?
2. in figure 10, i wouldn't say generating repeated [eos] tokens are meaningless. LLaDA is pretrained with eos tokens as padding tokens, this is an expected behavior. It is intriguing to see model trained with SAPO doesn't generate repeated eos tokens after training. This means the model will always generate tokens to fill entire fixed generation space which seems unnatural. 

Regarding experiments:

3. the performance improvement seems a little bit marginal as compared to diffu-grpo on gsm8k and math500 tasks in table 1. 
4. Their motivation stems from structured decomposition of reasoning steps, yet their R_process reward is applied uniformly to all tokens rather than to specific reasoning steps. Additionally, their efficiency trick of setting t_2 = T (all masked tokens) means the reward granularity only measures progress from complete masking to t_1, which seems too coarse to capture the hierarchical step-by-step decomposition they theoretically motivate. There lacks an experiment on where t_2 not equal to T to align better with the motivation part in this paper in section 3.
5. Their method incurs increased computation by sampling N additional rollouts from intermediate state t_1 (roughly 50% more generations than diffu-GRPO). However, they provide no analysis of sample efficiency—comparing performance against total compute or number of generations.

### Questions
please see above.

minor problems:
1. figure 3 should have swapped the position of t2 and t1 in the right part of the figure?
2. Why only apply R_process to positive examples? If a response has both a wrong answer (A_i < 0) and poor intermediate reasoning (R_process < 0), will the model be discouraged more strongly?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper is concerned with post-training of diffusion language models (dLLMs). Specifically, they focus on complex reasoning, and argue that the use of sparse, outcome-based rewards 
can lead to models producing correct answers via incorrect reasoning: what they label as "unstructured refinement". The authors propose a framework which considers the full reasoning chain as a latent hierarchical process. They then try to align the dLLM with latent hierarchy using RL through SAPO ( Step-Aware Policy Optimization ). SAPO involves using a process reward that can capture 'implicit progress' between two stages in the denoising path, and trying to align the denoising with the corresponding hierarchical reasoning structure. Empirical results on mathematical and logical reasoning benchmarks (GSM8K, MATH, etc.) are presented to show performance improvements.

### Strengths
Applying process-reward models to dLLMs is novel to the best of my knowledge. The paper is also generally well-structured and easy to follow. The problem of inducing hierarchical reasoning to solve the "unstructured refinement" problem is interesting. dLLM chains are in general harder to interpret and understand, and improving the reliability of the generation process is of practical value.

### Weaknesses
he theoretical construct for reasoning has weak links to the proposed method. It seems to have been forced in, instead of naturally translating to it. On the other hand the high level proposal is intuitive, and can be presented directly. The issues I see are: a) the framework assumes an identifiable process, but this has no analogue for the actual process,  b) sparsity is considered important for the identification but does not appear at all in the proposal, c) how does the random intervals of the denoising process connect to 'a set of logical constraints' from the theory.  Finally  and importantly I do not see how the model actually learns the postulated hierarchy.

Somewhat relatedly, we have the upweighting heuristics. The authors provide the process reward, only if the base advantage is positive ( or effectively the answer is correct). The authors justify this as avoiding boosting process chains which are 'correct/better' but go nowhere. This is prioritizing final answer accuracy over reasoning integrity, and is not punishing models that get the right answer for the wrong reason undermining the core thesis of the paper. Furthermore, the whole point of process models (in general) is to promote the model to do correct reasoning even on wrong chains, as these correct subchains will then be promoted and the model generalize better. This makes me consider that the process reward models are not learning something effective in the first place. 

I am also confused by equation 4. Do you sample multiple random intervals and use the empirical average, or is it one sample only. The latter seems surprising, and if its an empirical average, Eq 4 seems wrong. I also think  a more structured sampling scheme would be important to check.

### Questions
I think the empirical evaluation is weak. The paper fails to demonstrate that SAPO provides a unique advantage over simply applying a similar process reward from a standard ARM. A good baseline might be to use a auto-regressive LLM process model, by considering the mask as sometype of space or pad tokens. I think SAPO should be compared against a strong baseline that uses an analogous process reward. It seems difficult to determine if the differences come from the dLLM based rewards or the process reward strategy.


The up-weighting strategy in Eq. 5 undermines the goal of eliminating "correct-by-chance" solutions. Can you run an ablation without the indicator function there and normal process reward.

The model seems to be worse than even d1/diffu-GRPO. Why would process reward models make the model worse? The difference on GSM are also small. Finally other models, which do not need  any process model etc. and have better results ( See [1])
Given the goal is the connection to process model, I think one should add an evaluation of the process reward model.

If I am understanding the reward over a single random interval correctly (and I may not be), is there a validation of this single random interval is an "effective and efficient approximation." 



[1] wd1 Weighted Policy Optimization for Reasoning in Diffusion Language Models

### Soundness
2

### Presentation
2

### Contribution
2
