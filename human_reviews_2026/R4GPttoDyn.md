# Learning to Reason Efficiently with Discounted Reinforcement Learning

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
Large reasoning models (LRMs) often consume excessive tokens, inflating computational cost and latency. More broadly, in goal reaching sequential decision problems we often want to reach the goal quickly, and LRM reasoning can be viewed through this lens. We challenge the assumption that longer responses improve accuracy. By penalizing reasoning tokens using a discounted reinforcement learning setup (interpretable as a small token cost) and analyzing Blackwell optimality in restricted policy classes, we encourage concise yet accurate reasoning, analogous to preferring shorter successful trajectories in a stochastic shortest path problem. Experiments confirm our theoretical results that this approach shortens chains of thought while preserving accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper investigates reasoning efficiency in verifier-guided reinforcement learning by casting reasoning as a discounted MDP. The key theoretical result (Theorem 3.10) states that Blackwell-optimal policies not only maximise expected reward but also minimise the expected time until the reward is obtained. This motivates a practical training scheme where only reasoning tokens are discounted, implemented with GRPO and KL regularisation. Empirically, the approach reduces average reasoning length while maintaining Pass@1 across several mathematical reasoning benchmarks.

### Strengths
* The problem---achieving concise yet correct reasoning---is well motivated and practically relevant.
* Theorem 3.10 is a clear and intuitively appealing formalisation of how discounting encourages shorter successful trajectories.
* The proposed method is simple to implement and shows consistent empirical effects across model scales and datasets.
* The paper is easy to follow, and the theoretical exposition appears rigorous and technically careful.

### Weaknesses
* *Limited theoretical novelty.* Section 3.1 essentially restates known results on Blackwell optimality and the connection between discounted and average-reward MDPs (e.g. Grand-Clément & Petrik 2023). Although Theorem 3.10 might not have been stated explicitly before, follows straightforwardly from the classical theory.
* *Restrictive and unnecessarily complex assumptions.* The deterministic-MDP setting appears more restrictive than necessary: the result should extend to general finite MDPs with binary rewards (i.e. reachability problems). The definition of a finite “deployment” policy class via greedified softmax policies also seems artificial and more complicated than required for the theoretical guarantees.
* *Unused and uninformative bound.* It is quite clear from standard convergence arguments that *some* bound on the Blackwell discount factor must exist. The explicit bound in Theorem 3.9 is never used or evaluated, yet it makes the exposition noticeably more cumbersome. Since the experiments determine $\gamma$ by naive search, it would have been more interesting to assess how tight or informative this theoretical bound actually is.
* *Missing empirical comparison.* The experiments do not compare to other approaches designed to shorten model outputs. Without such baselines, it is difficult to gauge the practical benefit of the proposed approach.

### Questions
1. Could the main result (Theorem 3.10) and associated guarantees extend to general finite MDPs with binary rewards, and can the need for the contrived finite “deployment” policy class be eliminated?
2. Did the authors attempt to evaluate the tightness or practical relevance of the theoretical bound on the Blackwell discount factor?
3. How does the proposed method compare empirically to other approaches aiming to produce shorter responses?

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
4

### Summary
The paper deals with efficient reasoning language models, specifically solving problems with the minimum number of required reasoning tokens. The key idea is that using discounting, i.e. the standard technique of discounting the reward in time by a multiplicative factor, can result in an efficient policy. The authors give theoretical results using the idea of a Blackwell optimal policy. They show that in deterministic verifier problems with finite horizon and binary rewards, Blackwell optimal policies are shortest path policies. Experimentally, they show that discounting can maintain pass@1 while reducing length in their tested settings.

### Strengths
- The paper makes a nice connection between discounting and reducing trajectory length in the context of reasoning language models.
- The authors provide a formal underpinning to the connection.
- Their recipe for training a model seems reasonable, and leads to a reduction in reasoning token usage in their tested settings.

### Weaknesses
- The experimental results do not seem to address the standard long chain-of-thought setting. For instance, in Table 1 the lengths are in the hundreds, and in table 2 the lengths are typically 1000 or less. These inference budgets are lower than those typically used to evaluate long chain-of-thought models. Based on the results it's unclear whether the proposed methods work for these settings. 
- There are many related methods for efficient reasoning models; the area is a very active research area. However, no comparison with other methods is done in the experiments. 
- The theoretical results are presented in a way that did not give me much insight. For instance, the main result Theorem 3.10 is stated but there is not a discussion of the key intuition for why it is true. In addition to the formalized results, it would be helpful to extract the intuition for why discounting leads to an efficient policy, e.g. through a less dense argument.
- Based on the assumptions in 3.10 it is unclear whether the results would hold in settings like multi-step agent or tool calling settings that may have noise in environment transitions or rewards, or settings with real-valued rewards.
- The theoretical results and the experiments use greedy decoding. However using stochastic sampling is a standard approach when evaluating reasoning language models and their efficiency techniques.
- The training data is not clear.
- The paper briefly mentions that "similar conclusions can be drawn for methods that assign a reward proportional to the response length". It implies that prior length penalty methods have the same properties, meaning that the primary contribution is the theoretical contributions in this paper rather than practical methods. In that case, it would be beneficial to expand the results and experimental analysis to the length penalty methods in addition to the discounting methods.

Overall there is an interesting idea here and I like the simplicity of the method, but the experimental validation is below the bar for ICLR for reasons described above. I also think the scope could be improved by elaborating on the length penalty methods, and the accessibility could be improved by providing insight into the key ideas for why the theoretical results are true.

### Questions
Please respond to the points contained in the review above, including these questions:

1. Can you provide experimental comparisons with other efficient reasoning methods to help judge the relative benefits of your approach?
2. How does the method perform in settings where the original model produces much longer reasoning traces?
3. Can you add discussion of the key intuition for why Theorem 3.10 holds and why discounting leads to an efficient policy?
4. Do the theoretical results extend to stochastic sampling, multi-step agents, or real-valued rewards?
5. Can you clarify the training data used?

### Soundness
3

### Presentation
3

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
The  problem addressed in the paper  is that RL post training improves
accuracy but lengthens responses, raising computational cost and latency. It
says that the response can be reduced to a threshold without reducing the
accuracy.

The authors have modeled a verifier-based reasoning as a finite horizon MDP
with determinist transitions and binary terminal reward. And then they train
with discount factor (γ < 1). Here only the environment reward is discounted,
leaving   the   formatting/shaping   reward   undiscounted.   Also,   only   the
reasoning tokens are discounted, regularized with a KL penalty to a moving
reference policy. This ensured that the token budgets comparable.

They show that in a fixed policy class Blackwell optimality policies maximizes
undiscounted   success.   And   among   accuracy   maximizers,   minimize   the
expected trajectory length. They reduce average response length on GSM8K,
MATH and other benchmarks while matching pass@1 base line Using GRPO
with discounted objective with (Group Relative Policy Optimization) with
discounted objective. 

When tested on Quen 2.5 7B-Instruct and Lama 3 8B-Instruct models, the
response length was reduced in discounted method. When compared to
undiscounted method the GSM8K dataset showed 22% reduction in response
length while MATH dataset showed 13% reduction. The Phi4 and Qwen 2.5
14B-Instruct models with AMC 2023, AIME 2025, MINERVA and OLYMPIAD
datasets also showed reduction in response length.

### Strengths
1)	The four design components used here are Discounting only the environment reward, Regularizing KL to a changing reference, Discounting only reasoning tokens and Comparable token budgets across methods. These components are simple to adopt and are also defended in the theory.
2)	In the finite restricted policy classes, for γ close to 1 the Blackwell optimal policies are accuracy maximizing and have shortest mean response length. This is proved in the theorems 3.4, 3.7 and 3.10 which strengthens the claim.
3)	The GSM8K dataset showing a 22% reduction and MATH datasets showing a 13% reduction in response lengths is significant when considering computational cost and latency.

### Weaknesses
1)	Modeling as a finite horizon with deterministic transitions and binary terminal reward worked theoretically. But many real-world reasoning workflows violate these assumptions.
2)	In Blackwell optimality analysis, the γ is selected as for from 1 as possible. However, this is done by a simple bisection search.
3)	Discounting is only applied to reasoning tokens. The authors say that discounting entire response slightly hurt the accuracy. This suggest there may be some issues that prevents errors frond being detected.

### Questions
1)	In addition to binary verifier rewards, have you tried Dense reward models or Hybrid rewards?
2)	Instead of a bisection search for the selecting γ, will a predictive hybrid approach provide a better result?
3)	Besides reduction in response length, did the latency and computational cost get reduced?
4)	For GSM8K and MATH datasets you limited the completion length to 786 tokens & 2048 tokens. What happens when the runs hit the limit? Have you tried with larger limits?

### Soundness
3

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
LLM reasoning models generate verbose intermediate reasoning to solve complex problems, incurring high computational costs. While reinforcement learning improves accuracy, it often lengthens responses. This paper challenges the assumption that this accuracy-length trade-off is unavoidable (as discussed in prior works).

The authors apply discounting (γ < 1) to the reward function, motivated by Blackwell optimality from decision theory. They prove that within any fixed policy class, such a policy exists, establishing that no inherent trade-off between accuracy and reasoning length occurs up to an instance-dependent regime. The key practical insight seems to be to discount only the environment (correctness) reward while leaving intrinsic (formatting) rewards undiscounted. Combined with KL regularization and discounting only reasoning tokens, this approach substantially reduces response length without sacrificing accuracy.

To back up the theoretical work, the authors conduct experiments on math benchmarks (GSM8K, MATH, AMC, AIME, MINERVA, OLYMPIAD) and demonstrate 22% length reduction for Qwen2.5 7B and 13% for Llama 3 8B while maintaining pass@1 accuracy, even when controlling for total tokens processed during training.

### Strengths
* The paper makes a principled design choice to discount only extrinsic (correctness) rewards while leaving intrinsic (formatting) rewards undiscounted, which is well-justified both theoretically and practically.

* The theoretical exposition clearly communicates Blackwell optimality concepts and the main results, making the mathematical framework accessible (I am not a hard RL person and found it pleasant to read). The paper provides thorough theoretical analysis that comprehensively covers finite policy classes, softmax training, and greedy deployment scenarios.

* The experimental methodology is good (e.g., model selection), with results averaged over multiple seeds to properly control for variance inherent in RL-style post-training. The efficiency claims are validated even when matching token budgets across methods, ruling out the confound that discounted methods simply see less data. Ultimatively, the results consistently support the theoretical predictions across multiple benchmarks and model architectures.

### Weaknesses
The paper 
* evaluation is limited to math benchmarks (GSM8K, MATH, AMC, AIME, MINERVA, OLYMPIAD); it remains unclear whether these findings generalize to coding tasks or other reasoning domains.

* theory section relies on the deterministic transitions assumption, but the paper does not specify whether training uses stochastic sampling (T>0) or greedy decoding (T=0), creating a potential theory-practice gap.

* assumes the policy class is finite, which is needed for the existence of the Blackwell discount factor, but it is unclear how realistic this assumption is or whether violations would significantly impact the results.

* has no systematic analysis showing where the difficulty cutoff lies, i.e., at what problem hardness the efficiency gains from discounting diminish or disappear.

* applies a single discount factor γ per dataset but does not justify whether this is appropriate or whether different problems within a benchmark should use problem-specific discount factors.

* does not discuss why discounting intrinsic rewards would be problematic, missing an opportunity to strengthen the design justification.

### Questions
* How dependent is the "no trade-off" result on the quality of the base model and the difficulty of the problem? Could this approach only be effective for sufficiently large models on sufficiently simple problems and fail to scale as a general training paradigm?

* Why has no one already tested this approach empirically given that discounting seems elegant and theoretically motivated? I am mainly asking as you mention ByteDance's existing implementation (Sheng et al. 2025) that already seems to do something similar, and if so, what exactly is novel here?

* Beyond maintaining pass@1 accuracy, what is the impact on reasoning quality (coherence, correctness of intermediate steps)?

* When problem difficulty increases and discounted RL is applied, does training become less effective or does the mean chain-of-thought length simply increase again, negating efficiency gains?

* In practical deployment across multiple tasks, should one use a universal discount factor or does γ need to be tuned per dataset or per problem?

### Soundness
3

### Presentation
3

### Contribution
3
