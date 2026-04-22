# On-Policy RL Meets Off-Policy Experts: Harmonizing Supervised Fine-Tuning and Reinforcement Learning via Dynamic Weighting

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) are two prominent post-training paradigms for refining the capabilities and aligning the behavior of Large Language Models (LLMs). Existing approaches that integrate SFT and RL often face the risk of disrupting established response patterns and inducing overfitting to expert data. To address this, we present a novel investigation into the unified view of SFT and RL through an off-policy versus on-policy lens. We propose CHORD, a framework for Controllable Harmonization of On- and Off-Policy Reinforcement Learning via Dynamic Weighting, which reframes SFT not as a separate stage but as a dynamically weighted auxiliary objective within the on-policy RL process. Based on an analysis of off-policy expert data's influence at both holistic and granular levels, we incorporate a dual-control mechanism in CHORD. Specifically, the framework first employs a global coefficient to holistically guide the transition from off-policy imitation to on-policy exploration, and then applies a token-wise weighting function that enables granular learning from the expert, which promotes on-policy exploration and mitigates disruption from off-policy data. We conduct extensive experiments across various practical tasks, providing empirical evidence that CHORD achieves a stable and efficient learning process. By effectively harmonizing off-policy expert data with on-policy exploration, CHORD demonstrates significant improvements over baselines. We will release the source code to inspire further research.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates the interplay between Supervised Fine-Tuning (SFT)  and RL. It proposes a simple yet effective method called Chord, where GRPO and SFT loss are dynamically combined, where the loss computation of SFT is further adjusted based on each token’s policy probability.

### Strengths
The major focus — addressing the shift-readapt-overfit phenomenon — is well motivated, and the analysis is insightful. Overall the paper has conducted comprehensive experiments with detailed analysis to explain various design choice.

### Weaknesses
The proposed weighting mechanism is not very novel, and the empirical effectiveness is mostly observed on Math.

### Questions
1. I don’t see much difference btw GRPO (pure RL) v.s. proposed method in BFCL, even though 5k instances are used, any intuition why the improvement is very large on AIME but not in tool-use?

2. Continue from Q1, since Math benefit more from longer reasoning chain, chord-$\mu$ could mostly learn from expert data (Deepseek-R1)’s format and tendency to generate long reasoing trace, which greately improve the result. From Table 1, the result seems to confirm about this assumption (comparing SFT-best+RL v.s. CHORD). This makes me question if the model is well SFT-trained, is there still any need for combining RL + SFT. 

3. Is there any result on base model Qwen3-8B-base? As newer model has much better performance on math/tool-use domain, I am curious if the method still brings improvement across math/tool-use domain.

4. the method of smoothly combining with SFT should be generalizable to non-verifable task as well (and arguably more useful since SFT data could help constrain exploration space to prefered style/format). Any thoughts on such setting?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper provides a method to unify supervised fine-tuning (SFT) and reinforcement learning (RL) into a composite objective that weights the two individual objectives with a dynamic weighting value. Furthermore, the paper proposes to weight individual tokens to avoid issues that can arise from over-reliance on off-policy data and to encourage on-policy exploration. The effectiveness of these two weighting factors are studied empirically on Math and Tool-use datasets. The paper uses datasets provided by an expert model to learn policy. Empirical results suggest that the proposed method exceeds several reasonable baselines that includes both SFT and RL variants as well as  recently proposed works (LUFFY, SASR).

### Strengths
- The paper provides a clear description of the two objectives used in their method called CHORD. Furthermore, the paper clearly describes the experimental setup and results.
- The analysis conducted to motivate the method is clear. 
- Empirical results suggest that the proposed method (CHORD-\phi) improves over several reasonable baselines on Math and Tool-use cases. The ablations included in the main paper suggest that the transition from offline imitation to online-RL learning is effective as it allows for exploration by the policy.

### Weaknesses
- The paper proposes one simple way to combine the two objectives. It's not clear why a convex combination of SFT and RL objectives is the right approach. Would it be possible to have generic weights for SFT and RL and let the model and data decide their optimal values? 

- The objective for Chord-\phi uses a weight that looks like the variance of a Bernoulli random variable. Just like above, is this the optimal value for this weight? Are there any insights on what might happen if the base model is not as strong as the one considered in the experiments?

- Related to above, the analysis is conducted on Qwen2.5 for Math and Llama-3.2 for Tool use. What are the reasons for using these models the way they were used? Would the findings translate to other/newer models released in the (near) future?

### Questions
(repeated from weaknesses)

- Would it be possible to have generic weights for SFT and RL and let the model and data decide their optimal values? 
- Are there any insights on what might happen if the base model is not as strong as the one considered in the experiments for the Chord-\phi's token weight?
- Are the findings applicable to newer models, especially architectures like MoEs that have been released or will be released in the future.

I would like to discuss the first two questions with the authors during rebuttal. The third question is asked to help the paper make generic but does not require a response.

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
In this paper, the authors propose CHORD, a unified framework that integrates Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) through a dynamically weighted objective. CHORD introduces two main components: a global coefficient μ that decays over time to balance imitation and exploration, and a token-wise weighting function ϕ(p)=p(1−p) to stabilize off-policy updates by emphasizing uncertain tokens. The method aims to mitigate instability when combining on-policy RL signals with off-policy expert data. Experiments on mathematical reasoning (OpenR1-Math) and tool-use tasks (ToolAce) demonstrate improved stability and modest gains over SFT→RL and recent hybrid methods such as LUFFY and SASR.

### Strengths
- The paper targets an important and timely problem in large language model post-training: how to combine supervised expert data with reinforcement learning in an effective way.


- The proposed framework is simple, well-motivated, and easy to implement in existing RLHF pipelines. The dual-control design (μ and ϕ) provides both stage-level and token-level balance between on- and off-policy learning.


- Experiments are extensive and include ablations (fixed vs. dynamic μ, with vs. without ϕ), entropy/reward analyses, and qualitative case studies that support the claimed stability improvements.

### Weaknesses
The novelty of CHORD is limited. The method reweights two existing loss terms (SFT and RL) using a dynamic coefficient and a heuristic token-wise weighting. Similar annealing strategies and uncertainty-based regularization have been explored in LUFFY, SRFT, and PPO variants with KL or imitation penalties.

- The token-level weighting ϕ(p)=p(1−p) is conceptually similar to entropy-based weighting and lacks theoretical justification for its specific form.

- The improvement margins over baselines are modest, and the experiments do not cover diverse post-training domains such as instruction-following or dialogue, leaving generality uncertain.

- The framework lacks a formal connection to off-policy correction theory or mixed-policy optimization, making it primarily heuristic rather than theoretically grounded.

### Questions
- Can the authors clarify how CHORD differs algorithmically from LUFFY or SRFT, beyond changing the weighting coefficients?


- Is there any theoretical interpretation (e.g., weighted policy gradient under mixed distributions) that supports the design of μ and ϕ?


- Have the authors tried learning μ adaptively (e.g., via reward variance or gradient norms) rather than fixing a decay schedule?


- Does the token-level ϕ weighting introduce significant computational overhead?


- How sensitive is the model’s stability to the exact shape of ϕ(p)? Would other functions (e.g., entropy-based) work similarly?

### Soundness
3

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
3

### Summary
This paper examines why SFT+RL can underperform pure RL when expert data diverges from the policy, characterizing a “shift–readapt–overfit” dynamic. It proposes CHORD, which mixes GRPO with an auxiliary SFT loss: a global weight controls the overall expert influence; a token-wise weight emphasizes mid-probability tokens. Results on math and tool-use show improvements over several baselines.

### Strengths
1. This paper gives a clear empirical documentation of SFT instability under off-policy expert trajectories.

2. The hybrid objective (μ-weighted SFT + RL) is easy-to-implement. 

3. The token-wise weighting is a simple stability heuristic; and the ablations on μ and training dynamics are decent.

### Weaknesses
1. The idea of combining supervised learning and RL during fine-tuning has been explored in prior works (e.g., SRFT, SimpleMix, LUFFY). CHORD uses a similar structure by optimizing a weighted sum of SFT loss and GRPO — with the addition of a global schedule μ and a token-level weight φ(y)=p(1–p).

2. The heuristic p(1−p) is plausible but lacks theoretical backing or strong comparisons to alternative uncertainty weights (entropy/focal/margin).

3. There is a heavy reliance on DeepSeek-R1 experts; and the analysis is limited for weaker/similar experts or different stylistic gaps.

### Questions
1. Can you provide controlled comparisons to SRFT/SimpleMix with matched compute/data and identical rollout settings? What is fundamentally new beyond weighting choices?

2. Can you compare the weighting funciton to alternatives (entropy, focal-style, margin/clipping) and report sensitivity?

3. Test experts that are weaker or stylistically closer to the policy; does μ decay still help, or does CHORD harm?

### Soundness
2

### Presentation
3

### Contribution
2
