# SPARK: Synergistic Policy And Reward Co-Evolving Framework

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 2, 6

## Abstract
Recent Large Language Models (LLMs) and Large Vision-Language Models (LVLMs) increasingly use Reinforcement Learning (RL) for post-pretraining, such as RL with Verifiable Rewards (RLVR) for objective tasks and RL from Human Feedback (RLHF) for subjective tasks.
However, RLHF incurs high costs and potential reward–policy mismatch due to reliance on human preferences, while RLVR still wastes supervision by discarding rollouts and correctness signals after each update. To address these challenges, we introduce the Synergistic Policy And Reward Co-Evolving Framework (SPARK), an efficient, on-policy, and stable method that builds on RLVR. Instead of discarding rollouts and correctness data, SPARK recycles this valuable information to simultaneously train the model itself as a generative reward model. This auxiliary training uses a mix of objectives, such as pointwise reward score, pairwise comparison, and evaluation conditioned on further-reflection responses, to teach the model to evaluate and improve its own responses. Our process eliminates the need for a separate reward model and costly human preference data. SPARK creates a positive co-evolving feedback loop: improved reward accuracy yields better policy gradients, which in turn produce higher-quality rollouts that further refine the reward model. Our unified framework supports test-time scaling via self-reflection without external reward models and their associated costs. We show that SPARK achieves significant performance gains on multiple LLM and LVLM models and multiple reasoning, reward models, and general benchmarks. For example, SPARK-VL-7B achieves an average 9.7\% gain on 7 reasoning benchmarks, 12.1\% on 2 reward benchmarks, and 1.5\% on 8 general benchmarks over the baselines, demonstrating robustness and broad generalization.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a method (SPARK) that jointly trains the LM to solve tasks and judge its own generated response, by "recycling" the rollouts generated during RL with Verifiable Rewards (RLVR). It also bakes self-reflection into the inference time, by utilizing its own judgement to prompt for reflection when mistake is detected.

### Strengths
The experiments over Math related domains are comprehensive and show improvements compared to baseline. The model is also ablated with Policy-only and Reward-only objective.

### Weaknesses
On the methodology, I am not quite sure if I understand the necessity of baking generation and reward modeling together. 

1. the task is already verifiable with rule-based calculation, the benefit of incorporating GRM is not obvious. What about other preference task where GRMs are more useful?
2. No experiments on tasks that are non-verifiable to verify the effectiveness of proposed method. In my opinion, the “co-evolving” framework will likely result in RM overfit or model collapse when the learned judgement is not correct (in RLVR task, the learning signal is guaranteed to be correct for the RM). The generalizability of the setup is not verified.
3. The experiment is not compared with setting that **trains a separate reward model** to help with test-time scaling. The experiments design should stress the difference between (a) LM + a pre-trained and fixed capable RM (b) the proposed co-evolving framework, but lacks such evidence.
4. I’m not sure if comparison between the ablated version and proposed method is fair (i.e., whether judgement and self-reflection are both applied during test-time) but I might be wrong. Please see my question for detail.

### Questions
1. For your evaluation (e.g., Table 1), can you clarify the setting a bit on how SPARK-VL-7B is evaluated? Is test-time scaling with judgement and self-reflection used? My understanding is YES. Please correct me if I am wrong.
2. Then for your ablated version Qwen2.5-VL+GRPO + Policy&Reward, can you explain in more detail how it’s trained? Is it first trained on original data, then trained on crafted preference data for reward modeling, and then evaluated with judgement and self-reflection step as well? Because from the current description (line 318-323), I don’t know if self-reflection is applied during test-time.
3. Do you have experiments that show results using two systems (a LM trained to generate CoT and solve problems, another LM trained on the collected rollout for reward modeling, then a combination of both during test time + self-reflection)?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
SPARK proposes a reinforcement learning framework that jointly evolves the policy and reward model within a single LLM/LVLM. Built on RL with Verifiable Rewards (RLVR), SPARK recycles correctness signals and rollouts that are normally discarded to train the same model as a generative reward model. This co-evolutionary mechanism reduces reliance on human preference data and external reward models, improving efficiency, stability, and test-time self-reflection.

### Strengths
1.	Elegant unification of policy and reward training—reduces cost and improves stability.
2.	Addresses reward-policy mismatch, a key issue in RLHF pipelines.
3.	Demonstrated improvements across reasoning and reward benchmarks (+9.7% / +12.1%).
4.	Conceptually aligns with scalable self-reflective AI trends.

### Weaknesses
1.	Incomplete technical specification:
The paper lacks full detail on how co-training signals are balanced or stabilized (e.g., gradient separation, EMA targets). Without this, it is hard to reproduce or verify convergence.
2.	Potential circularity problem:
Training a model to generate and simultaneously evaluate its own responses risks self-confirmation. The authors claim that the verification step prevents collapse, but empirical or theoretical backing is weak.
3.	Limited ablation studies:
The contribution of each component (e.g., reflection, recycling, policy iteration) to the overall gain is unclear. Ablations would strengthen causal claims.
4.	Generality of results:
All experiments rely on Qwen-family models. It remains uncertain whether SPARK generalizes to other architectures like Llama, Gemini, or GPT-style systems.
5.	Lack of qualitative failure analysis:
The paper focuses on positive results but does not explore where SPARK underperforms—e.g., in ambiguous reward conditions or low-confidence verification.
6.	Presentation clarity:
While conceptually sound, some notation and flow between RLVR and SPARK updates are dense and under-explained. Figures could better illustrate the co-evolution process.

### Questions
1.	How do you prevent reward drift or self-confirmation when both policy and reward share parameters?
2.	What stability techniques (e.g., target networks, KL penalties) are employed to ensure learning convergence?
3.	How often is the verifier updated relative to the policy loop?
4.	Can SPARK operate on preference data when available, or is it strictly designed for verifiable signals?
5.	How would SPARK handle tasks without binary verifiability (e.g., open-ended generation)?

### Soundness
3

### Presentation
3

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
This paper proposes a method called “SPARK”. Its major contribution is to jointly optimize the RL policy and the reward model. It uses the RLVR-derived correctness scores to train the model itself to become a generative reward model. The proposed method is verified on three categories of benchmarks. Experimental results show that SPARK achieves significant performance gains on multiple LLM and LVLM models and multiple reasoning, reward models, and general benchmarks.

### Strengths
1. The motivation to get both the optimized RL policy and reward model is good.
2. The adoption of the reflection mechanism in both training and testing is helpful.
3. The experimental results are supportive.

### Weaknesses
1. The idea of co-training the policy with the reward model will result in divergence. Without a well-trained and fixed reward model, the RL policy will lose the target to optimize. Indeed, a stable target is the priority in optimization. For example, the Deep Q Network, it uses the target network, which is a delayed version of the network to be optimized, as the evaluation network, just to keep the optimization target fixed during a period of time. On the contrary, this paper’s reward model (optimization target) is dynamically changing. Very likely, in the very beginning, the reward model is naive, and the RL policy will not get useful information from it. The RL policy will collapse, and as a result, the reward model itself will not be optimized. Finally, both the reward model and the RL policy will not be improved during the training process.
2. The reflection process can be improved. The idea of reflection is helpful, but simply using the LLM to directly reflect on itself may cause overfitting, which can limit the improvement.
3. As far as I comprehend, this paper attempts to improve the RL with verifiable reward (RLVR) framework by proposing the co-training strategy. It didn’t improve the reward limitation on objective tasks of RLVR, nor does it have a direct relationship with RLHF. The advantage of requiring no human preference data is inherited from the vanilla RLVR. Therefore, the purpose of depicting the limitations of those two methods in the description section (Paragraph 2) is confusing.
4. The manuscript needs polishing. For example, grammar errors like “Our key insight is to recycle the rollouts and correctness data to…”, “reward&reflection”; It is not clear what the “reference model” refers to in Equation 4; It is not clear what “\box{}” is.

### Questions
NA.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces SPARK, an on-policy framework that trains a single model to be both the policy and the judge. Instead of discarding rollouts in RL with verifiable rewards (RLVR), the method recycles the n-best candidates to build on-policy supervision for pointwise judgments, pairwise comparisons, and reflection. The unified model then uses this judging ability at test time for self-reflection–style TTS (no external reward model). On Qwen2.5-VL-7B, the authors report average gains of +9.7% on seven math benchmarks and +12.1% on two reward benchmarks, with smaller but consistent improvements on broader multimodal tasks. The paper argues this reduces RM cost/complexity while improving stability and data efficiency.

### Strengths
1. Unified loop that wastes less signal. Recycling RLVR outcomes into pointwise/pairwise/reflection supervision for the same model is neat and practical; it cuts one model class out of the stack and removes frequent RM calls.
2. On-policy supervision. Using current behavior to create judgment/reflection data reduces distribution shift versus offline RM datasets and explains why TTS helps SPARK but hurts baselines.
3. Consistent wins. The +9.7% (math) and +12.1% (reward) numbers on VL-7B are solid; the smaller general-domain bump is still directionally positive.
4. Reasonable ablations. Clear separation of answer vs. CoT data and a TTS study that highlights why a weak judge can degrade performance, whereas a trained judge helps.

### Weaknesses
1. Efficiency/accounting is light. The paper claims cost wins over RM-based RL, but lacks hard numbers: wall-clock hours, tokens/sec, GPU memory/FLOPs, and verifier runtime (#unit tests per sample, pass rate). Table-style qualitative comparisons are helpful but not enough for practitioners.

2. Verifier dependence. Rewards are binary and rule-based; the paper doesn’t probe robustness to noisy or partial verifiers (very common in code/math). A noise-injection or partial-credit ablation would make the claim more convincing.

3. Self-confirmation risk. Policy and judge live in the same model. The KL to a reference helps, but there’s no quantitative analysis of judge calibration (ECE/Brier) or safeguards against over-confident self-approval during TTS.

4. Repro details. Core knobs for TTS (max reflection rounds, acceptance rule, early stopping), prompt formats, and the exact n-best sampling policy should be surfaced in the main text.

### Questions
1. Compute & throughput. Could you report end-to-end wall-clock, effective tokens/sec, and GPU hours for SPARK vs. (i) GRPO Policy-Only, (ii) GRPO Policy&Reward, and (iii) an RM-based pipeline? Also break out verifier cost per batch (pass rate, retries). This would substantiate the cost argument beyond Table 7.

2. Judge–policy coupling. Did you try decoupled heads or stop-gradient tricks so the “judge” pathway can drift a bit from the “policy” during data generation? Even light dropout/temperature on the judge might reduce confirmation bias.

3. TTS protocol. Please specify maximum reflection rounds and acceptance criteria (first judged-correct vs. best-of-k). In Table 5, can you attribute the baseline degradation to specific judge errors over rounds?

### Soundness
3

### Presentation
3

### Contribution
2
