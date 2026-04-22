# Chasing Moving Targets with Online Self-Play Reinforcement Learning for Safer Language Models

- Avg Score: 5.33
- Decision: Reject
- Scores: 4, 6, 6

## Abstract
Conventional large language model (LLM) safety alignment relies on a reactive, disjoint procedure: attackers exploit a static model, followed by defensive fine-tuning to patch exposed vulnerabilities. This sequential approach creates a mismatch: attackers overfit to obsolete exploits, while defenders perpetually lag behind emerging threats. To address this, we propose Self-RedTeam, an online self-play reinforcement learning (RL) algorithm, where a single model alternates between co-evolving attacker and defender roles---generating adversarial prompts and safeguarding against them---while a reward model adjudicates outcomes. Each role uses hidden Chain-of-Thought, which enables agents to reason about how to formulate and defend against attacks. Grounded in the game-theoretic framework of two-player zero-sum games, we establish a theoretical safety guarantee that motivates our method: if self-play converges to a Nash Equilibrium, the defender is assured to generate safe responses against any adversarial input. Empirically, Self-RedTeam demonstrates strong generalizability across four model sizes from both the Llama and Qwen families. We not only uncovering more diverse attacks (e.g., +17.80% SBERT), but improve the safety of models trained with industry-standard safety fine-tuning procedures like RL from Human Feedback (RLHF) by as much as 95% across 12 safety benchmarks.Our results motivate a shift from reactive patching to proactive co-evolution, enabling scalable and autonomous self-improvement of LMs via MARL.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a novel online multi-agent reinforcement learning framework for improving the safety of Large Language Models (LLMs). The core idea is to move away from the conventional static, reactive patching of vulnerabilities towards a proactive, co-evolutionary process. The method frames LLM safety as a two-player, zero-sum game where a single, shared-parameter LLM alternates between an attacker role (generating adversarial prompts) and a defender role (safeguarding against them).

### Strengths
- It formulates red-teaming as a two-player zero-sum game with a formal safety guarantee at Nash Equilibrium.

- It shows strong empirical results, showing consistent gains across 12 benchmark and multiple model families and sizes.

- Extensive ablations show the effectiveness  of each proposed components.

### Weaknesses
- Reward model  and policy (defender and attacker) share the same parameter $\theta$, which looks confusing. Given that the WildGuard is used for reward model, it must be a notational mistake.  I think it would be better to use different parameters and explicitly state the reward model is frozen during entire training. 

- The KL term in Eq. is undefined. I guess the authors might use token-wise reverse KL, but it would better to explicitly define the term for clarity.

- There is no direct head-to-head against red-teaming baselines. Rainbow Teaming [1] looks a relevant baseline if we use the same seed prompts for Rainbow Teaming.

- It is unclear why using the same backbone for both defender and attacker is helpful other than computational efficiency. The same model competes with conflicting objective. I wonder how the proposed method enables stable training and performance improvement. 

- It heavily relies on initial seed prompts. Even though the attacker generated diverse attack prompts, they are still variants of the initial seed prompts, rather than new type of attacks.



## References

[1] Samvelyan, Mikayel, et al. "Rainbow teaming: Open-ended generation of diverse adversarial prompts." Advances in Neural Information Processing Systems 37 (2024): 69747-69786.

### Questions
- If we use a model that is already capable of generating think part, how the proposed method would work?

- What happens if seed prompts are not available for training? 

- Are there any examples of reward hacking? The reward model is not perfect, which might lead to false harmful attack prompts.

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
2

### Summary
This paper proposes SELF-REDTEAM, an online self-play reinforcement learning framework for LLM safety alignment, where a single model alternates between attacker and defender roles with a hidden Chain-of-Thought (CoT). The idea of framing safety training as a zero-sum game between co-evolving agents is novel and theoretically grounded, providing a clear motivation for proactive rather than reactive safety alignment.

### Strengths
Strengths:

- Conceptually appealing formulation of LLM safety as a self-play MARL problem with a Nash equilibrium–based safety guarantee.

- Solid empirical results across multiple model families (Llama, Qwen), demonstrating significant robustness gains (up to 95% ASR reduction) with minimal performance degradation.

- The Hidden CoT mechanism is an elegant addition, improving attack diversity and mitigating over-refusal.

### Weaknesses
Weaknesses:

- The theoretical guarantee relies heavily on the quality of the reward model; practical convergence to Nash equilibrium is not verified.

- Some evaluation benchmarks (e.g., WildGuard/WildJailBreak) overlap with training data, potentially inflating results.

- Experimental section could be more transparent about compute cost and stability during training.

### Questions
1. How do you measure or verify convergence toward the proposed Nash Equilibrium in practice?

2. How sensitive are the results to the choice or bias of the reward model used for safety evaluation?

3. Does the observed 95% ASR reduction generalize to unseen or multi-turn adversarial prompts?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Self-RedTeam, an online self-play RL algorithm that trains a single model to co-evolve attacker and defender roles in a two-player zero-sum game, generating adversarial prompts and safeguarding against them. They show that if the models reach Nash Equilibrium, the model is theoretically guaranteed to be safe — although this is likely impossible to achieve in practice. The approach demonstrates empirical improvements across multiple safety benchmarks (including WildGuard and HarmBench), and two model families (Llama3, Qwen2.5). The authors also offer a nice study on the distribution of the attacks generated by their method, showcasing how their method generates more diverse attacks over only fine-tuning an attacker LLM against a static defender. I have a few comments with regards to the theoretical justification as well as the evaluations (which could potentially be improved), but overall I am leaning towards a weak accept as the approach is nice, the work is very polished and well executed, and would be of interest to the research community.

### Strengths
1. The method is practical and efficient with a thorough analysis on the overhead of their approach (~45% longer than baseline with online generation). The framework is general and can be applied to any safety training pipeline, with a reasonable improvement to the refusal rate on the models tested.
2. The paper is very well written and polished; the authors conduct many experimental results, including comparisons to other safeguarding baselines (LAT, CircuitBreakers), as well as ablations on the various components of their approach (self-play, CoT, SFT). The appendix is detailed and provides all necessary implementation details one would need to reproduce their work.

### Weaknesses
1. If I understood correctly, the evaluations were done against *static* adversarial prompts (with the exception of X-teaming); stronger non-static attacks should be considered for the evaluations (i.e. applying some of the algorithmic methods to the final trained model itself, rather than using the preexisting attacks on other models). If the paper is indeed missing these evals, I would strongly recommend them for the discussion period.
2. Results indicate that the improvements are decent but not spectacular; some evals have good improvements but others are very modest; I think it would be critical to see how well this approach fares to what was discussed in W1.
3. The observation that over time, the attacker sometimes refuses to generate harmful attacks does suggest that it might not be ideal to use the same attacker/defender model
4. I understand that the Nash Equilibrium arguments are just to show that you are optimizing for the correct target. However, it feels a bit ad-hoc because it was presented as a justification for the approach, then acknowledged that convergence is unlikely in practice, and then no longer discussed. I think it would be nice to have more discussion of either how close the training can converge to the NE in practice, and/or what happens when it doesn’t.

### Questions
1. Re: emergent attacker refusal, it does feel like the attack/defense are in tension because the model is rewarded for breaking itself whilst also trying to be come more robust. Intuitively I would expect the results to improve if you had different models for the attacker and defender; have you experimented with this and seen anything to verify/disprove this hypothesis? How does this tie in to your theoretical motivation, as you require need strong attacker for robust defense?
2. The reliance on WildGuard 7B to evaluate components used in the reward could have biases and be prone to reward hacking; it would be nice to see some discussion on how sensitive the method is to the judge model’s quality/biases, what happens if the judge has exploitable weaknesses, and if attacks are able to game the specific reward model.
3. Why does the CoT help more for Llama than Qwen models?
4. Do attacks transfer to other LLMs?

### Soundness
3

### Presentation
4

### Contribution
3
