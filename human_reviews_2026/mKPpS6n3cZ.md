# Pessimistic Reward Modeling in RLHF against Reward Hacking

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4

## Abstract
This work proposes `PET', a novel pessimistic reward fine-tuning method, to learn a pessimistic reward model robust against reward hacking in offline reinforcement learning from human feedback (RLHF). Traditional reward modeling techniques in RLHF train an imperfect reward model, on which a KL regularization plays a pivotal role in mitigating reward hacking when optimizing a policy. Such an intuition-based method still suffers from reward hacking, and the policies with large KL divergence from the dataset distribution are excluded during learning. In contrast, we show that when optimizing a policy on a pessimistic reward model fine-tuned through PET, reward hacking can be prevented without relying on any regularization. We test our methods on the standard text generation datasets. We find that one can learn a high-quality policy on our pessimistic reward without using any regularization. **The learned policy has a high KL divergence from the dataset distribution while having high performance in practice. We also observe that the length bias phenomenon in reward modeling is significantly mitigated by PET.** While the proxy reward trained in traditional approaches shows bias to long responses, the pessimistic reward model finetuned by PET shows little bias to long responses. In summary, our work shows the feasibility of learning a pessimistic reward model through PET against reward hacking. The agent can greedily optimize a policy on the pessimistic reward without suffering from reward hacking. PET can be applied to solve the length bias problem in reward modeling.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes PET,  a method to fine-tune a pessimistic reward model that can prevent reward hacking in offline RLHF without using KL regularization. The main idea is to adversarially train a reward model against a BoN policy. More specifically, the policy (actor) seeks to maximize the pessimistic reward. The reward model (critic) minimizes its score on that policy while fitting the preference dataset.

### Strengths
- PET-PPO seems to outperform or match SOTA RLHF methods across the tested benchmarks.
- The authors also provide theoretical guarantees on the performance of the pessimistic reward fine-tuned by PET. 
- The observation that PET mitigates length bias is a nice practical insight.

### Weaknesses
**Confusion around the role of KL divergence:**
It is unclear whether the authors view KL regularization as too pessimistic or not pessimistic enough.
The text alternates between claiming that KL induces “over-pessimism” and that it is “not pessimistic enough” to prevent reward hacking.
This needs a clear explanation and empirical support (when and why KL over- or under-compensates). It is unclear why making the reward models pessimistic without using the KL divergence is an important problem to solve. 

**Ambiguity in labeling**:
Unclear what the difference is between tables 3 and 4. 

**Algorithm 2 does not seem like a substantial contribution**
It mainly wraps PET into the standard two-step RLHF framework.The contribution lies in Algorithm 1 (PET).

**Trustworthiness and pessimistic claims not substantiated:**
- The statement “This empirically verifies that the pessimistic reward fine-tuned by PET is trustworthy” is not justified. The evidence given (strong downstream performance) is indirect.
- It is unclear whether the PET reward models are actually pessimistic. There is no quantitative measure showing PET’s rewards are systematically lower or more conservative than the proxy’s. They can produce policies with a larger KL divergence. But the authors should compare predicted rewards from the proxy and PET models for the same responses.

**Limited empirical scope and lack of statistical analysis**
- Experiments use only two small datasets (TL;DR, IMDB) and a single model (Pythia-1B) with LLM-based evaluation (Qwen-2.5-32B).
- The experiments appear to use only a single random seed. No confidence intervals or variance estimates are provided, making it difficult to judge whether improvements are statistically meaningful or within noise. Especially given the small magnitude of some gains.

### Questions
- How many random seeds were used for each experiment?
- The authors note that policies learned with the PET reward can have high real performance while having a large KL divergence. Why is this happening, and why is it beneficial to have a large KL divergence? 
- Can the authors explain the difference between Tables 3 and 4?
- The choice of β (pessimistic coefficient) is fixed at 1/β = 0.1. Why was it set this way? What happens if we change that value?
- Do the authors think the other baselines would work better if they used a higher value for n? Since it was noted several times that n was set to 64 because of fewer computational resources?

### Soundness
2

### Presentation
2

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
This paper introduces PET (Pessimistic Reward Fine-Tuning), a method for training pessimistic reward models in Reinforcement Learning from Human Feedback (RLHF) to prevent reward hacking without using KL regularization. Instead of constraining policy updates through KL penalties, PET adversarially fine-tunes the reward model using a Best-of-N sampling process so that it remains faithful to human preference data while assigning lower scores to over-optimized outputs. This produces a pessimistic reward model that enables greedy policy optimization without overfitting or reward exploitation. Experiments on summarization and sentiment-generation tasks show that PET-trained policies achieve higher real-world performance, reduced length bias, and robustness to reward hacking, while maintaining efficiency and outperforming several state-of-the-art RLHF baselines.

### Strengths
The paper introduces PET, a novel pessimistic reward fine-tuning method that mitigates reward hacking without relying on KL regularization. Combining adversarial training with BoN sampling indeed opens up new opportunities.
And the presentation is clearly formalized with both theoretical guarantees and experimental evaluations provided.

### Weaknesses
1. The biggest weakness is limited experiments. It would be more convincing if more recent models were tested, like Qwen, llama, and GPT (they do support fine-tuning via API). The current task is too simple to conclude that PET helps the training without using KL.

2. The adversarial training framework reminds me of self-play and also [PAIRED](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://research.google/blog/paired-a-new-multi-agent-approach-for-adversarial-environment-generation/&ved=2ahUKEwj2u-6I1dCQAxUcEFkFHTWKA3sQFnoECBsQAQ&usg=AOvVaw2uO7Rk9lfGYI4faitR3oyu). It would be better to add some citations and discussions.

### Questions
1. Have you tested the method in tasks like math reasoning, coding, or hallucination benchmarks?
2. Regarding the reward model quality, could you evaluate it against the [reward bench](https://huggingface.co/spaces/allenai/reward-bench)?

### Soundness
3

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
The paper proposes PET, an adversarial reward-model fine-tuning method for offline RLHF that aims to prevent reward hacking without KL regularization. PET proposes a minimax game between a reward model and a BoN policy; leveraging a key property of BoN, the game reduces to a tractable single minimization solved by SGD. The authors then run a two-step pipeline: train the pessimistic reward with PET, and optimize a policy on it greedily. Empirically, the authors show BoN on the PET reward improves win-rates over BoN on a proxy reward; PPO without KL on the PET reward reaches competitive or top performance while having large KL, which contradicts the usual "high-KL leads to reward hacking" case. PET also reduces length bias in reward models. A finite-sample theorem guarantees that, for policies covered by data, the PET solution competes with any BoN policy.

### Strengths
Originality
1. The paper reformulates adversarial reward-policy training by choosing BoN as the adversary, which yields a clean reduction to SGD (as a single step update), which overcomes the instability of standard adversarial RLHF and dispenses with KL regularization during policy optimization. 

Quality
1. Clear derivation, where the claims are backed with proofs. 
2. Implementation details and complexity analysis are provided. 

Clarity
1. The two-step diagram is very clear. 

Significance
1. No-KL PPO on PET outperforms KL-PPO on PET and avoids reward hacking despite high KL which is a notable empirical result.
2. Length bias is substantially reduced by PET (scatter analysis).

### Weaknesses
1. Coverage assumption and BoN-specificity. The theorem requires dataset coverage of candidate BoN policies; in realistic deployments, high-quality but poorly covered behaviors can matter most. PET’s guarantees do not extend there, and the method targets around BoN (though PPO is used post-PET), which may constrain generality.
2. Parameter selection. PET introduces a pessimistic coefficient $\beta$ while in the experiments the authors only follow specific setup in previous paper. Does this selection follow the best practice suggested by Theorem 3.1? Does this selection allows convergence to a good policy for most usecases? Any practical suggestions?
3. Theory. The theorem and its proof seem a direct implication from Liu et al. (2024b) and  Zhan et al., (2023). Can the authors clarify if any theoretical contributions they may have?
4. Confound between KL and training setup. The claim that PPO without KL works well could be confounded by training length/step budget, early stopping, or implicit regularizers (e.g., entropy bonuses, clip ranges).
5. BoN. Does PET generalize well, i.e., does it work for non-BoN inference-time optimizers?

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
2
