# FlowRL: Matching Reward Distributions for LLM Reasoning

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 8

## Abstract
We propose FlowRL: matching the full reward distribution via flow balancing instead of solely maximizing rewards in large language model (LLM) reinforcement learning (RL). Recent advanced reasoning models adopt reward-maximizing methods (e.g., PPO and GRPO), which tend to over-optimize dominant reward signals while neglecting less frequent but valid reasoning paths, thus reducing diversity. In contrast, we transform scalar rewards into a normalized target distribution using a learnable partition function, and then minimize the reverse KL divergence between the policy and the target distribution. We implement this idea as a flow-balanced optimization method that promotes diverse exploration and generalizable reasoning trajectories. We conduct experiments on both math and code reasoning tasks: FlowRL achieves a significant average improvement of $10.0\%$ over GRPO and $5.1\%$ over PPO on math benchmarks, and performs consistently better on code reasoning tasks. These results highlight reward distribution-matching as a key step toward efficient exploration and diverse reasoning in LLM reinforcement learning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces FlowRL, a novel reinforcement learning algorithm for LLM reasoning that tackles the "mode collapse" problem inherent in traditional reward-maximizing methods like PPO and GRPO. Instead of merely seeking the highest reward, FlowRL's core contribution is to reframe the objective as reward distribution matching, aiming to make the policy's output distribution over reasoning paths proportional to their rewards. Drawing inspiration from GFlowNets, the authors formulate this as a tractable "trajectory balance" objective and introduce crucial technical adaptations—length normalization and a modified form of importance sampling—to make it effective for long-chain-of-thought training. Through strong empirical results on challenging math and code benchmarks, the paper convincingly demonstrates that this new paradigm significantly outperforms strong baselines by promoting greater solution diversity and improving generalization.

### Strengths
1. By creatively connecting GFlowNets with LLM reasoning, this paper provides an elegant and novel solution to the well-known problem of mode collapse, moving from the standard reward-maximization objective in RL to a more principled reward distribution matching paradigm.
2. The authors introduce crucial technical solutions—length normalization for stability and modified importance sampling for data efficiency—demonstrating a strong command of the challenges in large-scale RL and bridging the gap between theory and practice.
3. The paper is exceptionally clear, logically motivating and deriving the FlowRL algorithm from first principles. The claims are backed by strong and comprehensive empirical results on challenging math and code benchmarks, where FlowRL consistently and significantly outperforms established baselines like PPO and GRPO.
4. This paper is significant as it introduces a new, effective direction for LLM post-training. By promoting diverse and generalizable reasoning, FlowRL not only pushes state-of-the-art performance but also opens promising avenues for developing more robust and exploratory generative agents, which could have a broad impact on complex problem-solving applications.

### Weaknesses
1. The paper only reports final performance metrics, making it impossible to assess whether FlowRL is more or less sample-efficient than GRPO and PPO. Including learning curves (accuracy vs. training steps/samples) is essential to substantiate the method's practical training efficiency, especially given that GFlowNet-based objectives can be sample-inefficient.
2. While the proposed solutions to GFlowNet's instability (e.g., length normalization, detached importance sampling) are empirically effective, their necessity and impact are not deeply ablated, leaving them feeling more like engineering choices than rigorously justified algorithmic components.

### Questions
See "Weaknesses".

Additional question:
1. The choice to parameterize the partition function Zφ as a 3-layer MLP is presented without justification. This is a critical architectural decision, yet its impact on performance is unevaluated. What was the rationale for this specific depth, and how sensitive is the model's performance to variations in the Zφ architecture?

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
1. FlowRL shifts RL from maximizing scalar rewards to matching the full reward distribution, eliminating mode collapse and enriching reasoning diversity.
2. A learnable partition-function network approximates the intractable normalization constant. It also employs length normalization and importance sampling to enhance training stability.
3. Experiments demonstrate that the proposed method outperforms both PPO and GRPO on math and code reasoning tasks.

### Strengths
1. The insight is highly meaningful—it enhances the representational capacity of rewards and constitutes a valuable innovative attempt.  
2. Supported by solid theoretical justifications and in-depth analysis.  
3. The paper is well-written, fluent, and easy to follow.

### Weaknesses
1. Please refer to the "Questions" section for details.

### Questions
1. The ablation study omits an ablation on “reward shaping via length normalization.”  
2. Why a 3-layer MLP is chosen to parameterize Z_ϕ could be further investigated and compared with alternative designs.  
3. A statistical analysis of reward distributions under different training scenarios—detailing their shapes, similarities, and differences—would be valuable.

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
3

### Summary
FlowRL introduces a new RL objective for LLM post training that is motivated by connecting reverse KL with GFlowNet. In this new formulation, the advantage is regularized by the important sampling term and a learned partition function (via a small MLP from the mean pooled LLM hidden state) so that the model's output diversity is preserved. The author demonstrates effectiveness of the proposed method on Qwen-2.5 and DeepSeek-R1-Distilled Qwen models on math and reasoning benchmarks.

### Strengths
The paper is generally well written and easy to follow.

The biggest contribution from the paper is, in my opinion, the formulation of using GFlowNets to implement reserve KL for distribution matching training objective. The idea is elegant and the ablations seem to cover important tricks that are necessary for implementation on models at scale. It is great to see works that are both theoretically grounded and implementationally sound.

### Weaknesses
Given my lack of familiarity of the GFlowNet lineage of work, I can only comment on the algorithm from a purely pragmatic perspective.

The main drawback is that the FlowRL's benefits mainly come from important sampling to control model entropy, which could be achieved more easily with other methods without additional learnable parameters (in the partition function). In particular, the objective has 2 main components that are augumenting the advantage function: 1) Importance sampling, 2) partition function. From the ablations in the work, it appears that the IS part of the augmentation is very important, however, it is not clear whether the partition function is needed at all in practice. _If_ partition function is not necessary from a final model performance perspective, then FlowRL is similar to GRPO with a penalty on pi_theta/pi_ref, which in spirit looks quite similar to a KL penalty. Therefore, from a pragmatic standpoint, the question is whether the complexity of FlowRL necessary for the performance gain that we observe.

### Questions
It would be really interesting to ablate Z_phi to see how much partition function is helping. In principle this should help reducing training variance, but then the question is what if we simply tuning other hparams like number of rollouts/sample.

It would also be interesting to see how the model evolves during training from things like AIME vs. generation length, or AIME vs. training flops. It is not immediately clear that the gains reported in FlowRL would scale with model size and training flops.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper proposes FlowRL, a reinforcement learning algorithm that matches the full reward distribution via flow balancing instead of solely maximizing rewards for LLM post-training. Unlike reward-maximizing methods (REINFORCE++, PPO, GRPO) that tend to over-optimize dominant modes while neglecting less frequent but valid reasoning paths, FlowRL transforms scalar rewards into a normalized target distribution using a learnable partition function Zφ(x), then minimizes the reverse KL divergence between the policy and this distribution.The authors prove gradient equivalence between this KL objective and the trajectory balance formulation from GFlowNets, and introduce length normalization and importance sampling to address gradient explosion and sampling mismatch in long CoT reasoning.

### Strengths
- Novel application of GFlowNets to LLM reasoning through distribution-matching framework
- Rigorous mathematical formulation with gradient equivalence proof
- Exceptionally clear presentation with intuitive figures
- Achieves substantial improvements

### Weaknesses
- No human evaluation: All diversity assessments rely on GPT-4o-mini judgments
- Only tested on reasoning tasks (math, code); unclear if FlowRL works for other domains
- All experiments use thinking models or reasoning-focused tasks; what about non-reasoning tasks?

### Questions
As mentioned in Weaknesses.

### Soundness
4

### Presentation
4

### Contribution
4
