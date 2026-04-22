# RLFR: Extending Reinforcement Learning for LLMs with Flow Environment

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 6, 4

## Abstract
Reinforcement Learning with Verifiable Rewards (RLVR) has recently emerged as a promising framework for improving reasoning abilities in Large Language Models (LLMs). However, policy optimized with binary verification prone to overlook potential valuable exploration in reasoning trajectory. In view of heavy annotation cost of golden Process Reward Models (PRMs), recent works attempt using auxiliary signals for reward shaping of process tokens, involving entropy and likelihood collected from logit space. In this work, we offer a novel perspective on shaping RLVR with flow rewards derived from latent space, and propose \textbf{RLFR}, where the flow fields of model latents are constructed from either off-policy high-quality data and on-policy rejection sampling data, and the velocity deviations of policy latents within it are quantified to serve as a reward signal. \textbf{RLFR} first demonstrates that a well-established flow field can be a sound environment for reward signal collection, highlighting the expressive latent space is much underexplored. Moreover, \textbf{RLFR} is able to compress any off-policy expert data as reference for constituting reward signals and we show that the practical execute tokens are prefered by flow reward rather than connection tokens. Experiments on both language and multimodal reasoning benchmarks demonstrate the reliability of flow rewards, and suggest a promising paradigm for reward shaping with auxiliary signals.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes RLFR, a reinforcement-learning framework for large language models that introduces flow rewards derived from latent representations. The method extends RL with verifiable rewards by constructing flow fields in the latent space from both off-policy expert data and on-policy rejection-sampling data. The deviation of policy latents from the reference flow field is quantified as a reward signal. Experiments on language and multimodal reasoning benchmarks show modest gains over RLVR and entropy-based shaping methods.

### Strengths
- Addresses a limitation of RLVR (coarse, binary reward signal) and explores a new direction of reward shaping.

- Attempts to leverage latent-space information rather than logit-space entropy, which is conceptually interesting.

- Includes relatively broad experiments across multiple model families and tasks.

- Implementation details and ablation studies are clearly reported.

### Weaknesses
- The flow-matching variable is borrowed from continuous diffusion literature but has no meaningful definition in the discrete LLM latent process. The “velocity deviation” term is therefore ill-posed; the claimed connection between velocity deviation and likelihood lacks grounding.


- LLM hidden states are conditionally dependent and layer-specific, not time-continuous trajectories. Treating them as samples evolving under a flow field appears ad hoc and unsupported by either theory or evidence.


- The flow reward effectively measures reconstruction error in a surrogate latent regression task, but it is unclear why this metric correlates with reasoning quality or exploration value.


- The reported performance gains are small (often 1-2%), without statistical tests or variance analysis to show significance. It is unclear whether improvements stem from flow rewards or simply additional fine-tuning.

- Key equations are overloaded with symbols whose roles shift across sections. The method description mixes notations from RL and flow matching, making the algorithm hard to reproduce or verify.

### Questions
What is the precise definition of the time variable in the context of LLM latent representations?

How does the velocity field interact with discrete transformer layers or token positions?

Is there empirical evidence that smaller “velocity deviations” correspond to higher-quality reasoning steps?

How sensitive are results to the choice of layer sampling, timestep, and reward scaling parameters?

Could the improvements be replicated if the same auxiliary loss were applied directly on latent reconstruction without RL?

### Soundness
1

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
This paper introduced RLFR, reinforcement learning with flow rewards. It is a new reward shaping to extend RLVR. Instead of relying on binary signals, RLFR constructs flow fields in model's latent space using both off-policy data and on-policy rejection-sampled data. The velocity deviation serves as the flow rewards which provides a continuous signal to guide reasoning trajectories. The experiments are conducted on Math and logic tasks, and multimodal tasks, which shows consistent improvements over RLVR.

### Strengths
Standard RLVR requires verifiable rewards which limit the use cases. This paper introduced a new perspective, leveraging flow environment to train models via RL. This offers a novel method to the community to further improve model's reasoning ability.

The paper is clearly written with logical flow. The authors also provides clear ablation studies, which comprehensively analyzed the contribution of key components.

### Weaknesses
1. The empirical improvement is limited. From Table 1 and Table 2, RLFR achieved limited improvement on many tasks. For example, for logic Avg. the improvement is only 0.1 which is within noise level. For Math Avg. the improvement is 2.4. This raises my concern about how effective this method is. 

2. Can this method deal with non-verifiable rewards? For example, some tasks are difficult to verify, like long report generation. Is this method  more applicable to such tasks than RLVR?
 
3. Some typos and formatting issues:
- Line 193-194, analysis -> analyze
- Line 195-196, should use \citep instead of \citet. Similarly, there are many formatting issues with misused \citet. 
- There are some overlaps in Figure 1 for numbers. For example, the rightmost 52.1 overlaps with 52.0.

### Questions
In Table 1, on AIME24 Pass@1, why there is a drop around 3.4 for RLFR compared with RLVR?

### Soundness
2

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
2

### Summary
This paper proposes RLFR, a novel reward shaping framework to extend Reinforcement Learning with Verifiable Rewards (RLVR) for LLMs. The authors argue that standard RLVR with binary verification overlooks valuable exploration, while existing auxiliary signals from the logit space (like entropy) are limited. RLFR introduces a "flow reward" derived from the model's latent space. This method constructs a flow field using off-policy high-quality data and on-policy rejection sampling. The reward signal is then calculated as the "velocity deviation" of the policy's current latent states within this flow field, penalizing drift from the reference distribution. The central claim is that the latent space is an underexplored but highly expressive substrate for reward collection and that this flow-based reward provides a stable signal to improve LLM reasoning.

### Strengths
1. The primary strength is the novel use of flow matching and velocity deviation from the latent space to define a reward signal. This is a creative departure from common approaches that focus on outcome verification or logit-space signals (e.g., entropy, likelihood)

2. he method demonstrates consistent and robust performance gains. RLFR outperforms the base model, the standard RLVR baseline, and the Entropy Advantage (logit-space) baseline across a suite of six language and multimodal reasoning benchmarks (Figure 1). This holds true in detailed tables for both language (Table 1) and multimodal (Table 2)  tasks.

3. The authors provide a comprehensive set of ablations that validate the key components of the RLFR framework. The studies confirm the necessity of the offline start, online rejection sampling, and fluctuation filtering (Table 3). Furthermore, the ablations justify specific design choices, such as using outcome correctness as the rejection metric (Table 4) and the superiority of the proposed timestep debiasing method (Figure 4).

4. The paper provides good qualitative analysis. Figure 3 offers a clear visual motivation, suggesting that latent space distributions (especially for tail tokens) provide a more distinguishable signal for correctness than logit distributions.

### Weaknesses
1. The proposed method is highly complex, which may be its greatest drawback. Unlike a simple entropy bonus, RLFR requires training a separate flow network. This network must first be pre-trained on offline data and then continuously updated online using a rejection sampling buffer. This dual-optimization loop introduces substantial computational overhead and implementation challenges not present in the baselines.

2. The "Entropy Adv." baseline, used as a key comparison for logit-space shaping, performs poorly. In Figure 1, it underperforms the standard RLVR baseline in three of the six benchmarks. This is also seen in Table 1 for the Qwen2.5-Math-7B model. This suggests the entropy baseline may be weakly implemented or inherently flawed, making RLFR's outperformance of it less significant than its gains over the stronger RLVR baseline.

3. The paper is not perfectly clear about how latents from multiple layers are processed. It states latents are extracted from percentiles {0.25, 0.5, 0.75} and fed into a 4- or 6-layer flow network. However, the exact mechanism for combining these multi-layer features (e.g., concatenation, averaging) and integrating the "layer position embedding"  is not detailed, leaving a gap in reproducibility.

4. The repeated use of "Flow Environment"  is imprecise. In RL, the environment is the process being optimized (the LLM's decoding). The flow network is a component of the reward function, not a new environment. This terminological choice, while evocative, slightly obscures the method's actual mechanics.

### Questions
none

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces RLFR (reinforcement learning with flow rewards), a novel framework that enhances LLMs' reasoning capabilities by moving beyond the limitations of binary-reward reinforcement learning. The core innovation is the use of flow matching in the model's latent space to generate dense, process-oriented reward signals. Specifically, RLFR constructs a flow field from high-quality data and quantifies the velocity deviation of the policy's latent states within this field. More minor deviations are rewarded, while larger deviations are penalized. The authors theoretically connect velocity deviation to the evidence lower bound of log-likelihood, establishing a foundation for this reward signal.

### Strengths
The idea of using flow matching to generate a continuous reward signal from latent-space dynamics is interesting. While flow matching is established in generative modeling, its application as an environment for quantifying the quality of intermediate reasoning steps in RL is a fresh and compelling combination of ideas. It effectively addresses a known weakness of RLVR (sparse, binary rewards) with a technically sophisticated solution. The experiments on language and multimodal reasoning benchmarks demonstrate that RLFR consistently outperforms baseline RLVR and entropy-based shaping methods, showing more stable training and improved performance.

### Weaknesses
The primary baselines are basic RLVR and an entropy-based shaping method. The paper would be significantly strengthened by comparing against more competitive baselines. The authors note PRMs' high cost as a bottleneck, but it would be valuable to see a direct comparison, even on a smaller scale, to show whether RLFR can be a lower-cost alternative that achieves good performance. Compared to the baselines, the proposed method shows limited improvements. Further detailed experimental analysis is needed to show the reliability of the proposed method.

### Questions
Please refer to "Weaknesses".

### Soundness
3

### Presentation
3

### Contribution
3
