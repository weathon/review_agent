# Learn the Ropes, Then Trust the Wins: Self-imitation with Progressive Exploration for Agentic Reinforcement Learning

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 2, 6, 6, 2

## Abstract
Reinforcement learning (RL) is the dominant paradigm for sharpening strategic tool use capabilities of LLMs on long-horizon, sparsely-rewarded agent tasks, yet it faces a fundamental challenge of exploration-exploitation trade-off. Existing studies stimulate exploration through the lens of policy entropy, but such mechanical entropy maximization is prone to RL instability due to the multi-turn distribution shifting. In this paper, we target the progressive exploration-exploitation balance under the guidance of the agent's own experiences without succumbing to either entropy collapsing or runaway divergence. We propose SPEAR, a self-imitation learning (SIL) recipe for training agentic LLMs. It extends the vanilla SIL, where a replay buffer stores good experience for off-policy update, by gradually steering the policy entropy across stages. Specifically, the proposed curriculum scheduling harmonizes intrinsic reward shaping and self-imitation to 1) expedite exploration via frequent tool interactions at the beginning, and 2) strengthen exploitation of successful tactics upon convergence towards familiarity with the environment. We also combine bag-of-tricks of industrial RL optimizations
for a strong baseline Dr.BoT to demonstrate our effectiveness. In ALFWorld and WebShop, SPEAR increases the success rates of GRPO/GiGPO/Dr.BoT by up to 16.1\%/5.1\%/8.6\% and 20.7\%/11.8\%/13.9\%, respectively. In AIME24 and AIME25, SPEAR boosts Dr.BoT by up to 3.8\% and 6.1\%, respectively. Such gains incur only 10\%–25\% extra theoretical complexity and negligible runtime overhead in practice, demonstrating the plug-and-play scalability of SPEAR.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper aims to address the exploration-exploitation tradeoff for agentic
LLMs that use tools to complete certain tasks. The paper proposes a method
called SPEAR that enhances existing RL methods for finetuning LLMs. The main
function of SPEAR is to control the entropy of the policy to be within a narrow
range. It does this with methods such as intrinsic reward shaping and
self-imitation learning. Results on various benchmarks demonstrate that SPEAR
improves performance (success rates) when integrated with various contemporary
RL algorithms.

### Strengths
1. The paper seems to achieve high performance on the benchmarks presented.
1. My understanding is that the SPEAR method is achieving what it claims to,
   i.e., it is managing the entropy. This understanding is based on results like
   Figure 3.
1. The paper provides good low-level details of how exactly the method works.

### Weaknesses
As this paper is outside my field, I am unable to comment on the novelty of the
method or the validity of the low-level details, so my comments mainly focus on
the presentation of the paper:

1. While this paper excels at providing low-level details, I find that too much
   time is spent on these low-level details. As such, I overall had a hard time
   reading this paper and understanding the insight that it provides. For
   example, the abstract and introduction both dive quite deep into the
   low-level details of SPEAR, which causes them to be far too long. Meanwhile,
   in Section 4.1 and 4.2, it is unclear what the purpose is for each of these
   components, as the low-level details are just presented immediately.
1. Overall, the paper seems to be written for a highly technical audience who is
   already familiar with RL finetuning for LLMs. I think revising the paper to
   be written for a slightly broader audience would make it much easier to
   understand. For example, the abstract and introduction should seek to explain
   the high-level insights of the paper, rather than directly describing the
   modifications introduced by SPEAR.
1. Writing the paper for a high-level audience may help in shortening it. One
   aspect that seriously hurts this paper's readability is that important
   sections like Related Work and Discussion are left to the appendix. If all
   the sections truly need to be as long as they are, this paper may be better
   fit for a journal. That being said, I think this paper can be shortened a
   considerable amount by optimizing the usage of space. For example, there are
   many bulleted lists in the paper that could be switched to paragraphs.

I think a good starting point for addressing my feedback is: What is the key
insight of this paper? That is, if you had to summarize the contributions of
this paper in one sentence, what would you say?

### Questions
The following are my additional comments and questions on the manuscript.

1. The axis labels in all the figures tend to be quite small and hard to read.
   In particular, the plots in Figure 1 are illegible.
1. The main performance metric seems to be the success rates on the tasks; these
   success rates are treated as validating the changes introduced by SPEAR. Are
   there any measurements showing that SPEAR controls the entropy across a wide
   variety of tasks? Figure 3 seems to only show SPEAR's capabilities with
   respect to entropy in one task.

### Soundness
3

### Presentation
2

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
The paper tackles the exploration-exploitation trade-off in reinforcement learning (RL) for large language model (LLM) agents on long-horizon, sparsely-rewarded tasks. It proposes SPEAR, a curriculum-based self-imitation learning (SIL) method with intrinsic reward shaping and regularization, to progress policy entropy smoothly. SPEAR achieves performance improvements over baselines like GRPO, GiGPO, and Dr.BoT on ALFWorld, WebShop, and AIME24/25 benchmarks, with manageable computational costs.

### Strengths
**Innovative Method**: Proposes SPEAR, extending vanilla SIL with curriculum scheduling, advantage recalibration, and covariance-based clipping to address entropy collapse/divergence in LLM agents (Sec. 4; Eq. (5); Sec. 1). Integrates intrinsic reward shaping with a curriculum, distinguishing skill-level and action-level exploration tailored to LLM dynamics (Sec. 4.2; Fig. 4). Introduces Dr.BoT, a strong baseline combining RL tricks, enabling robust comparison of SPEAR’s performance (Sec. 4.3; Table 1, Table 2).

**Curriculum Design**: The integration of curriculum-based SIL with intrinsic reward shaping enables progressive transition from skill-level to action-level exploration, addressing entropy collapse (Sec. 4 and Fig. 2). Advantage recalibration using a FIFO buffer and 50th percentile baseline mitigates off-policy estimation drift and aligns with policy improvement (Eq. 2 and Sec. 4.1). Reward composition with curriculum scheduling (μ) for tool-call reward prevents competition with outcome reward (Eq. 6 and Fig. 4).

**Extensive Experimental Validation**: Evaluates on diverse benchmarks: ALFWorld (multi-turn), WebShop (e-commerce), AIME (math with code), covering different action spaces (Sec. 5.1; Table 1, Table 2). Compares against multiple baselines (GRPO, GiGPO, etc.), showing consistent gains (Table 1, Table 2). Uses ablation-like figures (Fig. 3, Fig. 4) to demonstrate self-imitation and intrinsic reward’s impact on entropy and performance.

### Weaknesses
**Mathematical Formulation Ambiguity**: Equation (2) for advantage recalibration uses the 50th percentile without theoretical justification; it is empirically described as "conservative but robust" but lacks derivation or comparison to other percentiles (Sec. 4.1). The SIL objective in Eq. (3) and (4) introduces multiple conditions, but the interaction between these advantages and their combined effect on policy convergence is not analytically explained.

**Limited Generalization**: Experiments focus on specific tasks (ALFWorld, WebShop, AIME) and models (Qwen series), with no main-text results on broader domains like vision-language agents, though mentioned in Appendix A.17 (Sec. 5.3). The method assumes that successful trajectories are clearly identifiable via positive advantages, but in stochastic environments with sparse rewards, this may lead to noisy experience selection, as acknowledged in Sec. 6.

**Vague Definition of Good Experiences**: The paper notes the vague definition of good experiences in highly complex, stochastic environments with unreliable tools, limiting generalizability (Sec. 6). No concrete examples or analyses in evaluated benchmarks illustrate this vagueness’s impact or occurrence frequency. The proposed stepwise supervision solution is mentioned but not elaborated or evaluated (Sec. 6).

### Questions
1.	How does SPEAR’s curriculum schedule adapt to different LLM sizes and task complexities? For instance, does a larger LLM need a different entropy progression rate, and does SPEAR adjust this dynamically (Sec. 4.1; Fig. 3)?
2.	In AIME24/25 experiments, how does SPEAR balance tool-call frequency and final accuracy under context length constraints (Table 2), and are there qualitative reasoning path examples illustrating this trade-off?
3.	How does the 10–25% extra theoretical complexity of SPEAR translate to actual training time and GPU memory usage across different benchmarks and LLM sizes (Sec. 1; Sec. 5.2)?

### Soundness
2

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
This paper proposes SPEAR, a curriculum-based self-imitation learning recipe for training agentic LLMs that progressively balances exploration and exploitation by (i) shaping intrinsic rewards to encourage early, skill-level tool use, (ii) strengthening self-imitation later for action-level refinement, and (iii) stabilizing learning via advantage recalibration and covariance-based clipping to control entropy and policy drift. SPEAR stores only promising trajectories in a replay buffer (positive-advantage episodes), then: (1) schedules intrinsic "tool-call" rewards that decay over training so outcome rewards dominate; (2) recalibrates off-policy advantages using a rolling, median baseline to avoid costly recomputation; and (3) clips tokens whose log-probabilities covary strongly with advantage, preventing over-confidence and entropy collapse. Together, these yield a smooth transition from broad tool-skill exploration to focused action-selection exploitation.

### Strengths
- The paper targets a concrete failure mode in agentic RL for LLMs—entropy collapse from naive self-imitation—and designs a remedy that explicitly balances exploration and exploitation. This focus is well-motivated and illustrated (Figure 3) with clear problem framing.

- The method SPEAR is a cohesive recipe rather than a single trick: curriculum-style intrinsic tool-call rewards for early skill exploration, action-level self-imitation later, advantage recalibration for off-policy stability, and entropy/covariance-based clipping to prevent over-confidence and policy collapse. The components and their roles are specified precisely.

- The paper identifies and mitigates a practical reward-design pitfall: competition between outcome reward and tool-call reward causing oscillations. The scheduled intrinsic reward is a simple, effective fix.

- Empirical results are broad and positive across distinct action spaces (ALFWorld, WebShop, and DAPO-MATH-17K), demonstrating that the recipe transfers beyond a single domain. Metrics and evaluation protocols are clearly stated. 

- The authors provide a strong, modern baseline (Dr.BoT) assembled from recent "bag-of-tricks," improving the fairness and relevance of comparisons; SPEAR then yields additional gains on top.

### Weaknesses
1/ The method hinges on a fuzzy notion of "good experiences" in stochastic, tool-heavy settings; sparse outcome rewards can make positive advantages look like noise rather than genuine policy improvement. This undermines the replay filter’s reliability. Would step-wise process rewards be mandatory in noisy tool ecosystems?

2/ Entropy control depends on hand-designed schedules and covariance-based clipping, whose optimal settings likely vary by task. This rigidity risks under-/over-regularization. Can the paper provide principled guidelines or adaptive rules beyond fixed schedules?

3/ The advantage recalibration and replay-filter condition $\hat{A} > 0, \tilde{A} > 0$ may bias learning toward "easier/high-return" slices of the data, potentially narrowing exploration and reducing robustness to hard cases. How sensitive are results to relaxing this double-positive gate?

4/ Parts of the qualitative analysis rely on an external LLM to label code intent, which could introduce evaluation bias. Is there an annotator-agreement check or a non-LLM rubric to validate these labels?

### Questions
Please refer to the questions in Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes SPEAR (Self-Imitation with Progressive Exploration for Agentic Reinforcement Learning), which addresses the exploration–exploitation trade-off for large language model (LLM) agents in long-horizon reinforcement learning. The method builds upon a curriculum-based self-imitation learning (SIL) framework to progressively control exploration during training.

### Strengths
1. The direction of agentic reinforcement learning for LLMs is highly relevant and important.
2. The paper demonstrates solid engineering effort, with comprehensive implementations and comparisons.
3. The writing is clear and easy to follow.

### Weaknesses
1. From my perspective, the paper seems to be more of an engineering combination than a foundational innovation. The paper reads as a well-crafted integration of existing techniques, e.g., self-imitation, curriculum learning, intrinsic reward shaping, and covariance-based regularization, etc., rather than a conceptually new algorithmic contribution. While the integration is effective, it feels closer to a sophisticated *recipe* or *best-practice collection* for training LLM-based agents than a novel RL framework with theoretical depth.
2. The paper defines exploration primarily as maintaining policy entropy to avoid collapse or divergence, and ties it closely to tool-use behavior. The notion of “skill-level exploration” is implemented through intrinsic rewards that encourage more tool calls. This design works well for tool-using environments but doesn't generalize to settings where exploration challenges arise from reasoning complexity, world modeling, creative planning, etc. As a result, SPEAR seems to address a narrow sub-problem of exploration–exploitation, rather than offering a broadly applicable framework.
3. SPEAR’s success depends on several manually set hyperparameters, like warm-up steps, decay schedules, clipping rates, covariance thresholds, and so on. The paper states that these are empirically chosen, and its own ablation results show significant sensitivity to them. This dependence undermines reproducibility and may limit generalization across tasks or models. Without principled guidance or adaptive mechanisms, achieving the reported results could require substantial tuning effort, reducing its appeal as a “plug-and-play” solution.
4. The paper states that the progressive entropy scheduling is one of main innovations, but all scheduling parameters (stages, thresholds, ratios) are fixed manually, with no criterion for determining when the policy is “ready to exploit.” This makes the method appear heuristic and task-specific, rather than a self-consistent or theoretically grounded algorithm.

### Questions
See the "Weakness".

### Soundness
3

### Presentation
3

### Contribution
2
