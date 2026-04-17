# RLVMR: Reinforcement Learning with Verifiable Meta-Reasoning Rewards for Robust Long-Horizon Agents

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
The development of autonomous agents for complex, long-horizon tasks is a central goal in AI. However, dominant training paradigms face a critical limitation: reinforcement learning (RL) methods that optimize solely for final task success often reinforce flawed or inefficient reasoning paths, a problem we term inefficient exploration. This leads to agents that are brittle and fail to generalize, as they learn to find solutions without learning how to reason coherently. To address this, we introduce RLVMR, a novel frame-work that integrates dense, process-level supervision into end-to-end RL by rewarding verifiable, meta-reasoning behaviors. RLVMR equips an agent to explicitly tag its cognitive steps—such as planning, exploration, and reflection—and provides program-matic, rule-based rewards for actions that contribute to effective problem-solving. These process-centric rewards are combined with the final outcome signal and optimized using a critic-free policy gradient method. On the challenging ALFWorld and ScienceWorld benchmarks, RLVMR achieves new state-of-the-art results, with our 7B model reaching an 83.6% success rate on the most difficult unseen task split. Our analysis confirms these gains stem from improved reasoning quality, including significant reductions in redundant actions and enhanced error recovery, leading to more robust, efficient, and interpretable agents.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces RLVMR, a RL framework designed to improve the robustness and generalization of long-horizon agents. The authors identify a key problem they term inefficient exploration, where standard outcome-only RL reinforces flawed or redundant reasoning paths that happen to lead to success. To address this, RLVMR integrates dense, process-level rewards based on verifiable meta-reasoning behaviors. The agent is trained to explicitly output tags for cognitive steps. These tags are then rewarded programmatically based on rules that encourage efficient and logical problem-solving (e.g., rewarding exploration that discovers new states). These process-centric rewards are combined with the final task success reward and optimized using a group-based policy gradient method. The authors demonstrate SotA performance on the challenging ALFWorld and ScienceWorld benchmarks, showing significant improvements in success rates, generalization to unseen tasks, and reductions in inefficient actions.

### Strengths
*   **Originality & Significance:** The paper introduces a novel approach to supervising the reasoning process of LLM agents, moving beyond sparse outcome-based rewards. The idea of rewarding verifiable meta-reasoning steps is a significant and practical contribution to building more robust agents that not only solve tasks but do so efficiently and logically.
*   **Quality:** The experimental evaluation is of high quality. It is comprehensive, including multiple strong baselines, different model sizes, and rigorous generalization testing. The results are compelling and demonstrate improvement over prior work.
*   **Clarity:** The paper is written with clarity. The motivation, methodology, and results are presented in a clear and intuitive manner, making the work accessible.
*   **Impact:** The work directly addresses a fundamental challenge in agent training and provides a practical, scalable solution that achieves SotA results.

### Weaknesses
*   **Heuristic Reward and Advantage Design:** The programmatic rules for assigning meta-rewards and the method for combining advantage signals are heuristic. For example, the planning reward is still tied to the final task success, making it sparse. The paper would be strengthened by a discussion of alternative designs or a more formal justification for the current choices. A sensitivity analysis on the weighting parameter $\alpha$ is also missing.
*   **Vagueness in Reward Implementation:** The paper could be more specific about the implementation of the programmatic reward rules. For instance, how is a "corrective action after a sequence of failures" (for the reflection reward) precisely defined and detected? Providing more detail would improve reproducibility.
*   **Dependence on "Cold-Start" SFT:** The method relies on an initial SFT phase using a powerful teacher model (GPT-4) to learn the tag syntax. The ablation study shows this step is critical. While the authors frame it as "lightweight," it still constitutes a dependency that could introduce teacher model biases and complicates the training pipeline compared to a pure RL approach.
*    **Source Code:** While the pseudocode for RLVMR is provided in Appendix D, sharing the actual implementation would greatly enhance reproducibility.

### Questions
1.  Could you provide more concrete details on the programmatic implementation of the meta-reasoning rewards? Specifically for the `<reflection>` reward, what is the exact rule used to determine a "sequence of failures" and a subsequent "corrective action"?
2.  The advantage signal is a linear interpolation $A_t = \alpha A^{\text{traj}} + (1-\alpha) A^{\text{MR}}$. How was $\alpha=0.5$ chosen? Have you performed a sensitivity analysis on this hyperparameter, and how do the results change with different values of $\alpha$?
3.  The reward for `<planning>` is granted only if the entire trajectory succeeds. This seems to defeat the purpose of a dense, process-level reward, as the feedback for the initial plan is still sparse and delayed. Have you considered alternative, more immediate rewards for planning, such as evaluating plan quality independently of the final outcome?
4.  The cold-start SFT phase relies on annotations from a superior model (GPT-4). Did you analyze whether the agent simply learns to mimic the meta-reasoning style of GPT-4, and could this potentially limit the agent's own emergent problem-solving strategies during the RL phase?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes RLVMR, a new reinforcement learning framework that incorporates verifiable meta-reasoning rewards to improve long-horizon reasoning in large language model (LLM) agents. Traditional RL methods in this domain (like GRPO) optimize for sparse, final outcome rewards, which often reinforce inefficient or illogical reasoning paths.  RLVMR addresses this by introducing dense, process-level supervision via rule-based rewards for meta-reasoning behaviors such as planning, exploration, reflection, and monitoring.

The method combines a brief supervised “cold-start” phase, where a teacher model (e.g., GPT-4) annotates reasoning tags, with a critic-free policy gradient optimization phase (GRPO-MR). Each reasoning tag receives local verifiable rewards that shape the reasoning process in addition to global task rewards. Experiments on ALFWorld and ScienceWorld benchmarks show that RLVMR achieves state-of-the-art (SOTA) performance

### Strengths
- clear motivation and well presented
- Improved efficiency and generalization

### Weaknesses
- lack of theoretical analysis of the composite reward which can leads to reward hacking
 
- Dependence on teacher annotation, since one powerful teacher LLM is used without guarantee

- Limited ablation on tag definitions, missing ablation on the contribution of individual meta-reasoning tags

### Questions
First of all, I would like to thank the authors for their work. I agree that reinforcement learning (RL) post-training for large language models (LLMs) often encourages inefficient or illogical reasoning paths, and I appreciate that this paper tackles such an important research question.

Here are a few concerns. With the introduction of a custom reward design, how do the authors ensure that the model does not engage in reward hacking? Since the model is trained to maximize cumulative rewards, this could potentially exacerbate the issue the paper aims to address.

Additionally, how do the authors ensure that the teacher model generates correct or reliable tags? And finally, how do different types of tags (e.g., planning vs. reflection) contribute individually to the overall performance?

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
4

### Summary
This work proposes RLVMR to augments end-to-end RL for LLM agents with rule-verifiable process rewards tied to four meta-reasoning tags (planning, exploration, reflection, monitoring). A brief SFT phase teaches the tag format; online training then optimizes a clipped policy objective using a blend of trajectory-level and tag-grouped step advantages. On ALFWorld and ScienceWorld, RLVMR reports SOTA success and large drops in invalid/repetitive actions, attributing gains to improved reasoning quality rather than shortcut paths.

### Strengths
* Clearly targets inefficient exploration and quantifies it with invalid action rate and repetitive action rate, tying process quality to task success.
* Simple, practical method: explicit meta-reasoning tags, verifiable meta-reasoning rewards, and a tag-grouped relative advantage blended with a trajectory-level relative advantage in a clipped objective.
* Consistent gains across base models and benchmarks, especially for the harder split; also shows shorter, more stable solution paths.
* Goes beyond final accuracy with behavior-quality metrics and training stability, reducing degenerate loops.
* Ablation studies indicate each component matters (outcome advantage, meta-reasoning advantage, cold-start SFT, format penalty).

### Weaknesses
* The claim that this work is “the first study offering a definitive explanation and comprehensive analysis of the inefficient exploration issue” overstates its novelty.
The idea that outcome-only RL reinforces flawed reasoning paths has already been recognised in prior works on process reward models and step-level or action-type-conditioned rewards. These earlier studies also analyse how intermediate reasoning quality affects exploration efficiency and generalisation.

* Despite claiming “verifiable” process rewards, the paper does not provide the exact rule logic needed to compute them.

### Questions
* Could you report a brief ablation of $\alpha$ (Eq. 4) and the format-penalty weight?

* Could you precisely define the process-level rewards (for planning, exploration, reflection, monitoring) in both ALFWorld and ScienceWorld?

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
This paper invesitgates the sparse reward problem in RLVR on LLMs for multi-turn long-horizon tasks. They identify an "inefficient exploration" problem which leads the agent to frequently output invalid or redundant actions, as the LLM is optimized to solely maximize the final succuss rate. They propose a reward shaping solution by adding more reward terms to reward meta-cognition behaviors like planning, exploration, reflection, and monitoring. Experimental results show that their method significantly outperform baseline RLVR methods on two benchmark ALFWorld and ScienceWorld, and nearly matches the performace of some strong close-sourced models like GPT-4o.

### Strengths
1. The paper is clearly motivated with a detailed investigation on the ALFWorld benchmark. 
2. The paper is clearly presented, and the method is clearly explained. 
3. The paper shows significantly improvement upon baseline methods on the two benchmarks they use.

### Weaknesses
See my questions below.

### Questions
1. Do you apply a discounting factor to your sparse reward function? As intuitively, I think using discounting factor is an easy approach to reduce repetive and invalid actions such that the task can be solved in a shorter time to gain higher reward. 
2. Have you applied the same cold start phase to the baseline methods? Just to make sure that different methods are compared in a fair way. If yes, do you apply SFT to the baseline methods with just the observation-action pairs? Or also with meta-reasoning tags?
3. How scalable is your method to other long-horizon tasks? E.g., can the meta-reasoning rewards defined in your paper be useful also for other benchmarks, or are they specifically tuned for the two benchmarks used in your paper? And if I've got it right, you need some way to label the meta-reasoning reward for each step right? Do you label with some hand-designed rules or with another LLM? Can these labeling method generalize to other benchmarks we are interested in?
4. Can you give some explanations on why the 7B and 8B variants tuned with your method significantly outperforms GPT-4o on ALFWorld and L0 of ScienceWorld, but gradually underperforms GPT-4o in L1 and L2 of ScienceWorld?

I'm happy to raise my score if the authors can help clarify on these points and address my concerns.

### Soundness
3

### Presentation
3

### Contribution
3
