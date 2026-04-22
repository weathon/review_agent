# MLE-RL: Reinforcement Learning for Self-Improvement in Machine Learning Agents

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Language models have shown significant promise in complex reasoning and coding tasks. However, coding for machine learning engineering presents unique challenges due to the iterative nature of development, long execution times, and the need for continuous self-improvement. In this paper, we introduce MLE-RL trained with reinforcement learning  to address these challenges. 
Our approach reframes the learning process by breaking down long-horizon trajectories into single-step optimizations.
We employ a reinforcement learning strategy that selectively learns from the most informative attempts, optimizing the policy on valuable steps.
In addition, to overcome context limitations, our agent uses a scaffold with a memory module to store and recall high-performing past solutions, facilitating cumulative learning. 
The evaluation on the MLE-Bench demonstrates that our MLE-RL-32B achieves 4.9% improvement over the baseline model in the competition ranking on ML tasks and
achieves competitive performance against state-of-the-art open-source models like DeepSeek-R1-0528. MLE-RL is open-sourced
at https://anonymous.4open.science/r/MLE-RL-CC61

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
MLE-RL addresses the challenge of training LLMs for machine learning engineering (MLE) tasks that require long-horizon, iterative self-improvement through two key contributions. First, it proposes a reinforcement learning strategy that decomposes multi-turn agent trajectories into single-step optimization units (where each "step" represents actions between two valid code submissions) to enable precise credit assignment. Rather than training on all generated attempts, it selectively learns from informative steps using three filters: invalid action masking (removes format errors), valuable step selection (retains approximately top 30%), and dynamic running mean normalization that maintains separate windows for memory-based and scratch trajectories to filter out negative samples. To overcome context length limitations, the MLE-RL introduces a memory module equipped with add and read operations that maintains a pool of high-scoring solutions from past attempts, using AST similarity filtering to maintain diversity. This enables both local (within-trace) and global (cross-trace via memory) self-improvement. MLE-RL-32B demonstrates competitive performance with state-of-the-art models like DeepSeek-R1-0528 despite using only 32B parameters.

### Strengths
1. This paper tackle the hard problem of training llms to learn long trajectory tasks ( Kaggle ML tasks was used in this case ). The proposal of trajectory decomposition and rollout data selection is novel.

2. The MLE-bench results are quite competitive given the scale of the model is just 32B. This shows that MLE-RL is effective in training a small models to specialize in highly complex and long trajectory task well.

### Weaknesses
This paper lacks the required details to reproduce ( lack of training data transparency, training hyperparams is least minimum and no details of what hardware or budget does it cost ).

Analysis section is bit weak on the side specifically the paper only trains one single model : QwQ-32B, I expect in the analysis or ablation to train MLE-RL on a smaller LLMs to ensure its reproducible even for slightly lower compute budget institutes. The other weakness are addressed in my major concerns section.

Only one dataset is evaluated which means questioning the effectiveness of this method but this is minor since no similar benchmarks are close enough to MLE-bench ( minor concern ).

### Questions
**Major concerns:**

1. Which specific training data is the MLE-RL trained on? Throughout the entire paper, it does not mention what is the source of train set, as MLE-bench do not contain any train set at all, all the 75 tasks were classified as test set.

2. Since QwQ is a reasoning model does the reasoning budget has any affect on the final score？

**Minor concerns:**

3. Does the competitive performance from distillation (Sec 5.4) suggests that rejection sampling finetune might be a strong baseline? Since based on the paragraph the distillation just means its trained on the positive rollouts from the RL experiments rather than responses from the final RL model ( which is how Qwen3, Deepseek distillation is done ).

4. What kind of training framework was used in the training of MLE-RL? Is it a proprietary one or existing framework like verl? My understanding is that only MLE-dojo was mentioned but its just a environment to run these rollouts.

5. What hardware is used to trained the MLE-RL, my understanding is the A10 GPU mentioned is just for rollout of the MLE-bench experiments.

6. Typo at line 194: “an agentenvironment interaction”

### Soundness
4

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
4

### Summary
This paper introduces MLE-RL, an RL framework for training large language model agents for MLE tasks. MLE tasks have unique challenges, such as iterative refinement over extended time periods and heterogenous reward signals. The contributions are as follows. First, a novel RL training strategy that decomposes multi-step trajectories into a single-step optimization. Second, a scaffold with memory module to overcome context length limitations. Third, empirical validation on MLE-Bench showing a 4.9% improvement over the baseline model.

The strengths of the approach are as follows. First, the MLE-RL framework is an effective framework for training agents for MLE tasks. Second, the scaffold is an improvement over previous agent scaffolds since it's able to account for long-context issues. Third, the authors do a significant empirical analysis on QwQ-32B, showing that their MLE-RL framework is effective for training language model agents.

The weaknesses of the approach are as follows. First, there is not an analysis on generalization to other domains beyond MLE-Bench. It is well established that training on agentic tasks will improve performance, but it can decrease performance on other coding tasks e.g. general coding asks. Second, the improvement is significant but it would be good to have an analysis on what the model is learning during RL training or where the improvements are coming from.

### Strengths
The strengths of the approach are as follows.
- First, the MLE-RL framework is an effective framework for training agents for MLE tasks.
- Second, the scaffold is an improvement over previous agent scaffolds since it's able to account for long-context issues.
- Third, the authors do a significant empirical analysis on QwQ-32B, showing that their MLE-RL framework is effective for training language model agents.

### Weaknesses
The weaknesses of the approach are as follows.
- First, there is not an analysis on generalization to other domains beyond MLE-Bench. It is well established that training on agentic tasks will improve performance, but it can decrease performance on other coding tasks e.g. general coding asks.
- Second, the improvement is significant but it would be good to have an analysis on what the model is learning during RL training or where the improvements are coming from.

### Questions
n/a

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper MLE-RL presents a reinforcement learning framework to enable LLMs to learn from their previous actions in ML Engineering tasks. Training MLE agents using RL is known to be hard because of several reasons - long execution times, credit assignment etc. This paper decomposes long-horizon trajectories into single-step optimizations and employs selective learning from informative attempts using three filtering strategies: invalid action masking, valuable step selection, and relative improvement selection. They address the issue of long context by introducing a memory module that stores and reuses high-performance solutions. The method shows improvement in performance with a 32B trained model to match the performance of DeepSeek-R1. It beats GPT-4o in many cases.

### Strengths
This paper is one of the first works to explore training machine learning agents using reinforcement learning, compared to past methods that primarily rely on prompting techniques. A key strength lies in using a memory module, which allows the agent to learn from past successful interactions and significantly improves performance across various tasks. The proposed method shows good results matching the performance of a 32B model to Deepseek-R1 and surpassing GPT-4o

### Weaknesses
1. Masking invalid actions: The paper "masks the loss" on any turn that yields an invalid tool call, formatting error, or invalid submission. This removes the supervision signal that the model needs to identify bad actions. This would lead to policy being less robust, being unable to recover from format errors under distribution shift. Moreover, this would provide less reinforcement signal on difficult tasks, as the policy is more likely to make an error on difficult tasks. This would lead to slow and inefficient learning on the difficult ML tasks.
2. The reward function is dependent on the existence of a competitive leaderboard with human participants. This makes it difficult to apply the same methodology to novel problems that don't have an established benchmark or community of competitors. The "Human Rank" is a relative metric, and its meaning can change as the pool of competitors changes. 
3. There is no discussion of the effect of hyperparameters, such as the absolute performance threshold ($\tau_{abs}$), the relative improvement threshold ($\tau_{adv}$), and the size of the running mean window, on the performance of the agent. 
4. The binary reward collapses all the solutions that pass the threshold as good solutions. This might lead to reward hacking and the agent generating just good enough solutions rather than exploring the best solutions. Moreover, the paper maintains separate running mean windows for memory vs reset traces to prevent bias against those starting from scratch. However, this could introduce its own bias - memory traces inherently start from better positions and should perhaps be held to a higher standard. The decision to use binary reward also throws away potentially useful fine-grained reward information.
5. Hyperparameter sensitivity not fully explored: The method introduces many hyperparameters ($\tau_{abs}$ calibrated per competition to retain ~30%, $\tau_{adv}$=0.01, $\tau_{rm}=0.03$, window size W=8, reset ratio p=0.7, memory size=5, etc.) but only ablates the running filter. The robustness and sensitivity to these choices remain unclear.
6. Missing Computations Cost Analysis: No information is provided about total GPU hours for training, wall-clock training time, number of rollouts/sample collection, or comparison of computational efficiency vs baselines. This makes it difficult to assess practical feasibility and reproduce the work.
7. Statistical Significance: While standard deviations are reported, no formal hypothesis testing is performed. Some improvements fall within overlapping confidence intervals, especially for specific medal categories.
8. Incomplete Action Space Description: While three primitives are mentioned, the paper doesn't provide sufficient detail about what information `request_info` can access, or the full specification of the action space structure.

### Questions
1. How does masking the loss on invalid actions affect the agent's ability to learn to recover from errors at test time, compared to assigning a small negative reward? 
2. How sensitive is the agent's final performance to the specific values chosen for the absolute ($\tau_{abs}$) and relative ($\tau_{adv}$) reward improvement thresholds? 
3. Does assigning a uniform reward of 1 to all positive samples disincentivize the agent from distinguishing between good and exceptionally great solutions? 
4. How would the Human Rank-based reward be formulated for new ML problems that do not have an existing competitive leaderboard for comparison?
5. Memory module details: What is the actual distribution of solution diversity in the memory pool? What percentage of training/evaluation traces successfully utilize memory vs fail to improve upon memory solutions?
6. Please provide some details on how the specific values for the hyperparameter were chosen.
7. What are the total training costs in GPU-hours? How does this compare to the computational cost of simply running more rollouts with a stronger model?
8. Can you provide the complete specification of the action space? What specific information can `request_info` access?
9. Why use binary reward after running filter instead of preserving the normalized reward values? Doesn't this discard useful gradient information?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces MLE-RL, a training framework focused on hard tasks like MLE-Bench using RL training that leverages informative rollouts as opposed to all attempts to solve a problem. The authors reframe the optimization problem by splitting long-horizon trajectories into single-step optimization steps to address the credit assignment problem for multi-turn interactions. They also introduce a memory module with push and get operations to overcome long-context limitations and leveraging high-performing solutions which help in iterative self-improvement. Results on MLE-Bench show around 5% improvement using MLE-RL over the baseline policy.

### Strengths
- The paper tackles an important problem of long-horizon RL and how to improve single-turn RL for harder tasks like MLE-Bench.
- Credit assignment for long-horizon problems and multi-turn setups is hard where getting signals for intermediate correct/incorrect actions is important, so the authors convert the multi-turn setup into single-step by splitting actions between two `execute_code` that produce a valid submission as a step.
- The authors propose various choices like masking invalid actions and selecting only valuable steps for training to stabilize the RL training.
- Addition of a memory module to retain historical information which helps in self-improvement improves the performance further.

### Weaknesses
- Presentation is a bit weak. Some figure/table references are wrong which lead to a bit of a confusion, for example in lines 236/255, it should be Figure 3 instead of Figure 5. In line 357, it should be Table 1 instead of Table 6. Another example is missing information, for example, what is $M$ in equation 4?
- A simple baseline of single-step RL is missing where you don't split into multiple steps and take the last `execute_code` call as the step end.
- No discussion on where the RL training data is sourced from in Section 5.1.
- The authors propose various design choices but do not ablate all of them, for example the effect of masking invalid actions and selecting only valuable steps.
- The proposed improvements are shown only wrt a baseline policy, but a baseline RL algorithm should be used to show the improvements like single-turn RL.

### Questions
I've asked most of my questions in the weakness section above. One other question is what's the effect of MLE-RL on OOD evaluation datasets like standard math/code benchmarks like AIME/Livecodebench etc. Does it degrade the performance compared to the baseline policy?

### Soundness
3

### Presentation
2

### Contribution
2
