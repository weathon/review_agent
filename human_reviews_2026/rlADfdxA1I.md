# Enhancing Social Intelligence in LLMs with Hierarchical Reasoning and Utterance-Level Goal Rewarding

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 6, 4, 6

## Abstract
Large language models (LLMs) excel in structured tasks but struggle with dynamic social interactions, where success requires long‐term goal coordination and rapid adaptation. Current methods often apply uniform goal‐based rewards to every utterance, overlooking the specificity of objectives at each dialogue turn and failing to account for the rationale of potential strategies. Inspired by the Theory of Planned Behavior, we propose the Think‐Strategy‐Response (TSR) framework, which decomposes social dialogue into two hierarchical stages: high‐level strategic planning and low‐level linguistic execution. To optimize TSR, we introduce Linearized Hierarchical Reinforcement Learning with Variance‐Gated Rewards (LHRL‐VGR), a novel algorithm that dynamically routes rewards—balancing goal completion and strategy adherence—based on the variance of goal achievement scores. Experiments on the SOTOPIA benchmark show that our approach fine‐tunes a Qwen2.5-7B agent to surpass the GPT‐4o baseline by 7.32% in goal completion success, demonstrating state‐of‐the‐art performance in multi‐agent social negotiation tasks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces a new reward function to perform RL on LLMs to improve their performance over interactive conversations. It does so with a strategy-based reward (Sec. 4.1.2) and then a 2-stage GRPO (Sec. 4.2). The method is called: Linearized Hierarchical
Reinforcement Learning with Variance-Gated Rewards (LHRL-VGR), and it consists of multiple components, such as 
Reward for the strategy content, Transferred goal completion reward, and more.

### Strengths
In my opinion, this work provides some incremental improvement over existing RL works for LLM training. Experimental results show promise for a few datasets, and cover quite a few LLMs. Ablation study is also done to tease out the influence of various components in their proposed reward function and training mechanism.

The paper is easy to read in terms of relaying their method across. However, some flaws exist in their presentation, where not much design justification is done to explain why certain components are chosen.

### Weaknesses
I feel the paper has several flaws. I'm happy to discuss this with the authors.

1. This paper adopts several components to improve the RL training of LLMs. Although some justification is provided in the paper, it is hard to interpret or understand fully why certain things are designed in a certain way. For instance, why was the Goal-completion reward chosen in that way? I understand there's some explanation to justify the reward structure, but how does the resulting LLM responses change w.r.t. the choice of reward that the author chose? There is basically no explanation or interpretable insights into how the LLM response changed due to the special reward structure chosen by the authors. As such, we even wonder whether these reward structures are chosen amongst many different candidates until we reach the desired incremental performance improvement over existing baselines.

2. Another example of this happening: " In multi-turn interactions, the model can repeat content from its
previous responses, which reduces informational value and inflates context length" -> Perhaps what the authors can do is to show that with/without the Non-repetition Reward, how does the actual LLM response change? Doing this to justify the design of the RL framework not only makes the paper stronger, but also makes the design more convincing. If not, it just feels like the authors are presenting things that worked best in their experiments, without any generalizable findings. If the paper is on multi-turn conversation, perhaps give some good illustrative examples of such conversations?

3. The third flaw is that the paper is quite incremental in nature because it combines a few different rewards and training mechanism with no big novelty or fundamental improvements. I'm not sure if this is a subjective point (let's see if other reviewers agree with me), but unfortunately I do not have a good suggestion to fix this flaw because it is quite fundamental. All I can suggest is provide more interpretable and empirical insights into how different components this paper introduced influenced the LLM responses, which could possibly make the paper stronger. But I really do not have a good suggestion to fundamentally improve the paper's incremental nature. Good luck! I'm happy to discuss this further.

### Questions
See weaknesses.

### Soundness
2

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
This paper presents a novel two stage training pipeline based on GRPO for the joint optimization of step-level and high-level rewards. The authors formulate and motivate each of their reward and algorithm choices well and demonstrate statistically significant improvements over strong baselines usch as SOTOPIA-RL.

### Strengths
1. They design the reward in a very novel way, with a goal-completion, strategy-followed, non-repitition, as well as a reward routing mechanism based on the variance of the underlying goal reward. This is important as it allows for them to select between when the goal completion is the deciding factor versus when the step level reward is that 
2. The two stage GRPO formulation for optimizing two rewards simultaneously is clearly explained, as well as well motivated and ablated for the motivation behind structuring the reward in this way
3. The objective function in equation 13 does something interesting where they combine the global group 
4. The paper reports statistical significance results on their results table, assuring the reader that the performance gains observed are significant

### Weaknesses
1. Somewhat misleading statement about +7% on sotopia pi. The abstract frames this as a percentage improvement while the results section does not present the results in percentages. This makes it confusing to the reader. A note in table 1 explaining this discrepancy would be helpful
2. Typo in figure 2 “stage” instead of “satge” 
3. A heavy reliance on LLM as a judge based rewards in both reward estimation and evaluation of SOTOPIA and a lack of verification of whether these rewards are accurate
4. For reward specification, some details are missing for what criteria are being used by the LLM to evaluate the strategies on a likert scale.

### Questions
1. The strict structure of the LLM based rewards is surely helpful in making the rewards more aligned and accurate. However, does the paper show any human studies or analysis of whether these rewards are accurate? 
2. This reviewer is slightly confused in equation 13, why is the KL divergence within the summations rather than outside of it? Based on my understanding of the standard GRPO formulation the KL divergence should applied within the first summation over G but not within the interior one over o
3. What critiria are evaluated on a likert scale in equation (5), line 219? 

To improve the score I give on this paper, I would like more information on reward calibration and specification, an acknowledgement in the ethics statement about potential misuse and an answer to question 2.

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
This paper proposes the Think-Strategy-Response (TSR) framework, a two-stage approach to improving social intelligence in large language models (LLMs). They decompose social dialogue into strategic planning and linguistic execution (generating contextually aligned responses). To optimize TSR, they perform Linearized Hierarchical Reinforcement Learning with Variance-Gated Rewards (LHRL-VGR) — a reinforcement learning algorithm that routes rewards dynamically between goal completion and strategy adherence based on variance in performance. They evaluate their method on SOTOPIA and SOTOPIA-Hard benchmarks and shows a 7.32% improvement in goal completion success over the GPT-4o baseline.

### Strengths
- The design is explicitly grounded in the Theory of Planned Behavior (TPB)
- The reward provide adequate signals to help improve performance on the SOTOPIA benchmark
- Fine-tuned Qwen2.5-7B has better performance than GPT-4o baseline

### Weaknesses
- There are several related works that perform hierarchy with LLMs, including different versions of ReACT that are goal-conditioned. How does your method compare to them?
- The comparisons in Table 1 appear incomplete. It is not fair to directly compare the performance of GPT-4o with Qwen2.5-7B-Instruct (ReACT), as they differ significantly in scale and capability. How does GPT-4o perform when equipped with the ReACT framework? Does it outperform your proposed method? I recommend rerunning the second part of Table 1 using all algorithms on the same, strongest model backbone to ensure a fair and controlled comparison.
- From my understanding, your algorithm relies heavily on multiple LLM evaluators which could incur high costs. Would this be an issue in more long-horizon tasks?

### Questions
- How long are the dialogues generated? Is this truly testing long-horizon capabilities?
- I am not sure if the three stages of Thinking, Strategy, and response is necessarily hierarchical, and seems more so like planning which has been done. Additionally, the linearization of the Hierarchical Reinforcement Learning (HRL) structure is a bit confusing.
- Did you consider other social interaction benchmarks?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes the TSR framework, a hierarchical algorithm that decomposes actions in social dialogue into high-level thinking and planning, and low-level responses. The key contribution is in jointly training high-level strategy and low-level response using LLM-evaluated rewards, using novel variance-gating to dynamically assign the reward of responses. The method achieves strong performance in the SOTOPIA benchmark.

### Strengths
1. The authors propose a principled framework for training hierarchical dialogue agents using custom rewards to jointly train both high-level planning and low-level execution. This is a clear improvement beyond traditional approaches that train using a single reward signal. 

2. The variance-gated reward mechanism is novel, and addresses the challenge of noisy LLM evaluators when using LLM-as-a-judge for training.

3. The method achieves strong empirical performance on challenging benchmarks.

### Weaknesses
1. A major downside is the complete  reliance on LLM evaluation for reward signals. This makes it unclear whether potential biases in the evaluator will limit the generalizability of the approach to other social dialogue benchmarks. 

2. It is unclear how important the non-repetition reward is, as it seems like mostly a "hack" to avoid repetitive behavior. It would be interesting to see an ablation with this reward removed from the objective, to understand if it is truly necessary for performance. 

3. The method itself should be noticeably more complex than evaluated baselines, as to my knowledge, it is the only one that performs 2-stage training. It would be helpful if the authors made clear what the additional overhead is during training and inference.

### Questions
1. Have the authors considered using different policy models for high-level strategy and low-level response generation? It would be interesting to see if that has any impact on final performance.

2. The authors use a scaling function on the goal-completion reward. How important is this scaling, i.e. would the method still work well with standard normalization or just using the raw difference?

### Soundness
3

### Presentation
3

### Contribution
3
