# Towards Agentic Self-Learning LLMs in Search Environment

- Decision: Reject
- Scores: 2, 6, 4, 4

## Abstract
We study whether self-learning can scale LLM-based agents without relying on human-curated datasets or predefined rule-based rewards. Through controlled experiments in a search-agent setting, we identify two key determinants of scalable agent training: the source of reward signals and the scale of agent task data. We find that rewards from a Generative Reward Model (GRM) outperform rigid rule-based signals for open-domain learning, and that co-evolving the GRM with the policy further boosts performance. Increasing the volume of agent task data—even when synthetically generated—substantially enhances agentic capabilities. Building on these insights, we propose Agentic Self-Learning (ASL), a fully closed-loop, multi-role reinforcement learning framework that unifies task generation, policy execution, and evaluation within a shared tool environment and LLM backbone. ASL coordinates a Prompt Generator, a Policy Model, and a Generative Reward Model to form a virtuous cycle of harder task setting, sharper verification, and stronger solving. Empirically, ASL delivers steady, round-over-round gains, surpasses strong RLVR baselines (e.g., Search-R1) that plateau or degrade, and continues improving under zero-labeled-data conditions, indicating superior sample efficiency and robustness. We further show that GRM verification capacity is the main bottleneck: if frozen, it induces reward hacking and stalls progress; continual GRM training on the evolving data distribution mitigates this, and a small late-stage injection of real verification data raises the performance ceiling. This work establishes reward source and data scale as critical levers for open-domain agent learning and demonstrates the efficacy of multi-role co-evolution for scalable, self-improving agents. The data and code of this paper are released at https://anonymous.4open.science/r/Towards-Agentic-Self-Learning-4D63

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a self-learning paradigm for improving the capabilities of LLM based agents. The authors propose a method that leverages GRMs, which they find to outperform the rule based methods. The benefit of the GRM is that the GRM can be further improved through RL training as the prompt generator produces harder and harder prompts, the GRM can be further trained to produce rewards for this task.

### Strengths
This area of self-learning is of significant importance for producing autonomous agents that can improve their performance for new tasks. As far as this reviewer can tell, the work is original and distinct from the previous works in the literature.

### Weaknesses
1. Results seem to be across a single seed of training (I couldn't find information about training setup i.e. number of seeds in the manuscript) which is very limited for research in RL. Even a small handful of seeds (i.e. 5) combined with the correct statistical methods (i.e. IQM[1]) can provide valuable insights. I strongly suggest that the authors work to include additional training runs of their method

2. Combined with #1, results are shown as point estimates and therefore we cannot evaluate the statistical significance of the results. Leveraging IQM [1] would provide insights into whether the experimental results are significantly different from each other or not. 

3. The figures are hard to read as the text and labels are much too small.

4. The authors re-introduce acronyms for their methods (i.e. Generative Reward Model (GRM) line 283 and line 068) throughout the text. Please introduce the acronym once when the method is first introduced, and then use the acronym going forward. 

5. I think that a discussion of scaling in RL is needed to give the reader full context of the literature. From the way the paper is written, a reader could think that scaling isn't done in RL outside of this proposed setting which is untrue. There are many works [2-4] that study parameter scaling in RL, while a few works [5, 6] provide similar insights into the relationship between scaling and the amount of data used during training, albeit in a more traditional RL problem setting. 

Citations
1. https://arxiv.org/abs/2108.13264
2. https://arxiv.org/pdf/2506.17204
3. https://arxiv.org/abs/2405.16158
4. https://openreview.net/forum?id=kfYxyvCYQ4
5. https://arxiv.org/pdf/2506.03404?
6. https://arxiv.org/abs/2503.05126

### Questions
1. In the GRM training phase, the GRM performs N independent rollouts that generate a series of predictions for the score. These predicted scores are then compared against the reference scores - where do the reference scores come from? This seems like a bottleneck that the authors attempted to avoid by using the GRM but if GRM training relies on an outside verifier (i.e. human labels), then the positives of the GRM are lost.
2. How could this method be translated into more traditional RL settings (i.e. interactions with Atari or Mujoco environments)?

### Soundness
2

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
The work introduces a new framework, Agentic Self-Learning, that aims to train an LLM with minimal human effort. This framework comprises three components: a prompt generator that creates tasks for a policy model to solve, and a generative reward model trained against a reference score to provide a reward signal to the prompt generator, which maximizes the entropy of the return, and to the policy model, which maximizes the return. Agentic Self-Learning is evaluated against competitive baselines, and multiple ablations are presented to understand its behavior.

### Strengths
1. The presented method is novel and creative, focusing on a recent topic of artificial intelligence.

2. The reading flow is good. The method is clearly presented.

### Weaknesses
I. Many details are missing to assess the presented method properly:

   a. By learning to increase the entropy of the return over a batch of predicted tasks, the prompt generator is incentivized to make simple tasks simpler and hard tasks harder. A comment that provides intuition for why this behavior leads to tasks well-adapted to the current policy model is missing.

   b. No details are provided to explain the way the difficulty feedback is implemented. Given that the prompt generator is trained to maximize entropy, listening to the feedback would increase the loss because it would guide the model to generate hard tasks simpler and simple tasks harder. Clarifying this aspect would be helpful.

   c. Line 280, the statement "This comparison produces a binary correctness indicator for each rollout" is not specific enough. Moreover, in Line 279, "we compare it against the reference score s", no details are given to describe the reference score.

   d. In Section 3.1, the environment in which the proposed method is evaluated is not described. Adding a short description would be beneficial.

   e. Line 312, "all roles share parameters", no information on which parameters are shared is disclosed.

   f. VeRL is used as a base algorithm; however, no justification for this choice is provided. Additionally, it would be preferable to discuss how the performances would change if another algorithm were selected.

   g. The lack of a preliminary Section increases the entry barriers of the presented approach, which lowers the potential audience for this work. Adding a preliminary section would strengthen the impact of the presented work.

   h. Adding a comment on the importance of the hyperparameters such as the length of the different phases would improve the usefulness of the presented approach. Additionally, commenting on the influence of the capability of the initial policy on the performance of the presented framework would be interesting.


II. Overall, the presentation can be improved:

   a. In Figure 5, the confidence intervals can be added.

   b. Parentheses are missing around many citations. For example, Line 139, "Shi et al." should have parentheses. Moreover, many citations do not have dates.

   c. The acronym PURM, in Line 98 is not defined.

   d. In Line 188, "(1)" should be written instead of the first "(3.1)", and "(2)" should be written instead of the second "(3.1)".

   e. The font size of the all figures should be much bigger.

### Questions
N/A

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
3

### Summary
This paper examines whether self-learning using LLM generated rewards can outperform rule based verifiable rewards.Their agentic self-learning approach outperforms strong baselines such as Search-R1. The primary concern of reward hacking is mitigated by continued training of the Generative Reward Model and late-stage verification data injection. This framework consists of multiple components: the prompt generator, rewarded by the entropy of the score distribution of the downstream policy; the generative reward model, this is trained to mimic the verifiable reward and a policy model, which provides candidate solutions.

### Strengths
The paper tackles a challenging problem and proposes a novel framework which has some promise of self-improvement. It identifies the generative reward model is the bottleneck for self improvement, as well as identifying/mitigating an instance of reward hacking. There are interesting design decisions such as sharing of policy and reward model parameters.

### Weaknesses
The primary concern with RLAIF is the difficulty in avoiding reward hacking, whilst interesting I am yet to be convinced the proposed approach solves this problem.
 
Prompt Generator: Whilst rewarding based on the entropy of responses seems like an intuitive solution, it isn't clear how it would avoid reward hacking for solutions with many answers, i.e. always asking to roll a 1000 sided dice just has high entropy in responses/scores but is a degenerate prompting strategy.
 
Reward Model: As I understand it, this model is trained on the binary signal of if the reward model matches the verifiable reward output. Therefore, I am unsure how it can outperform the verifiable reward signal it is approximating when evaluated on the same task (training a model of the verification signal could perhaps generalize to domains where a signal is not present).
 
Policy: Asking the prompt generator for easier or harder prompts based on the score initially seems sensible however it could induce failure modes induced by the policy being punished for doing better, as if it answers correctly the questions become harder. Whilst perhaps not breaking it warrants discussion, particularly regarding the efficiency of learning.
 
Figure text is too small, with no information regarding seeds or uncertainty.
 
4.4 Ablation is useful to see that each component is learning however, as it's a closed loop system, it's hard to tell if the model has reward hacked without witnessing model outputs. For instance, I'm struggling to infer how you would notice the reward hacking behavior you mention from this plot. Therefore we're just taking the claim that at least one reward hack exists at face value without knowledge of how you discovered it and if there are any other degeneracies.

### Questions
1. How does continual training mitigate reward hacking? What details can you provide toward the exact training mechanism?
2. If injecting verification data seems to help, is this not contrary to the hypothesis that GRM data is better?
3. Fig 2, why the sudden drop in reward for the rule-based method at the end?
4. Is there a tradeoff between learning efficiency and final performance?
5. How did you discover the reward hacking behavior in 4.4? How can we be assured there aren't other failure modes?
6. You mention the generative judge "can provide more informative and tolerant reward signals", are you able to elaborate? Which aspect of training induces these qualities

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
2

### Summary
This paper presents Agentic Self-Learning (ASL). This closed-loop reinforcement learning framework enables LLM agents to self-improve without human-labeled data or rule-based rewards. Through controlled experiments in a search-based agent environment, the authors identify two key determinants of scalable agentic training: the reward source and the scale of the data. ASL unifies task generation, policy execution, and reward evaluation within a shared LLM backbone, allowing the Prompt Generator, Policy Model, and GRM to co-evolve through iterative reinforcement. Empirical results show that ASL achieves sustained performance gains, surpassing strong baselines such as Search-R1, Absolute Zero, and R-Zero, even under zero-labeled-data conditions.

### Strengths
- Human-free learning: the authors demonstrate scalable self-learning without human annotations or rule-based rewards, a significant step toward autonomous LLM training/self-improvement.
- Generalizability of the method and experiments: the work offers, in principle, a scalable, domain-agnostic recipe for self-improving agents, extensible to broader open-domain or multimodal tasks.

### Weaknesses
- Limited domain scope: the experiments are confined to a text-based search QA environment, leaving it unclear how ASL would generalize to other domains (e.g., coding or multimodal tasks).
- Complex training loop: the multi-role co-evolution process increases implementation complexity and computational cost.
- Non-significant experimental results: most experimental curves do not seem to show statistical significance.

### Questions
1. How well does ASL generalize beyond search-based QA? Could the same framework be applied to domains like coding?
2. Since the GRM is identified as a bottleneck, have you explored ways to scale or diversify it?
3. How do you ensure that the synthetically generated tasks remain diverse, realistic, and representative of real-world challenges?

### Soundness
2

### Presentation
4

### Contribution
2
