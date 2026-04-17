# Evaluating LLM Goal-Directedness

- Decision: Reject
- Scores: 4, 4, 2, 6

## Abstract
To what extent do LLMs use their capabilities towards their given goal? We take this as a measure of their goal-directedness. We evaluate goal-directedness on tasks that require information gathering, cognitive effort, and plan execution, where we use subtasks to infer each model's relevant capabilities. Our evaluations of LLMs from Google DeepMind, OpenAI, and Anthropic show that goal-directedness is relatively consistent across tasks, differs from task performance, and is only moderately sensitive to motivational prompts. Notably, most models are not fully goal-directed. We hope our goal-directedness evaluations will enable better monitoring of LLM progress, and more deliberate design choices of agentic properties in LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper aims to define and evaluate "goal-directedness" (GD) in LLMs. This is different from task performance, which conflates capability (is the model capable of completing the goal) and GD (how strongly does the model *try* to complete the goal). The key question is how to measure GD in a way that is independent from capabilities. To do so, the authors design text-based "Blocksworld" tasks which naturally decompose into subtasks. For example, one task is to stack three blocks into two towers of similar height. The model has access to a "stack" action and a "measure" action, which returns noisy measurements. The natural way to complete this task is to (1) take many measurements of each block's height, (2) decide on the optimal configuration, and (3) make that configuration. Each of these tasks can also be tested in isolation, e.g., "determine the height of block A". If a model performs well on each subtask in isolation, one can conclude that it has the necessary capabilities for the overall task. If such a model performs poorly on the overall task despite this, the authors argue that this indicates poor GD. The authors evaluate 8 proprietary LLMs in this way and find that most models are fairly goal-directed but not fully goal-directed, and GD is relatively consistent across the tasks they consider. The authors also show that (de)motivational prompting (e.g., "really go for it" or "this doesn't really matter") has nonzero but limited impact on GD.

### Strengths
GD in LLMs is clearly an important topic. I think the core idea of the paper -- separating capabilities from GD by comparing subtask performance and task performance -- is quite interesting and creative. While this idea has limitations, I think it's a pretty good first attempt. The experiments show that the idea works reasonably well, at least in this setting. I particularly appreciated the frank discussion of limitations by the authors.

### Weaknesses
I have some interrelated writing concerns and methodology concerns. I don't think any of these concerns are dealbreakers individually, and for an initial proof-of-concept paper like this, I don't expect completely airtight experiments. However, it does seem like there are a lot of things that could be cleaned up. Overall, I'm inclined towards acceptance *if* the issues below can be addressed. My current rating of weak reject is based on the current state of the paper (i.e., that would be my rating if no changes are made).

**Concern 1**: The task design and the subtask decompositions seem somewhat arbitrary. I also initially found the descriptions quite confusing.

1a. I initially was very confused about what the tasks were (Section 3.3). My understanding now is that there are four main overall tasks: Information Gathering (IG), Cognitive Effort (CE), Plan and Execute (PE), and their Combination. IG only has one subtask, which is height estimation. CE and PE have mostly overlapping subtasks, including one which confusingly is called "plan execution". Also, I initially thought IG, CE, and PE were subtasks of the main overall task. Can you confirm that my understanding is correct?

1b. IG is listed as having one subtask (height estimation). But even after the heights are estimated, it still has to select to two tallest blocks and stack them. Why is that not also a subtask? And if a model is good at height estimation but bad at IG, how do you know that shows low GD and not that it's bad at stacking?

1c. GD is conditioned on a capability level c, and Definition 3.1 requires optimizing over the set of policies with that capability level. My understanding is that you do this by using a Monte Carlo algorithm which takes the subtask outputs as inputs, i.e., the optimal policy given the subtask outputs. But it seems to me that this only works if the overall task fully decomposes into subtasks, which I already argued is not true for IG. It's harder for me to tell whether it's true for the other tasks.

1d. How is regret defined? I understand that 0 is best, but it's hard to interpret Figures 5 and 6 without knowing the scale of regret. For example, what does regret = 10 indicate? Is that good or bad?

**Concern 2**: Some of the methodological choices seem concerningly ad-hoc, specifically with regards to excluding certain data.

2a. "When a model failed to complete the task within a reasonable number of steps (usually around a 100, but depending on the exact task and number of blocks), we excluded the run from the analysis." Why are failed runs excluded rather than considered failures? Also, how did you choose the max number of steps? Do the results change if this maximum is varied? What percentage of runs is this relevant to?

2b. "Models that failed to understand a task have been dropped." Similarly, why are failed runs excluded rather than considered failures? Also, how did you determine whether a model "failed to understand" the task?

2c. "We also experimented with thinking models, but they frequently hallucinate their own environment responses, making systematic evaluation difficult." If thinking models are truly so bad at these tasks that GD cannot even be evaluated, that is quite surprising, since thinking models are supposed to be better at multi-step tasks like these. It's especially surprising that this holds for _all_ of the thinking models the authors tested (although the paper doesn't mention how many thinking models were tested). I think this deserves more exploration. Assuming the authors are correct, I don't think it's their responsibility to fully debug these thinking models, but I think they should rigorously confirm that their evaluation is correct (and provide evidence in the paper to support that). 

**Concern 3**: These tasks seem overall fairly easy. If future models solve both the subtasks and the overall tasks perfectly, does that imply that those models are 100% goal-directed? Or are the tasks just so easy that it doesn't require much GD to solve them? The authors state that GD was fairly consistent across tasks in their experiments, but maybe that was just because all of the tasks are quite similar in structure and difficulty level. The authors do acknowledge that their restricted Blocksworld environment has limitations, which I appreciate, although they don't mention this specific limitation. I don't think the authors need to solve this problem in order for the paper to be publishable, but it would be nice to include a discussion of this in the paper.

**Minor feedback**. Some of these are phrased as questions, but I don't expect responses. These points are all minor enough that none of them are affecting my overall evaluation of the paper.
1. "But task performance arguably says more about capability than goal-directedness". Why? It seems to me that task performance is basically the combination of capability with goal-directedness.
2. Definition 3.1: I think "capability" should probably be defined before using that term in a definition.
3. Definition 3.1: What about division by zero? This can happen if e.g. a random policy is optimal, right?
4. You might consider a different name than "Blocksworld", which is quite similar to "Gridworld", which is quite different.
5. I think "Theorem 3.1" should be "Definition 3.1" everywhere.

### Questions
I mixed in my questions with the weaknesses above. My top priority is making sure that I fully understand the experiment design and the rationale behind it. Once I'm confident that I understand everything, I can make more concrete recommendations for revisions and what specific factors would influence my final rating.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper analyzes the differences in LLMs' performances on sub-tasks and the combined tasks, denoted as goal-directness. The metric is defined as $(R_\pi-R_\text{random})/(R_\max-R_\text{random})$ where $R_\max$ is infered from sub-tasks' performances. It evaluates LLMs' goal directness in blocksworld where the model can measure the heights of blocks (with noise) and stack blocks to achieve tasks (such as to build two towers of similar heights). It mainly evaluates older LLMs such as gpt-3.5, 4o, and claude-3-7-sonnet and analyzes their performances.

### Strengths
It's interesting to evaluate the goal-directness of LLMs.

The paper is generally well-written.

### Weaknesses
* The blocksworld environment is synthetic and simple. It would be better to evaluate the models' performance in more realistic settings such as web/code agents. 
* The evaluated LLMs are old. The results may not be informative enough to guide current actions. 
* There are other factors to explain performance differences on combined tasks other than the goal-directness. For example, some models may prefer shorter outputs / fewer iterations than other models, which would make the models choose to stop earlier in height estimations and thus get worse performances.

### Questions
* Are there results on more realistic tasks and more recent LLMs?
* How to distinguish other factors to explain the performance differences on combined tasks?

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
2

### Summary
The work considers the notion of goal-directedness of existing LLM systems. The authors distinguish this property from that of performance and use it to capture the notion of the ability of an AI system to use all of its capabilities/resources to achieve its end goal. The authors then use this definition to evaluate the goal-directedness of many of the existing models. The evaluations are focused on blocksworld tasks. They also carry out additional evaluation on motivational prompting.

### Strengths
I think the paper makes a very interesting attempt at taking a concept from psychology and philosophy, and providing it with a mathematical grounding, which is then used for the evaluation of the models. I also appreciate the fact that the authors state all their assumptions and evaluation limitations quite clearly.

### Weaknesses
Now coming to the weakness, I am still a bit confused about whether the notion of goal-directedness adds something to the current discourse about the model behavior, which existing metrics cannot capture. For one, I am still not clear what the difference is between an optimal policy and one that makes use of full capabilities ($R_{\pi_C^*}$). Of course, you would have to account for a decision-making framework that is able to perform meta-reasoning, but I believe that could still be captured within the definitions of an optimal policy. It would be interesting if the authors could present a theoretical argument as to what aspects of behavior an agent can be provably captured by your new measure that existing ones cannot capture. This is different from an empirical evaluation, which, in the case of LLMs, could suffer from numerous confounding factors.

On the other hand, I feel like introducing notions like goal-directedness might introduce a level of anthropomorphization into LLM evaluation that could be harmful. While the authors try to claim that the effects seen here aren't simply due to an increase in context length, couldn't the composition of the subtask cause an increase in the complexity of the overall task that could explain the perceived reduction in goal-directedness? In general, a lack of clear understanding of the theoretical factors that affect LLM performance makes it hard to truly understand the phenomena we are observing here.

### Questions
I request the authors to respond to my comments provided above.

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
The paper investigates the extent to which LLMs are goal-directed when dealing with composite tasks involving information gathering, cognitive effort and plan execution, using the involved sub-tasks' performance as indicator of their capabilities. Evaluations of state-of-the-art LLMs indicate that their goal-directed behaviour remains consistent across tasks (i.e. involving information gathering, cognitive effort, planning and execution), and is only moderately affected by motivational prompts.

### Strengths
- The authors seek to tackle a timely question: to what extent LLMs exhibit goal-directed behaviour when handling composite tasks requiring multi-step planning and capability integration.
- Sub-task performance is used to infer capabilities, and subsequently, to measure goal-directedness as the extent to which these capabilities are deployed is an interesting methodological contribution.
- The paper evaluates several state-of-the-art LLMs across different prompt types and demonstrating consistent trends across tasks.

### Weaknesses
- Goal-directedness is partly used as a proxy for task difficulty increase from individual sub-tasks to composite tasks.
    - The manuscript does not sufficiently dissect whether low goal-directedness truly reflects motivational failure or inherent inability to solve complex tasks.
- The definition of what is considered an atomic task into which composite tasks are decomposed appear loosely defined.
- The notion of regret is somewhat embedded in methodological details and lacks a clear, intuitive standalone definition early in the text.

### Questions
- Have you considered how reinforcement learning–oriented post-training methods may systematically influence the goal-directedness of LLMs, especially in contrast to approaches relying solely on supervised fine-tuning with or without instruction-following data?
- The manuscript could benefit from the inclusion of a more intuitive explanation of the regret metric early in the text to improve clarity and reader comprehension.
- There are minor editorial issues, notably a missing reference in Appendix A on line 701 and a grammatical error on line 877 (i.e. "The result" instead of "They results").

### Soundness
3

### Presentation
3

### Contribution
3
