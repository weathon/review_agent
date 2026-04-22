# Assessing the Reverse-Engineering Abilities of Large Language Models

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Using AI to create autonomous researchers has the potential to accelerate scientific discovery. A prerequisite for this vision is understanding how well an AI model can identify the underlying structure of a black-box system from its behavior. 
In this paper, we explore how well a large language model (LLM) learns to identify a black-box function from passively observed versus actively collected data. We investigate the reverse-engineering capabilities of LLMs across three distinct types of black-box systems, each chosen to represent different problem domains where future autonomous AI researchers may have considerable impact: programs, formal languages, and math equations. Through extensive experiments, we show that LLMs fail to extract information from observations, reaching a performance plateau that falls short of the ideal of Bayesian inference. However, we demonstrate that prompting LLMs to not only observe but also intervene---actively querying the black-box with specific inputs to observe the resulting output---improves performance by allowing LLMs to test edge cases and refine their beliefs. By providing the intervention data from one LLM to another, we show that this improvement is partly a result of engaging in the process of generating effective interventions, paralleling results in the literature on human learning. Further analysis reveals that engaging in intervention can help LLMs escape from two common failure modes: $overcomplication$, where the LLM falsely assumes prior knowledge about the black-box, and $overlooking$, where the LLM fails to incorporate observations. These insights provide practical guidance for helping LLMs more effectively reverse-engineer black-box systems, supporting their use in making new discoveries.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper evaluates whether large language models (LLMs) can reverse-engineer unknown black-box problems, which are usually handled by Bayesian algorithms. The authors formalize this problem, curate the noise-free dataset from previous works and benchmark the selected LLMs (GPT-4o, in the main text) across three domains, including programs, formal languages, and mathematical equations and comparing passive observation versus active intervention paradigms. The authors identify two main failure modes, overcomplication and overlooking, appearing in the usual case without intervention and try to improve it with the proposed intervention. The evaluation is based on natural language reward/score (judged by another LLMs). Results show that LLMs can plateau under passive observation but show improvement with intervene across three experiments.

### Strengths
- Built a measurable reverse-engineering benchmark and evaluate the LLMs’ performance on it, which can be relevant to build autonomous research agents.
- Curated three setups from previous studies across formal language, math, and program for experiments and evaluations, which may be used as evaluation for future works

### Weaknesses
- Relies on LLMs (in specific GPT-4o in the experiments) as the nature language “judge” model, which may bias scoring and introduce circular evaluation effects compared to verifiable score or rewards.
- The obtained performance improvement under “intervention” might reflect prompt length or potential CoT-like reasoning help instead of hypothetically being contributed by actively collecting data. Necessary ablation studies on this is absent or not enough given Table 1.

### Questions
- The Bayesian inference baseline presumes full access/configuration to hypothesis space and need prior knowledge to define, does this set an unfair or unachievable upper bound for LLMs? If not, what is a more comparable Bayesian baseline to LLMs?
- Can you show with corresponding experimental evidence to justify the LLMs truly plan informative interventions and then make better “reverse-engineering” through intervention, instead of by (relatively trivial) increased reasoning length. In other words, what is the effect of reasoning verbosity (CoT length) versus actual hypothesis refinement?
- Can you also show the variance/std of the improvement margin in Fig 5?
- In Fig 8, it seems for quite a few (LLMs, Task) pair, the intervention does not help for specific data points used, as well as does not show consistent improvement along with increased data points. Could the authors explain around this phenomenon?

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
4

### Summary
The paper evaluates the abilities of LLMs at forming hypotheses to explain the behavior of a black-box system given input/output observations (and the ability to query the system further). Through experiments on three tasks (python lambda expression, formal language rules, math equation inference), the authors find that (1) LLMs from frontier still lag behind Bayesian systems (oracle baseline), (2) through an expert human evaluation, the failures can be commonly attributed to either overly complicated rules, or ignoring/overlooking some information in the examples, (3) when allowing the LLM to interact with the system, performance improves most commonly when the the intervention is unconstrained and the LLM can reason from past examples (as opposed to verbalizing the black box as python code).

### Strengths
1) The task is simple and well motivated, the experiments are well designed to evaluate model behavior and then analyze failure cases and the writing is clear and easy to follow. 

2) The extension from pure observation-based to interaction-based reverse engineering is intuitive, and the experiments showing that the active experience does not transfer to other models (Sec 5.3) is compelling as it implies that the reverse engineering process is unique to the particular model. 

3) The manual analysis is informative in categorizing model behavior, and allows us to ask more qualitative questions to go beyond individual benchmark numbers (see Q2 and 7 below).

### Weaknesses
One result that is important for verifying the claim is the variance in performance across different sets of observations (of the same size). What do you mean by variance by seeds in L.234? Is it the generation hyperparameter, or also the sampling algorithm for the particular observations used for reverse engineering?

This could potentially be in Questions for the authors, but I think it's worth bearing mention that the work could do a better job engaging other work, particularly published literature that tries to verbalize the decision functions for labeling in ICL examples [1], describe datasets and distributions in text [3] and elicit preferences of users in natural language [2, 4]. The tasks are different, but the principles of the experiment seem similar. This would be fine, but I bring it up here as the paper lacks a strong, surprising finding beyond our priors of LLMs from the literature. To mitigate this, I think moving from a single amorphous 'LLM' category to analysis between models could help. A potential finding that is already in the results is the relative performance of different models, which got pushed to Appendix F. Do effects such as scale, or training recipe, or model family affect performance? Are models well calibrated in their rule judgments?

[1] Si, Chenglei, et al. "Measuring Inductive Biases of In-Context Learning with Underspecified Demonstrations." The 61st Annual Meeting Of The Association For Computational Linguistics. 2023.

[2] Li, Belinda Z., et al. "Eliciting Human Preferences with Language Models." The Thirteenth International Conference on Learning Representations.

[3] Zhong, Ruiqi, et al. "Goal-driven discovery of distributional differences via language descriptions." Advances in Neural Information Processing Systems 36 (2023): 40204-40237.

[4] Handa, Kunal, et al. "Bayesian preference elicitation with language models." arXiv preprint arXiv:2403.05534 (2024).

### Questions
1) One concern is that the experiments report performance over multiple seeds. Do you also vary the particular examples given to the model in your experiments? Is N=10 a proper subset of the examples in N=50? What is the variation as a 

2) For figure 8, is the judge always GPT-4o? Do the different models have similar or very unique interaction patterns, in terms of the examples they seek clarification about? Do they tend have have different profiles among constructing new queries, response pairs, and concluding the analysis?

3) L.236-237- When you say that the "LLM performs M = {5, 10, 20, 50} rounds of interventions conditioned on the initial set of 10 observations", I took the points on the graph to mean 10 + $m \in M$ turns. Is this correct? Does the interaction baseline performance change if you provide more initial observations?

4) Could you comment on issues of potential data leakage of the rules (to be inferred) into the model pre-training? From [1], LLMs are able to decipher certain algorithms more than others due to prevalence in pre-training. 

5) Given that the experiments with other models are in the appendix, why present GPT-4o in the main paper when it is the same as the judge model? 

6) I think the absence of a non-oracle baseline is making it hard to interpret some of the results (since the Bayesian methods have a set of rules to search within, L.882). This could be where something like comparing the performance of different models on one graph (either of model families or scales or training steps) could help. 

7) Do different models fail systematically at certain examples (i.e. longer or more complex ones)? Do different models overcomplicate the same examples?

[1] Prabhakar, Akshara, Thomas L. Griffiths, and R. McCoy. "Deciphering the Factors Influencing the Efficacy of Chain-of-Thought: Probability, Memorization, and Noisy Reasoning." Findings of the Association for Computational Linguistics: EMNLP 2024. 2024.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper tries to find out how well LLMs can reverse-engineer 3 black box systems: program, math and formal languages. They run three kinds of experiments: giving the LLMs only a randomly sampled set of observations, giving them the power to intervene and interact with the black box systems and allowing an LLM to see the observations obtained from intervention of another LLM. These results are compared to Bayesian systems — and show that LLMs lag behind them, especially when not allowed to intervene. The paper also identifies two failure models: over complication and over simplification which leads to errors in hypothesis generation.

### Strengths
- The paper is well written and easy to follow with helpful illustrations.

- The experiments are rigorous. The authors use 3 distinct tasks (program, formal languages and mathematical equations) along with a graded difficulty level to provide a fine-grained evaluation of model's capabilities. The evaluation spans six models with proper statistical analysis of the results.

- The paper focuses on the timely topic of assessing AI models as autonomous researchers for scientific discovery. Given these models are increasingly being used for research workflows, this work provides an important step forward.

### Weaknesses
- The LLM as a Judge paradigm here uses only 1 model with a 0-10 scale and although there’s agreement with humans, there isn’t 100% agreement. The prompt for judge is also slightly subjective which can lead to noisy results. Along with that, using a judge model from the same family as the one being analysis has been proven to be noisy and unreliable. How do you think you can address this concern in the paper? 

- The two types of failure cases mentioned are very intuitive — irrespective of a model or human, one can only overcomplicate or oversimplify a given problem. It will be great if authors can add more concrete examples and analysis for the same, this would improve the readability of the paper. 

- It will be really nice if authors can think of qualitative analysis on understanding the failure modes more closely and write about it in the paper. This will help. 

- The experiments regarding transfer are not very rigorous and require more information to ensure we can count it as a significant contribution.

- More analysis of the limitations of the models would be helpful. See Questions section for more details. 

- It is expected that the model will perform well when given more information, the result that it performs better in the observation-intervention setting compared to observation-only setting seems may be considered trivial. An analysis of how interventions differ from additional passive data would strengthen the paper.

### Questions
- Why does the performance usually plateau after 10 observations?

- In Line 412, it is said that in the experiments the LLMs are allowed to reason once every five queries. This needs to be explained better because some of the strategies inherently require reasoning in each step or before generating each query. Can authors add more details here.

- Can you share any examples where the models dynamically refine their hypothesis in response to their own interventions. It would help better understand why repurposing LLM's intervention data for another model does not work well.

- What intervention method was used for reporting results in Figure 3 and 4?

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
2

### Summary
This paper explores the abilities of LLMs to reverse engineer black-box systems. Three different types of black-box systems are explored: list-mapping programs, formal languages, and inferring the parameters of an economics math equation. Models are evaluated under three different conditions: a passive-only condition where they only receive observations from the program; an intervention condition where they're allowed to query the program; and an intervention-yoked condition, which is similar to the passive condition except that some of the observations are queries that were produced by another LLM. The failure modes of models are broken down into two different types: namely, overcomplication, which is inferring additional behavior that isn't supported by the data, and overlooking, where the LLM does not make full use of the available data. 

LLMs behave below an ideal Bayesian observer in the passive-only condition, but the ability to actively query and intervene on the system recovers some, but not all, of that performance gap. Moreover, this seems to be tied to the process of active exploration itself, and not simply the informativeness of the examples, as the intervention-yoked condition does not produce the same performance.

### Strengths
- Originality
    - Weakness: I'm not terribly familiar with the active inference literature, but I am familiar with tasks of the type, LLM evaluations, that require LLMs to reverse engineer black-box systems. And it doesn't seem to me that either the tasks or the methodology used in this paper represent meaningfully novel contributions.
    - Strength: At the same time, I could simply be misinformed about the state of the literature.  It's possible that everyone is trying to jump the shark with end-to-end scientist AI and not doing careful analysis of each of the cognitive components, in which case this paper would represent a good grounding in the fundamentals.
- Quality
    - Weakness: It seems like only three attempts per task were performed. At least that's how I'm interpreting the phrase “report performance over three seeds.” This seems like a very low number of attempts. Granted, there are multiple “variants” of each task, corresponding to different numbers of data points, but only three runs on each task variant seems very low.
    - Strength: I was surprised by the use of LLMs as a judge, given that these seem like tasks with closed-form solutions that can be algorithmically verified, but the appendix showing its agreement with human annotators was reassuring.
        - Weakness: However, I don't know how to interpret the descriptiveness score. It's not clear to me how moving from a descriptiveness score of, for example, 30 to 50 would affect the model's ability to make use of it’s understanding of the system to do useful work.
    - Weakness: The LLM transfer experiment feels distracting. I would not expect to see improved performance when transferring results from one model to another if we don't see improved performance within a model. It feels like the paper would have been tighter and more cohesive by omitting that experiment.
- Clarity
    - Strength: The graphs were clearly presented, clearly labeled, and very readable.
    - Strength: Inclusion of example programs and agent responses was helpful.
    - Weakness: The diagram at the start of the paper was not great; it was a bit busy and not terribly informative. It did not do much to improve my understanding of overcomplication and overlooking, for example.
    - Weakness: There are four different intervention types mentioned in Section 5.2, but it's not clear to me which of those is the intervention type that is used when comparing observation-only versus observation with intervention in the approach. I'm assuming it's the variant that's just titled intervention, but I'm not sure.
- Significance
    - Weakness: Perhaps this is my bias toward more realistic tasks, but I don't feel like this paper meaningfully informed my understanding of model capabilities and the bottlenecks to accomplishing productive work in the real world.

### Weaknesses
See Strengths section (listed them together)

### Questions
1. Am I correct that only three attempts per task variant were performed?
2. Can you provide more details on how to interpret the descriptiveness score?
3. Why was the LLM limited to reasoning only once every five queries? Were any experiments done to show that that would be the optimal amount? For example, do LLMs get confused if they reason more often?

### Soundness
2

### Presentation
3

### Contribution
2
