# Selective Perception: Learning Concise State Descriptions for Language Model Actors

- Avg Score: 4.25
- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3, 6, 3

## Abstract
It is increasingly common for large language models (LLMs) to be applied as actors in sequential decision making problems in embodied domains such as robotics and games, due to their general world knowledge and planning abilities. However, LLMs are not natively trained for embodied decision making problems, and expressing complex state spaces in text is non-trivial. Exhaustively describing high-dimensional states leads to prohibitive inference costs and impaired task performance due to distracting or irrelevant information. Previous LLM actors avoid the issue by relying on hand-engineered, task-specific protocols to determine which features to communicate about a state and which to leave out. In this work, we propose BLINDER (Brief Language INputs for DEcision-making Responses), a method for learning to select concise and helpful sets of state features for LLM actors. BLINDER  learns a value function for task-conditioned state descriptions that approximates the likelihood that a state description will result in optimal actions. We evaluate BLINDER  on the challenging video game NetHack and a real-world robotic manipulation task. Our method improves task success rate by 77% and 14% on NetHack and robotic manipulation respectively, reduces model input length by 83%, and generalizes well to LLM actors of various size and quality.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes BLINDER, a module that learns to summarize task-relevant state features for the LLM actor. The primary motivation is that distractions in the full state description lead to bad performance of the LLM actor. The training pipeline aligns well with the MDP formulation, and the authors provide experiments to support the efficacy of BLINDER.

### Strengths
1. I fully agree with the motivation behind the work. Obtaining a succinct and informative state representation for LLM-based task planning is crucial since state representation is a major hurdle in applying LLM to task planning. However, VLM might significantly alleviate this situation.

2. In the work, the LLM actor is actually considered part of the reward function for BLINDER. As a result, the entire training pipeline aligns well with the MDP formulation, and the formulation is clear.

### Weaknesses
1. I believe a more robust baseline should be compared. Considering that the training of BLINDER utilizes privileged information, such as expert demonstrations, comparing it to zero-shot summarization might not provide a fair comparison. Moreover, with the evolving in-context learning capabilities of LLM, it's important to compare its performance with GPT-4's state feature extraction to understand the technical merits of BLINDER.

2. The experimental section would benefit from a clearer presentation. For instance, the terms 'zero-shot' and 'few-shots' are somewhat confusing. Some instances of 'zero-shot' referred to zero-shot summarization, while 'few-shot' referred to a few-shot LLM actor. The NetHack experiment is presented only in textual description, making it difficult for readers to visualize the experimental setup.

3. The paper would benefit from a more in-depth analysis. I would like to see some evidence, based on the experimental results, regarding the inherent limitations of LLM that necessitate an additional component like BLINDER. 1) It fails to utilize the relevant information amidst all the distractions, even with GPT-4, much like its struggles with simple 4-digit algorithms. 2) It cannot extract relevant information from the distractions, even with GPT-4 and despite examples and prompting. This would help underscore the significance of BLINDER at this moment.

**Update after rebuttal:**

I appreciate the author's response and clarification. Here are my thoughts after reading the rebuttal:

1. The improved performance after using GPT-4 isn't entirely unexpected, given the improved capabilities of a more advanced language model. While creating more challenging tasks is possible, I believe the bottleneck may mainly lie with the LLM agent rather than the state description. However, I agree that minimizing redundant state descriptions is valuable, particularly in aiding human understanding of the LLM’s reasoning process and in identifying key information for the model. Unfortunately, the manuscript lacks an analysis of this aspect.

2. I understand why the authors chose not to separate the state description from the LLM agent, considering BLINDER is tailored for a specific language model. Nonetheless, I believe there's merit in doing so. Whether certain states are relevant or not should be consistent across different language models. Since most language models are somewhat 'black boxes' and can be seen as repositories of commonsense knowledge, the internal reasoning mechanisms are not fully understood. Decoupling could potentially offer deeper insights into BLINDER, such as why it generates irrelevant information.

While BLINDER has potential as a general 'plug-in' for LLM agents, a more thorough analysis and deeper insights would enhance its value.

### Questions
1. In Figure 2, it says GPT-3; I guess that should be GPT-3.5?
2. In the reward function, how does one differentiate whether the low reward is due to an incorrect state description or the inability of the LLM to reason?
3. Does the order of state features affect performance?
4. How will BLINDER generalize to different state features without additional training?
5. Is there any reason for BLINDER to generate hallucinated state descriptions? This is a bit concerning.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The manuscript proposes a method for learning a value function for obtaining better state descriptions for LLM-based actors in interactive games and tabletop object rearrangement tasks.

### Strengths
The paper is well-written

The paper addresses an important problem is utilizing LLMs for sequential task-execution

### Weaknesses
Section 1 — The manuscript states that the "the system should learn to automatically generate task-conditioned state descriptions". The manuscript should discuss here about the proposed mechanisms that allow the learning process remain both general and task-customized enough, to have the best of both worlds. It would be detrimental to the novelty if the generalization of the feature selection process is too dependent on the set of tasks it is predicated on. Discuss in the main content how the tasks and their expert trajectories were chosen. 

Section 1 — The manuscript states that "[u]nlike prior work, BLINDER automatically constructs state descriptions without requiring task- specific prompt engineering." This is not strong a novelty, as BLINDER requires a carefully-chosen set of set of archetypal tasks.

Section 1 — "Given a small set of expert task demonstrations ..." — Certainly, if the tasks are seen, there would be little novelty to the proposed method. However, if the tasks are unseen during training, how can we say that the value function is trustworthy? Would there not be a distribution shift in value (which is conditioned on the expert set of tasks), just as there is in the set of tasks itself? The inference we can draw seems only to be that the training tasks are not significantly different from the test tasks.

Section 2.1-2.2 — It is not sufficient to merely survey the related work. In each subsection, which compares the proposed work with prior art (according to a specified theme or dimension), the manuscript should identify salient weaknesses and concisely specify how the proposed approach alleviates these concerns. For the rebuttal, please provide these statements, for each subsection.

Section 2.1 — The manuscript is missing discussion about how its proposed approach alleviates the grounding problem, which it identified as the prevailing weakness of the prior art in the ‘Language for State Descriptions’ dimension.

Section 2.2 — Same issue as above: missing discussion of how the proposed approach alleviates the weaknesses

Section 2.3 — "[the] approach [in Zhang et al., (2023)] differs significantly from our own and focuses on static NLP tasks rather than sequential tasks." This is not a sufficient basis for discrimination, as the approach proposed by the prior art, Zhang et al., (2023) can be very easily applied to sequential problems, since each decision-making step is, itself, a static NLP task. In the present manuscript’s problem formulation, nothing about the state description process depends on the surrounding agent’s action-generation.

Section 3 — According to this formulation, the state features seem only additive. Are there situations in which the reverse formulation (i.e., starting with omega-hat and iteratively removing irrelevant features) may be better? Would the same value function be learned? Why did you chose this formulation, or why does it not matter?

Section 3 — Does the “size” (or “capacity”) of the state representation only monotonically increase? What does this look like for long-horizon tasks?

Section 4.1 — The manuscript states "We use π to construct a state description x out of state features ω. Starting with x0 = ∅, π iteritively [sic] adds ωt∗ to xt as long as it increases in value and there are still state features to add (see Algorithm 1)." Does the policy learn an ordering bias? The manuscript should provide an “ablation” in which the ordering of the state elements is perturbed, at the end of each feature selection process.

Section 4.2 — the manuscript states "we reward BLINDER to maximize the likelihood that xf elicits the same action from the LLM actor as the target action a∗ from the expert trajectory." This carries an implicit assumption that the expert trajectories are somehow representative archetypes for test-time execution environments. If they aren’t representatives, the reward structure could be severely biased towards the training environment; if they are representative, the test-time execution environment would then not be so different from the training environment and the claim of unseen generalisation suffers (and the empirical performance improvement would be insignificant).

Table 1 (Caption) — The manuscript states "Although BLINDER sometimes includes unnecessary information, any extra included state features do not distract from the current task." How should we know? The manuscript should provide evidence that removal of this unnecessary information has no effect on performance.

Section 4.3 — The manuscript states "While we refer to x as a set, the text input to Vθ is ordered using x’s insertion order. " Ordering bias? See comment above

Section 5 — The manuscript states "In each of our experiments, BLINDER is trained on a set of training tasks and is provided five expert trajectories for each task." Missing discussion about how expensive it should be expected to be to provide five expert trajectories for each task, if the proposed method were applied to different settings. Does the required number of expert trajectories need to change as the number of tasks changes? What is the scaling factor for the number of demonstrations needed for an arbitrary number of (diverse) tasks?

Section 5.1 — Lack of fair comparison / strong baseline. The manuscript states "On average, BLINDER removes 83% of the original state features." Unfortunately, by itself, this number is meaningless. The manuscript should adapt an alternative approach as a strong baseline, e.g., from Zhang et al., (2023), in order to produce fair comparisons. Then, the manuscript could discuss how much distracting information a strong baseline or ablation removes, here, compared to the proposed approach

Section 6 (Intro) — An entire subsection should be dedicated to describing why just these selected tasks were chosen for training and testing. The current discussion is insufficient and leaves open the possibility that these tasks were selected to favour the proposed method. Why not evaluate on a much larger set of tasks? Why not evaluate on all the tasks? Why not organise different task splits, e.g., where the test variants are completely different from the training variants (rather than selecting test tasks that only differ by whether an object starts in the player’s inventory or not)?

Figure 6 (Caption) — It seems that a large-capacity model is needed for leveraging the state description produced by BLINDER.

Section 7 — Lack of fair comparison / strong baseline. Missing a fair comparison with a strong baseline that attempts test-time prompt tuning.

### Questions
N/A — see weaknesses above.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes BLINDER, a method for automatically selecting task-relevant text features for input to an LLM actor. On NetHack and a robot manipulation task, BLINDER decreases the length of the input text while improving performance relative to the full-text input. BLINDER also learns intuitive text selection and zero-shot generalizes to larger, more powerful LLMs.

### Strengths
- Experiments demonstrate that BLINDER improves the success rate over providing a full, zero-shot LLM generated and manually engineered state description (Fig 2).
- BLINDER is evaluated on two diverse domains: Nethack and a robotic manipulation task.
- BLINDER learns generalizable state descriptions. The method is able to generalize to variants of the Nethack training tasks and is able to transfer the descriptions zero-shot to a different LLM. 
- The paper includes qualitative examples of the selected state description and how it evolves throughout the episode.

### Weaknesses
- BLINDER can select relevant text features, but will this be important as LLM context lengths increase? NetHack has at most 40 features, and the robotic task has 90 features. A more powerful LLM, like GPT-4 could perform better from the full state description. Sec. 6.2 is perhaps intended to address this point, yet the section feels incomplete. It doesn't discuss the results in Fig. 4, and Fig. 4 is not discussed elsewhere in the text.
- BLINDER requires manually defining the set of grounded state features. Defining these features could be more difficult in more complex environments involving open-ended descriptions of environments. Instead, BLINDER will only work in environments where the state consists of a set of fixed state features. 
- BLINDER requires a precollected dataset of experience to learn the value function. The paper collects this dataset with a random policy. How can this scale to more complex problems where this random policy would not collect meaningful data?

### Questions
- Why does BLINDER outperform the manual state descriptions? 
- What prompt is used to zero-shot summarize the state description (zeroshot baseline in Fig 2)? Is it the same prompt as in 5.1? Why not use GPT-3.5 as the summarizer since it is a more powerful LLM than Flan-T5 (as demonstrated by Fig 2)?
- What do the error bars in Fig 2 represent? This is not described in the caption or elsewhere in the paper. 
- Section 2.3 states that prior works that explore compressing input tokens for shorter context lengths don't meet the needs of sequential decision-making. Why is that the case, and why can't such methods be used in the same setting as BLINDER (at least in the case of Flan-T5 Actor)?
- What does Fig. 4 mean by "context length per action"? What exactly does the x-axis quantity refer to in Fig. 4?
- Table 1 separates state descriptions into relevant and irrelevant text with blue and red colors. But what does the black text in the state description from BLINDER correspond to?

### Soundness
4 excellent

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors propose BLINDER, a method for selecting state descriptions for LLM actors in sequential decision making problems.  It does this by learning a value function for task-conditioned state descriptions that approximates the likelihood that the descriptions will lead to optimal actions.  BLINDER reduces context length and removes distracting information forming an optimal state description for the actor.

### Strengths
The paper is well written and clear.  Technical details are easy to follow and the motivation is clear.  Evaluations are provided on multiple applications and compared to a competing approach.

### Weaknesses
"We hypothesize that BLINDER does better on these tasks by learning that different state features are relevant at different points in a trajectory". Are there some qualitative examples that show this?  It would be interesting to see how different features are selected at different stages of the task.

In Figure 2, BLINDER is only achieving significant improvement in 3/5 tasks with the flan-T5 actor and 2/5 with the GPT3 actor.  In Figure 3, BLINDER shows significant improvement over manually specified descriptions in 3/5 tasks.  In Figure 6, BLINDER only shows significant improvement on 2/6 tasks.  These results are not very convincing given the simplicity of the zeroshot approach in comparison.  Can the authors better highlight the benefits of the approach and potential tradeoffs of the zeroshot approach?  I think the context length section touched upon it but it was not immediately clear why I would use the proposed approach over the simple LLM feature selection.

### Questions
see above

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
