# ROSETTA: Constructing Code-Based Reward from Unconstrained Language Preference

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
Intelligent embodied agents not only need to accomplish preset tasks, but also learn to align with individual human needs and preferences. Extracting reward signals from human language preferences allows an embodied agent to adapt through reinforcement learning. However, human language preferences are unconstrained, diverse, and dynamic, making constructing learnable reward from them a major challenge. We present ROSETTA, a framework that uses foundation models to ground and disambiguate unconstrained natural language preference, construct multi-stage reward functions, and implement them with code generation. Unlike prior works requiring extensive offline training to get general reward models or fine-grained correction on a single task, ROSETTA allows agents to adapt online to preference that evolves and is diverse in language and content. We test ROSETTA on both short-horizon and long-horizon manipulation tasks and conduct extensive human evaluation, finding that ROSETTA outperforms SOTA baselines and achieves 87% average success rate and 86% human satisfaction across 116 preferences.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces the ROSETTA framework, which uses LLMs to directly translate human natural language preferences into executable reward code. It addresses a problem in embodied intelligence: enabling agents to adapt to human preferences in real-time and online. ROSETTA can generate new reward functions and train policies, demonstrating performance significantly superior to existing baselines.

### Strengths
- The paper addresses a critical problem by directly modeling human language, tackling the challenge of aligning with unconstrained feedback.

- The capability for online adaptation is a major strength. The agent can adapt to a completely new human preference within a single interaction, which enables dynamic and continuous interactive alignment.

- The evaluation methodology is comprehensive. The authors propose three-dimensional evaluation metrics that assess the policy from multiple perspectives, establishing a strong benchmark for this problem.

- The framework was evaluated on a large-scale set of human preferences, achieving very high success rates and human satisfaction.

### Weaknesses
- Portability is questionable: The method relies heavily on domain knowledge. Migrating it to a new robot or a new environment would likely require significant manual setup.

- The policy learning process is simplified: While the paper highlights "single-step adaptation," it glosses over the policy learning process. The overall training loop is complex: after the reward function is generated, a policy must be fully trained in the simulator before the user can watch it and provide new feedback. The authors should provide details on the time and resources required for this training phase, as well as the iteration time for the LLM to generate the reward function.

- The necessity of the human selection step is unclear: The system relies on training multiple reward code variants and then requires a human evaluator to watch videos and select the best one. It is unclear if this human-in-the-loop selection step is essential and how performance would be affected without it.

- Experiments are simulation-only: Despite the large number of preferences and comprehensive experiments, the evaluations are conducted exclusively in simulation. Unlike a paper proposing a purely novel algorithm, this paper's primary contribution is a feasible framework. Therefore, it needs to demonstrate the feasibility of the full real-sim-real loop, perhaps with a simple case study. Simulation-only experiments overlook numerous potential errors that can occur in this cascaded process.

### Questions
1.Has the author verified the entire process in a real robot scenario?

### Soundness
3

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
This paper proposes Rosetta, a framework that leverages foundation models to ground and disambiguate natural language preferences, construct multi-stage reward functions, and implement them via code generation. The core contributions include the Rosetta framework itself, an evaluation framework measuring alignment, semantic match, and optimizability, and extensive experimental validation on 116 human preferences across five manipulation tasks. Key results report an 87% average success rate, 86% human satisfaction, and outperformance over state-of-the-art baselines like Eureka and Text2Reward, particularly for non-corrective preference types.

### Strengths
**Well-Structured Framework**: The three-module design (Preference Grounding, Staging, Coding) effectively handles unconstrained language by disambiguating references and structuring rewards, as illustrated in Fig. 2 and described in Sec. 4. Integration of domain knowledge through verification questions in the Coding module (Sec. 4.3) enhances reward reliability, with 95.38% of generated functions running after error correction. The framework supports online adaptation to dynamic preferences, maintaining performance over multiple iterations (Tab. 1), which is a significant advancement over prior fixed-goal methods.

**Comprehensive Evaluation**: The proposed evaluation framework (Sec. 3.1) with alignment, semantic match, and optimizability metrics provides a holistic assessment beyond standard success rates, addressing the subjectivity of human preferences. Extensive human evaluation on 116 preferences across five environments (Sec. 5.1) demonstrates robustness, with results broken down by preference type (Fig. 3, Fig. 4) and language/content diversity (Fig. 5, Fig. 7). High performance metrics (86% human satisfaction, 87% success rate) across short-horizon and long-horizon tasks indicate practical utility for real-world applications.

### Weaknesses
**Insufficient Details and Reproducibility**: Hyperparameters for policy training are not detailed in Sec. 4.4 or Sec. 5.1, making replication difficult. Computational costs and resource requirements are omitted, despite the use of multiple LLM calls per preference in Sec. 4. The policy selection process in Sec. 4.4 relies on human annotators choosing from variants, but no inter-annotator agreement or selection criteria are reported, potentially introducing bias.

**Limited Generalization**: The framework is task-agnostic but tested only in ManiSkill 3 environments; transfer to other simulators or domains is not evaluated, as acknowledged in Sec. 6 (Limitations). Semantic match evaluation in Sec. 3.1 relies on expert assessment, which may not scale and lacks objectivity compared to automated metrics; no inter-rater reliability is reported. The approach depends on specific foundation models (gpt-4o, ol-mini), and while open-source model tests in Sec. 5.2.1 show promise, full ablation on model choices is missing.

### Questions
1.	How does ROSETTA perform with long sequences of preferences in terms of performance degradation and computational efficiency? Does the foundation model context window become a bottleneck? (Sec. 5.2.3; Table 1)
2.	What is the impact of alternative error correction strategies (beyond iterative ol-mini) on success rate and human satisfaction? Could reinforcement learning for code correction be beneficial? (Sec. 4.3; Sec. 5.2.1)
3.	How generalizable is ROSETTA to non-manipulation embodied tasks? Are there module design limitations requiring modification? (Sec. 1; Sec. 5.1)

### Soundness
3

### Presentation
2

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
The paper tackles an important problem of adapting the behavior of embodied systems to natural language feedback. It adopts the approach of using foundational models to understand diverse and free form language commands and feedback to generate code that steers the actions of the agent towards the desired objectives. Further, the authors present additional evaluation metrics to analyse the behavior of the method beyond success rate, in terms of human satisfaction, feasibility and semantic similarity of the adapted behavior. The authors demonstrate that they out-perform the baselines across multiple tasks and users, with improvements in human satisfaction.

### Strengths
- The papers focus on the setting of unconstrained human preferences where the task is unknown initially and the agent has to adapt on the fly. It is a very important skill that is required for AI systems to be deployed in the real-world.
- The paper is very well written with very descriptive figures, with the motivation, method and experiments clearly laid out.
- The authors transfer their method across both simulation and real-world tasks which shows the strong applicability of the proposed solution.
- The paper includes extensive qualitative examples from their experiments to show the different success and failure cases, while also thoroughly discussing the limitations of the paper.

### Weaknesses
- The method and the different components are explained well, but is it difficult to understand the individual contributions of each module to the final method? It would be interesting to see other additional baselines that run an LLM end2end for the whole tasks (with prompts to reason or use CoT to perform staging, ground and code generation by the same model)? This could make the contribution of having this modular stack as compared to making a single call to the LLM.
- The authors include average reward values across different models in their method, but what is the average reward achieved by the different baselines? 
- In this method, it requires humans in the loop to provide the fitness function, which seems to be a very expensive process. It would be very interesting to talk about the sample efficiency and human effort required in the overall method?
- Overall, the paper seems very solid, with some missing details in the presentation. I would be happy to raise my score as my questions are answered.

### Questions
- Do we expect Eureka to perform similarly if you can provide it with the correct fitness function at each step?  Is it an upper bound on performance? A small explanation of how the baselines were implemented under the interactive setting would be helpful to understand the differences from prior work.
- What instructions are provided to the annotators? Because since the tasks in simulation have a single goal, how does the diverse preferences arise under the same objective for the relatively simpler tasks? Could the authors include some information about the number of participants, and other details about the experiments?
- For the real-world tasks, it would be helpful for reproducibility to understand the implementation details.

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
This paper proposes a multi-stage LLM framework that is used to design reward functions that match indicated preferences at each time-step in open-ended behavior generation tasks. The paper constructs detailed prompts to help with this design process and appears to show improvement in human satisfaction above the baselines.

### Strengths
- I really appreciate figure 5, the authors have a very thorough categorization of the different ways that users can express their preferences.
- The paper appears to improve over the results of its baselines. 
- The paper provides an interesting, somewhat new goal of generating arbitrary behaviors in each iteration as selected by a human annotator.

### Weaknesses
Most of the "weaknesses" in this work are placed in the questions category below. I am uncertain if similar levels of tuning was done on the baselines relative to how much work was done in designing the prompts for Rosetta and would like to understand that better before being able to make a final judgement on the work.

## Very minor (does not affect my score)
- This paper puts me in mind of two papers, "Efficiently Generating Expressive Quadruped Behaviors via Language-Guided Preference Learning" by Clark et al. and "In-Context Preference Learning", but Yu et. al. which also have a similar iterative loop that uses human preferences to select amongst generated candidates. You may want to look at these works as neither are cited but seem highly related.

### Questions
- I don't understand the amount of human annotation used in these. The paper points to section F.2 but the relevant content is not there? Where are the details on your experimental protocol for the annotators? Is it one human watching 80 videos? 80 different people? How long are the experiments for them?
- I'm not sure what is meant by Rosetta's "domain knowledge", a frequent claim throughout the paper. Where is the domain knowledge contained in Rosetta as opposed to in the LLMs that comprise Rosetta? More particularly, in what way does Rosetta have domain knowledge that your baselines does not (in particular, I'm responding to line 153)? Do you just mean the prompts in section C? If so, are similar prompts provided to your baselines?
- What LLMs are used for the Eureka and Text2Reward baselines? Are they also using 4o and o1-mini?
- Can you provide more information on the protocol used by the roboticists to evaluate your reward function? The text points to section D in the appendix but section D appears to be about something else. 
- In 372 to 377, what are these scores that are being given? Is it the average satisfaction score across several categories? Given that the performance degrades quite a bit when you paraphrase the prompts, why do the authors claim that "the specific prompt matters less than the conceptual structure"?
- Why does the Sat. Score fall off so much after the first round?
- In the real robot experiments, are the "two-iteration preference chains" generated by human annotators? What is the state dictionary that must be populated there?
- How much did this all cost, particularly the annotations? This is relevant information for those who might choose to build on this paper.

### Soundness
3

### Presentation
2

### Contribution
2
