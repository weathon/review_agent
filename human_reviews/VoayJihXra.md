# NeSyC: A Neuro-symbolic Continual Learner For Complex Embodied Tasks In Open Domains

- Avg Score: 7.00
- Decision: Accept (Poster)
- Scores: 8, 5, 5, 10

## Abstract
We explore neuro-symbolic approaches to generalize actionable knowledge, enabling embodied agents to tackle complex tasks more effectively in open-domain environments. A key challenge for embodied agents is the generalization of knowledge across diverse environments and situations, as limited experiences often confine them to their prior knowledge. To address this issue, we introduce a novel framework, NeSyC, a neuro-symbolic continual learner that emulates the hypothetico-deductive model by continually formulating and validating knowledge from limited experiences through the combined use of Large Language Models (LLMs) and symbolic tools. Specifically, we devise a contrastive generality improvement scheme within NeSyC, which iteratively generates hypotheses using LLMs and conducts contrastive validation via symbolic tools. This scheme reinforces the justification for admissible actions while minimizing the inference of inadmissible ones. Additionally, we incorporate a memory-based monitoring scheme that efficiently detects action errors and triggers the knowledge refinement process across domains. Experiments conducted on diverse embodied task benchmarks—including ALFWorld, VirtualHome, Minecraft, RLBench, and a real-world robotic scenario—demonstrate that NeSyC is highly effective in solving complex embodied tasks across a range of open-domain environments.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This work introduces a continual learning framework for dynamically restructuring neural-symbolic knowledge into rules via a Rule Reformulation process. It employs a contrastive generality improvement scheme based on Learning from Interpretations (LFI), a form of Inductive Logic Programming (ILP). The learned rules are utilized for interaction during Rule Application, supported by a memory-based monitoring mechanism within the Answer Set Programming (ASP) framework. Feedback from the Rule Application phase collects new data and periodically triggers Rule Reformulation to further refine the logical rules.

The experiments feature a diverse set of baselines, robustness evaluations, and ablation studies on key components, such as backbone VLMs and prompts. The results highlight the framework’s potential to seamlessly integrate low-level execution with logical reasoning and natural language instructions.

### Strengths
The work provides a solid introduction to the foundational neural-symbolic knowledge required for its framework, including Learning from Interpretations (LFI) within Inductive Logic Programming (ILP) and Answer Set Programming (ASP), as part of the background preliminaries.

The explanations effectively convey the system's core ideas, offering essential details about key functions, such as the use of LFI in Rule Reformulation and the mechanisms for memory modification.

Comprehensive information about NeSYC is provided in the Appendix, along with examples of neural-symbolic designs, making it easier for readers to understand the underlying processes.

The experiments are extensive, covering diverse environments and settings, demonstrating NeSYC’s capability and robustness.

### Weaknesses
Some aspects of the method remain unclear. For example, what exactly is the role of $\mathcal{I}_d$, and how does it relate to $\mathcal{G}_d$ as introduced? A reorganization of Section 2.3 may be necessary, as it forms the foundation of the problem setup.

Additionally, while the authors seem intent on making NeSYC capable of handling a broad range of situations, many of the symbolic formulations explaining the method feel too abstract. This is not a major issue, but it would be helpful if the authors provided more concrete examples of how these symbols are implemented within the main body of the method section, rather than placing all such details in the appendix.

The left portion of Figure 1 is also confusing. It is not adequately addressed in the annotations, leaving the reader unsure of its significance.

Finally, certain experimental settings could benefit from clearer descriptions. For example, in the Dataset section (Section 4.1), the format of the expert-level episodic experience data is ambiguous. Is the data presented in natural language, code, or another format? Providing compact and clear explanations of these settings would make it easier for others to understand the data processing steps necessary for reproducing a similar system.

### Questions
### Questions Regarding $itermax$ and Rule Learning ($R$)

- I am curious about the role of the fixed **$itermax$** in learning the rules $R$. Does setting a fixed $itermax$ always guarantee a reasonable system with appropriate rules? Could you provide results for different $itermax$ values to show their impact on performance and the resulting rules?
- Additionally, I am particularly interested in the notion of **convergence** within the NeSYC system. It might be valuable to explore how convergence is defined in your case. How sensitive is the system to environmental changes—would significant changes disrupt convergence? Also, is it possible for the rules $R$ to exhibit a **cyclical pattern** rather than converging to a stable state?
- On the topic of **continual learning**, I noticed that your system appears to stop growing its rules $R$ at some point (**$itermax$**), with subsequent testing based on a fixed set of rules. This seems different from my understanding of continual learning as a life-long process where rules evolve continuously based on new data. Could you conduct experiments where NeSYC and baseline agents keep learning and testing in various settings without a fixed $itermax$? It would be interesting to compare their performance, especially in terms of how quickly each system improves its success rate across tasks.

### Questions Regarding Experiments

- I have some concerns about the "Step" evaluation in your experiments. Although this evaluation follows prior work, I worry about potential overfitting. Achieving tasks does not always require strict adherence to ground truth; instead, achieving key milestones or sub-goals should suffice. This is particularly true for manipulation tasks—for example, a robot’s trajectory does not need to exactly match the demonstration. In Table 1, I noticed that both NeSYC and other systems show high "Step" evaluation scores, indicating close similarity to demonstrations. This raises questions about these systems' generalization capabilities in new environments. Could you clarify how NeSYC ensures generalization beyond following exact demonstrations? Are there additional experiments or insights that address this concern?
- Similarly, in the **Ablation Study** (Section 4.3), could you provide more insight into why simplified planning improves Goal Completion (GC) and "Step" evaluation? 

- For the **Robustness to Experience Incompleteness** experiment in Section 4.3, could you elaborate on what you mean by *Noisy* and *Imperfect* experiences? What preprocessing steps did you apply to the datasets or system to create these noisy or imperfect conditions?

- In the **comparison of different dynamics predicates** (Section 4.3), could you clarify what your **F1 score** represents? How was the evaluation data collected to calculate the F1 score? What insights are you aiming to provide with the changes in F1 score, especially in relation to the characteristics of the predicates?

- Lastly, I would appreciate more detailed observations from your **real-world experiments**. Much of the current description is qualitative. Were these real-world scenarios tested multiple times, and did you observe recurring patterns? It would also be helpful to quantify these observations with metrics, rather than relying solely on descriptive results.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This submission proposes NeSyC, a neuro-symbolic continual learner for open-domain embodied tasks. It combines LLM with symbolic tools to improve the generalization capability with limited experience. Extensive experiments are conducted on four benchmarks and on a Franka robot in the real world. The proposed framework shows competitive performance compared to multiple existing methods.

### Strengths
- Good presentation.
- Extensive evaluation on multiple benchmarks and on a real robot.

### Weaknesses
- The overall system seems very complicated. It is unclear which part contributes to the overall performance. More detailed analysis and ablations are necessary to shed light into the framework design.
- Tasks are too easy. How dynamics are the environment is also unclear (in paper it only mentioned low dynamics and high dynamics). 
- Since it's a planning framework, we'd expect it can handle long-horizon tasks. Considered tasks are instead short-horizon.

### Questions
See above.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces NeSyC, where a large language model is given access to ILP like framework, specifically Answer Set Programming (ASP), alongside an ASP solver like clingo, where the agent iteratively comes up with hypotheses, attempts to solve tasks, and builds up a knowledge base about the world, aided by a symbolic tool. The authors evaluate their approach on open-domain environments, and show performance improvements over previous results.

### Strengths
- The proposed method seems novel, and the combination of neural agents and symbolic tools in open-domain settings is of interest to the community at large.
- The method outperforms previous methods on open-domain environments.
- The proposed method seems to be able to handle high-dynamic settings where the world changes in unpredictable ways.

### Weaknesses
- The experimental results seem to have different numbers from the baselines, particularly in a comparable environment like ALFWorld (see questions about Reflexion and ReAct). The baseline methods in their own existing work did not use some of the environments used in this work. The numbers seem to be different enough to ask the question on performance, though it is possible this is not a weakness and the difference can be explained.
- There is no comparison to the proposed approach for RLBench on existing work on that environment like ARP+ or RVT.
- The writing can be improved to be clearer, as there are several complicated modules at play. It would be good to include qualitative examples of tasks, plans, and trajectories (see questions).

### Questions
- In Line 354, the authors mention temperature = 0, why was greedy sampling used?
- What was the observation space for the various environments examined? Were RGB images provided, like was done in e.g., LLM-Planner?
- Why are the ReAct performance numbers in the static case so different from the ReAct paper (71 SR reported in their paper for ALFWorld vs 35.8 SR). Same for Reflexion (>0.9 SR reported in their paper on ALFWorld vs 39.0 SR). Is this purely from the difference in the LM used?
- How many trials were used for Reflexion? How many trials were used with the proposed method?
- In Figure 12, the Tower of Hanoi example, why did the agent decide to first put the rings in pegs in the wrong order, only to finish it correctly later?
- In Appendix B.5, do the authors still use VoxPoser for low-level robot control?
- The robotics tasks seem to rely on VoxPoser for low-level control. The reason for the performance of the proposed approach is “constrained by factors such as actuator range, grip force, and balance”, but the plans in Table 11 don’t have any such parameters. How is NeSyC influencing things like grip force? Does the model come up with things like “fragile” in Appendix D.3.1?
- What do some qualitative planned trajectories on a task from all the baselines, and the proposed method look like? Would be great to include those.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
This paper addresses the challenge of generalizing knowledge for embodied task planning from prior experiences to novel scenarios and domains using a neural-symbolic continual learning approach. The proposed approach, NeSyC (Neural-Symbolic Continual Learner?), represents general knowledge and experience (hypotheses and grounded instances of action preconditions and effects, respectively) using Answer Set Programming and relies on existing ASP solvers to find plans that satisfy goals according to the best current hypothesis. Given a set of experiences (the effect of actions taken given the precondition state), an LLM is used to parse these experiences into positive and negative examples based on the action affordances. Given these positive and negative examples, an LLM is used to generate a set of hypotheses by first extracting background knowledge from the examples then generating hypotheses that satisfy batches of the examples. The hypotheses are further curated via an LLM-based implementation of \theta-subsumption to include the most general of the generated hypotheses. Given these hypotheses, a symbolic hypothesis interpreter validates each with respect to all of the examples, providing feedback and a score for each hypothesis. This process of hypothesis generation and interpretation repeats for a maximum number of iterations, resulting in a final, most general hypothesis to be used as the general knowledge during planning.
	
Given a natural language instruction of a task and the current history of execution, an LLM is used to generate both goal states associated with the instruction and the current state of the world. A symbolic task planner uses the general knowledge from the iterative hypothesis generation-interpretation phase to find plans that reach the goal state. During execution, an error handler identifies after each action whether it succeeded or failed (or whether the effects match the general knowledge) and updates a working memory accordingly. In the case of failure, the current working memory is appended to the initial set of experiences, and the system re-enters the hypothesis generation-interpretation phase to update the general knowledge used for planning.

The proposed approach is evaluated through extensive experimentation on multiple standard benchmarks and compared against existing state-of-the-art methods, as well as demonstrated on a real robot in a complex tabletop scenario.

### Strengths
- The problem being addressed is of high significance to the embodied intelligence community.
- The paper is well written and organized; the method is clearly and thoroughly explained and illustrated.
- The contributions are clear, novel, and significant.
- The experimental evaluation is extensive and thorough. The approach is evaluated across multiple diverse benchmarks with increasing difficulty within each benchmark (increasing dynamics). The methods against which the approach is compared are, to the best of my knowledge, the most appropriate state-of-the-art methods.
- The performance is a significant improvement over other methods for dynamic environments.
- Several relevant ablations are included that help answer questions about the relative performance of different components; an appropriate suite of state-of-the-art LLMs are included, showing that the hypothesis generation-interpretation phase is very successful across LLMs, with the exception of Llama-3-8B.

### Weaknesses
- The proposed approach performs worse than CLMASP (another neuro-symbolic approach) in static environments for 3 of the 4 benchmarks.

### Questions
I recommend strong accept at this time because of the strengths listed above.

Here are a few questions that I would appreciate having answered:
- What is the motivation for batching the examples during the hypothesis generation step? I imagine that fewer examples makes it easier to generate a good hypothesis, and batches could promote hypothesis diversity. Did you notice any interesting relationship between batch size and performance? Did you try including all examples?
- Why is the "satisfy" feedback only provided once the max iterations are met (equation 6)? Is there no early termination condition for hypotheses that satisfy the examples?
- The approach does not reach 100% on any of the metrics on any benchmarks; what are some common failures?
- Llama-3-8B performed quite poorly, as reported in Table 3. Future work is proposed to use model distillation. Have you tried more recent small models, like LLama-3.1-8B or LLama-3.2-3B?
	
	
Here are a few minor grammatical errors to fix:
- Line 421 "in Figure 3" -> "In Figure 3"
- line 422 "alignment between the H1 score" H1 should be HI
- line 514 "which constraints" -> "which constrains"

A small part of the appendix seems to have been mistakenly erased. See line 1201, which seems to start in the middle of a sentence.

### Soundness
4

### Presentation
4

### Contribution
4
