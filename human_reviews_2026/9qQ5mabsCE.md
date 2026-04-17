# EmboMatrix: A Scalable Training-Ground for Embodied Decision-Making

- Decision: Reject
- Scores: 4, 2, 6, 6

## Abstract
Embodied decision-making enables agents to translate high-level goals into executable actions through continuous interactions within the physical world, forming a cornerstone of general-purpose embodied intelligence. Large language models (LLMs), with their general decision-making capabilities, offer a promising path to realize this potential; however, LLMs trained solely on language lack exposure to physical environments, limiting their true embodied understanding. To bridge this gap, we propose the concept of a \textbf{training ground}: a comprehensive infrastructure that provides task and scene simulation, embodied interaction, and feedback signals, offering a one-stop solution for LLM acquire genuine embodied decision-making skills. In this work, we present EmboMatrix, the first training ground of its kind, providing massive and diverse tasks with efficient simulation and precise rewards. EmboMatrix incorporates a series of novel techniques: a multi-agent data engine for large-scale task and scene generation, a distributed heterogeneous-hardware system for scalable simulation, and a multi-level reward architecture for precise supervision. Leveraging EmboMatrix, we cultivate \textbf{EmboBrain}, an LLM whose embodied decision-making abilities emerge from extensive embodied interactions. Experiments show that EmboBrain-7B surpasses the 671B DeepSeek-R1 baseline by 9.5\% on two challenging embodied decision-making benchmarks, demonstrating the power of interactive, environment-grounded learning for building truly intelligent embodied agents. The code will be released upon the paper's acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
To bridge the gap between Large Language Models' (LLMs) reasoning abilities and the physical understanding required for embodied decision-making, this paper introduces EmboMatrix, a scalable "training ground" for interactive, simulation-based learning. EmboMatrix features three key innovations: a multi-agent data factory for generating massive and diverse tasks, a distributed heterogeneous-hardware system for scalable simulation, and a multi-level reward architecture for precise supervision. By training an LLM within this framework to create "EmboBrain," the resulting 7B model significantly outperformed much larger baselines, including the 671B DeepSeek-R1, on two challenging embodied decision-making benchmarks.

### Strengths
+ This work introduces EmboMatrix, the first "training ground" framework specifically proposed to enhance the embodied decision-making capabilities of LLMs through interactive learning.
+ The framework's effectiveness is clearly demonstrated by significant performance improvements; models trained within EmboMatrix (EmboBrain) substantially outperform much larger models, including the 671B DeepSeek-R1, on challenging benchmarks.
+ The system's architecturally decoupled design, which separates the LLM trainer from a distributed pool of heterogeneous simulation workers, enables high-throughput parallel rollouts and significantly improves the efficiency of the entire training loop.
+ The paper effectively uses an ablation study and reward curves to prove the necessity of its hierarchical reward design, showing that the 'semantic relevance reward' ($r_r$) is critical for guiding the agent and enabling efficient, stable learning.

### Weaknesses
+ The pre-cached language-physics interface, which substitutes full physical simulation with pre-computed post-conditions, sacrifices dynamic fidelity and generalization for speed. This approximation may lead to policies that lack robustness in the real world, as it omits crucial details like contact dynamics, sequential dependencies, and realistic failure modes. The paper does not provide a systematic error analysis of this approximation or address potential inconsistencies from pre-rendering.
+ The model, though the main focus is built on top of LLMs, with its inputs consisting of simulator-generated textual descriptions of the scene, not raw sensory data like visual perception, which bypasses the critical end-to-end challenge of perception and action.
+ Key details regarding the pre-caching system are not disclosed, including the storage overhead of the pre-computed outcomes, the runtime cache hit rate, and the fallback mechanism used when an interaction is not in the cache (e.g., whether it reverts to a full, costly physical simulation).
+ The statistical rigor of the evaluation is limited. The main results table only reports the average of 10 samples per task, without reporting variance, confidence intervals, or results from multiple independent random seeds, and lacks statistical significance testing.
+ The paper's formatting appears to deviate from the official ICLR 2026 template, seemingly using `\usepackage{geometry}` to alter the page layout.
+ There are minor formatting and typographical errors. For example, the caption in Figure 5 overlaps with text in the image, and some figure captions (Figures 2, 3) lack sufficient detail. There are also typos, such as "A detailed comparation is shown in Tab 4" (line 126) instead of "comparison", and "EmboMatrix improve its performance by" (line 337).

### Questions
1. How does the pre-caching approximation, which substitutes physics with post-conditions, affect the policy's robustness to real-world dynamics and unscripted failure modes?

2. How does the multi-agent data factory ensure genuine task diversity, and isn't this diversity ultimately bottlenecked by the limited set of objects and 45 base scenes from BEHAVIOR-1K?

3. What are the technical specifications of the pre-caching system, such as its storage overhead, typical runtime cache hit rate, and the fallback mechanism used for a cache miss?

4. To improve statistical rigor, can the authors provide variance, confidence intervals, or results run with multiple independent random seeds for the main benchmark results in Table 1?

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduce the EmboMatrix infrastructure and "training ground" for high-level embodied action sequencing using large language models (LLMs). EmboMatrix combines a multi-agent data factory, a distributed simulation backend , and a hierarchical reward architecture. The framework is used to train EmboBrain, an LLM adapted via reinforcement learning within simulated embodied environments.

### Strengths
1. The idea of scaling embodied environments to create a data engine for LLM training is interesting and somewhat novel. If executed well, can make a great resource for training embodied planners.

### Weaknesses
1. Authors claim strong performance gains on the EAI benchmark in abstract L26. However, experimental results on embodied agent interface that the authors report (Overall, Pick and Place, Appliances Using, Kitchen Operation, Compound Task) are not the standard axis of evaluation for EAI (Goal Interpretation, Subgoal Decomposition, Action Sequencing, Transition Modeling). The authors fail to clarify exactly what subset of tasks and what number of examples are used for each track, and the "Overall" score is highly misleading. It seems that the paper is only working on the Action Sequencing subtrack, which should be properly stated in obvious locations.

2. If the trained model is only capable of action sequence generation and the training envionment is only capable of generation action sequence prediction task data, then it is an overstatement to claim that the environment and model is for embodied decision making.

3. There's not enough comparison of EmboMatrix to other existing benchmarks and data generation methods, also there isn't sufficient analysis / experiments with "Our Agent-Generated Benchmark" to demonstrate it being a reliable benchmark.

### Questions
1. Typos: we present a comprehensive experiments (L264)
2. Additional citations: RoboVerse, Embodied World Models, etc.
3. How many tasks are there in each of the EAI eval subtracks (Pick and Place, Appliances Using, Kitchen Operation, Compound Task)?

### Soundness
1

### Presentation
2

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
This paper proposes EmboMatrix, a training ground for embodied decision making. The system has three key components: a multi-agent data engine for scene synthesis, a decoupled simulator with pre-cached language-to-physics mappings, and a hierarchical reward curriculum. The curriculum progresses from format correctness to semantic relevance and goal completion. Models trained in EmboMatrix acquire robust planning abilities through interactive learning. The resulting system, EmboBrain, substantially outperforms competitive baselines.

### Strengths
1. The paper is well-written, with clear figures and detailed explanations that make the work easy to follow.
2. The paper tackles a central gap in embodied AI by proposing a unified training ground that connects language-only LLMs to physical, interactive decision-making.
3. The system is well-engineered, combining multi-agent task generation, a scalable decoupled simulator with a pre-cached language–physics interface, and a hierarchical reward curriculum.
4. The evaluations are compelling and scale-efficient. A smaller EmboBrain model consistently surpasses a much larger DeepSeek-R1 across multiple embodied decision-making benchmarks and task categories, underscoring the practical impact of the unified training ground.

### Weaknesses
1. The method relies on a fixed, predefined skill library; the high-level policy can only select from these primitives, so generalization is limited by their coverage and granularity, and the set is not learned or expanded online.

2. The evaluation relies on GPT-4 to score task diversity and scene aesthetics, which risks creating a self-referential loop. When an LLM-trained system is judged by another LLM, results may be biased toward the evaluator's preferences.

3. The paper has limited environmental coverage. The training corpus and evaluation are built on 45 Behavior-based scenes, leaving cross-domain generalization and robustness to distribution shift unclear.

### Questions
1. The experiments focus exclusively on the DeepSeek-distilled Qwen series. Including results from a different model family would help demonstrate the generalization more broadly.

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
4

### Summary
This paper proposes a 'training ground' for LLM-based embodied agents. This environment, referred to as 'EmboMatrix', provides an end-to-end framework that encompasses scene generation, simulation for model rollouts, and reward signals for both training and evaluation. The authors introduce various techniques to optimize the simulation and rollout processes.  LLM models were subsequently trained using the EmboMatrix. Experimental results indicate that the resulting model outperformed both the base models and general-purpose, large-parameter models when evaluated on scenes generated within the EmboMatrix.

### Strengths
1. The reviewer appreciates the framework's ability to enable high-throughput rollouts, noting that rollout speed is often a bottleneck in embodied AI RL training. The proposed components—specifically the pre-cached language-physics interface, resource scheduler, and task dispatcher—appear to work well. The ablation study also shows they improve simulation speed, which the reviewer agrees is critical for RL practitioners.

2. This paper is well-structured and very easy to read. While some further clarification is needed, most of the technical details are clearly presented. 

3. The experimental results look promising, the EmboBrain models outperform baseline models by and large general purpose LLMs.

### Weaknesses
1. The current documentation for the multi-agent-driven automated data factory lacks essential details regarding the intricate interactions between its various components. Specifically, the reviewer requires a more comprehensive explanation of how these multiple agents communicate, coordinate, and influence the generation of instructions and subsequent scene constructions. Without this crucial clarification, the underlying mechanisms and overall impact on the data factory's output remain ambiguous and difficult to assess.

2. Confidence intervals are not included in the results. Given the potentially high variance of RL methods, the absence of confidence intervals makes it challenging to accurately assess and compare the different approaches.

### Questions
1. In the pre-cached language-physics interface, physically plausible outcomes from a pre-computed set are used instead of running a full simulation. This raises two critical questions: first, what is the accuracy of this approximation, and second, is this technique employed solely during training or also during evaluation?

2. In the context of EmboMatrix rollout and training, which aspect consumes the most time: the environment step, model inference, or gradient update? Please provide a detailed explanation.

3. Regarding the parallel rollouts, are the model updates performed synchronously or asynchronously? If the updates are synchronous, could you please comment on the associated synchronization overhead? Conversely, if the updates are asynchronous, what impact does this have on training performance?

4. Regarding the semantic reward, could you elaborate on the design rationale for basing it on the intersection of needed objects and objects the agent has interacted with? How does this reward structure generalize to tasks that are not object-centric? Additionally, did you experiment with alternative reward formulations? 

5. Given that intermediate rewards are often prone to reward hacking, did you observe any such unintended behaviors or policy exploitation stemming from this specific design?

### Soundness
3

### Presentation
3

### Contribution
3
