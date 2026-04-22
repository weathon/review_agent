# CortexVLA: Bridging the Gap between Cognition and Action via Function Calling

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 8, 4, 2

## Abstract
Vision-Language-Action (VLA) models have shown promise for embodied intelligence, but they often struggle with long-horizon tasks due to error accumulation or planning failure. To address these challenges, we propose CortexVLA, a novel paradigm that bridges cognition and action by leveraging large language model (LLM) function calling. CortexVLA consists of three modular components: the Central Cortex, an LLM-based cognitive hub for planning and function calling; the Visual Cortex, which provides perception through callable vision tools; and the Motor Cortex, which exposes robotic action control as functions. To improve robustness and enable recovery from execution errors, we further propose Cortex-PPO, a reinforcement learning (RL) algorithm that trains CortexVLA to make optimal function calls while supporting failure recovery. We provide theoretical analyses to further demonstrate the soundness and generalization abilities of Cortex-PPO. Comprehensive experiments demonstrate the effectiveness of CortexVLA on ultra-long-horizon tasks. In our main experiment, CortexVLA achieves an average success rate of 85.40\%. More importantly, it sustains a 72.73\% success rate with an average sub-task length of 11.55 when tackling the most challenging 14 sub-tasks, whereas end-to-end VLA baselines fail beyond 3 or 4 steps. In a flexible manufacturing scenario with 31 sub-tasks, CortexVLA achieves an 81.25\% success rate with an average sub-task length of 26.69, demonstrating strong scalability and adaptability. Codes will be released after publication.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes CortexVLA, a modular framework designed to bridge cognition and action in robotic manipulation via LLM-based function calling. It consists of three components—the Central, Visual, and Motor “Cortices”—that handle planning, perception, and control respectively. The authors further introduce Cortex-PPO, a reinforcement learning algorithm that improves function-call robustness and enables failure recovery. The paper provides theoretical analyses of the algorithm’s unbiasedness and generalization, and reports strong experimental results on ultra-long-horizon and multi-scenario tasks, showing higher success rates and scalability compared to existing VLA and hierarchical baselines

### Strengths
- System integration and clarity: The paper presents a clean, modular architecture that unifies planning, perception, and control using LLM function calling. The system diagram and explanations are clear and easy to follow.
- Reproducibility: The methodology and baselines are well documented, and the authors commit to releasing code.
- Empirical performance: Extensive experiments across long-horizon tasks demonstrate consistent improvement over end-to-end and hierarchical VLA baselines. The evaluation is thorough and includes multi-scenario case studies (e.g., manufacturing, bartender tasks).

### Weaknesses
- Conceptual originality: While the system is presented under the “VLA” terminology, CortexVLA is more accurately a system-level orchestration framework that leverages existing LLM tool-calling capabilities, rather than a fundamentally new vision-language-action learning model. Its contribution lies primarily in engineering design and modular integration, rather than new algorithmic or scientific insights.
- Algorithmic contribution: Cortex-PPO appears to be a modest extension of standard PPO with reward shaping and noise injection. Theoretical analysis is mathematically sound but adds limited novelty or practical insight beyond existing RL formulations.
- Scope of evaluation: The experiments convincingly demonstrate performance, but it remains unclear how much of the gain stems from better task decomposition and function orchestration versus true advances in policy learning. The framework’s dependence on predefined tools may limit generalization in open-world settings.
- Positioning: The framing of CortexVLA as a “VLA model” might be misleading, as it does not train a unified vision-language-action policy. The work would be more accurately characterized as a tool-based orchestration framework for robot control.

### Questions
- How critical is Cortex-PPO to the reported performance improvements? A clear ablation isolating its contribution would strengthen the claim.
- How scalable is the system to tasks requiring unseen tools or novel function compositions?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes CortexVLA, a modular vision-language-action framework that bridges cognition and action via LLM-based function calling across three components: a Central Cortex for planning and orchestration, a Visual Cortex for perception, and a Motor Cortex for control. To enhance robustness on long-horizon tasks, the authors introduce Cortex-PPO, a recovery-aware PPO variant with noise injection, and provide theoretical analyses proving unbiasedness of the learning signal and an information-theoretic generalization bound. Experiments on ultra-long-horizon manipulation show strong performance and scalability, with an average success rate of 85.40% and sustained success on sequences up to 14 sub-tasks, outperforming end-to-end and hierarchical baselines. Case studies in flexible manufacturing (31 sub-tasks) and a bartender scenario further demonstrate adaptability, tool modularity, and failure recovery capabilities.

### Strengths
- Originality: The paper presents a distinctive modular VLA paradigm that leverages LLM function calling to explicitly bridge cognition (planning/state tracking) and action (perception/control) via the Central/Visual/Motor Cortices and a unified tool library. This design, coupled with failure-aware orchestration and persistent task-state handling, offers a novel alternative to end-to-end or purely hierarchical VLAs for long-horizon manipulation.

- Quality: The technical contribution of Cortex-PPO is well-motivated and supported by theory and practice. The empirical evaluation is thorough, spanning ultra-long-horizon benchmarks, ablations, and scenario adaptations (manufacturing and bartending), with clear performance gains over strong baselines.

- Clarity and Significance: The framework is clearly described with a concrete architecture, tool interfaces, and prompting templates, aiding reproducibility and deployment.

### Weaknesses
- The Central Cortex requires task-dependent prompts.
- End-to-end RL does not fine-tune the Visual Cortex. Gradients propagate only to the Motor Cortex and Central Cortex.

### Questions
- What are the concrete, real-world execution protocols for Cortex-PPO (e.g., safety gating, on-robot data collection pipelines, reset strategies, and hardware wear management)?
- How many environment interactions (samples) are required for real-world RL with Cortex-PPO?
- What are the components of Cortex-PPO’s reward function, and how are they evaluated in real-world settings?
- How can we systematically generate function-call training data for supervised fine-tuning (SFT)?
- For the evaluated tasks, how is the Motor Cortex fine-tuned?

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
The paper introduces CortexVLA, a modular Vision¨CLanguage¨CAction (VLA) architecture that leverages LLM-based function calling to bridge high-level cognition and low-level control. The system separates planning (Central Cortex), perception (Visual Cortex), and action execution (Motor Cortex), with a dedicated task-handler for maintaining task state and managing bounded prompts. To enhance robustness and recovery from execution errors, the authors further propose Cortex-PPO, a PPO-based finetuning algorithm that incorporates a recovery-aware bounded reward (Cortex-Reward) and additive Gaussian noise. Theoretical analysis is provided to show that the noise-injected reward remains an unbiased estimator for policy gradients. Empirically, CortexVLA achieves higher success rates on ultra-long-horizon manipulation tasks and a flexible manufacturing scenario compared with several end-to-end and planning-based baselines.

### Strengths
1. The modular VLA architecture is well-motivated and clearly described, with distinct components for perception, planning, and control, as well as a task-handler for state tracking. Long-horizon error accumulation is a concrete challenge for current VLAs; decoupling cognition, perception, and action via callable tools is a natural and promising direction to train in end-to-end way.
2. The introduction of Cortex-PPO addresses a practical weakness of supervised fine-tuned VLAs: their inability to recover after partial execution errors. The recovery-aware bounded reward and the theoretical justification for the noise injection strategy are well presented and contribute novel insights to improving stability and generalization.

### Weaknesses
1. The main experiments rely almost exclusively on repetitive pick-and-place tasks, differing only by object identity or order. Even the so-called ¡°ultra-long-horizon¡± tasks (e.g., those with 14 subtasks) are essentially the same operation repeated multiple times. This design inflates the horizon length without increasing task diversity or reasoning complexity. To convincingly demonstrate the benefits of CortexVLA, the paper should include a broader set of tasks that require different skills, reasoning steps, or compositional dependencies.

Therefore, in this kind of repetitive tasks, it is no surprise that end-to-end methods will fail in the long-horizon setting. Moreover, the hierarchical baseline is not described in enough detail to understand why it fails; 
With a well-tuned hierarchical method, these tasks should, in principle, be solvable at least in the planning stage. As such, the current experiments are not strong enough to substantiate the claimed advantages of CortexVLA, and the single task family limits the generality of the conclusions.


2. For the Case Studies part, the paper propose two kinds of scenarios: flexible manufacturing and cocktail making. The results show that CortexVLA performs well, but there is no comparison with any baseline methods. It is hard to judge how much improvement CortexVLA brings without any baselines. Including even a simplified baseline (e.g., an end-to-end VLA or hierarchical planner) would make these results more credible.

### Questions
The most significant questions are already listed under weaknesses. In addition, I would be interested to get a clarification on the following point:
+ Line 1099: The paper claim that CortexVLA substantially outperforms all baseline methods as shown in Table 3. But Table 3 does not appear to include any baseline results.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposed to address several challenges arising from the long-horizon tasks in robotic tasks, via modularized model from a three-cortex architecture: Central Cortex (LLM-based cognition hub), Visual Cortex (perception toolset), and Motor Cortex (action controller). The Central Cortex is designed with a planner and a task handler for persistent sub-task state; the visual cortex is designed with callable visual models like grounding DINo, SAM + D3RoMa for visual information; and the motor cortex designed with pose predictor or motion generators to generate motions. These 3 cortices are linked via function calls and trained with a novel Cortex-PPO reinforcement learning algorithm, with a recovery-aware reward and additive noise injection. CortexVLA reports 85.40% average success rate and 72.73% success at 14 steps with avg. sub-task length 11.55, outperforming end-to-end and hierarchical baselines.

### Strengths
* Architectural clarity: there is a clean separation of cognition (planning + allocation), perception, and control, with a unified tool library and function descriptions that encode recovery strategies. This directly targets error propagation and context bloat in long sequences.
* Design of recovery-aware RL: the authors propose Cortex-reward thay can stabilize training, encourage retries, and mitigate reward sparsity, with a theoretically clean unbiasedness argument for GAE-PPO under additive noise and an information-theoretic rationale for better cross-environment generalization, providing a solid theoretical bound for the method.
* Long-horizon capability: task handler explicitly tracks sub-task progress to prevent loss in the long context; the results are also promising.

### Weaknesses
* Computational cost could be high: as the system consists of several external tools, like for the visual cortex, it uses GroundingDINO and other pretrained vision models, which could cause a high computational burden of the whole method, also cause high latency for function-calling and execution of the external tools. Seems there is no timing, token budget, or cycle-time analysis; this matters for real-time manipulation and scalability to denser action frequencies.
* Predefined tool schemas make it less robust and adaptive: I notice that the system is dependent on well-written tool descriptions and explicit recovery strategies in function specs. This might cause the method to be prompt-sensitive and brittle across new tool APIs without careful prompting.
* Evaluation metrics are insufficient & lack of ablation: the authors claimed their method is capable of recovering from the errors; however, there is no corresponding evaluation metric designed or used to evaluate this capability, or qualitative results to showcase the effectiveness, instead, there is a heavy emphasis on success rate and avg. success length.

### Questions
* Can the authors provide more experimental results of the method on other commonly used evaluation benchmarks? Or using more metrics like per-subtask accuracy, number of recoveries invoked, action efficiency (steps/time), and compounding error curves by category would be helpful as well. These could provide more context and insights into your method's capability.
* If there are new tool schemas to be added to the framework, how can this be done?
* How does the model's generalization capability improve by introducing the modular components? Are they becoming more robust?

### Soundness
3

### Presentation
2

### Contribution
2
