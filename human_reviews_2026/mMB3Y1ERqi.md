# SimuAgent: An LLM-Based Simulink Modeling Assistant Enhanced with Reinforcement Learning

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 0, 6, 8

## Abstract
Large language models (LLMs) have revolutionized text-based code automation, but their potential in graph-oriented engineering workflows remains under-explored. We introduce SimuAgent, an LLM-powered modeling and simulation agent tailored for Simulink. SimuAgent replaces verbose XML with a concise, dictionary-style Python representation, dramatically cutting token counts, improving interpretability, and enabling fast, in-process simulation. A lightweight plan–execute architecture, trained in two stages, equips the agent with both low-level tool skills and high-level design reasoning. To tackle sparse rewards in long-horizon tasks, we propose Reflection-GRPO (ReGRPO), which augments Group Relative Policy Optimization (GRPO) with self-reflection traces that supply rich intermediate feedback, accelerating convergence and boosting robustness. Experiments on SimuBench, our newly released benchmark comprising 5300 multi-domain modeling tasks, show that a Qwen2.5-7B model fine-tuned with SimuAgent converges faster and achieves higher modeling accuracy than standard RL baselines, and even surpasses GPT-4o when evaluated with few-shot prompting on the same benchmark. Ablations confirm that the two-stage curriculum and abstract-reconstruct data augmentation further enhance generalization. SimuAgent trains and runs entirely on-premise with modest hardware, delivering a privacy-preserving, cost-effective solution for industrial model-driven engineering. SimuAgent bridges the gap between LLMs and graphical modeling environments, offering a practical solution for AI-assisted engineering design in industrial settings.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work addresses the challenge of applying LLMs to Simulink code generation, which differs substantially from traditional coding tasks. Simulink adopts a hierarchical, graphical paradigm with complex block diagrams, signal routing, and strict topological constraints. These characteristics impose significant challenges for LLMs: requiring them to respect rigid graph structures, handle very long contexts, and suppress hallucinations to produce reliable and interpretable code.
The authors propose an RL-based approach to mitigate these challenges. In particular, ReGRPO leverages tool-invocation feedback to guide model learning, effectively improving code quality through iterative reinforcement. The proposed framework demonstrates competitive performance across multiple experiments.

### Strengths
1. The model adopts a two-stage curriculum learning strategy to handle complex tasks, fostering higher-order capabilities such as planning, abstraction, and modular composition.
2. The Abstract–Reconstruct mechanism alleviates data scarcity while ensuring the structural integrity and accuracy of the generated outputs.
3. The ReGRPO component enhances model performance through tool-based reflection and reinforcement, enabling more consistent reasoning.

### Weaknesses
1. While the experimental results are promising, the method appears limited in scalability. The Abstract–Reconstruct loop does not introduce new reward signals, meaning that improvements still rely heavily on the model’s inherent abilities. The authors require to include ablation studies showing how performance varies with different data scales.
2. The ReGRPO mechanism may be susceptible to reward hacking. Without proper supervision of the reflection phase, the model could exploit shortcuts, e.g., performing unnecessary or repetitive reflections to maximize reward. The paper would benefit from a more detailed discussion or empirical analysis of this issue.

### Questions
See Weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
The paper presents SimuAgent, an LLM-based agent to automate and assist with Simulink modeling and simulation tasks. The claimed contribution is a Python-dictionary representation for Simulink models, although that's nowhere to be found in the core text. Supposedly, it improves interpretability of Simulink models.

### Strengths
- Interesting problem, certainly high industry impact

### Weaknesses
- Basically zero scientific novelty. This is an engineering project without many generalizable takeaways.
- Presentation is inconsistent and unclear what the actual contribution is: toolbox, method, architecture, benchmark... All of these are claimed in the paper, but unclear which one is it. For some reason, it is claimed that a "Python-based model representation," which is a dictionary, is a contribution. Certainly not for a top conference. It supposedly improves interpretability. This obviously cannot be true as visual modeling languages, such as Simulink's causal-block diagrams, have superior interpretability for humans; and serialization into a JSON file achieves the same affect as serializing into whatever other file format as there's no added semantic information.
- No real evaluation. Without evaluating this on various classes of Simulink models, the utility of the approach remains questionable.

### Questions
No questions

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents SimuAgent, an LLM-based agent for constructing, modifying, and querying Simulink models. The core contributions are (1) a compact, Python-dictionary representation of Simulink models that reduces token usage and enables fast in-process validation and debugging, (2) a two-stage staged training curriculum (execution --> planning) augmented with a self-supervised Abstract–Reconstruct data augmentation loop, and (3) Reflection-GRPO (ReGRPO) i.e., an extension of Group Relative Policy Optimization that injects automatic self-reflection traces to provide intermediate textual feedback for sparse-reward, long-horizon tasks. The authors also release SimuBench, a 5,300-task, multi-domain benchmark for LLM-based modeling (creation, editing, QA). Experiments with Qwen-2.5-7B show that SimuAgent (Stage1+Stage2 + ReGRPO) converges faster and attains the best overall accuracy on SimuBench (51.89% average), narrowly outperforming a GPT-4o XML/image baseline (50.45%). Ablations analyze the contribution of ReGRPO, curriculum stages, augmentation, group sizes, and reflection schedules; failure analysis pinpoints typical error modes (topology, block selection, parameter omission, premature termination, context limits).

### Strengths
- The integration of Reflection-GRPO with Simulink tool feedback is a notable contribution. The agent leverages intermediate reflection traces and programmatic validation signals (e.g., structural checks, execution feedback, block-level errors) to guide long-horizon updates. This mechanism improves sample efficiency, stabilizes training under sparse rewards, and provides a general recipe for scaling RLHF-style methods to complex tool-using domains beyond text-only reasoning.

- The Python-dictionary representation, in-process validation testbed, and tool integration directly tackle the large number of tokens, slow MATLAB engine loops, and debugging friction. These are crucial for deployment in model-driven engineering and show a practical system design that addresses real engineering issues in designing such automation.

- The authors provide many controlled ablations (stage curriculum, reflection schedules, group sizes, reward shaping, LoRA, model scale) and a failure taxonomy that identifies limitations.

- The paper provides 5.3k multi-domain tasks (models + schematics + QA), filling a benchmarking gap for graphical model automation and enabling reproducible comparisons.

### Weaknesses
- The Introduction section is very well-written and effectively motivates the need for an automation agent for Simulink. However, the proposed method and experimental sections lack critical implementation details and could be substantially improved through better organization. For instance, in the architecture description (Section 3), it would be far more informative if the pipeline stages were presented sequentially, explaining the order of operations and data flow, rather than only listing the tool’s individual features.

- The tool processes a natural language description to create, modify, or query Simulink models. It does so by first prompting an LLM to produce a step-by-step plan, which is then translated into a Python dictionary representation. These dictionaries are subsequently converted into executable Simulink commands (e.g., adding blocks, setting parameters). However, it remains unclear how the semantic fidelity of the plan to the original NL description is ensured during inference. While the Python-based executor can validate syntax, it does not guarantee semantic alignment or correctness of the generated plan.

- For the results in Table 2, particularly for the modification task, there is an inconsistency in input modalities. Competing SoTA models (e.g., GPT-4o) receive an NL prompt along with an image or XML input, whereas SimuAgent operates on a Python dictionary representation. For a fair comparison, all models should be evaluated under identical input formats and testing conditions, or at least the differences should be clearly justified and analyzed.

### Questions
See weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper presents SimuAgent for LLM based agent for modeling in Simulink. Key contributions are (1) using the python dictionary structure rather than XML or other token heavy schemes, (2) a two-stage curriculum with a a selfsupervised Abstract–Reconstruct loop and (3) an algorithm ReGRPO that deals with the sparse reward nature of the long horizon problem using self-generated textual reflection traces. It also intruces SimuBench, a large-scale dataset of tasks and show that a Qwen2.5-7B model trained with their pipeline outperforms other baselines (GRPO and GPT-4o). The paper also shows ablation studies to show effect of the two-stage training, reflection and VAE style abstract-reconstruct augmentation.

### Strengths
- The paper effectively frames the problem, shows how previous methods (XML) lead to a large number of tokens, and showcases Python-dictionary representation as a suitable choice
- Reflection and retry is a simple mechanism to tackle the sparse reward issue of just having the output of 0/1 at the end of the episode.
- The SimuBench dataset provides examples over various system-design domains. 
- The paper is well written, has done extensive experiments, with multiple ablations and transfer to other similar platforms, solidifying the contribution. The figures and plots add to understanding.

### Weaknesses
- The algorithm is only compared with GRPO. How does the method compare to other baselines for LLM tool-use and RL?
- Improvements on generic NLP benchmarks are small, code-based tasks show more gain, but SimuBench is the setting where reflection is most helpful.
- More methodological clarifications on reward structure, prompt differences for image-based inputs are needed.

### Questions
1. How are the different terms in the reward structure weighted?
2. What minimal hardware is needed for inference/deployment? The manuscript only describes training GPUs and claims laptop-grade GPUs.
3. How can we compare the multi-modal inputs/prompts used for the GPT-4o with the other models?
4. Why does setting the algorithm to Always reflect hurt performance?

### Soundness
3

### Presentation
3

### Contribution
3
