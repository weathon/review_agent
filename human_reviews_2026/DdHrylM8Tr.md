# AgentAlign: Navigating Safety Alignment in the Shift from Informative to Agentic Large Language Models

- Decision: Reject
- Scores: 4, 6, 2

## Abstract
The emergence of agentic capabilities in large language models fundamentally transforms their risk profile from passive information providers to autonomous action executors, introducing unprecedented safety challenges that existing alignment methods fail to address. Current approaches lack systematic frameworks for understanding and modeling the behavioral patterns underlying malicious agentic
activities, leading to brittle safety measures that collapse when confronted with multi-step harmful requests. We introduce AgentAlign, a novel behavioral modeling framework for agentic alignment that systematically captures malicious activity patterns through abstract behavior chains – structured representations of action sequences that characterize how harmful objectives are pursued across diverse tool-use scenarios. By instantiating these behavioral abstractions within comprehensive simulated environments, our framework enables scalable generation of authentic, executable training scenarios that preserve complex multi-step dynamics while avoiding real-world risks. Extensive evaluation across three model families demonstrates substantial safety improvements (35.8% to 79.5% enhancement in refusal rates) while maintaining or improving utility on benign tasks, significantly outperforming existing prompting-based defenses.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work introduces AgentAlign, a new training framework to improve agent safety in which the attackers instruct the agents to execute malicious tasks. They show positive results where AgentAlign outperformed basic prompting defense baselines such as CoT, ReAct, and refusal prompts.

### Strengths
- Solid synthetic data generation pipeline with human validations.
- Great diagrams and plots that help explain things clearly.
- Showed great performance when compared to prompting baselines.

### Weaknesses
- Only compared to weak prompting baselines. For example, I'd appreciate it if you could at least add one common guardrail baseline, such as Llama Guard 4? This is a common defense method and will be helpful to include to see whether it complements AgentAlign or it's already very effective enough on the benchmarks evaluated. 
- No adversarial pressure studied. In reality, attackers will not give up after one try, and will likely apply the existing, common jailbreaking approach to the original harmful instructions. It'd be great to see a section that studies how robust this training approach is to for example, automated red teaming, multi-turn or decomposition attack, prefill attack, etc.
- There is no limitation section. This work studies the single-turn attack for the agents, but it's unclear whether this works for multi-turn attack, etc. some papers you can cite in multi-turn or decomposition attacks include: (1) Monitoring Decomposition Attacks in LLMs with
Lightweight Sequential Monitors and (2) Breach By A Thousand Leaks: Unsafe Information Leakage in `Safe' AI Responses
- No error bars were provided in any of the stats reported.

### Questions
Questions are basically the weakness i mentioned:
- Could you include llama guard 4 as a baseline?
- Could you test this approach on a few common jailbreaking techniques mentioned in Weakness 2?
- Could you add a limitation section?
- Could you at least report the standard errors for table 2? because the baseline and AgentAlign are quite close.

Happy to raise my score if all of these are addressed, and AgentAlign still outperforms the new baseline and still holds the positive results under the adversarial pressure.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes AgentAlign, a framework for improving the safety alignment of agentic large language models (LLMs) that can execute multi-step actions and use external tools. The key idea is to represent potential harmful or benign behaviors as Abstract Behavior Chains. The authors construct a simulation environment with 86 functional APIs across 9 tool categories to safely generate and validate a large-scale dataset (~18K samples) of harmful and benign agentic instructions. AgentAlign demonstrates significant improvements on the AgentHarm and ToolSword benchmarks. Specifically, the method substantially increases the refusal rate on harmful tasks while maintaining utility on benign tasks.

### Strengths
* Clear motivation and presentation: The paper provides a well-motivated discussion of the emerging safety challenges in agentic LLMs, supported by concrete examples and quantitative evidence. The writing is clear and easy to follow.
* Originality: The idea of modeling safety through Abstract Behavior Chains is novel and insightful, as it captures multi-step harmful behaviors at the behavioral logic level rather than relying on surface text filters.
* The proposed simulation environment and accompanying dataset are strong contributions, enabling safe and systematic synthesis of agentic tasks for alignment training. The semantic and execution validation framework is particularly rigorous and enhances data reliability.
* The experiments are comprehensive, covering multiple open-source models and benchmarks (AgentHarm, ToolSword).

### Weaknesses
* While the paper is strong overall, it would benefit from a more comprehensive discussion of related work on plug-and-use safety guardrails for agents, such as GuardAgent, Conseca, and Agrail, to better position AgentAlign within this growing research space.
* The training setup is not clearly described in the main text; readers may find it difficult to understand how the proposed dataset and objectives are applied during fine-tuning. Including a concise summary of the training process (currently only in the appendix) would significantly improve clarity.
* Similarly, the simulation environment is central to the paper’s contribution, but its implementation details and accessibility are limited. Open-sourcing or providing more technical documentation on the environment and dataset would enhance reproducibility and impact.
* On the empirical side, there is a slight drop in benign task performance for some models after applying AgentAlign, and results on Ministral and Qwen remain below Claude-3.5-Haiku. Moreover, it would strengthen the work to include comparisons against other representative guardrail systems such as Llama-Guard 3.

### Questions
Most of my questions overlap with the weaknesses mentioned above. In addition, I have one question regarding the data generation process:
* Could the authors clarify why Claude-3.5-Haiku was chosen to generate refusal responses for harmful instructions, while Mistral-Large was used to generate benign trajectories?

### Soundness
2

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The authors introduce AgentAlign, a framework that models malicious agent behaviors through "abstract behavior chains" - structured representations of multi-step harmful action sequences. These chains are instantiated in simulated environments to generate training data that balances safety and utility. Experiments across three model families show substantial improvements while maintaining performance on benign tasks.

### Strengths
1. Relevant problem: The safety gap between conversational and agentic LLMs is a real concern worth investigating.
2. Systematic data generation: The abstract behavior chain framework provides a structured approach to generating multi-step harmful scenarios, which is important in AI safety research.

### Weaknesses
1. The method is missing critical comparisons. This is fundamentally a data generation method, yet there are no comparisons to: 
- Existing safety datasets: GuardSet-X [1], ToolAlign (only briefly mentioned in the related work), and other multi-step safety datasets. How does training on your dataset compare to training on these?
- Guardrail systems: Why not compare against ShieldAgent [2], LlamaGuard, or other input filtering approaches? These operate at inference time without requiring model retraining. The paper doesn't justify why fine-tuning is necessary when you could simply filter inputs with an existing safety classifier.
- Other data generation approaches: What about simple augmentation of existing red-teaming datasets? Or using LLMs to generate harmful agent scenarios with different prompting strategies?

2. The evaluated models (GPT-4o, Qwen-2.5) are already outdated. More recent models should be evaluated.

3. Transferability issue not addressed. If the agent is equipped with new sets of tools (or APIs), will the model still show a good refusal rate?

[1] Kang, Mintong, et al. "Guardset-x: Massive multi-domain safety policy-grounded guardrail dataset." arXiv preprint arXiv:2506.19054 (2025).
[2]

### Questions
1. Why not compare to LlamaGuard or ShieldAgent as input filters? This seems like the most obvious baseline.
2. How does performance compare when training on other existing safety datasets? 
3. What happens with completely different tool ecosystems? If I deploy your trained model with an entirely new set of APIs, does the safety transfer?
4. Can you show this works on current frontier models? The models tested are outdated.

### Soundness
2

### Presentation
3

### Contribution
2
