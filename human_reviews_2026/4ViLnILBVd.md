# CoBel-World: Harnessing LLM Reasoning to Build a Collaborative Belief World for Optimizing Embodied Multi-Agent Collaboration

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4

## Abstract
Effective real-world multi-agent collaboration requires not only accurate planning but also the ability to reason about collaborators' intents-a crucial capability for avoiding miscoordination and redundant communication under partial observable environments. Due to their strong planning and reasoning capabilities, large language models (LLMs)  have emerged as promising autonomous agents for collaborative task solving. 
However, existing collaboration frameworks for LLMs overlook their reasoning potential for $\textit{dynamic intent inference}$, and thus produce inconsistent plans and redundant communication, reducing collaboration efficiency. 
To bridge this gap, we propose $\textit{\textbf{CoBel-World}}$, a novel framework that equips LLM agents with a $\textit{\textbf{co}llaborative \textbf{bel}ief world}$-an internal representation jointly modeling the physical environment and collaborators' mental states. 
CoBel-World enables agents to parse open-world task knowledge into structured beliefs via a symbolic belief language, and perform zero-shot Bayesian-style belief updates through LLM reasoning. This allows agents to proactively detect potential miscoordination (e.g., conflicting plans) and communicate adaptively. Evaluated on challenging embodied benchmarks (i.e., TDW-MAT and C-WAH),  CoBel-World significantly reduces communication costs by $\textbf{22-60\\%}$ and improves task completion efficiency by $\textbf{4-28\\%}$ compared to the strongest baseline. Our results show that explicit, intent-aware belief modeling is essential for efficient and human-like collaboration in LLM-based multi-agent systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces the Collaborative Belief World framework, equips LLM agents with an internal representation that jointly models the physical environment and collaborators’ mental states. The paper introduces Symbolic Belief Representation for belief representation and Bayesian Belief Collaboration for belief update. Experiments on two embodied benchmarks show the proposed method reduces communication costs and improves task performance.

### Strengths
- The introduction of a symbolic belief representation language is novel and well-motivated.

- The experimental results are good and convincing with the newly introduced communication token cost metric.

- Detailed prompt report and code for reproduction.

### Weaknesses
- The presentation could use some improvements.
   - Better to put Symbolic Belief Language Definition in the main paper rather than the Appendix, as it's the grounding of the work. Deferring it makes the method hard to follow.
  - Several typos in the manuscript, could use another round of proofreading. E.g., l132-l133 in the contribution
- The method seems to have a strong assumption of homogeneous agents' cooperation, which does not hold for human-agent cooperation.
  - Coming up with agreed Belief Rules (building common ground) itself is a hard problem in decentralized multi-agent cooperation; current implementation seems to be assuming agents of the same architecture (the belief rule construction phase may not be followed perfectly by other agents of other methods, including humans)
  - There's no experiment with heterogeneous agents or human-agent cooperation.

### Questions
- How are the ablation experiments of CoBel-World (No SBR) implemented?

- Since the proposed method introduces additional llm call overhead, how's the efficiency tradeoff regarding llm tokens used?

- Missing discussion of recent work about Belief Modeling in multi-agent systems [1][2][3]

[1] COMBO: Compositional World Models for Embodied Multi-Agent Cooperation
[2] Neural Amortized Inference for Nested Multi-agent Reasoning
[3] Too many cooks: Coordinating multi-agent collaboration through inverse planning

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose a novel framework, CoBel-World (Collaborative Belief World), to improve decentralized LLM agents collaboration using belief modeling - specifically to produce consistent plans and reduce communication costs. 

They create a symbolic belief language to represent mental state, current plan, and collaborator intents for each agent. They also describe first-order belief (i.e. I believe agent B believes..). They perform zero-shot Bayesian belief updates with theory-of-mind reasoning based on observations and communication. Then, the belief prediction for future intents and states is done based on the prior beliefs. The communication only happens when there is a misalignment after the belief update. 

The authors use transport rates for TDW-MAT, and average steps to complete all tasks for C-WAH for efficiency calculation. For communication, they use the average number of takes generated by all agents for communication on average.

They deploy Qwen3-32B and ChatGPT-4o models as agents. They compare against traditional (MHP, RHP) as well as other LLM-based frameworks (CoELA, CaPo) as baselines on the TDW-MAT and C-WAH benchmarks. The CoBel-world framework shows significantly reduced communication costs and improved completion efficiency, especially with ChatGPT-4o. They also perform ablations by removing each component - symbolic belief representation, and the bayesian belief collaboration, and show that while both are important, bayesian belief collaboration is essential for high performance for CoBel-world. They also experiment with more agents to show that framework can scale.

### Strengths
1. They apply a useful idea from traditional multi-agent RL (belief networks/modeling) to zero-shot MLLM agents and show improved performance and costs on benchmarks.
2. They use actual embodied benchmarks instead of grid-like or text-only worlds, which is closer to the real-world.
3. Lack of task-specific fine-tuning allows the framework to potentially generalize and scale to any environment.

### Weaknesses
1. The novelty of the approach is very limited. While the approach works better than baselines, the ideas of belief modeling are not uncommon in multi-agent RL. This is just application of the idea in the LLM-agent setting.
2. The authors do not cite or compare against very easy-to-find, relevant, and perhaps similar works. A simple Google Search leads to many Belief Modeling + Belief Language + LLM ideas: https://openreview.net/pdf?id=TWC4gLoAxY, https://arxiv.org/pdf/2506.08292
3. The benchmarks are very small (24 episodes and 10 episodes) and have limited tasks, which are perhaps not enough to signal a high-performance in the real-world. The benchmarks they use are typically only good to test 2 agents in collaboration, 3 at best.
4. There are many typos in the paper, and missing letters.
	1. Line 132 - averafe -> average
	2. Line 146 - tep-by-step -> step-by-step
	3. Line 224 -> defination -> definition
	4. "Collaborative representing progress" -> do the authors mean process?
	5. Line 293 -> pontential -> potential
	6. Line 501 - Baesian -> Bayesian
	7. Line 505 - reproduct -> reproduce

### Questions
1. Can the authors clarify the differences against the recent works, and any other that incorporate belief modeling with LLMs?
2. How can we get a better signal as to whether this approach applies generally to the real-world? Have the authors considered PARTNR: https://ai.meta.com/research/publications/partnr-a-benchmark-for-planning-and-reasoning-in-embodied-multi-agent-tasks/?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces CoBel-World, a framework that augments LLM agents with an explicit "collaborative belief world." A symbolic belief language encodes task knowledge and agent mental states. At the same time, a Bayesian-style update loop, with prediction and measurement steps executed via LLM prompting, maintains and refines those beliefs during execution. On two embodied multi-agent benchmarks, compared with rule-based planners and two recent LLM collaboration baselines, CoBel-World reduces communication tokens by 22–60% and yields 4–28% better task-completion efficiency.

### Strengths
1. By explicitly distinguishing zero- and first-order beliefs, the authors provide a concrete representation that is easier to inspect than pure free text.

2. The Bayesian filter implemented through LLM prompting is a creative use of reasoning capability that decides whether to speak or act.

3. On both TDW-MAT and C-WAH, CoBel-World achieves the best transport rate / step count while cutting token usage substantially. In Figure 3 the qualitative trajectory shows how belief prediction avoids room-duplication and unnecessary chat.

### Weaknesses
1. The “Bayesian” update is performed by textual prompting rather than probabilistic computation, and the normalization and sampling policies are not well-defined.

2. Recent work on ToM reasoning and belief-driven LLM collaboration/debate is not cited or contrasted, such as [1][2]. Differences in representation and update procedure should be spelled out.

3. All results appear to be single-run numbers. No variance or statistical tests are provided. Given the inherent stochasticity of LLM outputs (temperature 0.7), this casts doubt on the reported average gains of 4%.

4. The method may trade fewer tokens between agents for more tokens issued to the LLM internally (belief prediction, misalignment detection). There is no report of total LLM calls or wall-clock time, so efficiency is not well-supported.

5. There are too many grammatical mistakes and typos that affect readability. For example, in the caption for Figure 2: "All agent" should be "All agents", "collaborative reasoning progress" should be "collaborative reasoning process", "to analysis" should be "to analyze", "structured format" should be "a structured format", "each agent construct a initial belief" should be "each agent constructs an initial belief", "each agent update" should be "each agent updates", and "adaptive collaborative decision" should be "an adaptive collaborative decision".

[1] AutoToM: Scaling Model-based Mental Inference via Automated Agent Modeling. Zhang et al, 2025.

[2] From Debate to Equilibrium: Belief-Driven Multi-Agent LLM Reasoning via Bayesian Nash Equilibrium. Yi et al, 2025.

### Questions
1. How many internal LLM calls (update + prediction + alignment checks) are issued per environment step compared to CoELA/CaPo? Please report average total tokens (internal + external) and time.

2. Can you conduct some failure analysis to understand when the framework will fail?

### Soundness
3

### Presentation
2

### Contribution
2
