# SkillEvo: An Experience Learning Framework with  Reinforcement Learning for Skill Evolution

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 2, 6, 6

## Abstract
Large Language Models (LLMs) have evolved into agents capable of perception, reasoning, and acting in open environments. Yet, in long-horizon tasks with sparse rewards, existing methods are often inefficient. Group-based reinforcement learning (e.g., GRPO) provides critic-free and stable optimization, but its coarse credit signals cannot distinguish high-quality trajectories from those that merely succeed but contain redundant or invalid actions, leading to weak generalization. We propose SkillEvo(Skill Evolution), a two-stage framework for efficient and sustainable agent learning. In the first stage, WebGRPO integrates a Reasoning and Execution Reward Model (RXERM) to deliver fine-grained feedback, and employs a dual-uncertainty filtering strategy to select informative tasks, improving sample efficiency and stability. In the second stage, SkillGenesis transforms trajectories into reusable skills, organized in a dynamically evolving Skill Path Graph (SPG). This enables skill composition, reuse, and the emergence of composite skills for long-term adaptability. On WebArena-Lite, SkillEvo raises the success rate of Llama-3.1-8B from 4.8% to 60.4% and GLM-4-9B from 6.1% to 57.6%, achieving new state-of-the-art results. These findings highlight that effective long-horizon learning requires not only refined credit signals but also systematic mechanisms for skill evolution.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the limitation of coarse credit signals, which fail to distinguish high-quality trajectories from those that merely succeed but contain redundant or invalid actions in web environments. To mitigate this issue, the authors propose SkillEvo, a two-stage framework that consists of a reasoning validity and execution efficiency module, along with a mechanism for transforming experiences into reusable skills.

### Strengths
Inspired by active learning theory, this work automatically selects the most informative task instances by evaluating the standard deviation of rewards across multiple rollouts for each instance.

### Weaknesses
In the RXERM mechanism, reasoning rewards are evaluated by an LLM, which is expected to be sufficiently advanced and reliable to avoid hallucinations. However, the true reasoning and planning capabilities of LLMs remain a subject of debate. LLMs primarily excel at sophisticated pattern recognition and statistical correlation rather than genuine logical deduction or causal inference ([1], [2]).

**Minor**

Several figures appear to be AI-generated, which makes them look less professional and somewhat difficult to interpret. It is recommended to replace them with clearer, manually designed visuals.

The font size of the legends is too small and should be increased to improve readability.

[1] Subbarao Kambhampati. Can large language models reason and plan? 1534(1):15–18. ISSN
1749-6632. doi: 10.1111/nyas.15125.

[2] Karthik Valmeekam, Kaya Stechly, and Subbarao Kambhampati. LLMs Still Can’t Plan; Can
LRMs? A Preliminary Evaluation of OpenAI’s o1 on PlanBench

### Questions
(1) What is the definition of a skill in this paper? Does it refer to a sequence of actions, as in the general formulation of skill learning, or to a form of domain-specific language (DSL)? Additionally, does the agent output a high-level skill or a sequence of low-level actions?

(2) Could you clarify the overall training process? Is the language agent trained directly using Equation (8)? At which stage are composite skills incorporated during training?

(3) Why does reward formatting lead to improved performance?

(4) Please consider including an additional experiment that compares the proposed method with existing group-based RL approaches such as GRPO or DAPO.

### Soundness
2

### Presentation
2

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
The paper presents SkillEvo, a two-stage framework designed to improve long-horizon task learning in LLM-based agents. At the first stage, WebGRPO integrates a reasoning and execution reward model, where LLM provides fine-grained feedback on both reasoning and task execution. Then, SkillGenesis performs skill evolution, using LLM to propose, validate and compose new skills into a skill path graph. Experiments on WebArena-Lite demonstrate strong performance improvements over existing imitation and reinforcement learning based baselines.

### Strengths
The combination of LLM-based reward modeling and dynamic skill graph evolution is conceptually interesting and extends prior RLHF-style methods toward log-horizon task learning. SkillEvo achieves improvements over RL-based baselines, demonstrating its effectiveness in long-horizon tasks. Moreover, the ablation studies provide that both RXERM and SkillGenesis contribute meaningfully to performance gain.

### Weaknesses
1. The current evaluation is limited to the WebArena-Lite environment, leaving it unclear whether the proposed framework generalizes to other log-horizon domains such as embodied reasoning [1] or mobile navigation [2]. Border evaluation across heterogenous environments would strengthen the capability of SkillEvo framework.

    [1] Reflexion: Language Agents with Verbal Reinforcement Learning, NeurIPS 2023

    [2] Mobile-Agent-v3: Fundamental Agents for GUI Automation. 2025 

2. The framework's reliance on LLM as both in RXERM for reward evaluation and SkillGenesis for post-training skill evolution raises concerns regarding scalability and reproducibility. The training pipeline is extremely resource-intensive and thus difficult to extend to domains requiring real-world interactions or limited computational budgets.

3. All (open-sourced) baselines rely solely on environmental reward feedback without any LLM-based fine-grained supervision. 
Consequently, SkillEvo benefits from a substantially richer supervision signal and external knowledge source. For a fair evaluation, the authors should include baselines that incorporate external LLM feedback [3,4].

    [3] RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback, ICML 2024
    
    [4] DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning, 2025
	
4. After policy convergence, the SkillGenesis stage employs GPT-4o to evolve skills. If LLMs can already perform such evaluation and skill composition after training, it would be more straightforward to use them directly during inference rather than integrating them into the training framework.

### Questions
1. Could SkillGenesis be appiled to GPT-4o or other LLM agents? In other words, does SkillGenesis provide additional benefits beyond what a standalone LLM could already achieve? 

2. RXERM and SkillGenesis both depend on LLM. Have the authors tried replacing GPT-4o with smaller or open-sourced LLMs?

### Soundness
2

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
4

### Summary
This paper addresses the challenge of LLM agents in long-horizon, sparse-reward web tasks by proposing the complex yet effective SkillEvo framework. The method additionally validates and designs a reasoning reward ($R_{reason}$) and introduces a method for automatically inducing a skill library, significantly improving the agent's capabilities.

### Strengths
1. The paper's proposed idea of leveraging high uncertainty to filter for the most information-rich tasks is intriguing, as it effectively optimizes the training data for LLM reinforcement learning.

2. The authors' empirical validation of the reasoning reward ($R_{reason}$), demonstrating the effectiveness and accuracy of their RXERM model through comparison with human annotations, represents a notable contribution to the broader agent research community.

3. The proposed training framework (SkillEvo) is highly effective overall, as evidenced by the significant performance improvements it delivers.

### Weaknesses
1. The paper does not appear to offer a correction mechanism for the erroneous use of skills that are already in the library, though this limitation is acknowledged as a direction for future work. 

2. The calculation of uncertainty (fluctuation) introduces a higher interaction cost, as it requires multiple rollouts per task instance to compute the standard deviation of rewards.

3. The overall framework design is heavy, which may pose challenges for reproducibility and subsequent tuning or adaptation.

### Questions
1. Could the authors provide a visualization of a complete Skill Path Graph (SPG) induced from a real task? 

2. The paper does not detail the specific algorithm or heuristic rules for selecting the prefix $\tau_D$ truncation point (i.e., step $D$). Could the authors clarify this?

3. The trigger for the SkillGenesis phase seems entirely dependent on "when a task T is successfully completed". What happens if the policy consistently fails? Is there an exploration mechanism to handle this cold-start problem?

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
3

### Summary
The authors propose SkillEvo, a two-stage framework for long-horizon web interaction tasks. The first stage, WebGRPO, enhances group-based reinforcement learning by introducing a Reasoning and Execution Reward Model (RXERM) for fine-grained feedback and a dual-uncertainty filtering strategy to select informative task instances. The second stage, SkillGenesis, transforms successful trajectories into reusable skills organized in a dynamically evolving Skill Path Graph (SPG).

### Strengths
- The paper is well-written and clearly articulates why existing group-based RL methods struggle with long-horizon tasks. 

- Innovative skill evolution framework: The SkillGenesis component with its three-stage process (proposal, genesis, evolution) provides a convenient framework for organizing and composing skills.

- Well-motivated filtering mechanism and reward mechanism: When combined with the introduced RXERM reward mechanism, the dual-uncertainty filtering strategy for selecting informative task instances is intuitive and addresses the curriculum learning problem naturally.

- Comprehensive results and ablations: The experimental evaluation is thorough, with comparisons across multiple baselines. Given the method's complexity, the ablation studies (Tables 1-2, Figures 5-6) effectively demonstrate most component's contributions.

### Weaknesses
- Missing qualitative skill evolution examples: while Table 2 shows quantitative improvements from skill evolution, the paper lacks concrete examples of how the skill library actually evolves (e.g., what atomic skills are discovered, how composite skills emerge, what the SPG structure looks like over time).
- Figure 4 is difficult to parse. The diagram contains many overlapping elements and lacks clear visual hierarchy, making it hard to understand without extensively reading the text.
- Several citations use incorrect formatting (e.g., lines 370 and 377 appear to use \citet where \citep would be appropriate).

### Questions
- How sensitive is the method to the filtering hyperparameters $\delta_p$ , $\delta_q$, p%, q%?
- As the Skill Path Graph grows, does it require any pruning mechanism to remain efficient?
- Could you provide specific examples of atomic skills and composite skills that emerged during training?

### Soundness
4

### Presentation
3

### Contribution
3
