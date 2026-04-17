# EvoAgent: Self-evolving Agent with Continual World Model for Long-Horizon Tasks

- Decision: Reject
- Scores: 4, 4, 6

## Abstract
Completing Long-Horizon (LH) tasks in open-ended worlds is an important yet difficult problem for embodied agents. Existing approaches suffer from two key challenges: (1) they heavily rely on experiences obtained from human-created data or curricula, failing to autonomously update and select multimodal experiences, and (2) they may encounter catastrophic forgetting issues when faced with new tasks, failing to autonomously update world knowledge. To solve these challenges, this paper presents EvoAgent, a self-evolving agent with a continual World Model (WM), which can autonomously complete various LH tasks across environments through self-planning, self-control, and self-reflection, without human intervention. Our proposed EvoAgent contains three modules, i.e., i) the memory-driven planner which uses an LLM along with the WM and interaction memory, to convert LH tasks into executable sub-tasks; ii) the WM-guided action controller which leverages WM to generate low-level actions and incorporates a self-verification mechanism to update multimodal experiences; iii) the experience-inspired reflector which implements a two-stage curriculum learning algorithm to select experiences for task-adaptive WM updates. Moreover, we develop a continual World Model for EvoAgent, which can autonomously update the multimodal experience pool and world knowledge through closed-loop dynamics. We conducted extensive experiments on Minecraft and Atari, compared with existing methods, EvoAgent can achieve an average success rate improvement of 111% and reduce ineffective actions by more than 6x.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes **EvoAgent**, a self-evolving agent with a **Continual World Model**, which addresses the challenges of **experience dependency and catastrophic forgetting** in long-horizon tasks. By enabling **self-planning, self-control, and self-reflection**, EvoAgent autonomously decomposes tasks and updates world knowledge. In experiments on **Minecraft** and **Atari**, it achieved an **average success rate improvement of 105%** and **reduced ineffective actions by over six times**.

### Strengths
1. **The Novel Self-evolving Agent Framework**
    The paper introduces **EvoAgent**, a self-evolving agent with a **Continual World Model (CWM)** that enables autonomous *self-planning, self-control,* and *self-reflection* without human intervention. The proposed *planning–control–reflection* loop forms a coherent self-improving mechanism, addressing the limitation of one-directional learning in prior agents.
2. **Effective Continual Learning and Forgetting Mitigation**
    **EvoAgent** achieves continual world knowledge updates and prevents catastrophic forgetting through weighted experience replay, Fisher-based regularization, and LoRA fine-tuning for adaptive task learning.
3. **Strong Technical Integration**
    The framework seamlessly combines **LLMs** for task decomposition, **World Models (RSSM)** for control, and **Curriculum Learning** for experience selection—enhancing autonomy and generalization.
4. **Comprehensive and Convincing Experiments**
    On **Minecraft** and **Atari**, EvoAgent improves average success rates by **105%** and reduces ineffective actions **sixfold**. Ablation studies show the continual world model contributes **72%** of the performance gain.
5. **Clear Structure and Logical Presentation**
    The paper is well-organized, with a coherent narrative from motivation to validation, making its technical and empirical contributions easy to follow.

### Weaknesses
1. **Typographical  issues.**
    For example, the paper states “including WM-based agents (such as PPO Schulman et al. (2017), DreamerV3 Hafner et al. (2025)),” but PPO is a **model-free** algorithm rather than a WM-based method.
2. **Lack of quantitative evidence for the “6× fewer ineffective actions” claim.**
    The abstract mentions this improvement, but no explicit numerical data or training performance curves are provided to support it—only indirect reflection via the final EE metric. Adding corresponding training curves would greatly strengthen this claim’s credibility.
3. **Insufficient details on the reward predictor.**
    Unlike DreamerV3, EvoAgent’s reward predictor is goal-conditioned. However, the paper does not specify its structure and training objective.
4. **Unclear visual encoding path in the Experience-driven Task Planner.**
    Since the experience pool $D_{MEP}$ includes visual observations (Obs), these should logically pass through an encoder and projector. Yet this process is not clearly illustrated or described, which may cause confusion about the data flow.
5. **Incomplete experimental comparison.**
     The experimental section only compares EvoAgent against PPO, DreamerV3, Jarvis-1, and Optimus-1, but omits **more recent model-based baselines**, such as *LS-Imagine* [1].
6. **Unreported computational cost of GPT-4o usage.**
    The paper employs GPT-4o for online learning, but the token consumption and computational cost are not reported. It would be helpful to clarify whether this process is computationally expensive or practical for large-scale applications.

[1] Wang Qi, er al. Open-World Reinforcement Learning over Long Short-Term Imagination,  **ICLR 2025 Oral**

### Questions
1. **On text tokenization of multimodal data.**
    How can (D_{exp})—which contains actions and structured numerical information—be directly fed into a text tokenizer? Can GPT-4o truly interpret such structured multimodal data, or is there an intermediate formatting or symbolic conversion process?
2. **Alternative world models.**
    Have the authors experimented with different world model architectures, such as **Transformer-based approaches (e.g., DiT or Decision Transformer)**? If not, what motivated the choice of RSSM?
3. **Transferability to robotics.**
    Could EvoAgent be extended to real or simulated robotic settings (e.g., **Libero** or manipulation tasks)?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents EvoAgent, a novel framework designed to address the challenges of long-horizon (LH) task completion in open-ended environments. The core contribution is a self-evolving agent that integrates a continual World Model (WM) with three key modules for planning, control, and reflection, enabling autonomous experience accumulation and knowledge updates without human intervention. The paper is well-structured, tackles a significant and timely problem in embodied AI, and is supported by extensive experiments in Minecraft and Atari, showing impressive quantitative improvements over strong baselines.

### Strengths
The proposed integration of a continual WM within a closed-loop planning-control-reflection cycle is a compelling and timely contribution. Addressing the limitations of existing methods (reliance on human curricula, catastrophic forgetting) by enabling agents to autonomously update and select multimodal experiences is a significant step forward for long-horizon task solving.

The experimental evaluation is thorough and convincing. The use of the challenging Minecraft benchmark provides strong evidence for the method's effectiveness and generalization capability.

### Weaknesses
1. The paper's writing needs to be improved a lot. For example, almost all the citations are not appropriately shown (without the parenthesis), which makes the understanding difficult; On line 207, the model-based RL is introduced, while it seems no format text refers to it. For action selection, there is no technical details provided about how to solve Eq 14. On the other hand, the basic machnism of LLM (line 247-253) can be omited to save the space.

2. The core contributions needs to be more clear. There are too many new modules within EvoAgent, including the Task Planner, the Action Controller, the continual World Model, and the Reflector. Are all these modules new? If yes, their novelties should be made more clear, with necessary ablation experiments need to be include, for example:
- why there must be two stages of CL
- why there must be some many terms on Eq 18,  20 and 22
- why these must be a self-verification for controller. 

(Considering the page limit of a conference paper, it is suggested to narrow the topic such that the core contribution can be more clear.) 
On the other hand, if the answer is no, there should be more content moved to preliminary and the author can focus to interpretate the novelty of the rest part.

3. Why use only the failure expeirence to LoRA finetune the planner? Why the multimodal experience can improve the planner? How the mechanism  altered compared to the experience-free version, or with text-only experience? There should be more in-depth analysis in the experiment.

### Questions
- How is the computational overhead of the full EvoAgent loop (especially the two-stage CL and LLM fine-tuning) compared to a baseline like DreamerV3?
- why there must be two stages of CL
- why these must be a self-verification for controller. 
- why use only the failure expeirence to LoRA finetune the planner
- why the multimodal experience can improve the planner?

### Soundness
3

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
This paper proposes EvoAgent, a self-evolving agent for long-horizon tasks that combines an LLM-based planner, RSSM-based world model controller, and a reflector that updates the world model continually by a hierarchical trajectory selection from the Multi-model Experience Pool (MEP). The selection is adaptively based on relevance, efficiency, importance, and complete ratio. The key contribution is a continual world model that autonomously updates through closed-loop planning-control-reflection, addressing catastrophic forgetting in sequential task learning. Experiments are conducted on Atari and Minecraft, showing an average 105% improvement in success rate over baselines on the latter setting.

### Strengths
- Important problem formulation: The paper addresses autonomous experience accumulation and world knowledge updates for long-horizon tasks, eliminating reliance on human-designed curricula or demonstrations, which is critical for real-world deployment.
- Novel continual world model design: The two-stage curriculum learning using relevance, efficiency, world model divergence, TD-error, and gradient norms as weight signals is innovative, prioritizing experiences that maximally update environmental understanding efficiently.
- Strong empirical results: Achieves 105% average success rate improvement and 6× efficiency gains over strong baselines, with performance advantages increasing systematically on harder, longer-horizon tasks.
- Sound experimental design: Tests across Minecraft (67 tasks) and Atari environments, compares against diverse baselines (model-free, model-based, LLM-based), uses appropriate metrics (success rate, exploration efficiency).

### Weaknesses
- Minor spelling errors: "Atari" is spelled "Atair" in multiple instances.
- Some ambiguities in method description:
  - No explicit description on how to determine "relevant" experiences from MEP for planner fine-tuning.
  - Figure 2 is daunting and unintuitive, without clear temporal pipeline of in what order each module infers or updates.
- Missing analysis on cost of LLM API calls. This is significant because the method relies on LLM inference for frequent planning.
- Some personal concerns on the soundness of the method. Please see questions.

### Questions
- Section 3.2 states "experience trajectories relevant to the subtask $g_i$ are extracted" for LoRA fine-tuning after failures, but never specifies how relevance is determined. Is it exact label matching, embedding-based similarity, or another method?
- The fine-tuning uses pairs $\{(X_{in}^{(k)}, X_{out}^{(k)})\}$ where inputs are lower-level experiences and outputs are higher-level subtasks. Both are denoted "X" despite representing different modalities, making the supervision structure unclear. Can you clarify this notation and confirm the understanding?
- Subtask failures trigger planner fine-tuning, but failures can stem from poor world models, suboptimal control, or environmental stochasticity, not just planning errors. Why should the planner always update on failures? Moreover, fine-tuning requires successful subtask sequences as supervision. How does this work initially when MEP contains no successful experiences?
- The self-verification threshold σ=0.9 is labeled "set empirically" without principled analysis. Is there sensitivity analysis, validation-based tuning, or theoretical justification? How are state embeddings from WM and goal embeddings from LLM aligned for meaningful cosine similarity?

### Soundness
3

### Presentation
2

### Contribution
3
