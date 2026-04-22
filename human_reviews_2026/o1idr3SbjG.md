# RoleArena: A Multi-Agent Role-Playing Environment for Long Multi-Turn Dialogues with Autonomous Plot Progression

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 6, 2, 4

## Abstract
Role-playing language agents (RPLAs) use large language models (LLMs) to simulate character dialogues, enabling immersive interactions with users.
Improving their plot progression ability plays a key role in enhancing user experience and the sense of immersion in the dialogue, with significant practical value.
However, research on plot progression in role-playing language agents is still limited.
Existing datasets often lack sufficient dialogue rounds and are missing sentences that include large-scale, clear plot twists, limiting the deep development and evaluation of related models.
To bridge this gap, we propose RoleArena, an innovative multi-agent role-playing environment designed to generate highly robust long multi-round dialogues with plot progression.
RoleArena contains several functionally independent large-model agents. To enable automatic plot progression, we introduce an environment agent that generates multiple plot storylines validated by self-reflection. After discretizing these, character agents complete plot points through dynamic dialogue, while the environment agent judges the plot completion state to smoothly drive the discrete plot forward.
To generate robust and controllable long multi-turn dialogues, we introduce a critic agent to calibrate the plot progression judgments made by the environment agent.
Using this environment, we constructed the PlotStream dataset, which includes 40.1k character configuration metadata files.
Numerous experiments have demonstrated the effectiveness of RoleArena.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces RoleArena, a multi-agent simulation environment designed to generate long, plot-driven multi-turn dialogues for role-playing language agents (RPLAs). It proposes an integrated framework that combines environment agents, critic agents, and character agents to achieve controllable plot progression. The authors also release PlotStream, a large-scale dataset (39.3k characters, 170k dialogues), and train two RPLAs, PlotRole-7B and PlotRole-72B, evaluated through the proposed RoleArena-Eval benchmark.

### Strengths
1. Clear Motivation and Problem Definition: The paper correctly identifies a gap in RPLA research: the lack of datasets and environments emphasizing plot progression in long multi-turn dialogues.
2. Comprehensive Evaluation: The proposed RoleArena-Eval provides a well-defined five-dimensional metric system (Dialogue Fluency, Character Fidelity, Emotional Expression, Story Quality, Plot Progression).

### Weaknesses
1. Limited Novelty of Framework: RoleArena builds directly upon the design principles of prior systems like CharacterBox, COSER, and RolePlot, extending them mainly by adding a critic module. The conceptual leap, while practical, may be viewed as an engineering improvement rather than a methodological innovation.
2. Evaluation Bias (LLM-as-Judge): Heavy reliance on GPT-4o as an evaluator raises questions about LLM bias, even though a human evaluation subsection is provided. It would strengthen the paper to include human–LLM agreement statistics beyond Table 4 (e.g., Pearson correlation coefficient, Cohen’s Kappa).
3. Limited Evaluation Coverage: The evaluation omits several recent high-performing closed-source models such as Gemini-2.5 and Claude-4, even though they were released before DeepSeek-V3.1. Instead, only older versions were tested. This incomplete coverage limits the credibility of the comparative analysis. Moreover, if those newer models show stronger dialogue and narrative abilities, they might serve as better candidates than DeepSeek-V3.1 for constructing higher-quality PlotStream data.
4. Dataset Quality Control: The dataset appears largely synthetic, generated via DeepSeek-V3.1 and other LLMs, with “partial human screening.” It would be beneficial to provide a detailed explanation of the quality control strategies that have been adopted.

### Questions
1. When the publications are not included in the sentence, the citations should be in parentheses using \citep{}.
2. Does the paper have detailed statistics or examples for the PlotStream dataset, such as average dialogue length, language ratio, and plot type distribution? Including such analyses or sample dialogues could help clarify the dataset’s quality and diversity.
3. Are there evaluation results for Qwen2.5-7B-Instruct and Qwen2.5-72B (base) models? Was PlotRole trained on the base models or the instruct variants of Qwen2.5? If it used the base models, what was the reason for not choosing the instruct versions, which might be better aligned for dialogue generation tasks?
4. Does RoleArena suffer from plot stagnation, where dialogues drift or repeat without reaching the point for plot progression?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes RoleArena, a multi-agent framework for generating and evaluating long-horizon role-playing dialogues. The system comprises three large language model agents: an environment agent responsible for plot control, a role agent that performs role-playing, and a commentator (critic) agent that manages pacing and turn balance. These three agents collaborate to advance the plot through a consensus mechanism defined as P(E,C)=1. Based on this framework, the authors construct PlotStream (≈39K characters and 170K dialogues) and train the PlotRole-7B/72B models. They also propose a five-dimensional evaluation system, RoleArena-Eval. Experimental results show that PlotRole substantially improves performance over comparably sized open-source models, achieving results close to some mid-tier closed-source LLMs in certain dimensions, though still trailing behind top models like DeepSeek-V3.1 and GPT-4.1-Mini.

### Strengths
1.	The paper precisely identifies a key deficiency in long-horizon role-playing dialogues—the absence of an autonomous plot-progression mechanism—and proposes a coherent and systematic solution to address this challenge.
2.	The introduction of a dedicated critic agent for adaptive control of plot advancement and pacing is conceptually novel. Its effectiveness is demonstrated through ablation experiments.
3.	The proposed PlotStream dataset and RoleArena-Eval benchmark are valuable community resources that provide a much-needed foundation for studying narrative consistency and controlled story generation.
4.	The paper presents comprehensive experiments, including comparisons across 15+ LLMs, module ablation studies, and human–LLM alignment tests, lending strong support to the empirical findings.

### Weaknesses
1.	The central formula P(E,C)=1 relies entirely on the semantic agreement between two LLMs, lacking explicit rules or theoretical grounding for when a plot transition should occur. This opaque, black-box mechanism limits transparency and weakens the theoretical rigor of the contribution.
2.	The description of “secondary checks/manual verification” suggests light post hoc inspection rather than systematic annotation. The paper provides no details about the sampling ratio, annotation criteria, or inter-rater consistency (e.g., Cohen’s k). Given that both data generation and annotation are LLM-driven, potential model biases may be embedded in the dataset.
3.	All metrics are judged solely by GPT-4o, which introduces a risk of stylistic or linguistic bias toward GPT-4o’s own preferences. The paper lacks multi-judge validation or variance analysis to ensure ranking stability. Furthermore, it remains unclear how models producing fewer than K turns are handled, possibly biasing results toward verbose models.
4.	The paper compares RoleArena with CharacterBox, but the two frameworks serve distinct purposes: RoleArena enhances generative narrative control, while CharacterBox focuses on evaluating role-playing fidelity. Treating them as direct baselines conflates different objectives, reducing the validity of the comparison.
5.	Certain sections lack clarity—for instance, Section 4.2 should more explicitly distinguish between narrative outputs and character responses in the training objective. In Section 6.2, “roleplot” may be a typographical error for “plotrole.”

### Questions
1.	If the environment and critic agents use different base models (or models of differing strengths), how would this affect turn controllability and plot progression stability? Have you conducted cross-model sensitivity experiments?
2.	What is the proportion of data subjected to manual verification in PlotStream, and what were the annotation criteria and consistency metrics? Could the authors provide a comprehensive data quality report, including statistics on linguistic, emotional, and narrative diversity?
3.	Has RoleArena-Eval been tested with alternative evaluators (e.g., Claude, Gemini, DeepSeek) to assess ranking robustness and inter-judge consistency? If not, are there plans to release scripts or datasets enabling multi-judge evaluation?
4.	How are dialogues with fewer than K turns handled during evaluation? Are such samples regenerated or discarded?
5.	If extensive manual verification is required to ensure annotation reliability, how does this approach retain its advantage over traditional datasets that provide manually annotated plot-progression statement?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper presents 4 things: a role-playing environment (RoleArena), a dataset (PlotStream), models (PlotRole), and the evaluation system (RoleArena-Eval).

The environment contains: (i) an Environment agent that creates a topic, scene, and a discretized plotline of nodes, guides characters between turns, and decides when to progress to the next node; (ii) Character agents that produce thoughts, actions, and speech; and (iii) a Critic agent that overrides/accepts the Environment agent’s progression decisions to control the total number of turns given the remaining turns and nodes.

The Environment agent first creates a discretized plotline and guides character agents turn-by-turn with short narrations; after each character action, the environment agent decides whether to move to the next plot node. The critic agent vets these progression decisions using simple heuristics that consider remaining required turns r and remaining nodes, aiming to keep the total number of turns controllable and each node sufficiently developed

The dataset is based on interactions in this environment and has 39.3k characters and 170k dialogues (avg >50 turns) with markers for plot‑driving utterances (EN/ZH). The model used for the dataset creation is DeepSeek-V3.1.

The models are fine-tuned Qwen 2.5. Training simultaneously optimizes narration generation and character response generation.

The evaluation system has 5 Likert‑style LLM‑as‑a‑judge metrics.

Experiments across many open/closed LLMs and ablations (w/o env agent, story line, critic) suggest RoleArena improves generated dialogue quality and that PlotRole models outperform base Qwen models, with strong results particularly on Plot Progression. Limited human preference checks are provided.

### Strengths
Plot progression is a well-scoped and motivated problem. The focus on long multi-turn conversations is nice. A combination of discretized plot nodes with a critic agent enforcing turn-level pacing is a reasonable approach. The paper is easily understandable, though sometimes the text is too verbose.

### Weaknesses
1. The main weakness: it's not clear what problem this environment actually solves. Usually, in all role-playing applications, models interact with users. The primary complexity in building a role-playing benchmark lies in user simulation. This paper (and several other papers) just removes users from the picture. The remaining self-playing framework evaluates whether models are good at interacting with each other. But why should anyone care about that? One of the evaluation criteria mentions "tension to keep the user engaged", but there is no user as a part of the environment.

2. Evaluation is almost entirely LLM-as-judge; human evaluation is small-scale (50 role sets) and only alignment rates are reported without details on annotator agreement, protocol, or per-dimension reliability (Table 4). It is unclear how evaluation results translate to human satisfaction in real applications. The same with filtering, authors state that the dataset "has undergone manual verification to ensure data quality". Who were the annotators? What was the protocol?

3. The critic and plot-advance rules are heuristic and simplistic (Sec. 3.2.1–3.2.3): e.g., force progress if turns ≥6, or compare remaining turns with remaining nodes ×(3–5). No learning or adaptive mechanism; limited analysis of sensitivity to these thresholds."

4. Everything is generated by DeepSeek-V3.1. It is unclear whether models generalize beyond this synthetic distribution.

5. The procedure for discretizing plotlines, how key plot-driving sentences are marked, and how the narration versus character response targets are extracted need more algorithmic details or pseudo-code.


Minor points:
1. "Dialoge" in Table 1 and prompts, "RoArena-Eval" in line 145.

2. In Figures 2 and 3, there is no point in smoothing the loss curves.

### Questions
Suggestions:

1. Expand the human evaluation: at least 3 expert annotators, 100+ dialogues per language, per-dimension scoring with inter-annotator agreement (Krippendorff’s alpha). Carefully log the whole protocol. Report correlation with LLM judge per dimension and significance tests.

2. Demonstrate generalization beyond PlotStream distributions: evaluate with out-of-domain characters/scenes not generated by DeepSeek and report whether gains persist.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes a novel multi-agent role-playing framework named RoleArena, designed to generate long multi-turn dialogues with autonomous plot progression. 

Traditional role-playing language agent (RPLA) datasets are insufficient in dialogue depth and plot progression. To address this, RoleArena introduces three types of agents: character agents, environment agents, and critic agents. The environment agent is responsible for generating discretized plot lines and dynamically judging when to advance the plot. Meanwhile, the critic agent provides a secondary evaluation of the environment agent's decisions by assessing the remaining dialogue turns and plot complexity, ensuring that the dialogue is sufficiently developed and the total turns are controllable. The critic agent is one of the key innovations in the paper. 

Based on RoleArena, the authors constructed a large-scale dataset, PlotStream, and trained the PlotRole series models. Furthermore, the paper proposes a five-dimension evaluation benchmark, RoleArena-Eval, which specifically introduces a quantitative assessment of plot progression capability.

### Strengths
The paper identifies a key shortcoming in existing RPLA datasets and models: the lack of active and logical plot progression in long multi-turn dialogues. This work presents not just an environment, but a complete package: a large-scale dataset (PlotStream), stronger models trained on it (PlotRole), and a targeted evaluation benchmark (RoleArena-Eval). The paper's ablation studies effectively demonstrate the necessity of each component.

### Weaknesses
1. **Insufficient evidence of dataset quality**: The paper provides almost no description of how the PlotStream dataset was manually checked and verified.
2. **Concerns about data diversity**: We note that the paper's method relies on the Environment agent pre-generating "discretized plot lines," which are then fleshed out to enrich the plot. Does this lead to similar or repetitive plots under different settings? This concern is twofold: (1) Are the storylines in the constructed dataset repetitive or strongly biased towards a certain style? We note the authors emphasize ensuring data diversity for character profiles, but there seems to be a gap between this and storyline diversity. (2) Can the RoleArena-Eval benchmark detect plot repetitiveness? The evaluation system lacks a measure for Inter-story Diversity.
3. **Formatting and citation issues**: There are several oversights, such as the missing citation on Line 435 and issues with the formatting of the References(e.g., the same article appears twice in the reference list).

### Questions
1. Could you provide more details on the human verification of PlotStream's quality? For example, the verification criteria, acceptance rate, and inter-annotator agreement (IAA)?
2. Given that the system uses pre-generated "discretized plot lines," how did you ensure the plots in the PlotStream dataset are diverse and not repetitive? Did you analyze the distribution or novelty of the plotlines in the dataset?

### Soundness
2

### Presentation
2

### Contribution
3
