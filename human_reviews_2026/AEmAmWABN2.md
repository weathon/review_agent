# Mind-Map Agent: Enhancing Cooperative Task Planning through Communication Alignment with Large Language Models

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Embodied agents that collaborate with humans through natural language have become an active area of research, offering flexibility in cooperative planning and execution. Debate-based approaches often depend on repeated consensus procedures, which can increase dialogue frequency and risk over-communication. At the same time, LLMs are prone to hallucination during dialogue processing, sometimes causing confabulation and reducing consistency in long-term strategies. We introduce the Mind-Map Agent, an approach that guides reasoning with explicit cooperative strategies while maintaining structured long-term memory to disentangle dialogue, task state, and planning context. The generated Mind-Maps support coherent long-horizon planning, reduce redundant dialogue, and enhance interpretability in multi-agent interaction. Evaluations on Communicative Watch-and-Help and ThreeDWorld Multi-Agent Transport indicate that the Mind-Map Agent achieves more stable efficiency compared to classical planners and LLM agents across different model scales and environments. Our results suggest that Mind-Map reasoning enables cooperative agents to accomplish tasks with fewer conversations while sustaining effective collaboration.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents the Mind-Map Agent, an extension of the cooperative embodied agent CoELA, which introduces an explicit internal reasoning structure—the mind map—to help mitigate confabulation and reduce redundant dialogue in language-based cooperation effectively. Experiments conducted on C-WAH and TDW-MAT demonstrate that the proposed approach outperforms both CoELA and traditional baselines. An ablation study further confirms the contribution of each component within the Mind-Map framework.

### Strengths
- The paper identifies a critical limitation in communicative cooperative planning and proposes a well-motivated approach.
- The paper is well-written.
- The ablation experiments are commendable.

### Weaknesses
- The contribution appears incremental relative to existing works such as CoELA and CaPo.
- The paper lacks sufficient details on how the Mind-Map, the core component of the proposed method, is updated. Does this process involve additional LLM calls? If so, what are the efficiency trade-offs in terms of total computational cost and LLM usage?
- While evaluating the method across four different LLM backbones is commendable, the observation made in line 347 that Qwen2.5-7B cannot even follow the prompts to finish the task makes these results meaningless. Also, there is only one LLM-based baseline (CoELA) included. Incorporating additional baselines, such as CaPo mentioned in the paper, would strengthen the empirical validation. Moreover, as noted in line 357, testing CoELA with unadapted prompts and different LLM backbones from its original setup raises concerns about fairness and comparability. Evaluating the proposed method using the same backbone originally employed by the baseline would make the results more convincing.

### Questions
- What's the actual implementation of updating the Mind-Map? Is there LLM call required? How many?

- Why not test other existing baselines like CaPo?

- Why not report performance with the original llm backbone for CoELA?

### Soundness
2

### Presentation
3

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
The paper introduces **Mind-Map Agent**, a framework designed to enhance cooperative task planning and communication in multi-agent settings. The key idea is to guide LLM-based reasoning with a **structured memory system**, called the *Mind-Map*, which explicitly separates others’ intentions, cooperative strategy, and communication strategy. This structure aims to reduce confabulation, maintain consistency in long-horizon reasoning, and lower communication overhead during collaboration.

The authors evaluate the proposed framework on two embodied AI benchmarks, C-WAH and TDW-MAT**,** using multiple LLMs of different scales. Results show that the Mind-Map Agent generally achieves higher cooperative efficiency and fewer communication turns compared to baseline methods such as COELA and classical planners. An ablation study further analyzes the contribution of each Mind-Map component.

### Strengths
- The paper is well-written and easy to follow, raising important unsolved issues in multi-agent communication and long-horizon plan consistency.
- By introducing an inspectable memory schema, the work moves toward making language-agent reasoning more explainable. The Mind-Map entries (intentions, cooperative strategies, communication plans) allow researchers to trace decision-making steps and understand more about how LLM agents communicate with each other.

### Weaknesses
- The framework of the Mind-Map Agent largely builds upon the existing COELA architecture, with the primary modification being the introduction of a structured memory system. While this addition is interesting, the overall contribution appears somewhat limited in scope. The concept of structured or persistent memory has already been explored in prior embodied agent systems, such as WAH [4], where mechanisms like MCTS were used to record information and guide decision-making.
- The experimental comparison is currently insufficient to substantiate the claimed improvements. The evaluation primarily contrasts Mind-Map with COELA, omitting more recent and relevant baselines such as REVECA [1], CaPo [2], and CoTS [3], which have advanced the design of collaborative embodied agents after COELA. Although some of these works are cited in the related work section, quantitative or qualitative comparisons are missing.
- It remains unclear whether the Mind-Map Agent can effectively cooperate with heterogeneous partners, including agents of different architectures or human collaborators. Since the proposed system relies on internally inferred intentions and self-developed cooperative strategies, it is uncertain how well these inferences align with partners that follow different reasoning paradigms.

*[1]* Seo, SeungWon, et al. "Reveca: Adaptive planning and trajectory-based validation in cooperative language agents using information relevance and relative proximity." *Proceedings of the AAAI Conference on Artificial Intelligence*. Vol. 39. No. 22. 2025.

*[2] Liu, Jie, et al. "CaPo: Cooperative Plan Optimization for Efficient Embodied Multi-Agent Cooperation." The Thirteenth International Conference on Learning Representations.*

*[3] Zu, Lizheng, et al. "Collaborative Tree Search for Enhancing Embodied Multi-Agent Collaboration." Proceedings of the Computer Vision and Pattern Recognition Conference. 2025.*

*[4] Puig, Xavier, et al. "Watch-And-Help: A Challenge for Social Perception and Human-AI Collaboration." International Conference on Learning Representations.*

### Questions
- See Weaknesses.
- Providing examples that illustrate the evolution trajectories of the Mind-Map memory throughout task progression would greatly help readers understand how effectively the memory system supports long-horizon planning. In particular, case studies showing how the memory representation changes in response to a partner’s messages or environmental updates could clarify the dynamic behavior and interpretability of the proposed structured memory.
- The ablation results in Table 3 indicate that the Communication Strategy alone achieves notably strong performance compared to more complex combinations of components. Regarding this, the author mentions that flexibility is important. However, the design of Mind-Map is a structured schema. This raises the question of whether certain structural constraints or components of the Mind-Map might be unnecessary—or even detrimental—to overall performance.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes Mind-Map Agent, an LLM-driven framework that separates (i) dialogue history and working memory from (ii) a structured long-term “Mind-Map” (intentions, cooperative strategy, and communication strategy). The system reasons in two phases: Mind-Map Generation and Plan Selection, and aims to mitigate confabulation, reduce redundant dialogue, and improve long-horizon coordination. Experiments on C-WAH and TDW-MAT (with symbolic and visual observations, several LLM backbones) show consistent reductions in steps and often communication turns versus CoELA and classical planners. An ablation on the schema’s components is also presented.

### Strengths
* Clear, motivated idea: Explicit, inspectable schema for intentions and strategies is well aligned with Theory of Mind literature and practical multi-agent needs.
* General mechanism: The JSON-style schema and two-phase prompting is architecture-agnostic and easy to port across models and environments.
* Compelling empirical signal: Across multiple backbones and two benchmarks, Mind-Map variants typically achieve notably fewer steps than CoELA and classical baselines; communication turns often drop substantially as well.
* Interpretability: The design encourages transparent, auditable reasoning (e.g., fields like “Reason to Send” vs “Reason not to Send”), which is valuable for human-robot interaction.
* Thoughtful discussion: The paper acknowledges trade-offs such as communication efficiency versus cautious reasoning.

### Weaknesses
1. Evaluation scope and baselines

   * Only two simulation benchmarks are used, and both are from the CoELA framework. Protocolized baselines (e.g., CaPo, REVECA) are discussed but not included. This weakens generality claims.
   * CoELA prompts were optimized for GPT-4, possibly disadvantaging other LLMs. Prompt engineering effort should be balanced across methods.

2. Metrics and statistical rigor

   * C-WAH evaluation only reports Average Steps (AS) and Average Communication Steps (ACS), with no explicit success rate. How failures are treated is unclear.
   * Only three runs are used per experiment, and no significance tests are provided. Variances are large in places.

3. Confabulation reduction is argued but not measured

   * The paper highlights that Mind-Map mitigates confabulation, but provides no quantitative measure or error count to support this.

4. Ablation clarity

   * Section 4.3 omits the structural key-matching process, which makes it hard to assess the true incremental contribution of each component.

5. Method specifics and reproducibility

   * More detail is needed: exact schema template, in-context examples, prompt templates for both reasoning phases, memory retrieval and pruning rules, token budgets, and compute details.

### Questions
Check weaknesses please.

### Soundness
3

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
4

### Summary
This paper introduce Mind-Map Agent, a LLM-based structured reasoning framework for multi-agent embodied AI tasks. The key idea is to prevent confabulation and excessive dialogue in LLM-based cooperative planning by introducing a Mind-Map memory, which is a structured long-term representation that explicitly separates dialogue history, task state, inferred intentions, cooperative strategy, and communication strategy. Experiments on C-WAH and TDW-MAT demonstrate that the proposed framework achieves better performance and costs fewer dialogue turns, better task completion than CoELA and MCTS-based baselines.

### Strengths
1. Mind-Map agent is an interpretable and effective way to structure LLM reasoning in cooperative tasks.
2. Experiments includes multiple models (GPT-4o-mini, GPT-OSS-120B, Llama 3.3-70B, Qwen2.5 7B) and two benchmarks, showing consistent performance gains and reduced communication rounds.
3. Ablation studies demonstrates how each Mind-Map component contributes differently to efficiency.

### Weaknesses
1. Although emergent behaviors are mentioned, richer qualitative visualization or human evaluation are needed.
2. While Mind-Map reasoning reduces hallucination and dialogue turns, it can increase computational cost with more LLM queries and memory management.
3. The proposed Mind-Map is essentially a reorganization of existing memory modules. Its “Long-Term Memory” resembles mechanisms already explored in LLM-based agents such as Long-Term Memory: The Foundation of AI Self-Evolution, while the maintenance of cooperative strategies are proposed by CaPo (ICLR 2025), and the representation of intentions and dialogue history are proposed from CoELA (ICLR 2024). This paper mainly combines these memory elements into a unified JSON format.

### Questions
See weaknesses

1. Could you provide more qualitative examples, including the evolution of Mind-Map over steps?
2. Could you add more baselines, such as CaPo, REVECA, and CoTS?

CaPo: Cooperative Plan Optimization for Efficient Embodied Multi-Agent Cooperation. ICLR 2025

Collaborative Tree Search for Enhancing Embodied Multi-Agent Collaboration. CVPR 2025

REVECA: Adaptive Planning and Trajectory-based Validation in Cooperative Language Agents using Information Relevance and Relative Proximity. AAAI 2025

### Soundness
3

### Presentation
2

### Contribution
2
