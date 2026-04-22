# CAViAR: Critic-Augmented Video Agentic Reasoning

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 4, 6

## Abstract
Video understanding has seen significant progress in recent years, with models' performance on perception from short clips continuing to rise. Yet, multiple recent benchmarks, such as LVBench, Neptune, and ActivityNet-RTL, show performance wanes for tasks requiring complex reasoning on videos as queries grow more complex and videos grow longer. In this work, we ask: can existing perception capabilities be leveraged to successfully perform more complex video reasoning? In particular, we develop a large language model agent given access to video modules as subagents or tools. Rather than following a fixed procedure to solve queries as in previous work such as Visual Programming, ViperGPT, and MoReVQA, the agent uses the results of each call to a module to determine subsequent steps. Inspired by work in the textual reasoning domain, we introduce a critic to distinguish between instances of successful and unsuccessful sequences from the agent. We show that the combination of our agent and critic achieve strong performance on the previously-mentioned datasets.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes CAViAR (Critic-Augmented Video Agentic Reasoning), a training-free framework for video understanding. It uses an LLM as a reasoning agent that iteratively builds/executes short programs, leveraging many video tools (e.g., temporal grounding, retrieval+QA, ASR understanding), and then uses the LLM as a critic to select among multiple reasoning traces/strategies. On video QA benchmarks, including LVBench, Neptune, and ActivityNet-RTL, CAViAR reports improvements over direct inference and various baselines; ablations show that most of the improvements come from the critic and from agent multi-step reasoning.

### Strengths
- Sound framework. A simple yet working training-free agent framework that uses an LLM agent to generate/execute code with many tools for video understanding.

- Good results. The CAViAR system achieves strong results across multiple video QA benchmarks, including LVBench, Neptune, and ActivityNet-RTL.

- Useful ablations. The paper clearly ablates the effects of the critic and multi-step reasoning on the final video QA performance.

### Weaknesses
- Limited novelty. Training-free, agentic frameworks for video understanding have already been explored (e.g., VideoAgent, VideoTree). Programmatic tool use has likewise been studied (e.g., ViperGPT). This paper primarily combines an agent framework with code execution. Given the limited novelty, the work would benefit from deeper analysis, e.g., when code execution helps, when it fails, and guidelines for effective system design.

- Missing baselines. Several relevant training-free agentic systems are absent from the comparisons, notably VideoAgent and VideoTree. Including these (ideally under a shared backbone and context budget) would strengthen the empirical case.

- Additional benchmarks. The evaluation focuses largely on older video-QA datasets. It would be valuable to include more recent and challenging benchmarks such as VideoMME, MovieChat, and NextQA.

### Questions
See weaknesses. 

Can you also report average #tool calls, tokens, latency, and $/query for LVBench/Neptune/RTL (distribution helps)?

### Soundness
2

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
CAViAR is a training-free agentic system for long-video reasoning that introduces a two-stage approach: an LLM planner/executor generates multiple programmatic strategies using vision and audio tools (frame retrieval, temporal localization, ASR analysis, etc.), and a separate critic LLM compares the resulting execution traces to select the best answer.

A distinct **trace-level selection** step: a separate LLM **critic** compares multiple *executed* tool programs (with timestamps, ASR spans, intermediate rationales) and **chooses** the best evidence chain—going beyond single-run self-confidence or answer-only voting. The contribution is a **preference-over-programs** formulation that operationalizes cross-trajectory comparison as the decision rule, turning tool-grounded reasoning traces themselves—not just final answers—into the object of evaluation and selection.



Key contributions:

* Multi-strategy execution + cross-trace selection: Unlike prior video agents that rely on single-trace self-evaluation, CAViAR generates diverse reasoning paths and uses a critic to compare them—crucial for handling noisy tool outputs in long videos
* Strong empirical gains: Achieves 62.0% on LVBench (+14-35pts over baselines), 77.2% on Neptune, and 32.3 mIOU on ActivityNet-RTL, outperforming both direct inference and supervised methods

### Strengths
* Separates planning/execution from a selection stage where a critic LLM compares multiple evidence-bearing traces and picks the winner—more robust than single-trace self-confidence or fixed pipelines (clear methodological originality).
* Consistent, large gains across diverse long-video settings: strong, training-free improvements on vision-only QA, audio-augmented QA (via ASR tool), and temporal localization, with clean ablations isolating the critic’s contribution (high empirical quality).

### Weaknesses
* **Novelty**: limited. Meny works explore utilizing LLMs with Video-LMMs in agentic frameworks. The difference is the utilization of two distinct LLMs for generating programs and as a critic.
- **Insufficient benchmark coverage**: Skips Video-MME, LongVideoBench, and MLVU (temporal ordering/counting—Neptune's known failure modes). Doesn't report Neptune's own GEM metric for evidence grounding.
- **No open-model validation**: Zero replication on open LMMs & LLMs despite claiming "training-free portability." Open model results are important to see if OS models can be utilized in this framework.

### Questions
* Your paper's core claim rests on the superiority of a separate critic over other selection methods. While the ablation in Table 5 convincingly shows the critic outperforms a confidence-based self-evaluation module, it omits a comparison to simpler ensemble methods
* The evaluation is limited to LVBench, Neptune, and ActivityNet-RTL, omitting other critical long-video benchmarks like Video-MME and MLVU
* The paper claims the framework is "general" and enables performance scaling "with no additional training," yet all primary results depend on a single proprietary model, Gemini 1.5 Flash. Please include more base models to make sure method generalizes

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
3

### Summary
This paper proposes a multi-agent model named CAViAR for video understanding. The model introduces two types of agents: a reasoning agent and a critic. The reasoning agent generates multiple reasoning strategies, while the critic acts as a verifier, ranking these strategies and selecting the most appropriate one. When the critic is incorporated, experimental results show that the model achieves improved performance on several popular benchmarks.

### Strengths
- The paper proposes CAViAR, a multi-agent–based video understanding algorithm. The method appears to be training-free, utilizing existing LLMs (Large Language Models) as agents, and designs a modular system where each agent plays a different role through prompting.


- The framework includes two types of agents: a program-generating agent and a critic. The program-generating agent interprets the video and proposes multiple reasoning strategies, while the critic ranks these strategies to determine the most appropriate one. This design enhances the model’s robustness in handling complex reasoning tasks.


- Unlike existing methods such as visual programming, which rely on a fixed reasoning procedure, the proposed algorithm allows the reasoning process to adapt dynamically based on the critic’s feedback.

### Weaknesses
- Lack of Novelty in the Modular System


   - As a natural limitation of this type of work, constructing a modular system purely through prompting in a training-free manner appears to lack visible novelty in terms of model architecture or training framework. Therefore, additional experiments or analyses are needed to compensate for this limitation.


- Experiments Across Diverse LLMs


   - It would be valuable to evaluate the proposed method not only with Gemini-Flash, but also with other closed-source models (e.g., ChatGPT, Claude, etc.) and open-source models, to verify whether the approach remains equally effective across different LLMs.


- Comparison of Time Consumption and Token Usage with Baselines


   - Since the proposed approach employs a multi-agent setup, it is expected to incur higher time consumption and token usage compared to baseline methods. A more detailed and quantitative analysis of these aspects is necessary.

### Questions
Please provide your responses with reference to the weaknesses mentioned above.

### Soundness
3

### Presentation
3

### Contribution
3
