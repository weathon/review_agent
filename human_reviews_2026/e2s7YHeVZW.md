# AgentMath: Empowering Mathematical Reasoning for Large Language Models via Tool-Augmented Agent

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 4, 4, 6

## Abstract
Large Reasoning Models (LRMs) like o3 and DeepSeek-R1 have achieved remarkable progress in natural language reasoning with long chain-of-thought. However, they remain computationally inefficient and struggle with accuracy when solving problems requiring complex mathematical operations. In this work, we present AgentMath, an agent framework that seamlessly integrates language models' reasoning capabilities with code interpreters' computational precision to efficiently tackle complex mathematical problems. Our approach introduces three key innovations: (1) An automated method that converts natural language chain-of-thought into structured tool-augmented trajectories, generating high-quality supervised fine-tuning (SFT) data to alleviate data scarcity; (2) A novel agentic reinforcement learning (RL) paradigm that dynamically interleaves natural language generation with real-time code execution. This enables models to autonomously learn optimal tool-use strategies through multi-round interactive feedback, while fostering emergent capabilities in code refinement and error correction; (3) An efficient training system incorporating innovative techniques, including request-level asynchronous rollout scheduling, agentic partial rollout, and prefix-aware weighted load balancing, achieving 4-5× speedup and making efficient RL training feasible on ultra-long sequences with scenarios with massive tool calls. Extensive evaluations show that AgentMath achieves state-of-the-art performance on challenging mathematical competition benchmarks including AIME24, AIME25, and HMMT25, substantially outperforming frontier open‑source models of comparable size.  Specifically, AgentMath-30B-A3B attains 90.6\%, 86.4\%, and 73.8\% accuracy respectively, surpassing OpenAI-o3-mini and Claude-Opus-4.0-Thinking while remaining competitive with OpenAI-o3, Gemini-2.5-Pro, and DeepSeek-R1-671B-0528. These results validate the effectiveness of our approach and pave the way for building scalable mathematical reasoning agents.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents AgentMath, an agentic framework integrating code interpreters with LLMs to enhance math problem solving and mathematical reasoning to be more accurate and computationally efficient through tool-augmented learning. It contains two major contributions. The first is the introduction of a three-step data synthesis pipeline for generating tool-augmented training trajectories. The second is a reinforcement learning (RL) techniques that combine natural language reasoning with symbolic code execution. To accelerate RL training, the authors design an efficient infrastructure leveraging asynchronous rollout scheduling and adaptive load balancing. The paper primarily uses math reasoning benchmarks (AIME24, AIME25, and HMMT) for experiments, and reports commendable improvements on those datasets over baseline models/approaches.

### Strengths
* The paper conducts an important study on integrating the Code Interpreter into LLMs to improve mathematical reasoning via symbolic computation.

* The data synthesis pipeline and training strategies (including the code execution sandbox and adaptive load balancing) are technically sound and thoughtfully designed to improve training efficiency.

* The proposed RL with Code Interpreter integration is shown to be effective through extensive experimental results.

* The experiments are extensive and systematically conducted.

* The paper is well written and easy to follow.

### Weaknesses
* Concern about benchmark: The math benchmarks (AIME24, AIME25, HMMT) each contain only about 30 questions. Prior work has shown that these datasets may have leaked into open-source model training corpora. Thus, it is unclear whether the reported results are entirely reliable. The authors are encouraged to provide more evidence on the independent contribution of the proposed approach.

* Training-testing overlap: The authors report 346k training questions for SFT and 42k for RL, while the total number of testing questions is under 100. It is questionable whether the training data already contains components or similar structures to the test questions. Although an n-gram filtering algorithm is mentioned, the specific implementation details are missing. Have the authors performed ablation studies to evaluate the effect of training data volume on test performance?

* Unclear heuristic data synthesis: The data synthesis process involves multiple human heuristics (e.g., computational component segmentation, code complexity filtering), and several components appear to require iterative manual adjustment. These steps may limit scalability to new benchmarks or domains. Moreover, the data synthesis and cleaning costs seem high given the repeated use of LLMs.

* The paper does not clarify how textual answers are segmented into computational parts or how the model determines when code usage is beneficial. Different models may prefer distinct code-usage patterns even for the same question.

* The dataset synthesis employs larger teacher models (DeepSeek R1 and V3, Qwen-30B). This may imply knowledge distillation from more powerful teacher models. Comparing such distilled models with those trained without teacher guidance may be unfair.

* The authors should consider moving key details, such as dataset sizes for SFT and RL, to the main paper because they are critical for comprehension.

* Many designs in the algorithms, such as the four quality refinement modules in Step 2 of the data set synthesis, and reward design in RL, need further justification or ablation studies. Otherwise the overall method seems to contain too many ad-hoc details that may limit the generalizability of the proposed approach.

* Typo: In line 1600, the figure reference is missing.

### Questions
1. Given that AIME24, AIME25, and HMMT each contain fewer than 30 questions and have been reported as leaked in prior work, how do the authors ensure that these benchmarks were not inadvertently seen during pretraining or fine-tuning? Have the authors considered using larger or newly constructed evaluation sets to validate the robustness of the results?
1. With 346k training questions for SFT and 42k for RL but fewer than 100 test questions, how do the authors ensure that the training data do not overlap or contain structurally similar questions to the test set?

1. What exact n-gram filtering algorithm and similarity threshold were used to remove overlapping data?

1. Have the authors conducted ablation studies to quantify how training data volume affects final test accuracy? Would performance remain stable if a smaller, filtered subset of training data were used?

1. Many steps in the data synthesis pipeline involve human heuristics such as computational component segmentation and code complexity filtering. How much manual adjustment or iteration do these steps require?

1. Can the synthesis and cleaning pipeline be generalized or automated for other benchmarks or non-math domains?

1. What is the total computational cost (e.g., GPU hours or LLM API calls) associated with synthesizing and verifying the datasets?

1. How are textual reasoning steps segmented into computational components when deciding where to inject code? What criteria or metrics determine whether using code is more effective for a given part of the reasoning chain?

1. Since different models can prefer different code usage patterns for the same question, how robust is the segmentation procedure across models?

1. The paper uses large teacher models (DeepSeek R1, DeepSeek V3, and Qwen-30B) for data synthesis. How do the authors ensure fairness when comparing with models trained without such teacher guidance? Would the performance gap shrink if competing models were also allowed to distill from comparable teacher models?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes AgentMath, a tool-augmented agentic framework for competition-level math reasoning. It tackles three pain points: (i) scarcity and noisiness of tool-use data; (ii) lack of agentic RL that interleaves generation with code execution; (iii) infrastructure bottlenecks for ultra-long contexts and many tool invocations. Concretely, it introduces an automated tool-augmented trajectory synthesis pipeline (code injection → executability verification → environment-feedback alignment → self-correction examples), an agentic RL scheme (GRPO-style optimization with interleaved code execution and selective masking of environment outputs; a reward combining correctness with tool-usage efficiency), and a scalable training system (request-level async scheduling, agentic partial rollout, prefix-aware weighted load balancing). On AIME24/25 and HMMT25, the approach reports strong results; e.g., AgentMath-30B-A3B = 90.6/86.4/73.8%, competitive with proprietary frontier models and surpassing open counterparts of comparable size. The system ablations claim 4–5× end-to-end RL throughput gains from the engineering stack

### Strengths
1. The refinement pipeline (format consistency, executability checks, feedback alignment, self-correction) shows sizable SFT gains and scaling trends before RL.
2. The async scheduler + partial rollout + prefix-aware balancing reportedly lift RL throughput 4–5×, and the paper provides a breakdown and sensitivity to segment count.
3. The 30B model achieves 90.6/86.4/73.8% on AIME24/25/HMMT25

### Weaknesses
1. Novelty over prior tool-augmented reasoning may feel incremental.
The community already has multiple tool-augmented RL frameworks (e.g., ReTool-style RL teaching strategic tool calls) and mature RL infra with asynchronous rollout and truncation/partial-trajectory techniques (e.g. AREAL, ROLL). Much of AgentMath’s lift appears to stem from more elaborate SFT data synthesis and a careful infra implementation rather than a fundamentally new RL principle. From a novelty lens, the async scheduler and partial rollout echo patterns well-established in modern RL systems; similarly, the use of truncated/segmented rollouts conceptually aligns with prior truncated policy optimization ideas (e.g. Truncated Proximal Policy Optimization). The work’s value is therefore primarily systems consolidation + scale rather than conceptual RL novelty.
2. I could not find a code repository or an explicit release statement in the current manuscript (PDF). Given that a large portion of the contribution is engineering (decoupled async architecture, agentic partial rollout, prefix-aware balancing, data-synthesis tooling), lack of code release significantly limits reproducibility and adoption. Please clarify whether code, data pipelines, and training/eval scripts will be released (e.g., upon acceptance) and provide a URL if available.

### Questions
See my comments on Weaknesses.

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
This work introduces AgentMath, a framework that enhances the mathematical reasoning ability of large language models by combining their natural‐language chain‐of‐thought with precise external tool use (e.g., code execution). The authors propose (1) an automated method to convert natural language reasoning traces into structured tool-augmented interaction trajectories (for supervised fine-tuning), (2) a multi-turn agentic reinforcement‐learning paradigm that interleaves language reasoning and real‐time code/tool invocation to learn optimal strategies for tool use and error correction, and (3) a highly efficient training infrastructure (with asynchronous rollout scheduling, partial rollouts, and prefix‐aware load balancing) enabling meaningful scaling on ultra-long sequences with many tool calls. The result is a model better able to tackle complex mathematical problems beyond pure language reasoning—off-loading heavy computation to tools while retaining robust step‐by‐step reasoning in language.

### Strengths
- This paper did a good work in presentation. All the components in the pipeline are illustrated in good details and intuitions. This provides a good recipe for open-source math prover training.

- A very comprehensive comparison between the model trained from this work and other open/closed-source models are provided, making the results very convincing.

### Weaknesses
- The paper's contribution is mostly on the development of the entire pipeline from my perspective. As many technical innovations claimed in the paper are either standard or mostly from engineering aspects, as agentic RL training is not a fresh concept nowadays (including coding-assisted math reasoning). I value a lot of the paper's efforts on developing such a comprehensive pipeline (SFT data collection and large-scale RL system building), which I understand is very challenging and helpful for the open-source community; however, the lack of technical contribution is the major factor preventing me providing a higher score.

### Questions
My major concern, as mentioned in weakness, is on the technical innovation of this paper, which I would love to hear more from the authors.

### Soundness
3

### Presentation
3

### Contribution
2
