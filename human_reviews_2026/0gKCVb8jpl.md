# Are Large Reasoning Models Interruptible?

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 6, 2, 4

## Abstract
Large Reasoning Models (LRMs) excel at complex reasoning but are traditionally evaluated in static, "frozen world" settings: model responses are assumed to be instantaneous, and the context of a request is presumed to be immutable over the duration of the response. While generally true for short-term tasks, the "frozen world" assumption breaks down in modern reasoning tasks such as assistive programming, where models may take hours to think through problems and code may change dramatically from the time the model starts thinking to the model's final output. In this work, we challenge the frozen world assumption and evaluate LRM robustness under two realistic dynamic scenarios: interruptions, which test the quality of the model's partial outputs on a limited budget, and dynamic context, which tests model adaptation to in-flight changes. Across mathematics and programming benchmarks that require long-form reasoning, static evaluations consistently overestimate robustness: even state-of-the-art LRMs, which achieve high accuracy in ideal settings, can fail unpredictably when interrupted or exposed to changing contexts, with performance dropping by up to 60% when updates are introduced late in the reasoning trace. Our analysis further reveals several novel failure modes, including _reasoning leakage_, where models fold the reasoning into their final answer when interrupted; _self-doubt_, where performance degrades while incorporating update information; and _panic_, where under time pressure models abandon reasoning entirely and return incorrect answers.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper challenges the “frozen world” assumption underlying most reasoning-model evaluations—the idea that model reasoning unfolds in static, unchanging contexts. It systematically investigates how Large Reasoning Models (LRMs) behave when their reasoning process is interrupted mid-inference or when the context dynamically changes. Two main scenarios are explored: (1) Time-constrained interruptions, where the model is prompted to stop reasoning early (“hard interrupts”) or to accelerate reasoning (“soft interrupts”); and (2) Update-driven interruptions, where new information is injected mid-reasoning, requiring the model to adapt to revised problem specifications. The authors design dedicated evaluation protocols across mathematical and coding benchmarks, analyze distinctive failure modes such as reasoning leakage, self-doubt, and panic, and propose a simple prompt-guidance technique to improve update incorporation.

### Strengths
- The paper addresses a genuine gap by moving beyond static evaluations and introducing interruptibility as a new dimension of reasoning robustness. This perspective is practically relevant for real-world applications such as interactive agents and long-running inference systems.

- The proposed evaluation protocol systematically covers both time-constrained and update-driven interruptions, yielding useful insights across multiple datasets and models. The experimental design is clear.

- The identification of reasoning leakage, self-doubt, and panic offers interpretable and human-intuitive failure categories that help conceptualize how reasoning breaks down under dynamic conditions.

- The proposed prompt-guidance method is easy to implement, and empirically effective in improving model stability and update incorporation—without requiring additional training or model modification.

### Weaknesses
- While the paper exposes the vulnerability of reasoning models to interruptions, it does not propose solution for this issue. The introduced prompt-guidance strategy is heuristic and only partially mitigates the issue, offering limited insight into how truly interruption-resilient reasoning architectures might be designed.
- Experiments are limited to open-source LRMs (e.g., Qwen3, GPT-OSS, Magistral). Validation on closed systems such as Gemini 2.5 Pro or GPT-5 would strengthen the claim that interruptibility reflects a general property of reasoning models.
- Although the paper presents interruptibility as a novel dimension of reasoning robustness, similar mechanisms have been explored in prior studies on token-budget control and inference-time adaptation. The conceptual distinction, especially between soft interrupts and thinking budget, should be articulated more clearly.
- The use of relative fractions of the reasoning trace implicitly depends on model verbosity and generation length. This may bias the interpretation of “early” versus “late” interruptions across models with differing reasoning styles.

### Questions
Please see the weaknesses.

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
4

### Summary
This paper introduces a new evaluation suite that is designed specifically to measure the robustness of reasoning models when interrupt during reasoning, as well as context changes during reasoning.
The experments are done using math reasoning and programming tasks.
The results show that current reasoning models can often fail to handle interruption or context changes during reasoning.
The paper shows the effect of introducing interruption at different reasoning lengths and demonstrates results that can be meaningful for future model development.

### Strengths
1. The problem this paper aims to show is an important one for actual model deployment.
2. The investigation of this paper covers many aspect of the proposed challenge of interruption that would be a useful resource.

### Weaknesses
1. The evaluations in this paper, while informative, still focuses on tasks that could be completed in a short period of time. Therefore, I think the proposed issue of interruption and context change may not happen if the reasoning time is relevatively short. 
2. The paper could be stronger if it includes more models to compare with, such as Deepseek etc.
3. The evaluations are mostly shown in a plot, it would be more clearer if the paper could introduce a single number metric to describe the behavior of the models when interrupt.

### Questions
1. Can the authors include tests on longer horizon tasks like SWE bench and evaluate the effect of interruption and context change there?
2. One interesting setting could be that the original context/question contains error which is later corrected during the interruption/context change, and see if the model can recognize the error and reason with the correct change.

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
4

### Summary
This work explores how large reasoning models respond to interruption scenarios. Specifically, the authors argue that current large models generate responses in a turn-based static manner, which does not align with the continuous interaction demands faced by agents in modern language model applications. To address this, the authors propose a novel framework and benchmark for evaluating model performance under interrupted conditions, and summarize several common failure patterns observed in models.

### Strengths
1. This paper presents a highly novel perspective by exploring critical challenges that future language models may encounter in environment interaction scenarios. The proposed evaluation framework and benchmark establish a foundation for assessing agent capabilities, potentially contributing to the development of autonomous agents in future research.

2. The study provides reasonably comprehensive experiments with persuasive results. Particularly, the systematic categorization of failure patterns offers valuable insights into fundamental limitations of current language models.

### Weaknesses
1. While the authors propose that models require interruption handling in agent scenarios, the selected benchmarks (GSM8K, MATH500, AIME 2024/2025, and LiveCodeBench v6) do not inherently involve agent-based tasks. This raises questions regarding the motivation and applicability of their proposed framework. From a language modeling perspective, both hard and soft interruptions disrupt the causal nature of language generation. If feasible, incorporating tasks that explicitly require agent-like behavior would help substantiate the motivation and contextual relevance of their approach.

2. The strategy of selecting interruption points proportionally appears suboptimal. A more principled approach would be to treat individual sentences as reasoning units and use them as logical boundaries for inserting interruptions.

3. The design of the update mechanism remains unclear in its validity. By maintaining the original problem statement in the augmented problem with updates, the evaluation may simply encourage the model to "memorize" the base benchmark rather than genuinely assessing its ability to resolve interruptions in complex, generalized scenarios.

### Questions
1. Could the evaluation framework and benchmarks be extended to include more agent-oriented datasets? This would significantly strengthen the rationale for studying model interruption in practical scenarios.

2. Please elaborate on the advantages of interrupting language models compared to alternative approaches such as backtracking or turn-based methods.

3. Are there more principled and dynamic methods for interrupting models? The current approach, while valuable, might still be considered somewhat "frozen" in its implementation.

### Soundness
2

### Presentation
3

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
This paper explores an unusual setting where language model generations are interrupted in the middle of chain of thought. 

### **Novel Dynamic Evaluation Scenarios**
1. Interruptions: Test partial output quality under limited computation budget.
2. Dynamic Context: Assess adaptation to changes occurring mid-reasoning.

The paper introduces a novel dataset and evaluation benchmark to assess in math and coding domain, testing whether language models are interruptible by simple stop signs and correctly interpret newly introduced content from the world. They explore soft and hard stopping for the model, and the respective effectiveness of each interruption scenario.

### **Key Findings on Benchmarks**
They found out that reasoning models are able to gracefully end their thinking regardless of the model type and sizes. Moreover, to information update scenarios, most LRMs exhibit self-doubt when provided with additional information from the user's side. Even top LRMs (high accuracy in ideal conditions) suffer severe drops (up to 60%) when interrupted or facing late context updates in math/programming tasks requiring extended reasoning.

They identify three modes of reasoning failures:
- Reasoning Leakage: Models embed partial reasoning into final answers when interrupted.
- Self-Doubt: Performance worsens while trying to integrate update information.
- Panic: Under pressure, models skip reasoning and output wrong answers.

### Strengths
### **Novelty of Dynamic Evaluation for LRMs**
This paper suggest a important and novel direction of evaluation problem of reasoning models. 
The dynamic environment setting for reasoning model seems unusual, but the provided examples below are reasonable.
1. User interruption for partial answers: In long-running tasks (e.g., proving a math theorem or debugging code), users often want progress updates or to abort early if the direction is wrong. Waiting hours for a final answer is unacceptable in interactive workflows.
2. Mid-task context updates: In collaborative coding (e.g., GitHub + AI agents), the codebase evolves while the model is thinking. A model that can't adapt to a new commit or user clarification is brittle.
3. Multi-agent environments: Future AI systems (e.g., coding agents in a shared repo) must handle concurrent changes—restarting from scratch breaks coordination.

> This dataset reflects the real-world demands as LRMs move into interactive, long-running, and collaborative roles.

### **Thorough analysis on Mid-reasoning Interruption**
1. Accuracy increases the later the interruption occurs in the reasoning trace.
- Models exhibit progressive reasoning - early interruptions yield poor results; late ones approach full performance.
- Magistral on coding tasks slightly improves when interrupted late (possibly due to overthinking avoidance).
2. Given soft interruption like end the reasoning faster, some model gives immediate wrong answer on hard tasks
3. Forcing to stop reasoning makes the model continue reasoning even in the output (e.g. \\boxed{} in math)

This shows that the models are hard to interrupt, either in soft or hard user instruction, and this implies that current reasoning models are hard to be deployed in interactive settings.

### **Analysis on model behavior when provided with extra context mid-reasoning**
1. Late-stage context updates cause sharp performance drops; models fail to recover or adapt reasoning trace.
2. Self-doubt causes models to ignore updates and stick to original reasoning, even when warned.
3. Prompt guidance (user-verified update tag) boosts adaptation, especially in math; coding remains robust.
4. Efficiency gain: Guided models recover with <110% token cost vs. full restart.
5. Late-stage updates still challenging on hard tasks (e.g., AIME).

### Weaknesses
### **Scope of Dataset Composition**

Although the paper introduces a novel evaluation direction, its dataset construction and analysis yield only incremental insights. It highlights compelling real-world scenarios requiring dynamic robustness in LRMs—such as multi-agent debate, mid-task context updates from agentic systems, and user interruptions—yet the dataset simulates only single-turn interruptions during model reasoning. The impact would be significantly greater with generalization to multi-turn, diverse dynamic scenarios.
Additionally, the authors should report dataset statistics, including:

- Distribution of interruption types,
- Total dataset size,
- Per-domain breakdown.

### **Lack of Diverse Model Analysis**

Furthermore, evaluations are limited to Qwen3 (1.7B/8B/32B), GPT-oss, and Magistral—a narrow set that does not reflect the broader landscape of open-source or closed-source LRMs. Testing widely used frontier models (e.g., DeepSeek-R1, o3, Claude 3.5, Gemini 1.5 Pro) is essential to substantiate claims of universal weakness in interrupt adaptation.Fast

### Questions
Please answer to the weakness part.

### Soundness
3

### Presentation
3

### Contribution
3
