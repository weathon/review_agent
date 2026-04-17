# UltraHorizon: Benchmarking LLM-Agent Capabilities in Ultra Long-Horizon Scenarios

- Decision: Reject
- Scores: 4, 2, 6, 6

## Abstract
Autonomous agents have recently achieved remarkable progress across diverse domains, yet most evaluations focus on short-horizon, fully observable tasks. In contrast, many critical real-world tasks, such as large-scale software development, commercial investment, and scientific discovery, unfold in long-horizon and partially observable scenarios where success hinges on sustained reasoning, planning, memory management, and tool use. Existing benchmarks rarely capture these long-horizon challenges, leaving a gap in systematic evaluation. To bridge this gap, we introduce $\textbf{UltraHorizon}$, a novel benchmark that measures the foundational capabilities essential for complex real-world challenges. We use exploration as a unifying task across three distinct environments to validate these core competencies. Agents are designed in long-horizon discovery tasks where they must iteratively uncover hidden rules through sustained reasoning, planning, memory and tools management, and interaction with environments. Under the heaviest scale setting, trajectories average $\textbf{200k+}$ tokens and $\textbf{400+}$ tool calls, whereas in standard configurations they still exceed $\textbf{35k}$ tokens and involve more than $\textbf{60}$ tool calls on average. Our extensive experiments reveal that agents powered by state-of-the-art LLMs consistently underperform in these settings, whereas human participants achieve much higher scores, underscoring a persistent gap in agents' long-horizon exploration abilities. We also observe that simple scaling fails in our task. To better illustrate the failure of agents, we conduct an in-depth analysis of collected trajectories. We identify eight types of errors and attribute them to two primary causes: in-context locking and functional fundamental capability gaps.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents UltraHorizon, a benchmark framework designed to evaluate the long-horizon reasoning and planning abilities of large language models (LLMs). It constructs a suite of hierarchical, multi-step reasoning tasks inspired by planning and reinforcement learning literature. Each task involves sequential subtasks that require maintaining coherent intermediate states and long-term dependencies.
UltraHorizon introduces two main components: (1) a hierarchical task generator that synthesizes long-dependency tasks across text-based domains (e.g., logistics, scheduling, and simulation-based puzzles); and (2) an evaluation engine that measures models along three axes—planning correctness, consistency across steps, and cumulative goal completion. The authors benchmark a wide range of LLMs (from GPT-4o and o3-mini to open-source instruction models) and claim that current LLMs show steep degradation as task horizon length increases.

### Strengths
1. Comprehensive and well-documented benchmark design.
2. Careful control of task horizon and complexity, enabling clear empirical trends.
3. Extensive experiments covering a wide range of open and closed LLMs.
4. Clear empirical finding: reasoning and consistency sharply deteriorate with horizon length, even for advanced models.
5. The work may serve as a useful diagnostic suite for model developers focused on reasoning persistence and error accumulation.

### Weaknesses
1. Incremental innovation. The concept of evaluating long-horizon reasoning is not new, and UltraHorizon does not introduce a fundamentally different methodology from existing multi-step reasoning or planning benchmarks.
2. Limited interpretability. The evaluation is purely outcome-based and does not analyze the internal reasoning process or state representations of models. As a result, the paper offers limited insight into why models fail as horizons grow.
3. Questionable ground truth. The rule-based “oracle planner” cannot account for alternative valid solutions, especially in tasks involving open-ended reasoning or multi-path planning, which may penalize models unfairly.
4. Synthetic overfitting. The benchmark tasks are entirely synthetic and templated; they risk overrepresenting pattern-based reasoning rather than real long-horizon dependencies found in interactive or embodied settings.

### Questions
1. Please add experiments validating the robustness of your evaluator—e.g., measure agreement between the rule-based oracle and human judgment on a subset of long-horizon outputs (20–50 samples). This would strengthen claims about evaluation fairness.
2. Include a comparison between models with and without memory or scratchpad reasoning (e.g., using RAG or token caching). This would demonstrate whether UltraHorizon effectively measures “long-horizon” reasoning or simply longer output persistence.
3. Provide examples of model failure trajectories across horizon lengths (e.g., horizon = 5, 10, 15). What specific types of reasoning breakdowns (logical drift, context forgetfulness, step duplication) occur? A small taxonomy of failure patterns would greatly increase insight.
4. Could you quantify inter-run stability to assess the sensitivity of evaluation outcomes to sampling randomness?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper presents a new benchmark called UltraHorizon, targeting long-horizon tasks (e.g., involving hundreds of steps and 200K+ tokens). The benchmark includes three environments: Mystery Grid, where an agent explores a 10x10 grid under energy and step budgets to infer the hidden mapping from symbols A to E; Sequence Exploration, where agents deduce transformation rules applied to pairs of letter sequences; Alien Genetics Laboratory: agents investigate inheritance mechanisms in a triploid alien organism through controlled genetic experiments. The paper then evaluated five LLMs using UltraHorizon and found that they consistently underperform than human.

### Strengths
- Long-horizon is an increasingly important set of tasks. Benchmarking in this direction is hugely needed. 
- The paper conducts a comprehensive evaluation using the benchmark, with a detailed error analysis

### Weaknesses
- I don't really understand any of the three environments set up
- The three environments are all made up, while what would be valuable is to use real-life long-horizon tasks
- The CRNR approach is promising but relatively simple

### Questions
Can you explain these three environments a bit more?

### Soundness
2

### Presentation
1

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
UltraHorizon is a benchmark designed to evaluate large language model (LLM)-based agents in ultra-long-horizon and partially observable environments, addressing limitations of existing short-horizon benchmarks. It introduces three diverse tasks, Mystery Grid, Sequence Exploration, and Alien Genetics Laboratory, that collectively test sustained reasoning, planning, memory management, and tool use across extended interactions averaging over 200k tokens and 400 tool calls. Experiments with leading LLMs (e.g., Gemini-2.5-Pro, GLM-4.5, DeepSeek-V3, Qwen3-235b) reveal that all models substantially underperform compared to humans, with performance deteriorating as task horizons lengthen. Scaling up interaction steps or context size does not resolve these failures, which stem from in-context locking and foundational capability gaps in reasoning and memory. UltraHorizon thus highlights the urgent need for structured memory mechanisms, adaptive exploration, and long-term reasoning strategies to advance autonomous agent intelligence.

### Strengths
(1) The paper is clearly written and well-motivated.

(2) The designed experiments are interesting and indeed very challenging.

### Weaknesses
(1) Some experimental setting is missing.

(2) Some experimental outcome analysis is a little bit subjective and does not have empirical evidence to support, e.g., the irrelevance of the reasoning difficulty to the task performance, and also Takeaway 5, see questions for more details.

(3) Among the three designed benchmarks, what are some real-world implications of the first two? This might further motivate the reason why we want to study agentic capabilities in this setting, as compared to just designing some random challenging scenarios, exploring the limits of agentic AI.

(4) For many of the failure analyses, if we can propose some corresponding solutions and demonstrate they are effective in mitigating the issue, that would be much more useful.

### Questions
(1) The current LLM-as-Judge evaluation is a little bit subjective, especially given that the prompt itself sets a certain number of rules there, and also, evaluating the mapped rule itself is a problem with complex logic. I am wondering whether we can conduct a human evaluation benchmark to verify the goodness of using LLM-as-Judge to evaluate against the human evaluator performance.

(2) For Takeaway2, where has the claim "despite often surpassing humans on math-reasoning benchmarks" been verified?

(3) Between line 369 and line 374, although the decreasing performance has been demonstrated as the number of horizon levels increases, it cannot indicate that the reasoning difficulty will not affect agentic performance. It is suggested to design some additional experiments that can vary the reasoning difficulty to verify this. However, currently, it is unclear how to vary the reasoning difficulty of the Mystery Grid Environment Task. However, the observed relation between decreasing performance and the horizon itself is valuable, though.

(4) For in-context locking, although Figure 6 demonstrates the decreasing entropy as the sequence progresses, it does not show that this decreasing trend has any relation to the final success/failure rate. Therefore, I am not sure whether this observation can support failure analysis.

(5) The experimental setting of Figure 7 is missing. How do you categorize the failure? Are you asking a human to conduct manual classification?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
UltraHorizon is a benchmark for agent evaluation in ultra long-horizon, partially observable settings. It introduces three exploration-centric environments that require sustained hypothesis formation, memory management, planning, and tool use. Trajectories in the UltraHorizon benchmark can reach 200k+ tokens and 400+ tool calls, with 'standard' runs still exceeding 35k tokens and 60+ tool calls. The authors evaluate five modern LLMs in fixed-step and free-step regime and report consistently humans outperforming LLMs. Further, they analyze failures, highlighting in-context locking and fundamental capability gaps.

### Strengths
- The problem description is interesting and well motivated. Current LLMs fail on long-horizon tasks and it is clearly elicited in the paper through the UltraHorizon benchmark.
- Paper is easy to read, and the writing style is easy to follow
- The paper discloses trace lengths, tool-call counts, and completion tokens per model per environment (fixed & free).
- Interesting insight on naive scaling hurting performance.

### Weaknesses
- Generalizability of CRNR: Scaling curves focus only on GLM-4.5. To claim CRNR as a general recipe, show cross-model results (e.g., Gemini, Qwen) on other models as well. 

- LLM-as-a-judge evaluation: Evaluation uses a single DeepSeek-R1 judge. There’s no report of agreement with humans. I would be curious to see if some human eval results change the scores.

- Typos: Abstract says “eight types of errors,” but the body later says "nine recurring error patterns." (L93). Moreover, The figure shows eight categories.

### Questions
Please refer to the weakness section.

### Soundness
3

### Presentation
3

### Contribution
3
