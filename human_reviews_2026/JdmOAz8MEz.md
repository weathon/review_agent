# Tracing LLM Reasoning Processes with Strategic Games: A Framework for Resource-Constrained Decision Making and Revision

- Decision: Reject
- Scores: 4, 2, 6, 6

## Abstract
Large language models (LLMs) are increasingly applied to tasks that require complex reasoning. While most benchmarks focus on evaluating final reasoning outcomes, they overlook the internal processes that lead to those outcomes—such as how a model revises and makes decisions under constraints. We argue that evaluating these internal reasoning steps is essential for understanding model behavior and improving reliability in real-world applications. To make these processes observable and measurable, we propose using strategic games as a natural and effective environment. These games operate within closed, rule-based systems and provide interpretable states, limited resources, and automatic feedback. Therefore, we propose a framework to evaluate LLMs along two core process dimensions: resource-constrained decision making and revision. To support this, we introduce a set of evaluation metrics that extend beyond traditional Win Rate(WR), incorporating measures such as Over-Correction Risk Rate (ORR), Correction Success Rate (CSR), Improvement Slope ($\beta$), and Over-Budget Rate (OBR). In a set of 4320 adversarial rounds across 12 state-of-the-art models, we find that ChatGPT-o3-mini, which demonstrates strong reasoning capabilities, achieves the strongest results across process metrics(74.7\% WR, 78.6\% CSR, and $\beta=+0.041$). In contrast, Qwen-Plus, despite a high ORR of 81.6\%, achieves only a 25.6\% win rate, primarily due to excessive resource use. We observe a consistent negative trend between ORR and CSR, suggesting that more frequent corrections do not always improve outcomes. This pattern may reflect high-frequency revisions made prematurely, which often reduce overall effectiveness, whereas more targeted revisions are associated with higher accuracy. We hope this work offers a new direction for LLM evaluation—focusing not just on what models decide, but on how they decide it.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces AdvGameBench, a framework for evaluating/understanding large language models’ reasoning processes rather than only their outcomes. Instead of static question answering, the benchmark places 12 leading LLMs in adversarial, rule-based game environments (tower defense, battle card, and turn-based combat) that produce interpretable, resource-constrained decision traces. The authors define five core process-level metrics, Win Rate (WR), Over-Correction Risk Rate (ORR), Correction Success Rate (CSR), Improvement Slope (β), and Over-Budget Rate (OBR), to quantify how models adapt, revise, and manage resources across thousands of simulated rounds. The results suggest that selective correction and constraint adherence are better indicators of reasoning quality than raw win rate.

### Strengths
1. Process-aware evaluation paradigm: The paper shifts LLM evaluation from outcome-based correctness toward process-level reasoning analysis, addressing an important gap in current benchmarking.
2. Novel and interpretable metrics: The proposed ORR, CSR, β, and OBR move beyond win rate to capture and understand revision behaviour and resource constrained decision making, both crucial for real-world reliability.
3. Comprehensive experimental coverage: Twelve major models are evaluated across 4,320 adversarial rounds, spanning distinct game types that probe different strategic competencies.
4. Well-controlled environments. The authors explicitly redesign game mechanics to eliminate data contamination and ensure fair, rule-governed evaluation.

### Weaknesses
1. Lack of formal notation for the main metrics. The five central measures (WR, ORR, CSR, β, OBR) are described narratively but never defined mathematically. In contrast, the appendix formalises secondary metrics. Given the emphasis on reproducibility and interpretability, formal definitions for the core metrics would be valuable.
2. Ambiguity around revision policy. The framework assumes that models may optionally revise after outcome-based feedback, but does not clarify the normative expectation for how a rational agent should respond to negative feedback. If a model loses, why would not revising be desirable? The paper suggests that high ORR reflects “over-correction,” yet does not specify when persistence is warranted versus not.
3. Limited generalisation to self-verification. The benchmark centres on externally supplied feedback (win/loss signals). It does not explore settings where models must detect and correct their own reasoning failures, a more realistic self-verification regime.

### Questions
1. Clarifying revision policy: Given explicit negative feedback (a loss or rule violation), under what conditions should a model not revise its strategy? How can one distinguish “strategic confidence” from missed adaptation opportunities?
2. Formal definitions: Could the authors provide explicit mathematical formulations for WR, ORR, CSR, β, and OBR similar to those in Appendix B?
3. Extension to self-verification: How might the framework adapt to cases where feedback is internal (e.g., model-generated reflections or verifiers) rather than external outcome signals?
4. The results would benefit from further statistical analysis of the results found in Tables 1 and 2, to convince the reader that the results are sound.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes AdvGameBench, an evaluation framework that tests LLMs' process-level reasoning capabilities through three strategic game environments (tower defense, battle card, and turn-based combat). It innovatively introduces four metrics—Over-Correction Risk Rate (ORR), Correction Success Rate (CSR), Improvement Slope (β), and Over-Budget Rate (OBR)—going beyond traditional win rate evaluation. Through 4,320 adversarial rounds across 12 models, the study finds ChatGPT-o3-mini performs best (74.7% win rate), while high correction frequency negatively correlates with success rate (r=-0.47). The paper is a pure evaluation study without proposing improvement methods, emphasizing that "how decisions are made" matters more than "decision outcomes."

### Strengths
The paper employs game-based environments to evaluate LLMs' reasoning abilities.

### Weaknesses
1. The paper's core conclusions rely on correlation analysis with n=12, but all p-values in Figure 3 exceed 0.05 (ORR vs WR: r=-0.468, p=0.125). While acknowledging "failure to reach significance" on page 7, the paper asserts on page 8 that "precision in revision is the hallmark of effective strategy adjustment," constituting over-interpretation. Expanding to ≥26 models or reframing as exploratory research is necessary; otherwise, it violates basic statistical principles.
2. Key parameters of the three games (11×5 grid, 7-character limit, budget settings) lack theoretical justification. Table 1 shows Gemini-2-Flash achieves only 15.8% in TDG but 49.2% in BCG (3× difference), yet the paper fails to distinguish whether this reflects "capability differences" or "parameter sensitivity." In BCG, Gold characters have a cost-efficiency of 0.67, strictly inferior to Bronze's 1.0, raising design validity concerns. Sensitivity analyses are needed to verify conclusion robustness.
3. The paper provides no human player data, making it impossible to judge whether ChatGPT-o3-mini's 74.7% win rate is excellent or poor, or to verify whether ORR/CSR truly reflect reasoning ability. The tests may measure instruction-following rather than reasoning (e.g., Qwen-Plus's OBR=0.51 could stem from arithmetic errors). Data from at least 20 human players is needed as an anchor; otherwise, performance metrics lack practical meaning.
4. The paper does not propose any methods for improving large language model performance.Retry

### Questions
Why not use games from existing benchmarks for evaluation? For example:
1. TextArena
2. Orak: A Foundational Benchmark for Training and Evaluating LLM Agents on Diverse Video Games

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
This paper presents a concise, process-aware benchmark with three strategic game environments and multi-round traces. Metrics (ORR/CSR/β/OBR) go beyond WR to capture revision discipline and budget compliance. Results indicate that “fewer but higher-quality revisions” and strict budget control correlate with more stable wins, while a contamination-resistant design strengthens validity.

### Strengths
1. The process metrics (ORR/CSR/β/OBR) are really interesting, and they capture behaviors we actually care about in reasoning.
2. Experiments show these metrics carry real signal beyond WR, which is reasonable and convinsing.
3. Reporting both outcome-level and process-level views makes the results more actionable for model selection/ablation.

### Weaknesses
1. No open-source release. Code + the UI demo mentioned in the paper would hugely help reproducibility and adoption.
2. First results table is hard to read. Highlight key trends or add some visualization would be better, just like the radar figure.
3. Model set needs to be updated, like the Qwen2.5/3 series. A refreshed leaderboard would make conclusions more convincing.
4. I can’t tell exactly how game states are serialized into prompts and how actions are parsed back. More illustration and input output examples are needed.
5. “Reasoning ability" is loosely defined. It’s unclear how these game behaviors relate to standard reasoning benchmarks or transfer beyond these environments. A comparison with these models on the general reasoning benchmarks may help.

### Questions
1. Are ORR, CSR, β stable across seeds, prompt variants, and small rule changes? Please report variance or confidence intervals.
2. Do these metrics generalize to other game or planning settings ? Any external validations?
3. How sensitive are WR and other metrics to the chosen opponent? Do we see rank flips under different opponent pools?

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
4

### Summary
The paper proposes to use strategy games as a controlled, rule-based environment to evaluate the reasoning process of large language models (LLMs). The authors propose three types of strategy games that depart from popular existing games to avoid strategy leakage. Beyond a simple outcome-based evaluation metric, authors propose multiple novel process-oriented metrics to evaluate the quality and stability of LLM’s reasoning process. Multiple LLM baselines are implemented into this evaluation framework. Experiment results revealed that high win rates correlate with calibrated revision and resource discipline, not simply frequent corrections.

### Strengths
Using strategy games, a rule-based environment with clear feedback, as the backbone of the evaluation framework, is novel and insightful. Such a framework not only considers the outcome performance but also dynamically evaluates LLM’s reasoning process.

Based on the round-based strategy game, the authors proposed several novel evaluation metrics to evaluate the quality and stability of each LLM’s reasoning process, which are meaningful to understand the performance of a language model more comprehensively.

### Weaknesses
One of the main concerns I have is whether the evaluation results obtained from strategy games can be generalized to real-world reasoning situations. For example, in the round-based strategy game, the reasoning chain might be shorter than real-world cases, where long contextual windows are required to deal with multiple factors and uncertainties. A discussion of the generalization of this work might be helpful. 

The experiment source code/log is not released, which limits the community adoption and reproducibility.

The methodology description, including some key terms, is not clear. 

For example, in line 150, the authors mentioned that “after each round, models receive outcome-based feedback,” but did not describe how such “outcome-based feedback” is generated. 

Another instance is that in line 192, the author mentioned that “CSR measures how often a revision leads to improved results, either by eliminating a rule violation or by turning a loss into a win.” To my understanding, the reward for eliminating a rule violation and turning a loss into a win is different, and it seems inappropriate to be weighted the same in CSR calculation. Moreover, given the nature of a multi-round game, how can one make sure that one single revision is the key factor that turns a loss into a win?

### Questions
see weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
4
