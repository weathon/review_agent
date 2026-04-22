# Beyond Turn Limits: Training Deep Search Agents with Dynamic Context Window

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
While recent advances in reasoning models have demonstrated cognitive behaviors through reinforcement learning, existing approaches struggle to invoke deep reasoning capabilities in multi-turn agents with long-horizon interactions. We propose DeepMiner, a novel framework that elicits such abilities by introducing high-difficulty training tasks and dynamic context window. DeepMiner presents a reverse construction method to generate complex but verifiable question-answer pairs from authentic web sources, which ensures the challenge and reliability of training data while injecting cognitive capabilities into multi-turn reasoning scenarios. We further design an elegant yet effective dynamic context management strategy for both training and inference, utilizing sliding window mechanisms while eliminating the dependency on external summarization models, thereby efficiently empowering the model to handle continuously expanding long-horizon contexts. Through reinforcement learning on Qwen3-32B, we develop DeepMiner-32B, which achieves substantial performance improvements across multiple search agent benchmarks. DeepMiner attains 33.5\% accuracy on BrowseComp-en, surpassing the previous best open-source agent by almost 20 percentage points, and demonstrates consistent improvements on BrowseComp-zh, XBench-DeepSearch, and GAIA. Notably, our dynamic context management enables sustained interactions of nearly 100 turns within standard 32k context length, effectively addressing the context limitations that constrain existing multi-turn interaction systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents DeepMiner, a framework designed to address two core challenges in long-horizon search agents: insufficient training data complexity and context management limitations. The authors introduce a "reverse construction" method to generate complex, verifiable QA pairs from authentic web sources. Concurrently, they design a dynamic context management strategy that uses a sliding window to compress (discard) distant tool outputs while preserving the assistant's reasoning chain, enabling nearly 100 turns of interaction within a standard 32k context length.

### Strengths
- The authors correctly identify that existing multi-hop QA datasets (like HotpotQA) are insufficient for eliciting the deep reasoning, verification, and backtracking abilities required in long-horizon tasks. The proposed "reverse construction" and "obfuscation" strategies are an effective attempt at generating high-difficulty, high-fidelity training data, and the comparison in Table 3 validates the superiority of this data.

-  DeepMiner tackles a highly practical and critical problem in multi-turn search agents: context length limitation. Achieving nearly 100 interaction turns within a 32k window is an accomplishment. The substantial performance leap on `BrowseComp-en` by nearly 20 points demonstrates the combined potential of the new data and framework.

### Weaknesses
- The paper lacks  ablation studies, which leaves its core claim—the effectiveness of the dynamic context window—unsubstantiated. How much of the massive performance boost (e.g., +20% on BrowseComp) is attributable to the superior training data versus the dynamic context window? The authors compare DeepMiner data to HotpotQA data in Table 3, but this only proves the importance of the data.

- The paper is missing a crucial comparison: [DeepMiner Data + Vanilla Context] vs. [DeepMiner Data + Dynamic Context]. Without this ablation, it is impossible to determine if the dynamic context window itself actually enhances reasoning capabilities, or if it merely serves as a tool to enable longer runs, with all capability gains stemming from the DeepMiner dataset.

- The context efficiency analysis in Table 2 is "training-free" and uses a completely different base model (GPT-OSS-120B). This makes its results difficult to correlate with the main training results of DeepMiner-32B (based on Qwen3-32B) and thus cannot serve as supporting evidence for the dynamic window's effectiveness.

- Compared to summarization-based methods, summarization, while lossy, at least retains some signal. This paper's method discards the information entirely. For complex tasks requiring long-range dependencies and fine-grained information retrieval, this "all-or-nothing" dropping strategy could lead to catastrophic failures in the reasoning chain.

### Questions
1. Can authors provide a performance comparison on $BrowseComp$ between [DeepMiner Data + Vanilla Context] (e.g., truncating when the context limit is reached) and [DeepMiner Data + Dynamic Context]? This would be the only direct way to evaluate the contribution of the dynamic context window.

2. Why did authors choose to completely discard tool outputs instead of using a (potentially trainable) summarization module? While summarization adds complexity, discarding information entirely seems like a fundamental constraint for deep research tasks. How does your strategy handle tasks that require backtracking and comparison of early evidence?

    Besides, have you analyzed the failure cases of DeepMiner? How many failures are caused by the model needing an early tool output that was already discarded (e.g., needing content from turn 5 at turn 40)?

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
This paper presents DeepMiner, a training framework for long-horizon, multi-turn reasoning agents. The authors identify two main limitations in current multi-turn systems — insufficient task complexity and ineffective context management. To address these, DeepMiner introduces: (1) a reverse construction pipeline to generate complex, verifiable, multi-source question-answer pairs from authentic web data; (2) a dynamic context window with a sliding-window mechanism to preserve reasoning traces while selectively omitting older tool outputs. The method is implemented on Qwen3-32B, and after reinforcement learning (RL) training, the resulting DeepMiner-32B model achieves large gains on benchmarks such as BrowseComp-en, BrowseComp-zh, and XBench-DeepSearch, outperforming previous open-source agents by nearly 20 percentage points.

### Strengths
1. The paper tackles an important and timely challenge — enabling deep, long-horizon reasoning in open-domain search agents.
2. The proposed sliding-window mechanism offers a practical way to manage growing contexts without relying on external summarization.
3. The paper presents experiments across multiple challenging benchmarks (e.g., BrowseComp, XBench-DeepSearch, GAIA), showing consistent improvements over previous open-source systems.

### Weaknesses
1. The paper claims that tool responses mainly influence only the model' s immediate next decision, but it is unclear how this conclusion was obtained. No quantitative or ablation evidence supports this key assumption.
2. The reward design is overly simple (binary 0/1 correctness). More nuanced signal (e.g., step-wise or process-based reward) could lead to deeper policy learning.
3. The training mechanism that converts a full trajectory into multiple sub-sequences via sliding windows is conceptually interesting but not clearly specified. In particular, the paper does not explain how gradients are propagated across these truncated sequences and how batches are constructed to ensure consistency between local and global optimization.
4. The default sliding window configuration (size=3, step=2) is used throughout all experiments, but the paper does not include an ablation or sensitivity study exploring how different window settings affect performance, stability, or context retention.
5. The reward design relies on an LLM-as-judge evaluation, but the paper lacks discussion about the reliability and theoretical justification of using such models as verifiers in reinforcement learning.

### Questions
The following questions are proposed based on the above weaknesses.
1. Regarding the claim that tool responses mainly influence only the model' s immediate next decision — could the authors elaborate how this conclusion was empirically derived? For example, were ablation or correlation analyses conducted to quantify the short-term versus long-term effects of tool responses on reasoning outcomes?
2. The paper employs a simple binary (0/1) correctness reward. Have the authors explored or considered more granular reward designs?
3. The paper mentions that training trajectories are segmented into multiple sub-sequences via sliding windows. Could the authors explain in detail how gradients are propagated within this setup?
4. Since the default sliding window configuration (size=3, step=2) is fixed in all experiments, have the authors conducted ablation or sensitivity analyses to evaluate how varying window sizes or strides impact performance, stability, or context retention efficiency?
5. As the reward design depends on LLM-as-judge evaluation, could the authors provide theoretical or empirical justification for its reliability? Has the team investigated possible bias, variance, or inconsistency in LLM-judged rewards compared with human or rule-based evaluation?

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
4

### Summary
Authors propose a method for training Deep Research agents. There are two key contributions. The first is data contribution: authors propose a method of synthesizing complex questions with verifiable answers which elicit deep research capabilities. Authors also propose a methodology to compact the context window by maintaining tool results for a recent time window. GRPO is slightly extended to accommodate the fact that each turn depends on a different context from past turns. Compared to some of the previous open-source agents, authors' fine-tuning of Qwen3-32B performs better on BrowseComp and competitive on GAIA and Xbench-DS.

### Strengths
Quality:

The proposed method achieves significance performance improvement on BrowseComp, which is a very challenging benchmark. It is impressive authors' 32B model outperforms DeepSeek V3.1-671B model, which is an order of magnitude larger, on this benchmark. Authors run good ablations for the training method (context management and tool call budget) although there is no ablation on the data synthesis side.

Significance, Originality:

The management of context window across multi-hop trajectories has been explored in the prior work, for example https://arxiv.org/abs/2505.16421 , but to my knowledge, previous methods were more rudimentary and did not discuss its implication with GRPO-style monte carlo advantage estimation. This paper calls out the attention on the importance of context window management with GRPO-style training, which has the potential of being adopted as standard approach in the area.

Clarity:

The paper is clearly written and easy to follow.

### Weaknesses
The idea of synthesizing training data for deep research with progressively evolving questions and fetching web data have been explored in many concurrent work, for example 
Websailor https://arxiv.org/abs/2507.02592 , Webshaper https://arxiv.org/abs/2507.1506 , DeepDive https://arxiv.org/abs/2509.10446 . While authors do refer these papers and compare against results from these papers, it is unclear whether the improvement is due to better synthesis or better context management, and if authors' method of synthesis is any better than previous methods, which design decisions led to how much improvement and why. Without an ablation separating the contribution of reverse-constructed data from dynamic context management, it is difficult to attribute the performance gain to one factor or the other.

While authors share their prompt for quality judgment in the appendix, many details of the data pipeline including exact prompts, question construction logic, and the LLM used to synthesize data are missing. This makes it difficult for other research groups to build upon results of this paper.

### Questions
In Figure 4, the reward and BrowseComp performance seem to be continue improving at the end of the training. Why didn't authors train the model for longer steps?

### Soundness
3

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
This paper presents a framework DeepMiner for training long-horizon LLM agents that can reason and search across extended multi-turn sessions under a fixed context window.
In particular, it introduces a sliding-window mechanism that omits outdated tool outputs while preserving the assistant’s reasoning trace, ensuring training–inference consistency through windowed sequence-level training.
The system is trained in two phases: a SFT stage using reverse-constructed trajectories from authentic web sources, followed by RL via sequence-level training with trajectory-level advantages.

### Strengths
1. Sliding-window design is an effective context-efficiency solution. The method enables around 100 turns within 32 k tokens. Unlike static truncation/summarization, this mechanism scales gracefully with context length and maintains coherence across windows.

2. The empirical motivation is solid with preliminary analysis. Figure 2 clearly shows that without the sliding window, the number of incorrect trajectories increases steeply as context length grows, as tool outputs expand exponentially and squeeze assistant reasoning tokens.

### Weaknesses
1. The reverse-constructed DeepMiner dataset is not open-sourced; details such as question samples, source selection, and verification criteria are missing. Reproduction and fair comparison are thus impossible.

2. The same trajectory-level reward is propagated to all windowed subsequences. This can over-credit early steps and under-credit later corrective reasoning.

3. Missing comparison to related baselines. The paper omits baselines such as Search-R1 [1] and Synthetic Data RL [2], both using RL for multi-hop QA.

4. Limited Ablations and Hyperparameter Studies. No analysis of different window sizes, sliding size, and base model sizes.

[1] Jin, Bowen, et al. "Search-r1: Training llms to reason and leverage search engines with reinforcement learning." arXiv preprint arXiv:2503.09516 (2025).

[2] Guo, Yiduo, et al. "Synthetic Data RL: Task Definition Is All You Need." arXiv preprint arXiv:2505.17063 (2025).

### Questions
1. Why were Search-R1 and Synthetic Data RL omitted as baselines, given their direct methodological overlap (RL training on hard-QA dataset)?

2. In Figure 4, the reward and performance curves show persistent fluctuations without clear convergence, even after 80 training steps. Could the authors further explain the missing training steps results?

### Soundness
2

### Presentation
3

### Contribution
2
