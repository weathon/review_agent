# Fortune: Formula-Driven Reinforcement Learning for Symbolic Table Reasoning in Language Models

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4

## Abstract
Tables are a fundamental structure for organizing and analyzing data, making effective table understanding a critical capability for intelligent systems. While large language models (LMs) demonstrate strong general reasoning abilities, they continue to struggle with accurate numerical or symbolic reasoning over tabular data, especially in complex scenarios. Spreadsheet formulas provide a powerful and expressive medium for representing executable symbolic operations, encoding rich reasoning patterns that remain largely underutilized. In this paper, we propose Formula Tuning (Fortune), a reinforcement learning (RL) framework that trains LMs to generate executable spreadsheet formulas for question answering over general tabular data. Formula Tuning reduces the reliance on supervised formula annotations by using binary answer correctness as a reward signal, guiding the model to learn formula derivation through reasoning. We provide a theoretical analysis of its advantages and demonstrate its effectiveness through extensive experiments on seven table reasoning benchmarks. Formula Tuning substantially enhances LM performance, particularly on multi-step numerical and symbolic reasoning tasks, enabling a 7B model to outperform OpenAI o1 on table understanding. Beyond empirical gains, we present several insights into the role of RL in symbolic table reasoning, highlighting the broader potential of formula-driven RL to advance reasoning capabilities in LMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces **FORTUNE**, a reinforcement learning (RL) framework for training large language models (LLMs) to generate **spreadsheet formulas** as an explicit symbolic reasoning mechanism for table-based question answering. Instead of relying on annotated for-mulas, FORTUNE uses **binary answer correctness** as the reward signal, teaching the model to derive executable formulas that yield correct results when evaluated. The authors pro-vide **theoretical justifications** showing the superiority of symbolic over textual reasoning and RL over supervised fine-tuning (SFT), followed by **experiments** across seven table reasoning benchmarks. Results indicate substantial gains over both SFT and prior state-of-the-art (SOTA) systems, with the FORTUNE++ variant (combining symbolic and textual reason-ing) outperforming larger closed-source models such as OpenAI o1.

### Strengths
1. **Novel conceptual framing of formula-driven RL**
   The idea of representing symbolic table reasoning through spreadsheet formulas, rather than the more typical SQL or Python program synthesis, is both novel and practically motivated. The paper convincingly argues that formulas provide a more lightweight and accessible sym-bolic interface, with empirical results validating this claim.

2. **Comprehensive empirical evaluation**
   Results span seven datasets (WikiTQ, TabFact, HiTab, FinQA, MultiHiertt, AIT-QA, TableBench), covering both in-distribution and out-of-distribution settings. The comparisons are thorough—against prompting, supervised, and hybrid baselines—and the improvements are both consistent and meaningful.

3. **Clear analysis and ablation**
   The paper goes beyond headline results to provide analysis of textual vs symbolic reasoning, the effects of RL vs SFT, and the relative strengths of SQL, Python, and formula-based reason-ing. These comparative results lend strong credibility to the claims.

4. **Strong empirical impact**
   Achieving performance that surpasses OpenAI o1 with a 7B open model is a compelling re-sult and likely to attract interest from both academia and industry.

### Weaknesses
1. **Limited novelty in RL methodology**
   The reinforcement learning setup is a relatively standard application of PPO with scalar cor-rectness reward. While effective, the technical innovation on the RL side is incremental. The contribution is more conceptual (using formulas as the medium) than algorithmic.

2. **Limited theoretical depth**
   The two presented theorems (symbolic ≥ textual, RL ≥ SFT) are intuitive restatements of re-ward-maximization principles rather than new theoretical insights. Proofs lack quantitative bounds or convergence guarantees and do not address RL instability or reward-variance issues.

3. **Spreadsheet-formula limitations underexplored**
   While the paper highlights the flexibility of formulas, it underplays practical drawbacks—limited scalability on large tables and potential maintainability issues compared to structured systems like SQL/Pandas.

4. **Reward design and training stability**
   The reward structure (1 / 0.2 / 0) is coarse and heuristic, with no reported analysis of training stability, reward variance, or convergence behavior. It remains unclear how robust FORTUNE is to sparse or noisy rewards and whether different reward formulations would yield similar improvements.

5. **Experimental comparability**
   Some baselines (e.g., TabAF, TableGPT) may differ in data scale or compute budgets, mak-ing it unclear whether observed improvements stem purely from reinforcement learning or broader training differences. The paper reports its own configurations but cannot guarantee parity across external baselines.

### Questions
1. **Reward Sensitivity:**
   Have you experimented with alternative reward functions (e.g., graded rewards based on par-tial correctness or token-level formula similarity)? How sensitive is FORTUNE’s performance to the specific (1 / 0.2 / 0) reward scheme? Also, why did you choose PPO over newer RL variants such as GRPO or REINFORCE++?

2. **Training Stability:**
   Did you observe instability, mode collapse, or reward hacking during RL optimization? How did you ensure convergence, given the sparse binary reward?

3. **Fairness of Comparison:**
   Can you clarify whether FORTUNE and baseline models were trained on comparable data scales and compute resources? Were all models fine-tuned on the same merged corpus de-scribed in Section 4.1?

4. **Quantitative Effect of RL:**
   Beyond accuracy, do you track executability rate, average formula length, or entropy to show how RL qualitatively changes formula generation?

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper propose a LLM framework that firstly use symbolic reasoning in reinforcement learning. By using the correctness of formula execution results as the reward signal, the framework reduces dependence on supervised formula annotations, guiding models to learn formula derivation through reasoning.

### Strengths
- This paper is the first to use symbolic reasoning in RL and gains huge enhancement.
- The tabular dataset used is sufficiently large in scale and the experimental setup is detailed, effective and complete enough.
- The proofs in appendix is detailed.

- This paper is the first to integrate symbolic reasoning (via spreadsheet formulas) with RL for table reasoning tasks, yielding significant performance improvements.
- The tabular datasets employed in this study are comprehensive, covering 7 diverse table reasoning benchmarks (e.g., WikiTQ, TabFact, FinQA). Moreover, the experimental setup is detailed and rigorous—including clear descriptions of model backbones, training protocols and evaluation metrics to ensure the reproducibility and validity of the results.
- The theoretical proofs presented in the appendix are comprehensive and detailed.

### Weaknesses
- Robustness is one of the key features in LLMs. However, this paper don't design experiments for this feature. When it comes to real world tables like NAN or nil data, it's important to know the results.
- While the paper reports performance under zero-shot, supervised fine-tuning, and RL settings, the evaluations are limited to unimodal LLMs. With the rapid advancement of multimodal large language models which can process tabular data alongside other modalities (e.g., image charts, textual descriptions in table captions)—the paper fails to extend its scope to MLLMs. This omission limits the framework’s generalizability to increasingly common multimodal table understanding tasks.

### Questions
- Symbolic methods are widely used in TableQA. For instance, GSM8K, a well-known mathematical reasoning dataset, can be converted into tabular formats to test table-based numerical reasoning. Given this, if the proposed Formula Tuning framework directly adopts formula-based parameterization instead of relying on LM-generated formula exploration, would this constitute a more effective modeling approach? If so, how might it impact the framework’s ability to handle complex multi-step reasoning tasks?
- For zero-shot reasoning, GPT-series models (e.g., GPT-4o, GPT-4o-mini) exhibit notably stronger performance compared to open-source models like Qwen2.5-Coder7B. First, could the authors explain the potential reasons for this significant zero-shot gap? Second, have the authors tested other state-of-the-art models such as Gemini in zero-shot settings to further validate whether the observed performance trends are specific to GPT-series models? Third, including more powerful models in the baseline comparisons would help better contextualize the superiority of the proposed FT framework. Are there plans to supplement such experiments?

### Soundness
3

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
This paper introduces a training framework called Fortune to enhance reinforcement learning prompt models' ability to process tabular data. By having the model output executable formula outputs and using executability and correctness as reward functions, the model's tabular data processing capabilities significantly improve after reinforcement learning.

### Strengths
The proposed training method enables a 7B model to achieve strong tabular data processing capabilities, outperforming commercial models on certain datasets.

### Weaknesses
Support for tabular operations is insufficient; features like sorting should be added to better align with practical software like Excel. The specific implementation of the Formula Executor lacks description. Exploration of OOD (Operations Over Data) symbols is missing.

### Questions
1, Is there a specific reason for setting executable but incorrect results to 0.2 in the reward function? 
2, A description of the Formula Executor's concrete implementation should be included. 
3, It is recommended to test the model's performance before and after encountering data requiring operations not included in training to demonstrate its generalization ability.

### Soundness
2

### Presentation
2

### Contribution
2
