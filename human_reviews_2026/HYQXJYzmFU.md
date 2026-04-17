# SWE-Ext: Extending and Scaling Augmented Data for Repository-Level Coding Tasks

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 6, 2, 2

## Abstract
Repository-level benchmarks such as SWE-Bench have highlighted the challenges of scaling language models to complex software engineering tasks. However, current training data remains narrow in scope, primarily focusing on monolingual issue resolving and feature implementation. In this work, we introduce SWE-Ext, a large-scale effort to extend and scale augmented data for repository-level coding tasks. SWE-Ext broadens existing data along two key dimensions: multilingual coverage (spanning 10 languages) and an auxiliary code completion task. We uncover distinct transfer mechanisms: data from other programming languages provides transferable signals that generally enhance localization and editing capabilities in single-language (Python) settings, while code completion data strengthens code editing capabilities, particularly for feature implementation tasks requiring substantial new code generation. These extensions yield consistent improvements on Python repository-level benchmarks like SWE-Bench and FEA-Bench. Our method offers a simple yet effective way to leverage more open-source data for advancing repository-level code models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
SWE-Ext proposes a scalable pipeline to extend augmented training data for repository-level coding beyond the current Python-only issue-fixing focus by adding multilingual pull-request data (10 languages) and an auxiliary code-completion task. The dataset factorizes supervision along four stages of an agentless pipeline: file localization, component localization, code editing, and code completion. GPT-4o is leveraged as the expert model to synthesize training data. Hints from groundtruth are provided when the expert model fails. Experiments show consistent downstream gains on Python-only benchmarks (SWE-Bench, FEA-Bench). In particular, training on multilingual and completion data both enhances performance on Python tasks. Scaling analysis shows predictable gains across model sizes (7B→32B) and data volume when applying these extensions.

### Strengths
* SWE-Ext presents a scalable way of synthesizing training data for software engineering tasks, by leveraging supervision from groundtruth to steer the expert model to generate correct samples. The method does not rely on runtime environments, significantly reducing the cost.
* Experiments demonstrate clear gains on SWE-Bench Verified and FEA-Bench. Ablations show that training on multilingual or code completion data can both improve the performance on python-only SWE tasks, proving that those are both promising approaches for data scaling.
* The paper is written clearly and easy to follow.

### Weaknesses
* Existing data augmentation frameworks with executable environments (e.g., swe-gym, swe-smith) are generally scaffold-agnostic, in the sense that one can recreate training trajectories for any agent scaffold by running rejection sampling. However, I do not see a straightforward way to redo the exercise of SWE-Ext for a different agent, for example, OpenHands, as it may require a separate design to inject hints.

* The use of hints in the form of groundtruth leakage may hurt data quality. In some cases where LLMs do not understand the full path towards the correct final answer, they may hallucinate the intermediate reasoning steps. Did you observe such cases, and how were they handled?

### Questions
* What motivates you to choose an agentless scaffold for this work instead of an end-to-end multi-turn agent with tools?
* Is the $P$ in L077 meant to be $L$?
* L144 seems missing an "and".

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
2

### Summary
This paper proposes SWE-Ext, a framework for extending repository-level datasets by adding multilingual and code-completion data. It decomposes software engineering tasks from GitHub pull requests into subtasks (localization, editing, completion) and uses GPT-4o as a data generation model to fine-tune Qwen2.5-Coder models. Experiments show consistent but moderate improvements on SWE-Bench and FEA-Bench benchmarks.

### Strengths
The paper proposes a well-structured and scalable data collection process for repository-level code tasks. It leverages real GitHub PRs, ensuring realism and high coverage.

Introducing multilingual and completion-based augmentation is original in the context of SWE-Bench-like setups, addressing the monolingual bias of prior datasets (mostly Python).

SWE-Ext offers a potentially reusable dataset and pipeline for training and benchmarking LLMs in real-world software engineering contexts, without requiring execution-based data collection.

### Weaknesses
The dataset relies entirely on augmented supervision from PRs and expert model annotations, without verification through execution or testing (unlike SWE-Gym or R2E). This limits the data’s reliability and may include noisy or incomplete samples.

The reported improvements (+1–2% on SWE-Bench, +2.5% on FEA-Bench) are modest relative to the scale of data expansion. The paper does not fully analyze the efficiency trade-off versus dataset size.

Despite the multilingual data generation, benchmarks are only in Python, so cross-lingual generalization is inferred rather than directly measured.

The manuscript could benefit from clearer explanation in the experiments section, especially the description of the CosAgentless system and training configurations.

### Questions
Have you tested zero-shot performance on non-Python repository-level tasks to confirm actual multilingual transfer rather than auxiliary effects?

How do you filter out low-quality PRs or synthetic reasoning errors from the GPT-4o expert outputs? Any human validation?

Could SWE-Ext be combined with verification methods like SWE-Gym or SWE-Smith to yield hybrid supervision?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes two data augmentation methods, multilingualism and additional code completion tasks, for SWE-Bench-like tasks. Based on these two augmentation methods, this paper proposes a new training dataset, SWE-Ext. Trained on SWE-Ext, models perfoms well on SWE-Bench and FEA-Bench.

### Strengths
1. The studied topic is valuable.
2. The paper is well-written and easy to follow.
3. Training with the proposed SWE-Ext, the performance increases.

### Weaknesses
1. The experiments are not comprehensive. There are three experimental settings in the paper: 1) without augmentation 2) with multilingual 3) with code completion. Though both 2) and 3) are shown to have a positive influence on performance, there are no experimental results of the combination of 2) and 3).
2. The two proposed augmentation methods are quite common in coding-related tasks, which limits the novelty of the paper.
3. The authors only train Qwen series LLMs with SWE-Ext. More results from other LLMs are needed to demonstrate the general benefit of SWE-Ext.
4. For FEA-Bench, the authors only compare their models with general LLMs. More results from specific coding LLMs are needed for a more comprehensive evaluation.
5. I don't agree that the experimental settings and results support the claim that multilingualism augments LLM training. The further improvement of performance may not come from the multilingualism of the data, but rather from the diversity of the data. A more rigorous setting is to translate exactly the same python projects to other programming languages.

### Questions
Please see my comments above.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces SWE-Ext, a pipeline for constructing training data for repository-level coding tasks. The authors extend existing approaches along two key dimensions: multilingual coverage (spanning 10 programming languages) and an auxiliary code completion task. The pipeline processes GitHub pull requests to create four complementary datasets targeting file localization, component localization, code editing, and code completion. Through experiments on SWE-Bench and FEA-Bench using Qwen2.5-Coder models (7B and 32B), the authors demonstrate that multilingual data improves localization and overall performance, while code completion data primarily enhances editing capabilities.

### Strengths
+ Collecting training data to solve repository-level SWE tasks is an important topic.

### Weaknesses
**Unsatisfactory Performance compared to closely related work**

The best trained SWE-Ext-32B models report 32.6% on SWE-Bench Verified, which is notably worse than the most directly comparable baselines: SWE-Smith-32B (40.2%) and R2E-Gym-32B (34.4%). These two methods are both training data construction approaches using the same 32B model scale, making them the most relevant comparisons for evaluating SWE-Ext's data augmentation strategy. The performance gap of 7.6% and 1.8% respectively suggests that SWE-ext may be less effective than existing data construction techniques, such as SWE-smith and R2E-Gym.

**Missing some 32B baselines**

The paper misses recent strong 32B baselines that have demonstrated significantly better performance on SWE-Bench Verified. Notably, DeepSWE [1] and SWE-Swiss [2] report substantially higher performance on the same benchmark. Without comparing against these strong baselines, it is difficult to assess where SWE-Ext stands relative to the strong baselines in swe-bench at similar model scales.

[1] DeepSWE: https://www.together.ai/blog/deepswe

[2] SWE-Swiss: https://github.com/zhenyuhe00/SWE-Swiss/tree/main

**Lack of controlled apple-to-apple comparison.**

Importantly, as a training data construction work, the paper fails to provide an apple-to-apple comparison that isolates the effectiveness of the data itself. The comparisons with SWE-Smith, SWE-Gym, and R2E-Gym involve different base models (e.g., SWE-Smith uses different expert models for data generation), different agent scaffolds (OpenHands vs. Agentless vs. CosAgentless), and different system configurations. To truly validate the contribution of SWE-Ext's multilingual and completion-based data augmentation, the authors should conduct controlled experiments where only the training data varies while keeping the base model, agent system, and inference setup constant. For instance, training on SWE-Smith's data versus SWE-Ext's data using the same Qwen2.5-Coder-32B model and CosAgentless system would provide a direct assessment of data quality. Without such controlled comparisons, it remains unclear whether the observed performance differences stem from superior data construction, better agent design, or simply different experimental configurations.

**Marginal Improvements from Multilingual and Code Completion Extensions**

The highlighted contributions of extending data to multiple languages and incorporating code completion tasks yield only marginal improvements that raise questions about their practical value. The multilingual extension provides only +1.4% improvement (31.2% → 32.6%) on SWE-Bench. Similarly, the code completion extension shows limited impact with only +1.0% improvement (31.2% → 32.2%) on SWE-Bench Verified, and notably comes at the cost of degraded localization performance (component Hit@1 drops from 55.0% to 52.8% in Table 4), suggesting misalignment with the full repository-level coding pipeline. The paper lacks critical analysis on whether combining both extensions yields additive improvements, whether there are diminishing returns, or whether simply collecting more high-quality Python issue-resolution data would be more efficient.

**Static Data Collection Misaligned with Dynamic Agent Paradigms and Lacks Practical Extensibility**

The paper's purely static data collection approach is fundamentally misaligned with the dynamic, interactive nature of modern LLM-based coding agents, limiting both its effectiveness and practical applicability. Current state-of-the-art coding agents rely heavily on iterative interactions with execution environments, incorporating feedback from test runs, linting errors, compilation failures, and runtime observations to guide code modifications. In contrast, SWE-Ext constructs training data entirely from static GitHub pull requests without any execution-based verification or tool-using, essentially restrict the training recipe from dynamic agentic capabilities. 

Moreover, it seems practically infeasible to extend SWE-ext with dynamic agentic trajectories, since it does not provide unified configuration interface for too much varied repositores that span across multiple languages. This restriction makes the value of SWE-ext even more downgraded.

### Questions
- Could you provide an apple-to-apple comparison where you train models on SWE-Smith's data, R2E-Gym's data, and SWE-Ext's data using the exact same base model, agent scaffold, and inference configuration? This would isolate the contribution of your data construction approach from confounding factors like different agent scaffolds or model choices.

- Could the authors add comparison with DeepSWE and SWE-Swiss?

- Could the author explain why they focus on static trajectories without any dynamic interactions and tool using?

### Soundness
1

### Presentation
2

### Contribution
1
