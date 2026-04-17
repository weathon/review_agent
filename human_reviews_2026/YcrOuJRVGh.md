# ReSyn: Autonomously Scaling Synthetic Environments for Reasoning Models

- Decision: Reject
- Scores: 2, 4, 2

## Abstract
Reinforcement learning with verifiable rewards (RLVR) has emerged as a promising approach for training reasoning language models (RLMs) by leveraging supervision from verifiers. Although verifier implementation is easier than solution annotation for many tasks, existing synthetic data generation methods remain largely solution-centric, while verifier-based methods rely on a few hand-crafted procedural environments. In this work, we scale RLVR by introducing ReSyn, a pipeline that generates diverse reasoning environments equipped with instance generators and verifiers, covering tasks such as constraint satisfaction, algorithmic puzzles, and spatial reasoning. A Qwen2.5-7B-Instruct model trained with RL on ReSyn data achieves consistent gains across reasoning benchmarks and out-of-domain math benchmarks, including a 27% relative improvement on the challenging BBEH benchmark. Ablations show that verifier-based supervision and increased task diversity both contribute significantly, providing empirical evidence that generating reasoning environments at scale can enhance general reasoning abilities in RLMs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces RESYN, a scalable pipeline that autonomously generates diverse reasoning environments, where each environment includes a problem generator and a code-based verifier to enable training via reinforcement learning. By leveraging the generator-verifier gap and scaling task diversity, a model trained on RESYN data achieves significant improvements on reasoning benchmarks, including a 27% relative gain on the challenging BBEH benchmark.

### Strengths
1. The target of this paper for automatically constructing tasks for training LLMs is critical.
2. The proposed method achieves improved performance compared to baselines.

### Weaknesses
1. The paper claims the proposed method achieves OOD improvement on Big-Bench Hard (BBH). However, in section 2.2, the paper uses subtasks of BBH as input for constructing the training dataset. This may lead to data leakage, weakening the claim of OOD evaluation.
2. The diversity of environments is not validated. The pipeline of this framework fully relies on Claude 3.5 to evaluate and filter the generated environments, and generate from 100 keywords. This process may incur mode collapse, which means the 418 tasks share similar logic or patterns. Therefore, the diversity of tasks is needed to validate and analyze.
3. The proposed method only reports 418 survived environments, but does not provide the pipeline or method for how many other environments are filtered out and why they are filtered.
4. The claim of generalization to math benchmarks (like GSM8K) is invalid. The paper claims this is an out-of-domain gain. However, the seed keyword list in Appendix A.1, used to generate the environments, is heavily populated with explicit mathematical and algorithmic concepts such as "Number Theory," "Math Operations," "Dynamic Programming," and "Combinatorial Optimization". Therefore, the strong performance on GSM8K is an expected "in-domain" transfer from synthetic algorithmic tasks, not a proof of general reasoning transfer.

### Questions
See weaknesses.

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
5

### Summary
This paper proposes generating diverse environments using large language models (LLMs) instead of reasoning traces, along with verifiers implemented in code to create question–verifier (Q, V) pairs. The core idea is that verifier-based supervision offers stronger learning signals than solution-based data. Unlike previous work that relies on hand-picked problems, this method automatically generates Q, V pairs via LLMs.

### Strengths
1. The approach demonstrates impressive improvements (Table 4) over both the instruct model and baselines across four datasets.
2. The dataset generation strategy integrates multiple filtering mechanisms and appears methodologically sound.
3. The framework allows dynamically generated tasks rather than fixed ones as in prior work, which contributes to performance gains.

### Weaknesses
1. There is substantial variance when the number of environments or tasks changes. It remains unclear whether the upward scaling trend holds beyond 400 environments, as performance sometimes falls below prior work (e.g., on BBH).
2. The set of baselines is limited. Only **SynLogic** (Liu et al.) is included. The authors should also compare against **TinyZero** (Pan et al.), **Logic-RL** (Xi et al.), and **Synthetic Data RL** (Guo et al.) to contextualize performance more broadly. Another possible candidate: https://arxiv.org/abs/2505.24760 (NeurIPS 2025)
3. The advantages of the proposed method over **Synthetic Data RL** (Guo et al.) are not mentioned. Since Synthetic Data RL can also scale with many task definitions, a direct comparison would strengthen the paper’s claims.
4. The effect of the RL training algorithm beyond **DAPO** is unexplored. As prior work (e.g., Liu et al.) uses **GRPO**, it is unclear whether improvements stem from the algorithm or from the proposed synthetic data.

### Questions
See Weaknesses above.

### Soundness
3

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
4

### Summary
This paper proposes ReSyn, a pipeline that automatically generates synthetic reasoning environments with code-based verifiers and instance generators. The authors argue this approach can scale “reasoning-focused” reinforcement learning with verifiable rewards (RLVR) beyond a small number of handcrafted tasks.

### Strengths
- The paper is clearly motivated by the recent trends in RLVR.

- The idea of combining automatic environment synthesis with code-based verifiers is conceptually appealing.

- The authors include ablations on verifier vs. answer-based supervision and task vs. instance scaling.

### Weaknesses
1. **Lack of experimental rigor and details.** The experimental section omits key information—how exactly were the questions and environments generated from keywords, what proportion were filtered out, and how many survived each stage? The authors mention using “Claude 3.5 Sonnet v2” but do not provide prompts, seed examples, or reproducibility details. It is also unclear whether any existing datasets (e.g., BBH templates) were reused or rephrased.


2. **Minimal improvement from baselines.** The reported gains are modest. For instance, the BBEH improvement from 11.2 → 14.3 is still near chance level. There is no comparison with strong recent baselines such as R1-Zero-like methods, or self-play verifiers.

3. **Unclear generalizability.** The environments appear narrow—mostly code-style puzzles or rule-based tasks (Appendix A.1). It remains unclear whether the learned skills transfer to open-ended reasoning tasks such as commonsense or natural-language reasoning. No experiments were provided on such domains.

4. **Missing explanation of keyword and environment generation quality.** The paper briefly states that LLMs were prompted with “keywords from BBH/KOR-Bench” but provides no justification of why these keywords cover diverse reasoning types or whether the generated tasks are semantically distinct. The role of the LLM judge is also under-specified.

### Questions
1. How were the ~100 keywords selected and filtered? Was this list hand-curated?
2. Did any of the generated tasks closely resemble existing datasets (e.g., GSM8K templates)?
3. How were diversity and correctness of environments quantitatively measured?

### Soundness
1

### Presentation
1

### Contribution
2
