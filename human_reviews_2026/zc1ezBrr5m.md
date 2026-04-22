# TaTToo: Tool-Grounded Thinking PRM for Test-Time Scaling in Tabular Reasoning

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 8, 4

## Abstract
Process Reward Models (PRMs) have recently emerged as a powerful framework for enhancing the reasoning capabilities of large reasoning models (LRMs), particularly in the context of test-time scaling (TTS). However, their potential for supervising LRMs on tabular reasoning domains remains underexplored. 
Through detailed empirical analyses, we identify that existing PRMs, though widely adopted for supervising text-only reasoning steps, struggle with table-specific operations such as sub-table retrieval and schema interaction, leading to critical performance bottlenecks. To address this limitation, we propose TaTToo, a novel table-grounded PRM framework that (i) reasons explicitly over tabular reasoning steps and (ii) integrates tool-based verification to provide precise reward supervision. Concretely, we first design a scalable data curation pipeline that constructs over 60k high-quality step-level annotations by integrating table verification rationales with tool-based executions. Building on the collected data, we train TaTToo with a dual-stage paradigm: cold-start supervised fine-tuning to capture tool-use reasoning patterns, followed by reinforcement learning with tool-grounded reward shaping to align our model with table-based verification.
We provide a comprehensive evaluation of the policy improvement induced by our newly designed PRM.
Across 5 challenging tabular reasoning benchmarks covering numerical reasoning, fact-checking, and data analysis, TaTToo improves downstream policy LRMs by 30.9\% at inference, surpasses strong PRM baselines such as Qwen-2.5-Math-PRM-72B with only 8B parameters, and demonstrates strong generalizability across diverse TTS strategies.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces TaTToo (Tool-Grounded Thinking PRM), a Process Reward Model (PRM) framework specifically designed for Test-Time Scaling (TTS) in the domain of tabular reasoning. The authors empirically demonstrate that existing PRMs, which are widely adopted for supervising text-only reasoning steps, struggle with table-specific operations such as sub-table retrieval and schema interaction. To address this, TaTToo proposes a novel Tool-Grounded thinking process to finely supervise Large Reasoning Models (LRMs) during inference on structured data. The work leverages Reinforcement Learning (RL) and Reward Shaping to enhance the PRM's ability to capture fine-grained tabular reasoning signals, highlighting the significant potential of PRMs in structured reasoning domains like fact-checking, scientific analysis, and decision support.

### Strengths
S1: The paper is highly original, successfully extending the advanced PRM paradigm to the complex, structured domain of tabular reasoning. It precisely identifies the core limitation—the inability of current PRMs to handle table-specific operations—and provides a critical solution with the TaTToo framework. This is a vital step toward improving the reliability of LLMs on structured data.
S2: The methodology is sophisticated, employing Reinforcement Learning and Reward Shaping to effectively capture and optimize fine-grained signals of the reasoning process, demonstrating high-quality research execution.
S3: The paper is clear in articulating the problem, the proposed method, and its potential limitations. Method details (e.g., data pipeline) are well-described.

### Weaknesses
W1: The paper acknowledges the significant computational overhead from the RL stage, which may "hinder reproducibility and accessibility in low-resource environments." The authors should provide a deeper discussion and specific efficiency analysis in the main body (e.g., training time/cost comparison to SFT-Only models). 
W2: Shallow comparison with recent tabular reasoning RL methods (e.g., Table-R1 series), only mentioned as related work.
W3: A critical risk, as highlighted in the ethics section, is that reliance on automated verification could amplify errors if the underlying tools or training data are flawed. The authors should include experiments to measure TaTToo's robustness against tool errors (e.g., errors in SQL compilers/executors) or noisy training data, and propose concrete, preliminary mechanisms for auditing verifier reliability.

### Questions
Q1:How do tool execution errors (e.g., code runtime exceptions) impact PRM reward supervision? The paper integrates tool outputs into rationales but lacks error handling mechanisms.
​​Q2:The paper notes schema interaction steps fail due to locality bias, and TATTOO uses table prefixes as a fix. Does this scale to very long reasoning chains (e.g., >10 steps)? Are there plans for dynamic prefix mechanisms?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper addresses a critical issue that existing Process Reward Models (PRMs) struggle with table-specific operations and result in performance bottelnecks in tabular reasoning. To this end, this paper proposes TaTToo, a novel table-grounded PRM framework that provides precise reward supervision in reasoning steps. Valuable contributions contain the large-scale high-quality step-level annotations and tool-grounded dual-stage training paradigm involving table-aware supervised fine-tuning and RL training with step-level reward shaping. Experimental analysis demonstrates the effectiveness and generalizability of TaTToo.

### Strengths
1. This paper is very well organized and presented, integrating clear and logical descriptions of motivations, pre-analysis, theoretical analysis and experimental validation. I am very impressed to read this manuscript and learn much from it. 

2. The motivation of addressing step-level supervision in tabular reasoning is critical and impressive as a very novel contribution, as the limitations of previous LRMs have been demonstrated in the pre-analysis section. 

3. The design of the RL reward integrates multiple reasonable components, and the policy improvement has been validated by robust theoretical analysis.

4. Experimental results clearly show the effectiveness and generalizability of TaTToo, revealing the contributions of the key modules of TaTToo.

### Weaknesses
1. TaTToo is trained based on Qwen3-8B and outperforms previous verifers with larger model sizes. I am curious about the result of the setting that utilizes the raw Qwen3-8B-model without further training, yet this setting is missing.

2. As TaTToo integrates large-scale dual-stage training within TTS scenarios, the computational overhead may be significant. Is there some possible solutions to further improve the efficiency?

### Questions
See the weaknesses part.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a process reward model (PRM) named TATTOO to address the common failure modes in current RMs, like subtable retrieval, schema/column interaction, and so on. The authors build a dataset of 60k+ step-level annotations and design a two-stage training pipeline (SFT+RL) for the better PRM. Specifically, they propose a GRPO variant with label matching, confidence calibration, tool grounding as shaping signals to optimize stepwise validation and scoring. Experiments spanning five tasks show that TATTOO yields monotonic, non-saturating gains as N increases under Best-of-N, beam search, and DVTS, and surpasses strong baselines like Qwen2.5-Math-PRM-72B and GenPRM-32B.

### Strengths
1. Re-diagnoses PRM failure from table-specific steps, clearly separating error sources into retrieval and schema interaction. The “table prefix” study shows existing PRMs’ insensitivity to distant retrieval context, motivating table-aware rewards + tool execution for stepwise verification.
2. End-to-end data pipeline: from expert LRM trajectory pools and dual validation filters to replacing subjective heuristics with executable tools for verification.
3. Introduces tool-supported reward shaping (label matching, confidence calibration, tool grounding) that explicitly rewards using tool outputs inside verification rationales—distinct from prior PRMs that rely on purely textual discrimination/generation.

### Weaknesses
The main weaknesses are in experiment settings. Addressing these would substantially strengthen the paper.
1. The competitive baseline about outcome reward model is absent. And I wonder how an outcome reward model trained on your training samples (which are surely a solid, data-aspect contribution) performs.
2. The policy model in experiments is Distilled-Qwen-14B while the reward model is trained from Qwen3-8B (weak policy model + strong reward model). And I wonder if TATTOO still works on strong policy models such as Qwen3-series (like Qwen3-32B, Qwen3-30B-A3B) or gpt-oss.

### Questions
Please see the weaknesses above

### Soundness
3

### Presentation
3

### Contribution
3
