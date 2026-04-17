# WebShaper: Agentically Data Synthesizing via Information-Seeking Formalization

- Decision: Accept (Poster)
- Scores: 4, 6, 8, 6

## Abstract
The advent of Large Language Model (LLM)-powered agents has revolutionized artificial intelligence by enabling solutions to complex, open-ended tasks through web-based information-seeking (IS) capabilities. The scarcity of high-quality training data has limited the development of IS agents.  Existing data synthesis approaches typically adopt an information-driven paradigm that first collects information and then refines question-answer pairs through retrieval. However, this may lead to inconsistency between information structure and reasoning structure, as well as between the question and the corresponding answer. To mitigate, we propose a formalization-driven IS data synthesis framework WebShaper, which systematically formalizes IS tasks using set-theoretic constructs.
Central to the formalization is the concept of Knowledge Projections (KP), which enables precise control over reasoning structure by KP operation compositions.  During synthesis, we begin by creating seed tasks, then use a multi-step expansion process.
At each step, an agentic Expander expands the current formal question more complex through retrieval and validation tools grounded in our formalization. We train our model on the synthesized dataset. Experiment results demonstrate that WebShaper achieves state-of-the-art performance among open-sourced IS agents on competitive benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces WebShaper, a formalization-driven framework for synthesizing training data for information-seeking agents. Unlike existing information-driven methods such as WebDancer, it defines IS tasks through a set-theoretic formalization based on knowledge projections, and uses these to guide the synthesis process. A key innovation is the Expander agent, which autonomously performs multi-step, layer-wise expansions of seed tasks by retrieving and validating information from the Web, ensuring consistency between reasoning and information structures.

### Strengths
1.	The paper introduces a set-based formalization of IS tasks with knowledge projections and compositional operations (Union, Intersection), which provides a clear structure for reasoning.

2.	The Expander module is well-motivated and technically detailed, by embedding the Thought–Action–Observation loops and specialized tools including Search, Summarize, and Validate.

3.	The experiments are comprehensive, showing consistent improvement across model scales and benchmarks.

### Weaknesses
1.	The paper is not self-contained enough, with multiple components (e.g., the WebDancer framework, the QwQ model) depending on specific infrastructures, which limits understanding of general audience and reproducibility. 

2.	The paper’s notation-heavy exposition, especially in Section 2, hinders accessibility for non-expert readers. The figures and tables, though informative, are dense and underexplained, and the text occasionally assumes familiarity with prior systems like WebDancer.

3. The paper does not discuss potential risks in synthetic data generation (e.g., bias propagation, fact inconsistency, or data contamination). An explicit acknowledgment of these issues and their mitigation strategies would strengthen the credibility and responsibility of the work.

### Questions
1.	How robust is the KP formalization to noisy or incomplete retrieval results? Does (or to what extent) the model degrade under imperfect web data?

2.	Could you provide a quantitative evaluation of synthesized question quality (e.g., factual accuracy or diversity)?

3.	To what extent can WebShaper be generalized to non-English or multimodal information-seeking tasks?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the lack of training data for information-seeking agents by proposing WebShaper—a formalization-driven data synthesis framework that formalizes search tasks within a set-theoretic framework. WebShaper introduces Knowledge Projection as the fundamental unit to represent complex IS tasks systematically. During data synthesis, WebShaper employs a layer-wise expansion strategy and uses an Expander agent to iteratively generate and validate questions, producing 5K high-quality trajectories for SFT and RL training.

### Strengths
1. The paradigm shift from "information-driven" to "formalization-driven" demonstrates originality and theoretical depth. 
2. The paper presents a complete pipeline: the KP representation, layer-wise expansion strategy, and Expander agent collectively implement a closed loop of autonomous data generation and quality assurance, demonstrating engineering rigor.
3. The paper validates WebShaper's performance across multiple backbones. Ablation studies across various models and additional analyses confirm WebShaper's practical utility and generalizability.

### Weaknesses
1. The paper suffers from serious writing and data consistency issues, with multiple conflicting details: (1) Content misalignment between Sections 4.4.3 and 4.4.4; (2) Inconsistent key performance data: Qwen2.5-32B's SFT performance is reported as 44.66 in Figures 3 and 4, but 43.6 in Table 2; Qwen2.5-72B similarly changes from 46.66 to 45.6. Such conflicts undermine the paper's rigor.
2. The paper lacks evaluation on benchmarks such as BrowseComp and XBench, limiting fair comparison with concurrent work. Notably, BrowseComp and BrowseComp-zh are explicitly mentioned in the Introduction, but experiments are completely absent.
3. WebShaper's Visit tool uses Qwen-2.5-72B for content summarization, while baseline methods may not have this design. Part of the performance gains may be attributed to tool design rather than data quality.
4. While the paper demonstrates the advantages of formalization, certain IS tasks may be difficult to express, such as those that require logical or causal reasoning in GAIA.

### Questions
1. According to Table 2, QwQ-32B's SFT performance is identical to the post-RL result. Why does QwQ show no improvement after RL? 
2. Figure 3a compares Formalization (FL) vs. Natural Language (NL), and Figure 3b compares Layer-wise (G) vs. Sequential (S), but lacks a baseline that removes both components simultaneously (i.e., the combination of NL + Sequential). This baseline is essential for understanding the independent contributions and interaction effects of the two components.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper presents a dataset generation approach who's output can be used to
train information seeking agents via the SFT and RL paradigms. The key of the
method lies in its formalization, essentially a blueprint for complex questions
which the method can generate from a simple seed question though expansion. The
resulting trajectories are then used for training. Their WebShaper model
achieves the best performance among all information seeking agents on two
public benchmark datasets.

### Strengths
S1: Novel training data construction method that improves over previous "information-driven" paradigms.

S2: Method improves on performance compared to existing IS agents.

S3: WebShaper's training data is more effective compared to the output of other
training data generation methods.

### Weaknesses
W1: It seems the created dataset itself is not available. This would have been a very valuable contributions for the community. The same goes for the code, you claim it is open-source, but it was not provided in the supplement.

### Questions
Q1: Is the dataset and code available? You claim that the method is open-source, but I could not find any supplementary material.

Q2: Can you quantify the training data generation effort with WebShaper compared to traditional methods?

### Soundness
4

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
The paper presents WebShaper, a framework for synthesizing high-quality training data for information-seeking (IS) agents. Unlike existing information-driven approaches that retrieve text first and then generate questions, WebShaper adopts a formalization-driven paradigm, starting from a structured definition of the IS task using set-theoretic constructs. Its core unit, the Knowledge Projection (KP), represents entity relations and can be composed through Intersection and R-Union to form complex reasoning tasks. A multi-step Expander agent progressively builds valid and diverse questions while avoiding redundancy and reasoning shortcuts. Models trained on the WebShaper dataset achieve state-of-the-art performance on GAIA and WebWalkerQA, showing improved reasoning consistency and generalization over prior methods.

### Strengths
+ The paper presents a meaningful shift from an information-driven to a formalization-driven approach for data synthesis, directly addressing the inconsistency issues observed in prior methods.
+ The formalization based on Knowledge Projections (KPs) and set operations provides fine-grained control over both the reasoning structure and task complexity of synthesized data.
+ The Layer-wise Expansion Strategy effectively tackles redundancy and reasoning shortcuts common in previous synthesis frameworks.
+ Experiments on GAIA and WebWalkerQA benchmarks demonstrate sota performance, reflecting the utility and quality of synthesized data.

### Weaknesses
- The synthesis process—spanning seed generation, multi-agent expansion, and online retrieval—appears computationally expensive. A quantitative comparison of cost (e.g., API calls, runtime, or compute hours) against traditional information-driven synthesis methods would clarify its practical feasibility.
-  The current set-theoretic grammar (KPs, R-Unions, and Intersections) may not sufficiently capture complex real-world IS tasks, such as those involving temporal, comparative, or counterfactual reasoning.
-  The paper omits a “Limitations” section, leaving readers uncertain about the known weaknesses, failure cases, or scalability challenges of the proposed method.

### Questions
+ How does the formalization handle IS tasks that are difficult to express in set-theoretic form, such as comparative reasoning , temporal reasoning beyond filtering, or counterfactual reasoning？
+ Could the authors provide an estimate of the computational cost of generating the WebShaper dataset (e.g., time, number of API calls, GPU hours)? How does it compare to the information-driven baselines it outperforms?

### Soundness
3

### Presentation
2

### Contribution
3
