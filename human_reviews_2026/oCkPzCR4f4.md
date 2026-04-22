# From Charts to Code: A Hierarchical Benchmark for Multimodal Models

- Avg Score: 2.67
- Decision: Withdrawn (Treated as Reject)
- Scores: 0, 2, 6

## Abstract
We introduce Chart2Code, a new benchmark for evaluating the chart understanding and code generation capabilities of large multimodal models (LMMs). Chart2Code is explicitly designed from a user-driven perspective, capturing diverse real-world scenarios and progressively increasing task difficulty. It consists of three levels: Level 1 (Chart Reproduction) reproduces charts from a reference figure and user query; Level 2 (Chart Editing) involves complex modifications such as changing chart types or adding elements; and Level 3 (Long-Table to Chart Generation) requires models to transform long, information-dense tables into faithful charts following user instructions. To our knowledge, this is the first hierarchical benchmark that reflects practical chart2code usage while systematically scaling task complexity. In total, Chart2Code contains 1,947 tasks across 22 chart types, paired with multi-level evaluation metrics that assess both code correctness and the visual fidelity of rendered charts. We benchmark 25 state-of-the-art LMMs, including both proprietary and the latest open-source models such as GPT-5, Qwen2.5-VL, InternVL3/3.5, MiMo-VL, and Seed-1.6-VL. Experimental results demonstrate that even the strongest models struggle to generalize across levels and chart types, highlighting the significant challenges posed by Chart2Code. We anticipate this benchmark will drive advances in multimodal reasoning and foster the development of more robust and general-purpose LMMs.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This paper introduces Chart2Code, a new benchmark for evaluating LMMs on "chart-to-code" tasks. The authors evaluate 25 LMMs (GPT-5, Gemini 2.5 Pro, Claude-Sonnet 4, Qwen 2.5-VL, MiMo-VL, etc.) using metrics that combine code executability and visual fidelity (assessed by GPT-5-mini). The paper claims to be the first hierarchical benchmark capturing user-driven chart editing and long-context table-to-chart generation.

**However**, upon close inspection, **the submission shows substantial textual and conceptual overlap with the previously published ICLR 2025 paper ChartMimic[1].** The overlap includes **shared motivation**, **identical section organization**, **parallel task definitions**, and similar **evaluation methodology**, with only minor naming changes (e.g., “Direct/Customized Mimic” → “Reproduction/Editing”). 

**The new paper lacks clear attribution and differentiation from that prior benchmark.**

[1] ChartMimic: Evaluating LMM's Cross-Modal Reasoning Capability via Chart-to-Code Generation. arXiv/2406.09961

### Strengths
None

### Weaknesses
- **Severe overlap with prior work**. The paper shares near-identical phrasing, figures, and experimental design with ChartMimic (ICLR 2025). Core sections such as “Task Definition,” “Data Curation,” and “Evaluation Metrics” appear rewritten with only superficial changes (e.g., GPT-4o → GPT-5-mini).

- **Lack of proper citation and differentiation.** ChartMimic is cited only once in passing; there is no explicit statement that Chart2Code builds upon it. This constitutes a potential research-integrity issue (plagiarism or self-plagiarism).

- **Questionable originality of data and metrics.** The claimed new benchmark design mostly reuses ChartMimic’s data sources and evaluation protocols, offering minimal methodological advancement.

- **Ethical and reproducibility concerns.** The overlap raises uncertainty about data ownership, reuse permissions, and potential double submission of overlapping content. Moreover, although the paper claims to have released code, the accompanying repository does not contain the dataset itself, only stating:

> “Due to file size limitations, only a small set of demo data is included in this repository. We plan to open-source our full dataset in the near future—stay tuned!”
This lack of accessible data severely limits the reproducibility of reported results and prevents independent verification of the benchmark’s claimed scale and diversity.

### Questions
The paper should **explicitly** claim and demonstrate the differences from ChartMimic, not only in task definition but also in dataset composition, data curation pipeline, and evaluation methodology. At present, many sections (e.g., task formulation, evaluation metrics, and even notation) are nearly identical to ChartMimic, making it difficult to judge genuine novelty.

In particular, **Table 1 (“Comparison of existing chart-to-code benchmarks”) omits ChartMimic entirely**, despite it being the most directly related prior work. This omission reinforces concerns that the authors may be attempting to present an extended version of ChartMimic as a new benchmark without proper acknowledgment. The authors should update this table and discuss the differences explicitly to dispel such suspicion.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces Chart2Code, a hierarchical benchmark designed to evaluate large multimodal models' capabilities in chart understanding and code generation. The benchmark comprises three progressively challenging levels. The authors construct a dataset of 1947 tasks spanning 22 chart types and propose multi-level evaluation metrics assessing both code executability and visual fidelity. Through comprehensive evaluation of 25 state-of-the-art LMMs, the paper demonstrates that even the strongest proprietary models struggle significantly with visual fidelity, particularly on editing tasks and long-context table-to-chart generation, revealing substantial gaps between code-level correctness and pixel-level accuracy in automated chart generation.

### Strengths
- Well-motivated hierarchical design: The three-level structure effectively captures realistic usage patterns, from simple reproduction to complex editing and long-context processing;
- Comprehensive evaluation methodology: The paper proposes a thoughtful multi-dimensional evaluation approach combining execution rate, code-level metrics , and chart-level visual fidelity scores;

### Weaknesses
- Insufficient novelty over ChartMimic: While the paper positions itself as advancing beyond ChartMimic, the core contribution appears incremental. The main differences are: (1) adding Level 3 long-table tasks, and (2) scaling up data collection. However, ChartMimic already established the chart-to-code evaluation paradigm, and the fundamental task formulation remains largely unchanged. 
- Limited scope of incremental contributions: The workload presented, while substantial in terms of data collection and experimental evaluation, does not sufficiently justify acceptance at a top-tier venue like ICLR.

### Questions
1. Beyond scale and task categories, what fundamental new insights does Chart2Code provide that ChartMimic could not? Can you more clearly articulate the unique scientific contributions or interesting findings?
2. Can you provide concrete failure case analysis showing what specific aspects cause the code-visual fidelity gap?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Chart2Code, a benchmark for evaluating LMMs in chart understanding and code generation. It defines three levels of increasing difficulty: chart reproduction, chart editing, and long-table to chart generation. Using metrics for execution, code correctness, and visual fidelity, the study benchmarks 25 SOTA LMMs. The results show that while these models perform well on simple tasks, their performance drops sharply on more complex ones, highlighting the challenge of real-world chart-to-code generation.

### Strengths
1. The paper introduces a well-structured hierarchical benchmark that decomposes chart-to-code reasoning into three progressive levels. The proposed benchmark is challenging, and the statistical details are sufficient.
2. The paper proposes a set of multi-dimensional evaluation metrics, and the proposed base evaluation method appears thorough and reliable.
3. The analysis and experiments are extensive.

### Weaknesses
1. The evaluation relies on GPT-5-mini, which may introduce circular reasoning, bias, or limitations in visual fidelity due to limited human validation.
2. Metric Reliability and Anomalies: The paper's results reveal unexpected patterns that may affect confidence in the LMM-Score. For example, in Table 3, Seed-1.6-VL (0.812) is reported to outperform GPT-5 (0.633) on L1 visual fidelity. In Table 4, the 7B MiMo-VL-RL model (0.471) scores more than twice as high as GPT-5 (0.220) on L2 fidelity. These surprising results are not fully examined. They suggest that the LMM score may not always correspond to correctness and could favor outputs that are visually appealing to the judge-LMM, even if they differ factually or stylistically.
3. Level 3 (Long-Table to Chart) is described as the paper's most novel and important contribution, as it addresses the critical bottleneck of long-context reasoning. Nonetheless, this level currently has only 150 tasks from 39 files, which limits the ability to draw firm conclusions about model performance on this complex, open-ended task.
4. Shallow "Long-Table" Reasoning: Level 3 seems to test primarily "long-context retrieval" (locating the right data in a long file) rather than "complex data reasoning." Real-world tasks often require several steps of data wrangling, such as filtering, merging tables or sheets, pivoting, and handling missing values before visualization. The benchmark may not fully reflect this important layer of pre-visualization reasoning.
5. Limited Scope of Chart Libraries: The evaluation focuses exclusively on Matplotlib-based libraries and does not address interactive libraries such as Plotly or declarative options like Altair. A benchmark aiming for real-world relevance might benefit from acknowledging this limitation in scope.
6. Some unexpected results are observed. For example, Seed-1.6-VL attains a high LMM score of 0.81 (L279), while MiMo-VL-7B-RL generates relatively few successful codes yet still achieves high LLM and LMM scores (L339). This may indicate that the metrics are somewhat sensitive.
7. A unified or weighted composite metric might provide additional insights. Although there are many evaluation metrics, the absence of a unified composite score could make direct model comparisons less intuitive. For example, while it is observable that Gemini-2.5-Pro and Claude-Sonnet-4 perform relatively well, it remains somewhat challenging to determine the ranking of other models at a glance.

### Questions
1. Could you please provide a more detailed explanation regarding the metric anomalies, specifically the unexpectedly high LMM scores for Seed-1.6-VL and MiMo-VL-7B-RL? Are these results indicative of a sensitivity in the LMM-Score (as judged by GPT-5-mini) to particular model output styles or artifacts that may not align with human-perceived correctness?
2. Have you conducted a human evaluation on a subset of the benchmark to calibrate the LMM-Score? Have you also assessed the correlation between LMM and expert scores? Linking these would strengthen the paper's claims about visual fidelity.
3. Why were the "base evaluation" (8-dimensional) scores not reported for Level 1 tasks? I think it would be interesting to see this objective breakdown for the "simpler" reproduction tasks as well, if available.
4. Would it be possible to add a comparison between MoE and dense models to help analyze the architectural effect?
5. Would you consider including results for Qwen3-VL to further complete and update the benchmark coverage?

### Soundness
3

### Presentation
2

### Contribution
3
