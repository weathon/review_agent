# MM-BrowseComp: A Comprehensive Benchmark for Multimodal Browsing Agents

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4, 6

## Abstract
AI agents with advanced reasoning and tool use capabilities have demonstrated impressive performance in web browsing for deep search. While existing benchmarks such as BrowseComp evaluate these browsing abilities, they primarily focus on textual information, overlooking the prevalence of multimodal content. To bridge this gap, we introduce MM-BrowseComp, a novel benchmark comprising 224 challenging, hand-crafted questions specifically designed to assess agents' multimodal retrieval and reasoning capabilities. These questions often incorporate images in prompts, and crucial information encountered during the search and reasoning process may also be embedded within images or videos on webpages. Consequently, methods relying solely on text prove insufficient for our benchmark. Additionally, we provide a verified checklist for each question, enabling fine-grained analysis of multimodal dependencies and reasoning paths. Our comprehensive evaluation of state-of-the-art models on MM-BrowseComp reveals that even top models like OpenAI o3 with tools achieve only 29.02\% accuracy, highlighting the suboptimal multimodal capabilities and lack of native multimodal reasoning in current models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
MM-BrowseComp consists of 224 challenging, hand-crafted questions distributed across 22 distinct subtasks covering five broad categories (Media, Technology, Society, Geography, and Academics). The questions are intentionally multi-hop and difficult, with a construction criteria ensuring they remain unanswerable by strong VLMs with web search in a single attempt, or by unfamiliar human annotators within five minutes. A crucial component is the verified checklist provided for each question, which represents the minimal irreducible reasoning path required to reach the correct answer. This checklist enables the use of Strict Accuracy (SA) alongside Overall Accuracy (OA), allowing for a fine-grained analysis that distinguishes genuine reasoning from "lucky guessing". Experimental results demonstrate that the top performer, OpenAI o3 with tools, achieved only 29.02% OA and 19.64% SA, confirming the benchmark’s challenging nature.

### Strengths
- The benchmark successfully bridges the gap left by previous textual benchmarks (like the original BrowseComp).
- The checklists provide fine-grained evaluation, moving beyond simple correctness to assess the path taken.
- Evaluates 18 models across multiple dimensions with detailed error taxonomy and modality-specific performance breakdown.

### Weaknesses
- While the authors convincingly justify the size through the rigor of construction and high filtering rate, a total of 224 instances across 22 distinct subtasks may be insufficient for reporting reasonable scores at this granularity.
- Heavy reliance on GPT-4o-2024-11-20 as the sole evaluator for checklist, and I believe this might add certain evaluation bias.

### Questions
- Could the authors elaborate on the strict criteria used by annotators to ensure the reasoning checklist is truly "irreducible"?
- Why evaluate open-source agents on only 54 instances? This seems too limited for reliable conclusions. What were the selection criteria?

### Soundness
2

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
3

### Summary
The paper introduces MM-BrowseComp for evaluating agents that must browse the web and reason over multimodal content. Each instance comes with an irreducible reasoning checklist that specifies the minimal sequence of retrieval and reasoning steps required to reach the answer, enabling fine-grained assessment beyond final-answer accuracy. On this benchmark, strong systems achieve only ~29% accuracy, underscoring the challenge and current gap in native multimodal browsing.

### Strengths
1. Clarity: The paper is readable and well-structured, with intuitive examples and comprehensive task taxonomy/mixture. Construction principles and validation steps are communicated with sufficient detail.

2. Significance: Addresses a timely need: deep web browsing with native multimodality—central for real-world assistants. The results and analyses (e.g., modality-specific performance, test-time scaling, error taxonomy) are likely to shape evaluation practices and agent design.

### Weaknesses
1. Scale: 224 instances is on the small side for a general-purpose benchmark spanning 22 subtasks; per-subtask sample sizes are too thin for robust statistics. Consider releasing a larger dev/test split or staged expansions, and report confidence intervals (e.g., bootstrap over items) in the main text. The dataset probably won't be very meaningful if the data size is too small.

2. Potential construction bias and leakage checks. During dataset construction, there could be several stages with risk of potential biases. Difficulty criteria include “unanswerable by strong models in one attempt.” can be subjective. The dataset construction lacks inter-annotator agreement. This risks encoding model-specific blind spots. Add contamination audits, time-stamped sources, and a multi-attempt human check protocol report (agreement, time-to-solve). 

3. While the paper has some analysis on the evaluation results, the LLM-judge based analysis seems not very scalable if the data size grows large. Also the llm judge backbone may also introduce extra bias for analysis.

### Questions
1. Checklist design & validation. How do you ensure minimality and non-redundancy of checklists across annotators? Report inter-annotator agreement on checklists and provide a public rubric of what counts as “completed.” (This would reduce reviewer subjectivity when others extend the benchmark.) 

2. Tool standardization. Could you release a reference tool suite (OCR, layout/grounding, video frame sampler) and a tool-capability checklist per model/agent so results are not confounded by missing/different tools? This would also clarify where o3’s edge stems from (backbone vs. toolset). 

3. Dataset growth & governance. Any plan for a continually updated MM-BrowseComp with frozen yearly snapshots and public leaderboards? Is there any plan to make the data collection pipeline more scalable and generalizable.

4. Error taxonomy reliability. The failure analysis uses GPT-4o to label errors. Please report labeling agreement (e.g., dual-judge consistency) and try to use different LLM to label the errors. Would GPT-4o have bias towards OpenAI models like o3?

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
The authors introduce a new benchmark to evaluate web-browsing agents in multimodal environments, where text shortcuts are not available. It contains 224 carefully designed questions that often include images in the prompt or require agents to pull information from visuals on real web pages. Each question is paired with a verified checklist of reasoning steps and supporting evidence. The best system only achieves only about 29% accuracy. The findings show that current models still struggle significantly with multimodal browsing and reasoning.Models perform much worse on visual content than on text and tend to depend on shallow image-captioning shortcuts rather than genuine visual understanding.

### Strengths
- The data construction process ensures questions require multimodal browsing, effectively eliminating text shortcuts.
- Queries in the dataset go through rigorous difficulty-based filtering.
- The human-verified checklist of minimal finegrained reasoning steps provides a valuable signal, it provides a way for evaluation to go beyond just right/wrong final answers.

### Weaknesses
- There is missing a human baseline to calibrate what model accuracy means. It would provide an estimate for the performance ceiling of this task.
- In 3.1.1, authors assert that essential information to solve a task should not appear in any text source. However, there is no mention of how this verification is done.
- Although the authors repeatedly refer to “video-dependent” tasks, the paper never specifies how models are expected to engage with videos. Are agents intended to interact with video content directly, or are they simply expected to rely on accompanying textual information, like transcripts or descriptions?

### Questions
- Figure 3 could be clearer. Rather than focusing on whether each input includes an image, it might be more informative to present statistics on the actual modalities required or used by each task.
- Relevant work from earlier this year with a similar objective of measuring multimodal interactions with no textual shortcuts: `BEARCUBS: A benchmark for computer-using web agents`

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces MM-BrowseComp, a new benchmark designed to evaluate multimodal browsing agents that integrate reasoning and tool use. It comprises 224 hand-crafted questions across 22 subtasks, requiring retrieval and reasoning over both textual and visual information. Each question includes a verified checklist that tracks reasoning steps, allowing fine-grained analysis beyond final-answer accuracy. Experimental results show that even state-of-the-art models like OpenAI’s o3 achieve an accuracy of 29%, highlighting the difficulty of multimodal reasoning and the limitations of current models. Overall, the paper provides a challenging dataset that fills an important gap in multimodal browsering agent evaluation.

### Strengths
* The proposed benchmark is constructed through multiple rigorous verification phases.
* The experiment part systematically compares a wide range of state-of-the-art closed- and open-source models, offering a clear view of current limitations and performance gaps.

### Weaknesses
* The tasks in this benchmark are often intentionally complex and involve multi-hop reasoning, which may not accurately reflect the typical multimodal search behaviors encountered in real-world web browsing scenarios.
* The heavily hand-crafted nature of the benchmark may limit real-world generalizability.

### Questions
See weakness.

### Soundness
3

### Presentation
3

### Contribution
2
