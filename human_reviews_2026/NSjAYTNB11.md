# PlotCraft: Pushing the Limits of LLMs for Complex and Interactive Data Visualization

- Decision: Reject
- Scores: 6, 6, 6, 4

## Abstract
Recent Large Language Models (LLMs) have demonstrated remarkable proficiency in code generation. However, their ability to create complex visualizations for scaled and structured data remains largely unevaluated and underdeveloped. To address this gap, we introduce \textbf{PlotCraft}, a new benchmark featuring 1k challenging visualization tasks that cover a wide range of topics, such as finance, scientific research, and sociology. The benchmark is structured around seven high-level visualization tasks and encompasses 48 distinct chart types. Crucially, it is the first to systematically evaluate both single-turn generation and multi-turn refinement across a diverse spectrum of task complexities.
Our comprehensive evaluation of 23 leading LLMs on PlotCraft reveals obvious performance deficiencies in handling sophisticated visualization tasks. To bridge this performance gap, we develope \textbf{SynthVis-30K}, a large-scale, high-quality dataset of complex visualization code synthesized via a collaborative agent framework. Building upon this dataset, we develope \textbf{PlotCraftor}, a novel code generation model that achieves strong capabilities in complex data visualization with a remarkably small size.
Across VisEval, PandasPlotBench, and our proposed PlotCraft, PlotCraftor shows performance comparable to that of leading proprietary approaches. Especially, on hard task, Our model achieves over 50\% performance improvement. We will release the benchmark, dataset, and code at https://anonymous.4open.science/r/PlotCraft-E320.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces PlotCraft, a benchmark of 982 text-to-visualisation tasks spanning 7 high-level intents and 48 chart types, explicitly targeting the ability of LLMs to generate complex plotting with multiple subplots and multi-turn refinement from structured raw data. It evaluates 24 LLMs (23 external and their fine tuned model PlotCraftor) and finds substantial gaps on medium/hard visualisation tasks. To address this, the authors create SynthVis-30K, a large scale dataset of multi-modal task–code pairs created via a multi-agent pipeline and then create PlotCraftor by fine tuning (SFT) Qwen3-Coder-30B-A3B on SynthVis-30k. PlotCraftor reaches top open-weight performance and approaches strong proprietary models across PlotCraft, VisEval, and PandasPlotBench, with particularly large gains on hard tasks. The benchmark, dataset, and code are open sourced.

### Strengths
- **Clear problem and gap:** Complex, multi-subplot, multi-turn visualisation is under-evaluated; PlotCraft directly targets this. It also provides quite an extensive series of test, well varied.
- **Well-specified evaluation:** Two-stage pipeline (valid execution + automated visual judge) with decomposed compliance/quality metrics and reported human agreement (Cohen’s κ).
- **Breadth of baselines & analysis:** 23–24 models across difficulty tiers; insightful scaling analysis showing hard-task are not really solved with models under ~100B params.
- **Substantive details:** SynthVis-30K via multi-agent pipeline and PlotCraftor training details; reproducibility and release plan.
- High quality figures (for instance 1 and 3 are really clear and qualitative)

### Weaknesses
- **Judge dependence:** Primary scores come from Gemini-2.5-Pro; although correlated with humans, single-judge bias and failure on subtle visual errors remain concerns. Multiple models could be used concurrently for a more robust way of obtaining the score
- **Licensing/annotator clarity:** Ethics statement says no external workers; elsewhere, “human annotators” crafted refinement prompts—clarify who they were and dataset licence variability.
- **Scope limitations:** Focus on matplotlib/Python; portability to other viz ecosystems is not evaluated. (Authors note this implicitly as future work.)
- **Automated synthesis cost/filters:** Multi-agent generation is costly; strict termination criteria may bias examples toward what the judge favours.
- Results lack variance estimates (no std/CIs across seeds).

### Questions
- **Judge robustness:** How do results/rankings change if Claude/GPT-4o serve as judges, or using 2 LLMs + the human factors? Any statistically significant rank flips?
- **Human evaluation scale:** Beyond the 500-chart subset, could you provide CIs and inter-rater stats per sub-metric and per difficulty tier?
- **Licensing & annotators:** Were the “human annotators” solely the author team? Any variability in Kaggle dataset licences that restrict redistribution of derived images?
- **Generalisation:** Do findings carry over to Vega-Lite/Altair, Plotly, or ggplot2 targets? Any preliminary cross-backend experiments? Or has it not yet been tested by the authors ?
- Please add **mean±std over seeds** for main metrics (by difficulty + turn setting). If the judge is stochastic (which i would assume as you use Gemini-2.5-Pro), include **std over multiple seeds**
- Small typo to correct:
    - Title Table 3 “benchamrks”

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the limitations of Large Language Models (LLMs) in generating complex data visualizations. The authors introduce three key contributions: 1) PlotCraft, a new benchmark with ~1k challenging tasks to evaluate both single-turn generation and multi-turn refinement. 2) SynthVis-30K, a large-scale, high-quality dataset of visualization code synthesized via a multi-agent framework. 3) PlotCraftor, a fine-tuned 30B parameter model that achieves performance comparable to leading proprietary models like Claude-4-Sonnet on complex visualization tasks. The work demonstrates that while current LLMs struggle with complexity, targeted fine-tuning on high-quality, domain-specific data can significantly bridge this performance gap.

### Strengths
- **Timely Benchmark** (PlotCraft): The paper introduces a much-needed benchmark that moves beyond simple text-to-chart tasks (e.g. VisEval, NVbench). I think this benchmark is a timely step towards better alignment of llm’s coding capability and people’s practical needs for visualization generation.
- **Strong Empirical Results and Model** (PlotCraftor): The comprehensive evaluation across 24 models provides clear evidence of the current state-of-the-art and its limitations. The proposed PlotCraftor model demonstrates impressive results, showing that a smaller, specialized model can achieve performance on par with much larger proprietary systems.

### Weaknesses
- **Reliance on Automated Evaluation**: While the authors validate a MLLM-based judge (Gemini-2.5-Pro) against human scores, current MLLM can still miss important visual flaws / quality aspects that can be easy for human to inspect. Moreover, relying on a certain version of proprietary MLLM (Gemini-2.5-Pro) for benchmark could lead to potential bias and stability risks. These limitation should be discussed more prominently. Some relevant works that are recommend to review and discuss:
[1] VisJudge-Bench: Aesthetics and Quality Assessment of Visualizations
[2] VIS-Shepherd: Constructing Critic for LLM-based Data Visualization Generation
[3] Closing the Feedback Loop in Text2Vis: Refining Visualization with Vision-Language Models

- **Reproducibility and Cost of Data Synthesis**: The data synthesis pipeline appears to rely on powerful, proprietary models (e.g., Claude), which may create a high cost and reproducibility barrier for other researchers. A discussion on the associated costs and the feasibility of using open-source alternatives would strengthen the paper.

### Questions
- Could you provide a few concrete examples where the automated visual judge's score differed significantly from human evaluators?

- Can you comment on the approximate cost (e.g., API calls/cost per sample) of your SynthVis-30K data generation pipeline? Have you explored the feasibility of replacing the proprietary models in your pipeline with open-source ones? Which model does you use for the Planner, Coder and Debugger? Could you list them explicitly in the manuscript for better transparency and reproducibility? 

- Could you elaborate on the specific types of reasoning you believe are crucial for "Hard" tasks and are lacking in smaller models?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents PlotCraft, a benchmark for evaluating LLM's capability to generate charting code from detailed instructions. Comparing to prior benchmarks, PlotCraft includes harder data visualization tasks that involve subplots and layers that improves upon previous benchmarks. The paper also presents a training synthetic dataset SynthVis-30k to support visualization tasks similar to PlotCraft (based on a disjoint set of Kaggle data). The evaluation shows that the model trained on this dataset perform as good as out of the box closed-source models. The benchmark also shows interesting observations on limitations of simple benchmarks and reasoning/non-reasoning model performance.

### Strengths
- new dataset capturing a much richer set of visualizations comparing to previous benchmarks, better for measuring model's data visualization performance.
- the synthetic benchmark provides source for finetuning models towards the goal.

### Weaknesses
- The task in this dataset is quite detailed and verbose, which may not match how such models would be used in practice. But on the other hand, I don't think this is a big issue since the paper focuses on model capability measurement. But this worth emphasis, since many practical visualization tasks are quite more open-ended from a high-level question.
- From examples in appendix, some charts with layout issues could be the result of the data and instruction characteristics (i.e., if the model follow instructions exactly, the chart may already be over-crowded). There could be some discussion about whether the task + data combinations may just lead to non-optimal visualizations.
- The LLM as judge evaluation, while standard, worth some deeper investigation. For example, given reasonable consistency between human and LLM judge agreement on different aspects of evaluation, whether the assemble into the final score for evaluation.

### Questions
- It would be ideal for authors to discuss visualization creation from detailed instruction vs abstract goal (which could lead to multiple feasible charts, not just one), and clarify the paper falls into the first category in revision. 
- better elaborate LLM-as a judge quality.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper introduces PLOTCRAFT, a benchmark for LLM-based generation of complex visualization code, featuring single-turn generation and multi-turn refinement tasks across three difficulty tiers and a rich taxonomy of intents and chart types. The authors also release SYNTHVIS-30K, a sizable dataset sourced from real-world data, and fine-tune PLOTCRAFTOR, which achieves competitive performance with a relatively small model size. The evaluation covers 24 models with detailed analyses and discussion. Results highlight current deficiencies of large language models on visualization code generation and demonstrate the effectiveness of SYNTHVIS-30K for complex plot generation.

### Strengths
- Relevant topic: The paper addresses an emerging area in LLM-based complex visualization generation and outlines a research gap that is worth exploring. It also proposes a benchmark to facilitate further study in this space.

- Use of real-world data: The SYNTHVIS-30K dataset is reasonably large and built from real data sources, which helps enhance its practical value.

- Evaluation design: The paper presents experiments across 24 models and provides an analysis of their behavior in visualization code generation. The comparison between single-turn and multi-turn settings offers some insights into refinement effects. The reported alignment between the automatic evaluator and human judgments provides an initial indication of its reliability.

### Weaknesses
- **Undefined notion of “multi-chart” generation:** The paper does not clearly define what constitutes *multi-chart* or *multi-plot* generation. If it merely refers to placing several plots side-by-side, the novelty and significance are limited. A more meaningful definition should clarify whether the goal involves cross-chart coherence, shared data semantics, or insight-driven multi-view coordination. It is also unclear whether the dataset considers the relationship between generated charts and the underlying data insights.

- **Evaluation fairness concern:** PlotCraftor is fine-tuned on its own benchmark and then evaluated on the same benchmark. This setting favors the proposed model and limits fair comparison with other models that were not adapted to the dataset.

- **Mismatch between claims and actual content on layout:** Although the Introduction frames layout design as a key challenge, the paper provides limited technical treatment of layout generation. The notion of *layout* seems reduced to basic Matplotlib subplot configuration, but a more comprehensive explanation of what layout entails (e.g., spatial organization, visual hierarchy, annotation distribution, cross-chart alignment) is missing.

- **Inaccurate dataset comparison (Table 1):** The paper states that prior datasets do not include multi-chart scenarios, but ChartArXiv does contain multi-chart examples. The comparison table should be corrected to reflect this.

- **Reliability of the multi-turn evaluation setup:** The multi-turn refinement scenario is valuable and aligns with how humans iteratively improve charts. However, the evaluation relies on a single model judge. Given that chart quality is inherently subjective, relying on one judge may bias the results. A more robust approach could involve a *multi-judge ensemble* or voting across different models to reduce bias and increase scoring reliability.

- **Limited exploration of edit-based refinement:** The idea of refining existing chart code is interesting, but current refinement requires regenerating the entire code. A more intelligent and realistic approach would support *edit-based* modifications—e.g., adjusting key parameters, fixing specific components, or applying localized patches—rather than full code rewriting.

- **Dataset construction details insufficiently described:** Since the dataset is positioned as a main contribution, the construction methodology needs greater transparency. The paper lacks detail on dataset design criteria, annotation guidelines, quality control processes, and whether multiple annotators were involved to ensure correctness and consistency.

### Questions
1. **Clarification on multi-chart definition:**  
   How do the authors formally define *multi-chart* generation in this work? Does it involve cross-chart coordination or insight-driven composition, or is it limited to placing multiple plots within a single figure?

2. **Evaluation fairness:**  
   Since PlotCraftor is fine-tuned on the proposed benchmark and then evaluated on it, how do the authors ensure fair comparison with other models that were not trained or adapted to this dataset? Would zero-shot and fine-tuned baselines be considered for a fairer evaluation?

3. **Reliability of single-judge scoring:**  
   The multi-turn refinement setting is interesting, but chart quality can be subjective. Why did the authors rely on a single model judge? Have the authors considered using multiple judges or an ensemble voting mechanism to improve scoring robustness?

4. **Dataset construction transparency:**  
   As the dataset is a primary contribution, could the authors provide more details on the construction criteria, annotator workflow, and quality control procedures? Was multi-annotator verification performed to ensure correctness and consistency of refinement instructions?

### Soundness
2

### Presentation
2

### Contribution
2
