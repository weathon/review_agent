# RAViG-Bench: A Benchmark for Retrieval-Augmented Visually-rich Generation with Multi-modal Automated Evaluation

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 4, 6, 6

## Abstract
Retrieval-Augmented Visually-rich Generation (RAViG) extends RAG by integrating textual explanations with multiple visual elements in a well-structured layout. Despite its growing adoption, no existing benchmark offers a holistic evaluation of RAViG. Current RAG benchmarks focus on text-only generation, while natural language to visualization (NL2VIS) benchmarks focus on "show-data-as-chart" style queries and do not follow the RAG paradigm.
To address this deficiency, we present RAViG-Bench, the first comprehensive benchmark specifically designed for RAViG. The benchmark features a diverse collection of authentic user queries, each paired with real-world web retrievals to simulate realistic RAViG scenarios.
Besides, we introduce a novel multi-modal automated evaluation framework that holistically assesses the quality of RAViG outputs. This framework scrutinizes the generated content by evaluating the functionality, design quality, and content quality of both textual and visual components.
Our extensive experiments on leading commercial and open-source LLMs provide a comprehensive analysis of their current capabilities, highlighting significant limitations and charting key directions for future research in this emergent area.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces a new task named Retrieval-Augmented Visually-rich Generation (RAViG), which involves automatically generating HTML-style reports that integrate multiple visualizations (e.g., charts, tables) with explanatory text, based on retrieved web documents. 

The authors highlight the gap in existing benchmarks: 1) RAG benchmarks primarily evaluate text-only outputs, and 2) NL2VIS (Natural Language to Visualization) benchmarks assume the underlying data is already provided, consequently, they fail to cover the unique failure modes of this task, such as HTML rendering errors, design flaws, text-visual misalignment, and data hallucinations. 
To address this problem, the authors constructed the RAViG-Bench dataset and proposed a comprehensive, automated evaluation framework to assess LLM performance on this task.

### Strengths
1. Comprehensive Benchmark：As a benchmark paper, the authors establishes a complete framework for this new RAViG task, comprising a dataset, detailed task definitions, and an end-to-end evaluation pipeline.

2. Multi-faceted Evaluation Framework: 
The proposed evaluation framework is comprehensive, employing a assessment from various aspects like Functionality, Design, and Content, that progressively filters and evaluates model outputs.

### Weaknesses
1. Limited Task Novelty: Even the authors discuss a lot about related works in RAG and NL2Vis, actually RAViG task closely overlaps with existing ''deep research'' AI agents/task/benchmark (e.g., OpenAI Deep Research [1], Tongyi DeepResearch [2]) and Data Insight Agent/Task (e.g., InsightBench [3]) that also generate reports with visualization elements (i.e., various analysis chart with explaining text) from web retrieval or multiple data sources (e.g. databases). Requiring HTML output doesn't fundamentally differentiate it, undermining its claimed innovation. For example, if we requires a Deep Research Agent’s output to be restricted to a HTML format by adding this requirement statement in the input of the agents, then the general DeepResearch task becomes this RAViG task. 

2. Questionable Task Design: Compared with the general Deep Research task, which requires the LLM to act like a data analyst with searching, reasoning and report generation capability, this RAViG task performs like forcing the LLM to act as both a data analyst and HTML engineer, which is inefficient and a bit unreasonable for real-world applications. These two steps could be effectively solved by separate agents/tasks—one generating a visualized report (not necessarily in HTML, like the general deepresearch task) and another one acting like a coding agent which converting the report into other formats (PDF, HTML, markdown, etc.). Combining them into a single task offers no clear additional benefit. 

[1] https://openai.com/index/introducing-deep-research/

[2] https://github.com/Alibaba-NLP/DeepResearch

[3] https://insightbench.github.io/

### Questions
I notice that RAViG-Bench does not provide a direct Ground-Truth visual output for each query against which to compare model responses. Instead, it relies heavily on an LLM-as-a-Judge paradigm combined with rule-based checks for evaluation.

Could you discuss the potential reasons/implications of this design? e.g.,, does the absence of a fixed ground-truth and the high dependence on the LLM-based judges make the scores susceptible to judge-specific biases? This concern seems particularly relevant when the judge model is similar in architecture or training data to the models being evaluated.

Furthermore, without a concrete ground-truth, how can the benchmark ensure precise and objective accuracy measurement, rather than just a relative score?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors present a novel benchmark on visually-rich RAG for data summary and visualisation. It includes a dataset of natural language queries that are appropriate for visually-rich summarisation, and an automated LLM-as-judge evaluation protocol that separately assesses technical functionality, aesthetic presentation, and content quality/consistency. They provide a number of performance baselines from existing VLMs, and a reasonable analysis of agreement with human raters.

### Strengths
The use case is well-motivated and seems to address a genuine gap in current benchmarks, based on a convincing demonstration of failure modes in existing VLMs. 

The dataset construction protocol is sensible, and the listed examples of queries that are suitable for visually-rich RAG appear appropriate. I had my doubts about whether purely automated query filtering would produce an adequate dataset, but the presence of a final human review is welcome. 

The LLM-as-judge evaluation method is difficult to trust, so the extensive validation experiments performed are also welcome. Agreement with human raters is encouraging, and in particular, the analysis of self-preference bias is a good inclusion.

### Weaknesses
I would like to see a more detailed analysis of how much the protocol is reliant on these specific models, or a more robust protocol that can verifiably work well with open-source models. I would also like to see some analysis of where these models disagree with human evaluators, i.e. whether there are specific or systematic biases, and whether these differ across models.

Minor points that did not affect the review score: 
- In the appendix, on page 25 (around lines 1311-1314), the query does not match the image. The query asks about “densely populated countries”, actually used in the next page, but the image is about F1 grand prix races.
- In table 3, the colouring scheme is counter-intuitive. It should not really matter, but my brain insists that green is good (high performance) and red is bad (low performance). Maybe this is a cultural thing, but I suggest reversing the scale.
- Small typos: “Releated work” (line 810) should be “Related work”, “Lable” (lines 1385 and 1392) should be “Label”

### Questions
1. The evaluation protocol is said to rely solely on GPT-4o for design quality assessment, and Gemini-2.5-pro for content quality evaluation. Is the quality of the evaluation procedure highly sensitive to these particular VLM/LLM choices? 

2. Would the automated evaluation have more agreement with human raters if an ensemble of different models was used, or an ensemble of different prompts? 

3. Is there a failsafe method in place if these APIs become inaccessible in the future, during the lifetime of the benchmark? 

4. (minor) In 4.2, the section on occlusion (lines 261-268) is not as convincing as the others. Does this approach fail to detect when elements from separate semantic modules (i.e. under different headers) are overlapping? Does it fail if the generated output uses few header tags or none at all?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces RAViG-Bench, a benchmark for evaluating Retrieval-Augmented Visually-Rich Generation (RAViG) systems. It targets models that must both retrieve multimodal evidence and generate grounded visual outputs. While prior work has separately evaluated retrieval reasoning or visual generation, RAViG-Bench assesses the complete retrieval-to-generation pipeline.
The benchmark defines three complementary evaluation axes, functionality, design quality, and content fidelity, measured through a hybrid approach combining rule-based structural metrics and an LLM-as-a-judge for semantic and stylistic assessment. A complexity-corrected holistic score further normalizes performance across layouts of varying difficulty, discouraging trivial generations.
Ten multimodal generative models, including GPT-5 as the latest, are evaluated under a unified protocol. Results show that current models produce visually plausible yet semantically inconsistent outputs, exposing systematic weaknesses in factual grounding and compositional reasoning. Beyond reporting scores, the paper analyzes failure types and suggests directions for future models that better integrate retrieval accuracy, design coherence, and content faithfulness.

### Strengths
The paper is convincing in its motivation. It identifies the lack of a benchmark that jointly evaluates retrieval and visual generation, and addresses this gap with a reproducible and well-structured framework. The proposed three-axis design is coherent and logically motivated. The introduction of new evaluation metrics is particularly commendable, as it reveals cases where models succeed through oversimplified outputs rather than genuine multimodal reasoning. The benchmark’s overall design is practical, reproducible, and aligned with real-world multimodal generation workflows. Its evaluation protocol, illustrated in Fig. 3, is comprehensive, assessing both functional and aesthetic quality beyond textual correctness. The hybrid evaluation scheme combining rule-based validation with an LLM-as-a-judge, achieves a pragmatic balance between objectivity and interpretive depth. The empirical section is substantial. It presents a meaningful taxonomy of failure patterns and articulates clear directions for what stronger models should achieve. The inclusion of recent frontier models under identical zero temperature conditions ensures reproducibility and enables reliable cross-model comparison of multimodal generation capabilities. The paper demonstrates careful engineering, solid coverage of prior literature, and thoughtful reflection on how evaluation can drive progress in multimodal generation research. Section 5.3 provides an exemplary in-depth analysis that dissects error patterns and design trade-offs across model families, revealing nuanced distinctions in retrieval grounding and layout control.

### Weaknesses
Despite the paper’s quality, several conceptual limitations prevent RAViG-Bench from being fully convincing as a standard benchmark for visually rich generation.

1. The reliability of the LLM-as-a-judge protocol is not convincingly established.
While the authors report correlations with human judgments, such correlation alone does not guarantee fairness or stability.
It remains unclear whether the same model family acting as both generator and evaluator introduces self-preference bias, or whether version drift and sampling variance were systematically tested.
Since the reported rankings rely heavily on these automatic judgments, a more rigorous quantitative validation of scoring reliability is necessary.


2. Fixing temperature to zero improves reproducibility but oversimplifies the evaluation.
This deterministic setup removes stochastic diversity, which is crucial for understanding how multimodal generators behave in real deployment.
Consequently, the benchmark captures only a single deterministic output per query, making it difficult to assess model robustness or creative range.
A complementary analysis using moderate temperature values would provide a more realistic picture of generative variability.


3. Although the appendix includes numerous rendered-page examples, the paper does not analyze them in sufficient depth.
These visuals appear without detailed textual interpretation, leaving readers uncertain about which specific failure modes or design patterns they represent.
While Section 5.3 offers valuable qualitative insights, its narrative remains largely disconnected from the visual evidence provided.
Incorporating even a small number of representative rendered examples directly into the analysis would enhance both interpretability and transparency.

4. The benchmark primarily measures factual consistency and renderability, yet overlooks higher-level qualities such as design coherence, narrative flow, and alignment with human intent.
These aspects are central to evaluating multimodal generation as creation, not merely as factual assembly.
Without accounting for such qualitative dimensions, the benchmark, though methodologically rigorous, remains somewhat narrow in how it operationalizes visual generation quality.

### Questions
Q1. Can the authors provide additional quantitative validation to assess scoring robustness? For example, evaluating cross-family judgments (e.g., Gemini judging GPT outputs) or repeating the evaluation with different model versions would reveal whether ranking stability holds across evaluators.
Such analysis would strengthen confidence that the reported rankings are not artifacts of evaluator bias.


Q2. Would the authors consider an additional experiment that compares deterministic and mildly stochastic decoding settings?
Such a comparison could reveal whether the benchmark rankings remain stable when generation variability is introduced, thereby improving ecological validity.



Q3. Could the authors include a few representative rendered-page examples within the main text or extend Section 5.3 with brief commentary linking each example to observed failure types?
This addition would make the qualitative analysis more transparent and verifiable, without requiring major additional experiments.


Q4. Connecting to Q2, can the authors discuss additional evaluation dimensions to capture creative quality?
Even a small-scale human study or expert annotation subset could substantiate the broader claims about visually rich generation.

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
3

### Summary
This paper discusses the topic of Retrieval-Augmented Visually-rich Generation (RAViG), which extends RAG by combining textual explanations with multiple visual elements in a structured layout. It identifies a benchmarking gap: existing RAG benchmarks are text-only, and NL2VIS benchmarks focus on charting rather than the RAG paradigm. To address this, the authors propose RAViG-Bench, a comprehensive benchmark built from authentic user queries paired with real-world web retrievals to simulate realistic RAViG scenarios. They also introduce a multi-modal automated evaluation framework that assesses functionality, design quality, and content quality across both textual and visual components. While HTML is used as the working representation, the dataset, evaluation framework, algorithms, and criteria are format-agnostic and can be adapted to other structured formats by modifying input/output modules. Experiments on leading commercial and open-source LLMs reveal notable limitations and significant room for improvement. The current system reliably detects readability-related defects and performs basic visual complexity correction but does not evaluate subjective aesthetics or richer factors like typography and layout, and it excludes judgments about “over-design.” The authors suggest future work should incorporate user-centered aesthetic evaluation and finer-grained visual refinement to assess visual appeal, accessibility, and overall user experience.

### Strengths
* Clear problem formulation and novelty  
The paper clearly positions RAViG-Bench as the first benchmark to comprehensively evaluate Retrieval-Augmented Visually-rich Generation (RAViG), i.e., reports that integrate textual explanations with multiple visual elements under the RAG paradigm. The contributions (dataset, automated evaluation framework, and empirical study) are well delineated.

* Realism-oriented data construction  
Queries are derived from real user search queries and paired with top web retrievals, preserving noise, with human validation applied. This design faithfully simulates realistic RAG conditions where models must ground to multiple noisy sources.

* Alignment between automatic and human evaluation  
The paper reports agreement analyses showing that automated evaluation correlates with human judgments, supporting the validity of the framework.

* Broad model coverage and actionable findings  
A diverse set of commercial and open-source LLMs is evaluated under uniform settings. Results reveal that functionality is relatively strong but design quality is a major bottleneck, providing concrete guidance for future model improvements.

### Weaknesses
* $\gamma_{vc}$ and $VC_{score}$ design choices may be somewhat ad hoc  
The visual richness score ($VC_{score}$) aggregates z-scored counts (modules/charts/tables) with fixed weights.
Similarly, $\gamma_{vc}$ is defined using a linear factor of $VC_{score}$ (e.g., $\alpha$ = 0.3).
The generality of these settings has not been fully stress-tested.

* Limited reproducibility due to partial prompt disclosure  
For confidentiality reasons, complete prompts are not released, constraining exact replication, ablations on prompt sensitivity, and fine-grained reevaluation.

* As shown in Table 3, the performance of GPT-5 seems very high and nearly perfect according to the evaluation metrics introduced in this paper. It seems that RAViG-Bench is almost solved by current top-tier commercial LLMs. This implies that RAViG-Bench will be difficult to use for distinguishing performance among top-tier commercial LLMs in the future. This may reduce the usefulness of the proposed dataset.

### Questions
* The agreement between automated multi-modal evaluation and human assessments as shown in Table 2 seems very high.
It is basically a good point.
However, I suspect that the sampled problems might be extremely easy.
Could the authors elaborate on this point?

* There are many proposed evaluation metrics. 
It would be better to clearly show the chance rate that can easily understand the effectiveness of each method on each metric.
Could the authors provide the chance rate for each metric?

### Soundness
2

### Presentation
3

### Contribution
2
