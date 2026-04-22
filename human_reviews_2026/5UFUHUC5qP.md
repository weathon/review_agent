# GDGB: A Benchmark for Generative Dynamic Text-Attributed Graph Learning

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 4, 8, 4

## Abstract
Dynamic Text-Attributed Graphs (DyTAGs), which intricately integrate structural, temporal, and textual attributes, are crucial for modeling complex real-world systems. However, most existing DyTAG datasets exhibit poor textual quality, which severely limits their utility for generative DyTAG tasks requiring semantically rich inputs. Additionally, prior work mainly focuses on discriminative tasks on DyTAGs, resulting in a lack of standardized task formulations and evaluation protocols tailored for DyTAG generation. To address these critical issues, we propose \underline{G}enerative \underline{D}yTA\underline{G} \underline{B}enchmark (GDGB), which comprises eight meticulously curated DyTAG datasets with high-quality textual features for both nodes and edges, overcoming limitations of prior datasets. Building on GDGB, we define two novel DyTAG generation tasks: Transductive Dynamic Graph Generation (TDGG) and Inductive Dynamic Graph Generation (IDGG). TDGG transductively generates a target DyTAG based on the given source and destination node sets, while the more challenging IDGG introduces new node generation to inductively model the dynamic expansion of real-world graph data. To enable holistic evaluation, we design multifaceted metrics that assess the structural, temporal, and textual quality of the generated DyTAGs. We further propose GAG-General, an LLM-based multi-agent generative framework tailored for reproducible and robust benchmarking of DyTAG generation. Experimental results demonstrate that GDGB enables rigorous evaluation of TDGG and IDGG, with key insights revealing the critical interplay of structural and textual features in DyTAG generation. These findings establish GDGB as a foundational resource for advancing generative DyTAG research and unlocking further practical applications in DyTAG generation. The dataset and source code are available at \url{https://github.com/Lucas-PJ/GDGB-ALGO}.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper highlights 2 key concerns with existing dynamic text-attributed graphs: a)Lack of high-quality textual attributes- either completely absent or limited to non-semantic quality, like only containing usernames or emails; b) Lack of standardized DyTAG generative tasks formulations and evaluation protocols: Existing evaluation frameworks don't incorporate textual characteristics.  To solve for this, the authors propose the following contributions
a) Eight DyTAG datasets covering diverse domains with rich textual semantic information
b) Two novel tasks on these DyTAGs: Transductive dynamic graph generation and Inductive Dynamic graph generation with metrics converting Graph Structural Metrics, Textual Quality Metrics, and Graph Embedding Metrics. 
c) Baseline: Given this suite of datasets, the paper proposes GAG-General, which is an LLM-based multi-agent framework adopted for DyTAG generation tasks. 

Collectively, the paper proposes a robust benchmarking for DyTAG generation.

### Strengths
a) There is a clear gap in standardized datasets for textual attributed graphs, which this paper attempts to fill.
b) Novel Eight text attributed dynamic graph datasets proposed
c) Comprehensive comparison against existing datasets in terms of richness and utility of these textual attributes to highlight their importance
d) The paper clearly motivates the problem and reports a detailed analysis of these datasets.

### Weaknesses
A) The paper somewhat dilutes its core contribution by introducing a Generative framework: GAG-General, which is adopted from existing work. Core contribution could only be a textual attributed benchmark with evaluation metrics. And this generative framework could be proposed as a baseline method, presenting it as a central contribution, overemphasizes an incremental component, as novelty here is very limited. 

B) Text quality and graph embedding metrics: There seems to be a dependency on the underlying LLM  on final performance. However, the paper provides no clear guidance on which model should be considered standard. Can the author guide which LLM should be used for evaluation, or can any one of them be used? The evaluation framework needs to be final, but the proposed one seems to be dependent on the underlying LLM, with no clarity on which to use.  There is a table 3 in the experiment section, but it's not clear what the conclusion is from it. 

An evaluation framework that is dependent on underlying LLMs will lead to different researchers obtaining inconsistent results. Any new LLM version update or slight change in prompt can lead to different results. The authors of this paper should define a fixed or recommended LLM backbone with recommended hyperparameters during LLM inference and input for consistency, robustness, or a way to minimize the LLM-specific biases.

### Questions
Overall, the paper is of significant use to the temporal graph learning community, which is in dire need of benchmark datasets, especially in dynamic attributed graphs and a consistent evaluation framework.   But paper in this current form needs clear positioning and a more robust evaluation framework. 

I will be open to accepting if authors can address these concerns meaningfully.

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
3

### Summary
GDGB introduces a high-quality benchmark for generative learning on dynamic text-attributed graphs (DyTAGs), accompanied by two novel tasks (TDGG and IDGG) and a multi-agent LLM framework (GAG-General). The datasets notably surpass prior ones in textual richness, and the holistic evaluation protocol covers structure, text, and time.

### Strengths
1. This paper is well written and easy to follow.
2. The paper proposes the 1st dedicated generative DyTAG benchmark with rich, realistic node/edge texts across eight diverse domains.
3. The TDGG and IDGG tasks is well defined, and the metrics (structure, text, embedding) give a comprehensive quality picture.

### Weaknesses
1. IDGG new-node evaluation may lack direct semantic-drift or human-amenity checks.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
GDGB presents a generative Dynamic Text-Attributed Graph benchmark with eight text-rich datasets, two tasks (transductive and inductive DyTAG generation), and metrics covering structure, time, and text. An LLM-based multi-agent framework, GAG-General, standardizes evaluation and delivers competitive results. Experiments show that high-quality text and structure are crucial, enabling rigorous assessment.

### Strengths
1. The paper is well written and easy to follow.

2. The authors introduce several dynamic text-attributed graph datasets with higher text quality than existing benchmarks.

3. The authors design an LLM-based multi-agent generative framework; an accompanying illustration would improve comprehensibility.

### Weaknesses
### **Major Concern**

1. **Claim of the “first” generative DyTAG benchmark.**
   The paper claims to be the first generative DyTAG benchmark, yet it does not evaluate **existing TAG-generation or DyG-generation methods**. Instead, it varies only the LLM backbones for DyTAG generation. Without comparisons to prior generative baselines, the “first” or “state-of-the-art” claim is overstated. Please include representative TAG/DyG generative methods (or strong non-LLM baselines) and report side-by-side results.

2. **Assessment of text quality and incomplete reporting on DTGB.**
   The paper asserts that existing DyTAG datasets have poor text quality; however, the reported results on DTGB appear strong in both graph-embedding metrics and textual quality scores. Moreover, only **partial** DTGB results are shown.

   * Please explain the selection criteria for the reported DTGB subsets.
   * Provide complete results on all DTGB datasets for both generation tasks (transductive and inductive), using the same metrics and settings.

3. **Direct comparison between the new datasets and existing ones.**
   To establish the value of the introduced datasets, the main paper should present **explicit empirical generation comparisons** between DTGB and the newly proposed datasets. This will make the dataset-quality argument far more convincing.

4. **Metric definitions and justification.**
   Evaluation is central in generative work, yet the three metrics are not formally defined. Please:

   * Provide precise mathematical definitions (inputs, outputs, normalization, and aggregation).
   * Justify why each metric is appropriate for DyTAG generation and discuss potential limitations or failure modes.
   * Clarify implementation details


### **Minor Concern**

   Could you please elaborate on concrete use cases for dynamic text-attributed graph generation?

### Questions
All my major and minor concerns are mixed in the Weaknesses part. Please kindly refer to it.

### Soundness
2

### Presentation
4

### Contribution
2
