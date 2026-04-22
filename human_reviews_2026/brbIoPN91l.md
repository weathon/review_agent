# OmniEarth-Bench: Probing Cognitive Abilities of MLLMs for Earth's Multi-sphere Observation Data

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 4, 4, 6

## Abstract
Existing benchmarks for multimodal learning in Earth science offer limited, siloed coverage of Earth’s spheres and their cross-sphere interactions, typically restricting evaluation to the human-activities sphere or atmosphere and to at most 16 tasks. Holistically evaluating MLLMs on observational data across all Earth spheres face three limitation: multi-source heterogeneous data, unlocking scientific formulation and cross-sphere reasoning. Therefore, we introduce OmniEarth-Bench, the first multimodal benchmark that systematically spans all six spheres—atmosphere, lithosphere, oceansphere, cryosphere, biosphere, and human-activity sphere—and cross-sphere. Built with a scalable, modular pipeline that ingests 33 native Earth-observation sources and expert-in-the-loop curation, OmniEarth-Bench produces 29,855 standardized, expert-curated annotations. All annotations are organized into a four-level hierarchy (Sphere, Scenario, Ability, Task), encompassing 109 expert-curated evaluation tasks.  Experiments on 9 state-of-the-art MLLMs reveal that even the most advanced models struggle with our benchmarks, where none of them reach 35% accuracy, revealing systematic gaps in Earth-system cognitive ability. The dataset and evaluation code were released.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work introduces a new benchmark for MLLMs on earth observation. Most existing earth observation benchmarks only focus on a single "sphere", e.g., human activities sphere or atmosphere (weather), whereas OmniEarth-Bench includes data from 6 spheres as well as cross-sphere interactions. Evaluations are also performed on a number of modern MLLMs and demonstrate relatively poor performance on OmniEarth-Bench, motivating future research on developing more scientifically capable MLLMs.

### Strengths
When released, OmniEarth-Bench will certainly be a great contribution to the community. Gathering all the data and expert annotations is an expensive endeavor and will hopefully accelerate future research in this area.

### Weaknesses
- Just to clarify, all the questions and categorizations (L1-L4) were created by domain experts and not LLMs?
- The related works section could be clearer about which of the data sources in OmniEarth-Bench (Table 2) are not already incorporated in a major ML benchmark? e.g., ERA5 is definitely a well-established benchmark for weather prediction models. Additionally, it sounds like the main contribution of OmniEarth-Bench is the sourcing of questions to go with the data in all the datasets, not just aggregating the data in a ML-ready package. Is this true? For non-earth science experts, these details will make your work more clearly distinguishable.
- Similarly, the text strongly emphasizes including 'cross-sphere interactions', but it's not clear to me what exactly the impact is. e.g., line 154: "[other benchmarks are] neglecting cross-sphere interactions essential to real-world Earth science challenges." What challenges exactly couldn't be captured in previous benchmarks that is captured now?
- On open-ended scoring via LLM-as-a-judge: What evals were done on this scoring mechanism? For example, I could see LLMs struggling with numerical tasks like comparing very similar numbers (within eps-tolerance) that are not exactly the same.
- how are visual grounding and captioning performance scored? this didn’t seem to be covered in A.4
- What is the role of the ENSO example in Sec 4.3? It comes a bit out of the blue and only has GPT-4o prediction. There are no details on the prompting setup or any other baselines. I would recommend either removing this or making it a proper task evaluated with the rigor of everything else in Sec 4.2.

### Questions
See 'weaknesses' for major questions.

Minor questions:
- Maybe L3 (the type of analysis needed to solve the problem) should actually be “L1”? since L1 L2 and L4 seem to be specific to the problem while “L3” describe general skills that MLLM needs to solve the problems. No need to change the paper for rebuttals, just a thought for future work.
- Table 3 says there are 27k MCQ and 27k open-ended when the total is 29k?
- Table 4: why are open-ended questions only evaluated on a small subset of models used for MCQ?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces OmniEarth-Bench, a large-scale multimodal benchmark for evaluating MLLMs on Earth-observation data spanning six Earth spheres and their cross-sphere interactions. The benchmark integrates 33 observation sources and defines 109 expert-curated tasks organized in a four-level hierarchy (sphere, scenario, ability, task). While the dataset is extensive and the expert-in-the-loop curation adds credibility, the paper mainly extends existing multimodal evaluation frameworks to a new scientific domain.

### Strengths
1. Comprehensive domain coverage. The benchmark is the first to systematically include all six Earth spheres and their cross-sphere interactions, providing broad and scientifically grounded evaluation coverage.
2. Expert-in-the-loop curation. Involving domain experts throughout task and data construction improves annotation quality and ensures scientific validity beyond crowd-sourced datasets.
3. Structured, multi-level design. The four-level task hierarchy (Sphere–Scenario–Ability–Task) and large-scale integration of 33 observational data sources establish a coherent and extensible framework for future geoscience MLLM evaluation.

### Weaknesses
1. Limited methodological novelty. The contribution lies primarily in dataset construction and organization; it does not introduce new evaluation paradigms, metrics, or analytical insights into MLLM reasoning.
2. Descriptive analysis. Experimental results are largely descriptive and lack deeper investigation into model behaviors, failure modes, or domain-specific reasoning limitations.
3. Scalability over insight. The benchmark’s breadth may obscure opportunities for more targeted, mechanism-driven evaluation (e.g., understanding cross-sphere reasoning or modality alignment).

### Questions
1. How were task difficulties and annotation consistency verified across spheres—were any inter-annotator agreement or difficulty statistics reported?
2. Could the authors provide evidence that “cross-sphere” tasks indeed require multi-sphere reasoning rather than parallel single-sphere cues?
3. Are there mechanisms to ensure fair comparison across models with different visual input formats or temporal reasoning capabilities?
4. Beyond accuracy, could the authors consider introducing analysis or metrics that better capture the reasoning process (e.g., causal inference, spatial-temporal consistency) rather than aggregate performance alone?

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
This paper introduces a new benchmark for assessing multimodal language models focused on earth observation data. The benchmark covers all six spheres: atmosphere, lithosphere, oceansphere, cryosphere, biosphere, and human activity sphere, as well as cross-sphere interactions. It includes approximately 30K standardized, expert-curated annotations. Currently available benchmarks in this area lack comprehensive coverage of all earth spheres. An additional feature of this dataset is its heterogeneous data types, which include multispectral satellite imagery, seismic signals, weather reanalysis, and microwave sea-ice concentration. Experiments with state-of-the-art multimodal language models (MLLMs) show they underperform on this benchmark, highlighting existing gaps in their cognitive abilities within this domain.

### Strengths
1. It is a comprehensive benchmark for the discipline of earth sciences. The benchmark covers all six spheres: atmosphere, lithosphere, oceansphere, cryosphere, biosphere, and human activity sphere, as well as cross-sphere interactions.
2. Existing MLLMs perform poorly on this benchmark, which shows gaps in their expertise in this niche domain.

### Weaknesses
1. The motivation for this benchmark is not clear. Earth science is a very niche domain and requires expert-level knowledge to answer the kind of specialized questions contained in this benchmark. Do we really need that kind of capability in general-purpose LLMs?
2. Missing retrieval-augmented generation baselines. Since this is a niche domain, I would expect the authors to retrieve some related corpora and try to check the performance. It is unlikely that the LLMs were trained on a lot of earth science data.
3. The LLM versions being evaluated are slightly obsolete. For instance, GPt-4o instead of GPT-5, claude-3.7-sonnet instead of claude-4.1-sonnet, and most importantly, Gemini-2.0 instead of Gemini-2.5.
4. Some input data, like seismic data, is time-series, and the image modality might not be a good fit for that.

### Questions
1. Would it be possible to evaluate some retrieval-augmented generation baselines on this benchmark?

### Soundness
3

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
2

### Summary
The paper introduces OmniEarth-Bench, a multimodal benchmark that evaluates MLLMs on observational Earth-system data across all six spheres plus explicit cross-sphere tasks. It standardizes 29,855 expert-curated samples into a four-level hierarchy (Sphere → Scenario → Ability → Task) with 109 L4 tasks spanning perception, general reasoning, scientific-knowledge reasoning, and CoT reasoning. A modular pipeline ingests 33 native EO sources; domain experts and annotators curate and quality-check the data. Evaluations of nine modern MLLMs show none exceed ~35% accuracy, with especially poor performance on cross-sphere reasoning, highlighting large capability gaps and motivating domain-aware modeling.

### Strengths
- First benchmark to systematically cover all Earth spheres and cross-sphere interactions using real EO data, not just exam-style questions
- There is a clear four-level framework and ability taxonomy (Perception, General, Scientific-Knowledge, CoT) tailored to Earth-science reasoning
- The benchmark results expose a large domain gap in Earth system cognitive ability, where even the most advanced models do not achieve higher than 35% accuracy
- The hierarchy, data source (Table 2), example tasks (Fig. 4), and tabled comparisons communicate clearly what is being measured; key numbers (tasks/annotations/sources) are easy to find

### Weaknesses
- There are only accuracy-related metrics being reported alone about the models tested. Authors could add ability-targeted probes or per-ability sub-scores to better isolate failure modes, which would also give a more nuanced understanding of why and where exactly these models failed
- Some L1 spheres (e.g., cryosphere) appear a lot smaller (230 questions) compared to others in L1 sphere, which can make averages sensitive to composition
- More detail/metrics is needed on refusal handling. The paper notes safety/refusals can depress scores; provide refusal-aware metrics (e.g., Acc@answered, abstention rate) and report both to avoid penalizing safe behavior.

### Questions
- The authors mention involving “MLLM specialists” alongside human experts and annotators. What concrete tasks are MLLM specialists responsible for, and how do these differ from the tasks performed by annotators and domain experts?
- Did the authors measure inter-annotator agreement for key tasks or spheres? Knowing this would be more helpful to understand how trustworthy the labels are.

### Soundness
3

### Presentation
4

### Contribution
3
