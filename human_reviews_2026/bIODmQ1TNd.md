# LiveWeb-IE: A Benchmark For Online Web Information Extraction

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Web information extraction (WIE) is the task of automatically extracting data from web pages, offering high utility for various applications.
The evaluation of WIE systems has traditionally relied on benchmarks built from HTML snapshots captured at a single point in time.
However, this offline evaluation paradigm fails to account for the temporally evolving nature of the web; consequently, performance on these static benchmarks often fails to generalize to dynamic real-world scenarios.
To bridge this gap, we introduce LiveWeb-IE, a new benchmark designed for evaluating WIE systems directly against live websites.
Based on trusted and permission-granted websites, we curate natural language queries that require information extraction of various data categories, such as text, images, and hyperlinks.
We further design these queries to represent four levels of complexity, based on the number and cardinality of attributes to be extracted, enabling a granular assessment of WIE systems.
In addition, we propose Visual Grounding Scraper (VGS), a novel multi-stage agentic framework that mimics human cognitive processes by visually narrowing down web page content to extract desired information. 
Extensive experiments across diverse backbone models demonstrate the effectiveness and robustness of VGS.
We believe that this study lays the foundation for developing practical and robust WIE systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents LIVEWEB‑IE, a benchmark for evaluating web information extraction systems on so‑called live websites, and introduces a baseline model, Visual Grounding Scraper (VGS). The benchmark is constructed through website selection, annotation, and verification, emphasizing content stability for reproducibility. However, since it deliberately focuses on static webpages and stable answers, the setting does not clearly demonstrate advantages over conventional snapshot‑based benchmarks in reflecting real temporal dynamics.

### Strengths
The benchmark clearly defines four task types covering single/multi‑attribute and single/multi‑value extraction, providing a systematic and comprehensive framing of the web information extraction problem.

The paper presents a coherent pipeline for website selection, annotation, and verification, showing solid engineering effort.

The introduction of the baseline model (Visual Grounding Scraper – VGS) illustrates the benchmark’s intended use and offers an initial reference for comparison.

### Weaknesses
Although the paper highlights the live nature of the benchmark, most of the evaluated webpages are essentially static and rarely updated. As a result, the work does not clearly demonstrate any advantage of using “live” data over conventional snapshot‑based settings that are already stable and reproducible. Even HotpotQA, which is built on fixed Wikipedia content and unchanging commonsense facts, offers a comparable level of variability.

The benchmark explicitly filters for stable pages and attributes “unlikely to change,” ensuring reproducibility but sidestepping what genuinely makes web extraction live—coping with temporal and structural drift. A truly live evaluation should treat the target as a function responsive to time and environment rather than a fixed, time‑agnostic value.

Incorporating multiple modalities in web information extraction is intuitively reasonable, but the paper does not clearly explain how these modalities are functionally integrated or why they are critical to the benchmark’s objectives. The description and experiments leave it unclear whether multimodality influences task formulation, model behavior, or evaluation outcomes, making this aspect feel under‑motivated despite being prominently emphasized.

### Questions
Could the authors clarify what specific phenomena of “liveness” are actually captured in the current setup, beyond what a static snapshot benchmark would provide?

Given that the dataset construction deliberately filters for stable content, how does the current benchmark evaluate a system’s robustness to temporal or structural changes—if at all?

Multimodality is highlighted as a major aspect of the work, but its functional role is unclear. Can the authors specify which tasks or results concretely rely on non‑textual modalities, and what insights would be lost if these were removed?

### Soundness
2

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
3

### Summary
This paper introduces LIVEWEB-IE, an online Web Information Extraction (WIE) benchmark, and proposes a multi-stage agentic framework called VGS (Visual Grounding Scraper). By mimicking human cognitive processes, VGS accurately identifies the information that needs to be extracted. Experiments demonstrate that VGS outperforms existing methods on both LIVEWEB-IE and several established WIE benchmarks.

### Strengths
The proposed Visual Grounding Scraper (VGS) framework that mimics human information-seeking behavior on web pages is novel and practical. Multi-stage visual grounding (region → element → XPath) effectively reduces HTML noise, achieving great performance on both LIVEWEB-IE and other offline benchmarks.

### Weaknesses
1. Weak Motivation. While the paper argues that performance on offline benchmarks fails to generalize to live websites due to temporal changes in web structures, this claim lacks sufficient empirical evidence. For instance, there is no direct comparison showing how existing methods degrade over time on the same websites, nor quantitative data on the frequency or impact of such changes. This undermines the core motivation, as it's unclear whether the offline-to-online gap is as significant as asserted, potentially overstating the need for LIVEWEB-IE.
2. Reproducibility Undermined. Live evaluation causes inconsistent results across runs/time windows, breaking fair comparison and replicability. Website states can vary at each evaluation (e.g., due to updates in layout or content), leading to inconsistent results across runs. Different systems or papers might be tested in non-overlapping time windows, making it impossible to ensure fair comparisons.
3. Expenditure and efficiency in VGS. The multi-stage VGS framework relies heavily on VLMs for visual grounding and pinpointing, which could incur significant computational costs and latency, especially for large-scale scraping. The paper does not discuss efficiency metrics (e.g., inference time or resource usage) or optimizations, making it less practical for real-world deployment compared to simpler HTML-based methods.

### Questions
No specific questions. Please refer to the weakness section for detailed concerns.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces a new benchmark, LiveWeb IE, which focuses on addressing the limitations of previous offline benchmarks that capture only a fixed snapshot in time. LiveWeb IE is claimed to evaluate models directly on live websites, thereby reflecting their performance on temporally evolving web content. The paper further proposes a Web IE framework, Visual Grounding Scraper, which leverages visual cues as guidance rather than directly locating the target element.

### Strengths
-Empirical results show that the newly proposed benchmark is more challenging, and the performance on state-of-the-art LLM/LMMs are less saturated; showing a gap between WIE systems and humans on more up-to-date live websites.
-The proposed VGS framework is effective on both closed-source and open-source models
-The paper writing is clear and contains sufficient ablations on the VGS components

### Weaknesses
-While the LiveWeb-IE benchmark is claimed to be “evaluating directly against live websites”, it is not clear how the benchmark automatically evolves as the website updates over time. The dataset construction pipeline is still based on a snapshot of a certain time and requires human verification to curate the data. It is potentially an overclaim that the benchmark is “Live”.
-It is also not clear how to handle layout changes through time while still keeping the evaluation/annotation valid; and how much human efforts are required to keep the benchmark up-to-date
-The WIE task is very related to GUI and Computer Use tasks (especially in the settings where the html/a11y tree is available for the perception step); there is a lack of discussion and comparison with the widely studied GUI agent models and literature.

### Questions
What is the key difference between the WIE task and a subset of GUI tasks (where the agent do not need to perform action but simply perform the perception)?

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
4

### Summary
The paper tackles the task of Web Information Extraction (WIE) directly on live websites, while prior work has mainly considered static websites.  For this, they introduce a new benchmark LiveWeb-IE which existing Web IE approaches seem to struggle on. The paper also introduces Visual Grounding Scraper (VGS) that leverages VLMs to narrow down the relevant web element to extract desired information. Experiments show that VGS outperforms current WebIE approaches on LiveWeb-IE while also demonstrating improvements on existing WebIE benchmarks.

### Strengths
1. The methodology is described well and the paper is easy to read. 
2. The experimental results are pretty comprehensive, with a variety of backbone LLMs used. 
3. The LiveWeb-IE benchmark can be a very valuable resource to the research community.

### Weaknesses
1. The novelty of the VGS approach is limited. The method mainly incorporates VLMs for prompting to narrow down relevant web elements, with the XPath generation part already being done in prior work [1]. 
2. The related work section is pretty lacking, with no discussion of distinctions/comparisons of VGS with prior WebIE methodologies.
3. While VGS is relatively performant, the authors should also show a cost comparison with prior baselines. The approach of iteratively pass all regions of the webpage to VLM can be cost intensive. The paper does not have any discussion of the additional cost of the proposed approach. 

[1] Automating xpath query generation using nlp for streamlined web crawling and gui testing; Kaur et al 2025

### Questions
1. Given that the use of live websites within this benchmark creates the problem of information on these websites changing over time, how do the authors  plan to address this? Will the answers regularly be refreshed, similar to FreshQA [2]?

[2] FreshLLMs: Refreshing Large Language Models with Search Engine Augmentation; Vu et al 2023

### Soundness
3

### Presentation
3

### Contribution
2
