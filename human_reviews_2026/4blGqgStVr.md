# PDFBench: A Benchmark for De novo Protein Design from Function

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 6, 4

## Abstract
Function-guided protein design is a crucial task with significant applications in drug discovery and enzyme engineering. However, the field lacks a unified and comprehensive evaluation framework. Current models are assessed using inconsistent and limited subsets of metrics, which prevents fair comparison and a clear understanding of the relationships between different evaluation criteria. To address this gap, we introduce PDFBench, the first comprehensive benchmark for function-guided de novo protein design. Our benchmark systematically evaluates eight state-of-the-art models on 16 metrics across two key settings: description-guided design, for which we repurpose the Mol-Instructions dataset, originally lacking quantitative benchmarking, and keyword-guided design, for which we introduce a new test set, SwissTest, created with a strict datetime cutoff to ensure data integrity. By benchmarking across a wide array of metrics and analyzing their correlations, PDFBench enables more reliable model comparisons and provides key insights to guide future research.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
PDFBench introduces a unified, reproducible benchmark for function-guided protein design across two settings: description-guided generation (natural-language functional descriptions) and keyword-guided generation (InterPro and/or GO terms). It evaluates eight modern systems (ProteoGAN, ESM3, CFP-Gen, Chroma, ProteinDT, PAAG, Pinal, ProDVa) on 16 metrics organized into six dimensions: Plausibility, Foldability, Language Alignment, Similarity, Novelty, and Diversity. The description-guided split repurposes Mol-Instructions to enable quantitative assessment, while the keyword-guided split is a new test set, built with a strict datetime cutoff to mitigate contamination. The paper also studies metric-metric relationships. Overall, PDFBench fills a gap by normalizing task definitions and metrics so competing model families can be compared fairly and by summarizing empirical regularities that can guide method design. Generally, I believe it is a novel and interesting benchmark in a wider range of protein design.

### Strengths
- PDFBench consolidates multiple factors into a single framework: two canonical input regimes, six evaluation dimensions, and 16 concrete metrics that span feasibility and goal-matching. This is a long-standing comparability gap where prior works selected disjoint metric subsets and left cross-family conclusions ambiguous.

-  The description-guided track systematically assesses Mol-Instructions; the keyword-guided track (SwissTest) applies a strict post-2025 annotation cutoff. The paper explicitly discusses overlap analyses and the rationale behind the splits, which strengthens confidence in leakge avoidance.

- The authors quantify relationships such as lower perplexity co-occurring with higher pLDDT and lower PAE and show retrieval scores can swing dramatically under different negative-sampling strategies underscoring why metric details matter as much as the model. I think it is a very meaningful try.

- On keyword-guided design, CFP-Gen, Pinal, and ProDVa frequently yield foldable sequences with strong language alignment under the chosen proxies, but show trade-offs against local repetition and novelty/diversity.

### Weaknesses
- Novelty and diversity are sensitive to database coverage and thresholds. Sequence/structure dissimilarity against large corpora is an intuitive starting point, but the conclusions can drift with database freshness, taxonomy composition, and cutoff conventions. Because the benchmark observes that stronger alignment often coincides with higher repetition and lower novelty, it would be valuable to report the sensitivity of novelty/diversity to alternative databases and thresholds and to visualize the trade-off among alignment, foldability, and novelty. For example, in structure-based protein design, papers tend to compare the effect on multiple datasets (cath4.3, casp, pdb...even AF2)

- The paper documents sizable gaps across strategies, which means absolute retrieval numbers are fragile and method rankings can change with protocol choices. A standardized retrieval protocol with fixed public seeds, paired with confidence intervals and significance tests, would reduce measurement noise and make cross-paper comparisons more robust.

- Descriptions, GO terms, and InterPro entries encode function at different levels of specificity and with different structural implications. Dose-response studies that vary input constraint strength (e.g., number and granularity of keywords, cross-level GO) and measure the impact across all six dimensions would clarify how methods adapt to tight versus loose guidance.

- Thresholds and binning rules need stronger statistical grounding. Practical heuristics like repetition cutoffs or high-confidence bands for structural scores improve readability but can be brittle across datasets. ROC/PR analyses, bootstrap confidence intervals, and sensitivity curves around chosen thresholds would make the findings more portable.

- Computational cost and evaluation throughput are not quantified. Some metrics can be computationally expensive at the benchmark scale. Summarizing per-metric runtime and resource footprints and suggesting cost-aware evaluation recipes (for example, perplexity filtering before foldability checks) would help teams plan and reproduce studies efficiently.


- Many tables present point estimates without variance bands over seeds or resamples, making it hard to judge whether narrow score gaps reflect real differences. As a solid benchmark, providing mean and standard deviation or confidence intervals, along with simple non-parametric tests on key comparisons, would make the takeaways sturdier. If the authors already have those results, I do encourage them to report it in the later version, but if not, I do not think it is required to conduct experiments during the rebuttal phase.

### Questions
I have a question about the contents in Table 1, which mentions that some baseline models don’t support the same kinds of keyword inputs, so comparing them in one unified “keyword” task can be unfair: a model forced to handle an input type it doesn’t really support will look worse than it truly is. For example, in the paper’s capability table, ProteoGAN supports GO terms but not InterPro, while ESM3 supports InterPro but not GO, so in an InterPro-only setting ProteoGAN is handicapped, and in a GO-only setting ESM3 is. Maybe the fix is to clearly label each method’s capabilities in the results tables and report two numbers per method: one under its own native, fully supported pipeline (to show its true ceiling) and one under the unified PDFBench pipeline. But as I did not run the model myself, it might be not doable. Hope the authors can provide further elaborations on this issue.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces PDFBench, a new benchmark aimed at providing a unified evaluation framework for function-guided de novo protein design. While the initiative to consolidate evaluation in this area is appreciated, it is questionable whether a text-guided approach is useful in real-world protein design. Also, the design choices of the benchmark itself are a major concern. Specifically, the metrics, comparison set, and the practical relevance of the structural results presented lead to significant concerns regarding the paper's contribution to the field. I do not currently believe that text-guided protein design offers a realistic path forward for practical protein engineering.

### Strengths
The paper's primary strength is the attempt to address a clear gap in the field: the lack of a unified and comprehensive evaluation framework for function-guided (text-guided) protein design models. The systematic effort to evaluate multiple state-of-the-art models across a variety of metrics is a valuable starting point.

### Weaknesses
The proposed benchmark suffers from several critical weaknesses that undermine its conclusions and practical relevance:
1. The core motivation of text-guided design is questionable. Coarse-grained descriptions like Gene Ontology (GO) terms, EC numbers, or keywords are fundamentally insufficient for specifying novel, complex functions (e.g., designing a high-affinity binder for a specific, newly discovered target). This limits the real-world utility of the entire methodology.
2. The benchmark utilizes existing models for evaluating the "language alignment score." However, the accuracy of existing GO annotation algorithms is notoriously problematic, meaning the core metric used to assess the models' alignment with the text prompt is potentially based on a flawed, inaccurate standard.
3. The benchmark significantly diverges from the state-of-the-art methods currently employed by biologists for de novo protein design. The paper doesn't include comparisons with highly successful and practically proven structure-based methods like RFDiffusion and ProteinMPNN. This makes it impossible to compare the performance of text-guided methods against realistic, successful baselines.
4. The reported plddt values for the designed proteins are far too low. With baseline models consistently achieving scores below 80, the generated sequences lack the predicted foldability required for practical use in biology or engineering.

### Questions
See above

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
3

### Summary
PDFBench set up a new benchmark for text-guided protein design task. This task went popular these years, but there is not comprehensive benchmark for this task. Authors have considered several aspects of protein-related metrics, most of which are wildly used in protein design benchmarks.

### Strengths
Authors have considered "Pausibility", "Foldability", "Language Alignment", "Novelty" and "Diversity", most of which are well-established metrics and are wildly accepted in different works. The text-guided protein design is a relatevly a new task without current benchmark. The design of the benchmark is good, divide tasks into "text-guided" and "keyword-guided" is practical and useful.

### Weaknesses
Generally, the task that using function to design protein is not well-accepted because of low controability of the design process, and using traditional pipeline to design protein using RFdiffusion and ProteinMPNN can also achieve similar task (describe function using structure, or using additional models). Add more discussion that comparing traditional workflow with the function2protein workflow is useful in this paper.

### Questions
1. Is it possible that the protein design metrics is different in different kinds of protein functions? Like the well-studied or well-documented function can have much better performance than those rarely discovered function. Can you add a comparison in keyword guided part?
2. Using large language model like ChatGPT to design protein can also be an interesting baseline for this task

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
3

### Summary
This paper presents PDFBench, a comprehensive benchmark for de novo protein design conditioned on function. It systematically evaluates eight representative models under both description-guided and keyword-guided settings, using metrics across six key dimensions: Plausibility, Foldability, Language Alignment, Similarity, Novelty, and Diversity.
The benchmark introduces two dedicated test sets and provides a unified framework for fair, reproducible comparison. The results offer deeper insights into the relationship between sequence design, structure formation, and functional alignment in generative protein modeling.

### Strengths
1. The benchmark fills an important gap in functional protein design and provides a unified, reproducible evaluation framework.
2. The paper is well-organized, and the rethinking section is particularly informative and insightful.

### Weaknesses
1. The dataset and code links are currently inaccessible (“The requested file is not found”), which limits reproducibility.
2. Many evaluation metrics overlap with those used in ProDVa, raising concerns about the degree of novelty beyond benchmark integration.

### Questions
1. The link to the codebase is inaccessible. Could the authors verify and restore the dataset/code access links to ensure reproducibility?

P.S. I also wonder whether this paper might better fit the Dataset and Benchmark Track, given its focus on standardization rather than methodological innovation.

### Soundness
3

### Presentation
3

### Contribution
3
