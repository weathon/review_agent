# CFDLLMBench: A Benchmark Suite for Evaluating Large Language Models in Computational Fluid Dynamics

- Avg Score: 4.67
- Decision: Reject
- Scores: 6, 6, 2

## Abstract
Large Language Models (LLMs) have demonstrated strong performance across general NLP tasks, but their utility in automating numerical experiments of complex physical system, a critical and labor-intensive component, remains underexplored. As the major workhorse of computational science over the past decades, Computational Fluid Dynamics (CFD) offers a uniquely challenging testbed for evaluating the scientific capabilities of LLMs. We introduce CFDLLMBench, a benchmark suite comprising three complementary components, CFDQuery, CFDCodeBench, and FoamBench, designed to holistically evaluate LLM performance across three key competencies: graduate-level CFD knowledge, numerical and physical reasoning of CFD, and context-dependent implementation of CFD workflows. Grounded in real-world CFD practices, our benchmark combines a detailed task taxonomy with a rigorous evaluation framework to deliver reproducible results and quantify LLM performance across code executability, solution accuracy, and numerical convergence behavior. CFDLLMBench establishes a solid foundation for the development and evaluation of LLM-driven automation of numerical experiments for complex physical systems.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces CFDLLMBench, a benchmark suite designed to holistically evaluate the comprehensive capabilities of LLMs in the specialized domain of CFD. The authors posit that while LLMs excel at general nlp tasks, their potential for automating complex scientific computing workflows remains insufficiently explored. CFDLLMBench consists of three components with progressively increasing difficulty: CFDQuery, CFDCodeBench, and FoamBench. Through rigorous experimental design and in-depth case studies, the paper reveals key shortcomings of current LLMs when applied to the scientific computing domain and provides directions for future research.

### Strengths
1.  This paper is the first to propose a comprehensive, layered LLM benchmark specifically for the CFD domain. CFD, a field that demands a high degree of physical understanding, numerical methods knowledge, and software operation skills, serves as an excellent scenario for testing the scientific capabilities of LLMs. This work fills a gap in existing LLM benchmarks concerning the automation of complex scientific workflows.

2.  The knowledge-reasoning-practice structure of CFDLLMBench is exceptionally well-designed. This progression from simple to complex allows for a clear dissection of the LLM's capability boundaries, revealing precisely at which stage the model encounters a bottleneck.

3.  The paper introduces evaluation metrics closely tied to real-world CFD practices: code executability, physical result accuracy, and numerical convergence. This set of metrics significantly enhances the scientific rigor and credibility of the evaluation.

### Weaknesses
No obvious weaknesses were found; please refer to the Questions section. :-)

### Questions
1.  The definition of success rate seems very strict, requiring the three metrics $M_{exec}$, $M_{NMSE}$, and $M_{conv}$ to simultaneously achieve a perfect score. Have you considered an evaluation system for "partial success"? Such an analysis might offer a more fine-grained perspective for understanding the model's capabilities across different sub-tasks.


2.  The experimental results demonstrate that both RAG and the Reviewer are crucial. From a qualitative perspective, what are the primary problems that each of these components addresses? For instance, when RAG is absent, is the model's most common failure the selection of the wrong solver or the inability to correctly set parameters?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents CFDLLMBench, a benchmark datset designed to evaluate large language models (LLMs) on tasks in computational fluid dynamics (CFD). This includes 1) CFDQuer : 90 multiple-choice questions testing graduate-level CFD knowledge, including fluid mechanics, numerical methods, and linear algebra 2) CFDCodeBench: 24 programming tasks requiring LLMs to generate Python code for solving 1D and 2D PDEs with specified boundary and initial conditions and 3) FoamBench: 126 OpenFOAM cases (110 basic, 16 advanced) requiring LLMs to generate complete input files and configurations for realistic CFD scenarios.

### Strengths
* The benchmark tasks are accopanied by tailored metrics for CFD (such as executability, numerical accuracy, convergence, and physical correctness).

* The paper covers diverse tasks, namely, conceptual understanding (CFDQuery), code generation (CFDCodeBench), and workflow automation (FoamBench).

* The works provides a benchmark for OpenFOAM, a widely adopted CFD suite for simulation tasks.

### Weaknesses
* Dataset statistics and distributions of problem types are unclear. It would be helpful for understanding the benchmark to provide sub-categories for problem difficulty (such as based on PDE characteristics, initial conditions, etc.)

* L253 - 257: The quality assessment of the dataset is unclear. How many human hours went into quality assessment? What criteria were used?

### Questions
* How does this benchmark account for different (but valid) numerical algorithms for a PDE problem which may have better or worse performance depending on the PDE and evaluation score in question ? 

* If the dataset was constructed from web-scraped content, how to decouple "the LLMs to have deep knowledge about topics in CFD like linear algebra, numerical methods and fluid dynamics (L 188)" with possible test-set leakage with LLM training?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The authors proposed a set of dataset and benchmark problems for evaluating different large language models (LLMs) such as ChatGPT, Gemini, Sonnet, etc, for evaluating computational fluid dynamics (CFD) problems. They proposed matrices for measuring the success rate of these LLMs and made a comparison through visual graphs and numerical values.

### Strengths
1) The topic is relevant to LLMs, which nowadays is hot.
2) Nice plots and figures
3) Good amount of supportive materials
4) Good writing

### Weaknesses
The current manuscript is a technical report and does not present any novel algorithm or methodology. Even in the context of a technical report (evaluating the performance of different LLMs for CFD), this report claims inaccurate information. For example, the authors wrote in conclusion (and similarly in introduction) that:

"In this work, we introduced CFDLLMBench, the first benchmark to holistically evaluate graduate level knowledge, numerical and physical reasoning, and practical simulation capabilities of LLMs for CFD"

It is not true and they are not the first. I believe that the manuscript suffers from the adequate and appropriate literature reviews. Pioneers of people who evaluated LLMs for CFD have been missed from the reference section and the authors claimed that they are first scientists performing such experiments. For example, please see the following journal paper:

https://www.dl.begellhouse.com/journals/558048804a15188a,498820861ef102d2,1255e053242c9a40.html

“ChatGPT for programming numerical methods”

It was just published in 2023, a few months after releasing ChatGPT by OpenAI.

In pages 23 and 27 of this journal paper, the authors evaluated the performance of ChatGPT for coding the 1D compressible flow and in pages 57, 58, and 59, they investigated the performance of ChatGPT for coding 2D incompressible flow in C++. These are two examples for CFD.

There are other journal papers that have been missing from this manuscript, for example see:

https://www.sciencedirect.com/science/article/pii/S2095034925000157

“DeepSeek vs. ChatGPT vs. Claude: A comparative study for scientific computing and scientific machine learning tasks”

A few more points to be considered:

--> The quality of Figure 1 is low, and some text cannot be read.

--> In Figure 2, we observe that the authors listed the performance of o3-mini, which has been removed by Open AI for a few months and instead we have GPT 5, which has been missed from the current manuscript. Of course, I understand that this is because the fast movement of AI technology, but this fact supports me total judgment about the manuscript, that this is simply a technical report and not a scientific paper suitable for publication in ICLR 2026. Perhaps, it could be considered for the workshop sections.

### Questions
The state of art of using LLM for CFD is to develop domain-knowledge LLM for CFD. I suggest considering this direction. Evaluating the current LLMs and providing benchmarks is good, but it is not really the concern of CFD community. My main questions and concerns have been listed in the previous box. Thanks for your submission to ICLR.

### Soundness
2

### Presentation
2

### Contribution
1
