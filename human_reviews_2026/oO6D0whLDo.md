# CLARC: C/C++ Benchmark for Robust Code Search

- Decision: Accept (Poster)
- Scores: 4, 4, 6

## Abstract
Efficient code retrieval is critical for developer productivity, yet existing benchmarks largely focus on Python and rarely stress-test robustness beyond superficial lexical cues. To address the gap, we introduce an automated pipeline for code search datasets and present CLARC, a C/C++ benchmark built from real-world GitHub repositories. CLARC contains 1,245 query-code pairs for evaluation and 5,472 pairs for training. The benchmark incorporates LLM-generated natural language queries validated through rigorous human scoring and hypothesis testing. To analyze contextual requirements effectively, our pipeline starts by ensuring code compilability. It then categorizes code snippets by dependency complexity, distinguishing whether the code relies on custom-defined types or helper functions. The pipeline also enables CLARC to stress-test retrieval robustness by introducing challenging settings, including identifier anonymization and compilation to low-level languages like Assembly and WebAssembly. Under these conditions, our evaluation of six state-of-the-art models reveals sharp drops in retrieval effectiveness. The experimental results highlight the models' persistent reliance on lexical features rather than code semantic understanding. Our dataset is publicly available at https://huggingface.co/datasets/ClarcTeam/CLARC.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
CLARC introduces a comprehensive C/C++ code search benchmark addressing the limitations in existing benchmarks. The benchmark contains 1,245 fully compilable query-code pairs sourced from popular GitHub repositories, categorized into three groups based on dependency complexity. Unlike previous benchmarks that focus primarily on Python, CLARC systematically evaluates model C/C++ code search through multiple settings: standard code, neutralized identifiers (generic placeholders), randomized identifiers, and low-level representations (Assembly/WebAssembly). The authors developed an automated pipeline using LLMs to generate natural language queries, validated through hypothesis testing against human expert annotations. The results demonstrate that current models perform poorly on low-level languages and lack robust understanding of code functionality.

### Strengths
1. **Fills gap for C/C++ retrieval**: First comprehensive C/C++ robustness benchmark addressing the field's Python bias with real-world compilable code
2. **Systematic robustness evaluation**: Structured testing across identifier anonymization and compilation settings isolates semantic understanding from lexical pattern matching
3. **Scalable automated methodology**: LLM-based query generation with statistical validation enables cost-effective benchmark expansion while reducing knowledge contamination

### Weaknesses
1. **Questionable prevalence of target language**
The paper claims C/C++ represents "industrially prevalent languages," but this assertion lacks supporting evidence. While C/C++ has importance in systems programming, languages like Java, JavaScript, Python, and C# arguably have broader industrial adoption across web development, enterprise applications, and data science. The authors should justify why C/C++ specifically addresses an industrial need, or broaden their claim to acknowledge the more diverse landscape of industrially relevant languages.

2. **Questionable use cases of benchmark setting**
The practical necessity of code retrieval across different abstraction levels is questionable. The anonymized identifier setting (func_a, var_b, etc.) represents an artificial scenario, which is contrasted by most professional developers that will follow naming conventions and write meaningful identifiers. Similarly, searching Assembly or WebAssembly code has limited real-world utility, as most developers work exclusively in high-level languages and rely on compilers for low-level translation.

3. **Limited Dataset Scale and Missing Training Components**
Despite introducing an automated pipeline, the benchmark contains only 1,245 query-code pairs, which is relatively small for modern machine learning evaluation. More critically, while the authors demonstrate automated dataset construction capabilities, they provide no training set. Given the automated nature of their pipeline, supplying training data would significantly enhance the benchmark's practical value for model development.

4. **Insufficient Validation of LLM-Generated Queries**
The quality assessment of LLM-generated descriptions relies on only 125 samples per category, which may not be representative of the full dataset's quality. More fundamentally, LLM-generated queries tend toward stylistic homogeneity, lacking the diversity found in real-world project descriptions and developer queries. This limitation reduces the benchmark's ecological validity and may not adequately test model robustness against natural query variation.

5. **Poor Results Presentation**
The experimental results are poorly organized across Tables 3-5, requiring excessive cross-referencing to compare model performance across different settings. A consolidated presentation showing all experimental conditions for each model/group combination would significantly improve readability and facilitate comparative analysis.

### Questions
While the work addresses the gap (lacking C/C++) in code search evaluation, the authors should reconsider the practical relevance of their chosen scenarios and significantly expand both the dataset size and validation methodology.

### Soundness
3

### Presentation
2

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
The paper introduces CLARC, a new benchmark for code search focused on C/C++. CLARC includes 1,245 compilable query-code pairs divided into groups of varying dependency complexity. It also provides several robustness settings, such as anonymized identifiers, randomized names, and compilation to Assembly or WebAssembly.

### Strengths
The motivation is solid and relevant. Focusing on C/C++ fills a clear gap in existing benchmarks. The dataset being fully compilable enhances reproducibility. The robustness settings are well-designed and informative. The automated data generation pipeline with statistical validation is technically sound. Experimental coverage is broad and clearly demonstrates the weakness of current models.

### Weaknesses
1. The dataset is small compared to existing large benchmarks (e.g., CodeSearchNet). 
2. The contribution is mainly engineering rather than conceptual. 
3. The analysis of why performance drops is shallow, with little insight into model behavior or representation. 
4. The LLM-generated query validation focuses on surface quality rather than semantic fidelity. 
5. The paper reads more like a dataset report than a research study, and the novelty is limited. Writing is clear but somewhat lengthy and repetitive.

### Questions
1. Have the authors tried fine-tuning models on CLARC to test whether robustness can be learned?
2. Can the anonymization and randomization pipelines be extended to other programming languages?
3. Is there any observed correlation between code complexity (e.g., cyclomatic complexity) and robustness degradation?

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
This paper introduces CLARC (C/C++ LAnguage Retrieval with Anonymized Code), a benchmark designed to evaluate the robustness of code search models on C/C++ code. The dataset consists of 1,245 query-code pairs and is categorized based on the complexity of the code dependencies. CLARC includes configurable settings like anonymized identifiers and low-level code representations (Assembly and WebAssembly) to test models across varying abstraction levels. The authors evaluate six state-of-the-art code search methods and demonstrate significant performance degradation when identifiers are anonymized or code is compiled into lower-level languages. Furthermore, the paper introduces an automated pipeline for scalable benchmark generation, making the dataset reusable and extensible for future work.

### Strengths
1. CLARC is a **novel benchmark** specifically for C/C++ code search, with unique settings for anonymized identifiers and low-level languages.
2. The experimental design is thorough, with a wide range of models tested across different evaluation settings. The inclusion of low-level language scenarios is a valuable addition.
3. The paper is mostly clear and well-structured, with good use of figures and tables to explain the methodology and results.
4. The work addresses a gap in code search research, particularly for industrial programming languages like C/C++, and the automated pipeline for benchmark generation has broad potential for future research.

### Weaknesses
1. While CLARC is comprehensive, the authors could explore **more complex real-world codebases** to further ensure the dataset's robustness.
2. While the automated benchmark generation pipeline is promising, the paper could offer a more in-depth discussion of how well this approach can scale to other programming languages or larger codebases.
3. The evaluation of low-level languages (Assembly, WebAssembly) is insightful but could benefit from a deeper analysis of why models struggle with such code (e.g., complexity of instruction sets, abstraction loss).

### Questions
1. Could the authors provide more details on the **scalability** of the automated benchmark generation pipeline, particularly when extended to other languages beyond C/C++?
2. The paper discusses performance degradation when identifiers are anonymized. Would the authors consider testing **semantic-based anonymization** (e.g., anonymizing function names based on their role) to better assess the models' understanding of code semantics?

### Soundness
3

### Presentation
3

### Contribution
3
