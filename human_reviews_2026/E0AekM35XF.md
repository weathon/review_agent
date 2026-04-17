# CTIArena: Benchmarking LLM Knowledge and Reasoning across Heterogeneous Cyber Threat Intelligence

- Decision: Reject
- Scores: 2, 2, 4

## Abstract
Cyber threat intelligence (CTI) is central to modern cybersecurity, providing critical insights for detecting and mitigating evolving threats. 
With the natural language understanding and reasoning capabilities of large language models (LLMs), there is increasing interest in applying them to CTI, which calls for benchmarks that can rigorously evaluate their performance.
Several early efforts have studied LLMs on some CTI tasks but remain limited: (i) they adopt only closed-book settings, relying on parametric knowledge without leveraging CTI knowledge bases; (ii) they cover only a narrow set of tasks, lacking a systematic view of the CTI landscape; and (iii) they restrict evaluation to single-source analysis, unlike realistic scenarios that require reasoning across multiple sources.
To fill these gaps, we present CTIArena, the first benchmark for evaluating LLM performance on heterogeneous, multi-source CTI under knowledge-augmented settings.
CTIArena spans three categories, structured, unstructured, and hybrid, further divided into nine tasks that capture the breadth of CTI analysis in modern security operations.
We evaluate ten widely used LLMs and find that most struggle in closed-book setups but show noticeable gains when augmented with security-specific knowledge through our designed retrieval-augmented techniques.
These findings highlight the limitations of general-purpose LLMs and the need for domain-tailored techniques to fully unlock their potential for CTI.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper introduces CTIArena, a benchmark designed to evaluate LLMs on diverse CTI tasks under both closed-book and knowledge-augmented settings. It defines nine tasks across structured, unstructured, and hybrid categories: spanning mappings among CVE, CWE, CAPEC, and ATT&CK taxonomies. The authors evaluate ten state-of-the-art LLMs and demonstrate that while general-purpose models perform poorly in closed-book CTI reasoning, performance improves significantly with security-specific retrieval methods.

### Strengths
The paper offers limited strengths in terms of novelty, clarity, or significance. While it identifies several limitations of prior CTI benchmarks, such as CTIBench and SEvenLLM, including their narrow task coverage and lack of retrieval-augmented evaluation, it does not necessarily address these limitations. Although the authors aim to develop a more comprehensive benchmark for CTI reasoning, as discussed below, the resulting CTIArena framework still falls short of constituting a true “reasoning” benchmark for CTI contexts.

### Weaknesses
The main limitation of this work is that CTIArena is not truly a benchmark for evaluating LLM reasoning. The included tasks primarily measure an LLM’s ability to extract or retrieve known knowledge, rather than perform any form of reasoning or inference. Consequently, the large performance gains observed with retrieval-augmented generation (RAG) are unsurprising, as once supporting documents are provided, the tasks essentially become lookup problems. For instance, in the RCM (Root Cause Mapping) structured task, the question simply asks for the CWE ID corresponding to a given CVE ID, without including any contextual information or description. Such a task can be trivially solved via a direct database or Google search, and since many CVEs postdate the training cutoff of the evaluated LLMs, it is expected that they would fail without retrieval access. This design does not test reasoning ability, nor is an LLM the appropriate tool for it. In contrast, CTIBench defines RCM using CVE descriptions rather than IDs, requiring models to infer the underlying weakness (CWE) from textual evidence, thus addressing a practical and reasoning-oriented use case for analyzing newly discovered vulnerabilities. CTIArena completely overlooks this reasoning aspect in its task design. 

Moreover, the paper makes several unsubstantiated claims, such as stating that CTIBench contains only 150 manually annotated queries, whereas in reality the dataset is substantially larger. For example, the RCM task alone includes around 1,000 questions, while CTIArena itself contains only 691 total questions. Moreover, the two hybrid tasks, CTI-VCA and CTI-ATA, are very similar in nature to the CTI-RCM and CTI-ATE tasks from CTIBench.

### Questions
I would recommend that the authors either revise the task designs to better capture and evaluate the reasoning abilities of LLMs in CTI contexts by incorporating tasks that require inference, abstraction, or contextual understanding, rather than factual lookup or reframe the benchmark’s stated goal to focus explicitly on assessing whether LLMs can retrieve and align CTI concepts from heterogeneous knowledge sources.

### Soundness
1

### Presentation
2

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
This work benchmarks how well LLMs understand and reason over CTI from different sources. The benchmark includes 9 tasks across structured, unstructured, and hybrid CTI settings, using 691 question–answer pairs. The authors evaluated 10 popular LLMs under both closed-book and knowledge-augmented settings.

### Strengths
1. The work covers structured, unstructured, and hybrid CTI tasks.

2. The tasks reflect what security analysts actually do (e.g., mapping CVEs to weaknesses, profiling threat actors).

### Weaknesses
1. The benchmark contains only 691 QA instances, which may not be enough to reflect the complex and ever-evolving nature of CTI. 

2. Much of the dataset is produced through LLM prompting with human filtering rather than entirely human-authored. It is thus unclear  about biases introduced during such a semi-automation process.

3. Using fixed templates with predefined associations may lack the linguistic and structural diversity seen in real-world CTI queries. This may limit how well the benchmark tests genuine reasoning ability versus pattern-matching on fixed formats.

4. The benchmark assumes that entities like CVEs, malware, or threat actor names can be cleanly mapped across sources using templates and matching rules. In reality, CTI is full of conflicts and ambiguous or conflicting evidence, which the benchmark largely ignores rather than tackles directly.

### Questions
Please see the comments above.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
(1) The paper introduces a benchmark for CTI reasoning. The benchmark is constructed through a 3-stage process:
- stage 1: annotate correlation among different sources
- stage 2: generating QA tasks 
- stage 3: human-curator 

(2) Empirical evaluation is performed by comparing several target models under different setups (closed-domain vs. knowledge-augmented), as shown in Table II.

(3) Based on these experiments, the paper presents several observations about CTI reasoning performance and challenges.

### Strengths
- The proposed benchmark is timely and relevant, contributing to standardized evaluation of cybersecurity-related agent.

- The proposed benchmark includes tasks that expand coverages of existing benchmarks.

### Weaknesses
- The paper employs GPT-5 as an LLM-based judge to assess data quality during dataset construction and as an automatic evaluator to score model predictions against reference answers in the testing phase. However, the authors also include GPT-5 in the performance comparison (Tables 2, 3, and 4), emphasizing its superiority. This practice may introduce evaluation bias, as the same model is used both as a judge and as a participant in the comparison.

- The evaluation process lacks detailed descriptions and does not include the evaluation of fine-tuned LLMs. In the Structured Tasks of Table 2, after injecting the related official entries of structured enumerations into the LLM, the performance improves significantly: from around 0 in the closed-book setting to around 1 with inference-time knowledge injection. There should be analysis to identify and verify the precise sources of the observed performance improvements.

- Interpretation of results is unclear. It is not entirely evident how to interpret Table II on the proposed  benchmark’s overall validity.

- Although Table I provides examples illustrating the benchmark’s coverage, there is no clear quantitative measure of how comprehensive or representative the benchmark is.

### Questions
- Identify the source of the significant  jump of improvement as mentioned earlier. 

- A quantitative comparison with other benchmark.

### Soundness
3

### Presentation
3

### Contribution
2
