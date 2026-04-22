# QuArch: A Benchmark for Evaluating LLM Reasoning in Computer Architecture

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 4

## Abstract
The field of computer architecture, which bridges high-level software abstractions and low-level hardware implementations, remains absent from current large language model (LLM) evaluations. To this end, we present QuArch (pronounced 'quark'), the first benchmark designed to facilitate the development and evaluation of LLM knowledge and reasoning capabilities specifically in computer architecture. QuArch provides a comprehensive collection of 2,671 expert-validated question-answer (QA) pairs covering various aspects of computer architecture, including processor design, memory systems, and interconnection networks. Our evaluation reveals that while frontier models possess domain-specific knowledge, they struggle with skills that require higher-order thinking in computer architecture. Frontier model accuracies vary widely (from 34% to 72%) on these advanced questions, highlighting persistent gaps in architectural reasoning across analysis, design, and implementation QAs. By holistically assessing fundamental skills, QuArch provides a foundation for building and measuring LLM capabilities that can accelerate innovation in computing systems.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces QUARCH, the first benchmark designed to evaluate the knowledge and reasoning capabilities of Large Language Models (LLMs) in the field of computer architecture. The benchmark contains 2,671 expert-validated question-answer (QA) pairs. These questions cover key areas like processor design, memory systems, and interconnection networks. The benchmark evaluation reveals that LLMs generally struggle with questions requiring higher-order thinking

### Strengths
1) This work presents an interesting benchmark for reasoning in computer architectures

2) This benchmark helps to demonstrate the shortcoming of LLMs in high-order thinking.

### Weaknesses
1) According to the sources of the benchmark, there is limited evidence that the QA tasks reflect the iterative, data-driven decision-making and trade-off analysis that human architects perform. This limits ecological validity — i.e., how well the benchmark reflects real-world reasoning in architectural design.

2) Despite the efforts, the benchmark’s scope is restricted to a relatively small set of expert-validated questions and may not scale to cover the full breadth of computer architectural domains.  

3) The work argues that architectural reasoning evaluation will accelerate innovation in system design, yet provides no evidence or use cases showing how improved benchmark performance correlates with tangible advances in computer architecture research or practice.

### Questions
see the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes QuArch, a benchmark on assessing LLMs in designing computer architectures. This benchmark goes beyond the prevailing benchmarks in software design and contributes to bridging the gap in hardware designs. The authors construct and validate the dataset with experts, ensuring the dataset is of high quality. Using this dataset, the authors benchmark state-of-the-art models and reveal that although some models perform quite well on the benchmark, some others do not. They also analyze and provide insights on common behavior patterns of LLMs in this task.

### Strengths
1. The investigated problem is interesting: there are many datasets on software design, but it is also important to understand whether we can use LLMs in hardware design.

2. The dataset consists of more than 2k QA pairs validated by experts, providing a large scale testbed with high-quality data instances for researchers to understand and improve model behaviors.

3. The paper provides interesting insights for LLMs' common behavior patterns on this benchmark. These insights make it possible to understand/predict to some extent how LLMs will perform in related tasks. It also makes it easier for researchers to design remedies strategically to further improve LLM performance.

### Weaknesses
1. The head room of the benchmark does not seem to be very big. GPT-5 is already having accuracy of above 70 on the reasoning task and around 90 on the recall task. It seems to me that even if we can improve model performance, the space for improvement is quite small. Considering that benchmarks recently tend to be easily saturated with increasingly powerful models, I'm a bit worried about for how long the benchmark can be used.

2. Is there any method we can use to improve LLM performance in related tasks? It will be really useful if the authors can briefly discuss what methods an potentially be helpful, either in inference or training time.

### Questions
Authors are encouraged to address concerns above.

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
3

### Summary
This paper introduces QuArch, the first benchmark specifically designed to evaluate the reasoning capabilities of LLMs in the field of computer architecture. The authors address the absence of specialized evaluation tools in this domain by creating a comprehensive dataset of 2,671 expert-validated question-answer pairs. Through the evaluation of ten frontier LLMs, the study reveals that these models exhibit considerable weaknesses in higher-order reasoning tasks such as analysis, design, and implementation.

### Strengths
1. This work addresses the lack of LLM evaluation benchmarks in computer architecture. The inclusion of expert validation for all 2,671 question-answer pairs lends credibility to the dataset's quality and technical accuracy.
2. The authors evaluate a wide range of frontier LLMs.

### Weaknesses
The "Implement" skill is tested by asking models to produce artifacts like code or simulation scripts. This is a step in the right direction, but it's a simplified version of real-world implementation, which involves complex toolchains, debugging, verification, and performance tuning that cannot be fully captured in this format.

### Questions
The expert validation is a key strength. Could you elaborate on the inter-rater reliability among the experts who validated the QAs? Were there specific topics or question types that caused more disagreement?

### Soundness
3

### Presentation
3

### Contribution
2
