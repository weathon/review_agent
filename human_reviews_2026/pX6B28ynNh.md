# CMT-Benchmark: A Benchmark for Condensed Matter Theory Built by Expert Researchers

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 8, 6, 8, 4

## Abstract
Large language models (LLMs) have demonstrated remarkable progress in coding and mathematical problem-solving; however, evaluation on advanced research-level problems in the hard sciences remains scarce. 
To fill this gap, we present \cmt, a dataset of 50 original problems covering condensed matter theory (CMT) at the level of an expert researcher. The solution for these problems involve analytical and computational approaches commonly used in quantum many-body physics and classical statistical mechanics. The dataset has been designed and verified by a worldwide panel of expert researchers through a collaborative environment. 
Topics in the dataset include Hartree-Fock mean-field theory, exact diagonalization methods, quantum Monte Carlo sampling, density matrix renormalization group, quantum statistical mechanics, classical statistical mechanics, and model building. We evaluate different LLMs by programmatically checking LLM-generated solutions against expert-supplied ground truth. 
To verify LLMs performance at scale, we developed an automated machine-grading pipeline suitable for advanced physics research problems. 
For example, we handle non-commuting operators that are essential for quantum many-body problems by symbolic manipulation and normal ordering. 
Our evaluations show that frontier models struggle with all of the problems in the dataset, highlighting a gap in the physical reasoning skills of current LLMs. Notably, experts identified strategies for creating increasingly difficult problems by interacting with the LLMs and exploiting common failure modes. 
While the highest-performing model, GPT5, correctly solves 30\% of the problems, average performance across 17 models (GPT, Gemini, Claude, DeepSeek, and Llama classes) is only 11.4$\pm$2.1\%.  Moreover, our benchmark contains 18 problems that not a single one of the 17 models considered here can correctly solve, and 26 problems that are solved by at most one model. 
These currently unsolvable problems span the fields of Quantum Monte Carlo, Variational Monte Carlo, and Density Matrix Renormalization Group. 
Furthermore, we illustrate how incorrect answers sometimes violate fundamental symmetries or have unphysical scaling dimensions. We believe that this benchmark set provides valuable guidance for the future development of language models, aiming to achieve the goal of AI research assistants and tutors.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper presents a new and very difficult benchmark for physics problems in condensed matter theory. It evaluates state-of-the-art LLMs and finds that they cannot solve most of these problems.The problems are categorized into different categories and created by experts from across the world. The benchmark is intended to help develop a research assistant grade AI assistant in this field.

### Strengths
- new benchmark dataset for a field in which data is lacking
- should enable the development of stronger models for physics problems
- the problems are checked and created by experts

### Weaknesses
- small dataset: there are only 50 problems because they are manually created by experts
- no other weaknesses to be found

### Questions
- do the problems have different difficulty levels or are they all at approximately the same level?
- do the problems generalize well, is an LLM that is able to solve these problems expected to generally be good at solving physics problems?  - How do you assess the coverage of these 50 problems, are there redundancies or gaps?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces CMT-Benchmark, a dataset of 50 expert-level problems in condensed matter theory (CMT). The benchmark was built by an international panel of expert researchers at the level expected of strong grad students. The authors then built automated evaluation infrastructure, including a novel parser that can handle non-commutative operator expressions in CMT. The paper found that all frontier models have uniformly low performance in the benchmark.

### Strengths
1. I really like the creation of the benchmark process. By asking the human experts from different countries to submit the questions, i think this benchmark truly captures what it means to be an expert in CMT. Thus, it will be more convincing to believe that the progress in this benchmark will imply the progress in CMT research. I think this is a significant contribution to the community. 

2. The automated parsing and grading system is carefully-designed and quite impressive, particularly the handling of non-commutative operator algebra through symbolic manipulation and normal ordering.

3. The four detailed case studies (Sections A.1-A.4) provide valuable insights into specific LLM limitations: language-geometry gaps, over-reliance on textbook heuristics, failure to apply fundamental principles, and weak spatial reasoning.

### Weaknesses
1. The paper only tests LLM capabilities without access to tools like web search, code executions or symbolic/numerical computation packages. However,  we know that even human research assistants don't work in isolation without any tool use. It would be interesting to test whether tool-augmented agents improve performance.

2. Requiring answers in boxed LaTeX environments and prohibiting new variables may artificially hurt model performance. The authors note some models (particularly Gemini 2.5 Pro) "occasionally disregard the formatting instructions," leading to parsing failures. How much does the strict format requirement degrade performance compared to free-form responses that could be human-evaluated on a subset?

3. While the authors claim problems are original, there's no systematic verification that similar problems (or solution strategies) don't appear in training data or online.

4. With only 50 problems and some categories having very few examples (PEPS has 3, VMC has 2), per-category performance can have large uncertainty.

### Questions
1. Can you design experiments to show that how much of the poor performance is due to formatting constraints versus actual physics reasoning failures?

2. Can tool-augmented agents substantially improve performance on this benchmark?

3. Is there evidence of training data contamination for any problems? How to design the experiments to test this?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents CMT-Benchmark, a dataset of 50 problems and expert-annotated answers in condensed matter theory. A variety of large language models are evaluated on this benchmark dataset, and they show a high gap to a satisfying physical reasoning skills.

### Strengths
- Create a very high-quality dataset in condensed matter domain. This dataset could potentially be very useful in evaluating LLM's capability in solving scientific research problems.
- Provide comprehensive benchmarking of multiple LLMs including GPT, Claude, DeepSeek and Llama family models.
- The writing of this paper is good, providing details in data curation process and necessary background knowledge.

### Weaknesses
- To improve the quality of experiments, authors are encouraged to analyze the failure cases, demonstrating LLM models consistently fails on which types of problems and makes which types of mistakes.
- As the benchmark is highly domain-specific, authors are encouraged to also evaluate the performance of web search agents (e.g., OpenAI & Tongyi DeepResearch) to show if LLM models could solve the problem through searching public Internet knowledge.

### Questions
Though it is released close to the ICLR submission date, I am interested in the comparison of CMT-Benchmark to another public benchmark CMPhysBench [1], which is also used to evaluate LLM in condensed matter physics. What are the major differences between them in the collected problems and answer annotation process?

[1] Wang, Weida, et al. "CMPhysBench: A Benchmark for Evaluating Large Language Models in Condensed Matter Physics." arXiv preprint arXiv:2508.18124 (2025).

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper curated a benchmark including 50 problems on condensed matter that are generated by domain experts. The benchmark covers a wide range of topics in the field and requires high levels of understanding and expertise to solve the problems. The LLMs perform badly on the benchmark.

### Strengths
1. The curated benchmark involves experts inputs and is carefully designed and evaluated.

2. The benchmark covers a wide range of topics in the condensed matter field.

### Weaknesses
1. The models evaluated are all general purpose LLMs that have not been fine tuned in the field of condensed matter. Therefore, they are unlikely to perform well on this challenging benchmark. The authors may want to fine tune a model on relevant tasks and then evaluate it on the benchmark to see if performance can be improved.

2. The benchmark only includes question and answer, without reasoning process. For such complicated tasks, few-shot prompts and a CoT guide may improve the performance. I think this would be valuable to test.

### Questions
1. Have the authors tried to fine tune some LLMs to learn to solve problems as challenging as the benchmark?

2. Would it possible to include reasoning process into the benchmark that can serve as the prompt or a reasoning benchmark?

### Soundness
3

### Presentation
3

### Contribution
3
