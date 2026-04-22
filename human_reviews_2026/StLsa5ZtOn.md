# EngiBench: A Benchmark for Evaluating Large Language Models on Engineering Problem Solving

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 4, 2, 2

## Abstract
Large language models (LLMs) have shown strong performance on mathematical reasoning under well-posed conditions. However, real-world engineering problems require more than mathematical symbolic computation—they need to deal with uncertainty, context, and open-ended scenarios. Existing benchmarks fail to capture these complexities. We introduce EngiBench, a hierarchical benchmark designed to evaluate LLMs on solving engineering problems. It spans three levels of increasing difficulty (foundational knowledge retrieval, multi-step contextual reasoning, and open-ended modeling) and covers diverse engineering subfields. To facilitate a deeper understanding of model performance, we systematically rewrite each problem into three controlled variants (perturbed, knowledge-enhanced, and math abstraction), enabling us to separately evaluate the model's robustness, domain-specific knowledge, and mathematical reasoning abilities. Experiment results reveal a clear performance gap across levels: models struggle more as tasks get harder, perform worse when problems are slightly changed, and fall far behind human experts on the high-level engineering tasks. These findings reveal that current LLMs still lack the high-level reasoning needed for real-world engineering, highlighting the need for future models with deeper and more reliable problem-solving capabilities. Our source code and data are available at https://anonymous.4open.science/r/EngiBench-05DF.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces EngiBench, a benchmark for evaluating LLMs on solving real-world engineering problems. The benchmark spans three difficulty levels 1) foundational knowledge retrieval, 2) contextual reasoning, and 3) open-ended modeling. Together, they cover multiple engineering subfields including systems & control, physical & structural, and chemical/biological. Each problem is further rewritten into three controlled variants to diagnose robustness, knowledge use, and mathematical reasoning respectively.

The experiments contain 16 latest models, including GPT-4.1, Claude 3.7, Gemini 2.5, DeepSeek-V3, and open-source models like Qwen2.5 and Llama 4. Benchmarking results reveal that LLMs perform well on simple, structured problems but degrade significantly as tasks become open-ended. Perturbations cause notable accuracy drops, highlighting weak generalization, and even SOTA models fall short of human experts on Level 3 tasks. Overall, EngiBench demonstrates that LLMs lack the reliable high-level reasoning required for real-world engineering problems.

### Strengths
- The problem framing is practical and novel. It moves beyond symbolic reasoning toward real-world engineering problems, which was previously underexplored.

- The design with three difficulty levels enables granular capability analysis across different dimensions.

- The comparison is comprehensive, including SOTA models and human expert baselines.
	
- Codebase is submitted for reproducibility.

### Weaknesses
Overall, I am positive about the paper's contribution. Some minor weaknesses include: 

- Scoring rubrics are constructed using LLMs to transform raw scoring descriptions into a capability-oriented rubric, this step may potentially introduce LLM bias and may not generalize across all engineering domains.

- Open-ended evaluation and human scoring might limit scalability when this benchmark is further developed, for example, include multimodal information.

### Questions
- How sensitive are Level 3 scores to the rubric granularity?

- Did you observe specific subfields, for example, physical problems, where LLMs perform better or worse?

- What would you expect about integrating multimodal information, like figures or diagrams? Will that change the conclusions about reasoning depth?

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
4

### Summary
This paper introduces EngiBench, a benchmark designed to evaluate LLMs’ ability to solve real-world engineering problems, which require not only mathematical computation but also contextual understanding, multi-objective trade-offs, and uncertainty handling.
The benchmark covers three hierarchical difficulty levels (from foundational engineering knowledge to open-ended modeling tasks) and spans multiple engineering domains. The authors further construct perturbed, knowledge-enhanced, and math-abstraction variants for fine-grained diagnosis of model weaknesses such as robustness or missing domain knowledge. They evaluate a wide range of LLMs and observe a strong performance drop when moving to more realistic and open-ended tasks. Overall, the benchmark exposes key limitations of current LLMs in engineering problem-solving.

### Strengths
1.The paper addresses an important but underexplored question: Do LLMs actually have the ability to solve real engineering problems, rather than just math? This goes beyond existing benchmarks focused solely on abstract mathematics.

2. Incorporation of open-ended engineering questions: Level-3 tasks reflect real-world scenarios with incomplete constraints and trade-offs, making the benchmark more practical and challenging.

3. The benchmark includes multiple problem variations, enabling detailed analysis of robustness, mathematical abstraction, and knowledge gaps.

4. Diverse model evaluation: A strong suite of models—across sizes and accessibility—provides credible comparison and demonstrates wide applicability.

### Weaknesses
1.Lack of evaluation with math-specialized or science-oriented models,
It would be insightful to test models like Math-specialized LLMs or scientific reasoning model to assess whether their strong math/science capabilities transfer to engineering problems.

2.Missing training-side investigations,
Since Level-3 is extremely challenging, the authors could explore how fine-tuning or data augmentation improves performance, offering insights into how LLMs might better acquire engineering reasoning.

3. Presentation issues:Some visual elements reduce readability:
-Reference hyperlinks in bright green are visually distracting
-Figure 4 font size is too small to read clearly
Better formatting would improve the overall user experience.

### Questions
Mention in aboved issues.

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
4

### Summary
This paper introduces EngiBench, a new benchmark designed to evaluate large language models (LLMs) on their ability to solve real-world engineering problems. The authors argue that existing benchmarks focus too narrowly on well-posed mathematical problems and fail to capture the complexities of real-world engineering, such as uncertainty, context, and open-ended scenarios.
EngiBench is a hierarchical benchmark with three levels of increasing difficulty:
Level 1: Foundational knowledge retrieval, which involves applying basic formulas in a single step.
Level 2: Multi-step contextual reasoning under defined constraints.
Level 3: Open-ended modeling, requiring information extraction, domain-specific reasoning, multi-objective decision-making, and uncertainty handling. 
To conduct a fine-grained analysis of model performance, the problems in Levels 1 and 2 are rewritten into three controlled variants: perturbed, knowledge-enhanced, and math abstraction. This allows the evaluation of a model's robustness, domain-specific knowledge, and mathematical reasoning abilities separately.
The paper's experiments show a clear performance gap across these difficulty levels, with models struggling more as tasks become harder. Results reveal that even state-of-the-art LLMs perform significantly worse than human experts on the high-level, open-ended tasks in Level 3, indicating that they currently lack the necessary high-level reasoning for real-world engineering problems.

### Strengths
- The idea of creating a benchmark focused on real-world engineering problems is good and rooted in application of LLMs in practical settings. The creation of a hierarchical benchmark, EngiBench, with three distinct difficulty levels from foundational knowledge retrieval to open-ended modeling is also good.
- The methodology of systematically generating controlled variants for each problem is interesting and acts as a diagnostic tool to pinpoint specific weaknesses in models.
- For the complex, open-ended tasks in Level 3, the use of expert-designed scoring rubrics and annotation by PhD students and professionals provides a reliable and rigorous evaluation framework.
- Evaluation across diverse models and task levels is thorough and the controlled variants help in analysing specific gaps in models and bring out insights.

### Weaknesses
- Level 1 and 2 problems are borrowed from existing benchmark datasets and follow a simple binary scoring evaluation. The non-trivial contribution is mainly the level 3 problems. This limits the overall contribution of the work. This does not agree with the authors' initial motivation that highlighting the gaps in the existing benchmarks.
- The paper acknowledges that many authentic engineering problems are inherently multimodal, relying on visual elements such as diagrams, schematics, and structured tables. By restricting the benchmark to text-only formats to ensure consistency, the evaluation overlooks a crucial aspect of engineering reasoning that involves interpreting and integrating information from both visual and textual sources.
- It is unclear if the benchmark covers long-context problems. Real-world engineering tasks often involve navigating extensive documentation and large datasets.
- Some of the insights around models performing better on knowledge-enhanced variants or low reasoning capabilities of smaller-scale models are fairly acknowledged already.

### Questions
- On page 7, you say that - "we introduce perturbed variants that modify surface details but keep the core structure unchanged" -- isn't this  already done as part of the dataset creation?
- There's some repetition in the writing. For instance, "We define this as engineering problem-solving ability, comprising four key dimensions: information extraction, domain-specific reasoning, multi-objective decision-making, and uncertainty handling" - this is introduced multiple times.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Authors introduces EngiBench, a hierarchical benchmark to test large language models (LLMs) on engineering-related tasks. The aim is to go beyond traditional math based tasks and involve various practical nuances that comes with engineering domain problems. EngiBench spans three levels of difficulty: namely, 1) foundational knowledge retrieval, 2) contextual reasoning, and 3) open-ended modeling. It also covers various engineering fields. Each problem has controlled variants (perturbed, knowledge-enhanced, and math abstraction) that isolate different reasoning capabilities, such as robustness, domain knowledge, and mathematical reasoning. Experimental results with sixteen major LLMs show a clear decline in performance from basic computation to open-ended problem-solving.

### Strengths
1. The specific focus on diverse engineering problem is very timely as "AI for science" work is catching on. 

2. Benchmark is structured with controlled problem variations that allow fine-grained evaluation (modulo concerns below)
3. Employs rubric-based evaluation for open-ended tasks
4. Authors have experimented with many models

### Weaknesses
1. The introduction is all about various engineering problems. Suddenly, we see in dataset construction, that the lower level problems actually consists of general domain problems from datasets such as SuperGPQA, GSM8k etc. Frankly, this makes me completely confused about the objectives of the work. 
2. Very little detail about human annotations (question below).
3. Level 3 seems to be the only important aspect of the paper. Its evaluation has no justification in the paper. Authors c ould have spent more time on justifiying why the metrics were chosen. Maybe show correlation with human evaluations etc.

Overall, while the paper discusses an important problem, its execution is immensely poor.

### Questions
L99: One problem is, upto the entire introduction, I do not find a proper example which shows the representative complexity of engineering problems.

L214: This is a very blanket remark for a very large field and not true at all. Starting from NLP specific benchmarks (SuperGLUE, TaxiNLI), to math and generalist benchmarks, it has been a natural choice to follow well-defined taxonomy. TaxiNLI, GSMore (Hong et al. 2025) goes further to define an ontology.

L249: I do not understand whats the benefit of having standard math questions at Levels 1 and 2. Why consider them as part of a generic engineering benchmark at all?

L258: How was the annotation process carried out? Was each problem looked at by multiple annotators? 
How varied where their expertise (or performance on this dataset itself), given number is quite large?   -- say for example, did you give questions annotated by others to a student and verified whether he/she could answer correctly?
Lot of details are unclear.

L292: You say it is impractical. Yet, you have given a motivating example from an engineering problem.

### Soundness
2

### Presentation
2

### Contribution
2
