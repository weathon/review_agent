# CrochetBench: Can Vision-Language Models Move from Describing to Doing in Crochet Domain?

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 4, 6

## Abstract
We present CrochetBench, a benchmark for evaluating the ability of multimodal large language models  to perform fine-grained, low-level procedural reasoning in the domain of crochet. Unlike prior benchmarks that focus on high-level description or visual question answering, CrochetBench shifts the emphasis from describing to doing: models are required to recognize stitches, select structurally appropriate instructions, and generate compilable crochet procedures. We adopt the CrochetPARADE DSL as our intermediate representation, enabling structural validation and functional evaluation via execution. The benchmark covers tasks including stitch classification, instruction grounding, and both natural language and image-to-DSL translation. Across all tasks, performance sharply declines as the evaluation shifts from surface-level similarity to executable correctness, exposing limitations in long-range symbolic reasoning and 3D-aware procedural synthesis. CrochetBench offers a new lens for assessing procedural competence in multimodal models and highlights the gap between surface-level understanding and executable precision in real-world creative domains.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper introduces CrochetBench, a benchmark for evaluating the ability of VLMs to perform fine-grained, low-level procedural reasoning in the domain of crochet. In CrochetBench, models are required to recognize stitches, select structurally appropriate instructions, and generate compilable crochet procedures. The benchmark includes 4 types of tasks, including stitch classification, instruction grounding, and both natural language and image-to-DSL translation.

### Strengths
The benchmark built around the crochet domain is conceptually interesting and could potentially introduce new challenges for VLMs.

### Weaknesses
I believe this paper has significant issues in its presentation, which makes it hard to follow and understand, and therefore difficult to assess its contribution.

- It is hard to follow most parts of the paper. There are no examples or figures to help readers understand what the benchmark is assessing. Given that this is a highly specialized domain (crochet), many readers from the ICLR community may not have the relevant background knowledge. The only section that can somehow know the background is in the introduction, but immediately after that, the paper shifts to dataset statistics and experiments. Because of the poor presentation, it is very difficult to understand the work and, consequently, to evaluate its contribution. I suggest the author to provide sufficient background in the main paper and polish the writing.

- The four task types are not described clearly. The paper only mentions what ability each task is supposed to test, without explaining in detail what the model is actually being asked to do. The descriptions rely heavily on complex terminology without clarifying their meaning. For example, in Lines 140–141, the paper says: “Task A (Stitch Recognition) evaluates a model’s ability to detect symbolic primitives in crochet images, establishing the foundation for multimodal perception.” However, it is unclear what “symbolic primitives” means. 

- The paper uses many uncommon or domain-specific terms without explanation --  for example, “procedural crafts,” “stitch abbreviations,” and “counts.”

- The paper only presents the performance, no detailed analysis why models fail and no insights. Despite Table 8 provide the error analysis, no description of how failure analysis is performed or conducted.  

- Several table references are incorrect or missing (e.g., Lines 98 and 106), which further adds to the confusion.

### Questions
The use of LLMs is not disclosed anywhere in the paper. Did the authors use an LLM for writing the paper, and if so, to what extent was it used?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces CrochetBench, a benchmark designed to evaluate the ability of multimodal large language models to perform fine-grained, low-level procedural reasoning in the domain of crochet. Unlike prior benchmarks focused on description, CrochetBench emphasizes execution, requiring models to recognize stitches, generate structured instructions, and translate natural language or images into CrochetPARADE DSL. The dataset includes tasks such as stitch classification, instruction selection, natural language generation, and DSL translation, with outputs evaluated for executable correctness. This paper reveal significant gaps between surface-level understanding and executable precision.

### Strengths
- It is a new approach to employ crochet, a craft defined by its intricate structure and creativity, as a framework for evaluating a model's reasoning and code generation capabilities.
- This benchmark highlights the limitations of existing vision-language models.

### Weaknesses
- The data, only sourced from the Yarn spirations website, may be biased toward specific design styles or formats, limiting its diversity and representativeness.
- It is unclear whether the use of GPT-4o-mini for PDF conversion and annotation involved any manual error checking.
- The evaluation for Task D only focuses on compilation success, without comparing the geometric and topological similarity between the compiled output and the reference design.
- There is a progressive relationship between the tasks, such as Task B potentially relying on Task A, which could lead to error propagation. Exploring the dependencies between tasks is necessary to obtain more accurate indicators of reasoning capability.
- The evaluation relies on a limited number of models, and the open-source models are small.

### Questions
- There are issues with the table labels, as some references are incorrectly displayed as question marks.
- In Task C: Instruction Generation, performance is evaluated by comparing outputs to a reference answer; however, it is worth considering that a single product may have multiple valid crochet methods.

### Soundness
3

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
4

### Summary
The author proposes an interesting benchmark called CrochetBench, to test whether multi-modal language model can understand how to perform crochet, which is a interplay of different actions: recognize stitches, select structurally valid instructions, and generate compilable crochet procedures, effectively performing 3D-aware reasoning. When evaluating current models on the proposed benchmark, the authors find that their performance drops sharply when they must generate instructions that are actually executable and correct, revealing major limitations in their reasoning and procedural skills.

### Strengths
- The authors introduced an interesting task, CrochetPARADE DSL, as it does have this nice property of verifiable, meaning it could be beneficial for other tasks, for example post-training RL.
- I can see the challenge of crochet code generation, as it requires 3D-aware reasoning, and because it is a quite niche task, it is possible that current language models have not been trained on this tasks, making it less contaminated task and might better reflect model's performance differences

### Weaknesses
- the author emphasized that their benchmark focuses on the instruction fidelity, if the model can generate valid, compilable DSL code, based on multi-modal input, and  opens a new direction for multimodal research, which I don't think they are the first to do this: for example, the whole area of letting LLMs/multi-modal LLMs to generate symbolic graphics programs like SVG (2D), CAD (3D) etc. which fullfill all the requirements and properties of this crochet DSL, have already being studied before. I think this task sounds novel, but the contribution claim might not be accurate. At least the authors should provide a more comprehensive analysis to these prior works to thoroughly discuss why their DSL / benchmark is more suited to benchmark LLMs, or more into the details of the DSL differences
- minor errors in writings (e.g., table reference wrong in line 98 / 106), related work section in the appendix
- there needs more explanation about the benchmark generation process (what tasks have been done), to highlight the author's contribution, because currently the paper seems to be focused on the task and the DSL CrochetPARADE (which is not part of the paper) and the result analysis

### Questions
- crochet DSL generation benchmarking seems to be an interesting and I can imagine that it might be relevant to some readers / research, but it is a very niche task. This does not undermine the valid and significance of this task, but I think only if it has proven to be beneficial for generic visual / 3d reasoning, its significance remains limited. How can this kind of DSL/benchmark benefit generic multi-modal llms? for example through instruction finetuning, it improves the model's performance on CrochetBench, but for example, this performance gain is also valid in other multi-modal visual reasoning tasks, for example like geometric reasoning problems?
- the benchmark results is inconsistent,  the tasks A, B, C and D have three different best performing models? even open-sourced model can perform the best (table 7), and even outperforming the other models by a large margin, showing maybe the performance on this proposed CrochetBench is not generic or varies a lot, or even data contamination within the benchmark? this might not be a good property of a benchmark

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Benchmark for procedural crochet understanding: perception (stitch recognition), retrieval (instruction selection), text generation, and both natural language and image-to-DSL (CrochetPARADE) with compilation/execution as the main metric. Their core finding: surface text metrics don’t predict executability; program synthesis is the bottleneck.

### Strengths
1. Execution-grounded evaluation. CrochetPARADE enables syntactic/structural validation and visualization/execution, providing a more faithful signal than BLEU/ROUGE alone.
2. Well-structured task ladder. Tasks escalate from perception to executable synthesis with clear metrics and sizes.
3. Dataset scale & coverage. Table 1 reports 6,085 patterns, 98.77% image coverage, 55 project types.
4. Clear gap at execution. Performance “declines as evaluation shifts to executable correctness”, project-level CSR is low.
5. Crochet is an under-explored domain for LLM code generation and a good test bed.

### Weaknesses
1. Semantic equivalence vs. compilation. Compilation checks syntax/structure but can miss semantically equivalent programs. The authors motivate execution-based metrics but do not pair them with visual render agreement in main results.
2. More qualitative results are needed for better interpretation of the results.

### Questions
I hope the author can address my concerns in the weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
3
