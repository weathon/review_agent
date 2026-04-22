# KnowledgeSmith: Uncovering Knowledge Updating in LLMs with Model Editing and Unlearning

- Avg Score: 6.67
- Decision: Accept (Poster)
- Scores: 4, 8, 8

## Abstract
Knowledge editing and machine unlearning are two popular approaches for large language models (LLMs) to stay up-to-date. However, the knowledge updating mechanism of LLMs remains largely unexplored due to insufficient, isolated, and small-scale evaluation. For instance, are LLMs similar to humans in modifying certain knowledge? What differs editing and unlearning as training data increases? This paper proposes KnowledgeSmith, a unified framework to systematically understand the updating mechanism of LLMs. We first cast editing and unlearning as instances of one constrained optimization problem. Then, we propose an automatic dataset generator that provides structured interventions across multiple graph levels and data scales, enabling controlled studies of how different modification strategies propagate through model knowledge. Extensive experiments demonstrate nuanced insights over knowledge propagation, plasticity scaling, consistency, and robustness. For instance, our results show that LLMs do not exhibit similar updating as humans for different levels of knowledge, and there exists consistency-capacity trade-off. We hope our findings can offer suggestions to the design of more reliable and scalable strategies. Code: https://github.com/AIFrontierLab/KnowledgeSmith

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This article proposes a unified framework KnowledgeSmith to systematically study the knowledge update mechanism in Large Language Models (LLMs), with a particular focus on the performance and differences between knowledge editing and knowledge forgetting methods. The theoretical modeling of the paper is rigorous, the experimental design is meticulous, and the results are highly inspiring. The author not only proposed new evaluation metrics such as Consistency Collapse and Conflict Rate, but also constructed a structured and scalable knowledge intervention evaluation benchmark, filling the gap in current research on the lack of systematic understanding of knowledge update mechanisms.

### Strengths
- Novel Unified Framework: The paper introduces KnowledgeSmith, a conceptually novel and theoretically grounded framework that unifies knowledge editing and machine unlearning as instances of the same constrained optimization problem, offering a new lens for understanding and comparing these distinct but related knowledge updating mechanisms.
- Innovative Benchmark Generation: It proposes a highly original and automatic pipeline for generating large-scale, structured evaluation benchmarks from existing knowledge graphs . This method allows for systematic probing of knowledge updates across different hierarchical levels (root, intermediate, leaf) and data scales, addressing major limitations of previous static and isolated fact-based evaluations.
- Insightful Empirical Findings: The extensive experiments yield several significant and nuanced insights into LLM knowledge updating behavior, including propagation asymmetry (over-spreading vs. under-spreading) , plasticity limits related to hierarchy , the consistency-capacity trade-off , subject-dependent update resistance , and a unified taxonomy of failure modes.

### Weaknesses
- Limited Scope of Investigated Update Methods: The study primarily focuses on AlphaEdit for knowledge editing and ReLearn  for machine unlearning. While these represent state-of-the-art approaches, the conclusions drawn about knowledge updating mechanisms (e.g., propagation asymmetry , consistency collapse ) might be specific to these particular algorithmic paradigms (locate-then-edit vs. retraining-based unlearning). The framework's generalizability across a broader spectrum of editing techniques (e.g., memory-based methods like SERAC , gradient-based methods like MEND ) or different unlearning approaches (e.g., gradient ascent ) remains underexplored, potentially limiting the universality of the observed phenomena.
- Reliance on Multiple-Choice Question Format: The automatic benchmark generation pipeline exclusively creates multiple-choice questions derived from knowledge graph triples . While this format facilitates standardized evaluation and leverages existing datasets like MMLU, it may not fully capture the nuances of knowledge representation and retrieval in LLMs. Evaluating updates solely through multiple-choice accuracy might overlook impacts on generative capabilities, reasoning chains, or the model's ability to handle ambiguity, potentially offering an incomplete picture of the update's true effect.
- Over-reliance on GPT-4o for Benchmark Generation and Insufficient Validation : The KnowledgeSmith framework heavily relies on GPT-4o for crucial generation steps, including initial KG construction, template generation, and potentially probe/distractor creation . This dependency raises concerns regarding reproducibility and potential biases inherent in the generator model, which could influence the benchmark's structure and content. While Appendix A.6 mentions quality control including external validation and manual spot checks , the paper lacks quantitative analysis from systematic human studies assessing the quality, factual accuracy, and potential hallucination rates specifically within the GPT-4o generated components (e.g., generated KG relations, question templates, or multiple-choice distractors), making it difficult to fully gauge the reliability of the benchmark itself.
- Focus on Static Knowledge Updates : The current experimental design primarily investigates the effects of single-step, isolated knowledge editing or unlearning interventions based on a static KG structure. It does not explicitly address the challenges of dynamic knowledge updating in a continual learning setting, where knowledge evolves over time, and updates arrive sequentially. Consequently, the framework does not evaluate crucial aspects such as catastrophic forgetting across multiple sequential updates, the interaction between consecutive edits/unlearning operations, or how the observed mechanisms (like propagation or consistency trade-offs) might compound or change in a lifelong learning scenario

### Questions
- Does the observed propagation asymmetry (over-spreading in editing vs. under-spreading in unlearning ) stem primarily from the specific algorithms chosen (AlphaEdit/ReLearn), or is it a more fundamental characteristic inherent to the editing versus unlearning tasks themselves, regardless of the method?
- Could the exclusive reliance on multiple-choice questions mask certain effects of knowledge updates, and how might findings, particularly the consistency-capacity trade-off , differ if evaluated using open-ended generation tasks?
- Can the authors provide quantitative results from human evaluations assessing the factual accuracy, clarity, and distractor plausibility of the GPT-4o generated benchmark components (KG relations, question templates, multiple-choice options) ?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposed KnowledgeSmith, a unified framework and benchmark for studying knowledge updating in large language models via editing and unlearning. By framing both as constrained optimization problems and evaluating 13 LLMs on a hierarchical knowledge-graph benchmark, the study reveals asymmetric propagation, including editing over-spreads and unlearning under-spreads, a consistency–capacity trade-off, and shared failure modes. While the approach is conceptually rather than technically novel, it provides a valuable, systematic analysis of how LLMs modify and maintain knowledge.

### Strengths
- This is the 1st work to a unified theoretical formulation linking knowledge editing and unlearning under a single constrained optimization framework.
- The paper proposes a structured, hierarchical benchmark based on knowledge graphs, enabling fine-grained evaluation across root, intermediate, and leaf concepts.
- The paper provides a comprehensive empirical analysis of 13 LLMs and four domains, revealing consistent phenomena such as propagation asymmetry and the consistency–capacity trade-off.

### Weaknesses
The benchmark construction heavily relies on GPT-based data generation, which raises concerns about reproducibility and annotation reliability. Although Appendix A describes prompt structures and includes multi-stage validation with manual spot checks and cross-checks against encyclopedic sources, the extent of human verification and generation parameters is not fully specified, limiting strict reproducibility.

### Questions
- Could the authors provide more details about the benchmark generation pipeline to improve reproducibility? For example, how consistent are the outputs across different runs? How extensive was the manual verification process mentioned in Appendix A — approximately what proportion of generated items were manually checked, and by how many annotators?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper regards knowledge ”editing" and "forgetting" as a unified constrained optimization problem and studies the performance of these two tasks by constructing multi-level data automatically generated based on a knowledge graph.
They mainly discovered phenomena such as Propagation Asymmetry, Plasticity Scaling and Branch-dependent Limits, and consistence-capacity Trade-off. And finally, a theoretical analysis was conducted from the perspective of singular value decomposition.

### Strengths
This paper treats knowledge "editing" and "forgetting" as a unified constrained optimization problem. 
This perspective is not very novel since some prior work has viewed unlearning as a subset of editing. However, the author conducts extensive empirical analyses from this unified viewpoint.
The experiments cover 13 models across 6 families of LLMs, ranging from 1B to 123B parameters. 
These analyses provide valuable insights and offer useful suggestions for the current chaotic development of these two fields.

### Weaknesses
This work presents a unified perspective, and the author conducts extensive comparative experiments on editing and unlearning.  However, it seems that the framework does not offer additional guiding functions, and after reading the paper, I find it hard to grasp the significance of this unified framework.  
Moreover, this framework closely resembles the editing framework;  if we consider editing in a broader sense, forgetting could be seen as a special case of editing with the "target empty."  
Lastly, the paper respectively employs one method for both editing and unlearning, but the paradigms of methods in these two areas differ significantly, and this difference is not mentioned in the analysis.

### Questions
1. Do you think different editing or unlearning methods have a significant impact on the analysis results? Why?
2. What advantages does your framework have over the view that "the model editing method is regarded as a strong baseline for unlearning", or what's your opinion on this issue?

### Soundness
4

### Presentation
3

### Contribution
4
