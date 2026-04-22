# Speak-to-Structure: Evaluating LLMs in Open-domain Natural Language-Driven Molecule Generation

- Avg Score: 4.67
- Decision: Reject
- Scores: 2, 6, 6

## Abstract
Recently, Large Language Models (LLMs) have shown great potential in natural language-driven molecule discovery. 
However, existing datasets and benchmarks for molecule-text alignment are predominantly built on a one-to-one mapping, measuring LLMs' ability to retrieve a single, pre-defined answer, rather than their creative potential to generate diverse, yet equally valid, molecular candidates.
To address this critical gap, we propose **S**peak-to-**S**tructure (**S$^2$-Bench**), 
the first benchmark to evaluate LLMs in open-domain natural language-driven molecule generation.
S$^2$-Bench is specifically designed for one-to-many relationships, challenging LLMs to demonstrate genuine molecular understanding and generation capabilities. 
Our benchmark includes three key tasks: molecule editing (**MolEdit**), molecule optimization (**MolOpt**), and customized molecule generation (**MolCustom**), each probing a different aspect of molecule discovery. 
We also introduce **OpenMolIns**, a large-scale instruction tuning dataset that enables Llama-3.1-8B to surpass the most powerful LLMs like GPT-4o and Claude-3.5 on S$^2$-Bench. 
Our comprehensive evaluation of 28 LLMs shifts the focus from simple pattern recall to realistic molecular design, paving the way for more capable LLMs in natural language-driven molecule discovery.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The author suggests that LLM-based molecular generation should adhere to a one-to-many principle and has constructed a benchmark dataset based on this principle. However, the core concepts and dataset construction method of this work are very similar to those in article [1]. However, the manuscript does not appropriately cite [1] or articulate the differences between the two works. Additionally, the LLMs evaluated in this work are all general-purpose LLMs. While they possess some understanding of SMILES data, there remains a significant gap with the molecular generation field. The author should assess more domain-specific LLMs.

[1] https://link.springer.com/article/10.1186/s12915-025-02200-3

### Strengths
The principle that molecular generation should adhere to a one-to-many approach is crucial, and the author has developed a benchmark based on this notion, evaluating it across multiple LLMs. However, the core concept and the method of constructing the benchmark in this paper are highly similar to those in work [1].

### Weaknesses
The methodology is highly similar to that in work [1] and lacks comprehensive research. The LLMs evaluated are all general-purpose models, with a noticeable absence of specialized domain-specific models.

### Questions
no

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces S²-Bench (Speak-to-Structure), a large-scale benchmark designed to evaluate the capability of LLMs in open-domain natural-language-driven molecular generation. Unlike traditional one-to-one text-to-SMILES datasets, S²-Bench adopts a one-to-many paradigm, allowing multiple valid molecular outputs for a single instruction to better mimic real-world chemical design. The benchmark includes three complementary tasks—MolEdit, MolOpt, and MolCustom—that respectively assess structural editing, property optimization, and constrained generation. To support this evaluation, the authors construct OpenMolIns, a 1.2-million-sample programmatically generated dataset of instruction–molecule pairs using RDKit-based property computation and LLM paraphrasing for linguistic diversity. The paper further proposes the Weighted Success Rate (WSR) metric to integrate chemical validity, success rate, and novelty. Experiments on 28 LLMs (e.g., GPT-4o, Claude-3.5, LLaMA-3.1, Qwen-2) reveal that current models struggle with true chemical reasoning, while instruction-tuned variants significantly outperform base models. Overall, the work provides a reproducible, scalable, and realistic benchmark for assessing how LLMs perform in molecular reasoning and creative design.

### Strengths
The paper makes a timely and impactful contribution by redefining how natural-language-driven molecule generation should be evaluated.  
(1) The benchmark is conceptually original, introducing the one-to-many mapping paradigm that better reflects chemical diversity and real-world design scenarios.  
(2) The dataset generation pipeline is well-engineered and reproducible, integrating chemical computation (RDKit) with LLM-based linguistic diversification.  
(3) The evaluation design—combining MolEdit, MolOpt, and MolCustom—offers a comprehensive assessment of LLMs’ structural reasoning and control capabilities.  
(4) Experiments are large-scale and convincing, covering both open and closed LLMs with consistent results.  
(5) The paper is clearly written, well-structured, and likely to serve as a standard benchmark for future research in chemical LLMs.

### Weaknesses
The main weakness lies in the limited methodological depth. While the benchmark is well-designed, the work does not provide theoretical insight into the relationship between language semantics and chemical structure reasoning. The Weighted Success Rate metric, though practical, appears heuristic, and its weighting choices are not empirically justified. The programmatic data generation may also introduce semantic drift between the instruction and molecule, especially after paraphrasing by LLMs. Additionally, the dataset relies exclusively on RDKit-computed properties, which may not reflect experimental conditions, and the benchmark focuses primarily on organic drug-like molecules, limiting generalization to other domains.

### Questions
Q1. Data reliability after paraphrasing  
Since all instructions are automatically generated and paraphrased by LLMs, how do the authors ensure that the final language remains semantically consistent with the intended molecular transformation? Would human verification or semantic-similarity filtering improve dataset fidelity?  

Q2. Distributional bias of molecular sources  
OpenMolIns is built mainly from ZINC, ChEMBL, and MOSES, which are biased toward drug-like molecules. Could this limit generalization to other chemical domains (e.g., materials or catalysts)?  

Q3. Scientific validity of the evaluation metric  
How are the weights in the Weighted Success Rate (WSR) determined, and do they align with human expert judgments of chemical success or usefulness?

Q4. Language understanding vs. template learning  
Since instructions are generated from deterministic templates, does S²-Bench truly measure language understanding or just template matching? Have the authors tested models on non-templated, free-form instructions?

Q5. Out-of-distribution generalization  
Have the authors evaluated the benchmark under compositional or OOD instructions, e.g., “Add a hydroxyl group while keeping molecular weight below 200”?

Q6. Alignment with real chemical measurements  
Since all target properties come from RDKit computations, how well do they correlate with experimental measurements? Could integrating experimental datasets improve realism?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces Speak-to-Structure, a benchmark for open-domain, one-to-many natural language–driven molecular design, comprising three task families, MolEdit, MolOpt, and MolCustom, which focus on precise editing, property-oriented optimization, and de novo constrained generation. It also releases OpenMolIns, an instruction-tuning dataset and reports results across 28 LLMs, showing that targeted one-to-one datasets overestimate real design ability while instruction-tuned models on OpenMolIns perform best.

### Strengths
1. The benchmark allows many valid molecules per prompt, not just one “right” answer, which is closer to how chemists actually design.
2. Success is checked automatically, and everything rolls up into a single headline number (WSR) so models are straightforward to compare.
3. Many models are evaluated side-by-side, revealing where current methods struggle (especially de-novo constraints) and showing that instruction-tuning on the released data can meaningfully boost performance.

### Weaknesses
1. The “weighted success rate” is computed as *success × one quality term* (similarity for MolEdit/MolOpt; novelty for MolCustom), then averaged uniformly across nine subtasks. Because there are no reported thresholds or sensitivity analyses, rankings may be unstable under this multiplicative choice and the equal subtask weights.
2. Prompts are generated from fixed templates, and MolCustom’s constraints largely boil down to counts of atoms, bonds, or functional groups. Important real-world specs—stereochemistry, 3D geometry, ring/bridge topology, and basic synthesizability—are not directly assessed.
3. The test molecules are drawn from Zinc-250K for convenience, and the builders pre-select which functional groups/atoms/bonds occur, including “normal/edge” subsets. These design choices can shift task difficulty and may not reflect the breadth of discovery-scale chemical space.

### Questions
Please refer to the weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3
