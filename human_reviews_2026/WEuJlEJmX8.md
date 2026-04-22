# CircuitSense: A Hierarchical MLLM Benchmark Bridging Visual Comprehension and Symbolic Reasoning in Engineering Design Process

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 8

## Abstract
Engineering design operates through hierarchical abstraction from system specifications to component implementations, requiring visual understanding coupled with mathematical reasoning at each level. While Multi-modal Large Language Models (MLLMs) excel at natural image tasks, their ability to extract mathematical models from technical diagrams remains unexplored. We present \textbf{CircuitSense}, a comprehensive benchmark evaluating circuit understanding across this hierarchy through 8,006+ problems spanning component-level schematics to system-level block diagrams. Our benchmark uniquely examines the complete engineering workflow: Perception, Analysis, and Design, with a particular emphasis on the critical but underexplored capability of deriving symbolic equations from visual inputs. We introduce a hierarchical synthetic generation pipeline consisting of a grid-based schematic generator and a block diagram generator with auto-derived symbolic equation labels. Comprehensive evaluation of eight state-of-the-art MLLMs, including both closed-source and open-source models, reveals fundamental limitations in visual-to-mathematical reasoning. Closed-source models achieve over 85\% accuracy on perception tasks involving component recognition and topology identification, yet their performance on symbolic derivation and analytical reasoning falls below 19\%, exposing a critical gap between visual parsing and symbolic reasoning. Models with stronger symbolic reasoning capabilities consistently achieve higher design task accuracy, confirming the fundamental role of mathematical understanding in circuit synthesis and establishing symbolic reasoning as the key metric for engineering competence. Our synthetic pipeline code is available at \href{https://anonymous.4open.science/r/CircuitSense-8AC7/README.md}{URL}.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces CircuitSense, a benchmark designed to evaluate the circuit-understanding capabilities of multimodal large language models (MLLMs). CircuitSense comprises over 8,000 problems organized into a six-level hierarchical structure. The paper also presents an evaluation of six state-of-the-art MLLMs, revealing that while these models demonstrate strong visual perception, they perform catastrophically poorly in the symbolic reasoning tasks essential for circuit analysis and design.

### Strengths
1. The paper is well-written and easy to follow.

2. Addressing multimodal engineering problems is an important and timely topic for current MLLMs.

3. The proposed hierarchical synthetic generation pipeline is interesting.

### Weaknesses
1. Lack of discussion on related work (EEE-Bench [1]).
EEE-Bench is a comprehensive multimodal benchmark for electrical and electronics engineering, which is highly relevant to this paper. The authors fail to acknowledge or compare their work against it.

2. Overstated claims and limited novelty.
The paper claims to introduce the first multi-level visual-to-analytical benchmark. However, EEE-Bench has already covered comprehensive circuit analysis problems. The authors do not clearly articulate the differences between CircuitSense and EEE-Bench. In my view, the main tasks (circuit-related problems) in this paper largely overlap with those in EEE-Bench, suggesting limited novelty. Moreover, EEE-Bench addresses a broader domain of electrical and electronics engineering.

3. Insufficient evaluation.
The paper only evaluates six large scale models, which is not enough to provide a comprehensive assessment. More open-source small models and recent proprietary models—such as o3 and o4-mini—should be included.

4. Lack of new insights in the experimental results.
Beyond the introduction of the benchmark itself, the experiments do not provide any particularly novel findings. The observation that MLLMs excel in visual perception but struggle with symbolic reasoning is already well known.

5. Missing implementation details.
The paper does not provide sufficient information about the evaluation setup, such as the prompts used for answer extraction or accuracy calculation or other experimental configurations.

6. Inconsistent complexity-level design.
The benchmark defines multiple complexity levels, but the experimental results do not reflect the expected performance degradation with increasing difficulty. If model performance does not decrease with complexity, it raises questions about the validity and meaning of the defined complexity levels.

Reference

[1] EEE-Bench: A Comprehensive Multimodal Electrical and Electronics Engineering Benchmark. CVPR 2025.

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents CircuitSense, a comprehensive hierarchical benchmark for assessing visual-to-symbolic reasoning in circuit understanding.  It focuses on the core engineering ability to translate schematic and system diagrams into mathematical equations, which has not been tested in previous multimodal benchmarks.
CircuitSense has 8,006 problems organized into six hierarchy levels (resistor, RLC, small signal, transistor, block, and system) and three task categories (perception, analysis, and design).  The authors propose a synthetic generation pipeline based on SPICE simulation and Mason's gain formula for block diagrams to ensure ground-truth symbolic equations.
Experiments on six cutting-edge MLLMs (GPT-4o, Gemini-2.5-Pro, Claude-Sonnet-4, InternVL3, Qwen2.5-VL, GLM-4.5V) show that while closed-source models achieve >85% accuracy on perception tasks, their performance on symbolic derivation remains <19%, indicating a severe gap between visual parsing and mathematical reasoning.

### Strengths
- Novel Benchmark Scope  
CircuitSense combines perception, analysis, and design in a single dataset that reflects real-world circuit design workflows. The six-level hierarchy captures domain progression from physical components to system architecture, and provides a uniquely structured view of how reasoning degrades with abstraction.

-Rigorous Synthetic Generation Pipeline
The dual-stage pipeline (schematic and block diagram) ensures electrical validity and symbolic correctness. The use of SPICE simulation, Lcapy-based symbolic derivation, and Mason's gain formula ensures physics-consistent ground truths, not simply textual labels.

- Comprehensive Evaluation and Diagnostics
Testing six leading MLLMs with curated and synthetic problems (multiple-choice -- open-ended -- symbolic derivation) reveals how pattern-matching fails when symbolic reasoning is needed.

- Fine-grained Task Taxonomy 
The benchmark explicitly distinguishes perception subtasks (component detection, connection identification, function classification) from analytical derivation, allowing for the identification of the true bottleneck: symbolic manipulation.

- Strong Domain Validity 
Questions are sourced from reputable textbooks (Gray, Razavi, Holberg) and university courses to ensure educational and professional relevance.

### Weaknesses
- Limited Novelty in Technical Methodology 
Although the domain is novel, the benchmark creation adheres to standard dataset synthesis paradigms (grid-based placement, random topology, template verification).  The originality is primarily in hierarchical symbolic labeling, not algorithmic innovation. Limited Novelty in 

-Imbalanced Data Composition 
The dataset heavily favors analysis (approximately 7,000 samples) over design (~150 samples), making it difficult to statistically interpret design-task outcomes.

- Synthetic Domain Gap 
The significant drop in synthetic performance could be due to distribution shift rather than pure reasoning failure. The paper could better quantify the visual and algebraic difficulty of synthetic problems to address potential domain difference of the synthetic data.

-Limited Evaluation Diversity 
All tested models are general MLLMs, with no domain-specific baselines (such as SPICE-aware symbolic solvers or retrieval-augmented MLLMs) included.  This absense makes it unclear whether the failures are the result of poor general reasoning or a lack of domain knowledge.

### Questions
- Annotation Quality and Validation
How were the 2,986 curated problems verified?  Were multiple domain experts involved, and how did the annotators agree? Were symbolic derivations manually re-checked after SPICE/Lcapy generation to avoid mismatch errors?

- Synthetic Circuit Generator
How does the generator handle non-linear or transistor-level behavior that symbolic solvers cannot detect? Could the authors explain how "adaptive timeouts" were used to balance completeness and feasibility in symbolic derivation?

- Data Balance and Hierarchical Sampling
The design subset (157 samples) is relatively small.  Do the authors intend to scale this portion using synthetic design tasks (for example, automated topology synthesis with constraints)? Are the counts roughly equal across hierarchy levels, or are some levels (such as transistor or system) underrepresented?

- Table 7 shows a significant drop in output impedance derivation (8%).  Could you provide qualitative examples that demonstrate the specific algebraic or topological misunderstanding? Do models consistently misinterpret node labeling conventions (e.g., sign or orientation errors) or fail algebraic simplification?

- Robustness in Equation Comparisons
The symbolic comparison pipeline (SymPy plus numerical validation) is great. What is the failure rate? Were there instances where algebraically correct but simplified forms (e.g., factorized expressions) were mistakenly classified as incorrect?

- Potential Extensions: Do the authors intend to include temporal behaviors, such as time-domain simulations or frequency-sweep interpretation? Could integration with RAG/agent-based method serve as future baselines?

Appreciate the author's work and contributions to the community!

### Soundness
3

### Presentation
2

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
This paper introduces CircuitSense, a hierarchical benchmark designed to evaluate multimodal large language models on circuit understanding and symbolic reasoning tasks across over 8,000 problems. The study shows that while models perform well on visual perception tasks, they struggle significantly with symbolic derivation, highlighting a critical gap between visual comprehension and mathematical reasoning in engineering design.

### Strengths
- The dataset is highly diverse, encompassing various difficulty levels and a wide range of analog question types. This diversity could significantly benefit future research on large language models (LLMs) in analog reasoning.

- The experimental evaluation is thorough, involving multiple LLMs with different capability levels, providing a comprehensive comparison.

### Weaknesses
- Although the work is valuable, several existing benchmarks—such as AMSBench and MMCircuitEval—appear to share similar objectives and formulations. The differences between this work and those benchmarks are not clearly articulated.

- In the supplementary materials, there are multiple zero-byte files (e.g., in the Perception/func directory), which may indicate missing or corrupted data.

Typos:

In Table 4, “GPT-4O” should be corrected to “GPT-4o”.

### Questions
How is the correctness of the LLM’s answers evaluated? For example, in the Synthetic Example Q5, there seem to be multiple mathematically correct equations that can represent the same diagram. How is such ambiguity handled during evaluation?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces CircuitSense, a large-scale (8,006 problems across six hierarchical abstraction levels spread across Perception, Analysis, and Design problems) benchmark designed to evaluate visual mathematical reasoning in circuit analysis and design. This remains an under-explored domain for multimodal large language models (MLLMs). 

The authors construct a hierarchical synthetic data generation pipeline capable of automatically producing novel circuit schematics and block diagrams with guaranteed symbolic ground-truth equations. They achieved this via SPICE simulation and symbolic analysis via SymPy and Lcapy. This allows systematic testing of models’ ability to convert visual circuit representations into symbolic transfer functions and equations, which is a step beyond conventional recognition or numerical tasks that exist in the current VLM’s. 

Evals across six leading MLLMs such as GPT-4o, Gemini-2.5-Pro, Claude-Sonnet-4, InternVL3, etc. reveal a sharp contrast between perception and reasoning: while models achieve >85% accuracy on component recognition and topology understanding, symbolic derivation accuracy collapses below 19%, with catastrophic failure on novel synthetic circuits. This demonstrates the baseline reliability of the benchmark in evaluating SOTA models on these tasks.

### Strengths
- The paper tackles an unexplored domain: hierarchical visual symbolic reasoning in circuit systems, with high-quality richly annotated data. It seems to be the first benchmark to evaluate how well AI systems connect circuit visuals to mathematical models. Prior work such as ChipVQA only partially evaluates this capability. 
- Six abstraction levels mirror real analog design processes and provide a structured way to locate where model failures occur
- SDG pipeline and datasets are open-sourced and readily reproducible and could constitute a major contribution to the community in solving the data scarcity problem
- Failure analysis is fine-grained, highlighting, for example, precise points of reasoning breakdown (output impedance and input impedance)

### Weaknesses
- Sim2Real gap remains: while the synthetic generation pipeline is well engineered, the circuits are constrained to 12-15 components; data sourcing from internet data might result in contamination 
- The benchmark relies on Gemini as the judge, which may result in potential biases toward the Gemini family 
- The paper could benefit in an analysis on cross-domain transfer; for example, training on schematic-level circuits and testing on block diagrams 
- Lastly, the paper would benefit in adding human baseline performance

### Questions
- Beyond algebraic identity via SymPy, have you considered functionally equivalent but structurally different forms? This might influence the reported accuracy at <19%. 

- How closely do generated circuits reflect real textbook or industrial designs?

- How did you ensure fairness when benchmarking Gemini models using Gemini as a judge?

- Could CircuitSense outputs be integrated into simulation-based tools to test end-to-end utility for analog designers?

### Soundness
3

### Presentation
3

### Contribution
4
