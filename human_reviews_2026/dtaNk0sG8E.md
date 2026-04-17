# What MLLMs Learn about When they Learn about Multimodal Reasoning: Perception, Reasoning, or their Integration?

- Decision: Reject
- Scores: 6, 4, 6, 2

## Abstract
Multimodal reasoning models have recently shown promise on challenging domains such as olympiad-level geometry, yet their evaluation remains dominated by aggregate accuracy, a single score that obscures where and how models are improving. We introduce MathLens, a benchmark designed to disentangle the subskills of multimodal reasoning while preserving the complexity of textbook-style geometry problems. The benchmark separates performance into three components: Perception: extracting information from raw inputs, Reasoning: operating on available information, and Integration: selecting relevant perceptual evidence and applying it within reasoning.
To support each test, we provide annotations: visual diagrams, textual descriptions to evaluate reasoning in isolation, controlled questions that require both modalities, and probes for fine-grained perceptual skills, all derived from symbolic specifications of the problems to ensure consistency and robustness. Our analysis reveals that different training approaches have uneven effects: First, reinforcement learning chiefly strengthens perception, especially when supported by textual supervision, while textual SFT indirectly improves perception through reflective reasoning. Second, reasoning improves only in tandem with perception. Third, integration remains the weakest capacity, with residual errors concentrated there once other skills advance. Finally, robustness diverges: RL improves consistency under diagram variation, whereas multimodal SFT reduces it through overfitting. We will release all data and experimental logs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
They introduce MATHLENS, a benchmark designed to disentangle the subskills of multimodal reasoning while preserving the complexity of textbookstyle geometry problems. The benchmark separates performance into three components: Perception: extracting information from raw inputs, Reasoning: operating on available information, and Integration: selecting relevant perceptual evidence and applying it within reasoning.

### Strengths
1. The paper is clearly written with demonstrative figures.
2. The idea of separating the reasoning process of MLLMs is interesting.
3. The analysis of the result of different methods is detailed and shines light on the inner workings of MLLM reasoning.

### Weaknesses
1. The perception probes test for a finite set of atomic facts. A model might correctly identify all the probed facts but fail to perceive another crucial, un-probed visual detail. This perceptual failure would be misclassified as an integration failure.
2. The primary benchmark, MATHLENS, is composed entirely of geometry problems. It cannot catch the full scope of MLLM reasoning. The add-on MATHLENS-GENERAL cannot maintain the same rigor.

### Questions
See weaknesses.

### Soundness
2

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
This paper addresses the problem that aggregate accuracy metrics are insufficient for evaluating and understanding the progress of Multimodal Large Language Models (MLLMs) on complex reasoning tasks. By comparing a model's performance across these tests, the authors can decompose errors into failures of perception, reasoning, or Integration (defined as failing the joint task despite succeeding on perception and reasoning individually). The methodology also involves generating semantic variations of diagrams to test robustness.

### Strengths
- Error analysis reveals RL shifts errors to integration, providing evidence-based guidance for future training, unlike aggregate benchmarks.
- Use of symbolic states ensures equivalence across modalities, supporting valid isolation of subskills with high diagnostic value.
- The evaluation of robustness using semantic-level diagram modifications, rather than just pixel-level noise.

### Weaknesses
- The primary analysis and all major findings are derived exclusively from the geometry domain. While this allows for tight control, it leaves the generalizability of the conclusions.
- The accuracy of the error decomposition hinges on the assumption that the perception probes are exhaustive.
- The paper presents many empirical comparisons but does not report confidence intervals or use statistical tests to confirm the significance of the observed differences.

### Questions
- Regarding the error decomposition, how did you ensure the set of perception probes for each problem was comprehensive? Is it possible that some errors classified as "integration" failures are in fact subtle perceptual failures not captured by the probes?
- What compute was used for evaluations, and how might nondeterminism affect API models?

### Soundness
2

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
2

### Summary
This paper introduces MATHLENS, a benchmark designed to disentangle the subskills of multimodal reasoning—specifically perception, reasoning, and their integration—in the context of geometry problems. 
The authors argue that existing benchmarks often rely on aggregate accuracy, which obscures the specific capacities where models fail or improve. 
MATHLENS is built from symbolic specifications to generate aligned annotations, including visual diagrams, textual descriptions, perception probes, and multimodal questions, ensuring consistency and robustness. 
Through controlled experiments on open multimodal models, the paper reveals that different training approaches
have uneven effects.

### Strengths
1. MATHLENS provides a rigorous framework for decomposing multimodal reasoning into perception, reasoning, and integration, addressing a gap in existing evaluations. The use of symbolic semantics ensures controlled and reproducible annotations.
2. The paper thoroughly evaluates multiple model families across diverse settings, including robustness tests with semantic diagram modifications. This allows for nuanced insights into how training strategies affect specific capacities.
3. MATHLENS demonstrates strong alignment with datasets like MathVista and MathVerse, enhancing its credibility and utility for the community. The release of data and tools promotes reproducibility and further research.

### Weaknesses
See Questions.

### Questions
1. Have you explored preliminary strategies to improve integration? If so, what were the results? If not, what directions do you prioritize for future work?
2. MATHLENS relies on geometry problems, which have well-defined symbolic representations. For tasks with ambiguous symbolic mappings, how would you adapt MATHLENS’s decomposition framework to ensure consistent evaluation of perception, reasoning, and integration?
3. The diagram modifications test structural changes, but how would MATHLENS perform under real-world perturbations like occlusions or lighting variations?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper proposes a math benchmark to evaluate MLMs through different lenses rather than a single one. Specifically, they present and made publicly available a dataset called MathLens for geometry to perform evaluation based on three aspects: perception, reasoning and integration, enabling evaluation beyond a single score accuracy. Authors then did several fine-tuning analysis and made these main observations: (1) RL boosts perception, (2) textual SFT indirectly strengthens perception through reflective reasoning, (3) integration is the weakest skill among all, and finally (4) robustness varies (e.g., RL vs. SFT).

### Strengths
1. Paper is well-written and easy to follow. The core idea is interesting and novel.

2. A public release of the dataset for the community is nice and will help future research.

3. Extensive experiments and analyses were provided. For instance robustness analysis investigates changes to the diagram (e.g., rotation, etc.).

### Weaknesses
1. Single data source is just very limited -- only FormalGeo-7K was used as the basis of the data. This makes it hard to trust the findings IMO (e.g., relation between RL and SFT).

2. The core analysis were performed using open-weight models and it's unclear how the fine-tuning of closed-source models such as Gemini would make any difference.

3. Integration measurement is done indirectly -- this could be conflated with other latent failures, like context-length limitations.

### Questions
1. As mentioned in the paper, integration is the main bottleneck in structured geometry. Is this true for real-world less-structured data as well? 

2. Since integration is the weakest skill, what training or architectural changes can be made to improve it?

### Soundness
3

### Presentation
3

### Contribution
3
