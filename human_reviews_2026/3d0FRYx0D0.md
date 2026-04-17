# CMPhysBench: A Benchmark for Evaluating Large Language Models in Condensed Matter Physics

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
We introduce CMPhysBench, designed to assess the proficiency of Large Language Models (LLMs) in Condensed Matter Physics, as a novel Benchmark. CMPhysBench is composed of more than 520 graduate-level meticulously curated questions covering both representative subfields and foundational theoretical frameworks of condensed matter physics, such as magnetism, superconductivity, strongly correlated systems, etc. To ensure a deep understanding of the problem-solving process,we focus exclusively on calculation problems, requiring LLMs to independently generate comprehensive solutions. Meanwhile, leveraging tree-based representations of expressions, we introduce the Scalable Expression Edit Distance (SEED) score, which provides fine-grained (non-binary) partial credit and yields a more accurate assessment of similarity between prediction and ground-truth. Our results show that even the best models, Grok-4, reach only 36 average SEED score and 29% accuracy on CMPhysBench, underscoring a significant capability gap, especially for this practical and frontier domain relative to traditional physics.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces CMPhysBench, a benchmark designed to evaluate LLMs for Condensed Matter Physics (CMP) problems. The benchmark consists of 520 graduate-level calculation problems, spanning subfields such as magnetism, superconductivity, semiconductors, and strongly correlated systems. Compared to prior benchmarks, CMPhysBench emphasizes symbolic reasoning. A key contribution is the SEED metric, which extends existing expression-based similarity measures to support multiple answer types through syntax tree representations. The authors evaluate 18 proprietary and open-source LLMs, finding that even top models like Grok-4 achieve only 28% accuracy, suggesting a significant gap in domain-specific reasoning for CMP.

### Strengths
- The dataset is a good contribution, as it is manually curated by domain experts and spans diverse CMP subfields.

- SEED provides a novel measure of partial correctness, addressing a gap in symbolic assessment.

- The evaluation across 18 models is comprehensive, offering generalizability and comparative insight.

- Evaluation with human alignment validation and the error analysis with detailed categorization of reasoning failures helps to verify the findings.

### Weaknesses
- Although the error analysis is informative, the reasoning behind LLM failures remains speculative rather than causal.

- Maybe it's because CMP is a very specific area beyond my research scope. I find the dot examples in the figures very confusing. If a bit more background can be provided to explain the meaning of those symbols, it will help non-experts to better understand the paper.

### Questions
- Could the SEED metric be adapted to multimodal problems involving diagrams or figures in physics textbooks?

- Would you expect domain-specific fine-tuned LLMs (e.g., physics-trained models) to exhibit qualitatively different failure modes than general LLMs?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Graduate level condensed matter physics benchmark manually curated in high detail. The authors main contributions are (1) mix of LLM and expert question generation (2) presentation of a new metric SEED which helps to reward both partial correctness as well as (intended) to help with symbolic equivalence matching of answers, correlating well with human preference, (3) evaluation of several SoTA models on CMPhysBench showing that models currently struggle with condensed matter physics.

### Strengths
1, Scope and subject: Graduate level benchmark based on standard graduate textbooks, requiring complex step-by-step solutions across diverse answer types. Specifically, CMYPhysBench is a non MC/QA benchmark, so much more difficult.

2. Diversity and coverage: Clear balance between categories, and clear explanation and validation of the source material from which benchmark is derived from. The authors also perform strong analysis on failure modes, which are possibly actionable and of interest to members of the AI for science community trying to understand LM capabilities on scientific domains (and when to (or not) use models for specific scientific tasks).

3. SOTA LLMs moderately struggle on this benchmark making it of key interest to the community and interesting for marking LLM progress over time.

### Weaknesses
1. Relevance of SEED as an benchmark evaluation metric versus actual accuracy.

The goal seems to reward partial correctness (which is understandable from an RL or intermediate reward feedback perspective), however in practice: does SEED actually properly weight when LMs make minor incorrect reasoning steps (or does it only purely give partial credit when LMs fail to decode a final correct answer)? Some more discussion on this would be helpful.

Related, from model thinking trajectories, how well does SEED correlate with minor incorrect steps in reasoning: I’m worried that SEED partial correctness of answer may not correlate with minor reasoning incorrectness, which may defeat some of the benefit of this partial correctness.

Related, is it true that partial misses in edit distance should be only partially penalized just at an AST level?

2. Human preference seems to be binary (0 or 1): are there studies on how accurate human labelling is here? Were questions + ground truth answers verified by multiple raters? As such would the human grading here be close to an “accuracy” or a “preference” type of statistic?

### Questions
Figure 6 very difficult to read even while zoomed in (Model text and also the colors)

Figure 4: I’m unclear on how SEED is actually computed here, where does the 60 in Model Response 1 come from?

### Soundness
3

### Presentation
2

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
This paper presents CMPhysBench, a new benchmark to test LLMs on graduate-level Condensed Matter Physics. It's made of 520+ hard calculation problems from textbooks. They also created a new scoring metric called SEED that gives partial credit for complex math answers (like equations or tuples) using tree-based analysis. Their tests on 18 LLMs show that even the best models, like Grok 4, perform poorly (36 SEED score, 28.9% accuracy), showing a big gap in this specific domain.

### Strengths
The paper's main strength is tackling a new, hard domain: graduate-level condensed matter physics. Most benchmarks are easier, so this is a needed step up. The SEED metric is also a big plus; it's a smart way to give partial credit on complex math answers instead of just right/wrong. This metric seems useful for other science benchmarks too. The testing of 18 models is thorough, and the error analysis in Figure 6 gives a good breakdown of why models fail, with "Concept and Model Misuse" being the biggest problem.

### Weaknesses
The main weakness I see is in the error analysis. The authors used GPT-4o to categorize all the model mistakes. While this is fast, it's not clear how accurate GPT-4o is at this task. It would be better if they had human experts check a sample of these to confirm the error breakdown. Also, the SEED score focuses on the final boxed answer. The prompt asks for step-by-step solutions, but it's not clear if the steps themselves are evaluated. A model could get the right answer with the wrong steps.

### Questions
For the error analysis, how do you know GPT-4o's categorizations are correct? Did you have any humans double-check its work? Does your evaluation look at the reasoning steps, or just the final answer in the box? It seems possible for a model to get the right answer by luck or by making mistakes that cancel out.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This manuscript introduces CMPhysBench, a novel benchmark that successfully establishes a rigorous standard for evaluating LLMs on advanced scientific reasoning tasks in physics. The adoption of a fine-grained evaluation protocol is a major contribution.

### Strengths
1. High Problem Difficulty

Focuses exclusively on graduate-level material, comprising more than 520 meticulously curated questions that require LLMs to generate complete, step-by-step solutions for complex calculation problems. This moves beyond the limitations of high school or undergraduate benchmarks, demanding advanced mathematical rigor and conceptual understanding.

2. Expert-Aligned Metric

The proposed Scalable Expression Edit Distance (SEED) score provides highly accurate, fine-grained, non-binary partial credit for mathematical responses. SEED exhibits the highest correlation (ρ = 0.90) with human expert ratings, demonstrating superior alignment in evaluating complex symbolic reasoning compared to prior metrics.

3. Extensive Model Analysis

The paper conducts a comprehensive empirical study evaluating 18 proprietary and open-source LLMs. This extensive analysis identifies a significant capability gap, with the best models achieving only a 36 average SEED score and 28% accuracy, providing quantitative illumination of specific failure modes across the LLM ecosystem.

### Weaknesses
1. It would be better if the authors could discuss how the issues of LLM in this domain identified in the analysis could be mitigated in the future research. Currently, the analyses only show LLM can make multiple types of error and it is still unclear how to improve LLM to avoid such errors. Proposing potential solutions for the identified errors could further improve the contribution of the paper.

### Questions
1. please add additional analyses or discussions towards future research directions or the contribution of this paper towards the future improvements of LLM in this domain.

### Soundness
3

### Presentation
3

### Contribution
2
