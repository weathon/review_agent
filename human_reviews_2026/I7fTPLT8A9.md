# We-Math 2.0: A Versatile MathBook System for Incentivizing Visual Mathematical Reasoning

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 6

## Abstract
Multimodal large language models (MLLMs) have demonstrated impressive capabilities across various tasks but still struggle with complex mathematical reasoning. Prior work has mainly focused on dataset construction and method optimization, while often overlooking two critical aspects: comprehensive knowledge-driven design and model-centric data space modeling. We introduce WE-MATH 2.0, a unified system that integrates a structured mathematical knowledge hierarchy, model-centric data space modeling, and a reinforcement learning (RL)-based training paradigm to enhance the mathematical reasoning abilities of MLLMs. Our contributions are fourfold: (1) MathBook Knowledge System: a five-level hierarchy covering 491 knowledge points and 1,819 fundamental principles; (2) MathBook-Standard and MathBook-Pro: datasets that ensure broad conceptual coverage and robust training through dual expansion, a three-dimensional difficulty space, and seven progressive variants per problem; (3) MathBook-RL: a two-stage RL framework including Cold-Start Fine-Tuning to align models with knowledge-oriented chain-of-thought reasoning, and Progressive Alignment RL leveraging average-reward learning with dynamic data scheduling for progressive difficulty alignment; (4) MathBookEval: a benchmark covering all 491 knowledge points with diverse reasoning step distributions. Experimental results show that MathBook-RL achieves competitive performance on four widely used benchmarks and demonstrates strong results on MathBookEval, suggesting promising generalization in mathematical reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a unified system to enhance the mathematical reasoning abilities of MLLMs. WE-MATH 2.0 combines a structured math knowledge base, a model-driven data design framework, and an RL-based learning curriculum. Results on various benchmarks suggest the effectiveness of the proposed framework for MLLM reasoning.

### Strengths
- WE-MATH2.0 Integrates data generation, training, and evaluation within a closed-loop system, ensuring consistent metrics between learning objectives and assessment benchmarks.

- The paper provides a large-scale, well-balanced benchmark covering both knowledge breadth and reasoning depth, enabling systematic evaluation of multimodal reasoning models.

### Weaknesses
- The performance improvements of MathBook-7B are mainly observed on its self-constructed benchmarks like We-Math, while gains on external benchmarks such as MathVista and MathVerse are relatively minor.
- Although MathBookEval is designed to test reasoning depth and knowledge coverage independently from the training data, MathBook-7B does not show significant advantages on this benchmark. 
- The system focuses only on multimodal reasoning. It has not been evaluated on standard textual datasets such as GSM8K or MATH, leaving its language-only mathematical reasoning ability unverified. It would be great to also discuss the capability on language-only but still challenging math reasoning tasks.

### Questions
- How well does MathBook-7B generalize to language-only mathematical reasoning tasks, such as GSM8K or MATH, given that the system is trained and evaluated only on multimodal data?
- Have the authors conducted human validation or cross-checking to verify the correctness of automatically assigned knowledge labels and reasoning steps?
- Although MathBook-7B achieved strong results on We-Math and MathBookEval, how do the authors ensure that evaluation items were not seen or paraphrased during training, given the shared data construction pipeline?

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
3

### Summary
This paper introduces We-Math 2.0, a system involving a knowledge system for mathematical knowledge, a pair of dataset for training, an RL framework, and a comprehensive benchmark for multimodal mathematical reasoning.

### Strengths
1. This work represents substantial engineering effort in an important area and should be applauded for this.
2. MathBook Knowledge System (MKS) is comprehensive with 491 knowledge points + 1819 fundamental principles
3. MathBook-Standard/Pro is Built on MKS with annotated problems which are shown later in experiments to be strong together with the proposed MathBook-RL.
4. MathBook-RL is presented well with ablation studies.
5. This paper offers a few insights including the observation that MLLMs performance in geometry is subpar compared to that in algebra, and that performance in general correlates negatively with knowledge points.

### Weaknesses
Maybe some more experiments on a few more model scale are warranted, but considering the amount of work put into the entire system, I do not see this as much of defect.

### Questions
Do we have a mechanism to use the knowledge system at inference?

### Soundness
3

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
4

### Summary
The paper proposes We-Math 2.0, a unified framework centered around a structured MathBook knowledge system with 491 knowledge points. It introduces associated datasets and the MathBookEval benchmark, and designs a reinforcement learning training strategy based on curriculum learning. The authors claim their trained model achieves a marginal performance advantage on several existing mathematical multimodal benchmarks

### Strengths
- The paper presents a unified framework (We-Math 2.0) that integrates a structured mathematical knowledge system, a model-centric data space, and an RL-based training paradigm.

- The trained model reportedly demonstrates a marginal advantage on some established mathematical multimodal benchmarks.

### Weaknesses
- The categorization and comparison in Table 1 appear inconsistent. For instance, comparing the granularity of the proposed 491-point system with datasets like MathV360k (which contains diverse content like charts and general QA) is not an apples-to-apples comparison and may be misleading.

### Questions
- How can the evaluation in Table 1 be made fair and consistent, especially regarding the granularity of category definitions?
- Could the evaluation on MathBookEval include more open-source reasoning models of comparable scale? This would help disentangle whether the observed performance gains primarily stem from the richness and structure of the training data or from the effectiveness of the MathBook-RL training pipeline itself.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors proposed We-Math 2.0, a comprehensive framework for visual mathematical reasoning benchmarks. The major contribution of this paper is providing a detailed way to curate the dataset alongside with itself, paving the path for future progress. The authors also include a two-stage RL framework for MLLMs, and a extensive evaluation.

### Strengths
1. The paper is well structured and easy to follow. The teaser figure is particularly well structured.
2. The intuition of the dataset is detailed and inspiring, which could be even more helpful than the dataset iteself.
3. The evaluation results show the improvement brought by We-Math 2.0.

### Weaknesses
No significant weakness within the dataset scope.

### Questions
1. I see in the teaser figure the definition has been emphasized a lot. I wonder if the authors want to state that the solution of the question is related to the definition, or the question is about the definition itself?

### Soundness
3

### Presentation
3

### Contribution
3
