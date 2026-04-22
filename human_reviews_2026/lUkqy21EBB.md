# Pushing LLMs to Their Logical Reasoning Bound: The Role of Data Reasoning Intensity

- Avg Score: 3.33
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4

## Abstract
Recent advances in large language models (LLMs) highlight the importance of training data structure and quality in shaping reasoning behavior. However, most existing approaches focus on transforming data formats while neglecting the internal reasoning complexity of training samples, leaving the reasoning potential of data underexplored and underutilized. In this work, we posit that LLM logical reasoning performance is jointly constrained by the potential of the training data and the cognitive capacity of the model. To make this relationship measurable, we introduce Data Reasoning Intensity (DRI), a novel metric that quantifies the latent logical reasoning complexity of samples by decomposing and aggregating their logical structures. This allows us to analyze how well current LLMs utilize logical reasoning signals and identify performance gaps relative to data potential. Based on this insight, we introduce a re-cognizing optimization strategy that systematically enhances the logical reasoning intensity of training data. Rather than increasing data volume, our method re-optimizes existing samples to better align with the LLM’s logical reasoning bounder. Extensive experiments show that our approach significantly improves performance and generalization over data-centric strategies. We further validate our method under a reinforcement learning framework. Our results indicate that prioritizing reasoning complexity in data rather than sheer scale or superficial form is essential to realizing LLMs' full cognitive potential. Our code is available in the supplementary file.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduce Data Reasoning Intensity (DRI), a novel metric that quantifies the latent logical reasoning complexity of samples by decomposing and aggregating their logical structures.
Based on this, the authors introduce a re-cognizing optimization strategy that systematically enhances the logical reasoning intensity of training data. 
Experiments show that the proposed approaches improves performance and generalization over data-centric strategies.

### Strengths
1. The paper develops a quantitative metric for the reasoning complexity of training data.
2. The authors draw on established theories like  Sweller’s cognitive load optimization principle to propose the re-cognizing optimization strategy.

### Weaknesses
1. Although the paper introduces a quantitative metric for data reasoning potential, the rationale behind the specific formulations (e.g., Equations 1, 5, 6, and 7) is unclear: why are they defined in their current forms rather than other possible ones (linear, polynomial, exponential, logarithmic, etc.)? The paper lacks sufficient theoretical justification, intuitive explanation, or empirical validation for these choices.
2. The definition in Equation 1 is somewhat confusing. It is unclear why it is termed effective reasoning capability: it seems the authors may intend to express the model’s efficiency or learning ability in utilizing data per parameter unit. Moreover, since model architectures may naturally exhibit different sensitivities to various data types, it is not clear how the coupling between C(M) and E(D) is considered.
3. The notations used in Sections 3.1 and 3.2 are also difficult to follow. It would be helpful if the authors could include concrete examples to illustrate the meaning or numerical form of these symbols.
4. Figure 1 lacks sufficient explanation. The meaning of the horizontal gray dashed line and the definition of the ideal state are unclear. According to the intended meaning of Equation 1, shouldn’t the line be linear? (Though I may have misunderstood the authors’ intention.)
5. Beyond natural-language-based logical reasoning, mathematics and code are important components of reasoning tasks today. The generalization and applicability of the proposed metric to these domains require further validation.

### Questions
see Weakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors suppose that the LLM's reasoning ability is constrained by the potential of the training data. Therefore, the authors design a metric, named data reasoning intensity, to re-order the training datasets. In this way, the LLMs are trained on re-ordered datasets from low DRI samples to high DRI samples. Experiment results prove the effectiveness of the proposed method. The DRI depends on the Logical Intensity Quantification and Reasoning Intensity Quantification.

### Strengths
1. The paper proposes a new metric to re-range the training samples of reasoning tasks.
2. The re-range dataset is balanced. According the Figure 3 (c), a high DRI score also means a higher error rate, which is consistent.
3. The fine-tuned models show better performance on reasoning tasks.

### Weaknesses
1. The improvements are not significant compared to baselines.
2. In Figure 1, the 8b-our method is closed to the 30b-base. However, there are no related experiments and results in Section 4.

### Questions
Does this method improve an 8B model to achieve 32B-level performance?

### Soundness
2

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
This paper introduces Data Reasoning Intensity (DRI), a novel metric for quantifying the latent logical reasoning complexity of training samples for large language models. The authors posit that LLM reasoning performance is jointly constrained by training data potential and model cognitive capacity. DRI decomposes samples into logical elements (predicates, constants, expressions) and aggregates them via a scoring function that accounts for structural complexity, nesting depth, and reasoning steps. Based on empirical analysis showing current LLMs underutilize available reasoning data, the authors propose a re-cognizing optimization strategy: Phase I reshapes model cognition through uniform data exposure, while Phase II emphasizes high-DRI samples via probability-weighted sampling. Experiments on four logical reasoning benchmarks (Reclor, LogicBench, LogiQA, LogiQA2.0) using LLaMA3.1-8B and Qwen2.5-7B demonstrate consistent improvements over curriculum learning and bin-based progressive learning baselines in both supervised fine-tuning and reinforcement learning settings.

### Strengths
- The paper includes extensive experiments across multiple datasets, model architectures, and training paradigms (SFT and RL), with careful construction of balanced test sets to avoid evaluation bias.
- The analysis reveals three important and clear observations: (1) low-DRI data can be pruned, (2) high-DRI data catalyzes improvements, and (3) diversity across DRI spectrum is necessary—providing actionable guidance for data curation.
- The connection to the Roofline Model and cognitive load theory provides intuitive motivation, even if the formalization could be strengthened.

### Weaknesses
- The quantity DRI well supports the data selection given the model complexity. However, this is how E(D) part works. If the author wants to advocate that DRI works as the ratio of E(D) / C(M), as the key contribution, the C(M) part should also be demonstrated.
- The E(D) quantity is defined on logical problems, specifically, a well defined first order language. This aligns well with the scope of this paper. However, one potential limitation is that if we work on a first order language where $|C_i|$ is particularly large, say a database on all athletics in Olympic games from 1900. where the |C_i| term dominate the calculation so that the DRI score can be biased. This can happen if LLM is given a question about athletics and a full list of their names are input in the context. How will you define the E(D) score for the sample?

### Questions
- Can we say that, the DRI ratio E(D) / C(M) means that more parameters of LLM leads to lower DRI score given the same dataset. So can we say, for a 30B model and a 8B model, if we train it on the same dataset D, the 30B model should perform worse than the 8B one because its DRI is low?

### Soundness
2

### Presentation
3

### Contribution
3
