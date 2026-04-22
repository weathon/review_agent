# Lean4Physics: Comprehensive Reasoning Framework for College-level Physics in Lean4

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 6, 8, 2

## Abstract
We present **Lean4PHYS**, a comprehensive reasoning framework for college-level physics problems in Lean4. To establish a solid foundation for formal reasoning in physics, **Lean4PHYS** launches *PhysLib*, a repository containing fundamental unit systems and essential theorems to formulate physics proofs in Lean4. It will be community-driven and long-term maintained. Lean4PHYS also includes *LeanPhysBench*, a college-level benchmark for evaluating LLMs' Lean4 formal physics reasoning capability. It contains 200 hand-crafted and peer-reviewed Lean4 theorem statements formalized from university textbooks and physics competition problems. Based on the *PhysLib* and *LeanPhysBench* we composed in **Lean4PHYS**, we perform exhaustive experiments of baseline results using major expert Math provers and state-of-the-art closed-source models, and provide an analysis of their performance. In the experiment, we identify that most expert provers do not outperform general models as they did in the math domain. This suggests potential overfitting to the math domain rather than learning formal reasoning for formal provers. We also conduct a comprehensive experiment showing that, with *PhysLib* in the context, LLMs' performance on *LeanPhysBench* increases by **11.90%** on average, proving the effectiveness of our repository in assisting LLMs in solving the Lean4 physics problem. To the best of our knowledge, we are the first study to provide a physics benchmark in Lean4.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents Lean4PHYS, a Lean4-based framework for formalizing college-level physics problems.   It includes PhysLib, a modular library with a systematic unit system and reusable theorems, and LeanPhysBench, a benchmark of 200 formalized physics problems.   The authors propose a pipeline to convert natural language physics questions into Lean4 proofs.   Experiments compare Lean-oriented provers with general-purpose LLMs on LeanPhysBench.   Results show LLMs outperform provers, with improved performance around 40.5% accuracy when using PhysLib context, highlighting the need for domain-specific knowledge in formal reasoning.

### Strengths
- The paper pioneers formal physics reasoning in Lean and introduces the first large-scale Lean4 physics benchmark covering topics from mechanics to modern physics. 

- PhysLib provides a modular, SI-based unit system and topic-structured theorems, ensuring dimensional consistency and extendability for accurate physical reasoning.

- Comprehensive experiments show that PhysLib context consistently boosts model performance.  LLMs outperform specialized Lean provers, revealing that current Lean provers, trained for math, struggle with physics tasks.

### Weaknesses
- The paper lacks an explanation of the advantages of PhysLib's modular structure over organizing theorems by specific physics domains.  It also does not clarify how this hierarchical organization aids in retrieval and reasoning.
- What defines the boundary between mathematical and physical problems, and why do Lean provers, which perform well in mathematics, fail to transfer their capabilities to the physics domain?
- Would including non-competition problems, such as physics questions from middle school or high school exams, in the experiments provide a more comprehensive comparison?

### Questions
see Weaknesses

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
This paper introduces a comprehensive framework named Lean4PHYS, designed for formal reasoning on university-level physics problems using the Lean4 proof assistant. The framework consists of two core components: PhysLib, a community-driven library that provides a foundational unit system and commonly used theorems for formal physics reasoning; and LeanPhysBench, a benchmark dataset of 200 problems manually constructed and formalized from university textbooks and physics competitions. Based on this framework, the authors evaluate the performance of several mainstream large language models (including both general-purpose models and those specialized in Lean mathematical proofs). The experimental results show that general-purpose LLMs generally outperform math-specialized models on physics reasoning tasks, revealing a potential overfitting issue of the latter to the mathematics domain. Furthermore, the study demonstrates that using PhysLib as contextual information significantly improves the performance of all tested models.

### Strengths
*   **Significant Contribution to the Research Community:** This paper contributes two extremely valuable resources: **PhysLib**, a modular and extensible foundational library for physics, and **LeanPhysBench**, the first benchmark dedicated to evaluating formal reasoning capabilities in physics. These two achievements provide a solid infrastructure and a fair evaluation standard for subsequent researchers to enter and work in this field, which will undoubtedly promote the development of the entire community.
*   **Exhaustive Experiments and Deep Insights:** The paper's experimental design is very comprehensive, not only testing multiple top-tier general-purpose LLMs but also comparing them with several Lean-specialized models that excel in mathematics. The results reveal an important finding that "specialized models have limited cross-domain (from math to physics) generalization ability," prompting deep reflection on model generalization and domain overfitting. At the same time, the experiments clearly quantify the effectiveness of the PhysLib library in assisting models with physics reasoning, proving its design value.

### Weaknesses
*   **Insufficient Discussion of Related Work:** The "Related Work" section mentions Lean's application in physics and other non-mathematical fields but fails to deeply discuss the differences and connections with some directly related works. For example, the paper mentions the `PhysLean` [Tooby-Smith & contributors (2024)] project but only briefly describes it as "theorem-specific, small-scale, and non-modular." Considering that `PhysLean` also aims to formalize physics in Lean4, the authors should have elaborated more on the fundamental differences and specific advantages of Lean4PHYS in terms of design philosophy, implementation methods (such as the construction of the unit system), coverage, and modular design compared to `PhysLean`. Adding such in-depth comparative analysis would better highlight the uniqueness and irreplaceable contribution of this work.

### Questions
1.  The experimental results indicate that providing PhysLib as context to the models significantly improves their performance. Given that PhysLib, as a foundational library, could be quite large, the paper seems to lack a specific description of how it was effectively integrated into the model's prompt context window. Did the authors provide the entire library's content, or was some form of retrieval mechanism used to select relevant theorems and definitions? Clarifying this implementation detail is crucial for the reproducibility and understanding of the results.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces Lean4PHYS, which is a Lean4-based framework for formal physics. Lean4PHYS includes PhysLib (a repository of a physics unit system and commonly used theorems) and LeanPhysBench (a benchmark of 200 hand-crafted theorems from high school competitions to elementary college level).

### Strengths
The first framework for physics problems in Lean4 for LLM. This may be of significant interest to the community that studies LLM applications to physics. The PhysLib library and LeanPhysBench test dataset can be a useful artefact for future studies.

### Weaknesses
- Copyright infringement.
    - **Due to this issue, I decide to assign a very low score to the paper albeit the important contribution. However, I am very open to changing my score once this issue is clarified/addressed.**
    - The authors mentioned that “rather than copying the questions verbatim, we reformulated and rephrased them based on the underlying physics ideas", however, I am not certain that this is sufficient. The key issue hinges on "substantial similarity" and whether the original work's creative expression has been copied, even in a modified form. Given that the underlying physics idea of the questions are copied (perhaps to the point that there exists a one-to-one mapping between the textbooks and the dataset questions), this seems to constitute substantial similarity. The authors may need to ask for **explicit permission** from the publishers.
- Statistical robustness
    - Given that the evaluations were done with non-zero temperature, the authors should report the stochasticity of the results (e.g., standard error).
- Lack of implementation elaboration
    - How do the authors present PhysLib to the model? Would it fit into the context window?
    - For non-experts, it is challenging to understand what the task looks like, particularly because the prompt asks the model to “complete the following Lean4 code” instead of the commonly known question-answering setup.

### Questions
- Did the authors attempt to compare the accuracy of the models in this Lean 4-based setup vs natural setup (i.e., asking the model the question in natural language)?
    - This would be important to check if the bottleneck is in the physics understanding or in the Lean 4 code understanding
- I would expect PhysLib to be used by the model in a tool-calling fashion such that we do not need to present all the available concepts to the model in the context window. Is that the case?
- L377-379: “models with weaker in-context learning perform reatively badly on this level of problems. It is because they cannot infer the new out-of-distribution syntax or unit-handling rules from context.” → This seems to be an overclaiming since none of the experiments are checking the in-context capabilities. Not to mention, we cannot confidently say that this data is OOD because we do not have access to the pretraining data of the models. Am I understanding the sentence properly?

### Soundness
2

### Presentation
2

### Contribution
3
