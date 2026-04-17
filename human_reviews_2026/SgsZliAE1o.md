# Prompting to Prompt: Meta-Template Learning for Transferable Prompt Optimization

- Decision: Reject
- Scores: 4, 4, 2, 6

## Abstract
Prompt optimization plays a key role in fully leveraging the capabilities of large language models (LLMs). Despite their respective advantages, offline, online, and hybrid prompt optimization methods all suffer from limited transferability and strong reliance on task-specific data. To systematically resolve these limitations, we propose Prompting to Prompt (PTP), a novel framework for optimizing meta-templates, inspired by the idea of learning to learn. PTP introduces meta-templates as structured intermediate representations that decompose prompts into transferable elements, enabling generalization across diverse task. PTP employs a bi-level optimization process: the inner loop that refines prompt elements for individual samples using gradient feedback and element list guidance, and the outer loop that captures transferable structural patterns by comparing element-level changes before and after inner-loop updates. Instead of learning task-specific features, the outer loop generalizes structural knowledge across tasks, continuously updating meta-template structures and selection strategies. This enables PTP to unify offline and online prompt optimization, supporting task-level and query-level prompt generation without retraining. Extensive experiments on six benchmark datasets show that PTP consistently outperforms state-of-the-art baselines, achieving up to 10.52-point gains on challenging tasks like Arena-hard. These results demonstrate that PTP offers a promising solution for more transferable and efficient prompt optimization.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Prompting to Prompt (PTP), a novel meta-learning–inspired framework for prompt optimization that introduces a structured meta-template as an intermediate representation. PTP decomposes prompts into reusable, transferable elements and employs a bi-level optimization process: an inner loop refines task-specific prompts using textual gradients, while an outer loop abstracts cross-task structural patterns to update the meta-template. The method is evaluated across six benchmark datasets under both offline and online settings, demonstrating consistent improvements over baselines.

### Strengths
1. PTP represents a meaningful conceptual advance by framing prompt optimization as a meta-learning problem. The introduction of a learnable meta-template, enables systematic decomposition and recombination of prompt components, moving beyond string-level or black-box prompt tuning toward structured, interpretable prompt engineering.
2. The dual-loop mechanism is well-motivated and aligns with established meta-learning principles. The inner loop performs fine-grained, error-driven prompt refinement, while the outer loop captures generalizable structural priors. This separation of task-specific adaptation and cross-task generalization is a key strength.
3. PTP enables plug-and-play prompt generation without retraining or auxiliary models, offering a unified solution for both offline and online settings. The ablation studies on loop iterations, instantiation models, and multi-task training further support its robustness and scalability.

### Weaknesses
1. While the empirical results are compelling, the paper offers no theoretical analysis of why the meta-template structure facilitates cross-task transfer. There is no discussion of convergence properties, generalization bounds, or conditions under which the bi-level optimization is guaranteed to improve transferability. Given the reliance on LLM-as-optimizer (a non-convex, stochastic process), even heuristic convergence analysis would strengthen the methodological foundation.
2. The 12-element list (Appendix E.1) appears to be manually curated based on prompt engineering heuristics rather than learned from data. The paper does not justify why these specific components (e.g., “Personality & Style,” “Insight & Depth”) are necessary or sufficient. It remains unclear whether $\theta$ is task-agnostic or whether its completeness affects performance—especially for tasks that do not naturally align with such a structured format (e.g., open-ended creative writing).
3. Although PTP is described as “low-cost,” the paper lacks a detailed comparison of computational or financial overhead. All methods are run for 16 total iterations, but PTP uses GPT-4o as M_opt in both loops, while baselines may use cheaper models. A breakdown of API calls, token usage, or wall-clock time would clarify whether the performance gains come at a prohibitive cost—particularly relevant for real-world deployment.
4. The paper demonstrates strong cross-task transfer but does not explore failure modes. For instance:
    - How does PTP perform when source and target tasks are highly dissimilar (e.g., training on arithmetic, testing on moral reasoning)?
    - Can the meta-template resolve conflicting requirements (e.g., “be concise” vs. “provide detailed steps”)?
    - Is PTP effective for tasks that do not benefit from structured prompting (e.g., summarization, translation)?

    Such analyses would better delineate the method’s applicability scope.

5. PTP relies on GPT-4o as M_opt to generate textual gradients. The paper does not evaluate whether weaker models (e.g., GPT-3.5, small open-source LLMs) can serve as effective optimizers. If performance degrades significantly with smaller models, the framework’s practical utility in resource-constrained settings is limited.

### Questions
Please refer to Weaknesses for details.

### Soundness
3

### Presentation
2

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
This paper addresses the limited transferability and poor reusability of existing prompt optimization methods for LLMs. The authors propose Prompting to Prompt (PTP), a meta-template learning framework that decomposes prompts into transferable structural elements (meta-templates) and adopts a bi-level optimization process (inner loop for task-specific prompt refinement, outer loop for cross-task meta-template updating). PTP unifies offline and online prompt optimization scenarios, eliminating the need for retraining when adapting to new tasks or queries. Extensive experiments on six benchmark datasets and six LLMs illustrate the state-of-the-art performance with up to 10.52-point gains on challenging tasks like Arena-hard.

### Strengths
**Originality**
This work demonstrates originality in two folds: (1) the proposed meta template learning unifies online and offline prompt optimization in one framework. (2) SOTA performances are achieved in extensive experiments. 

**Technical details**
1. The reusable meta-templates capture cross-task structural patterns play an important role in improving the transferability. 
2. The mathematical equations clearly formalize the prompt optimization objectives and concrete steps in both inner loop and outer loop optimization. 

**Quality & Clarity** 
1. The proposed method is technically sound and the unified meta learning framework is elegant. The overall structure of the manuscript is organized with a clear logic. Details on the meta learning and bi-level optimization is rigorously addressed in method section with comprehensive supplementary details in Appendix. 
2. Extensive experiments with 6 LLMs on 3 open-ended benchmarks, including both offline and online settings, validate PTP's effectiveness. Furthermore, comprehensive datasets and experiment details in appendices provide strong support for this work. 

**Significance**
1. This work is of highly practical value to a wide-range of scenarios because of three aspects: (1) the meta template learning is training data efficient (200 instances from GSM-8k or 74 instances from BBH-CJ) , (2) it costs much less GPU resources than previous soft prompt optimization methods, (3) it has the potential to be compatible with nearly all kind of LLMs including both proprietary LLMs with API access and open-sourced LLMs.

### Weaknesses
1. Typo in Eq. (4) Select -> Selcet?

2. Previous work PROmpting (OPRO) also introduced concept and method on "meta prompts" to improve the cross-task generalization, as mentioned in Line 160, and the authors claim the superiority of this work with only a few words in between Line 186-187. Since OPRO is the most related work, quantitative and qualitative comparisons between the learned meta-templates and meta prompts in OPRO should be carefully performed and discussed. 

3. I’m curious about the origin of the term 'text gradient'—where does it come from? Since the text gradients are not real gradients, how could the authors assure their claim in Line 188 "This data-driven extension retains the generalization ... while further improving adaptability in ..." 
  
4. The element list Θ (Appendix E.1) includes 12 components, but the paper does not explain how these components were selected? How would the total number of the list influence the overall results?

5. In Table 2, PTP’s average gain over P3 is 10.52 points on Arena-hard but only 0.12-0.25 points on Alpaca-Eval-2.0 for GPT-4-turbo and GPT-4-1106-preview. The paper attributes this to "task differences" but provides no further analysis. What is the main reason that the difference in tasks/model capabilities cause such huge performance gaps?

### Questions
Several questions are raised in "Weaknesses" part. The main concern is the lack of detailed analysis and comparisons with most related work OPRO. I would be willing to change my recommendation according to the authors' response.

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
This paper introduces a new Prompting to Prompt (PTP) framework for prompt optimization following the ideas of meta learning. It focuses on improving the disadvantages of the existing prompting methods, that they usually have poor transferability across tasks and a strong dependency on task-specific data. In PTP, a highly structured meta template/prompt is optimized through a bi-level optimization process. The inner loop adapts the meta template into task-specific prompts utilizing the sample-level errors, then the outer loop extracts important structural changes that are general enough across tasks. The paper shows empirical improvements in performance in both offline and online prompt optimization scenarios.

### Strengths
1. The meta prompting formulation enables application in both offline and online settings - the meta prompt template can serve as task-level prompts, and can also adapt to query-level prompts.
2. Transferability is the key to this paper. The paper shows strong generalization abilities to different tasks and models empirically.

### Weaknesses
1. The method shows dependency to the manually-curated element list for the meta template. More analysis and discussion on relaxing this requirement are needed.
2. The cost of meta training is not discussed in the paper, making the evaluation of the proposed method incomplete.
3. There is a high reliance on the powerful frontier model for optimizing the prompts through textual prompt updates. In the paper, GPT-4o is heavily used.
4. Some parts of the formulation need to be clarified.

I included more details in the questions section below.

### Questions
1. In the meta prompting formulation of the paper, should we find the argmax over the tasks $i$ as well? Currently, it is unclear whether the dataset $D = \\{ (D_i, C_i) \\}$ includes multiple tasks and whether they are optimized jointly according to the formulation.
2. Why don’t we need a validation set in the PTP algorithm? I see two potential problems: Firstly, the meta prompt can be easily overfitted to the training dataset, for example, by incorporating task-query-specific information explicitly in the prompt, etc. Secondly, prompt updates (or gradients) for both behavioral and structural prompts are executed by the optimizer LLM, there is no guarantee that the updated prompt outperforms the previous prompt on unless you evaluate it against the validation set.
3. Is the element list for the meta template static? How did you come up with these 12 elements and prompts? This begs two questions: Firstly, whether by explicitly designing the meta prompt using these 12 elements manually by a human achieves similar performance as compared to PTP? Secondly, is the PTP framework naturally extendable by making the LLMs curate the elements list by themselves?
4. The meta-training process seems to be computationally expensive to me. The paper still lacks a comprehensive analysis of the computational cost (and/or the API call budget requirements) of the proposed PTP framework. Preferably, an explicit cost comparison alongside the performance comparison (in Table 1 and Table 2) should be presented.
5. Serious typo in Line 236: “Selcet” → “Select”

### Soundness
2

### Presentation
3

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
This paper addresses the challenge of transferable prompt optimization for large language models (LLMs), a problem where existing methods tend to overfit to specific tasks and lack generalizability. The authors propose "Prompting to Prompt" (PTP), a bi-level meta-learning framework that optimizes meta-templates—structured, decomposable representations of prompts—enabling systematic reuse and adaptation across offline and online settings. The method incorporates inner-loop prompt refinement based on feedback and an outer-loop update that distills structural knowledge across tasks. Results on six datasets, including arithmetic, reasoning, and alignment challenges, demonstrate strong average gains over a suite of baselines, and ablations provide further insight into transfer mechanisms and design choices.

### Strengths
1.The paper comprehensively articulates the need for prompt optimization methods that transfer across tasks, highlighting both empirical and practical shortcomings of current approaches (see Section 1 and Table 1).
2.The effect of meta-template scale, instantiation models, and optimization loop parameters is dissected in Figure 4 and Figure 5. For example, Figure 5 systematically shows how accuracy saturates with increasing inner and outer loop iterations, supporting design choices.

### Weaknesses
1. Notation and references are ambiguous—especially defining “prompt elements,” mapping them to meta-templates, and explaining inner vs. outer loop mechanics; clearer legends/footnotes would help.
2. Needs deeper before/after prompt comparisons to show which elements transfer or recur; consider a main-text figure contrasting PTP with ProTeGi/OPRO.
3. Heavy use of API-only LLMs limits technical depth vs. prompt/soft-prompt tuning; misses post-hoc analysis of how meta-templates align with representations or how elements evolve across tasks.
4. Update/search are under-specified (feedback granularity, candidate scoring/beam policy, data flow/complexity) and lack pseudocode; no convergence guarantees, failure-mode diagnostics, or checks for overfitting/negative transfer.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
3
