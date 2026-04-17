# On the Eligibility of LLMs for Counterfactual Reasoning: A Decompositional Study

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Counterfactual reasoning has emerged as a crucial technique for generalizing the reasoning capabilities of large language models (LLMs). By generating and analyzing counterfactual scenarios, researchers can assess the adaptability and reliability of model decision-making. Although prior work has shown that LLMs often struggle with counterfactual reasoning, it remains unclear which factors most significantly impede their performance across different tasks and modalities. In this paper, we propose a decompositional strategy that breaks down the counterfactual generation from causality construction to the reasoning over counterfactual interventions. To support decompositional analysis, we investigate 11 datasets spanning diverse tasks, including natural language understanding, mathematics, programming, and vision-language tasks. Through extensive evaluations, we characterize LLM behavior across each decompositional stage and identify how modality type and intermediate reasoning influence performance. By establishing a structured framework for analyzing counterfactual reasoning, this work contributes to the development of more reliable LLM-based reasoning systems and informs future elicitation strategies.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
Large language models typically struggle on counterfactual reasoning tasks, however it is unclear *why* they struggle, particularly at which parts of such reasoning. This submission proposes a benchmark to evaluate different aspects in LLMs' causal reasoning.They isolate that LLMs struggle to properly identify causal variables and find the counterfactual mediator and outcome even when given the variables, their relationships and the intervention to perform.

### Strengths
- The approach is original; I have never seen any attempt at explaining the difficulties of LLMs in causal reasoning.

- The division into subtasks looks sound to me.

- Diversified source of tasks.

### Weaknesses
- The benchmark seems to impose the strict presence of four variables : covariates, treatment, mediator, outcome. However, the mediator could be absent from the context, as in the example:  “A person is running a marathon and collapses.” l.467-469. It is unclear why LLMs should explicitly,identify “Dehydration” as a mediator, and not other sources.

- It is unclear what prompts are given to LLMs : from my understanding, it is the "CRASS Prompt" given in Appendix A, but it seems to apply to all datasets?

- Typo? "the primary challenge lies not in" l.390-391 : it should be "lies in", right?

### Questions
- Can you answer/address the above weaknesses?

- In the datasets, are the examples *only* made up of the $X,T,M,Y$ variables, or are any other variables present such that the model must extract the relevant variables *among other options*?

- Did humans or LLMs annotate the paper's benchmark dataset? Which of these scored LLMs' answers, or was this done through another method?

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
5

### Summary
This paper makes three key contributions:

- It introduces a **decompositional framework** for evaluating counterfactual reasoning in LLMs, breaking it into four stages: causal variable identification, causal graph construction, counterfactual intervention identification, and outcome reasoning.
- It constructs a **comprehensive multimodal benchmark** using 11 datasets across text, image, code, and symbolic modalities, enriched with causal annotations and graphs.
- It provides an **in-depth evaluation and improvement strategy**, identifying key bottlenecks (especially in mediator reasoning and cross-modal tasks) and proposing tool augmentation and advanced prompting techniques to enhance LLM performance.

### Strengths
1.  **Systematic and Granular Evaluation Framework:** The paper's primary strength is its decompositional approach, which breaks down the complex task of counterfactual reasoning into four distinct, measurable stages. This allows for a much more precise diagnosis of *where* and *why* LLMs fail, moving beyond a monolithic "pass/fail" assessment to identify specific bottlenecks, such as the particular difficulty with implicit mediators.

2.  **Comprehensive and Multimodal Benchmark:** The authors construct a substantial and diverse benchmark by curating 11 datasets spanning various modalities (text, image, code, symbols) and tasks. This breadth ensures that the findings are not limited to a single domain and provides a robust testbed for evaluating the generalizability of LLMs' counterfactual reasoning abilities.

3.  **Actionable Insights and Proposed Solutions:** The paper goes beyond mere identification of problems by proposing and evaluating concrete strategies for improvement. The demonstration that tool-augmentation can alleviate modality-specific issues and that advanced prompting can help with implicit reasoning provides valuable, actionable directions for future research aimed at enhancing LLM reasoning capabilities.

### Weaknesses
I think there are 2 brief weaknesses of the paper:

*   **Potentially Artificial Evaluation:** The benchmark relies on pre-annotated causal structures, which may not reflect the challenge of inferring causality from raw, unstructured data.

*   **Surface-Level Diagnosis:** The analysis identifies performance bottlenecks but offers a high-level explanation (e.g., "working memory") without deeply investigating the underlying architectural mechanisms in LLMs that cause these failures.

### Questions
The benchmark relies on curated causal structures. To what extent do you believe the performance on these structured tasks generalizes to real-world scenarios where causal graphs are not provided and must be inferred from raw data? Did you run any experiments on "raw" data without this scaffolding?

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
5

### Summary
This paper examines counterfactual reasoning in LLMs across a range of modalities and models.
The paper shows that by decomposing the reasoning scenario to causality construction and counterfactual intervention, the reasoning ability of LLMs under counterfactual scenarios can be greatly improved.
By leveraging tools that can help the model identify the causal variable, the performance can be further improved.

### Strengths
1. The overall methodology of decomposing the counterfactual reasoning process is novel and the experiments show this really helps.
2. The experiments cover a wide range of dataset design specifically for counterfactual reasoning.
3. The final proposed method seems to be easy to adopt for any LLM for reasoning.

### Weaknesses
1. The experiments covers many datasets, but it lacks comparison on model scale, for example, Qwen 3 provides models across different scales, it could make the paper stronger if some results are shown there.
2. The NER tools are designed to use Bert like models, however, would it be possible that the tools are instantiated by another model using different prompts?

### Questions
See weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a decomposition of counterfactual reasoning into smaller sub-tasks: causal variable identification, causal graph construction, counterfactual identification, and outcome reasoning. The authors evaluate the performance of different LLMs in performing these sub-tasks across a variety of datasets, identify variable identification and counterfactual identification as a weak point, and propose strategies to improve it.

### Strengths
It has been observed repeatedly that LLMs perform worse when answering counterfactual queries relative to factual ones. The paper’s attempt at understanding why in a more fine-grained fashion is a significant problem.

Experimental evaluations are comprehensive in terms of the number of models and the variety of datasets they consider.

### Weaknesses
It is not clear how the performance on the four sub-tasks relate to the end-to-end performance. Establishing this relation is important when interpreting performance on these sub-tasks as decomposition.

For instance, the paper concludes that LLMs are generally better at Task 1 than Task 2 conditioned on correct results from Task 1 (which is supported by experiments and indeed seems to be the case). However, it could still be the case that starting from the inputs of Task 1 and directly querying for the outputs of Task 2 is as easy as performing Task 2 conditioned on correct results from Task 1. If, for instance, processing X, Y, Z variables implicitly is somehow easier than outputting them explicitly.

While the above is probably unlikely to be the case, it is a crucial validation to perform in order to confirm that Task 1 and Task 2 (conditioned on Task 1 results) behave as intended. Evidence provided in the paper does not provide support for attributing poor end-to-end Task-1 inputs to Task-2 outputs performance to poor Task 1 performance. 

I want to emphasize that this is a general consideration for all sub-tasks, the entire chain, not just Tasks 1 and 2 specifically.

The paper doesn’t discuss the related work to a sufficient depth. In particular, other methods of evaluating counterfactual reasoning and whether there have been attempts at decomposing this process into smaller steps in evaluation is not discussed.

The four sub-tasks that are identified, while generally sound, are arbitrary. Their choice is not motivated.

While evaluations in Section 4.1 are interesting, they are not exactly actionable. They help identify which sub-tasks are the weak points to be improved (which arguably all a decomposition is supposed to do) and the paper suggests improvement for these weak points. However, the suggestions take the decomposition as a given. When the goal is to answer a question end-to-end, starting from the given of Task 1 and providing the outputs of Task 4, the paper does not explore the impact of going through Tasks 1, 2, 3, and 4 as a pipeline rather than direct queries. Moreover, it does not explore whether the task specific improvements in Section 4.2 lead to a comparable end-to-end gain as well.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
