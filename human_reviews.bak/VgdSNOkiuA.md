# Adaptive-Solver Framework for Dynamic Strategy Selection in Large Language Model Reasoning

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 5, 6

## Abstract
As the field of artificial intelligence evolves, Large Language Models (LLMs) are showcasing impressive ability in handling complex reasoning tasks. Researchers have developed various techniques utilizing LLMs to tackle these challenges. In real-world situations, problems often span a spectrum of complexities. Humans inherently adjust their problem-solving approaches based on task complexity. However, most methodologies that leverage LLMs tend to adopt a uniform approach: utilizing consistent models, prompting methods, and degrees of problem decomposition, regardless of the problem complexity. Inflexibility of these methods can bring unnecessary computational overhead or sub-optimal performance. To address this problem, we introduce an Adaptive-Solver framework. It strategically modulates solving strategies based on the difficulties of the problems. Given an initial solution, the framework functions with two primary modules. The initial evaluation module assesses the adequacy of the current solution. If improvements are needed, the subsequent adaptation module comes into play. Within this module, three key adaptation strategies are employed: (1) Model Adaptation: Switching to a stronger LLM when a weaker variant is inadequate. (2) Prompting Method Adaptation: Alternating between different prompting techniques to suit the problem's nuances. (3) Decomposition Granularity Adaptation: Breaking down a complex problem into more fine-grained sub-questions to enhance solvability. Through such dynamic adaptations, our framework not only enhances computational efficiency but also elevates the overall performance. This dual-benefit ensures both the efficiency of the system for simpler tasks and the precision required for more complex questions. Experimental results from complex reasoning benchmarks reveal that the prompting method adaptation and decomposition granularity adaptation within the Adaptive-Solver framework enhance performance across all tasks. Furthermore, the model adaptation approach significantly reduces API costs (up to 50%) while maintaining superior performance.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes an adaptive approach to using LLMs that may switch between
different LLMs, change prompting, and decompose the problem based on the
observed performance of initial or partial solutions. The authors describe their
framework and evaluate it empirically.

### Strengths
The paper is well written and explores an interesting idea.

### Weaknesses
Details of it are unclear. In particular the decomposition seems to rely on
manually defined granularities, i.e. a given problem cannot be decomposed
automatically. This imposes a significant burden on the user. The results of the
empirical evaluation seem to suggest that this decomposition is often crucial to
achieving good performance; this should be discussed in more detail.

### Questions
As a user, how would I decide how to decompose, how many levels are needed, and what the
levels represent?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces an adaptive LLM solver method that iterates through different methods to arrive at a solution. Adaptive solver first checks if a given solution is accurate before trying different adaptations.  The adaptations are model, prompting and decomposition granularity. Experiments show how the different adaptations affect performance on various reasoning datasets.

### Strengths
- Originality: The idea of the framework introduced is unique for its flexibility. Most papers focus on trying to improve one of the given adaptations while the adaptive solver takes a different approach. 
- Significance: Searching or solving for the best way to approach a particular question is an important line of work, especially given the high cost of API and variety of inputs.
- The paper is well presented. In particular, the conclusions from the analysis and experiments are easy to find.

### Weaknesses
- There are a limited number of options for the solver and the method requires a lot of pre-processing (to write the in-context examples) for the prompting method. 
- The implementation is not as novel as the original idea. To the best of my understanding, the solver goes through different methods and chooses the best one. A significant improvement would come from making the solver more dynamic and based on the solution. Works that combine planning and LLMs are quite relevant.*
- The paper mentions that the number of solving rounds does not increase much but there is no discussion of the increase in inference time. Trying different approaches until a certain number of iterations has passed or some metric is satisfied will increase the inference time significantly. This could be problematic for real-time applications. 


*Relevant papers: 
- Wang, Lei, et al. "Plan-and-solve prompting: Improving zero-shot chain-of-thought reasoning by large language models." arXiv preprint arXiv:2305.04091 (2023).
- Hao, Shibo, et al. "Reasoning with language model is planning with world model." arXiv preprint arXiv:2305.14992 (2023).

### Questions
- For the Decomposition Granularity Adaptation experiment, what model is used? GPT-3.5? Is there a way to compare this with GPT-4?
- How is Decomposition Granularity different from prompting? From Figure 1, c and d look quite similar. 
- Were there experiments using a larger solver list? From Table 1, each of the 3 solvers has the best performance in at least one of the datasets. 
- Results clearly depend on what is in the solver. Is there a way to choose what methods to put into a given solver? 
- How did inference time change per types of adaptations?  
- How were in-context examples chosen?

### Soundness
2 fair

### Presentation
4 excellent

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper presents an approach for combining multiple different strategies in order to solve problems using LLMs. This is akin to portfolio selection since there is generally no "one-size-fits-all" approach to solving problems. In order to do so, the authors propose the adaptive-solver (AS) framework for LLMs. 

AS consists of three different adaptation strategies, (a) Model adaptation where the LLM models are changed from cheaper to more advanced albeit expensive models, (b) Prompting method adaptation wherein different prompting methods are utilized for problems, and finally decomposition granularity adaptation that tailors the decomposition granularity of prompts from coarse to finer.

An adaptation module consists of a portfolio of such solvers and the authors use an evaluation module with a consistency metric to determine the evaluation criteria for switching solvers. The authors then provide an empirical evaluation of their approach and perform several ablations of each of the modules.

### Strengths
1) The paper is generally well-written (even though the empirical could have been organized a bit better) and the ideas are expressed clearly

2) The idea of having a portfolio of such selection strategies makes sense since empirically it is known that there is usually not a single approach that can outperform others

### Weaknesses
The paper is quite interesting but it seems that the empirical evaluation section is (a) bit hard to follow, and more importantly (b) the results only show a marginal improvement over baselines.

a) The paper only improves over baselines by a nominal ~3% (Table 1). This does not seem very significant to me and is further exacerbated by the fact that different, hand-coded variations of AS are needed to outperform the baselines as such.

b) The paper claims that AS can cut down on API costs but a cost analysis vs baselines is not provided. Table 2 only provides cost analysis vs using two versions of GPT but does not include overall costs for the entire pipeline.

c) Similarly, Table 3 only shows marginal improvements for the decomposition granularity ablation. 

Overall, the ablations are interesting but the process seems overly hand-coded with not enough improvements over the baselines. (Even the strategies for choosing solvers is driven by expert-knowledge). For example, how many times was strategy 1 (choose the last solver in the list) selected in your evaluation. Such information is missing in the main paper.

### Questions
Id like to thank the authors for their extensive experiments. I've listed my questions below. I hope that the authors can resolve my queries.

1. Could you please comment on (b) and provide a reason as to why overall costs for the entire pipeline are not included in the paper.

2. Currently, it feels like most of the experiments are ablations. I would have preferred to have seen results with a general AS solver list and a more comprehensive comparison with baselines.

3. I can understand the reason for the ablations but is there any reason as to why all baselines were not tried on for all datasets? For example, Table 2 only uses ZeroCoT for prompting and only the model adaptation is explained. I think that the overall efficacy of the pipeline can only be clearly determined when the pipeline is used everywhere and not selectively applied to different datasets. I appreciate the authors trying to reduce the # of variables but this only made the evaluation more confusing for me.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper presents and demonstrates a simple algorithm for achieving a better cost-accuracy trade-off for reasoning tasks with LLMs. The high-level idea is to construct a cascade using different models, prompts, and/or granularities of decomposition. A crucial element is the ability to evaluate whether a solution is likely to be correct, which is achieved using the consistency of multiple samples at some non-zero temperature. The method is evaluated on a collection of reasoning datasets, and is shown to achieve a significant reduction in cost, sometimes even achieving an increase in accuracy.

### Strengths
1. Baselines and components are all highly recent (2022-2023).
1. Considers three different types of solver adaptation. Judicious choice of solvers in experiments.
1. Significant reductions in cost achieved. For those employing LLMs, it is useful to be aware of the efficacy of a cascade using the consistency check.
1. Ablative studies are well designed and well presented.

### Weaknesses
1. Little technical novelty - mostly an empirical study.
1. It seems like the temperature could have a significant impact on the consistency check, however there was no study of the effect of varying temperature.
1. It would be useful to present the ROC curve (FPR-FNR tradeoff) of the consistency check for each solver, ideally at a range of temperatures.
1. While Figure 3b shows which of the 3 decomposition prompts was used, it would be useful to know how many solvers were tried for each experiment, and how often the cascade "dropped through" to the final solver.
1. It would be good to include a discussion of determining an optimal cascade (perhaps assuming that the errors of different models are independent) per budget for a given dataset.

Suggestions (no need to address):
1. For a scientific context, I would tone down some of the grandiose language ("this innovative method represents a significant step in dynamic strategy selection", "holding vast implications for the realm of artificial intelligence").
1. Personally, I don't like the use of the word "solver" for the current purpose. Possible alternatives: tactic, strategy, protocol.

### Questions
Mainly just address weaknesses listed above. Additional questions:

1. Which model does each method use in Table 1? Could include this in the caption.
1. It's unfortunate that OpenAI's pricing affects the "cost saving"; changes in pricing will change the results. Is there any way around this? Is it possible to obtain flops (or kWh, but that too is technology dependent)? Otherwise at least note that this uses pricing as at [date].

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
