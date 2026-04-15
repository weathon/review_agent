# Improve Mathematical Reasoning in Language Models with Automated Process Supervision

- Decision: Reject
- Scores: 5, 3, 6

## Abstract
Complex multi-step reasoning tasks, such as solving mathematical problems or generating code, remain a significant hurdle for even the most advanced large language models (LLMs). Verifying LLM outputs with an Outcome Reward Model (ORM) is a standard inference-time technique aimed at enhancing the reasoning performance of LLMs. However, this still proves insufficient for reasoning tasks with a lengthy or multi-hop reasoning chain, where the intermediate outcomes are neither properly rewarded nor penalized. Process supervision addresses this limitation by assigning intermediate rewards during the reasoning process. To date, the methods used to collect process supervision data have relied on either human annotation or per-step Monte Carlo estimation, both prohibitively expensive to scale, thus hindering the broad application of this technique. In response to this challenge, we propose a novel divide-and-conquer style Monte Carlo Tree Search (MCTS) algorithm named \textit{OmegaPRM} for the efficient collection of high-quality process supervision data. This algorithm swiftly identifies the first error in the Chain of Thought (CoT) with binary search and balances the positive and negative examples, thereby ensuring both efficiency and quality. As a result, we are able to collect over 1.5 million process supervision annotations to train Process Reward Models (PRMs). This fully automated process supervision alongside the weighted self-consistency algorithm is able to enhance LLMs' math reasoning performances. We improved the success rates of the instruction-tuned Gemini Pro model from 51\% to 69.4\% on MATH500 and from 86.4\% to 93.6\% on GSM8K. Similarly, we boosted the success rates of Gemma2 27B from 42.3\% to 58.2\% on MATH500 and from 74.0\% to 92.2\% on GSM8K.
The entire process operates without any human intervention or supervision, making our method both financially and computationally cost-effective compared to existing methods.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes an efficient automated method for labeling each step in a process reward model. The core concept involves using binary search to identify the first incorrect step in a rollout and applying a tree-based search to explore the sample space. The resulting artifact contains 1.5 million stepwise annotations. The authors demonstrate the method's effectiveness by comparing the quality of process reward models (PRMs) trained with this dataset to those trained on previously proposed datasets.

### Strengths
1. The paper significantly enhances the efficiency of generating process labels by maintaining a tree structure of stepwise derivations and efficiently expanding the tree to minimize the number of required rollouts.
1. The application of binary search to pinpoint the first incorrect step is a novel approach.

### Weaknesses
1. Empirical results on the MATH and GSM8K datasets indicate a clear upper bound on this distillation-based self-improvement method, showing that performance improvement is constrained by the initial capability of the base model (e.g., Gemini Pro and Gemma 2).
1. The improvement observed on the GSM8K dataset is particularly questionable, as it shows a less than 2% gain compared to majority voting.
1. This paper needs to improve its presentation. On page 3, L159, the sub-indece in x are inconsistent. On page 4, L163 `which is and difficult _?_ to scale`. On page 5, L267, `(s,r)` stands for a full rollout, while `(s,a)` represents a on-step rollout. The duplicate notations are confusing.

### Questions
1. Given the higher inference cost associated with using reward models, does the method ultimately reduce the overall cost of inference?
1. Is the method scalable? For instance, how could one expect to improve MATH500 to 90% accuracy by iterating on this method?
1. How effective is the method when the base model is either very weak or very strong (e.g., performance improvement on o1)?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper introduces OmegaPRM, a divide-and-conquer Monte Carlo Tree Search (MCTS) algorithm aimed at automating the collection of process supervision data for large language models (LLMs). It efficiently identifies the first error in a reasoning chain, thus improving the performance of LLMs in mathematical reasoning tasks, demonstrated by a significant improvement in accuracy on the MATH benchmark.

### Strengths
- The automated process reduces data collection costs significantly compared to traditional human annotation methods.

### Weaknesses
- While OmegaPRM's efficiency is emphasized, a direct comparison with methods that rely on human annotations and noise mitigation strategies would enhance its practical evaluation.

- Present only the results of Gemini and Gemma on MATH500 and GSM8K.

- The accuracy gain is minimal and may be due to variance; the accuracy improvement mentioned in the abstract is not solely attributable to OmegaPRM.

- The proposed OmegaPRM primarily combines MCTS and binary search, which limits its novelty.

### Questions
- Consider including a broader range of experiments to evaluate the generalizability of OmegaPRM across different reasoning domains.

- It might be helpful to discuss more explicitly the limitations of relying on human-annotated data alongside your automated methods, perhaps by providing quantitative analysis on the impact of noise on performance metrics.

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes to train a PRM as verifier for LLM math reasoning. Since human annotation for step-wise reasoning is expensive, the authors propose to use mcts score as the annotation signals. To enhance the efficiency of construct tree, a binary search method is used to accelerate pruning. Trained on the collected step supervision data, the PRM is able to help LLM outperform previous methods on math reasoning in inference time.

### Strengths
- This paper collects a large size of step annotation data without human labelling for math reasoning. Step-wise data is more fine-grained and enable many potentials such as PRM training.

- When preparing the data, the authors contract a space-action tree in math reasoning environment with a efficient binary search approach to locate where a solution starts to make mistake.

- With the trained PRM as verifier, the math reasoning capability improves a bit compared to previous approaches.

### Weaknesses
- This paper aims to improve MCTS efficiency by binary search the first incorrect step and reusing rollouts. However, the efficiency gain described in the experiment section is very vague. The authors only report they can generate much more examples with same computational cost. Since efficiency is one major claim of this paper, it would be more clear to give more details, e.g., with same hardware environment, how much exact time the proposed method can save compared to existing approaches. 

- A candidate poll of all rollouts from the binary search is maintained for later selection in mcts, which might increases memory / storage overhead. The authors could discuss more, e..g, with experiments results, about the time-space efficiency trade-off.

### Questions
- How do the value of $\alpha$.$\beta$, L and $c_{puct}$ are selected empirically?
- Will the collected step annotation data, or the PRM, be released to public?

### Soundness
3

### Presentation
3

### Contribution
3
