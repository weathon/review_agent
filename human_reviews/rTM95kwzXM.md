# Language Model Preference Evaluation with Multiple Weak Evaluators

- Avg Score: 5.25
- Decision: Reject
- Scores: 5, 3, 8, 5

## Abstract
Despite the remarkable success of Large Language Models (LLMs), evaluating their outputs' quality regarding *preference* remains a critical challenge. Existing works usually leverage a powerful LLM (e.g., GPT4) as the judge for comparing LLMs' output pairwisely, yet such model-based evaluator is vulnerable to *conflicting preference*, i.e., output A is better than B, B than C, but C than A, causing contradictory evaluation results. To improve model-based preference evaluation, we introduce GED (Preference Graph Ensemble and Denoise), a novel approach that leverages multiple model-based evaluators to construct preference graphs, and then ensemble and denoise these graphs for better, non-contradictory evaluation results. In particular, our method consists of two primary stages: aggregating evaluations into a unified graph and applying a denoising process to eliminate cyclic inconsistencies, ensuring a directed acyclic graph (DAG) structure. We provide theoretical guarantees for our framework, demonstrating its efficacy in recovering the ground truth preference structure.  Extensive experiments across ten benchmark datasets show that GED outperforms baseline methods in model ranking, response selection, and model alignment tasks. Notably, GED combines weaker evaluators like Llama3-8B, Mistral-7B, and Qwen2-7B to surpass the performance of stronger evaluators like Qwen2-72B, highlighting its ability to enhance evaluation reliability and improve model performance.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This manuscript introduces GED (Graph Ensemble and Denoise), a novel framework for evaluating Large Language Models (LLMs) by constructing a denoised preference graphs from multiple weak LLM evaluators. The proposed framework aims to address the common problem of conflicting preferences in LLM evaluations. The manuscript provides theoretical guarantees for the framework's ability to recover the true preference structure and presents experimental results across several LLM evaluation tasks.

### Strengths
1. The manuscript clearly identifies a significant challenge in current LLM evaluation: the presence of conflicting preferences in model-based evaluators. The proposed GED framework offers a practical and promising approach to mitigate this issue by leveraging the collective wisdom of multiple weaker evaluators.
2. The theoretical analysis provides a solid foundation for GED, demonstrating its ability to recover the ground truth preference structure with high probability under realistic assumptions about noise in individual evaluator preferences.

### Weaknesses
1. The presentation of the experimental results could be significantly improved. Several details includes evaluator selection, experimental setup, and ablation study, require clarification and further elaboration to enhance understanding and reproducibility.

### Questions
1. The distinction between the "single model" and "single evaluator without graph denoising" settings is unclear. Please elaborate on the differences in these two evaluation setups. 

2. The use of "Mistral-7B" (line 764) might be misleading, as it refers to a different open-source model than the later mentioned "Mixtral-8x7B-Instruct-v0.1". Please clarify the specific model used and maintain consistent nomenclature throughout the manuscript.

3. The choice of weak evaluators varies across the different evaluation tasks (response selection, model ranking, and instruction tuning). Moreover, the evaluators employed for instruction tuning are not explicitly stated. Please elaborate on the evaluator selection rationale for each task and complete the missing details for instruction tuning.

4. In Section 5 (Model Ranking), using all baseline 70B level models as the weak evaluator set raises concerns about fairness. The computational cost of this approach could exceed that the sum of all baseline models. Please address this potential issue and justify the chosen evaluator set for model ranking. Consider exploring alternative configurations with smaller, more comparable evaluator sets.

5. The ablation study in line 374 describes using various rank aggregation methods to validate the effectiveness of the graph ensembling step. However, Table 2 presents results for different aggregation methods applied to both single evaluators and GED. This creates confusion. Is this aggregation process necessary in GED? If so, which method is adopted (in other experiments)?

6. The manuscript would benefit from a comparison with point-wise LLM evaluation methods (like MT-Bench). These methods assign absolute scores to responses and typically do not suffer from the conflicting preference problem. A discussion or comparison on the trade-offs between pairwise preference-based methods (like GED) and point-wise methods would strengthen the paper.

7. Typos (e.g., "Alabation" in line 372).

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper proposes an interesting way to denoise conflicting preferences of multiple weak models using graph ensembling and denoising. To mitigate the noise of individual evaluators, the authors aggregate multiple of them together to find the majority consistency and then filter out the inconsistencies. The authors provide multiple results from different benchmarking suites.

### Strengths
The authors provide extensive experiments to validate their proposed method.

### Weaknesses
Related Work Needs Further Development: The section on related work in preference evaluation for LLMs currently stops at the point that LLM evaluators can introduce inconsistencies. Since this is not the first study focused on denoising preference datasets for LLMs, the authors should expand their literature review to include relevant prior work.

There are two primary approaches in the existing literature about denoising preference learning: (1) within-model preference modeling approaches, such as robust DPO (rDPO) by Chowdhury et al. (2024) and conservative DPO (cDPO) by Mitchell (2023), and (2) modular approaches that denoise data before preference modeling, such as CURATRON (Nguyen et al., 2024). This paper appears more closely aligned with the latter, modular approach.

Aim of the paper: The stated aim of this paper is to denoise weak LLM evaluations. However, preference inconsistencies can arise with both weak and strong evaluators, including human evaluators. Why, then, is the scope limited only to weak evaluators? For instance, if GPT-4 were included as an evaluator but showed differing preferences from other LLMs, would the proposed methods filter out its preferences as well? How are "weak" evaluators defined, and by what criteria? Smaller models may perform better in certain domains based on their training data (than even larger GPT models), which can lead to cases where minority preferences are, in fact, optimal. This raises the question: how well does the proposed denoising approach preserve these potentially valuable minority perspectives? Could minority preferences, which may offer the correct or preferred outcome, be unintentionally excluded in the process?

Experiment: The paper misses some interesting aspects that they could dive deeper into. What is the improvement of the proposed approach against a majority voting baseline where we just take the majority for each pairwise comparison as the preferred answer (Some sort of Self-consistency based on evaluators)? Also, how does the amount of evaluators affect the denoising quality?

Results: When comparing the best single evaluator to the ensemble of multiple evaluators in Tables 1 and 2, the differences appear minimal, particularly given the high variance across other rows. The improvement shown by the proposed method doesn’t seem substantial, especially considering the increased computational cost of using 4 to 5 models instead of a single one. This raises questions about whether the performance gains justify the added complexity and resource demands.

References:
Chowdhury, S. R., Kini, A., and Natarajan, N. (2024). Provably Robust DPO: Aligning Language Models with Noisy Feedback.

Mitchell, E. (2023). A Note on DPO with Noisy Preferences & Relationship to IPO.

Nguyen, S. T., Naresh, N. U., & Tulabandhula, T. (2024). CURATRON: Complete and Robust Preference Data for Rigorous Alignment of Large Language Models.

### Questions
What is the detailed information about the internal parameters/prompts the authors use for each LLM and LLM evaluator?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper casts the LLM evaluation problem as a pairwise tournament. This is an important problem in model evaluation as it can be used to rank models (picking an LLM for a task), choosing a response among several responses, or in preference alignment. The idea of the paper is to use a suite of LLM-judges to make pairwise evaluations, construct a weighted graph (where the weights denote the number of judges that prefer A over B). A denoising process removes the feedback arc set that gives an acyclic graph from which rankings can be recovered.

### Strengths
I commend the authors on a well presented idea on an important problem. The paper is well executed and central claims have supporting evidence. The idea of aggregating several (unreliable) judgements to make more robust evaluation in this manner is, to the best of my knowledge, new. So this paper makes a good contribution to the literature and merits inclusion in the conference program.

The reduction of the problem to a tournament graph is an interesting angle to this. The experiment results are also well crafted and support the main thesis of the paper.

### Weaknesses
- If there is a large set of models/responses to evaluate, this may get expensive. The method requires pairwise evaluations from all evaluators, i.e. $|\mathcal{A}| \times |M|^2$ inferences. Ideas to limit the number of inferences could be useful.

- In the graph construction, it would appear all evaluators are given equal weight (Algorithm 1/2). In reality some models are better evaluators than others. Evidence on LLM-as-a-judge suggests that performance is not always good. Although pairwise evaluations are an easier task.

- Evaluators are also susceptible to various biases. Position bias for instance, where the order of the pairs shown to the evaluator yields different evaluations. Or token bias, where the evaluator prefers a specific option (e.g. latent preference for Option A vs. Option B). These are important factors in the the "data generation" process that could be factored in to the graph construction and analysis.

### Questions
What is the metric being reported in Table 1.?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces GED (Preference Graph Ensemble and Denoise), a new method for evaluating large language models (LLMs) using pairwise comparisons made by multiple "weak" evaluators, which could be smaller or less capable LLMs. The authors address a key challenge in preference evaluation: the tendency of even the best LLMs to produce conflicting preferences (e.g., cycles in a preference graph: A is better than B, B is better than C, but C is better than A). GED constructs preference graphs from these pairwise comparisons and combines these graphs from multiple evaluators, and applies a denoising process to remove inconsistencies (aka the cycles in the graph) and produce a more reliable ranking of LLM outputs. The paper demonstrates the effectiveness of GED on a variety of tasks, including response selection, model ranking, and instruction tuning.

### Strengths
- Addresses a critical problem in LLM evaluation: The paper tackles the issue of inconsistent preference judgments by LLMs, which is crucial for obtaining reliable evaluation results. 

- Good theoretical grounding: The authors provide a theoretical analysis demonstrating that GED can recover the ground truth preference structure with high probability under reasonable assumptions. (See below for further theoretical grounding that may be necessary)

- Extensive experimental validation: The paper evaluates GED on ten benchmark datasets across three distinct LLM evaluation tasks, showing consistent improvements over existing methods, and even more powerful individual LLM evaluators. 

- Practical implications: GED enables the use of weaker and more accessible LLMs as evaluators, which can be particularly beneficial in scenarios where access to powerful models like GPT-4 is limited.

### Weaknesses
- Cost: Using multiple evaluators and aggregating their preferences is an expensive process. In a dense preference graph case, one would require preferences across all pairs of weak evaluators. While it can be done randomly or strategically, the paper does not present a theoretical result on how many such labels are needed for statistical significance. I would encourage the authors to cover some literature on active learning, preference learning based on pairwise preferences. A quick google search leads to [1] which seems to have many other references related to this question. 
[1] Saha, A., Shivanna, R., & Bhattacharyya, C. (2019, July). How many pairwise preferences do we need to rank a graph consistently?. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 33, No. 01, pp. 4830-4837).  

Cost is a major limitation of this work and is essential to be answered for the work to find any practical application. 

- Limited analysis of evaluator selection: The paper focuses on demonstrating the effectiveness of GED when combining weaker evaluators but the choice of the set of weak evaluators is not really discussed. It would be insightful to explore how the selection of specific evaluators, e.g., their diversity, capabilities, etc., and how it affects the performance and validity of GED.

- Dependence on pairwise comparisons: While the preference graph framework effectively captures pairwise relationships, it might not fully capture more nuanced preference structures that could exist among multiple outputs. See questions below.

### Questions
- The paper primarily uses accuracy-based metrics for evaluating response selection and model ranking. Could you elaborate on how GED might be adapted or evaluated for tasks that require more nuanced quality assessments, such as evaluating the coverage, creativity or factuality of LLM generated outputs?
- The theoretical analysis assumes equal weighting of edges in the preference graphs. How sensitive is GED's performance to different weighting schemes? Would using confidence scores from evaluators, if available, impact the results?
- The denoising process aims to minimize edge removals to preserve information. Could you discuss alternative approaches to denoising, such as edge re-weighting or probabilistic methods, and their potential advantages or disadvantages?

### Soundness
3

### Presentation
3

### Contribution
2
