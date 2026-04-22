# Mordal: Automated Pretrained Model Selection for Vision Language Models

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Incorporating multiple modalities into large language models (LLMs) is a powerful way to enhance their understanding of non-textual data, enabling them to perform multimodal tasks.
Vision language models (VLMs) form the fastest growing category of multimodal models because of their many practical use cases, including in healthcare, robotics, and accessibility.
Unfortunately, even though different VLMs in the literature demonstrate impressive visual capabilities in different benchmarks, they are handcrafted by human experts; there is no automated framework to create task-specific multimodal models.

We introduce Mordal, an automated multimodal model search framework that efficiently finds the best VLM for a user-defined task without manual intervention.
Mordal achieves this both by reducing the number of candidates to consider during the search process and by minimizing the time required to evaluate each remaining candidate.
Our evaluation shows that Mordal can find the best VLM for a given problem using $8.9\times$--$11.6\times$ lower GPU hours than grid search.
We have also discovered that Mordal achieves about 69\% higher weighted Kendall’s $\tau$ on average than the state-of-the-art model selection method across diverse tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the critical and computationally challenging problem of selecting optimal pretrained vision encoders and large language models (LLMs) for constructing Vision-Language Models (VLMs) for a specific downstream task. The authors identify that existing model selection methods are inadequate for VLMs, as they fail to account for the necessity of vision-language alignment, and that a brute-force grid search is prohibitively expensive. To solve this, they propose Mordal, an automated framework that efficiently searches the model combination space. Experiments across seven datasets (GQA, VizWiz, ChartQA, DocVQA, ScienceQA, AI2D, MMMU) demonstrate that Mordal achieves faster search time and better performance compared to state-of-the-art model selection baselines.

### Strengths
1. The paper clearly identifies a significant pain point in the VLM development lifecycle: the high cost and unpredictability of selecting pretrained backbones. The problem is timely and relevant given the proliferation of open-source models.
2.  Mordal is a novel solution that smartly combines several techniques (CKA-based clustering, early stopping, scaling prediction) into a cohesive framework specifically tailored for the VLM selection problem. The two-pronged approach of reducing candidate count and evaluation cost is logical and effective.
3. The results are impressive. The reported speedups (up to 11.6×) are substantial, and the fact that Mordal finds the top-1 model in 6 out of 7 tasks demonstrates its high effectiveness. The ablation studies and sensitivity analysis provide strong evidence for the contribution of each component.

### Weaknesses
1. The sensitivity analysis (Appendix D.2) shows that Mordal's performance and efficiency are sensitive to clustering thresholds (t_ve, t_lm) and exploration parameters (topk_inter, topk_intra). While the paper provides default values, the need for task-specific tuning could be a practical hurdle for non-expert users. An automated or adaptive method for setting these thresholds would be a valuable future improvement.
2.  The evaluation is primarily focused on ~7B parameter LLMs. The authors briefly discuss the challenge of scaling to much larger models (e.g., 70B) in Appendix E, noting the increased computational overhead and potential ineffectiveness of similarity metrics. An empirical demonstration of Mordal's performance on a larger scale (even with a smaller subset) would strengthen the claims about its general applicability.
3. The CKA-based clustering, while reducing overall search time, introduces its own computational cost. Although likely smaller than full training, a quantitative analysis of this overhead (e.g., GPU hours spent on clustering vs. training) would provide a more complete picture of the total cost savings.

### Questions
1. The experiments are conducted using a simple MLP projector. As noted in Appendix E, other projectors like Q-Former exist. How would the representation similarity (CKA) and the observed scaling laws be affected by more complex, cross-attention based projectors? Have you done any preliminary experiments to validate Mordal's effectiveness in such architectures?
2. In the case of ChartQA where Mordal did not find the top-1 model, could you provide a more detailed post-mortem analysis? Was the failure primarily due to the clustering step (the best model was in a eliminated cluster) or the evaluation step (it was outranked by others within its cluster during scaling prediction)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
This paper focus on VLM selection problem and propose Mordal, an automated framework for efficiently selecting optimal pretrained models (vision encoders and LLMs) for VLMs training on downstream tasks. Mordal addresses two optimization dimensions simultaneously:
1. Reducing Search Space via Candidate Clustering: Mordal use centered kernel alignment (CKA) to clusters vision encoders first and then clusters LLMs based on fixed vision representations. For inter-cluster selection, Mordal evaluates one representative per cluster, eliminates weak clusters and the remaining top K clusters are evaluated in the intra-cluster stage.
2. Reducing Evaluation Cost per Candidate. Mordal applies: a) Early stopping: Uses Successive Halving Algorithm to eliminate poor performers during inter-cluster evaluation. b) Scaling prediction: Exploits observational scaling laws—trains on small data samples and predicts full-data performance via log-linear regression

### Strengths
1. Mordal addresses a practical problem where manual selection for VLM is unreliable and grid search is infeasible.

2. Mordal utilizes the proper metric to cluster VLM combinations.

3. The experiments are extensive and results are strong.

4. Mordal requires no manual intervention

### Weaknesses
1. Limited Scope and Generalizability. Mordal are architecture-specific and only evaluated on MLP projector-based VLMs. It is unclear if it extends to Q-former or other architectures.

2. Only 7 vision encoders and 7 LLMs are tested. It is unclear how it will work when scaling to 10x model zoo.

### Questions
1. The reviewer is wondering whether it is possible to perform additional experiments on Q-former or other architectures.

2. Could authors provide some analysis on how Mordal would work when the model zoo is scaled to 10 times or more?

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
2

### Summary
This paper introduces Mordal, an automated framework for selecting optimal pretrained vision encoders and language models when constructing Vision Language Models (VLMs). The key insight is that different combinations of pretrained components yield varying performance on different tasks, yet current VLM construction relies on manual selection. Mordal addresses this by: (1) clustering similar model candidates using CKA-based representation similarity, (2) employing a two-stage inter/intra-cluster evaluation process, and (3) utilizing early stopping and scaling prediction to reduce evaluation costs. The authors demonstrate that Mordal can identify top-performing VLM configurations 8.9×-11.6× faster than grid search while achieving 1.2×-3.3× better performance than existing model selection methods.

### Strengths
- This paper addresses a real problem faced by practitioners - selecting optimal pretrained components for VLMs is currently ad-hoc and computationally expensive.

- The authors have conducted extensive experiments across 7 datasets, 49 model combinations, with detailed ablation studies demonstrating the contribution of each component.

- The proposed method achieved 8.9×-11.6× speedup over grid search while maintaining great performance.

### Weaknesses
- The evaluation focuses on 7B parameter models with MLP projectors. As acknowledged, extending to smaller (1B) or larger (70B) models presents challenges. The framework's effectiveness on other projector architectures (e.g., Q-former) is unexplored.

- The performance is sensitive to clustering thresholds ($t_{ve}$, $t_{llm}$), which may hurt its generalization to other unexplored LLM and vision encoders.

- The current design optimizes for single tasks independently. It would be better if the authors could address multi-task learning scenarios, which are also common in practice.

### Questions
None

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper “MORDAL: Automated Pretrained Model Selection for Vision-Language Models” tackles the problem of efficiently identifying the optimal combination of a vision encoder (VE) and a language model (LLM) from an open model library, given a downstream task and a small alignment dataset.
The proposed framework, MORDAL, works in three stages. First, it clusters vision encoders using CKA, and within each visual cluster, it clusters language models based on fixed image representations transformed by a lightweight projector. The Cartesian product of these clusters forms candidate groups.
Then, it performs two-stage evaluation: inter-cluster comparison using cluster medoids to filter top-K candidates, followed by fine-grained intra-cluster ranking.
During evaluation, MORDAL employs Successive Halving for early stopping and fits a log-linear relationship between error and sample ratio using a few “observation points.” This allows it to extrapolate the expected full-data error without full training.
Across 49 VE×LLM combinations and 7 datasets, MORDAL achieves 8.9×–11.6× speedup over exhaustive search while matching the top-1 result in 6 out of 7 tasks. Compared to EMMS, LogME, LEEP, and NLEEP, it yields higher ranking consistency in terms of weighted Kendall’s τ.
Overall, the work frames “pretrained component selection” as an alignment-phase selection problem, combining candidate clustering, staged evaluation, and observation-based extrapolation to significantly reduce cost, demonstrating strong practical value.

### Strengths
MORDAL enables rapid screening of visual encoder–language model combinations within open model repositories, saving roughly 9–12× the time compared to exhaustive search, while maintaining top hit rates on most tasks. Its two-level clustering greatly reduces the computational cost of CKA, and the SHA and extrapolation strategies allow reusing intermediate checkpoints, achieving overall engineering efficiency.
Experimental results show that MORDAL’s ranking consistency surpasses that of EMMS, LogME, LEEP, and NLEEP, indicating more stable predictions of downstream adaptability. The paper also provides time decomposition and hyperparameter sensitivity analyses, helping to understand cost distribution. The authors also explicitly acknowledge limitations regarding small models, large models, and non-MLP projectors.

### Weaknesses
The experimental setup and implementation in the paper are not fully aligned; it should be clarified whether LoRA fine-tuning is enabled by default, and performance and resource overhead should be reported separately for both modes.
There appear to be typos in the extrapolation and early-stopping logic of Algorithm 1, which may require correction. The terms “maximum sample ratio” and “initial ratio” should also be made consistent.
The baseline comparison alters the “training-free” assumption, so a strictly training-free version should be added. The statistical robustness is insufficient — Tables 2 and 3 only report point estimates without confidence intervals, and the false stop rate of early termination is not quantified.
The similarity metric is relatively limited, relying solely on CKA without comparing other metrics such as SVCCA, PWCCA, or layer-weighted CKA.
The time decomposition does not list CKA computation and the warm-up phase GPU-hours separately, making the resource reporting incomplete. Some wording is potentially misleading: in the abstract, “1.2×–3.3× better performance” should be revised to “higher weighted Kendall’s τ”, and spelling and formula symbols should also be corrected.

### Questions
Please clarify the default setting used throughout the paper (i.e., projector only or projector + LLM-LoRA), and report τ, hit rate, and GPU-hours separately for both modes.
Could you add results for a strictly training-free baseline, as well as a version under the same extrapolation framework, to isolate the impact of partial training time?
Please report mean and variance across multiple random seeds, especially for ChartQA, and quantify the false rejection rate of SHA early stopping.
Regarding resource accounting, are the CKA computation and warm-up phase included in the total time reported in Table 2? Please supplement GPU-hour absolute values and proportions.
Finally, could you provide detailed reproduction code and guidelines?

### Soundness
2

### Presentation
3

### Contribution
3
