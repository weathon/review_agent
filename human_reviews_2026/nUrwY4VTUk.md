# OASIS: An Optimized Approach to Systematic Calibration Data Selection

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 4, 4

## Abstract
Post-training pruning is a critical technique for compressing Large Language Models. However, as shown in previous research, its effectiveness is highly sensitive to the small set of calibration data used for estimating parameter importance. Current calibration data selection relies on simple heuristics like random sampling or entropy, which often leads to suboptimal and inconsistent pruning outcomes: the same pruning method applied with different calibration data can cause up to 3× variance in post-pruning perplexity. In this work, we reveal the source of this inconsistency: calibration samples are not equally important; a quality hierarchy exists within any data pool. Not only does mixing high- and low-quality data cause a performance degradation, but the quality of the sample is context-dependent, changing with the specific model and pruning algorithm, rendering static filtering infeasible and necessitating an adaptive solution. Therefore, we introduce OASIS, the first end-to-end framework that directly optimizes calibration data selection with respect to the pruned model’s downstream performance. OASIS leverages a differentiable soft-mask proxy to propagate task-level gradients back to the calibration data, enabling dynamic discovery of the most beneficial subset. Experiments show that our approach improves the performance of diverse state-of-the-art pruning methods, establishing a new standard for data-aware model compression.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper provides an analysis of the influence of individual calibration quality and proposes a soft-mask-based pruning method combined with data selection to improve pruning performance. Experimental results show that the proposed method outperforms existing randomized and synthetic data selection approaches.

### Strengths
The paper provide detailed analysis for the importance of data selection based on multiple pruning methods, which provide good motivation for the problem studied.

### Weaknesses
- The paper needs a thorough proofread. Additionally, the overall presentation should be improved. Although I did not read every section in detail, I noticed a significant number of writing issues throughout the paper (see the Questions section for more specific examples).
- Beyond the writing, the contribution of this paper feels limited. The main contributions can be summarized in two parts: (1) an analysis of the influence of pruning data, and (2) the proposed OASIS method. However, the analysis largely revisits well-established findings—such as the impact of data quality and quantity—which have been studied in prior work. As for the method, it essentially builds on existing soft pruning frameworks, with a gradient-based weight for data selection. These contributions, in my view, are not substantial enough to warrant publication at ICLR.
- The experimental section is also quite limited, particularly in terms of baseline coverage. The authors should include more direct comparisons related to the data selection component, as this is the core novelty of the paper. Specifically, comparisons with prior data selection techniques would help clarify the relative effectiveness of the proposed approach.

### Questions
Here are some typos or mistakes I found:
- Line 154: The pruning score for Wanda is not correct.
- I’m curious about the definition of (golden, mediocre, detrimental) data. Maybe I missed something, but I think the author should give a clear definition for the criteria at the very beginning of the paper, since these terms are mentioned many times without any explanation.
- Line 289: There are typos in the parentheses.
- The saliency score is defined in Section 3 as S = |WX^2|; however, in Line 269, the parameter becomes a vector without an explanation or a new definition of the saliency score.
There are many other typos and mistakes in this paper, I highly suggest the author further revise the paper.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The OASIS framework proposes a novel approach to improving post-training pruning of large language models (LLMs) by addressing the problem of calibration data selection. Traditional calibration data selection methods rely on simple heuristics, such as random sampling or entropy, which often result in suboptimal and inconsistent pruning outcomes. The authors point out that this inconsistency arises because the importance of calibration samples varies and is context-dependent (i.e., it depends on the specific model and pruning method). A key feature of OASIS is its end-to-end framework, which formulates calibration data selection as an optimization problem and solves it using a differentiable soft-mask proxy. This allows task-level gradients to be backpropagated to the calibration data, dynamically discovering the subset most beneficial for pruning. Experiments show that OASIS improves the performance of various state-of-the-art pruning methods, establishing a new standard for data-aware model compression.

### Strengths
1. Context-aware calibration: The adaptive selection of calibration data allows pruning results to be optimized based on the specific model and pruning algorithm, providing high specificity.
2. Improved pruning performance: Compared with traditional heuristic methods, OASIS offers more consistent pruning outcomes and can reduce variance in pruning results.
3. Wide applicability: The method is compatible with various pruning techniques, making it practical and suitable for different types of model compression.

### Weaknesses
1. Poor figure readability: The legends and chart sizes in the paper are relatively small. Although the figures are information-dense, it is difficult to extract clear conclusions, which affects readers’ intuitive understanding of the experimental results.
2. Limited improvement for low-accuracy models: When the base model has low accuracy, OASIS provides only minimal gains in perplexity and downstream task performance. For example, for Llama-3.1-8B, the average accuracy is 79.36, which drops to 52.50 after pruning. With OASIS, it only increases to 52.97, indicating that the method does not significantly improve low-accuracy models and does not bring substantial performance breakthroughs.
3. Unclear iterative process and high time cost: OASIS relies on iterative optimization to dynamically select the optimal calibration data subset, but the paper does not specify the number of iterations needed or the computational cost per iteration. This may require significant computational resources and long training time in practical applications, limiting the feasibility of the method.
4. Generality issue: Experiments are conducted only for a 50% pruning rate and models under 8B parameters. It remains unclear how the method performs at higher pruning rates or on larger models, limiting the assessment of its general applicability.

### Questions
1. How many iterations are required for the optimization problem to converge, and what is the computational cost per iteration?
2. Does the method still work effectively at lower pruning rates?
3. Can the method be applied to large models, such as Llama-65B, and is the computational cost still acceptable?

### Soundness
3

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
The paper identifies that pruning large language models is highly sensitive to the calibration data used, and that existing heuristic-based methods often lead to inconsistent and suboptimal results due to data quality variance. To address this, it proposes OASIS, a fully differentiable framework that optimizes calibration data selection end-to-end by backpropagating task-level gradients through a soft-mask proxy, allowing the model to learn which samples most improve post-pruning performance. Experiments on structured and unstructured pruning across Llama and Qwen models show that OASIS consistently outperforms heuristic and synthetic data baselines, establishing a new standard for data-aware model compression.

### Strengths
1. This paper provides a thorough investigation of the impact of calibration data on pruning from both macro and micro perspectives, offering valuable insights for future research in this area.

2. The experiments are solid and comprehensive, covering multiple LLMs under both structured and unstructured pruning settings, which strongly support the paper’s conclusions.

3. The writing is well-organized and easy to follow.

### Weaknesses
1. The macro-level conclusions have already been established in prior work, so the novelty in this aspect appears limited.

2. The motivation for introducing noise perturbations into the input is not clearly explained. Although the authors demonstrate its effectiveness through ablation studies, it remains unclear why adding noise would lead to more stable optimization.

3. The paper does not report the time or computational cost of data selection. Excessive overhead could undermine the practical value of the proposed method. If I allocate the same computational cost for data selection to gradient-based iterative pruning or recovery training, would it yield better performance?

### Questions
1. Why would adding noise lead to more stable optimization?

2. If I allocate the same computational cost for data selection to gradient-based iterative pruning or recovery training, would it yield better performance?

### Soundness
4

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a data selection method for LLM pruning. The authors first investigate how different data selection strategies affect pruning performance from both macro and micro perspectives, and find that "heuristics fail in calibration data selection." They then propose an algorithm called OASIS to select datasets for pruning tasks. Experiments demonstrate that OASIS outperforms other data selection approaches and is suitable for both structured and unstructured pruning.

### Strengths
1. A comprehensive study on how various calibrated data selection strategies affect model pruning performance, encompassing both structured and unstructured pruning methods.

2. The proposed algorithm is straightforward and easy to implement, while consistently improving upon the performance of baseline methods in experiments.

### Weaknesses
1. The findings are sound but not very surprising. First, it is apparent that performance saturates as data size increases, while data diversity significantly impacts model performance including in model pruning, and the optimal data composition varies across different tasks. Additionally, the statement "A single low-quality ('detrimental') sample can contaminate the entire set and severely degrade the performance of a high-quality ('golden') set" is somewhat confusing. What exactly is the size of the "entire set"? For example, if we have a selected calibrated set chosen by OASIS and introduce just one low-quality sample, will the performance indeed degrade severely?

2. The perturbation of embeddings requires further justification. Specifically, it is unclear why such perturbation ensures stability. Moreover, the ablation studies do not report the final model performance without perturbation.

### Questions
1. It would be helpful to include a small experiment showing how performance degrades when a single detrimental sample is added to a high-quality calibrated set of realistic size.

2. Ablation study should report final model performance without perturbation to better illustrate its contribution.

3. It appears that the reported code site has no content.

### Soundness
2

### Presentation
3

### Contribution
2
