# Towards Understanding the Feasibility of Machine Unlearning

- Decision: Reject
- Scores: 6, 5, 5, 5

## Abstract
In response to recent privacy protection regulations, machine unlearning has attracted great interest in the research community. However, existing studies often demonstrate their approaches' effectiveness by measuring the overall unlearning success rate rather than evaluating the chance of unlearning specific training samples, leaving the universal feasibility of the unlearning operation unexplored. This paper proposes a novel method to quantify the difficulty of unlearning a single sample by taking into account factors such as model and data distribution. Specifically, we propose several heuristics to understand the condition of a successful unlearning operation on data points, explore difference in unlearning difficulty over training data points, and suggest a potential ranking mechanism for identifying the most challenging samples to unlearn.  In particular, we note Kernelized Stein Discrepancy (KSD), a parameterized kernel function tailored to each model and dataset, is an effective heuristic to tell the difficulty of unlearning a data sample. We demonstrate our discovery by including multiple classification tasks and existing machine unlearning algorithms, highlighting the practical feasibility of unlearning operations across different scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper tackles the problem of assessing the difficulty of unlearning individual training samples in machine learning models, a need highlighted by recent privacy regulations. While most existing unlearning methods focus on overall unlearning success rates, this work shifts attention to the unique challenges of unlearning specific samples, considering factors like the underlying model and data characteristics. The authors propose heuristics to predict the success of unlearning operations for individual data points and explore variations in unlearning difficulty across samples, with a ranking mechanism to identify samples that are more resistant to unlearning. A key contribution is the use of Kernelized Stein Discrepancy (KSD) as a model- and data-specific heuristic for gauging unlearning difficulty. The method’s effectiveness is demonstrated on multiple classification tasks, showcasing its applicability across diverse scenarios and highlighting its potential to refine the measurement of unlearning success at a granular level.

### Strengths
* This work introduces an original and timely contribution to the field of unlearning by tackling a previously overlooked question: the unlearnability of specific samples.
* The use of Kernelized Stein Discrepancy (KSD) in this context is both innovative and technically sound. The KSD-based unlearnability score, which incorporates model and data characteristics, is compelling.

### Weaknesses
* **Regulatory Implications.** It is not clear that the developed tools are useful for advancing unlearning techniques to comply with regulations. The authors claim that "With the proposed evaluation metrics, one may reduce unnecessary machine unlearning operations when data points are determined to be infeasible to unlearn.”, but this is not convincing since erasure is mandatory in any case, and it does not seem reasonable to decide to retrain a model from scratch because a heuristic score ranking method indicates that a single sample may be hard to unlearn. More discussion on the regulatory utility or limitations of unlearnability scores would strengthen this point.
* **Rigorous Unlearning Objective.** The unlearning objective presented in Section 2.1 is based on heuristics, such as maximizing the loss on the forget set, which does not guarantee that an adversary or auditor could not detect the presence of the forget data in the unlearned model. A more rigorous definition of unlearning – one that establishes a statistical similarity to retraining from scratch – would better support the authors’ contributions and align their methodology with recent work in statistically grounded unlearning, e.g., see (Guo et al. 2020).
* **Baseline Comparisons and Additional Techniques.** While the inclusion of KSD is interesting, the paper would benefit from a broader comparison with baselines like influence functions (Koh and Liang, 2017), which are efficient and widely applicable to different architectures. Additionally, incorporating more advanced unlearning techniques or defenses against membership inference attacks (Carlini et al. 2022) would strengthen the empirical evaluation, as only three unlearning algorithms are tested here, limiting the generalizability of results.

### References

Koh and Liang (ICML 2017). Understanding Black-box Predictions via Influence Functions.

Guo et al. (ICML 2020). Certified data removal from machine learning models.

Carlini et al. (S&P 2022). Membership Inference Attacks From First Principles.

### Questions
* **Impact of Unlearnability Scores:** Can the authors elaborate on practical applications of unlearnability scores? For example, could these scores help in refining existing unlearning methods to improve handling of difficult samples, or are there contexts in which they could aid in privacy-preserving model design?
* **Empirical Limitations:** What criteria were used to select the three unlearning techniques in the empirical evaluation? Could the authors comment on the generalizability of their methodology to other unlearning frameworks and provide insights on adapting it to handle more complex attack models?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper introduces a set of KSD-based metrics for quantifying forget difficulty, taking into account the characteristics of the target model and data distribution. It introduces a KSD-based heuristic approach to assess forget difficulty, where KSD is a parameterized kernel function tailored for each model and dataset. These metrics hold significant practical value in supporting decision-making processes.

### Strengths
1. The KSD-based metrics presented in the paper are particularly intriguing, as they offer valuable insights into the relationship between data and models in the machine unlearning field.
2. The paper is easy to follow. The authors have effectively communicated their ideas, making complex topics accessible and engaging for the audience.
3. Understanding which samples are more difficult to unlearn has the potential to aid the development of machine unlearning.

### Weaknesses
1. Table 1 shows many counterintuitive numerical results, such as the basic baseline GradAsct achieving 0% accuracy on the forget set while maintaining 99% accuracy on the test set. Even when the authors' metric indicates that the most difficult-to-unlearn samples to forget in SVNH can also achieve 0% accuracy on the forget set, the accuracy on the test set is mostly around 80%. This result is incredibly hard to believe, especially since the current state-of-the-art GradAsct (enhanced GradAsct baseline: NegGrad+ proposed by [A]) cannot achieve such results.

2. The authors claim that the metric proposed in the paper does not rely on a specific unlearning algorithm, making it unreasonable to only select the simplest baseline finetune and GradAsct for the experiments. This suggests that the metric may only be effective for finetune and GradAsct. Considering the existence of different methods such as teacher-student methodology [A], weight saliency [B], knowledge distillation [C], Fisher [D], and Newton Update [E], simple finetune and GradAsct cannot adequately represent these methods. As a primary contribution of proposing some metrics, the authors should select a representative method from various heuristic unlearning works to verify that the metric does not depend on any specific unlearning algorithm. Only when the phenomenon observed in the authors' metric consistently exists across these different methods can it be concluded that the metric does not rely on a specific unlearning algorithm. 

3. The author's citations can be quite misleading in several instances. Such as, in lines 98-99, the authors mention: "Gradient Ascent methods (Thudi et al., 2022; Graves et al., 2021.), adjust the model’s weights in the direction of the gradient to increase the model’s error on the data intended for forgetting." However, it's difficult to classify the methods of Thudi et al. (2022) and Graves et al. as ascent, since ascent implies the need to compute the negative gradient, as in [J], rather than merely adjusting the model.
    In line 112, they state, "Guo et al. (2020) [E] introduced the concept of certified unlearning, grounded in information theory and specifically tailored to the Fisher Information Matrix." However, to my knowledge, Guo et al. (2020) do not mention anything related to the Fisher Information Matrix. If the authors intended to reference the Fisher unlearning method, I suspect they meant to cite [D]. Alternatively, if they intended to reference the use of information theory and the Fisher metric to evaluate unlearning methods, I would guess they meant [I].
   
4. The authors lack descriptions of some baseline settings and the choice of evaluation metrics. Please refer to my question for specifics.



[A] Towards Unbounded Machine Unlearning. Meghdad Kurmanji, et al. NeurIPS 2023.

[B] SalUn: Empowering Machine Unlearning via Gradient-Based Weight Saliency in Both Image Classification and Generation. Fan, Chongyu, et al. ICLR, 2024.

[C] Can bad teaching induce forgetting? unlearning in deep networks using an incompetent teacher. Vikram S Chundawat et al. AAAI 2023.

[D] Eternal sunshine of the spotless net: Selective forgetting in deep networks. Golatkar et al. CVPR, 2020.

[E] Certified data removal from machine learning models. Chuan Guo, et al. ICML, 2020.

[H] Model Sparsity Can Simplify Machine Unlearning. Jinghan Jia et al. NeurIPS 2023.

[I] Evaluating Machine Unlearning via Epistemic Uncertainty. Alexander Becker et al. ECML 2021.

[J] Machine Unlearning of Pre-trained Large Language Models. Jin Yao et al. ACL 2024.

### Questions
1. The authors are suggested to explain the mentioned numerical results. 

2. Can the proposed metric be applied to [A]-[E]? It can be more convincing if the authors show these in experiments.

3. What is the expression for 'MIA-efficacy'? It would be best to explain what 'MIA-efficacy' is, either in the main text or in the appendix, rather than just directing the reader to a specific paper, as this is not a common MIA evaluation metric (e.g., AUC, attack success rate).

4. Which references did the authors use for the evaluation of GradAsc**, **FineTune, and Fisher?  To avoid confusion, the authors should clarify in line 112 whether these methods are taken from other papers or are their own designs. 

5. What is the overfit_threshold in line 670? The authors should ideally provide a brief description of these baselines and their settings, either in the main text or in the appendix.

6. Have the authors tried any NLP-related tasks? I'm particularly curious about the difficulty of forgetting data in NLP compared to CV tasks.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper considers the problem of determinining the feasibility of machine unlearning. This is done by (a) determining which are the easiest and hardest samples to unlearn based on metrics related to kernel Stein discrepancy (b) unlearning these samples using different unlearning algorithms and (c) see how this impacts the accuracy on the data to be forgotten as well as the test set. 

Overall the experiments are solid, but what I found lacking from the paper is discussion and understanding of the significance of the results.

### Strengths
Machine unlearning, even though well studied is not well-understood -- mostly because it is usually not well-defined. The problem studied is thus definitely well-motivated.

I feel like what the paper is trying to get at here is "in-distribution" and "out-of-distribution" samples -- in-distribution being those samples that are very close to, or combinations of the rest of the data, while out-of-distribution samples being outliers or others.  In general, one would expect the latter to be easier to unlearn. In addition, it is also unclear why unlearning in-distribution points should lead to lower performance on them -- for example, if we can classify a typical zero accurately from a classifier trained without this zero. A lot of prior work has ignored these subtleties in the definition and practice of unlearning, and this work does attempt to throw light on them.

### Weaknesses
1. The major weakness of the paper is that it does not offer much by way of discussion and conclusion from the experiments. The experiments are presented in form of tables, with a short discussion section about different algorithms and metrics, but at the end we do not learn much about what we learn overall from the exercise, and why we get to see what we see. Adding a proper discussion section that tries to explain the results would significantly improve the paper.

2. It is unclear to me why so many different variants of kernel Stein discrepancy are needed as they appear to needlessly complicate the message. Is it because the different measures emphasize different aspects? What kind of aspects?

3. The paper would be improved by adding references to other work that questioned model-based unlearning -- see, for example, [1] https://www.usenix.org/conference/usenixsecurity22/presentation/thudi

### Questions
See above.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper studies how to estimate the difficulty of effectively unlearning training data points from models. In Section 3, it summarizes six main factors that can impact the effectiveness of machine unlearning, including the size of the unlearning expansion, resistance to membership inference attacks (MIA), distance to the decision boundary, tolerance of performance shift, number of unlearning steps, and the distance of parameter shift. It then groups these factors into two categories: the existence of strong ties among data points and predictive confidence. In Section 4, the paper introduces the notion of kernel strain discrepancy (KSD) and four potential variants to convert KSD into aggregated pairwise kernel values for each data point. In Section 5, it conducts empirical evaluations to observe the following phenomena: i) the relationship between KSD-based scores and the predictive performance of the unlearned model; ii) the effectiveness comparison among the four variants; and iii) the effectiveness comparison of unlearning algorithms against hard-to-unlearn data points.

### Strengths
+++ This paper studies the variations in difficulty among different training data points, offering an interesting and novel perspective in machine unlearning.

++ It introduces KSD-based scores to measure unlearning difficulty.

+ Experiments are conducted on two CNN models and three image datasets to empirically investigate the effectiveness of the KSD-based scores.

### Weaknesses
--- The paper does not provide sufficient discussion on how to incorporate the KSD-based scores into the overall machine unlearning workflow. It also lacks results on the efficiency of computing these scores. Providing both the computational complexity and empirical evaluations of computational efficiency would strengthen the work. Additional discussion would also be valuable. For instance, could a new unlearning-difficulty-aware algorithm be developed to leverage the KSD-based scores for more effective and efficient machine unlearning? Alternatively, if computing the KSD-based scores is comparable in cost to running certain unlearning algorithms, it would be helpful to clarify how these scores could further enhance the machine unlearning process.

-- It is unclear how the KSD-based scores relate to the six difficulty factors. Ideally, the experiments should first verify that these six factors consistently represent the unlearning difficulty of different samples. Currently, however, the experiments lack systematic results on the relationship between unlearning difficulty and the six factors, which makes the effectiveness of these factors unconvincing. Establishing a solid relationship between the six factors and unlearning difficulty—demonstrating that the factors truly correspond to unlearning difficulty—would allow the paper to propose a unified, holistic metric of unlearning difficulty based on these factors. With this unlearning difficulty metric in place, the paper could then systematically validate that the KSD-based scores indeed reflect unlearning difficulty. The current factor-by-factor approach to evaluating unlearning difficulty (with some factors omitted or combined) is insufficient to convincingly verify that the KSD scores truly capture unlearning difficulty.

### Questions
1. Is it possible to provide efficiency results of the KSD-based scores?

2. Is it possible to leverage the KSD-based scores to develop unlearning-difficulty-aware algorithms for more effective machine unlearning?

3. Is it possible to develop a more unified and holistic metric for the unlearning difficulty?

4. Is it possible to provide more systematic empirical results to verify that the KSD-based scores indeed reflect the unlearning difficulty in terms of all six factors?

### Soundness
3

### Presentation
2

### Contribution
3
