# Data-Aware and Scalable Sensitivity Analysis for Decision Tree Ensembles

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 4, 8, 6

## Abstract
Decision tree ensembles are widely used in critical domains, making robustness and sensitivity analysis essential to their trustworthiness. We study the feature sensitivity problem, which asks whether an ensemble is ``sensitive" to a specified subset of features - such as protected attributes- whose manipulation can alter model predictions. Existing approaches often yield examples of sensitivity that lie far from the training distribution, limiting their interpretability and practical value. We propose a data-aware sensitivity framework that constrains the sensitive examples to remain close to the dataset, thereby producing realistic and interpretable evidence of model weaknesses. To this end, we develop novel techniques for data-aware search using a combination of mixed-integer linear programming (MILP) and satisfibility modulo theories (SMT) encodings. Our contributions are fourfold. Firstly, we strengthen the NP-hardness result for sensitivity verification, showing it holds even for trees of depth 1. Secondly, we develop MILP-optimizations that significantly speed up sensitivity verification for single ensembles and for the first time can also handle multiclass tree ensembles. Thirdly we introduce a data-aware framework generating realistic examples near the training distribution. Finally, we conduct an extensive experimental evaluation on large tree ensembles, demonstrating scalability to ensembles with up to 800 trees of depth 8, achieving substantial improvements over the state of the art. This framework provides a practical foundation for analyzing the reliability and fairness of tree-based models in high-stakes applications.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a new method for the sensitivity analysis of decision tree ensembles. 
Given a tree ensemble classifier, it aims to identify two data points that are close but receive different predictions from the model. 
The authors also introduce an additional constraint that encourages two data points to be close to the dataset and propose an efficient algorithm based on mixed-integer linear programming (MILP). 
By experiments with large tree ensemble models, the authors demonstrated that the proposed method achieved significant speedups over baseline methods.

### Strengths
1. This paper tackles an important problem of sensitivity analysis for tree ensemble models. The authors provide a theoretical analysis for the case that has not been addressed in prior work (Ahmad et al., 2025).
1. The experimental results demonstrate the scalability of the proposed method. Figure 3 clearly demonstrates that the proposed method could obtain solutions in a reasonable time compared to the baselines.

### Weaknesses
1. The presentation of Section 5 could be improved entirely. It is difficult to distinguish which terms are decision variables and which are fixed parameters in the proposed MILP formulation, making it hard to implement and reproduce. Adding clear notation definitions and a small illustrative example would help readers understand the formulation more easily and grasp the intuition behind the constraints.
1. While Figure 4 appears to show the L2 distance of each method for each instance, it is difficult to interpret and extract meaningful insights from it. It might be more precise and informative to summarize the results in a table reporting average values or win rates, rather than presenting all instance-level values in a single figure.
1. (Minor) The motivation of “data-awareness” has also been discussed in the literature on counterfactual explanations, which aim to generate realistic perturbations to alter prediction outcomes (e.g., Cui et al., 2015; Kanamori et al., 2020; Parmentier and Vidal, 2021). It would be helpful if the authors could clarify how their notion of data-awareness differs from or extends these prior concepts.

### Questions
1. The proposed method aims to generate pairs of data points $x^{(0)}$ and $x^{(1)}$ that are close to the dataset $\mathcal{D}$. Why not simply use data points already contained in $\mathcal{D}$? If one of the points, say $x^{(0)}$, were fixed to a sample from $\mathcal{D}$, can we formulate the problem as a more tractable one while still achieving the same objective? 
1. Could the authors clarify which part of the analysis in the prior work (Ahmad et al., 2025) makes it impossible to extend or adapt to the cases with depth 1 or 2? A brief explanation would help contextualize the limitation and highlight the novelty of the proposed approach.
1. In the experiments, both the proposed method and Kant rely on an MILP solver to obtain solutions. Which solver did the authors use?

### Soundness
2

### Presentation
1

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
This paper introduces several improvements, both in runtime and example quality, to the framework of Kantchelian et al. (2016) for finding pairs of sensitive points (or two inputs which are close to the dataset, and have very different probability predictions from a tree ensemble). The closeness to the dataset is one improvement, for which two methods are discussed. The first is an approach that measures a point's closeness to the dataset by the product of its marginals, and the second restricts the search to remove "cavities", or boxes in input space with no points. The authors also present several runtime optimizations based on adding additional constraints of consistency across predicate variables and leaf conditions. There are experimental results demonstrating improved performance on several binary and multiclass datasets, and a variety of parameter configurations.

### Strengths
1. The papers organization is clear and logical, and the paper is well situated in existing literature. It is clear from the authors' presentation what is novel, how it is novel, and what research problems each component of the paper are trying to address.

2. The new constraints that speed up optimization are well-reasoned and proven to be correct.

3. The ablation study is convincing that the combination of all methods leads to a greater improvement than any subset.

4. I am not an expert in this area, but I appreciate the effort to go above and beyond to compare to VERITAS by modifying this system to suit the sensitivity problem. 

5. The experiments seem to demonstrate a runtime improvement over existing methods, as well as finding examples closer to the training distribution.

### Weaknesses
1. The benchmarking results are too aggregated and are lacking statistical significance testing or error bars. The results should have been obtained over repeat trials for each configuration, with a corresponding mean and standard deviation. Moreover, it seems that most problem instances evaluated in Figure 3 are on only 4 datasets -- the ones with a lot of combinations of # trees and depths. I think that this work requires 1) statistical significance testing on the results over repeat trials for all results, and 2) disaggregated versions of the results by dataset (perhaps in the appendix; the aggregated results are fine as long as the disaggregated versions are accessible).

2. Figure 4 is really hard to read, and the definition of distance to the dataset is unclear. I assume that the distance to the dataset is defined by the distance to the nearest training point. I am concerned that, under this definition, a counterexample could be close to only a single point far off in a low-density region of space but still appear "close" to the dataset.

3. The use of cavities to restrict the search space is not compelling to me. Even in the provided example, it is odd that we ignore potential counterexample pairs close to the boundary of the green box. Those points are just as "in distribution" as any other. Furthermore, at least by inspection, it seems to me that we could form overlapping cavities over the entire search space, and thereby restrict our search only to exactly the points in the training dataset (for which there can obviously be no counterexample pairs). 

4. The examples of counterexample pairs in the appendix are not informative -- there is no relative scale to understand what different distance magnitudes mean, and there are no feature names to understand the actual relevance of the features involved. The conclusions in the paragraphs in section B in general ("quite far from any possible realistic data point and may not be very helpful", for example) are not well-justified and contextualized in terms of the actual features in the data.

5. The experiment in Figures 6 and 7, which do the ablation and vary the number of features to be searched over, seem to be conducted on different datasets (fewer instances) than the other experiments. These should be described.

### Questions
1. Are any of the approaches you're comparing against designed for multi-threaded systems? Is your approach suited for multi-threaded systems? I am skeptical of a single-core evaluation of MILP-solver based systems, which to my understanding (not my field, so please pardon the question) are designed to be parallelizable.

2. How do you place a limit on cavity constraint creation so that you don't cover the entire input space?

3. What is the relationship between this topic and the domain of adversarial example search, where the goal is to find an adversarial example very close (within some small $\varepsilon$) to a specific point in the dataset?

4. Does your approach model binary/categorical and continuous features in the same input space? Does this affect your distance-based methods? Modeling these in the same input space enforces an assumption about the relative "cost" of flipping a binary variable and moving around on a continuous feature. 

5. Very minor point, but the table headers in table 2 in the appendix are confusing. What is %V?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces efficient techniques to evaluate the sensitivity of an ensemble of trees to different subsets of features. The authors propose a specialized MILP formulation to construct counterexample pairs, while insuring that the constructed counterexample is realistic with respect to the data distribution. Empirical analysis shows that, across several datasets and models, this tool provides substantial runtime improvements over existing methods.

### Strengths
- The empirical results for SVIM are quite compelling. SVIM universally improves upon the runtime of prior methods, while producing counterexamples that are designed to be realistic.
- The proposed MILP formulation is interesting and involves several novel components.
- The problem studied here is interesting and well motivated.
- In general, the paper is well written and figures are clear.

### Weaknesses
- While the paper is generally easy to follow, it is quite notation heavy and suffers from some minor inconsistencies. In addition to the several specific comments listed below, I recommend adding a notation table to the appendix to help readers keep up.
    - In EQ 4/5, should $p_{kf}$ be $p_{fk}$?
    - In Eq Gap-Bin and Off-bin, $v_i$ is undefined.
    - In the data aware objective function defined under Utility Function, it seems like some things may be off. Should solution (2) be considered in some way, rather than just solution (1)? Additionally, I don’t think pi_f without an input value is well defined.
    - In the constraint introduced under Computing clause summaries, I believe w is undefined. Additionally, the subscript on the “and” is a bit off — should $j \in |F|$ be either $j \in F$ or $j \in [1, |F|]$?
- The L2 distance may not be an appropriate metric to measure counterexample validity because it may be thrown off if one entry in the original, unperturbed sample is extreme. Maybe an L_infininty distance would be more appropriate? This does depend on how features were pre-processed. Were they normalized in some way?

### Questions
- I would recommend adding the 1 feature case to Figure 7 to support a direct comparison.
- In the examples in the appendix, it would be nice to know what each of the reported features represents.
- I would consider this out of scope for the current work, but in the future I would recommend conducting a brief user study to see whether the counterexamples generated by this method are generally considered more realistic.

### Soundness
4

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
2

### Summary
The paper studies the feature sensitivity problem in decision tree ensembles. The authors proposed a data-aware sensitivity framework that builds on the combination of MILP and SMT encoding to verify whether prediction changes under feature perturbations. They developed a method SVIM and demonstrated the effectiveness of their methods compared to prior work on empirical data sets.

### Strengths
1. The paper shows sensitivity verification is NP-hard even for depth-1 trees.
2. The authors implemented their method SVIM and outperformed prior baselines such as SENSPB and KANT.
3. They extended the method beyond binary classification to multi-class problems.
4. The paper is clear in definitions and theoretically well-justified.

### Weaknesses
1. Whether the “data-aware” counterexamples lead to practical improvements in model fairness or robustness is not well explored, and remains an interesting practical direction.
2. The independence assumption is mitigated through restricting space, yet it still introduces a theoretical gap between the assumed and real data distribution.

Minor formatting: “Figures” vs “Fig”. (Line 456)

### Questions
1. Could this method be generalized for regression problems?
2. Will the “data-aware” counterexample be used for model retraining, will there be model improvements observed?

### Soundness
3

### Presentation
3

### Contribution
3
