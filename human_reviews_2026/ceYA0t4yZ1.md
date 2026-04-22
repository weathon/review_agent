# Bringing Light to the Threshold: Identification of Multi-Score Regression Discontinuity Effects with Application to LED Manufacturing

- Avg Score: 5.33
- Decision: Reject
- Scores: 4, 4, 8

## Abstract
The regression discontinuity design (RDD) is a widely used framework for threshold-based causal effect estimation in causal inference. Recent extensions incorporating machine learning (ML) adjustments have made RDD an appealing approach for researchers utilizing causal ML toolkits. However, many real-world applications, such as production systems, involve multiple decision criteria and logically connected thresholds, necessitating more sophisticated identification strategies, which are not clearly addressed in the recent literature. We derive a novel identification result for the complier effect in the multi-score RDD (MRD) setting by extending unit behavior types to multiple dimensions. Further, we show that under mild assumptions, this identification result does not depend on subsets of units with constant response. We apply our findings to simulated and real-world data from opto-electronic semiconductor manufacturing, employing estimators that adjust for covariates through machine learning. Our results offer insights into enhancing current production policies by optimizing the cutoff points, demonstrating the applicability of MRD in a manufacturing context.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper extends the classic Regression Discontinuity Design (RDD) framework to multi-score RDD (MRD), where treatment decisions are based on multiple thresholds. This extension is particularly useful in operational contexts like manufacturing, where decisions are made based on more complex rules rather than a single score. The paper presents a novel identification result for the complier effect in multi-dimensional cutoff settings. Through unit categorization(compliers, defiers, always takers, never takers, and indecisive units), the authors develop a rigorous framework for analyzing treatment effects in such contexts. The theoretical contributions are complemented by an empirical applicationto LED production data, demonstrating the practical utility of MRD in optimizing production policies and decision-making.

### Strengths
The proposed identification result for the complier effect in MRD is valuable, especially for complex decision-making systems like those found in industrial operations. By expanding the RDD framework to handle multi-dimensional decision rules, the paper contributes to making causal inference applicable to more realistic settings.

Unit categorization (compliers, defiers, always takers, never takers) is a useful framework for understanding how different units respond to treatment in multi-score settings. The inclusion of indecisive units is an important and novel contribution to the literature.

The empirical application to LED production shows how the MRD framework can be applied in a real-world manufacturing context. The results provide insights into how production policies could be adjusted to improve efficiency, demonstrating the practical applicability of the MRD framework.

### Weaknesses
The exposition of the paper is somewhat unclear, especially given the many definitions, lemmas, and propositions. These technical details are not always well-motivated or explained. A more intuitive approach would be to use the empirical application as a running example throughout the theoretical section, which would make the theoretical results more tangible. Since the empirical data is 2-dimensional, I suggest that the authors limit the theoretical results to the 2-dimensional case and move the more general case to the appendix.

The unit categorization section introduces multiple propositions and a lot of notation to formalize the different types of units (compliers, defiers, etc.). However, this section is not well connected to the empirical relevance of the results. Although Theorem 1 relies on unit categorization, the paper doesn’t clearly explain how these concepts apply to the data or how they influence the interpretation of the results. It might be more efficient to introduce these categories in a more intuitive way, rather than getting bogged down in formalism early on. The reader would benefit from a clearer link between the theoretical framework and the empirical application.

The discussion of assumptions related to the continuity and unit categorization in MRD is valuable but would benefit from further elaboration on how these assumptions might hold or fail in different empirical settings, particularly for industrial applications.

While the empirical section demonstrates the applicability of the framework, additional robustness checks on the sensitivity of results to changes in thresholds or decision rules would provide stronger evidence for the framework's validity in production contexts.

### Questions
Have you considered including visualizations to help illustrate the unit categorization and the identification results? In the RDD literature, visual aids, such as plots of discontinuities, are a valuable tool for empirical researchers. For example, Figure 4 is a great example of how to present the results visually, which aids in understanding the treatment effect at the cutoff.

While the paper focuses on identification, it would be helpful to provide more details on how researchers can implement the estimation procedure based on the identification results. For instance, how should researchers practically apply the unit categorization framework to estimate treatment effects using the MRD method? Providing a more concrete roadmap for applying the results would improve the paper’s practical value, making the theoretical insights easier to translate into real-world applications.

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
This paper proposes a generalized framework for the multi-score regression discontinuity (MRD) design, extending the classic RDD to scenarios with multiple cutoffs. In particular, it allows for arbitrary combinations of Boolean operations and further formalizes the classification of units and the identification of the compiler effect. Furthermore, the paper conducts empirical studies on semi-synthetic and real-world LED semiconductor manufacturing data.

### Strengths
1. The paper provides a generalized and formalized theoretical framework for MRD. Compared to traditional methods that often consider only special cases, the paper's framework accommodates multidimensional cutoffs under general Boolean-type threshold rules. It also introduces the concept of indecisive units, which do not exist in the single-dimensional RDD setting.

2. The paper conducts an empirical analysis on real-world data from opto-electronic/LED manufacturing, highlighting the practical significance of the MRD framework.

3. The paper systematically tests and compares various ML estimators within its theoretical framework, providing valuable guidance for the practical application of MRD.

### Weaknesses
1. The experiments are conducted only under a binary "AND" rule. Given the paper's core contribution, it would benefit from experimental demonstrations under higher-dimensional and more complex combinations of Boolean rules, even if real-world datasets for such scenarios are unavailable.

2. The experiments lack a comparison with other existing MRD methods (as mentioned in Appendix C). Such a comparison is important for demonstrating the advantages of the proposed method, even in the two-dimensional scenario.

3. The paper's emphasis seems limited to its value in industrial applications. A lack of discussion on other potential MRD scenarios (e.g., in healthcare or economics) might limit its perceived general applicability.

4. The notation used may lack sufficient explanation for readers who are not familiar with the potential outcomes framework.

### Questions
1. Could the authors experimentally demonstrate the estimation performance under higher-dimensional and more complex Boolean rule combinations?

2. Is it possible to provide a comparison with other existing MRD methods within the current two-dimensional setting?

3. Could the concept of indecisive units be discussed in greater depth? This discussion would be particularly valuable in the context of the practical application setting, as well as regarding the impact of violating Assumption 4 (which assumes their absence) on the proposed estimators.

### Soundness
2

### Presentation
2

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
Reasonable extension of RDD with novel real world data application

### Strengths
- This paper stands out due to its honesty about the strength of its assumptions: Line 483 onwards:“Causal identification is not possible without any of these assumptions. The question is whether such an assumption is interpretable enough to justify an approximate conformance in empirical studies. Compared to previously presented formulations, we draw from the intuitive language of unit categories to aid the argument for or against applicability.”
- It has a clear problem statement and provides reasonable extensions of established RDD methods as a possible solution.

### Weaknesses
- See questions

### Questions
- Line 037: can you provide a definition for ‘score’ here already?  Even informal intuition to increase accessibility might be good 
- Line 039 “When correctly specified”, can you briefly summarise what’s required for correct ‘specification’? There is no free lunch, so if the method can identify without assuming uncofundness or positivity, it usually needs to assume something else.
- Line 117: What does ‘credible’ mean here exactly? Is it a formal term in Statistics?
- Line470: You suggest that opposite signs can be “can be explained by the calibration of the semi-synthetic process.” Can you provide additional theories apart from calibration that could yield such a result?

### Soundness
4

### Presentation
4

### Contribution
4
