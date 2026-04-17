# Fair Decision Utility in Human-AI Collaboration: Interpretable Confidence Adjustment for Humans with Cognitive Disparities

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 8

## Abstract
In AI-assisted decision-making, human decision-makers finalize decisions by taking into account both their human confidence and AI confidence regarding specific outcomes.  In practice, they often exhibit heterogeneous cognitive capacities, causing their confidence to deviate, sometimes significantly, from the actual label likelihood. We theoretically demonstrate that existing AI confidence adjustment objectives, such as *calibration* and *human-alignment*, are insufficient to ensure fair utility across groups of decision-makers with varying cognitive capacities. Such unfairness may raise concerns about social welfare and may erode human trust in AI systems.
To address this issue, we introduce a new concept in AI confidence adjustment: *inter-group-alignment*. By theoretically bounding the utility disparity between human decision-maker groups as a function of  *human-alignment* level and *inter-group-alignment* level, we establish an interpretable fairness-aware objective for AI confidence adjustment. Our analysis suggests that achieving utility fairness in AI-assisted decision-making requires both *human-alignment* and *inter-group-alignment*. Building on these objectives, we propose a multicalibration-based AI confidence adjustment approach tailored to scenarios involving human decision-makers with heterogeneous cognitive capacities. We further provide theoretical justification showing that our method constitutes a sufficient condition for achieving both *human-alignment* and *inter-group-alignment*.
We validate our theoretical findings through extensive experiments on four real-world tasks. The results demonstrate that AI confidence adjusted toward both *human-alignment* and *inter-group-alignment* significantly improves utility fairness across human decision-maker groups, without sacrificing overall utility.
*The implementation code is available at* https://github.com/WEILaboratory/AI-Ethics-Safety-PaperCode/tree/main/Fair_HAI (ICLR2026).

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
In this paper, the authors explore how humans with different characteristics may derive unfair utility from the same advice. They introduce the new concept of inter-group alignment and propose an interpretable, fairness-aware objective for AI confidence adjustment. Their analysis shows that achieving utility fairness in AI-assisted decision making requires both human alignment and inter-group alignment. Building on these objectives, they present a multicalibration-based AI confidence adjustment approach tailored to scenarios involving decision-makers with heterogeneous cognitive capacities. Evaluations on four datasets with real human-behavior data demonstrate that the proposed method significantly improves utility fairness across groups without sacrificing overall utility.

### Strengths
1. The paper is clear, well organised, and adds several fresh ideas to human-AI interaction research. It first defines human-alignment and the new inter-group alignment, proves why both matter, and then shows an elegant multicalibration fix to improve the fairness utility across the group.

2. The experiment design and evaluation is thorough to justify the main claim of this paper.

3. The authors publicly release their code, making replication and future extensions straightforward for other researchers.

### Weaknesses
1. The framework assumes people can report confidence on a common, well-calibrated scale, yet prior work shows self-reported confidence is often noisy and inconsistent across individuals.

2. All experiments rely on archival human-AI datasets; the paper does not test how live users react to the adjusted confidences or how their perceptions and strategies might change.

3. The evaluation covers only binary decision tasks. Extending the theory and method to multi-class settings would broaden its practical reach.

### Questions
See above weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper tackles fairness in human–AI collaboration from the angle of ensuring that AI assistance provides equitable benefits to human decision-makers who possess heterogeneous cognitive capacities. The authors argue that existing alignment notions such as calibration and "human-alignment" are insufficient for guaranteeing fairness *across* decision-maker groups. They introduce the concept of "inter-group-alignment" to capture disparities in decision utility between subpopulations of humans. The paper provides a formal theoretical analysis linking human alignment and inter-group alignment. Building on this insight, the authors propose "cognition-aware multicalibration", and prove that it serves as a sufficient condition for simultaneously achieving both alignment objectives.
The claims are validated through experiments on four human-AI decision-making tasks. The results demonstrate that the proposed method successfully reduces utility disparities across groups while maintaining overall performance.

### Strengths
1. **Novel and Interesting Perspective:** The paper shifts fairness focus from data subjects to the decision-makers themselves. It's an underexplored and insightful perspective as it illuminates how AI systems can inadvertently widen performance gaps between varied decision-makers (e.g., experts and novices).
2. **Elegant Theoretical Framework with Principled Operationalization:** The derivation of a utility disparity bound provides an interpretable bridge between fairness notions, offering practitioners a clear diagnostic for assessing group-level disparities. Cognition-aware multicalibration operationalizes the theory in a computationally practical way.  
3. **Empirical Credibility:** Validation on multiple real-world tasks demonstrates both robustness and practical feasibility. Additional experiments in appendix demonstrate thoughtfulness and rigor.

### Weaknesses
1. **Scalability Challenges with Multi-Attribute Grouping:** The proposed framework requires partitioning the data and calibrating for each subgroup. While the experiments show this is feasible for a small number of groups, the computational and data requirements would grow exponentially with the number of sensitive attributes. A deeper discussion of the practical limits (e.g., the number of subgroups that can be realistically handled) would be beneficial.

2. **Rational Decision-Making and Static Setting:** The theoretical claims hinge on the assumption that humans follow a rational, monotone decision policy (Assumption 2.1). However, a large body of literature shows that human behavior often deviates from pure rationality, exhibiting various cognitive biases. The framework does not currently account for such suboptimal but more realistic decision policies. The analysis assumes that the cognitive capacities and decision policies of the human groups are static. In many real-world collaborations, humans learn and adapt their strategies based on repeated AI interactions. The current framework does not address these dynamic or online contexts, where calibration needs might evolve over time.

### Questions
I found the paper to be insightful and well-executed. The weaknesses section already incorporate most questions, primarily aimed at better understanding the practical implications and boundaries of the proposed framework. One additional question pertains to incorporating broader notions of utility within proposed framework. The paper defines utility as decision accuracy. In human-AI collaboration, however, overall utility can also encompass factors like decision time, cognitive load, or the user's trust and self-efficacy. How do you see the concepts of human-alignment and inter-group-alignment extending to these more multi-faceted, human-centric notions of utility?

### Soundness
4

### Presentation
4

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
The paper demonstrates that existing AI confidence adjustment objectives, such as calibration and human alignment, may lead to utility disparities across groups of decision-makers with varying cognitive capacities.  Moreover, the paper shows that the utility disparity between human decision-maker groups is bounded by a function of human-alignment level and inter-group-alignment level. Hence, the authors propose  a multicalibration-based AI confidence adjustment approach to mitigate this concern, validating their approach over four different datasets.

### Strengths
The main strengths are:

1. The paper covers an overlooked problem in the human-AI alignment literature, as it focuses on fairness in human-AI collaboration
2. The proposed multicalibration approach is principled and based on the proposed theory
3. Overall, the paper's goal is well-motivated and interesting

### Weaknesses
The main shortcomings of the paper are:

*Significance* While I appreciate the theoretical results, I think this new paradigm comes with some implementation choices that are not straightforward. For instance, in real-life scenarios, evaluating cognitive disparities might be even unethical, so such information might never be disclosed. Moreover, it is also quite challenging to assess if improving the fairness for decision-maker groups leads to better outcomes for the observed instances; hence, I think the authors should discuss these possible limitations.

*Assumptions Discussion* I think the authors should discuss Assumption 2.1 better, as understanding when such an assumption might fail is a key factor for the proposed method.

*Empirical Evaluation:* I think some of the results are not very clear. For instance, I would suggest that the authors reformat Table 1, as it does not report standard deviations and incorrectly displays the best entries in bold.

### Questions
I would like the authors to discuss my highlighted shortcomings.

Moreover, I wonder if the authors could detail some settings where $P(Y|S=1)$ is different from $P(Y|S=0)$? My intuition is that, for some reason, the instances that are considered by one group can be sampled from different parts of the feature space, but I want to be sure about it.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper studies fairness in AI-assisted decision-making, focusing on situations where humans have different cognitive capacities. The authors show that existing approaches, such as calibration and human-alignment, do not necessarily ensure utility fairness across different groups of human decision-makers.

They introduce a new concept called inter-group-alignment, which ensures that human groups with similar confidence levels receive comparable decision utility when supported by the AI. The paper provides solid theoretical analysis, deriving a clear upper bound on utility disparity as a function of both human-alignment and inter-group-alignment.

To achieve this dual goal in practice, they propose a Cognition-aware Multicalibration algorithm that adjusts AI confidences accordingly. Experiments on four real-world datasets (Art, Cities, Sarcasm, and Census) confirm that this method significantly reduces fairness gaps between human groups while maintaining overall performance.

The theoretical formulation, clarity of objectives, and validation through experiments make this a valuable contribution to the fairness and human-AI collaboration literature.

### Strengths
The introduction of inter-group-alignment provides a fresh and rigorous way to think about fairness in human-AI collaboration.

The paper is theoretically complete, with well-defined assumptions and proofs that connect intuitively to fairness.

The proposed Cognition-aware Multicalibration method is interpretable, practical, and mathematically justified.

The empirical validation is strong: four datasets, clear fairness and utility metrics, and consistent results showing improvements.

The work bridges human and algorithmic fairness, showing how cognitive differences among people can be formally addressed rather than ignored.

### Weaknesses
Monotonicity Assumption (Assumption 2.1):
This assumption states that humans act rationally, increasing their probability of making a positive decision as either their own or the AI’s confidence increases. While this is reasonable for theoretical proofs, it may not always hold in practice.
Real humans are often non-monotonic in their decision behavior. For instance, they may overtrust or undertrust AI advice due to biases, cognitive fatigue, or misunderstanding of confidence. In such cases, increasing the AI’s confidence may not increase the chance of a positive decision, which violates monotonicity.

The authors could strengthen the paper by (a) discussing how this assumption might break in real-world conditions, (b) analyzing the theoretical impact if monotonicity is only approximately satisfied, and (c) outlining how the proposed method could still remain effective. Even a small simulation with “noisy” or biased human decision policies would illustrate robustness and make the work more realistic and influential.

Ablation Study Clarity:
The paper does include human-only, AI-only, and human-AI results, but these are only mentioned in Figure 2 and not summarized in a table. It would be much clearer to show numerical comparisons between these three conditions across datasets. This would highlight how the collaboration truly benefits fairness and overall utility.

Hyperparameter Sensitivity:
The appendix covers sensitivity analyses for parameters like $\tilde{\alpha}$ and $\lambda$, but the main text should summarize them briefly. Readers need to know how robust the fairness improvements are if these parameters change. A short paragraph or figure in the main paper would make the method easier to trust and reproduce.

Game-Theoretic Perspective:
The problem naturally relates to concepts like Stackelberg games (AI as leader, human as follower) or Shapley-value-based fairness allocation. A short paragraph connecting the proposed framework to these ideas would situate the work more clearly within the broader literature on incentive alignment and cooperative fairness.

Behavioral Impact of AI Adjustments:
Since the AI modifies its confidence outputs to improve fairness, it would be interesting to consider whether this adjustment might influence how humans behave over time. For example, could one group become overly reliant on AI due to consistently higher confidence signals? Even a brief reflection on this would enhance the paper’s real-world applicability.

### Questions
Could you provide a clear quantitative table comparing human-only, AI-only, and human-AI decisions across all datasets to make the ablation more explicit?

How sensitive are the results to the parameters $\tilde{\alpha}$ and $\lambda$? Would adaptive tuning or data-driven adjustment improve generalization?

If the human decision policy is not strictly monotonic (for example, due to inconsistent trust or cognitive biases), how does that affect your theoretical bounds or empirical fairness outcomes?

Could the dual-alignment framework extend naturally to multi-class or regression tasks?

Have you considered a Stackelberg or cooperative-game interpretation of your framework, where the AI strategically adjusts its outputs to optimize a fairness-aware social welfare objective? Are they feasible to perform and related at all?

Do you think modifying AI confidence distributions could unintentionally change how different human groups rely on the AI (either over-trusting or disengaging)?

### Soundness
4

### Presentation
3

### Contribution
4
