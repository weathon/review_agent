# Quantifying the Accuracy-Interpretability Trade-Off in Concept-Based Sidechannel Models

- Avg Score: 3.33
- Decision: Reject
- Scores: 4, 4, 2

## Abstract
Concept Bottleneck Models (CBNMs) are deep learning models that provide interpretability by enforcing a bottleneck layer where predictions are based exclusively on human-understandable concepts. However, this constraint also restricts information flow and often results in reduced predictive accuracy. Concept Sidechannel Models (CSMs) address this limitation by introducing a sidechannel that bypasses the bottleneck and carry additional task-relevant information. While this improves accuracy, it simultaneously compromises interpretability, as predictions may rely on uninterpretable representations transmitted through sidechannels. Currently, there exists no principled technique to control this fundamental trade-off. In this paper, we close this gap. First, we present a unified probabilistic concept sidechannel meta-model that subsumes existing CSMs as special cases. Building on this framework, we introduce the Sidechannel Independence Score (SIS), a metric that quantifies a CSM’s reliance on its sidechannel by contrasting predictions made with and without sidechannel information. We further analyze how the expressivity of the predictor and the reliance of sidechannel jointly shape interpretability, revealing inherent trade-offs across different CSM architectures. Finally, we propose SIS regularization, which explicitly penalizes sidechannel reliance to improve interpretability. Empirical results show that state-of-the-art CSMs, when trained solely for accuracy, exhibit low representation interpretability, and that SIS regularization substantially improves their interpretability, intervenability, and the quality of learned interpretable task predictors. Our work provides both theoretical and practical tools for developing CSMs that balance accuracy and interpretability in a principled manner.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a framework for quantifying and controlling the interpretability–accuracy trade-off in Concept Sidechannel Models (CSMs). The authors introduce the Sidechannel Independence Score (SIS), a metric designed to evaluate how strongly a CSM relies on its sidechannel, and propose SIS regularization, a training scheme that explicitly penalizes such reliance to enhance interpretability. Building on a unified probabilistic meta-model that encompasses existing CSM variants, the work provides a deeper understanding of how sidechannel dependence and model expressivity jointly influence interpretability.

### Strengths
- The paper is well-structured and presents a comprehensive probabilistic framework that unifies various CBM variants in a clear and accessible manner.
- Its motivation is both timely and significant, tackling the crucial issue of uncertainty quantification in deep models, which remains central to the pursuit of reliable and trustworthy AI.

### Weaknesses
- While the definitions effectively cover multiple cases, the proposed metric lacks sufficient theoretical guarantees and appears somewhat limited in novelty.
- The empirical evaluation is rather limited, as the experiments are conducted on a narrow set of datasets.

### Questions
1. In Figure 4(b), it is unclear why applying stronger regularization does not lead to better performance when the number of interventions is below 10. This trend is also evident when comparing regularization weights of 0.01 and 0.0, where performance actually degrades. It would be helpful to clarify whether this effect is consistent across different CBM variants. Similarly, in Figure 9(b) of the appendix, the case with $w = 0.1$ shows worse results under a low number of interventions, raising questions about the robustness of the proposed approach.

2. In relation to this, the evaluation appears to be based on a rather small set of datasets. Extending the experiments to well-established CBM benchmarks such as CUB or AWA2 could help verify the robustness of the method.

3. If I understand correctly, the CSM model introduces a side channel to enhance task accuracy through reasoning steps that are not easily captured by concepts alone, thereby sacrificing some interpretability. In other words, it seems to trade interpretability for higher accuracy. Given this, I am not sure why one would intentionally reduce accuracy to regain interpretability in such a model — wouldn’t it be more straightforward to simply use a standard CBM instead? This seems especially relevant since there appears to be an inherent accuracy–interpretability trade-off, even when using SIS regularization.

4. In connection with the previous point, could you clarify what benefits are gained from applying SIS regularization?

5. In line 240, “use” should be corrected to “us”.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper is a follow-up research on concept bottleneck models. First, it unified CSMs and divided the interpretability into functional and representation interpretability. Based on this, this paper proposes a metric, SIS, that measures interpretability and can be used as a training objective. The paper conducts numerous experiments to demonstrate that SIS is highly useful and thoroughly discusses its interpretability.

### Strengths
- Though I give a negative rating, I really appreciate the unification of interpretability using functional and representation interpretability.
- The discussion on interpretability is insightful and reasonable.

### Weaknesses
- My primary concern is that the main contributions in this paper highly overlap with the existing work DCBM [1], which is not referenced. Specifically, they also design a residual branch and theoretically prove the trade-off between concept and label accuracy. More importantly, they propose a new metric, the concept contribution score (CCS), which is similar to SIS at a high level. 
- Apart from the unification of interpretability and the interpretability discussion, the technique itself appears to be weak, which only introduces a metric and trains it as a regularization.
- It seems that functional and representation interpretability cannot cover all situations. For example, the joint concept bottleneck model meets both functional and representation interpretability; however, it still suffers from information leakage due to the continuous values in the representation. Do you consider this case?
- A minor concern is that the writing needs to be improved. For example, the contents in Section 3.1 could be organized as mathematical assumptions and definitions rather than plain text. The methodology section (Section 3) and the experiment section (Section 5.2) are divided into too many subsections or subsubsections. Though it demonstrates a high workload, it might fail to focus on the most essential parts.
- This is not a weakness, but the abbreviations in this paper are different from those in most existing works. For example, CBM typically represents the concept bottleneck model in the literature, but it is CBNM in this paper. Instead, the authors denote CBM as concept-based models. This might be confusing for researchers familiar with Concept bottleneck models, though these abbreviations are clearly defined at the beginning.

[1] The Decoupling Concept Bottleneck Model.

### Questions
See weaknesses.

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper addresses representation interpretability in Concept Sidechannel Models (CSMs) by proposing a unified probabilistic framework, introducing the Sidechannel Independence Score (SIS), and demonstrating how SIS regularization controls the accuracy-interpretability trade-off. Authors also analyze how the expressivity of the predictor and the reliance of the side channel jointly shape interpretability, revealing inherent trade-offs across different CSM architectures.

### Strengths
Strengths:

1.  The authors clearly articulate that CSMs can rely heavily on side channels (even when unnecessary)—this is a valuable empirical finding.

2.  The distinction between representation interpretability and functional interpretability is well-articulated and addresses a genuine gap in the literature. This separation clarifies previously muddled discussions about what makes concept-based models interpretable.

### Weaknesses
Weaknesses:

1. The paper defines representation interpretability as binary. Predictions must be "derived exclusively from interpretable units" (lines 146-149). Yet treats it as continuous, claiming CSMs are "partially representation interpretable" (lines 152-153). This contradiction undermines the value proposition: at SIS=100%, CSMs reduce to expensive CBNMs; at SIS<100%, some predictions are completely uninterpretable by the paper's own definition. The paper does not justify why practitioners would prefer models with unpredictable interpretability over either fully interpretable CBNMs (accepting accuracy loss) or black-box models (abandoning interpretability claims entirely).

2. Limited Scope of experiments: The paper considers only vision datasets and lacks a comprehensive evaluation. 

3. The paper does not include a discussion on how the different CSM models differ from one another (Lines 366–371). Providing a more detailed explanation of these models would help readers better understand and interpret the results.

4. The paper does not include a clear definition of intervenability, which reduces the overall readability and clarity of the work.

5. The paper lacks the discussion on computational overhead of the SIS regularization.

### Questions
Questions to Authors:

1. Binary vs. Continuous Interpretability Contradiction: Your definition explicitly states predictions must be "derived exclusively from interpretable units" to be representation interpretable (lines 151-153), which is a binary property. However, you then claim CSMs are "partially representation interpretable" based on SIS scores. Can you clarify, doesn't any SIS<100% make the model unreliable for interpretability requirements for safety-critical applications?

2. For practitioners prioritizing interpretability, what is the concrete advantage of a CSM with SIS=60-80% over the two obvious alternatives, CBNMs with SIS=100% (fully interpretable)  or black box models (no interpretability, post-hoc explanations)?

3. How does SIS regularization affect the optimization landscape?

### Soundness
2

### Presentation
3

### Contribution
2
