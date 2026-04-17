# AtC: Aggregate-then-Calibrate for Human-centered Assessment

- Decision: Accept (Poster)
- Scores: 6, 8, 8, 4

## Abstract
Human-centered assessment tasks, which are essential for systematic decision-making, rely heavily on human judgment and typically lack verifiable ground truth. Existing approaches face a dilemma: methods using only human judgments suffer from heterogeneous expertise and inconsistent rating scales, while methods using only model-generated scores must learn from imperfect proxies or incomplete features. We propose Aggregate-then-Calibrate (AtC), a two-stage framework that combines these complementary sources. Stage-1 aggregates heterogeneous comparative judgments into a consensus ranking $\hat{\pi}$ using a rank-aggregation model that accounts for annotator reliability. Stage-2 calibrates any predictive model’s scores by an isotonic projection onto the order $\hat{\pi}$, enforcing ordinal consistency while preserving as much of the model’s quantitative information as possible. Theoretically, we show: (1) modeling annotator heterogeneity yields strictly more efficient consensus estimation than homogeneity; (2) isotonic calibration enjoys risk bounds even when the consensus ranking is misspecified; and (3) AtC asymptotically outperforms model-only assessment. Across semi-synthetic and real-world datasets, AtC consistently improves accuracy and robustness over human-only or model-only assessments. Our results bridge judgment aggregation with model-free calibration, providing a principled recipe for human-centered assessment when ground truth is costly, scarce, or unverifiable.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes a two-stage framework, Aggregate-then-Calibrate (AtC), for human-centered assessment tasks that lack verifiable ground truth. The authors provide theoretical results for supporting the modeling-selection, robustness, optimality.  Empirical evaluation on semi-synthetic and real datasets suggests that AtC improves robustness and accuracy over human-only or model-only baselines.

### Strengths
1. The paper focuses on an important research question—how to calibrate model's score for human-centered assessment tasks. 
2. The interdisciplinary considerations are insightful, for example, the design of HTM under the consideration of the Weber–Fechner laws. 
3. The theoretical analysis is sound under the modeling assumptions, and the framework performs consistently across multiple datasets, demonstrating its sound performance.

### Weaknesses
1. The paper lacks discussion of prior calibration works. The authors should better clarify how AtC differs from existing human-centered calibration approaches. 
2. The methods' effectiveness depends on several strict assumptions, which may limit its general applicability.
3. Some key design choices in this paper lack validation through real human studies — for example, whether the HTM-inferred “consensus ranking” truly corresponds to actual human agreement. Moreover, it remains unclear whether the proposed AtC framework ultimately enhances decision-making performance in real-world applications.
4. The score-level performance comparison (Table 1) is conducted only on semi-synthetic experiments, which limits its generalizability. Moreover, the calibration baselines  include only human-only, model-only, and AtC variants with different stage-1 aggregation methods, which do not constitute rigorous baselines. More appropriate comparisons would involve other human-centered score calibration approaches, rather than what appear to be ablation-style baselines.

### Questions
The isotonic projection enforces the model’s scores to align with the human consensus ranking, implicitly assuming that this consensus ranking is correct. However, the model’s role is also to correct systematic errors in human ranking outcomes. How do you view the balance between improving objective correctness and maintaining adherence to human-centered consensus in the calibration process?

I especially feel that real human studies are important in this research area.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
For certain tasks, obtaining the ground-truth is costly, unobservable, or only gets clear in the future. This is the case for tasks such as estimating paper quality (only find out in the future after finding out its impact) or worker workload (though workers are asked to use the same scale, how the scale is applied is up to the individual). 

This paper tackles the limitations of current practices that address such tasks by proposing the Aggregate-then-Calibrate framework (AtC). Aggregating over individual judgments that lead to proxy labels for the actual task does not allow the model to learn the actual features that are relevant for correct decision making. AtC combines both human and model judgments through two steps: 
1. Aggregation step: They yield an ordinal consensus ranking of items from multiple annotators, using a heterogeneous rank aggregation approach. 
2. Calibration step: The model’s initial predictions are calibrated using the consensus ranking through isotonic regression.  

The paper first provides theoretical guarantees for the framework: 
1. efficient in consensus w.r.t. annotator heterogeneity than homogeneous
2. robust under model misspecification
3. gives calibrated output that is more accurate than the uncalibrated model’s output 

The authors evaluate the method on two datasets: 
1. semi-synthetic: pairwise comparison reading difficulty level dataset where ground-truth scores are available but the annotators are synthetic
2. real-world: ranking and rating-based dots counting in an image, where the image can be obscured and made noisier

Results show that the framework outperforms baselines, especially when facing more noisy data.

### Strengths
1. The problem definition is novel, clear, and covers two aspects that heavily interact with each other but has (to the best of my knowledge) been rather understudied: unobservability of the ground truth that is estimated through noisy proxy labels, where humans can be inconsistent in their judgments. The latter adds another layer of complexity, which makes it an important task type to look at.  
2. The authors provide theoretical guarantees for the framework, which then highlights the strength of the setup. 
3. I also like the evaluation setup which is clearly mapped out with the 6 research questions. This helps the reader in getting a direct overview of what exactly will be evaluated and why. In addition, this helps in understanding the strengths of the AtC framework. The experiments also cover both a (semi-)synthetic setting and real-world setting.

### Weaknesses
1. I feel as though subjectivity can be a discussion point: I particularly view the inconsistency in how people use these rating scales as something subjectivee (e.g., people experience workloads differently) and the dots dataset also suffers from input ambiguity which leads to subjectivity. 
2. A high-level figure of the entire pipeline would be helpful, Figure 1 is nice but the paper would really benefit from an extra figure that shows an example input and then how AtC leads to a better solution. This is especially helpful since there are so many intermediate steps. 
3. Though the paper is nicely structured, I feel the reader can be taken a bit more by the hand, since the paper covers a lot of steps. I think an extra figure already helps with this. E.g., choice of evaluation metrics could be motivated a bit more, though I understand why, I think the reader can be taken a bit more by the hand here. 
4. The conclusion and discussion can benefit from a clearer takeaway: in which scenarios would you suggest that a researcher or practitioner uses AtC? 
5. Related Work is in the Appendix, which makes it difficult for the reader to adequately place the paper along existing papers. Some decision choices that I questioned before were only cleared up once I saw the Related Work in the Appendix. I understand that there are space constraints but it would be very helpful if at least the important papers are in the main body of the paper.

### Questions
Did you also control or verify whether the added noise does not to an extent actually change the ground-truth? I think as you add more obscurities, the actual number of dots could change. I would love to hear your thoughts on this or if I misunderstood any detail.

### Soundness
3

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
2

### Summary
This work introduces a notion of calibration and a technique to calibrate predicted model scores to fare well along this notion. 
The paper is concerned with settings where human judgments is assumed to be noisy, but where there's assumed value in the aggregated information they provide about relative merit of alternative options.

The paper fits a statistical model for the aggregation of heterogeneous data, and this provides us with a permutation of the judged items (say, in ascending order of merit). The calibration technique involves projecting predicted model scores to the manifold of score vectors whose coordinates are sorted in compliance with the ranking induced by aggregation of human ratings. The paper proposes to use l2 projection. 

The paper establishes a number of theoretical results and tests the proposed approach in meaningful settings, shedding light onto the significance of the theoretical results. 

I find the paper rather clear, and a very interesting read (admittedly, however, I am not knowledgeable of the specific literature it talks to, so my review is that of an educated but non-expert reader),

### Strengths
1. The technique is well-motivated and rather simple
2. The paper analysis the technique on theoretical and empirical grounds, the experiments cover all claims (incl. analysis of the theoretical claims)
3. The paper is well-written and I found it rather clear

### Weaknesses
I don't see any major weaknesses, but I have a minor question (for the other box).

### Questions
What is the significance of the Euclidean projection? Or, asking differently, what happens if we project differently? Are there any interesting results (theoretical or of computational efficiency) that depend/vary as a function of this decision?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes an Aggregate-then-Calibrate (AtC) framework to estimate latent true scores when only noisy human judgments and biased model scores are available. The paper provides theoretical results, including an optimality guarantee showing that integrating human ordinal judgments with model-based scores improves accuracy. Experiments on two datasets show that AtC consistently outperforms human-only and model-only baselines.

### Strengths
- The paper is generally well written and easy to follow.

- AtC is a lightweight approach with theoretical guarantees, improving model scores by leveraging human labels.

### Weaknesses
- [Q1] The motivation is somewhat unclear. The adjusted score $\hat s$ preserves the same order as the estimated consensus score vector $s^∗$; that is, $\hat \pi$ is consistent between Stage-1 and Stage-2. This makes the need to compute $\hat s$ questionable. Please provide concrete use cases where the adjusted score is required while the estimated consensus score alone is insufficient.

- There are also some issues regarding the experiment.
  - [Q2] The Kendall $\tau$ is a rank correlation. If values in $s^*$ and $\hat s$ induce the same order, they should yield the same correlation wrt $s$. Am I missing something about how \tau  is computed here?
  - [Q3] In Figure 2, the $s^∗$ distribution appears standardized to zero mean and unit variance, which can inflate the distance between $s^∗$ and $s$. Would using a different scaling (for example, matching mean and variance via a simple estimate) make the distributions more comparable?
  - [Q4] As in Q1, across both datasets it remains unclear when the adjusted score is practically useful beyond the consensus outputs. Examples would help.

### Questions
See weaknesses

### Soundness
2

### Presentation
3

### Contribution
2
