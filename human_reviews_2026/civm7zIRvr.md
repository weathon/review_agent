# Semantic F1 Scores: Fair Evaluation Under Fuzzy Class Boundaries

- Decision: Reject
- Scores: 4, 2, 2, 4

## Abstract
We propose Semantic F1 Scores, novel evaluation metrics for subjective or fuzzy multi‑label classification that quantify semantic relatedness between predicted and gold labels. Unlike the conventional F1 metrics that treat semantically related predictions as complete failures, Semantic F1 incorporates a label similarity matrix to compute soft precision-like and recall-like scores, from which the Semantic F1 scores are derived. Unlike existing similarity-based metrics, our novel two-step precision-recall formulation enables the comparison of label sets of arbitrary sizes without discarding labels or forcing matches between dissimilar labels. By granting partial credit for semantically related but nonidentical labels, Semantic F1 better reflects the realities of domains marked by human disagreement or fuzzy category boundaries. In this way, it provides fairer evaluations: it recognizes that categories overlap, that annotators disagree, and that downstream decisions based on similar predictions lead to similar outcomes.
Through theoretical justification and extensive empirical validation on synthetic and real data, we show that Semantic F1 demonstrates greater interpretability and ecological validity. Because it requires only a domain‑appropriate similarity matrix, which is robust to misspecification, and not a rigid ontology, it is applicable across tasks and modalities.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a semantic F1 metric for multi-label classification. The proposed metric captures the ambiguity of class labels and can provide a fair and interpretable evaluation of the model's performance in the related tasks.

### Strengths
(1) The proposed metric effectively captures the fuzziness of class labels and the correlations between different class labels by integrating the semantic similarities among them.

(2) The calculation process of the proposed metric is simple and reasonable, and when the similarity of class labels degenerates to the identity matrix, the proposed criterion degenerates into the standard F1 criterion.

(3) A large number of systematic experiments have verified the effectiveness of the proposed criterion, including the fairness, interpretability, and robustness of the evaluation results.

(4) The experiments demonstrate that the proposed metric can provide guidance for early stopping, thereby achieving better generalization performance.

(5) The article also provides an accelerated calculation scheme for the proposed metric.

### Weaknesses
The proposed metric heavily rely on the quality of the class label similarity matrix.

### Questions
(1) Does the proposed metric have corresponding issues in the single-label classification task? Can the proposed metric be generalized to the single-label classification task?

(2) What are the connections and differences between the proposed metric and the cost-sensitive classification loss functions?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors recognize the limitations of conventional F1 scores, which treat any non-exact label match as a complete failure, and introduce semantic F1 scores for multi-label classification tasks characterized by subjective or fuzzy class boundaries. The proposed metrics incorporate a domain-appropriate label similarity matrix to grant partial credit for semantically related predictions. Finally, the authors provides extensive empirical validation through multiple synthetic and real-world studies.

### Strengths
To address fuzziness of the semantic boundaries among labels, this paper introduces a two-step matching algorithm. It distinctly calculates semantic precision by mapping each prediction to its closest ground-trueh label and semantic recall by mapping each ground-truth label to its closest prediction. These components are then combined via harmonic mean to derive the final semantic F1 score, ensuring balanced penalties for both over-prediction and under-coverage in the semantic label space.

### Weaknesses
1. The novelty of this paper could be further strengthened. The proposed performance evaluation method based on semantic relevance appears quite similar to existing work on cost-sensitive learning; it would be helpful if the authors could clarify the distinctions between them.
2. The applicability of the semantic F1 score may require more discussion. Since it relies on a similarity matrix that is not entirely objective, the fairness of the final evaluation might benefit from further exploration.
3. The main contribution of the paper seems to be primarily centered around Equation (1), which seems relatively minor.

### Questions
1. Could you please clarify the key distinctions between the proposed evaluation methods and existing works in cost-sensitive learning? A more detailed discussion would help better highlight the novelty of your work.

2. Could you provide more justification for the use of the semantic F1 score? I have concerns regarding its fairness, as it relies on a non-objective similarity matrix. It would be helpful to discuss its potential limitations.

3. The paper's core contribution appears to be centered on Equation (1). Could you elaborate on whether there are other significant contributions, or discuss the impact and sufficiency of this primary contribution in greater depth?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper provides a new metric, Semantic F1 Scores, suited for multi-label classification tasks, that does not excessively penalize failures in classification, but rather employs a similarity matrix to account for fuzzy ground truth labels or disagreement between annotators. The authors provide some experiments to evaluate the strength of their metric with respect to real-world and synthetic tasks.

### Strengths
The paper tackles an interesting and timely problem concerning the evaluation of predictive algorithms in multiclass tasks when the ground truth can be fuzzy or could represent some disagreement with respect to, for example, human annotators.

### Weaknesses
(Disclaimer: this is not exactly my area of expertise. I skimmed through the appendices, but I did not check them too deeply.)

- I found the problem formulation not very clear in its exposition. For example, I assumed the F1 score formulation would have been defined in a classical supervised learning setting (e.g., we have samples (x,y) where $X \sim P(X)$ and $y \sim P(Y)$ comes from unreliable annotators…), but here the author opted for a set-based formalization that I believe it is less clear.

- The BestMatch function (Equation 1) only considers the sets of predicted and “ground truth” labels. Thus, it considers all labels as globally related, thus, independently of the particular instance $x$ being evaluated. As far as I understood it, this is a critical issue. For example, in one context, _“irony”_ may be close to _“humor”_, in another to _“sarcasm”_ or _“criticism”_. A global embedding-based similarity cannot capture that nuance. Consequently, _SeF1_ may assign high partial credit even when the predicted label is semantically incompatible in context, simply because it is related in the global embedding space.

- The synthetic study (Section 4.3.1) is too abstract. It would have been better to connect it to a real synthetic classification task (e.g., by creating a simple multiclass classification problem). The same applies to the “perturbed predictors” and to the relevance of the study concerning misspecified similarities. 

- No source code is available. It would have helped to better understand how to practically use the proposed metric (the pseudocode brings you only so far), and it would have solved my concern above. 

- The tone of the paper sounds a bit like “overselling” at certain points. For example, in line 483, I do not believe that the proposed method provides a “more informative evaluation”, given that, as the authors also argue, if the similarity matrix $S$ is misspecified, then you can basically obtain very wrong results.

### Questions
- Can you please explain why BestMatch (Equation 1) does not consider the instance $x$ being evaluated when computing the similarity between the labels?

### Soundness
2

### Presentation
2

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
This paper introduces Semantic F1 Scores, a family of evaluation metrics for multi-label classification that account for semantic similarity between labels. The core idea is to give partial credit when a predicted label is semantically related to the true label, instead of counting it as a complete miss as in standard F1. The metric uses a two-step matching (predict-to-true and true-to-predict) to compute “semantic precision” and “semantic recall,” which are combined into an F1 score. Semantic F1 is designed to be backward-compatible (it reduces to standard F1 when no partial credit is allowed). Additionally, to theoretical arguments, the authors conduct an empirical study on eight datasets (synthetic and real) and demonstrate that Semantic F1 provides potentially fairer and more informative evaluation in tasks with subjective or overlapping labels, correlating more closely with real-world outcomes than classical F1.

### Strengths
- The paper is easy to read and understand.
- Proposes a semantic F1 measure that is nicely motivated and theoretically elegant.
- The experimental study kind of demonstrates some desirable properties of semantic F1 over prior single-step and Hungarian-style approaches.

### Weaknesses
- I think the biggest weakness of the approach is dependency on the similarity matrix. If the matrix S is poorly specified or biased, scores could be misleading or unfair, which makes it tricky to use as an evaluation metric. The defining similarity may be non-trivial. Similarity derived from label co-occurrence or embeddings might not truly reflect conceptual closeness. Results with different S matrices are not comparable.
- Baseline comparison on real tasks is limited, as on real datasets, evaluation focuses on hard F1 vs Semantic F1, with no direct comparison to alternative semantic metrics, which also feels limited in a synthetic study.
- I'm generally a bit confused by experiments; it is not clear to me what is optimized and what is evaluated, and why should I care that Semantic F1 better correlates with the task than hard F1? And what about other soft/semantic metrics?
- Many labels on Figures 2 and 3 are so small that they are not readable when printed.

### Questions
Please, see weaknesses session.

### Soundness
2

### Presentation
3

### Contribution
2
