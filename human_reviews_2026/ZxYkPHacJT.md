# Classification vs. Deep Feature Learning in Normalized Spaces with Different Scaling

- Decision: Reject
- Scores: 2, 2, 4, 8

## Abstract
In supervised scenarios, deep feature learning is typically implemented through the training of classification models. However, it should be noted that classification reflects the sample-wise local properties of models on a dataset, while deep feature learning aims to learn features with good sample-independent global properties such as intra-class compactness and inter-class separability on the dataset. This paper conducts an in-depth comparison of classification and deep feature learning in normalized spaces. We first reformulate the binary cross-entropy (BCE) loss aligning with the fundamental requirements of feature learning; then, we theoretically analyze and compare its minima with that of the cross-entropy (CE) loss used for classification tasks. Informed by the above analysis, we explore the convergence behavior of the two losses when the scale factor $\gamma$ changes, revealing the differences between classification and deep feature learning. Specifically, when $\gamma$ increases linearly, the convergence rates of the two losses decay exponentially, resulting in poor feature properties for the trained models, although it does not affect their classification. As $\gamma$ decreases, the losses more readily reaches their minima, which helps to improve the feature properties. However, if $\gamma > 0$ decreases linearly and approaches zero, the convergence rate of the losses decay linearly, leading to unsatisfactory feature properties and making the models' classification highly sensitive to minor disturbances. Our experiments fully validate these conclusions. The experimental results also demonstrate the advantages of BCE over CE in more challenging scenarios such as long-tailed recognition and open-set recognition.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper first propose a loss function for deep feature learning based on global constraints—the binary cross-entropy (BCE) loss, applied to multi-class tasks. Then the authors perform a detailed comparison between cross-entropy (CE) and BCE losses in the normalized space, showing that normalized BCE can also achieve neural collapse (NC). As a last contribution, the authors highlight the key differences between classification and deep feature learning by analyzing how the convergence rates of CE and BCE vary with the scale factor. The authors conduct experiments and show that compared to the CE loss, their proposed BCE loss achieves better results on long-tailed recognition and open-set recognition.

### Strengths
The main strengths of the paper can be summarized as follows:
i) The authors propose a method that use binary cross-entropy (BCE) loss function (unfortunately the method is not novel) and provide a detailed comparison against the classical cross entropy (CE) loss. 
ii) The authors analyse how the loss functions CE and BCE behaves with varying scale factor $\gamma$.
iii) Conducted experiments show that their proposed BCE loss achieves better results on long-tailed recognition and open-set recognition compared to CE loss.

### Weaknesses
The primary weakness of the paper lies in its lack of novelty. Both the proposed loss functions and the analyses concerning the scale factor are not new contributions. My detailed comments are as follows:

The authors conduct an extensive analysis on the scale factor. However, this concept merely represents the radius of the hypersphere on which data are distributed in methods such as ArcFace and CosFace. This same hypersphere also contains the regular simplex in deep simplex classifiers [R2, R3], all of which pursue the same objective as neural collapse. The choice of hypersphere radius has been thoroughly investigated in the literature. For instance, the CosFace paper [R1] provides a theoretical lower bound for the radius based on the number of classes and the minimum posterior probability of class centers (see Eq. 6 in their paper). In practice, larger datasets with more classes benefit from larger radius values, and this is theoretically linked to the feature dimension. ArcFace, for example, fixes the radius at 64 for datasets with thousands of identities, a value shown to work well on large-scale benchmarks. Similarly, as stated in [R2], data samples theoretically lie on the surface of an expanding hypersphere as the feature dimension increases. Thus, smaller radii may suffice for low-dimensional illustrative experiments, but larger values are required for high-dimensional cases with high-dimensional spaces. There are also experimental results with varying scale factors in [R2] . Overall, the relationship between the scale factor and hypersphere radius is already well-established, making this part of the paper non-novel.

Regarding the proposed methodology, there exist closely related approaches in prior work. The proposed loss is highly similar to the Dot-Regression Loss in [R3], while [R2] presents an even simpler loss function that also induces neural collapse. Numerous other loss functions targeting neural collapse have been proposed, yet the authors neither cite nor compare against these alternatives.

Finally, the proposed method has a critical limitation that restricts its applicability to large-scale datasets. When d<C−1, the class centers cannot be arranged on the vertices of a regular simplex. This condition commonly arises in modern large-scale datasets, rendering the proposed approach impractical. The authors should therefore suggest modifications or alternatives that can address such cases.

In conclusion, while the paper aligns with the broader line of research on neural collapse, its contributions are incremental. The authors fail to provide compelling justification for why their method should be preferred over many existing alternatives.

Minor Issues: i) There are some typos and English Grammar mistakes that must be corrected, e.g., page 3, libe 155, ... implemented by divided with ...
ii) I did not get the part that is related to unique global minimum. In neural collapse, there are many global minimums that can be simply obtained by rotating the regular simplex.


[R1] H. Wang, Y. W. Z. Zhou, X. Ji, D. Gong, J. Zhou, Z. Li, and W. Liu, ‘‘Cosface: Large margin cosine loss for deep face recognition,’’ in IEEE Society Conference on Computer Vision and Pattern Recognition (CVPR), 2018.
[R2] H. Cevikalp, H. Saribas, B. Uzun, “Reaching nirvana: Maximizing the margin in both Euclidean and angular spaces for deep neural network classification,” IEEE Transaction on Neural Networks and Learning Systems, vol. 36, 2025.
[R3] Y. Yang, S. Chen, X. Li, L. Xie, Z. Lin, and D. Tao, “Inducing neural collapse in imbalanced learning: Do we really need a learnable classifier at the end of deep neural network?” in Proc. Adv. Neural Inf. Process. Syst., 2022, pp. 37991–38002.

### Questions
I raised some questions in Weaknesses part. I really appreciate if the authors answer them.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents Neural Collapse (NC) based analysis (using the unconstrained features model setup) for cross entropy (CE) and Binary CE (BCE) losses and feature normalization (to unit norm). Based on this, the authors make some claims on feature learning.

### Strengths
The writing is OK but I have concerns about the novelty, motivation, and relevance for the study of "feature learning".

### Weaknesses
1. 
Stronger NC, which is typically measured on training data, may degrade generalization, e.g., when the training set is small and/or in case of distribution shift and/or transferability to other tasks. Thus, it is not an indication for good feature learning.

2. 
I believe that "intra-class compactness" is not a desirable property for general feature learning, which should be transferable between tasks. Extreme compactness can be good in classification setups where NC is desirable, but then there is no difference between "feature learning" that is supervised and "classification" (which also includes feature learning).

3. 
Due to focusing on the unconstrained features model, the analysis does not imply anything on generalization (which is necessary for analyzing feature mappings).

4. 
Please provide detailed discussion on the differences between the current work and (Li et al. 2025) ("BCE vs. CE in deep feature learning"), which also analyzed unconstrained features models.

5. 
Please point to supervised learning references where the normalization model (z=\gamma Wh-b, with unit ||u||,||w||) has been used exactly.

6. 
In feature/representation learning, the goal is to learn good transferable embeddings that would perform well on new tasks/classes. The final layer of the downstream task need not be related to the classes of the samples during the feature learning. Thus, the arguments in Section 3.2 seem wrong, as they assume that the features will only be used for the same classification task.

7. 
The motivation for studying the BCE loss should be improved as it does not scale well with the number of classes. Also, as the goal is good feature learning, I suggest contrasting it with supervised contrastive loss and self-supervised approaches.

8. 
Theorem 1 seems to be extremely related to existing results in the literature on NC. 

9. 
Is there technical novelty in the proof of Theorem 2 compared to the proof of Theorem 1?

10. 
The discussions below Theorems 5 and 6 are not clear enough. State formally the convergence rates in the different cases.

11. 
The experiments section is not satisfactory.  
In Table 1 the classification accuracy and the quality of the features should be computed on the test set and not on the train set.
If motivation of the paper is to study "deep feature learning" it should more deeply examine transfer learning, distribution shifts, etc., and compare BCE+normalization with representation/feature learning approaches.

12. 
Compare BCE and CE with normalization with BCE and CE with small weight decay instead of normalization.

### Questions
Stated above.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper gives a neat, self-contained theoretical and empirical study of the gap between classification and deep feature learning (DFL) in a very specific regime: supervised, normalized, closed-set classification under UFM-style assumptions. The central idea is to decouple two objectives that are often conflated: (i) achieving nearly perfect, sample-wise local classification, and (ii) achieving a global feature geometry (NC and ETF-like) that enforces intra-class compactness and inter-class separability. To make this separation concrete, the paper associates standard CE with the local objective and reformulates BCE as a better proxy for the global DFL objective, arguing that, unlike CE’s shift-invariant bias, the BCE bias has a unique, substantial optimum that actually shapes the learned geometry.

A key contribution is the analysis of the scaling factor $\gamma$: the paper shows that as $\gamma$ increases linearly, the convergence toward the desired NC geometry decays exponentially. This creates a tension: large $\gamma$ can drive classification to 100% but stall feature-quality convergence; very small $\gamma$ improves geometric convergence but hurts accuracy. Experiments suggest a moderate $\gamma$ balances the two, and BCE-trained features transfer better to long-tailed recognition and open-set recognition.

### Strengths
- The paper clearly separates two confused goals: getting samples classified correctly vs. learning a good feature geometry.
- Within the stated regime (supervised, normalized, closed-set, UFM), the analysis is coherent, and the CE/BCE/NC/ETF relationships are tied together convincingly.
- The role of the scaling factor $\gamma$ is articulated concretely, showing how increasing $\gamma$ creates a trade-off by speeding up classification but slowing convergence to the NC/ETF geometry.
- The paper makes a useful point that, in this setup, the BCE bias is not incidental but actually shapes the feature geometry.

### Weaknesses
## 1. The applied scope is narrow and should be made explicit
The core Theorems [1-6] are all derived under a stylized configuration: fully supervised learning, a closed label set of size $K$, and feature and classifier normalization. This is essentially the NC (Neural Collapse) theory world. In such a context, using ETF-like target geometries and measuring feature–weight alignment is legitimate. However, today's deep feature learning is primarily driven by contrastive self-supervised methods, which aim for open-set vocabularies, class-weight ($w$)- agnostic batches, and sample-sample semantic alignment. These regimes do not satisfy the paper's core assumptions. The current paper sometimes slides from “we showed this under UFM + fixed-\(K\) + normalization” to “therefore this is how deep feature learning behaves”. This should be tightened. A fairer statement would be:
> “We provide a self-contained analysis for normalized, supervised, closed-set classification with fixed $K$; outside this regime, the behavior may differ.”

## 2. The advantage of bias in BCE can act as a constraint
This paper argues that BCE is superior to CE because the bias $b$ in BCE is substantial and unique, while the bias in CE is somehow useless due to its shift-invariance in softmax. However, we can have some opposite opinions:
- The shift-invariance of bias in CE can be a robustness property that makes the model not over-sensitive to the absolute scale of logits;
- The biases in BCE (Eq. 11) are a clean object only because we are in the aforementioned stylized settings. It is also tightly tied to the training-time number of classes $K$;
Therefore, the same thing the paper calls an advantage can also be read as a strong label-dependent constraint. This constraint removes the translation robustness of CE and locks the optimal geometry to the training value of $K$ in BCE in class-agnostic regimes.

## 3. On the "exponential decay" narrative
This paper claims that, for large $gamma$, the convergence rate decays exponentially; therefore, the model cannot reach the desired NC/ETF geometry. This is one possible reading, but there is also a very natural alternative: a large $gamma$ makes the optimization goal into a simpler one. Since it can make the data separably classified with high confidence ($P \to 1$), SGD can quickly succeed at this easier goal. Once this easy goal is reached, gradients necessarily get tiny (since $1-P \to 0$). What we observe as “exponential decay” is a symptom of success on the easy goal, not a cause of failure on the hard geometric goal. In other words, the paper currently treats a post-separability slowdown as evidence that "large $gamma$ harmed feature learning". However, a post-separability slowdown is expected once the “critical conditions” (I (Eq. 14) and II (Eq. 15) in the paper) are satisfied. Those conditions are themselves local separability conditions, inducing slower dynamics.


## 4. The Math Issue
To begin with, I'd like to say that the theoretical section would benefit from tightening the assumptions and motivating the key inequality from the problem structure, rather than presenting a long algebraic chain.

The most serious mathematical issue in this paper appears in the BCE part of the appendix, where the authors reuse an AM–GM type inequality but violate its own stated precondition. In Appendix E.2, the authors attempt to derive a lower bound for the BCE loss by invoking the standard AM–GM type inequality: $u^\top v \le \frac{c}{2}\|u\|_2^2 + \frac{1}{2c}\|v\|_2^2, \text{for } c > 0.$ They explicitly require $c > 0$ for this inequality to hold.

However, in the tightness part of the argument, where they try to make the inequality achieve equality and at the same time recover the NC/ETF-like structure, the derivation effectively enforces $h_i^{(k)} = - c_4 \gamma w_k$, and, combined with the target condition $h_i^{(k)} = w_k$, this forces $c_4 = -\frac{1}{\gamma} < 0$. That is, to make the proof work, the authors end up choosing a negative value for the constant, which must be positive according to the inequality they cited. This is a direct violation of the premise $c > 0$. As a result, the corresponding lemma (and the theorem that depends on it) does not currently have a valid proof in the appendix.

### Questions
- The theory in this paper is derived in a strict NC-style setting (supervised, normalized, closed-set, fixed $K$). What is the concrete insight or usefulness of your results for current self-supervised contrastive feature learning, which does not satisfy these assumptions?
- The most serious issue is the one in weakness #4. This breaks the stated condition and renders Lemma 12 invalid as written. Please clarify or fix it.
- Please also address the other weaknesses.

One tricky point is that, under today’s dominant self-supervised contrastive deep feature learning, the practical value of this paper’s classification-based setup is limited. That said, given ICLR’s openness to theory, I am willing to accept the authors’ chosen supervised closed-set feature learning framework; within this framework, the result that BCE can be preferable to CE is meaningful. However, the mathematical issue in Weakness #4 (violation of the inequality’s own precondition) affects Lemma 12 and is critical. I therefore currently place the paper below the acceptance threshold. If the authors can fix this issue, and if other reviewers are comfortable with the restricted learning paradigm, I would be happy to raise my score.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors analyze cross entropy (CE) and binay cross entropy (BCE)
in normalized spaces for classification and feature learning.  In
normalized spaces, both feature (h) and weight vectors (w_j) are
normalized with norm 1.  The decision vector (z) is adjusted with a
scaling factor (gamma), where z = gamma*Wh - b.  For classification,
class probabilities are obtained via softmax and the loss function is
cross entropy (CE).  For feature learning, intra-class compactness and
inter-class separation are considered.  For inter-class compactness,
min{w_k . h_i} is greater than some threshold.  Similarly for
inter-class separation, max{w_j . h_i} is less than some threshold.
Binary Cross Entropy (BCE) is used as is in multi-class classification.

Based on a number of assumptions, the theoretical analysis indicates
that normalized CE and BCE lead to neural collapse (NC) when minimum
is achieved.  The CE loss cannot enhance compactness and separability
and has many minima.  However, normalized BCE, which incorporates
compactness and separability, has only one minimum.

For normalized spaces, another analysis shows that when gamma is very
large, classification performs well while feature learning performs
poorly.  When gamma is very small, both classification and feature
learning perform poorly.

For empirical analysis, they use 2 existing network architecture over
3 datasets.  The use existing NC metrics to measure NC progress.  When
gamma is 8, the empirical results indicate that both BCE and CE can
lead to NC when they reach the minimum.  On varying gamma values, both
CE and BCE perform poorly when gamma is less than 1.  Both CE and BCE
performs well when gamma is larger than 1.  When gamma < 8,
convergence can be achieved.  As expected BCE generally has better
compactness and convergence.  On long-tailed recognition and open-set
recognition, BCE performs better than CE.

### Strengths
1.  Analyzing CE and BCE in normalized spaces is interesting.

2.  The theoretical analysis shows that CE and BCE under certain
assumptions can lead to NC, and larger gamma is useful for
classification.

3.  The empirical analysis indicates that the results roughly follow
the theoretical analysis.

### Weaknesses
1.  Since BCE explicitly incorporates compactness and separability,
BCE has better feature learning is expected.

### Questions
p8: "In contrast, when gammaγ = 32 or 64, the models’ accuracies reach
100%, while the intra-class compactness and inter-class separability
of their features are comparatively poor and worse than that learned
with small" Why compactness and separability did not further improve?
Is it because convergence was not achieved?

How would BCE perform without incorporating compactness and separability?

### Soundness
3

### Presentation
3

### Contribution
3
