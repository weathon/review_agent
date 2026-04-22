# A Universal Source-Free Class Unlearning Framework via Synthetic Embeddings

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 2, 2, 8

## Abstract
Class unlearning in neural classifiers refers to selectively removing the model’s ability to recognize a target (forget) class by reshaping the decision boundaries. This is essential when taxonomies change, labels are corrected, or legal or ethical requirements mandate class removal. The objective is to preserve performance on the remaining (retain) classes while avoiding costly full retraining. Existing methods generally require access to the source, i.e., forget/retain data or a relevant surrogate dataset. This dependency limits their applicability in scenarios where access to source data is restricted or unavailable. Even the recent source-free class unlearning methods rely on generating samples in the data space, which is computationally expensive and not even essential for doing class unlearning. In this work, we propose a novel source-free class unlearning framework that enables existing unlearning methods to operate using only the deployed model. We show that, under weak assumptions on the forget loss with respect to logits, class unlearning can be performed source-free for any given neural classifier by utilizing randomly generated samples within the classifier’s intermediate space. Specifically, randomly generated embeddings classified by the model as belonging to the forget or retain classes are sufficient for effective unlearning, regardless of their marginal distribution. We validate our framework on four backbone architectures, ResNet-18, ResNet-50, ViT-B-16, and Swin-T, across three benchmark datasets, CIFAR-10, CIFAR-100, and TinyImageNet. Our experimental results show that existing class unlearning methods can operate within our source-free framework, with minimal impact on their forgetting efficacy and retain class accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper is targeted at source-free class unlearning task. The authors propose one method that only needs to randomly sample from the intermediate distribution space of the classifier. Using the sampled features to calculate the loss of forgetting and retaining is sufficient. The authors use the proposition to support their claim. The experiments also show the effectiveness of their method.

### Strengths
1. The writing is easy to follow and understanding.

2. The proposed methods is clear and easy to implement.

3. The method is supported by the proposition proved by the authors.

4. The experiments show the effectiveness of the method.

### Weaknesses
1. The results in Table 1 for ResNet-18 is unusual, where the model retrained with access to original samples of forget and retain classes perform worse than all methods. But it should be the upper bound of this task.

2. All experiments are conducted when the number of class to forget is only 1 and the method is also based on this presumption. Could you please conduct experiments when the number of class to forget is greater than 1 and forget the classes one by one/at the same time?

### Questions
1. I wonder how the method perform if the sampled features are exactly the features extracted from real samples of forget/retain classes.

2. I wonder whether the method is applicable to other tasks such as detection.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper shows that class unlearning can be achieved without any data by randomly sampling intermediate embeddings, pseudo-labeling them, and applying existing unlearning losses, making class unlearning truly source-free, simple, and computationally efficient. Overall, while the idea is simple and empirically effective, the paper’s novelty is limited, theoretical analysis is shallow, and the motivation for random embeddings is not sufficiently justified. Strengthening these aspects could significantly improve the manuscript.

### Strengths
(1)The proposed method can be universally compatible with existing unlearning methods.

(2)Strong empirical performance across multiple datasets and architectures.

### Weaknesses
(1)Limited methodological novelty. Although the paper frames the proposed approach as a universal source-free unlearning framework, the core methodology essentially acts as a “wrapper” that adapts existing unlearning methods to a data-free setting. The central idea, randomly sampling embeddings from intermediate feature space and applying pseudo-labeling, is conceptually simple and lacks technical depth. No new loss function, model design, or optimization technique is introduced, making the methodological innovation relatively limited.

(2)The theoretical justification relies heavily on a monotonicity assumption on the logits with respect to the forget loss. This assumption is extremely weak and is satisfied by almost all existing gradient-based unlearning methods. Consequently, the theory provides limited explanatory power or insight into why random embeddings should be effective in practice. The current analysis feels more like a sanity check than a deep theoretical contribution.

(3)The paper lacks a strong motivation for why random embeddings should be preferable or even comparable to learned or generative embeddings. Existing source-free methods (e.g., GKT, DSDA) invest effort in synthesizing realistic or adversarial samples for a reason, namely, to approximate meaningful decision boundaries. This work assumes that random vectors are sufficient but does not provide a compelling argument or empirical analysis to explain why this is the case. As a result, the motivation behind the proposed approach feels shallow and underdeveloped.

(4)Why are random embeddings sufficient? The results suggest that using arbitrary embeddings is enough to drive decision boundary updates, but the manuscript does not explore why this is true empirically. Are there situations where random embeddings would fail (e.g., highly overfitted models or extremely imbalanced forget classes)?

(5)The current framework is class-specific and relies on pseudo-labeling; it is unclear whether this mechanism can be extended to instance-level settings, which are arguably more practical and challenging.

(6)While the reported results are promising, the main results section suffers from over-claiming, limited mechanistic analysis, lack of robustness discussion, and insufficient explanation of why random embeddings work as well as real embeddings.

(7)In figure 2, the paper shows an empirical trend that performance improves with more synthetic embeddings and saturates after a certain point, but offers no deeper reasoning for why 100–200 random embeddings are sufficient to approximate the decision boundaries of real classes. Without theoretical insights or geometric analysis of the embedding space, the conclusion remains largely observational.

### Questions
(1)Insufficient theoretical depth.

(2)Only supports class-level unlearning.

(3)Does the proposed method have failure-case?

### Soundness
3

### Presentation
2

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
This paper considers the problem of source-free class unlearning. This work proposes to randomly generate intermediate features and utilize them for unlearning. A theoretical analysis is provided under an assumption that the forget loss function is monotonically increasing/decreasing with respect to the logit for forget/retain classes. Experimental results show the effectiveness of the proposed methods on image datasets.

### Strengths
+ Machine unlearning is a timely topic, and taking advantage of randomly generated intermediate features sounds interesting.

+ Experiments are conducted with both CNN and ViT backbones.

### Weaknesses
- The goal of this paper might not be considered to be "unlearning." The proposed method deliberately makes the model not to classify inputs to a target class, no matter what the inputs are. This still requires the model to be aware of the target class, to avoid classification to the target class. Rather, the definition in the intro that removing "the influence of specific instances or classes" would be more widely accepted definition of unlearning.

- Theoretical analysis is flawed. Eq. (6) is invalid, as it argues that a scalar is equal to a vector.

- It is not clear how Proposition 1 can be interpreted to draw new information.

- No explanation on the proposed method. L234--240 is not enough to understand the details of the method. For example, what is the probability distribution p_z(z)? If it is parametric, how is it determined? Figure 1 gives a hint that the output of the feature extractor is used, but there is no further description. Furthermore, if p_z(z) depends on the output of the feature extractor, then it is not really "randomly generated."

- No efficiency analysis on the proposed method. As this paper argues that "generating samples in the data space is computationally expensive," the proposed method should be compared with this setting to support this claim.

- Experimental setting is questionable. All baseline methods are using retain and/or forget class data, and adding the proposed method does not directly imply that they do not use such data anymore. That is, it is necessary to explain how they modified baseline methods to eliminate the necessity of retain/forget class data.

- No ablation study on the design choices of the proposed method.

- This paper employs a different font and it paragraph is more dense, compared to other ICLR submissions.

### Questions
What is the definition of unlearning, and why do you think so?

### Soundness
1

### Presentation
1

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
This paper proposes a source-free class unlearning, utilizing the embedding in the intermediate space. It performs classification on these embeddings and generates pseudo-labels to construct synthetic forget and retain sets. And then use these sets for later training. The authors theoretically justify that effective unlearning can be achieved independent of the embedding distribution.  Empirical studies on  multiple datasets and backbones also demostrate the effectiveness of the proposed method. Further experiments show that existing unlearning algorithms can be seamlessly adapted to this source-free setting with minimal performance degradation.

### Strengths
1. the proposed method is simple but effective. The idea of leveraging intermediate embedding for creating the forget and retain sets is very good and effective. 
2. the theoretical proof strengthen the paper's contribution
3. It is very good that the method could be easily applied to existing methods without performance degrade.
4. writing is neat and easy to follow.

### Weaknesses
1. It would be better if the experiments could address some critical applications that unlearning is suitable.
2.  Many results show near-perfect AUS (~1.00), raising concerns about the sensitivity or discriminative power of the evaluation metric under this setting.
3.  it would be better to do some analysis regarding the pseudo-label。
4. novelty could be a problem as it is essnetially create forget and retain sets and access them in a traditional way.

### Questions
The questions are related to the weakness.

1. Could the authors provide examples or experiments demonstrating how the proposed method applies to critical real-world applications where unlearning is particularly useful?
2. How do the authors ensure that the AUS metric remains sensitive and discriminative in evaluating subtle performance differences?
3. Could the authors include an analysis of the pseudo-labels such as their reliability, stability, and effect on unlearning performance?

### Soundness
3

### Presentation
3

### Contribution
3
