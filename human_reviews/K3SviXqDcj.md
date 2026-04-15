# Invariance as A Necessary Condition for Online Continual Learning

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 6, 5

## Abstract
Traditional supervised learning aims to learn only features that are sufficient to classify the current given classes. 
This is highly problematic for continual learning (CL), which learns a sequence of tasks incrementally. It is also a major cause for 
catastrophic forgetting (CF). Although numerous CL methods have been proposed to mitigate CF, theoretical understanding of the problem is still limited. Recent work showed that if the CL learner can learn as many features as possible from the data (dubbed holistic representations), CF can be significantly reduced. This paper shows that learning holistic representations is insufficient and it is also necessary to learn invariant representations because many features in the data are irrelevant or variant, and learning them may also cause CF. This paper studies it both theoretically and empirically. A novel invariant feature learning method related to causal inference theory is proposed for online CL, which boosts online CL performance markedly.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper addresses the challenge of online continual learning, emphasizing the necessity of learner invariance to prevent catastrophic forgetting. The authors specifically focus on achieving invariance with respect to background and edge expansion, considering them as crucial target variables.

The proposed solution, IFO, is introduced as an approach to address this issue. However, in direct comparison with the competing baseline, OCM, IFO underperforms. Despite this, it is important to note that IFO complements OCM from an alternative perspective, providing a contribution to the field of continual learning.

### Strengths
S1:
This paper delves into a relatively unexplored realm within continual learning: the concept of invariance.

S2:
The authors offer a theoretical foundation for their argument, grounded in the principles of causality. However, I am not 100% sure about the validity of their findings, especially regarding the gradient norm. 

S3:
The authors conduct evaluations across various distinct class-incremental learning scenarios, including settings involving blurry and disjoint data, enhancing the comprehensiveness of their study.

### Weaknesses
W1:
The experimental contribution of this paper appears to be limited. Despite the central argument emphasizing the necessity of invariance for online continual learning, the experimental evidence provided is weak. Multiple instances indicate that the proposed approach, IFO, often underperforms the compared baseline, OCM. This suggests that a holistic approach might be more critical than mere invariance in preventing forgetting.

W2:
Regarding the technical contribution, the proposed IFO relies on an alignment loss, essentially a variant of the contrastive learning loss. This contribution, while present, seems limited in its originality and novelty.

W3:
The scope of the studied invariance setting appears to be constrained. The choice to generate foreground-background blendings, where the specific nuisance to be addressed is known, might be considered somewhat artificial. This artificiality raises questions about the broader applicability and relevance of the findings.

W4:
In terms of presentation, the quality of writing is notably low. The paper proves challenging to read due to unsupported claims, repetitiveness, excessively long sentences, and subpar figures and table presentation. These issues significantly impact the clarity and overall readability of the paper.

Considering these concerns, the paper is currently not deemed suitable for publication until these issues are addressed and improved in the rebuttal phase.

### Questions
N/A

### Soundness
2 fair

### Presentation
1 poor

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper highlights the importance of learning invariant features in mitigating the phenomenon of catastrophic forgetting in class-incremental continual learning through theoretical analysis.  A new method based on experience replay is then proposed for learning invariant features via creating environmental variants using 3 data augmentation techniques.

### Strengths
1. This paper underscores the importance of learning invariant features in mitigating the phenomenon of catastrophic forgetting in class-incremental learning, thus providing a promising direction for the development of novel CIL algorithms.

2. Despite the existence of prior research on the idea of learning invariant features in continual learning, this paper offers a review on the related works and places itself in a good position in the literature.

3. The integration of the proposed method with OCM results in exceptional empirical performance.

### Weaknesses
1. The proposed methods incorporates many hyper-parameter, including $\alpha$, $s$, $k$, $r_1$ in data augmentation. However, determining the optimal values for these hyper-parameters in continual learning can be challenging. Furthermore, this paper lacks comprehensive studies of the effects of the hyper-parameters and their determination.

2. The paper does not empirically verify the individual impact of learning invariant features in mitigating catastrophic forgetting. It is always combined with replay. Moreover, the accuracy and forgetting of the proposed IFO are not competitive compared to OCM and its optimal performance relies on integration with OCM.

3. Some minor issues:
- No experimental results on time efficiency comparison in the Appendix.
- $\theta$ is repeatedly defined as feature extractor parameter and as binary vector in Eqn. (2).

### Questions
1. How is the integration of IFO and OCM achieved? Does it involve the direct addition of two losses during model training? Furthermore, how does IFO coalesce with OCM in feature learning? While the objective of OCM is to learn as many features as possible, IFO solely focuses on invariant features, leading to some conflicting goals.
2. For background color augmentation, during the initial stages of training, when the classifier is not yet fully trained, there may be concerns regarding the accuracy of background color augmentation. There may be concerns where CAM fails to correctly identify the background, resulting in incorrect augmentation.
3. For method II proposed in 5.3, as mentioned, "We collect all new task data already stored in the buffer". Method II seems not applicable if there is no new task data in the buffer, which is often the case in continual learning when new task appears. Is this correct? Also, when the buffer is updated, do the k clusters get updated?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
Previous studies in this direction have shown that learning holistic representations in Continual Learning settings can help to mitigate forgetting. However, in this paper, the authors argue that the representations must also be invariant. This paper shows, theoretically and empirically, that having both holistic and invariant representations helps in online scenarios. After raising the need to remove spurious correlation of the previously learned model, the authors propose a new method for Online Continual Learning. They show good performance in traditional disjoint task scenarios in multiple benchmarks and also in blurry task boundaries and data shifts.

### Strengths
- Studying the theory behind a scenario and understanding the how and why of the problems is an excellent way of proposing new methods.
    - It is more promising when results on the empirical side also accompany these findings.
- Evaluating the method in various online scenarios proves its robustness.
- The augmentations proposed are interesting. Increasing the variability of the data can help improve the generalization capabilities of the model. 
    - These augmentations can work better than MixUp, under specific settings.

### Weaknesses
- My main concern is the Theoretical Analysis, which can have significant repercussions in the practical section. I agree with the intuition of the authors that if we focus on learning invariant and holistic representations, we can mitigate forgetting. However:
    - To learn these representations, the authors assume that the model converges. Something that is not the case in most Online scenarios. The paper does not show that the model learns invariant or spurious correlations. Showing that a model learns these kinds of representations is a challenging task since there is a whole research area that is focusing in this direction.
    - Assuming that one can select which representations are invariant or spurious is a big If, more with the limited distributions that are stored in the buffer. One may obtain an approximation, but solving this problem is not trivial, as multiple work in spurious correlation has proven.
    - I appreciate the great work done on the theoretical analysis of the work, together with the motivations and intuitions. However, the demonstrations and assumptions are not without issues.
- There are a few mentioned in the paper that the proposed method works in Class-Incremental Learning settings, without specifying the online setting. However, it could be better to refer to Online Learning. Changing between class-incremental and online class-incremental and use both interchangeable could create confusion.
- The augmentation methods proposed only work in particular scenarios. Where is one easy-to-detect object and is centered in the image. Something that occur in limited opportunities.
    - Also, as mentioned in the paper, the authors assume that the background is the primary source of spurious correlation. This limits the proposal even more.
- The paragraph “Learning invariant representations” in the results section is incomplete. It mentioned that the model was trained with original datasets, and I would assume until convergence. Training the model this way could create completely different representations. It is essential to show that the proposed method in Online Learning can generate those representations, something that, as mentioned, I am not entirely sure about.

### Questions
- Please explain the last paragraph of Appendix A. There are a few steps that are not trivial to me.
- Can we have a CL method that works under your assumptions but without a memory buffer? Or is access to a memory essential to achieve a model that does not forget?
- What are the similarities or differences between the “data environment shift setting” and a “domain incremental”?
- One significant limitation of Online scenarios is the computational capacity. Some argued that most methods underperform due to the low time to train the model. However, in the proposal, you are increasing the losses, by increasing the augmentations (Eq 1, 5 and 7). How much do you expand the batch size to be able to see all the samples? Or do you keep the batch size fixed and increase the interactions you train the model?
    - How much does the computational time increase?
- The size of the buffer and the samples stored can significantly influence the accuracy obtained in memory-based methods. How does the K value behave when changing the buffer size?
- Can you explain Table 3 in a different way? When it said, “no align”, you just removed the align? In most cases, the difference is less than 1% of accuracy. How can you identify if one of these components is not necessary?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
