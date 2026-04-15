# Knowledge Accumulation in Continually Learned Representations and the Issue of Feature Forgetting

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 6, 3, 3

## Abstract
While it is established that neural networks suffer from catastrophic forgetting ``at the output level'', it is debated whether this is also the case at the level of representations. Some studies ascribe a certain level of innate robustness to representations, that they only forget minimally and no critical information, while others claim that representations are also severely affected by forgetting. To settle this debate, we first discuss how this apparent disagreement might stem from the coexistence of two phenomena that affect the quality of continually learned representations: knowledge accumulation and feature forgetting. We then show that, even though it is true that feature forgetting can be small in absolute terms, newly learned information is forgotten just as catastrophically at the level of representations as it is at the output level. Next we show that this feature forgetting is problematic as it substantially slows down knowledge accumulation. We further show that representations that are continually learned through both supervised and self-supervised learning suffer from feature forgetting. Finally, we study how feature forgetting and knowledge accumulation are affected by different types of continual learning methods.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper analyzes the catastrophic forgetting issue in continual learning.
The authors focus on forgetting in the *representation*, which is narrowly defined as the last activation right before the final output.
The main findings are summarized as follows.
1. Catastrophic forgetting does occur in the representation.
2. This forgetting harms the final performance.

### Strengths
The text is easy to follow.
The main findings are clearly stated at the beginning.

### Weaknesses
### Novelty

The major weakness of this work is that the main findings are not new.
I think the two main findings are the most basic assumptions of continual learning.
Personally, the most surprising part of this paper was that there are several works claiming that forgetting is minimal in the representation.
But if forgetting in the representation is negligible, why would the entire field of continual learning exist?


### Limited Scope of Analysis

The analyses in this work were conducted in quite narrow settings.
The authors focused exclusively on the last activation of a network for classification tasks.
Furthermore, they concentrated solely on the offline continual learning scenarios, excluding online continual learning from their scope.

### Writing

While I didn't have much trouble understanding the paper, there is room for improvement in the overall writing, particularly in terms of grammar.

### Questions
N/A

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper tries to explore whether neural networks suffer from catastrophic forgetting at the level of representations. This paper focuses on two questions: Do continually trained representations forget catastrophically, and Does it matter that these representations are forgotten. To answer these questions, the main contributions of this paper are summarized as"

a. This paper shows that continually learned respresentations do forget catastrophically.

b. The respresentation forgetting negatively affects knowledge accumulation.

c. This paper also consider feature forgetting and knowledge accumulation in continual learning methods.

d. This paper explores the feature forgetting with self-supervised and contrastive losses.

### Strengths
a. This work is novel and explores whether neural networks suffer from catastrophic forgetting at the level of representations. 

b. This paper is well-written and easy to follow.

c. Extensive experiments. I appreciate that this paper provides extensive experiments to show the effectiveness of the proposed method.

### Weaknesses
Although the experimental phenomenon presented in this paper is very interesting, it is essentially an experimental work. It is hard to determine whether the conclusions drawn in this paper are widespread or only based on the experimental settings (models, datasets) used in this work. However, this type of work is OK and interesting.

### Questions
In Section 3, this paper argues that before learning task $t$, the model contains different level of information of task $t$ in different scenarios (red and blue lines). It is noteworthy that all tasks in CL are disjoint (mentioned in Section 2) and the model starts from scratch (mentioned in Appendix A). How does the model achieve the knowledge of task $t$ before learning task $t$ and why are you confirmed that this knowledge achievement is valid? Please discuss more about it and it is better to provide theoretical and experimental support.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper investigates the relationship between representation feature forgetting and knowledge accumulation during continual learning. The paper suggests relative forgetting that is very similar to Backward Transfer, yet dividing it with the performance improvement obtained by training the target task. To demonstrate the authors' claims, the paper provides multiple empirical results. In summary, they find that the continual learning models consistently forget the features severely, and this interferes with knowledge accumulation when learning new tasks. Additionally, this feature forgetting can be alleviated through adopting various continual learning approaches.

### Strengths
The authors scrab various claims on catastrophic forgetting during continual learning from multiple literatures. And suggests a new relative forgetting metric.

### Weaknesses
- Regarding feature forgetting, the paper simply repeats the observations of prior/conventional literature on continual learning: forgetting occurs, and it matters the performance of the model. In that sense, the suggested relative forgetting metric does not show any distinguished observations on existing metrics like Backward Transfer and Averaged Forgetting.

- Limited contribution: Although the paper is dedicated to studying well-known and sufficiently analyzed challenges in continual learning fields, the evaluation tasks, domains, models, method types (e.g., rehearsal-/architecture-/regularization-/prompt-based approaches), ..., are limited, and it is hard to catch 'new'/'novel' insights. - Most observations resort to image-based benchmark classification tasks. There are various continual learning approaches in vision/language/multimodal domains with diverse tasks, segmentation/object detection/generation/text classification/(visual) question answering, etc. 

- Presentation/writing can be further improved. It seems to include repeated claims and unnecessary sentences. For example, the first paragraph in Section 3 is about what is 'catastrophically', but this paragraph is not aligned with the overall flow and arguments of the paper. the word 'catastrophic' simply indicates critically bad, or severe, and no more implication. 

- The faithfulness/benefit of relative forgetting is not clearly described. Regardless of initial performance on target tasks in continual learning, the model contains the most beneficial representations of the task when its performance is the highest during continual learning, and the degenerated performance can be considered as knowledge loss, i.e., forgetting. As shown in the paper, this new metric shows a similar tendency to existing forgetting/backward transfer metrics without new insights, It is not clear why we need to care about the 'relative' forgetting.

- In section 4, the suggested ensemble baseline violates the conventional continual learning setting and is clearly different from the typical continual learning model. Let us store N backbone models by training N past tasks sequentially, the authors concatenate all features on evaluation data from these models and propagate the concatenated features to the classifier. Here, the input dimension of the classifier is different from the base continual learning model (proportional to the number of past tasks (i.e., stored models)), and this means the trainable parameters are N times larger. This is totally different model, and evaluation analyses with the assumption that 'the base continual learning model and ensemble models learn the continual learning tasks in the same way' may not be correct.

- Aligned with the second weakness, I strongly recommend that the authors provide further clear contributions against earlier works that extensively study representational forgetting and transferability in continual learning methods [1,2]. In particular, [2] also observed different behaviors among supervised, self-supervised, and contrastive continual learning in view of representation forgetting and knowledge accumulation.

[1] Chen et al., "Is forgetting less a good inductive bias for forward transfer?" ICLR 2023.  
[2] Yoon et al., "Continual Learners are Incremental Model Generalizers", ICML 2023.

### Questions
.

### Soundness
2 fair

### Presentation
1 poor

### Contribution
1 poor

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper studies the quality of learned representations in continual learning. With the help of two new metrics -- linear probe accuracy and relative forgetting, it is shown that representation learning also suffers from catastrophic forgetting in both continual supervised learning and continual self-supervised learning, and thus reduces the overall task performance.

### Strengths
- The two proposed metrics --- linear probe accuracy and relative forgetting --- are useful for the community.
- Experiments are performed in both continual supervised learning and continual self-supervised learning.

### Weaknesses
- The writing and presentation can be significantly improved. For example, Figure 4 is quite confusing. What does each square mean? What does each color represent? How about the sizes of the squares?
- Lack of deep analysis. This is my major concern. Since this paper does not propose new methods or new theories, I would expect to see more insights about continual representation learning, which the paper does not provide much. The main conclusions are within expectation and well known in the continual learning community. Knowledge accumulation and feature forgetting are another expression of the stability-plasticity dilemma. I would encourage authors to improve this work by providing deeper analysis. For example,
  - In the abstract, it is mentioned that "Some studies ascribe a certain level of innate robustness to representations, that they only forget minimally and no critical information, while others claim that representations are also severely affected by forgetting." Why does the contradiction exist? With two proposed metrics, can we explain and verify this contradiction effectively and directly? 
  - Different methods are tested to prevent representation forgetting. However, there is a lack of analysis or discussion to explain why method A is better than method B.

### Questions
See Weaknesses.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair
