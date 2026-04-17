# Understanding Adversarial Transfer: Why Representation-Space Attacks Fail Where Data-Space Attacks Succeed

- Decision: Reject
- Scores: 2, 2, 4, 2

## Abstract
The field of adversarial robustness has long established that adversarial examples can successfully transfer between image classifiers and that text jailbreaks can successfully transfer between language models (LMs).
However, a pair of recent studies reported being unable to successfully transfer image jailbreaks between vision-language models (VLMs).
To explain this striking difference, we propose a fundamental distinction regarding the transferability of attacks against machine learning models: attacks in the input data-space can transfer, whereas attacks in model representation space do not, at least not without geometric alignment of representations.
We then provide theoretical and empirical evidence of this hypothesis in four different settings.
First, we mathematically prove this distinction in a simple setting where two networks compute the same input-output map but via different representations.
Second, we construct representation-space attacks against image classifiers that are as successful as well-known data-space attacks, but fail to transfer.
Third, we construct representation-space attacks against LMs that successfully jailbreak the attacked models but again fail to transfer.
Fourth, we construct data-space attacks against VLMs that successfully transfer to new VLMs, and we show that representation space attacks _can_ transfer when VLMs' latent geometries are sufficiently aligned in post-projector space.
Our work reveals that adversarial transfer is not an inherent property of all attacks but contingent on their operational domain -- the shared data-space versus models' unique representation spaces -- a critical insight for building more robust models.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper studies the difference between the transferability of attacks in "activation-space" and "data-space". The authors show that attacks transfer for the latter but not for the former across several settings and provide a simple theoretical justification for this. The authors also attempt to connect these insights to the observations that jailbreaks do not transfer across vision language models (VLMs).

### Strengths
- The paper targets the very interesting problem of explaining why jailbreaks do not transfer for vision language models.
- The distinction between "representation-space" and "data-space" attacks and their differing rates of transferability are clearly explained an thoroughly established across different settings and some theoretical justification is provided.

### Weaknesses
1) The paper is motivated by the observation that jailbreaks do not transfer in VLMs, however, the papers cited showed this in "data-space" attacks. Therefore, this paper hinges on the claim that image perturbations in VLMs are akin to "representation-space" attacks. This is an interesting claim which is unfortunately glossed over, as in the following excerpt:
>Implicit in our thinking is that text is the data for vision-language models, and from their “perspective”, visual inputs are effectively perturbations to their activations, akin to a Neuralink implant in a human brain.

2) If the connection to the lack of transferability of jailbreaks in VLMs has not been established as discussed in 1), the study of "representation-space" attacks becomes less significant, particularly because very limited context or motivation is provided for this kind of attack. Is this a kind of attack that has previously been studied in the literature? Why is this kind of white box vulnerability important?

### Questions
If the jailbreaks which do not transfer in VLMs are perturbations applied to the images, why are these considered akin to "representation-space" attacks and not "data-space" attacks? Crucially, even if two VLMs learn rotated versions of each others representations, I would still expect their representations of "dog" to depend on similar parts of the input so that it is not clear that a perturbation for one model will not also affect the other.

Can you provide more context and motivation for "representation-based" attacks, how do they tie to existing literature and why is their lack of transferability important?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper addresses a paradox: adversarial examples transfer between image classifiers/LLMs, but image jailbreaks do not transfer between VLMs.

### Strengths
- Well written.
- The proposed method is interesting.

### Weaknesses
- The proof uses $\ell_2$-optimal representation attacks, but the empirical experiments use $\ell_\infty$ constraints.
- VLM data-space attacks (text jailbreak) are tested without image inputs, but VLMs are multimodal. 
- This paper claims "images behave like representation-space attacks " for VLMs, but provides no analysis of why this is the case.

### Questions
- Please see "Weaknesses".

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a fundamental distinction regarding the transferability of machine learning model attacks: attacks in the input (data) space are transferable, while attacks in the model representation space are not, except when the representations across models are geometrically aligned. The study provides both theoretical and empirical insights into this distinction and sheds light on why image jailbreak attacks often fail to transfer between vision-language models (VLMs).

### Strengths
1. The overall structure of the paper is clear and coherent, and the content is complete and easy to follow.

2. The research topic is meaningful, and the findings provide valuable explanations for the failure of cross-model transfer in image jailbreak attacks between VLMs.

3. The theoretical analysis is rigorous, with mathematical formulations and proofs that strengthen the paper’s theoretical foundation.

4. The insights presented are thought-provoking and could inspire future work on understanding attack transferability in multimodal and representation-based models.

### Weaknesses
1. Figure 1 occupies a large portion of the paper but fails to clearly convey the intended message. It is unclear how the perturbation directions in the second column (data-space perturbations) are chosen. From the figure alone, one cannot intuitively understand the reason why the second column depicts perturbations succeed while the third column shows aligned perturbations that still fail to cross the decision boundary. The decision-space illustration could be significantly improved to more clearly represent these geometric relationships.

2. The paper lacks a formal definition or mathematical formulation of attacks for Image Classifiers, Language Models, and VLMs, especially regarding the distinction between Data-Space Attacks and Representation-Space Attacks. Without this formalization, readers may find it difficult to understand how to construct or evaluate these two types of attacks in practice.

3. For the Image Classifier case, it would be helpful to include a simple derivation for a basic classifier and ensure consistency with the kernel regression analysis framework discussed former.

4. Some experimental figures lack sufficient details about the experimental setup, metrics, and interpretation of the plotted results. Each figure should include a concise explanation of what is being compared, how the axes should be interpreted, and what the main takeaway is.

5. The paper states that “transfer success seemingly does not depend on the number of models used to optimize the attack.” This is confusing, as prior work has shown that ensemble attacks tend to improve transferability. Could the authors provide a theoretical or empirical explanation for this apparent inconsistency?

6. Listing results from different models alone is insufficient to establish the paper’s contribution. The authors should elaborate on the key differences among these models and how these differences substantiate the paper’s theoretical claims. This comparison is essential to understanding the scope and impact of the proposed distinction.

### Questions
1. Can the authors elaborate on why attack transferability does not seem to improve with model ensembles, contrary to previous findings? Is it due to representation misalignment, model diversity, or optimization properties?

2. I find this paper presents an interesting and insightful discovery. However, I have several concerns as mentioned above, and I look forward to the authors’ clarifications. If these issues are adequately addressed, I would be willing to raise my score.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper attempts to explain the limited transferability of image jailbreaks across Vision-Language Models (VLMs).

### Strengths
1.	Investigating and explaining the mechanisms behind image jailbreaks against large models is an important and interesting topic.

### Weaknesses
1. The paper fails to provide meaningful insights into the underlying causes of image jailbreak transferability across VLMs.
2. The motivation for analyzing representation-space attacks is unclear. First, the applicability of representation-space attacks is limited. Second, the relationship between representation-space and data-space attacks is not clearly established.
3. Figure 1 does not effectively illustrate the claimed insight of “Why Representation-Space Attacks Fail Where Data-Space Attacks Succeed.”
4. Although the study focuses on VLMs, the authors also analyze other methods such as kernel regression and image classifiers. The rationale for including these analyses and their connection to the main topic is not well explained.

### Questions
Please refer to Weaknesses.

### Soundness
1

### Presentation
1

### Contribution
1
