# The Dual Power of Interpretable Token Embeddings: Jailbreaking Attacks and Defenses for Diffusion Model Unlearning

- Decision: Reject
- Scores: 2, 6, 4

## Abstract
Diffusion models excel at generating high-quality images but can memorize and reproduce harmful concepts when prompted. Although fine-tuning methods have been proposed to unlearn a target concept, they struggle to fully erase the concept while maintaining generation quality on other concepts, leaving models vulnerable to jailbreak attacks. Existing jailbreak methods demonstrate this vulnerability but offer limited insight into how unlearned models retain harmful concepts, limiting progress on effective defenses. In this work, we take one step forward by exploring a linearly interpretable structure. We introduce *SubAttack*, a novel jailbreaking attack that learns an orthogonal set of attack token embeddings, each being a linear combination of human-interpretable textual elements, revealing that unlearned models still retain the target concept through related textual components. Furthermore, our attack is also more powerful and transferable across text prompts, initial noises, and unlearned models than prior attacks.  Leveraging these insights, we further propose *SubDefense*, a lightweight plug-and-play defense mechanism that suppresses the residual concept in unlearned models. SubDefense provides stronger robustness than existing defenses while better preserving safe generation quality. Extensive experiments across multiple unlearning methods, concepts, and attack types demonstrate that our approach advances both understanding and mitigation of vulnerabilities in diffusion unlearning.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose two methods:

- SubAttack, a “human-interpretable” jailbreaking attack that constructs an orthogonal set of token embeddings via non-negative linear combinations of CLIP token embeddings;
- SubDefense, a defense mechanism that projects token embeddings orthogonally to remove residual concept subspaces.

The authors claim that SubAttack provides better interpretability, transferability, and efficiency than existing methods, and that SubDefense offers improved robustness with minimal degradation in generation quality.

### Strengths
The paper makes an attempt to interpret unlearned model vulnerabilities through embedding subspace analysis, which is important for safety and compliance of diffusion models.

### Weaknesses
1. Regarding the novelty of the method, although the concept of inserting v_att tokens is employed, this attack and defense strategy appears to essentially be a linear orthogonal transformation.
2. The method still requires training an MLP, which is not training-free, casting doubt on the stability of the attack strategy if there are many concepts that need to be attacked. 
3. The scalability of the method is questionable, as restoring a single concept requires extensive training. This limits the generalizability of the attack approach.
4. The baselines compared in Figure 8,9 and Table 4 only include ECE and RECE, which seems to lack reliability.

### Questions
See weaknesses

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper addresses a critical vulnerability in ​​diffusion model unlearning​​—where models fine-tuned to "forget" harmful concepts (e.g., nudity, copyrighted styles) remain susceptible to ​​jailbreak attacks​​ that regenerate erased content.

### Strengths
1. The proposed method is well motivated and demonstrates a strong level of interpretability, making the overall approach conceptually appealing.

2. The methodology is described with clarity, and the paper is easy to follow in terms of structure and technical presentation.

3. The experimental section is thorough and comprehensive.

### Weaknesses
1. The description of the authors’ main contributions in the methodology section is not entirely clear. If Section 3.1 corresponds to a prior baseline method and Section 3.2 presents the proposed SubAttack method, it appears that SubAttack might be an iterative extension or multi-token variant of the previous approach rather than a fundamentally new contribution. This raises some concerns regarding the novelty and the strength of the claimed contribution. I would appreciate clarification on this point in the rebuttal.

2. The evaluation of SubDefense appears limited in scope. The paper compares SubDefense with only a few baseline defense methods and under a relatively small number of attack scenarios. This experimental coverage may not be sufficient to convincingly demonstrate the superiority or generalizability of SubDefense.

### Questions
In the methodology section, the paper mentions that SubDefense is designed to defend against the CCE attack method. Does SubDefense require prior knowledge of the attack vector $v_{att}$ used by the adversary? If so, this assumption might be unrealistic in practical settings, and it would be helpful if the authors could clarify how such information is obtained or approximated.

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
4

### Summary
This work presents a new white-box attack (*SubAttack*) against concept erasure methods and a complementary erasure method (*SubDefense*). The attack is designed to be more interpretable by learning adversarial inputs as optimized linear combinations of existing token embeddings. These embeddings are compared against the ones derived from other white-box attacks (UnlearnDiff, CCE) and evaluated as a defense when applied via a proposed subspace projection of the token embeddings. Despite its appealing dual structure, the paper, unfortunately, lacks focus and misses important baselines. For example, it completely leaves more advanced adversarially robust erasure methods out of the picture that rely on adversarial fine-tuning (like Receler, AdvUnlearn, or STEREO). Nevertheless, this paper explores a generally interesting perspective, but feels outdated with its restriction to linear combinations of existing tokens, while even more thorough erasure approaches like Receler or AdvUnlearn have already been proven to struggle with white-box jailbreaking attempts. Ultimately, the effectiveness of SubAttack and SubDefense are constrained by their linear nature, limiting the relevance of both to the community as already more potent defenses and attacks exist. If the experiments were extended to show that their SubDefense's projection approach (with UnlearnDiff, CCE, or SubAttack attack embeddings) is a successful orthogonal method that can be applied together with adversarial fine-tuning of the model, this paper would suddenly become a lot more relevant.

### Strengths
- (S1) **Well written** work with an interesting dual structure by proposing both a linear attack and a linear defense.
- (S2) **Appealing new perspective**: This work studies the composition of attack embeddings as linear combinations of other, non-erased concepts, and then uses safe subspace projections as a simple defensive mechanism.
- (S3) **Honest mention of the limits of the linear framework** in which SubDefense operates as a defense against non-linear attacks.
- (S4) **Extensive supplemental material** with details on more experimental results and ablations.

### Weaknesses
I find the following list of things to be major weaknesses:
- (W1) **No comparison to "non-linear" defenses beyond RECE** such as AdvUnlearn, Receler or STEREO. I understand that these models do not rely on subspace projections in the token embedding space like SubDefense (or the way they apply CCE or UnlearnDiff). However, adversarial gradient-based fine-tuning of model weights, such as STEREO or AdvUnlearn, is a relevant baseline from my perspective that the linear SubDefense framework should be compared to. The question here is whether the subspace projection (used throughout this paper, even with CCE) is a better or comparable defense than a costly inner adversarial loop (like STEREO, which uses CCE internally)?
- (W2) **Unsubstantiated argument**: The argument (in line 468) that defenses against adversarial attacks like CCE are largely unexplored is simply not True. For example, STEREO proved to be robust even against it.
- (W3) **UCE is generally not a very robust baseline** as it is very easy to circumvent. Showing the superiority of SubDefense against RECE is indeed interesting, but unfortunately, not a very relevant contribution in itself. Of course, this entire work "lives" within the linear framework, and RECE is (to the best of my knowledge) the only linear defense framework so far. However, the picture painted in this study should be a bit more complete by quantitatively demonstrating the effectiveness of SubAttack against STEREO.
- (W4) **The results for SubDefense on ESD (Table 5)** reveal that SubDefense fails to defend against CCE (not surprisingly), but crucially, even against its linear complement, SubAttack. Is SubAttack or SubDefense now not effective?

And these are minor weaknesses:
- (W5) **Unclear motivation for interpretability**: The claim that interpretability is important to control the robustness of unlearned deep generative models lacks clarity for me. It definitely is fun to look at word clouds, but the information in it is not necessarily very useful.
- (W6) **The results are spread across too many tables and figures**, making each part feel shallow and the overall picture hard to see. This paper needs more focus and clearer storytelling for the reader.

### Questions
- (Q1) A better wording in Line 204 might be: "non-negative representation" -> "non-negative linear combination"
- (Q2) Line 269: What is the intuition behind this additional defense step for CCE?

### Soundness
3

### Presentation
2

### Contribution
2
