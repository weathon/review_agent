# Adversarial Data Robustness via Implicit Neural Representation

- Avg Score: 4.00
- Decision: Reject
- Scores: 3, 8, 3, 3, 3

## Abstract
Despite its effectiveness, adversarial training requires that users possess a detailed understanding of training settings. However, many common users lack such expertise, making adversarial training impossible and exposing them to potential threats. We propose ``adversarial data robustness'', allowing the data to resist adversarial perturbations. Then, even if adversaries attack those data, these post-attack data can still ensure downstream models' robustness at users' end. This leads to our new setup, where we store the data as a learnable representation via Implicit Neural Representation (INR). Then, we can train such a representation adversarially to achieve data robustness. This paper analyzes the possible attacks to this setup and proposes a defense strategy. We achieve a comparable robustness level without resorting to model-level adversarial training.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
Considering that adversarial training for models are sometimes difficult for common users, the paper proposes a data-level robustness method called Implicit Neural Representation (INR) for data-level adversarial training, which adversarially trains INR to get robust data features without losing their contents.

### Strengths
1. The adversarial data robustness proposed in this paper is original, since most of the previous works focused on models' robustness, while the abstacles in their applications were hardly considered.

2. The idea of double projection innovatively explored the functions of projection in gradient-based attacks.

### Weaknesses
1. Since data-level robustness is a new area, the meaning about robustness designed with spatial coordinates is not clear.

2. The explanation about why the robust data can only be created via adversarial training in Section 3.1 is not convincing enough.

### Questions
1. Since data-level robustness is a new area, the meaning about robustness designed with spatial coordinates is not clear.

2. The explanation about why the robust data can only be created via adversarial training in Section 3.1 is not convincing enough.

Update after discussion

After discussion, the other reviewers and I think this work has a serious flaw in the use of label information. Thus, I will revise the rating score to 3.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper investigates adversarial data robustness by applying Implicit Neural Representation (INR), storing the image as a network-based representation, and then reconstructing the representation with robustness enhancement before calling task models. With adversarial data robustness, the users do not need to worry about the model-level robustness with adversarial training in practice. The paper looked into two different attack stages and proposed adaptive attack DPGD. Last, the paper proposed a new defense-in-creation strategy for defense and extensively evaluate the performance.

### Strengths
The paper proposes a novel direction for adversarial robustness on data preparation stages. It prevents user enhancing model robustness with adversarial training while still achieving good robustness behavior.

According to the empirical experiments, the strategies achieve promising results compared to traditional model-level robust training.

The paper is well-written and easy to follow.

### Weaknesses
The authors motivate the adversarial data robustness by assuming many users do not have knowledge for adversarial training on their models. However, the proposed strategy still needs robust training for INR model. This is one caveat for the motivation.

It would be helpful to evaluate stronger attacks like Auto Attack besides CW and PGD.

### Questions
Is this approach compatible to model-level adversarial training? I am curious if applying both data-level and model-level approaches would lead to stronger robustness over SOTA?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a method that makes data resistant to adversarial perturbations, ensuring that downstream models remain robust even when subjected to attacks. The author achieves this by representing data as Implicit Neural Representations (INRs), without the need to modify the deep learning models’ weights.

### Strengths
The proposed concept of *adversarial data robustness* is interesting and leveraging the INR framework to achieve data robustness is novel.

### Weaknesses
This paper is subject to several weaknesses that need to be addressed.

1. **(Motivation)** The paper's motivation for storing data as a learnable representation via INR in the context of adversarial robustness is not well-established. Additionally, the proposed attack scenario, involving direct manipulation of INR parameters during data transmission, raises concerns about its practicality and real-world relevance. It would be beneficial for the authors to provide a more robust justification for these choices and consider the feasibility of such attacks in practical applications to ensure a more realistic context for the proposed defense strategy.

2. **(Method)** The paper introduces the concept of storing data as INRs for adversarial robustness, but it lacks clarity on how these representations can be derived for testing data. Given that testing data is typically only accessible during the testing phase, the paper should address the challenge of generating specific INRs for each testing image. For now, I would treat INR as a mere image generator. 

3. **(Experiment)** The experimental design in the paper appears to lack fairness, which fails to prove the efficacy of the proposed method. Also, it is unclear if the proposed framework could be used to defend against unseen attacks (e.g., [1]).


[1] Perceptual adversarial robustness: Defense against unseen threat models. (ICLR 2021)

### Questions
1. Please provide a more in-depth justification for the choice of INR as a data representation method for adversarial robustness.

2. How do the authors envision real-world applications where adversarial data robustness using INR would be particularly beneficial?

3. In the proposed setup, how can users derive INR representations for testing data, considering that in adversarial training, testing data is typically not accessible during training?

4. Are there any potential biases or confounding factors in the experimental setup that need to be addressed?

5. Can the proposed framework used to defend unseen attacks?

6. Please provide the computational cost of the proposed method and compare it with other baselines.

### Soundness
1 poor

### Presentation
2 fair

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
This paper proposes adversarial data robustness, aiming to allow the data to resist adversarial perturbations. The proposed method stores the data as a learnable representation via Implicit Neural Representation (INR), and trains such a representation adversarially to achieve data robustness. Empirical evaluations are done on CIFAR-10/100 and SVHN, against PGD/DPGD, FGSM, and CW attacks.

### Strengths
The strengths of this paper include:
- The writing is clear with intuitive explanations such as Figures 1 and 3. The formulas in Section 3 is straightforward and easy to follow.
- I like the high-level idea of data robustness, which is supposed to be (conceptually) more efficient than model robustness.

### Weaknesses
The weaknesses of this paper include:
- The attack during creation formulated in Eq (2) is not an *adaptive attack* [1,2]. Specifically, an adaptive attacking objective should be like
$$\\max\_{\\Delta I\_{i}}\\mathcal{L}\_{CE}(g\_{\\phi}(\\Psi\\circ f\_{\theta}(I\_{i}+\\Delta I\_{i})),y\_{i})\\textrm{,}$$
where the INR decoding and reconstruction process implemented by $\Psi\circ f\_{\theta}$ should be involved.

- There are two extra computations introduced by the proposed method: first is the optimization process of $f\_{\theta}$, which is required to be optimized for each input image (i.e., cannot be amortized); second is the defense-in-creation process of Eq (4), which requires adversarial training by perturbing model parameters $\\theta$. The authors should report the empirical cost (e.g., computational time) for these two operations.

- The considered attacking methods such as PGD/DPGD, CW and FGSM are not strong. The authors should evaluate their methods under strong attacks like AutoAttack[3] and compare with the state-of-the-art models listed on RobustBench[4].


References: \
[1] Athalye et al. Obfuscated gradients give a false sense of security: Circumventing defenses to adversarial examples. ICML 2018 \
[2] Carlini et al. On evaluating adversarial robustness. arXiv 2019 \
[3] Croce and Hein. Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks. ICML 2020 \
[4] https://robustbench.github.io/

### Questions
From my experience, the proposed method (with 92.51% accuracy against PGD) probably has gradient obfuscation [1,2]. The authors should evaluate their method under adaptive attacks that involving the defense mechanism (i.e., INR mechanism), as well as strong off-the-shelf attacks such as AutoAttack.


References: \
[1] Athalye et al. Obfuscated gradients give a false sense of security: Circumventing defenses to adversarial examples. ICML 2018 \
[2] Carlini et al. On evaluating adversarial robustness. arXiv 2019

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 5

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper investigates the problem of adversarial attacks in a new threat model designed around Implicit Neural Representations (INRs). INRs are a new family of data representations where each data point is represented via neural networks. To this end, often an MLP is used to overfit a data point. Then, the MLP weights can be used instead of each data point. Examples of INRs used in real applications include 2D images and 5D radiance fields (NeRFs).

In this context, this paper argues that INRs can be used to transmit data points between users, and as such, are prone to adversarial manipulation. In particular, an adversary might add a perturbation to the image data _before_ being encoded as INR, or they may opt to manipulate the INR weights _during transmission_. The former is called "attack during creation" while the latter is named "attack during transmission."

The paper presents a formulation for generating each of these attacks. Specifically, a projected gradient descent (PGD) attack is used to generate adversarial perturbations for attack during creation. Then, this adversarial example is encoded via an INR. Also, for attack during transmission a double projection gradient descent (DPGD) is used to ensure that while INR manipulation fools the downstream classifier, it generates attacks that are imperceptible in the image domain. Empirical evaluations indicate that both of these attacks are effective against CIFAR-10, CIFAR-100, and SVHN datasets.

Finally, as a potential defense against these attacks, the paper proposes to make the INR representation of the data robust to manipulation, resulting in an algorithm dubbed "defense in creation." To this end, the paper proposes a new objective function for creating the INR representation of the data by adding a regularization term. This term aims to add perturbations to the INR during creation while ensuring that these changes have minimal effect on the downstream classification task. Empirical results indicate that this strategy is helpful in making INRs robust to adversarial attacks.

### Strengths
- The paper investigates a new problem in adversarial machine learning. As far as this reviewer knows, there are no prior works that investigates robustness of INRs. Thus, the setting seems interesting, although its practicality needs to be discussed (see the Weaknesses).

- Setting the problem's definition aside, the paper explores the problem from different perspectives. In particular, exploring both the possible attacks and proposing a new defense strategy against them is rather appealing.

- Finally, the paper is very well-written and provides enough detail (such as figures and pseudo-codes) to understand each aspect of it.

### Weaknesses
- The major issue of this paper in this reviewer's view is its motivation and threat model. Specifically:

1. The paper starts its discussion by arguing that adversarial training is hard to deploy. The paper reads: "_Despite its effectiveness, adversarial training requires that users possess a detailed understanding of training settings._" In my view, this argument can even be used for using vanilla neural networks. One can make the same argument that even for using/training regular neural networks users need a detailed understanding of model architecture, optimization process, etc. There is no major differences that separates adversarial training from vanilla training, and I feel that motivating the core problem around such arguments is weak.

2. More importantly, the threat model is not intuitive. In particular, the paper assumes that one uses the INRs to encode the data, then send them to a model, which again decodes the INR to query a classifier. Why this process is efficient? Why the user doesn't send their data directly? What is special about this threat model? I believe that the proposes threat model is not making intuitive sense, and it requires a better design. For instance, using NeRF to encode a scene and then transmitting it would have made a much better threat model as NeRFs are encoding multiple scenes in one representation. However, using the current threat model for 2D images seems less intuitive and might not make any practical sense.

- The empirical evaluations are also limited. The paper only uses small scale datasets such as CIFAR and SVHN. It would be crucial to see how the proposed attacks and defense work in more large scale datasets such as ImageNet. Given that there are many pre-trained models available for the ImageNet dataset online, I believe that such evaluations would be feasible.

### Questions
- Please clarify what do you mean by arguments like "_our setup is rooted in the reality that many model users often fail to understand the settings associated with adversarial training._"

- Given that $\ell_p$ norms are used to enforce visual image similarity in the image domain, its use for other domains such as INR weights seems less intuitive. What are other alternatives for the similarity in the weight space that could have been used? In other words, are there any other alternatives to the $\ell_p$ norm used in Eq. (3)?

- What do you mean by $||\hat{I}\_{t-1}+\nabla-\hat{I}\_{\mathrm{org}}||_p \leq \zeta$? Is this a typo?

- It would be nice to add a few more sentences to the last paragraph of Section 3.2 explaining why projecting the gradient of the image space would help with having a better image quality. In other words, what motivated this step?

- What is the training time difference between finding an INR using Eq. (1) versus the defense in creation in Eq. (4)?

- Did the paper also test the transferability of the defense in creation? In other words, what happens if we find the robust INRs for a classifier $g\_{\boldsymbol{\phi}}^{(1)}$ while trying to defend another model $g\_{\boldsymbol{\phi}}^{(2)}$ during inference?

- Run experiments on large scale datasets such as ImageNet-1k or ImageNet-100.

- How do you specify the upper-bound for adversarial attack against INR weights? In other words, what makes a good $\delta$ for Eq. (3)? Because using $\ell_p$ norm in the INR weight space is not intuitive.

- In my view, using attack success rate (ASR) would be a better measure when trying to evaluate attacks. Using accuracy as the current version makes it difficult to interpret the results.

- How is it possible that the natural accuracies in Table 2 are 100%?

- Use a larger font size for the tables.


### Post-rebuttal Comments:
I want to sincerely thank the authors for their response.

As mentioned in my review, the proposed threat model makes no practical sense. The paper assumes that INRs instead of the query images are sent over the communication channels. Now, there are two major flaws with this threat model:

First, the data encoder has access to the target model (see the involvement of 
 in Eqs. (3) and (4)). If the user had access to the target model, why don't they just use that model to run their inference? It doesn't make sense that the user creates a robust INR with the full knowledge of the target model, sends the data through a medium which is susceptible to adversarial attacks, and then runs inference on the same target model! As can be seen from the transferability results of this method, it is not transferable between architectures at all.

Second, having access to the true target label can even be considered as a serious flaw in the threat model. Why when we have the true label, we need to run any inference at all? Reading through the authors' response on a comment on this matter, I am still unsatisfied with the paper's approach.

Considering these two points and the authors' response, I decided to lower my initial score and recommend rejecting this paper. I hope that the authors can address these points in the future versions of their paper.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
