# Adversarial Training on Purification (AToP): Advancing Both Robustness and Generalization

- Avg Score: 5.67
- Decision: Accept (poster)
- Scores: 6, 5, 6

## Abstract
The deep neural networks are known to be vulnerable to well-designed adversarial attacks. The most successful defense technique based on adversarial training (AT) can achieve optimal robustness against particular attacks but cannot generalize well to unseen attacks. Another effective defense technique based on adversarial purification (AP) can enhance generalization but cannot achieve optimal robustness. Meanwhile, both methods share one common limitation on the degraded standard accuracy. To mitigate these issues, we propose a novel pipeline to acquire the robust purifier model, named Adversarial Training on Purification (AToP), which comprises two components: perturbation destruction by random transforms (RT) and purifier model fine-tuned (FT) by adversarial loss. RT is essential to avoid overlearning to known attacks, resulting in the robustness generalization to unseen attacks, and FT is essential for the improvement of robustness. 
To evaluate our method in an efficient and scalable way, we conduct extensive experiments on CIFAR-10, CIFAR-100, and ImageNette to demonstrate that our method achieves optimal robustness and exhibits generalization ability against unseen attacks.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper introduces an adversarial defense technique called Adversarial Training on Purification (AToP). AToP conceptualizes AP into two components: perturbation destruction by random transforms and fine-tu . ning the purifier model using supervised AT. The random transforms aim to enhance the robustness to unseen attacks, while the purifier model reconstructs clean data from corrupted data, ensuring correct classification output regardless of the corruption. The purifier model is fine-tuned through an adversarial loss calculated from the classifier output. The paper presents experiments on CIFAR and ImageNette datasets, comparing AToP with other AT and AP methods. The results demonstrate that AToP achieves good generalization ability against unseen attacks, and maintains standard accuracy on clean examples.

### Strengths
- The paper addresses a significant problem in the field of deep learning, namely the vulnerability of neural networks to adversarial attacks.
- The proposed AToP technique offers a promising solution that combines the advantages of AT and AP, achieving good robustness, standard accuracy, and robust generalization to unseen attacks.
- The proposed AToP framework is well-defined, with a clear explanation of its components and their roles in achieving optimal robustness, standard accuracy, and robust generalization to unseen attacks.  
- The experiments are conducted on CIFAR and ImageNette dataset 
- I appreciate the evaluation with BDPA   
- The paper is well-structured, with a clear and concise presentation of ideas.

### Weaknesses
- Overall the technical novelty of this work is limited. In essence, AToP combines the orthogonal techniques of AT and AP. 
- The authors mainly evaluate ResNet and Wide-ResNet architecture. Hence it is not clear if this work generalizes to other model architectures. 
- These days, transformer architectures are gaining popularity. How does the proposed AToP perform on such architectures, such as ViT?
- While I appreciate the evaluation on BPDA, the authors should include a separate section on “obfuscated gradients”, where they follow the guidelines of BPDA and [A] to test for obfuscated gradients.  

[B] On Evaluating Adversarial Robustness; arXiv 2019

### Questions
Please address the points in my weaknesses section.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
1. This paper aims to enhance the adversarial robustness of deep models.

2. Combining adversarial training and purification techniques, a new pipeline is proposed, i.e., adversarial training on purification.

3. Different from purification methods that purify adversarial examples with a pre-trained generative model before classification, 
the generative model can be finetuned by adversarial training in the proposed pipeline. Moreover, the introduce random transformations play an important role in defensing adversarial attacks.

### Strengths
1. The paper is written clearly and easy to follow.    
2. Models trained by the proposed method show promising robustness and generalization.

### Weaknesses
1. What if there are no random transformations in the proposed pipeline?
    Does the robustness come from the gradient vanishing from random operations?

2. For adversarial training methods, auto-attack is one of the most effective attack methods to evaluate model robustness. This is because models with adversarial training usually show much better results under FGSM, PGD, and other attacks than under the auto-attack. 

    However, as indicated by Table 7 and Table 8 in this paper, the trained model even achieves much better robustness under auto-attack than under PGD20, FGSM attacks.
    From this perspective, the comparisons in Tables 2,3,4,5 can be unfair because models with adversarial training show worst-case robustness under auto-attack while it is not the case for the model trained by the proposed method.

3. Considering the issue in 2, to demonstrate their robustness, the models trained by the proposed methods should be fully tested under different attacks. For example, the CW attack with a large number of steps, and the PGD attack with 1000 steps.

### Questions
See above weaknesses.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes Adversarial Training on Purification (AToP) for adversarial defense, which is expected to improve the robustness while achieving good generalization ability. The proposed method consists of two components: perturbation destruction by random transformations and a purifier model that aims to recover the clean image which is adversarially finetuned. The experiments suggest that AToP can improve the robustness against both seen and unseen attacks without significantly sacrificing the accuracy on clean images, and achieve state-of-the-art results on several benchmarks.

### Strengths
- This paper makes an early attempt to finetune an adversarial purification model with adversarial training, and suggests that this can be beneficial.
- The experiments are complete and the results seem promising.
- The figures are clear and meaningful.

### Weaknesses
- Clarity of Section 3.2. 
  - Eq. (8) seems to be a mix of the loss functions for the generator (purifier) $g$ and the discriminator $D$, which are supposed to be different in GAN training. Besides, since AToP is not limited to GAN-based purifier models, it is better to present a general form of the loss function.
  - It is stated that "the discriminator $D$ is responsible for distinguishing between real examples (*including clean or adversarial examples*) and the purified examples", but this is not reflected by Eq. (8) since only the clean sample $x$ is involved in $L_{df}$.
  - Line 5 of Algorithm 1 is confusing. The notation $g_{\theta_g}$ is not explained.
- Some experimental details are not clearly stated.
  - For evaluation, it is claimed that "we randomly select 512 images from the test set for robust evaluation." It is not clear whether this applies to all results or all compared methods. The comparison may be unfair if different models are evaluated on different test sets.
  - The "Ours" in Table 2,3,4 are not specified: Which RT and purifier model are used? What are the hyper-parameters for the training attack?
- The results for AToP in Table 8 may be unreasonable. Specifically, considering the accuracy of the same model against different attacks, it is shown that FGSM < PGD < AutoAttack, which is exactly the reverse of common expectations since AutoAttack is believed to be a much stronger attack while FGSM should be the weakest attack. These results may significantly challenge the reliability of the robustness evaluation and should be seriously discussed.

### Questions
- Why FGSM is used for adversarial finetuning instead of PGD or other attacks?

- Can the reconstruction loss in Eq. (8) be applied to the reconstruction of adversarial images? Specifically, is it likely that adding $\mathbb{E}[\Vert x-g(t(x+\delta),\theta_g) \Vert]$ to the loss will increase the performance?

- Why does reconstruction loss take $\ell_1$-distance?

- What is the expected scope of purification models that AToP can generalize to?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good
