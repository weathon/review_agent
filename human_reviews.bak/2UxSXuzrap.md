# Learning the Unlearnable: Adversarial Augmentations Suppress Unlearnable Example Attacks

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 6, 6, 1

## Abstract
Unlearnable example attacks are data poisoning techniques that can be used to safeguard public data against unauthorized use for training deep learning models. These methods add stealthy perturbations to the original image, thereby making it difficult for deep learning models to learn from these training data effectively. Current research suggests that adversarial training can, to a certain degree, mitigate the impact of unlearnable example attacks, while common data augmentation methods are not effective against such poisons. Adversarial training, however, demands considerable computational resources and can result in non-trivial accuracy loss. In this paper, we introduce the UEraser method, which outperforms current defenses against different types of state-of-the-art unlearnable example attacks through a combination of effective data augmentation policies and loss-maximizing adversarial augmentations. In stark contrast to the current SOTA adversarial training methods, UEraser uses adversarial augmentations, which extends beyond the confines of $\ell_p$ perturbation budget assumed by current unlearning attacks and defenses. It also helps to improve the model's generalization ability, thus protecting against accuracy loss. UEraser wipes out the unlearning effect with loss-maximizing adversarial augmentations, thus restoring trained model accuracies. Interestingly, UEraser-Lite, a fast variant without adversarial augmentations, is also highly effective in preserving clean accuracies. On challenging unlearnable CIFAR-10, CIFAR-100, SVHN, and ImageNet-subset datasets produced with various attacks, it achieves results that are comparable to those obtained during clean training. We also demonstrate its efficacy against possible adaptive attacks. Our code is open source and available to the deep learning community.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
Unlearnable example attacks are used to safeguard public data against unauthorized training of deep learning models. To break the data protection of unlearnable example methods, this paper introduces UEraser, an adversarial augmentation (PlasmaTransform, TrivialAugment and ChannelShuffle) based method to help model to learn semantic information from unlearnable example. Similarly to adversarial training, UEraser tries to find an augmentation to maximize the training loss rather than to find a perturbation to maximize the training loss (adversarial training does).

### Strengths
1. This paper focuses on the under-explored topic to learn semantic information from unlearnable examples, which is interesting and impressive.
2. This paper is organized logically and written clearly. The visualization results of different unlearnable example methods are impressive.
3. Impressive accuracy improvements on on four datasets (CIFAR-10, CIFAR-100, ImageNet, SVHN) compared with standard training.

### Weaknesses
1. The motivation and insight of this paper are not clear. As mentioned in Sec. 1 " These perturbations can form “shortcuts” [12, 16] in the training data to prevent training and thus make the data unlearnable in order to preserve privacy.", [12,16] does not reveal that unlearnable perturbations are shortcuts from models. Such two references are not suitable and convincing. Is there any observation or experiment to confirm that unlearnable perturbations are shortcuts from models? The same confusion occurs in Sec. 3.2 " Geirhos et al. [12] reveal that models tend to learn “shortcuts”, i.e., unintended features in the training images.". [12] only expounded that models prefer to learn shortcuts. Whether and why unlearnable perturbations are shortcuts from models is not clear at all.
2. The explanation of why UEraser performs better than adversarial training is self-contradictory. As mentioned in Sec. 3.2, this paper believed that UEraser augmentation policies more effectively preserve the original semantics in the image than adversarial training. However, adversarial training set a $\ell_{p}$ bound to constraint larger change on original input. The distance between adversarial training input and original input is much more small than the distance between UEraser augmented input and original input. In this case, whether UEraser augmentation policies more effectively preserve the original semantics in the image than adversarial training is quite doubtful. This paper should focus on analyzing why UEraser augmentation works and adversarial training does not. More insights should be proposed rather than engineering experiments.
3. As shown in Table 2, CutOut, CutMix does not work on unlearnable examples, but ChannelShuffle does. Not sure if all spatial transformation-based augmentation does not work and color transformation-based augmentation does? If so, why not filter out spatial transformation-based augmentation in TrivialAugment like shear and rotate to improve performance?
4. Clerical errors in Sec. 3.2. "Compared to UEraser, although UEraser-Lite may not perform as well as UEraser on most datasets, it is more practical than both UEraser-Lite and adversarial training due to its faster training speed." should be "Compared to UEraser, although UEraser-Lite may not perform as well as UEraser on most datasets, it is more practical than both UEraser and adversarial training due to its faster training speed."

### Questions
1. It is confused about the experimental setup. In Sec. 4 this paper resizes all images of ImageNet-subset to 32x32. However, as mentioned in Table 8, the input size of ImageNet-subset is 224x224x3. Which one is the real experimental setup? In addition, the operation of resize should not be implemented to ImageNet-subset, because the efficient of UEraser should be verified on high-resolution images perturbed by unlearnable methods, which are aligned with the experimental setup of EM and REM.
2. This paper repeats that UEraser-Lite has a fast training speed. How fast it is? I'd like to know the comparison results of execution time between UEraser (UEraser-Lite, UEraser, UEraser-Max) and adversarial training.
3. Why the augmentation of UEraser-Lite is Channleshuffle? Is there any other augmentation (equalize, posterize, plasmabrightness) that can achieve the same result as Channleshuffle?
4. Table 2 and Table 3 should add the results of UEraser trained model on clean data. Considering as an attacker, you have no idea whether the training is clean or not. The results of adversarial training and other unlearnable methods (AR, OPS, TAP, NTGA, HYPO) in Table 3 should be shown out.
5. There is no sensitivity analysis of hyperparameter W. How to pick a suitable value of W and K when deploying UEraser?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper considers the following adversarial poisoning problem: An adversary uses training data with "small perturbations (so in particular not hard to distinguish with normal training data" to tamper the training process so that the trained model will generalize poorly in the distribution sense.

The paper proposes a new "data augmentation" mechanism, which is solve objective (3) under various augmentations.

Quite some experiments are performed to demonstrate the effectiveness of the method.

### Strengths
Overall this paper is a nice read. There is a clear principle that guides the paper (objective (3)), and the results seem pretty solid. UEraser seems to work pretty well in all experiments.

It is also a bit surprising that "data augmentation" actually works. What is missing in previous work?

### Weaknesses
For one thing, the paper seems rather straightforward in principle, we try to solve the objective (3) under various augmentation methods. So as we expand the augmentations, we should get better results.

One thing is what is really the cost (which I don't think the paper has much discussion), do we need a tremendous amount of augmentation in order to make sure the learning does not learn the short cut? Also, what happens if the adversary is aware of the augmentations the training method is using? (so in that sense the paper needs to be more clear about what is the security model -- that is, what knowledge does the adversary has?)

### Questions
I don't have specific questions, my main concerns are described above.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper tackles unlearnable example attacks in deep learning, introducing "UEraser," a defense method utilizing adversarial data augmentations to neutralize these attacks. UEraser extends the perturbation budget beyond typical attacker assumptions, aiming to maintain model accuracy on poisoned data without significant loss on clean data. A faster variant, "UEraserLite," is also presented. The authors demonstrate UEraser's effectiveness across various state-of-the-art unlearnable example attacks, outperforming existing defenses and showing resilience against adaptive attacks.

### Strengths
- The paper introduces "UEraser," a novel defense against unlearnable example attacks using adversarial data augmentations. This approach is novel and extends the perturbation budget beyond typical attacker assumptions, showcasing a new direction in defending against these types of attacks.

- The authors have conducted extensive experiments to validate the effectiveness of UEraser against various state-of-the-art unlearnable example attacks. The results demonstrate that UEraser outperforms existing defense methods and is resilient against potential adaptive attacks, providing a strong empirical basis for the proposed approach.

- The introduction of "UEraserLite" offers a faster and more efficient alternative to UEraser, making the proposed defense more accessible and practical for real-world applications.

### Weaknesses
- While the paper presents a novel approach to defending against unlearnable example attacks, the contribution can be considered incremental. The use of adversarial training and data augmentation for defense is not entirely new, and the extension to unlearnable example attacks, while valuable, builds upon existing knowledge and techniques.

- The paper could benefit from exploring additional evaluation of UEraser and UEraserLite, particularly in terms of computational efficiency/time cost compared with existing defenses. 

- The paper primarily focuses on empirical results, and a more comprehensive theoretical analysis of why UEraser works and under what conditions it is most effective could strengthen the paper.

### Questions
See above.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
1: strong reject

### Rating Number
1

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
Unlearnable example attacks refer to data poisoning techniques used to safeguard public data against unauthorized training of deep learning models. These methods introduce subtle perturbations to the original images, making it challenging for deep learning models to effectively learn from such training data. Current research indicates that adversarial training can partially mitigate the impact of unlearnable example attacks, while common data augmentation methods are ineffective against these poisons. However, adversarial training is computationally demanding and can lead to significant accuracy loss.

This paper presents the UEraser method, which surpasses existing defenses against various state-of-the-art unlearnable example attacks. UEraser achieves this through a combination of effective data augmentation techniques and loss-maximizing adversarial augmentations. Unlike current state-of-the-art adversarial training methods, UEraser employs adversarial augmentations that extend beyond the assumed 'p perturbation budget' used by current unlearning attacks and defenses. This approach enhances the model's generalization ability, thereby safeguarding against accuracy loss. UEraser effectively eliminates the unlearning impact through loss-maximizing data augmentations, restoring trained model accuracies.  On challenging unlearnable datasets like CIFAR-10, CIFAR-100, SVHN, and ImageNet-subset, generated using various attacks, UEraser achieves good results.

### Strengths
1. This paper highlights that unlearnable examples can be mitigated through various data augmentation techniques, potentially leading to the generation of more resilient unlearnable examples when facing adaptive poisoning.

### Weaknesses
1. After carefully conducting experiments of the UEraser using the official code provided by the authors, a notable disparity emerged in comparison to the reported results in the paper. On the CIFAR-10 datasets, specifically, for EM poisons, the best accuracy achieved during training was 74.46\%, while the accuracy in the final epoch was 68.46% (25\% lower than reported). In the case of LSP poisons, the best training accuracy reached 90.23\%, with a final epoch accuracy of 73.43\% (12\% lower than reported). This observation indicates that although the UEraser appeared effective during mid-training, the models eventually converged to shortcuts present in the unlearnable examples. Given that there is often no clean validation dataset available, many papers on unlearnable examples (UE) typically report the accuracy achieved in the last training epoch. Furthermore, with a fixed learning rate of 0.01 throughout the training process, I've observed that the accuracy of models trained on clean images struggles to converge to a satisfactory level, reaching approximately 92% on CIFAR-10 (2% lower than reported) and around 70% on CIFAR-100 (4% lower than reported) in my experiments. Therefore, it is recommended that the authors thoroughly review their results and consider reporting the accuracy in the last epoch for a more equitable comparison.

2. In the experiments conducted with the UEraser-Max approach, the performance on the EM and LSP attacks shows a gap compared to the reported results on the CIFAR-10 dataset (EM: 62.25\%, 33\% lower than reported; LSP: 88.33\%, 6\% lower than reported). Interestingly, the use of a fixed learning rate of 0.01 (too large to converge to the optimal model when conducting UEraser-Max) is not likely to reach an accuracy of 95.24% for EM poisons, which is even 0.5% higher than what is typically achieved in standard training on **clean** CIFAR-10.

2. It's essential to consider the standard evaluation settings and practices in the field. The results on ImageNet-subset are not convincing, as most operations on ImageNet-subset in papers related with Unlearnable Examples do not resize the images to $32 \times 32$. Instead, they follows the default data augmentations, and images are resized to $224 \times 224$ during training, it would be advisable to align with these practices in your experiments for better comparability with previous works. For your reference, usually, the clean performance on the ImageNet-subset should be around 80\%, and the performance of ISS facing several UE methods is around 55\%.

3. Training on CIFAR-10, CIFAR-100, and SVHN datasets takes a similar amount of time. To provide a more comprehensive evaluation of the proposed method's performance and robustness, it would be valuable to expand the experimental analysis to include CIFAR-100 and SVHN, similar to the approach taken for CIFAR-10. This extended evaluation would help assess how well the model generalizes across different datasets and under various UE attack methods.

4. The proposed methods do not exhibit robustness to adaptive poisoning. Table 4 demonstrates that when faced with adaptive poisoning (UEraser-Max), the defensive performance experiences a significant drop, ranging from 15% to 30%. To facilitate a fair comparison, it would be beneficial to report the performance of adaptive poisoning when facing ISS.

5. The concept presented in this work is not particularly innovative, and the achieved performance heavily relies on empirical augmentations for defense, which can be significantly undermined by adaptive poisoning.

### Questions
See weaknesses above.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
