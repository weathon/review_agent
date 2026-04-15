# Adversarial Guided Diffusion Models for Adversarial Purification

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3, 3, 3

## Abstract
Diffusion model (DM) based adversarial purification (AP) has proven to be a powerful defense method that can remove adversarial perturbations and generate a purified example without threats. In principle, the pre-trained DMs can only ensure that purified examples conform to the same distribution of the training data, but it may inadvertently compromise the semantic information of input examples, leading to misclassification of purified examples. Recent advancements introduce guided diffusion techniques to preserve semantic information while removing the perturbations. However, these guidances often rely on distance measures between purified examples and diffused examples, which can also preserve perturbations in purified examples. To further unleash the robustness power of DM-based AP, we propose an adversarial guided diffusion model (AGDM) by introducing a novel adversarial guidance that contains sufficient semantic information but does not explicitly involve adversarial perturbations. The guidance is modeled by an auxiliary neural network obtained with adversarial training, considering the distance in the latent representations rather than at the pixel-level values. Extensive experiments are conducted on CIFAR-10, CIFAR-100 and ImageNet to demonstrate that our method is effective for simultaneously maintaining semantic information and removing the adversarial perturbations. In addition, comprehensive comparisons show that our method significantly enhances the robustness of existing DM-based AP, with an average robust accuracy improved by up to 7.30% on CIFAR-10. The code will be available upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper proposed a diffusion model-based adversarial purification (AP) method called adversarial guided diffusion models (AGDM). AP methods face a trade-off between robust and standard accuracy since we have to drop some label semantics to preserve from recovering the adversarial perturbation during the reverse process. 

To overcome this, training-free conditional-based diffusion has been introduced into AP, where we can measure the distance between the intermediate results and the known adversarial samples to keep the label semantic. However, the trade-off still exists since the distance metric will recover the adversarial perturbation again. 

AGDM solves this challenge by introducing an auxiliary neural network pre-trained via an adversarial training paradigm. This auxiliary neural network could be regarded as a feature extractor that maps the images into the latent space. AGDM further argues that the adversarial training paradigm will help auxiliary neural networks extract the latent that does not suffer from the influence of adversarial perturbation. In this condition, we could use the clean latent feature as the condition to guide the generation process, thus alleviating the trade-off. Meanwhile, auxiliary neural networks could offer logit to enrich the condition, further enhancing the AP.

### Strengths
1. The overall method is technically sound. AGDM proposes to train an auxiliary classifier via adversarial training, which is sound. Adversarial training could help the classifier recognize the adversarial sample, which could naturally generate the most robust latent features to defend against the adversarial perturbation. Leveraging this, the overall conditional generation process will be more robust against the adversarial perturbation and alleviate the trade-off.

2. Introducing adversarial training to enhance AP is interesting. 

3. The experiments show that AGDM achieves the SOTA performance.

### Weaknesses
1. The contributions of this paper are limited. One of the main contributions of this paper is how to calculate the adversarial guidance, which is well explored under the training-free conditional diffusion such as FreeDom. In the view of the FreeDom, it is just a multi-conditional guidance, which could be easy to calculate. 

Specifically, the adversarial guidance in Sec. Methods contains two parts: 1) The MSE in the latent space between the intermediate results $x_{t}$ and the adversarial samples$x^{adv}$. 2) A logit of $p_{\phi}(x_{t})$, where $p_{\phi}$ is the classifier from auxiliary neural networks. This could be conducted in the FreeDom as $\nabla_{x_{t}}\log p ({c^{1},c^{2}}|x_{t})$. 
Then,
$\nabla_{x_{t}}\log p ({c^{1},c^{2}}|x_{t}) = \nabla_{x_{t}}\log p ({c^{1}}|x_{t})  + \nabla_{x_{t}}\log p ({c^{2}}|x_{t}) $, where
$ \nabla_{x_{t}}\log p ({c^{1}}|x_{t}) =p_{\phi}(x_{t}) $ and $\nabla_{x_{t}}\log p ({c^{2}}|x_{t}) = D(c_{\phi}(x_{t}) - c_{\phi}(x^{adv}))$. $c_{\phi}$ is the feature extracted from the auxiliary neural networks given the parameter $\phi$. In this condition, we could achieve the conditional generation based on $\nabla_{x_{t}}\log p ({c^{1},c^{2}}|x_{t})$ i,e, the adversarial guidance. This process is even simpler than Sec. Methods, which weakens one of the main contributions of this paper.

The other main contribution of this paper is introducing an auxiliary classifier via adversarial training. However, it directly leverages the TRADES method without any new insight. This raises a new question: Does AGDM really alleviate the trade-off for AP? The advantage of the AP compared to the adversarial training is to defend against unseen attacks. After introducing adversarial training (AT), how to ensure that AGDM can defend against unseen attacks? The weakness of AT is that it is difficult to defend against unseen attacks since there are no training samples from unseen attacks in AT. In this condition, AGDM generates a new trade-off between AP and AT again.

To sum up, the contribution of this paper seems to verify that an auxiliary classifier via AT could enhance the AP in some conditions, which seems limited.

2. The experiments are not enough to prove the superiority of the AGDM. (1). It lacks the BPDA [2] attacks. BPDA is one of the important attacks to test the performance of AP, which should be discussed. (2) The performance of AGDM is fair. ZeroPur [3] reports the robust accuracy of AutoAttack on CIFAR-10 ($\epsilon=8/255$) is 82.76%, better than the 78.12% reported in AGDM. Meanwhile, the AP for ZeroPur even drops the diffusion models. AGDM introduces the unconditional diffusion models and an auxiliary classifier, which should be better than the ZeroPur, since there is richer previous knowledge for AGDM.

[1] FreeDoM: Training-Free Energy-Guided Conditional Diffusion Model. Yu, Jiwen and Wang, Yinhuai and Zhao, Chen and Ghanem, Bernard and Zhang, Jian. ICCV. 

[2] Diffusion Models for Adversarial Purification. Nie, Weili and Guo, Brandon and Huang, Yujia and Xiao, Chaowei and Vahdat, Arash and Anandkumar, Anima. ICML 

[3] ZeroPur: Succinct Training-Free Adversarial Purification. Xiuli Bi and Zonglin Yang and Bo Liu and Xiaodong Cun and Chi-Man Pun and Pietro Lio and Bin Xiao. ArXiv.

### Questions
1. How does different AT influence the performance of AGDM? The motivation for this question is from the Weaknesses. 1. AT will inevitably introduce additional problems. For example, if we use PGD to generate training samples for the auxiliary classifier via AT, the classifier's performance will be worse when facing C\&W attack. Is there a way to make the auxiliary classifier better defend against unseen attacks?

2. How does the influence of $s$. For example, could we use the different weights for the MSE and the logit $p_{\phi}(x_{t})$, or could we increase $s$? The motivation for this question is that the training-free conditional generation method relies on the setting of the $s$. Thus, we have to discuss its influence.

To sum up, my main concerns are listed in Sec. Weaknesses and Sec. Questions. The overall contribution of this paper seems limited, and the experimental is not enough. Considering introducing the AT to enhance the AP is an interesting topic, I rate it as "marginally below the acceptance threshold".

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper introduces an improved diffusion-based adversarial purification (AP) method termed adversarial guided diffusion model (AGDM). To address the limitation of existing diffusion-based AP methods that the lack of proper guidance can lead to shifting towards incorrect classes, AGDM utilizes a robust auxiliary neural network obtained by TRADES to guide the diffusion model in AP. Experimental results suggest that AGDM can improve the robust accuracy compared with the diffusion-based AP baselines.

### Strengths
- The limitations of existing related methods are thoroughly discussed, and the proposed AGDM is well-motivated.
- The superiority of AGDM to existing diffusion-based AP methods indicates the significance of guidance for the diffusion model in AP.

### Weaknesses
- The notations and interpretations in Section 3.2 can be confusing. Specifically, the interpretation of $p_{\phi}(x' \mid x_t)$ (Lines 212-213) only concerns the semantic information of $x'$, but the notation itself seems to indicate that the specific pixel values of $x'$ are also concerned. If only the semantic information is considered, it should be something like $p_{\phi}(s(x') \mid x_t)$.
- In Lines 277-278, it is stated that the auxiliary network is not required to be a robust classifier but to have adversarially robust representations. From my perspective, the difference between the two requirements is that the latter does not consider the classification ability of the network (e.g., it can be a self-supervised model), but it seems that a non-classification model may not suffice for the guidance of AGDM.
- The architecture of the auxiliary network (or whether it is the same as the classifier) is not stated in the experimental settings.
- The number of PGD and EOT steps is not stated. Insufficient PGD and EOT iterations can lead to the overestimation of the robustness of AP methods.
- Judging from Figure 3(a), the robust accuracy of the AGDM is not significantly higher than that of AT methods under $\ell_{\infty}$ attacks. It might be the case that using state-of-the-art AT models as the auxiliary network may further improve the performance of AGDM, but no evidence is provided.
- The practicality of the proposed AGDM may be questionable. High computation costs are required for both training (AT of the auxiliary network) and inference (iterative denoising process for purification). It is argued that the adversarial fine-tuning of DMs in AToP is computationally expensive (Lines 079-080), but the proposed method also suffers from the same issue. There is also a lack of evidence for the transferability of a pre-trained AGDM to models of other architectures or for different tasks, which may indicate the practical value of the AP method.

### Questions
- In Lines 209-210, why can we assume that $y$ and $x'$ are conditionally independent given $x_t$?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This work proposes an adversarial purification method for defending against adversarial examples. The proposed model uses a robust classifier to guide the reverse process of DMs during purification, helping to preserve semantic information and improve robustness against adversarial perturbations. Experiments conducted on several datasets demonstrated the effectiveness.

### Strengths
1. The proposed method seems to be reasonable. Combining the robust classifier with diffusion models has the potential to improve the robustness.

2. The experiments are relatively comprehensive, with several datasets and several attack methods included.

### Weaknesses
1. The tricky illustrations. The diffusion step t is set to be 70 in the experiments, while in Fig 1, the step is set to 400. I recommend using the actual step for illustration to help readers comprehensively understand this work. Furthermore, How to create Figure 2 is not clear and there is no experimental support for Figure 2.
2. There is no theoretical analysis to show the reason why this process is better than other classifier-guided diffusion purification methods.
3. Lack of innovation and contribution. The core contribution of this work is to replace the standard classifier with a robust classifier in the framework of guided diffusion models [1], which do not bring a new perspective on diffusion-based purification. Therefore, the innovation is questionable.
4. The results  are not convincing. According to [2], the robustness of diffusion-based purification is significantly over-estimated and the robustness should be reported under adaptive attacks. I recommend all results in this work be reported following the settings in [2].
5. The chaotic formulations. Please follow the symbolics in DDPM or Score-SDE. For example, $\mathbf{x}$ instead of $x$ should be used to represent an image.

[1] Guided diffusion model for adversarial purification

[2] Robust Classification via a Single Diffusion Model

### Questions
Please see the weaknesses above.

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
Existing DM-based AP methods have no explicit guidance or improper guidance (e.g., existing guidances may also preserve adversarial perturbations in purified examples). To solve this issue, this paper proposes AGDM, which adversarially trains an auxiliary neural network to provide more robust guidance in the reverse process of DMs. Experiments show that AGDM can improve robustness by a notable margin.

### Strengths
1. The motivation of the paper is very clear, which effectively delivers the main insight of this paper.

2. The proposed method is intuitive and easy to understand.

3. The guided sampling can be extended to continuous-time DMs,  which means AGDM can be generalized to different DMs.

### Weaknesses
1. The proposed method may have very low efficiency. DM-based AP method is very slow during the inference stage (as it is a completely 'inference-time defense'). Based on this, this paper proposes to train an auxiliary neural network using AT, which will further increase the computational complexity of both the training (as AT is very slow by its nature) and the inference (as adversarial guidance is introduced to the reverse process of DMs). 

2. A followed-up weakness is: this paper did not report the training time for the auxiliary neural network and the inference time for the entire defense. Therefore, it is unclear whether the improvement obtained by AGDM is worthwhile in terms of the sacrifice in efficiency. I hope the authors could clarify this during the rebuttal.

3. The experiment settings are very unclear. Firstly, what are the seed numbers for those 512 images during the evaluation, are they consistent with DiffPure (as this paper said following DiffPure in line 327)? Secondly, what are the iteration numbers for PGD+EOT and AutoAttack, are they the same across all the baseline methods? Thirdly, it is unclear whether the adversarial examples for evaluations are generated under a white-box setting (i.e., attack AGDM + classifier as a whole) or a grey-box setting (i.e., only attack the classifier, or only attack the vanilla DMs). A fair evaluation process is very important in this field, so I would strongly encourage the authors to include detailed experiment settings during the rebuttal.

4. This paper lacks of ablation studies on the auxiliary neural network. What's the architecture of the auxiliary neural network? How the performance would be affected if other architectures are used?

### Questions
Most questions that I hope the authors can address are given in the weaknesses, and here are some additional questions:

1. Just for curiosity, given that DM-based AP methods and AT methods are completely different, what's the motivation for comparing DM-based AP methods with AT methods here? I can see most experiments follow what was done in DiffPure (except the PGD+EOT experiments), but DiffPure compared to AT methods because it is the first DM-based AP method and thus it cannot find another DM-based AP method to compare with. However, after a few years, now there are many DM-based AP methods in this field and I think comparing to DM-based AP methods is enough to demonstrate the effectiveness of AGDM. I hope authors could share their ideas about this question during the rebuttal.

2. A follow-up question is: if it is necessary to compare with AT methods, what do you think is the most fair way to do so? According to [1], AT methods perform worse on AutoAttack while DM-based AP methods perform worse on PGD+EOT. So a natural question is: how can we compare DM-based AP methods with AT methods in a fair setting?

[1] Minjong Lee and Dongwoo Kim. Robust evaluation of diffusion-based adversarial purification. In ICCV 2023.

I am willing to increase my rating if the authors can address my concerns during the rebuttal. Also, if I misunderstood any part of the paper, feel free to correct me.

### Soundness
2

### Presentation
3

### Contribution
2
