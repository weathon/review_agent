# ADBM: Adversarial Diffusion Bridge Model for Reliable Adversarial Purification

- Decision: Accept (Poster)
- Scores: 8, 6, 5

## Abstract
Recently Diffusion-based Purification (DiffPure) has been recognized as an effective defense method against adversarial examples. However, we find DiffPure which directly employs the original pre-trained diffusion models for adversarial purification, to be suboptimal. This is due to an inherent trade-off between noise purification performance and data recovery quality. Additionally, the reliability of existing evaluations for DiffPure is questionable, as they rely on weak adaptive attacks. In this work, we propose a novel Adversarial Diffusion Bridge Model, termed ADBM. ADBM directly constructs a reverse bridge from the diffused adversarial data back to its original clean examples, enhancing the purification capabilities of the original diffusion models. Through theoretical analysis and experimental validation across various scenarios, ADBM has proven to be a superior and robust defense mechanism, offering significant promise for practical applications. Code is available at https://github.com/LixiaoTHU/ADBM.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
Adversarial attack is a challenge for the neural network. This paper proposed a diffusion model-based adversarial purification (AP) to defend against adversarial attacks via adversarial training (AT) called ADBM. ADBM alleviates the trade-off between noise purification efficiency and recovery quality by proposing a new loss function to fine-tune the diffusion model via the AT paradigm. ADBM also offers detailed proofs to ensure the adversarial perturbation will be gradually removed. The experimental results show that ADBM achieves the SOTA performance.

### Strengths
1. This paper proposed a new adaptive pipeline based on the diffusion model's full gradient, which can further ensure the diffusion-based AP's robustness. The robustness of the diffusion-based AP is always questionable since it is a computation cost to get the full gradient of the diffusion model. Previous works proposed various gradient estimation methods, such as the adjoint method from DiffPure and the surrogate process from Lee. et al. [1], which remains a question: Is the improvement of such diffusion-based AP due to the estimation error of the gradient? Lee. et al. also prove this question. This paper is the first work to evaluate the robustness of diffusion-based AP by the full gradient to the best of my knowledge. I think this will benefit further study of diffusion-based AP.

2. The AT paradigm for diffusion-based AP proposed in this paper is novel and robust. To establish the loss function for AT, this paper chooses to connect the adversarial samples and the generation target $x_{0}$, which is interesting. Then, the detailed proofs verify that the loss function could ensure the removal of the adversarial perturbation after various times of the reverse process.

3. The experimental results show that ADBM achieves the SOTA performance. Specifically, the experiment also includes unseen attacks to illustrate how the proposed AT paradigm could work on unseen threats.  

[1] Minjong Lee and Dongwoo Kim. Robust evaluation of diffusion-based adversarial purification. In Int. Conf.
Comput. Vis. (ICCV), 2023

### Weaknesses
1. The writing for this paper should be improved. 1) Some math symbols are confused. For example $\hat{x}\_0$. It first indicated in 299 lines that $\hat{x}\_0$ should be the final point of the reverse process, thus having $\hat{x}\_0=x\_{0}$. But, in 879 lines, $\hat{x}\_0$ changed to the prediction of $x_{0}$ for $x_{t}$, thus causing the confusion. There are various similar mistakes, such as $x_{t}^{d}$ in 265 lines.  2) One-step DDIM is confusing since this paper includes the discrete and continuous VP-SDE. Setting $s=1$ is truly the one-step DDIM based on the discrete VP-SDE but is not one-step for continuous VP-SDE since the time sequence of continuous VP-SDE is in $[0.0, 1.0]$. In this condition, $s=1$ means the entire reverse process for continuous VP-SDE instead of the one-step. The authors should explicitly clarify which formulation (discrete or continuous) they use when discussing one-step DDIM and explain the implications for both cases.

2. The experimental results may not be enough. 1) The BPDA [1] should be considered to test the performance of the ADBM since it is a special attack designed for the AP.  Adding additional experiments on CIFAR-10 against BPDA ($\ell_{\infty}$,$\epsilon = 8/255$) will further strength the contribution. 2) AToP should be considered as the baseline, even if only making an ablation study on CIFAR-10 since ADBM is similar to AToP. The only difference is that AToP may be suitable for any generative model, and ADBM is for the diffusion model only.  Adding additional experiments on CIFAR-10 against PGD+EOT ($\ell_{\infty}$,$\epsilon = 8/255$, EOT=20, steps=200) to compare the performance of ADBM and AToP (the generative model chooses GAN, which is the best reported in AToP) will enhance the contribution for this paper. Additionally, adding an ablation study between ADBM and AToP (the generative model chooses VP-SDE) against PGD+EOT ($\ell_{\infty}$,$\epsilon = 8/255$, EOT=20, steps=200) will further show the validity of ADBM.

[1] Minjong Lee and Dongwoo Kim. Robust evaluation of diffusion-based adversarial purification. In Int. Conf.
Comput. Vis. (ICCV), 2023

### Questions
1. Does ADBM only work for the DDIM sampler? The motivation for this question is that the assumption for the proofs is the DDIM sampler. What happens if we change the DDIM sampler to the original DDPM sampler?

To sum up, there are two main contributions for ADBM: 1) proposed a full-gradient adaptive attack method. 2) proposed an interesting AT paradigm to enhance AP based on the unconditional diffusion model. All my concerns are listed in Weaknesses and Questions. Considering 1) the lacking the necessary baseline and attack method and 2) the writing problem, I rate it as "marginally above the acceptance threshold." If the author could solve these concerns, I'm willing to increase my rate.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes a new method for Diffusion-based Purification, which is named as Adversarial Diffusion Bridge Model (ADBM). Different from previous methods, ADBM directly constructs a reverse bridge from the diffused adversarial data back to its original clean examples. Experiments show the effectiveness of the proposed method.

### Strengths
The proposed ADBM has the complete theoretical support, and the experiments on the CIFAR10 demonstrate the effectiveness of the proposed method on some chosen datasets.

### Weaknesses
1. The proposed method is only verified on the datasets with small classification number, such as CIFAR10 and SVHN. The experiments on datasets with small sizes can not fully demonstrate the effects of the proposed strategy, since the applications in the real world will have large number of classification number. A more suitable dataset could be ImageNet with at least 1K class number. I think the complicated data distribution will largely influence the performance of diffusion models.

2. The experimental results are mainly built with the network of WRN, and the experimental results with more types of networks are not shown.

3. Compared with other methods, the accuracy on the adversarial examples are not improved a lot, while the accuracy on the clean examples are decreased clearly. The authors should analyze the factor which limits the accuracy on the clean examples.

### Questions
1. What is the performance on more datasets with more classification numbers?

2. What is the performance of the proposed method with various networks?

3. What is the performance of the proposed method towards different types of attack?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper trains a purifier based on the theoretical framework of a diffusion model, making it better suited for adversarial purification tasks.

### Strengths
- Introduces a comprehensive training process to adapt the diffusion model for adversarial purification tasks.
- Theoretically and empirically demonstrates superior performance compared to DiffPure.

### Weaknesses
- Unlike DiffPure, the Adversarial Diffusion Model (ADBM) requires reasoning about adversarial perturbations for purification, which may **limit its generalizability to unseen attacks**, similar to challenges faced by adversarial training. For instance, the experimental results in Table 3—beyond different settings of white-box attacks, how does it perform against Square attacks or StAdv?
- The author mentions the Robust Diffusion Classifier (RDC) method in the paper but **does not compare it with the purification component of RDC, namely Likelihood Maximization (LM)**. To my knowledge, RDC’s LM can handle unseen attacks better than ADBM without needing adversarial perturbation training.
- In the original diffusion model framework, as the time step approaches infinity, $x_T$ converges to a standard Gaussian distribution. With ADBM losing this property, there may need to be corresponding adjustments in the selection of variables such as $\bar{\alpha}$.

### Questions
please see weaknesses

### Soundness
2

### Presentation
3

### Contribution
2
