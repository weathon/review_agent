# Scalable Energy-Based Models via Adversarial Training: Unifying Discrimination and Generation

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 2

## Abstract
Simultaneously achieving robust classification and high-fidelity generative modeling within a single framework presents a significant challenge. Hybrid approaches, such as Joint Energy-Based Models (JEM), interpret classifiers as EBMs but are often limited by the instability and poor sample quality inherent in training based on Stochastic Gradient Langevin Dynamics (SGLD). We address these limitations by proposing a novel training framework that integrates adversarial training (AT) principles for both discriminative robustness and stable generative learning. The proposed method introduces three key innovations: (1) the replacement of SGLD-based JEM learning with a stable, AT-based approach that optimizes the energy function through a Binary Cross-Entropy (BCE) loss that discriminates between real data and contrastive samples generated via Projected Gradient Descent (PGD); (2) adversarial training for the discriminative component that enhances classification robustness while implicitly providing the gradient regularization needed for stable EBM training; and (3) a two-stage training strategy that addresses normalization-related instabilities and enables leveraging pretrained robust classifiers, generalizing effectively across architectures.   Experiments on CIFAR-10/100 and ImageNet demonstrate that our approach: (1) is the first EBM-based hybrid to scale to high-resolution datasets with high training stability, simultaneously achieving state-of-the-art discriminative and generative performance on ImageNet 256$\times$256; (2) uniquely combines generative quality with adversarial robustness, enabling faithful counterfactual explanations; and (3) functions as a competitive standalone generative model, matching autoregressive models and surpassing diffusion models while offering additional versatility.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents DAT (Dual Adversarial Training), a framework that integrates adversarial training into Joint Energy-Based Models (JEMs) to achieve both robust classification and high-quality image generation. The method replaces unstable SGLD-based learning with an adversarial optimization using PGD-generated samples and introduces a two-stage procedure to handle batch normalization issues. Experiments on CIFAR-10, CIFAR-100, and ImageNet show competitive robustness (similar to standard AT) and improved generative quality (FID 5.39 on ImageNet). The motivation is to bridge discriminative and generative modeling, combining the accuracy of classifiers with the data awareness of generative models. While the formulation is clean and the results promising, the practical benefits of unifying these two objectives remain somewhat unclear.

### Strengths
The paper addresses a well-known instability in JEM training through an elegant adversarial reformulation that improves both convergence and visual quality. The proposed method removes the need for gradient penalties and achieves stable training even on ImageNet, which is impressive for EBMs. The experiments are extensive and reproducible, comparing DAT against strong hybrid and adversarial baselines. Quantitatively, DAT outperforms RATIO and JEM in both robustness and FID, and qualitatively the generated samples show fewer artifacts. Overall, this is a careful and technically strong piece of work that meaningfully advances the robustness and scalability of energy-based hybrids.

### Weaknesses
The paper is technically competent and experimentally thorough, but the contribution feels incremental relative to prior hybrid frameworks. While the work pushes JEMs toward stability and competitive generative quality, it does not yet offer a compelling argument for hybrid modeling over modern specialized alternatives such as more recent, 2024 and beyond, diffusion models and GAN variants. Without stronger empirical evidence of unique practical benefits, the paper’s impact may be limited.

### Questions
How does it compare in efficiency and scalability to modern diffusion and GAN models?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose a hybrid framework that aims to unify discriminative robustness and generative modeling within a single network. The method replaces the JEM-style SGLD negative sampling with adversarially generated negatives (PGD) and trains the energy with a binary classification loss, while the classifier head is trained with standard adversarial training (AT). Experiments on CIFAR-10/100 and ImageNet report improved adversarial robustness relative to hybrid baselines and competitive (sometimes strong) generative quality.

### Strengths
1.	Replacing SGLD with PGD negatives makes the training loop simple and scalable. The two-stage BN recipe is an effective engineering fix to a well-known incompatibility in EBM-style training.
2.	The dual view (AT for the classifier and contrastive/AT-style learning for energies) offers a single model that can produce robust predictions, counterfactuals, and samples.
3.	Results are reported up to ImageNet, suggesting attention to large-scale feasibility.
4.	Public code (if complete and reproducible) increases practical impact and adoption.

### Weaknesses
1.	The conceptual novelty is limited. The core idea (learn energies with adversarial negatives and train the classifier adversarially) has clear antecedents in AT-EBM/CEM/JEM++/Robust-JEM[1] lines of work that (i) use contrastive or adversarially produced negatives to shape an energy landscape, (ii) draw formal links between AT objectives and energy-based views, (iii) report stability benefits relative to SGLD, and (iv) An empirical study on AT improving JEM/JEM++.
2.	The BCE-with-PGD objective likely corresponds to a contrastive density-ratio under a local worst-case neighborhood. It remains unclear what distribution is being learned and how/when this departs from MLE-style JEM; a formal treatment would strengthen the paper.
3.	The paper does not provide convincing diagnostics that AT stabilizes SGLD-based JEM training (e.g., training curves, divergence/failure rates, gradient-norm statistics), beyond final performance metrics.
4.	The proposed DAT is not compared head-to-head with prior AT-JEM methods (e.g., Robust-JEM) under matched backbones, budgets, and evaluation protocols, making it hard to attribute gains to the proposed ingredients rather than setup differences.

[1] Korst, R., & Asadulaev, A. (2022). Adversarial training improves joint energy-based generative modelling. arXiv preprint arXiv:2207.08950.

### Questions
1.	Please provide evidence that AT stabilizes SGLD-based JEM training: show training curves, failure/divergence rates, and input-gradient norm, not only final metrics.
2.	Please sharpen the novelty narrative relative to AT-EBM, CEM, JEM++, and Robust-JEM: what is fundamentally new here (objective, theory, or capabilities) beyond an engineering consolidation?
3.	The $l_\infty$ is the more common norm in adversarial robustness. Why are experiments restricted to $l_2$? Please either justify this choice or add $l_\infty$ evaluations with standard AutoAttack/RobustBench protocols. 
4.	Please report compute: FLOPs or wall-clock for training, plus sampling throughput (images/sec).
5.	What is the sampling procedure for generation—pure gradient-based synthesis or any MCMC steps? Please clarify the runtime and compute for each setting.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a framework to simultaneously achieve robust classification and high-fidelity generative modeling within a single network. The key contribution is the use of a Dual Adversarial Training (DAT) strategy as an alternative to unstable sampling-based (SGLD) in Joint Energy-based Models. One AT component ensures discriminative robustness against adversarial attacks, while the other uses PGD-generated contrastive samples. A binary cross-entropy loss is used to perform a stable AI-based generative modeling approach. Experimental evaluations on standard datasets (CIFAR-10, CIFAR-100, and ImageNet) show improvement in adversarial robustness over existing hybrid models while maintaining good generative performance.

### Strengths
* The paper addresses the major stability and scaling issues of previous Joint Energy-based Models by replacing the unstable sampling-based SGLD with adversarial training-based optimization.
* The proposed framework improves adversarial robustness in the discriminative tasks while simultaneously maintaining generative fidelity compared to existing hybrid models.
* The dual adversarial training setup stabilizes training compared to standard GAN frameworks as demonstrated through detailed analysis and ablation studies.

### Weaknesses
* The dual optimization adds computational overhead, which could make the training slower and impractical for large-scale high-resolution images. Computational inefficiency is mentioned but not reported in terms of training/inference time and parameter size.
* Experiments are performed mainly on standard image datasets (e.g., CIFAR-10/100, ImageNet), without extending to complex datasets or domains, which limits claims of broad applicability.
* The sample selection for FID calculation needs to be better justified. 
* Performance comparison is weak as the paper shows comparisons against some old methods (mostly 2020 or earlier). The paper could have included comparisons with more recent hybrid models (e.g., diffusion-classifier hybrids or energy-based approaches) for a stronger empirical baseline.
* Missing references: L-041 - "Recent research .... understanding of generative models" and L-094 - "sample generation .... semi-supervised learning"
* The paper claims significant improvement without reporting any statistical significance analysis. 
* The training procedure can be dataset-specific, requiring tuning parameters and adjustments for new datasets. The current experiments are confined to image data. Extending it to other data modalities could introduce new stability challenges, making its generalizability uncertain.
* L-365: "Our best generative configuration ... achieves an FID of 5.39 ... requiring significantly less sampling steps." Not clear which dataset it's referring to.
* The term PGD is never defined in the paper.

### Questions
* Have the authors explored whether the joint model maintains robustness when transferring to out-of-distribution (OOD) data or cross-domain settings (e.g., different image modalities or noise conditions)?
* Would the model’s scalability or performance differ on higher-resolution or more complex datasets?
* How do the dual objectives interact during training? For example, does improving the generative objective always enhance discriminative performance, or are there cases where they conflict? Some insight through visualization or empirical justification would be helpful.
* How sensitive is the DAT's performance to the relative weighting and scheduling between the discriminative and the generative AT losses?
* Can the authors provide insight into the computational cost? Specifically, what is the training time overhead (e.g., in GPU-hours) compared to training a non-robust JEM or a standard robust classifier on CIFAR-10/100?

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
The paper focusses on improving the Joint Energy Models (JEM) proposed by Grathwohl et al (2019) using techniques from Adversarial Training. They show that when such ideas are incorporated in the joint training of the generative + discriminative energy model, the training stability increases and surprisingly alleviates the need of gradient penalty regularizations, and improves the classification accuracy vs generative performance tradeoffs. Additionally, OOD detection and adversarial robustness and classifier calibration are auxiliary benefits with this approach.

### Strengths
- The objective of improving JEM training is made clear early on for the readers to get a grasp of the problem statement.
- The overview of Grathwohl et al. 2019 in the Method section was essential in setting up the context and mathematical notation for the problem. I thank the authors for the great job done here.
- The fundamental idea of incorporating Adversarial Training for more than just adversarial robustness is interesting and insightful. The authors successfully convince the reader how AT objectives improve the join distribution modeling in JEMs, which is quite different from the original focus of AT objectives in improving adversarial robustness.
- Insightful findings on stabilizing training while still having BN layers and data augmentations.

### Weaknesses
Minor:
- Numerous abbreviations are used before first expanding them. Example: PGD/SGLD in Abstract. PGD in the main paper before it's used in Line 63-65. "DAT" is used on line 249 without expanding it first.
- Missing Citations:
-- 1) Line 39 - "...rarely excelling at both simultaneously." - Please cite?
-- 2) Line 40 - "...but may underperform on downstream classification tasks." - Please cite?
- Misleading to say on Line 76: "datasets of increasing complexity from CIFAR-10 to ImageNet..." The paper only includes CIFAR-100 additionally and that is not a spectrum of datasets of increasing complexity as claimed.
- Section 3.1 under Method: Please define Z(\theta) before or after using it in the equations. Also, please make it clear and explicit that the Z(\theta) used in P(x) on line 161 is not the same as the Z(\theta) used in Eq. (2). Explicitly define them in each case with an integral etc to show the margnialized partition function.
- Section 3.1, please also talk about how Z(\theta) is handled in the loss function and how it is estimated.
- Please clarify Equation 11: Are the authors only using the adversarial loss on samples x_adv for classification objective and No regular CE loss on the input samples "x"? Or are both losses turned on and the AT-CE loss is just an auxiliary loss to a regular CE classification loss on input samples "x"?
- Unclear why OOD data is required in Line 328. Is it only for Eval? Or is it used during training too (which RATIO method requires)?

Major:
- The *primary* focus of the paper is unclear. Is the focus on improving the discriminative-generative performance tradeoffs? Is the focus on improving the training stability of JEMs mainly? Is the focus on improving the adversarial robustness or OOD detection? It is quite confusing despite novel insights presented in the paper.
- Continuing with the prev major weakness, the results in Table 1 are not clear to any extent where the method presented excels in relation to the other methods. For example, for CIFAR-100 hybrid models the DAT method is worse both in classification accuracy as well as FID than EGC method (sure, there is an extra benefit of Adversarial Robustness -- which comes back to the question of what aspect being the primary focus of the paper). Please BOLD the numbers of your method that you think are excelling at a certain aspect in relation to the rest of the models. 
- The datasets presented CIFAR-10/100 are not sufficient in convincing the readers of the merits of the work. These low res datasets are suited for a quick PoC run. The only substantial dataset used is ImageNet. Please consider including other datasets that compare to ImageNet in res like the LSUN dataset and/or CelebA faces etc. I think the results are insufficient to make a clear conclusion at the current form of the paper.
- Line 195 - "Preventing numerical overflow/ underflow" -- Please either cite or show a toy example as evidence that this indeed happens while training. You can follow the plots used in this paper (https://openaccess.thecvf.com/content/WACV2022/papers/Bhaskara_GraN-GAN_Piecewise_Gradient_Normalization_for_Generative_Adversarial_Networks_WACV_2022_paper.pdf) where to convince that the gradient explodes, an explicit plot of the gradient norm is presented in Fig 3. 
- Line 199 "The grad formulation stabilizes training at the cost of limiting the EBM to modeling the support of p_data...." -- This is not clear to the reader how the cost here is limiting the EBM to modeling the support but not the full dentsity. Please include a citation or a toy experiment to prove this is true.
- Line 215 - The paper mentions how the authors' method impoves the stability of training, however, no experiment is presented to back up this claim. For example, out of a random 10 experiments for each model, what fraction destabilize and diverge during training. Is the new method better in this quantitatively. See Fig 2 of (https://openaccess.thecvf.com/content/WACV2022/papers/Bhaskara_GraN-GAN_Piecewise_Gradient_Normalization_for_Generative_Adversarial_Networks_WACV_2022_paper.pdf) where a large FID/KID score implies training instability out of 5 random runs.

### Questions
Please see weaknesses.

Additional Questions:
1. - Line 319-320 - The authors say they use the RATIO pretrained models. However, they also show in Eq 13 how RATIO objective is different from their AT-CE objective. This totally changes their method fundamentally for CIFAR-10 and ImageNet models since the pretraining objective is RATIO objective which they elucidate how it's different from theirs in Eq 13. Since the datasets used in the paper are no where close to being considered large, I highly recommend the authors not use any pretrained models and train them from scratch using their proposed formulation without departure to other objectives like RATIO that makes it quite confusing to the reader on the exact proposal for training in this paper. 
2. - The weighted loss function modification in Eq (7) introduced by the authors is similar to the Focal Loss (https://arxiv.org/abs/1708.02002) albeit with gamma=1. Please compare this paper & suggest the readers why such a specific weight form is chosen (is it empirical? or is there a theoretical argument?)

Please also see weaknesses.

### Soundness
2

### Presentation
3

### Contribution
3
