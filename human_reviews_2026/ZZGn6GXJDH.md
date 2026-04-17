# Rethinking Data Augmentation for Adversarial Distillation: An Excess Risk Perspective

- Decision: Reject
- Scores: 2, 4, 2, 4

## Abstract
Adversarial Robustness Distillation (ARD) enhances the robustness of lightweight models by transferring knowledge from robust teacher models. Most studies focus on output alignment, while input-side augmentation remains underexplored. We reveal a surprising phenomenon: augmentation techniques such as CutMix and AutoAugment, which work well in standard Knowledge Distillation (KD), are ineffective in ARD and can even reduce student robustness. To explain this, we derive an excess risk bound for ARD based on uniform stability, revealing how augmentation diversity and teacher performance on augmented data jointly affect generalization. Our analysis shows that, while augmentation improves sample diversity and smooths the loss landscape, low-quality or overly strong augmentations can compromise teacher reliability during training. This insight highlights a fundamental trade-off in ARD: effective augmentation must balance diversity with teacher reliability. To achieve this balance, we propose ASDA (Active Selection for Diffusion-based Augmentation), which leverages diffusion-generated samples and actively selects informative and teacher-reliable data, guided by output fidelity and entropy. Experiments on CIFAR-10/100 show that ASDA outperforms baselines and surpasses SOTA, clarifying the role of augmentation in ARD and providing a practical solution for improving student robustness.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper investigates the role of data augmentation in Adversarial Distillation (AD), starting from the observation that standard augmentation techniques can degrade, rather than improve, student model robustness. To explain this phenomenon, the authors provide a theoretical analysis using an excess risk bound. Motivated by this analysis, the paper proposes a new framework, ASDA (Active Selection for Diffusion-based Augmentation). ASDA first leverages a diffusion model to generate a large pool of candidate samples and then actively selects a subset based on criteria designed to be both informative and reliable for the teacher model. The authors conduct experiments on CIFAR-10 and CIFAR-100 to demonstrate that their proposed method achieves state-of-the-art results compared to existing AD baselines.

### Strengths
1. The paper addresses an important and relatively under-explored research question: the role and effectiveness of data augmentation in AD.
2. The proposed method, ASDA, achieves strong empirical results, reaching state-of-the-art performance on the CIFAR-10 and CIFAR-100 benchmarks when compared against the selected AD baselines.

### Weaknesses
1. The introduction portrays robust overfitting as a general issue in AD largely on the basis of the vanilla ARD example; however, this characterization does not hold uniformly across AD baselines. For example, RSLAD does not exhibit the same overfitting behavior and can even improve test robustness when training is extended (e.g., to 300 epochs). Consequently, the motivation for positioning data augmentation as a necessary remedy for “robust overfitting in AD” is not fully persuasive.

2. The reported improvement from using 300k diffusion-generated samples as augmentation appears somewhat trivial and does not convincingly demonstrate the novelty of the proposed approach. Prior work [1] has already shown that diffusion-generated data can substantially enhance adversarial training robustness even without any distillation. Although the authors employ different model setups, the reported robustness levels are comparable to what standard adversarial training (e.g., TRADES) typically achieves when trained with diffusion augmentations (around 53–54% AutoAttack accuracy on ResNet-18). Therefore, it remains unclear whether ASDA provides a substantial advantage beyond what can already be achieved by straightforwardly applying diffusion-based augmentation.



3. To convincingly demonstrate the effectiveness of ASDA, the experimental design should directly compare against other adversarial training and distillation baselines under the same diffusion-augmented setting. As it stands, the reported improvement could largely reflect the benefit of diffusion augmentation itself rather than the proposed active selection mechanism. Without such cross-method comparisons, it remains unclear whether the observed advantage truly stems from ASDA’s selective augmentation strategy or simply from the use of diffusion-generated data.

4. If the authors intend to argue that diffusion-based augmentation itself enhances adversarial distillation, the paper should be reorganized to explicitly support that claim. Merely reporting performance gains does not reveal why diffusion data benefit AD. A more convincing argument would require analyzing how such augmentation affects the teacher–student relationship — for example, whether it facilitates better logit matching, improves teacher reliability under adversarial perturbations, or reduces the teacher–student output divergence. Without such mechanistic analysis, it remains unclear whether the proposed approach truly advances our understanding of how data augmentation contributes to adversarial distillation, or simply reports another case where diffusion-generated data empirically improve robustness.

5. The theoretical section closely mirrors [2] — it follows the same on-average stability framework, assumptions, and proof structure. Because of this, it reads more like a corollary than a new theorem. Furthermore, this framework fails to provide any insight into why diffusion-based augmentation is the preferred source of samples for AD. The theory only justifies the filtering component of ASDA, offering no principled justification for why diffusion models should be the source of augmentation in the first place. Therefore, this part does not warrant such a large portion of the main paper. It would be better to reduce its emphasis, strengthen the experimental analysis, or introduce a more original theoretical perspective that goes beyond existing data-dependent stability analyses.



6. The experimental comparison in Table 1 is not entirely fair because the Diffusion-300k setting introduces a substantially larger amount of training data than the other augmentation methods. While Cutout, CutMix, RandAugment, and AutoAugment operate on the same 50k CIFAR-10 samples, Diffusion-300k expands the training set by seven times through synthetic image generation. Consequently, the reported performance gain may stem largely from the increased data volume rather than the augmentation quality. 

7. The Diffusion-300k setting increases the training set from 50k to ~350k samples—a 7× increase in training steps under the same schedule. Yet the paper does not report training time, GPU hours, throughput, or the cost of generating/storing diffusion samples (even generate 1M and select 300k). 

From here, minor mistakes, which do not affect much on my evaluation.

8. The experimental setup focuses mainly on CIFAR-10 and CIFAR-100. Evaluations on additional datasets such as Tiny-ImageNet or SVHN are necessary to demonstrate broader generalization and verify whether the proposed method scales effectively to more complex or diverse data distributions.

9. Several reported baseline numbers deviate markedly from the original papers, casting doubt on the fairness of comparisons. For instance, on CIFAR-10 with ResNet-18, the paper reports 48.69% AutoAttack for AdaAD, whereas the original AdaAD reports 52.96% under the same dataset/architecture. Although the authors mention using different implementation practices, such discrepancies should be clearly acknowledged and explained.


[1] Wang, Zekai, et al. Better Diffusion Models Further Improve Adversarial Training. ICML 2023.

[2] Wang, Yihan, Shuang Liu, and Xiao-Shan Gao. "Data-dependent stability analysis of adversarial training." Neural Networks 183 (2025): 106983.

### Questions
- Can you quantify the practical benefit of the ASDA selection mechanism compared to simply using a randomly selected subset of the diffusion samples?
- Is your proposed scoring function superior to a simpler baseline, such as only selecting samples that the teacher model classifies correctly?
- It is known that diffusion-based augmentation can significantly improve standard TRADES, which does not use a teacher. While your ASDA selection mechanism requires a teacher, if a teacher is used only for the selection of the dataset, would training a standard TRADES model on these ASDA-selected samples be more effective than training TRADES on the full set of diffusion-generated samples?


Typos
- L672, L674, C&W attack label mismatch
- RandAugment (L159) and Randaugment (L79)

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
Adversarial Robustness Distillation (ARD) is a technique for creating a robust model using a student teacher training setup. Data augmentation techniques have been used to create additional variations in the training date to improve model performance. However, in this work it is claimed that data augmentation techniques may actually hinder ARD. A new technique is proposed called Active Selection for Diffusion-based Augmentation. Empirically the results show that this technique is effective on the CIFAR-10 and CIFAR-100 datasets.

### Strengths
The paper is well written and easy to follow. I was not able to go through the details of all the mathematical derivations but I assume they are correct. Experimentally the method is empirically demonstrated (on CIFAR-10/100).

### Weaknesses
Weakness #1: There are two related works you do not cite or discuss.  

1. Rethinking data augmentation:
https://www.sciencedirect.com/science/article/abs/pii/S0020025523014238
2. Another SOTA knowledge distillation approach:
https://dl.acm.org/doi/10.1016/j.procs.2025.07.114

Could you explain how these relate to your paper and/or add this into the main body?

Weakness #2: Experimental results are lacking. As a reviewer I cannot mandate that you do additional experiments. What I can say is that it would be much easier to advocate for your paper if experiments were done on Tiny-ImageNet or ImageNet or even any other non-CIFAR dataset. With only these two datasets (and CIFAR-100 essentially being a relabeling of CIFAR-10) it is hard to advocate for your paper on experimental grounds. 

Weakness #3: A lot of the attacks you use are redundant or old. I appreciate that you include auto-attack as that is more SOTA, but why is so much space taken up by FGSM and PGD? I would like to see that moved to the appendix. Also, no SOTA black-box attacks (beside what is already put in AA) is used. Why are there no black-box attacks, even as simple baselines just to show how effective your method is? 

Other minor points:
=Figure 1 is kind of blurry (especially the top text). Is it possible to use a higher quality figure? 
=FGSM is an ancient method at this point, I would prefer if those results were removed from Table 1, it is a waste of space. 
=Table 1, no epsMax value is given for the attack in the caption? This is a really important missing detail.
=I am very confused about how the T-WRA metric is actually derived. With which attack is T-WRA done? 
=Table 2, please don’t report FGSM, again. 
=The conclusion is extremely short and lacking in detail. If you were going to write such a sparse conclusion, why not just remove it entirely?

### Questions
Please answer the questions I posed in the weaknesses section. Specifically how you will handle weakness points #1, #2 and #3 and improve your paper the questions that are most important for my review.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper argues that augmentation methods, such as CutMix and AutoAugment, may degrade robustness transfer in logit-based distillation. To understand this problem, a comprehensive theoretical analysis is performed using excessive risk. The theoretical analysis provides a key insight: the need to balance between teacher reliability and sample diversity. Based on this analysis, the authors propose a filtering mechanism. The proposed method first generates a large pool of candidate images using a diffusion model and then applies the filtering method to select the best candidates. The filtering mechanism uses a linear combination of the entropy of the teacher's predictions and the distance between the teacher's prediction and the ground truth.

### Strengths
The paper is generally well written, investigates an interesting and important question, and provides a comprehensive theoretical analysis.

### Weaknesses
On line 154, the authors mention that there is a large generalization gap in adversarial distillation. 
> In Adversarial Robustness Distillation (Goldblum et al., 2020), a large robustness generalization gap is observed: as shown in Fig. 1, although training robustness can reach above 90%, test robustness may drop to around 40%, revealing a severe failure to generalize.

However, a similar gap is also observed in adversarial training [6, 7]. Best models often achieve 40-70% robustness even when their clean accuracy is 95-99%.

Moreover, in the next line, the authors posit whether data augmentation can bridge the gap.

> This raises the question of whether data augmentation, which is effective in standard Knowledge Distillation, can mitigate such a gap.

However, the authors do not mention any prior works explaining the connection between data augmentation and generalization (the difference between train and test time error).

An important aspect of this work is the idea that augmentation may degrade adversarial distillation. However, some prior works [4] have shown a strong connection between adversarial robustness and augmentation. More importantly, prior work [2] has also shown how some augmentation can help improve robustness distillation. 

Discussion on some important relevant robustness distillation  [1, 2, 3] and synthetic data for robustness [5, 8, 9] is missing.

[1] Chan, A., Tay, Y., & Ong, Y. S. (2020). What it thinks is important is important: Robustness transfers through input gradients. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 332-341).

[2] Awais. M., Zhou, F., Xie, C., Li, J., Bae, S. H., & Li, Z. (2021). Mixacm: Mixup-based robustness transfer via distillation of activated channel maps. Advances in neural information processing systems, 34, 4555-4569.

[3] Shao, R., Yi, J., Chen, P. Y., & Hsieh, C. J. (2021). How and when adversarial robustness transfers in knowledge distillation?. arXiv preprint arXiv:2110.12072.

[4] Zhang, L., Deng, Z., Kawaguchi, K., Ghorbani, A., & Zou, J. (2020). How does mixup help with robustness and generalization?. arXiv preprint arXiv:2010.04819.

[5] Schmidt, L., Santurkar, S., Tsipras, D., Talwar, K., & Madry, A. (2018). Adversarially robust generalization requires more data. Advances in neural information processing systems, 31.

[6] Nakkiran, P. (2019). Adversarial robustness may be at odds with simplicity. arXiv preprint arXiv:1901.00532.

[7] Zhang, H., Yu, Y., Jiao, J., Xing, E., El Ghaoui, L., & Jordan, M. (2019, May). Theoretically principled trade-off between robustness and accuracy. In International Conference on machine learning (pp. 7472-7482). PMLR.

[8] Alayrac, J. B., Uesato, J., Huang, P. S., Fawzi, A., Stanforth, R., & Kohli, P. (2019). Are labels required for improving adversarial robustness?. Advances in Neural Information Processing Systems, 32.

[9] Carmon, Y., Raghunathan, A., Schmidt, L., Duchi, J. C., & Liang, P. S. (2019). Unlabeled data improves adversarial robustness. Advances in neural information processing systems, 32.

### Questions
he proposed filtering method uses Entropy to make sure the samples are diverse. The entropy is maximum when all class labels are equally likely ($/dfrac{1}{K}$ for $K$ classes). Would not this be against the overall objective? In other words, the best samples are selected in such a way that the teacher would not provide useful information to the student. Correct me if I am wrong.

How do authors explain prior works showing a connection between robustness and augmentation [4], and how augmentation could be useful for robustness distillation?

How efficient is the proposed method, as an important aspect of distillation is efficiency? The active filtering requires generating an adversarial perturbation for each generated sample, and distillation also requires perturbation generation. 

It would be interesting to see a sample of selected images. 

On line 37, the authors mention 
> Adversarial Training (AT) (Szegedy et al., 2013), currently the most practical and effective defense, trains models directly on adversarial examples to substantially improve robustness.

I think the reference is wrong. It should have been Madry et al [1].

[1] Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018, February). Towards Deep Learning Models Resistant to Adversarial Attacks. In the International Conference on Learning Representations.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses why traditional data augmentation methods such as CutMix and AutoAugment, which have been known to be effective in standard Knowledge Distillation (KD), degrade performance in Adversarial Robustness Distillation (ARD). Through theoretical analysis grounded in uniform stability and excess risk bounds, the authors reveal a trade-off between augmentation diversity and teacher reliability in ARD, and then propose Active Selection for Diffusion-based Augmentation (ASDA) that leverages diffusion-generated samples and selects teacher-reliable, informative augmentations using entropy and teacher output fidelity.

### Strengths
- Interesting theoretical analysis: The excess-risk–based generalization analysis for ARD is new and offers valuable insight into the impact of data augmentation. The proposed “diversity–reliability trade-off” is intuitive and theoretically supported.
- Clear empirical motivation: The observation that common augmentations hurt ARD is well documented with empirical evidence (Table 1).
- Combining diffusion augmentation with active selection is conceptually simple yet effective.

### Weaknesses
- Simplistic selection mechanism: The active selection simply relies on a linear combination of entropy and teacher output distance, which seems heuristic and may not generalize beyond diverse data.

- Lack of ablation on teacher reliability: The claim that low teacher reliability increases generalization error is theoretically grounded but empirically underexplored.

- Insufficient evaluation dataset: CIFAR-10/100 are very small benchmarks. Large-scale or real-world evaluation would strengthen the claims.

- Limited comparative study: Baselines are restricted to ARD-style methods; there is no direct comparison to modern diffusion-augmented adversarial training or other data selection frameworks.

### Questions
- Could the method generalize to non-diffusion-based data or real-image augmentations?
- Can the authors provide a quantitative correlation between teacher reliability metrics (T-WRA, T-OS) and student generalization?
- Would integrating the selection process into joint training (rather than pre-selection) lead to better performance?

### Soundness
3

### Presentation
2

### Contribution
3
