# LetheViT: Selective Machine Unlearning for Vision Transformers via Attention-Guided Contrastive Learning

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Vision Transformers (ViTs) have revolutionized computer vision tasks with their exceptional performance. However, the introduction of privacy regulations such as GDPR and CCPA has brought new challenges to them. These laws grant users the right to withdraw their data, necessitating not only the deletion of data but also the complete removal of its influence from trained models. Machine unlearning emerges as a critical solution, with exact unlearning being computationally prohibitive and approximate methods offering a more practical approach. This work addresses the particularly challenging scenario of random data forgetting in ViTs, where the model must forget specific samples while retaining others, even within the same class. We first reveal the core characteristics of ViTs through selective masking experiments: when high-attention areas are masked, the model retains its recognition capability but significantly weakens its memorization ability. Based on the above insights, we propose LetheViT, a contrastive unlearning method tailored for ViTs. LetheViT uses masked image inputs to generate positive logits and original image inputs to generate negative logits, guiding the model to forget specific details while retaining the general cl category outlines. Experimental results demonstrate that LetheViT achieves state-of-the-art performance, effectively balancing privacy compliance with model efficacy.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper studies random data forgetting for Vision Transformers (ViTs), i.e. the harder machine-unlearning (MU) setting where you must forget only some samples within a class, while keeping the rest of the class usable. The authors make an empirical observation on CIFAR-100 with DeiT-T: if you take the self-attention map, mask the top-attended 5% patches by zeroing pixels, the test accuracy (TA) basically stays the same, but the membership inference attack (MIA) success rate drops sharply (from 24.49% to 10.16%). They interpret this as: ViTs retain class-level semantics even when key, high-attention details are removed, but sample-specific (“memorized”) information is degraded. Based on that, they propose LetheViT, a contrastive unlearning procedure. On CIFAR-10/100, SVHN and Tiny-ImageNet, and across many ViT/DeiT/Swin sizes, they show lower Average Gap (AG) to Retrain than previous approximate MU methods (FT, GA, IU, RL, ℓ1-sparse, SalUn). They also provide an efficiency plot showing LetheViT is must cheaper than full retrain and on par with other methods.

### Strengths
1. **Clear motivation from a ViT-specific property.** The selective masking experiment (their Table 1 + Fig. 1) is genuinely nice: “mask 5% of top-attention patches → TA stable, MIA collapses.” This is actually a good, model-specific insight and a good hook for MU on ViTs.
2. **Method is simple and implementable.** The contrastive setup with original-frozen logits as positives/negatives, and current logits as anchors is easy to re-implement. Algorithm 1 is clear. Using the original model as the source of “what to forget” vs. “what to keep” is an elegant reuse.
3. **Metrics and baselines are good.** Authors use the usual MU metrics: FA, RA, TA, same MIA as in ℓ1-sparse and SalUn, and they add Average Gap (AG) to summarize deviation from Retrain. This is actually a useful addition — without AG you indeed need to track 4 deltas.
4. **Breadth of architectures.** They run ViT-T/S/B, DeiT-T/S/B, and Swin-T/S on Tiny-ImageNet (Table 2). That’s a healthy sweep and it shows the method is not tied to one backbone.
5. **Efficiency analysis.** The bar plot vs. Retrain / FT / GA / IU / SalUn is a nice touch.

### Weaknesses
1. **Random data forgetting here is very related to instance-level recognition/retrieval, but this is not acknowledged**. The paper frames random data forgetting as “you must forget user A’s cat without forgetting user B’s cat”, i.e. intra-class discrimination for unlearning. This is exactly the kind of setting the instance-level community has been handling for years: instance-level retrieval, instance-level recognition, fine-grained, landmarking, fashion, artworks, where small, fine-grained details are what determine the identity. This line of work should definitely appear in Related Work, and in the discussion that motivates the masking:

[1] Kordopatis-Zilos, Giorgos, et al. “Ilias: Instance-level image retrieval at scale.” Proceedings of the Computer Vision and Pattern Recognition Conference. 2025.   
[2] Radenović, Filip, et al. “Revisiting oxford and paris: Large-scale image retrieval benchmarking.” Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.    
[3] Ypsilantis, Nikolaos-Antonios, et al. “The met dataset: Instance-level recognition for artworks.” Thirty-fifth conference on neural information processing systems datasets and benchmarks track (Round 2). 2021.    
[4] Liu, Ziwei, et al. “Deepfashion: Powering robust clothes recognition and retrieval with rich annotations.” Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.    
[5] Weyand, Tobias, et al. “Google landmarks dataset v2-a large-scale benchmark for instance-level recognition and retrieval.” Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020.    
[6] Wang, Shuang, and Shuqiang Jiang. “Instre: a new benchmark for instance-level object retrieval and recognition.” ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM) 11.3 (2015): 1-21.   
[7] Oh Song, Hyun, et al. “Deep metric learning via lifted structured feature embedding.” Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.    
[8] Psomas, Bill, et al. “Instance-Level Composed Image Retrieval.” The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.    

2. **Dataset choice is weak for the claim**. CIFAR-10/100 and SVHN are not the right datasets to argue about selective instance-level forgetting. Tiny-ImageNet is a bit better, but still 64×64. If your whole method zeros image patches, it is much more convincing to show it on higher-res, finer-grained data (see datasets above). 

3. **Missing related work on attention-guided masking (and potential experiments)**. AttMask [9] studies masking the most highly-attended tokens. It also includes some other masking strategies like block-wise random or highly-attended but leaving "hints" visible. It would be nice to extend the ablation of the masking strategies little bit more. 

[9] Kakogeorgiou, Ioannis, et al. “What to hide from your students: Attention-guided masked image modeling.” European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2022

4. **Discussion on the choice of performing contrastive on logits vs. features**. The instance-level memorization that should be removed is probably more naturally expressed in the feature space. Moreover, to my knowledge, contrastive is done before the classifier, not on logits. Please explain why you contrast on logits and not on features. If the reason is “we want to keep the classifier stable and only pull the representation toward the masked prediction”, please elaborate on this. Moreover, consider adding an experiment using features; features can be global image-level ([CLS]) or dense patch tokens. An ablation would be valuable.

### Questions
Suggestions and questions for authors: 

- Could you please add a paragraph in Related Work called e.g. “Connection of random data forgetting to instance-level recognition and retrieval”
- Why does MIA increase again at 20% and 30% masking (Table 1)? Please explain the mechanism
- Why contrast on logits and not on features? This is unusual in contrastive learning, and it’s exactly where the fine-grained information lives.
- Can you report results on at least one instance-level dataset (e.g. INSTRE [6] or ILIAS [1]) to support the “random data forgetting” story?
- Can you try attention-guided masking strategies from [9] and report results?

Overall, the paper has clear practical benefits for ViT-specific unlearning and introduces a useful aggregate metric (AG), and I’d be happy to raise my score if the authors add the instance-level experiments, integrate the related work [1–9], and include a small logits-vs-features masking discussion or ablation.

### Soundness
3

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
3

### Summary
This paper addresses the discriminative unlearning problem in the vision domain. The authors introduce an unlearning method called LetheViT that unlearns specific samples in vision transformers. The authors mask a tiny set of high-attention patches and use a contrastive loss that pulls the unlearned model toward masked logits and pushes it from original logits. In the experimental section, the authors validates the effectiveness of the proposed method on CIFAR, SVHN, and Tiny-ImageNet using various backbones including DeiT, Swin, and ViT. The experimental results shows that LetheViT narrows the gap to full retraining with much lower compute, while keeping task accuracy largely intact.

### Strengths
* The authors provide analytic motivation (however, it should be further improved, as mentioned in weaknesses)
* The proposed methods surpass other baselines up to 2024.

### Weaknesses
* The provided analytic motivation is not sufficiently persuasive 
    * Both Figure 1 and Table 1 supports the rationales for employing masking strategy during unlearning. However, why such small amount of patches significantly impacts on memorizing and forgetting samples seems not supported. From my perspective, understanding the role of these few patches and their impacts in unlearning are more essential than the methodological details.
* Comparison methods are not up-to-date
    * While current comparison methods including SalUn are also important approaches in unlearning, are the methods are introduced before 2025. Since this research area is increasing rapidly, comparison with up-to-date methods (e.g., Patel et al. [1]) seems to be important
    * More comparison with these papers is required in terms of intuition and performance

* Some explanation or claims should be further supported
    * e.g., in lines 54-56, there is no evidential analysis or references. The example case in lines 56-59 is also an explanatory supports without actual experiments. Without them, the claim cannot be supported
    * While the overall trends across various masking ratios in Table 1 are understandable, it would be good to also explain about the trend flipping near the masking ratio of 20%. 


* Minor issues
    * Lack of explanation in some figures and tables
        * e.g., Table 1 and Figure 2 and 3 are not described well
    * Cite more up-to-date unlearning papers [1,2,3]

[1] Learning to Unlearn while Retaining: Combating Gradient Conflicts in Machine Unlearning, Gaurav Patel, Qiang Qiu; Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025, pp. 4211-4221

[2] Towards Scalable Exact Machine Unlearning Using Parameter-Efficient Fine-Tuning, Somnath Basu Roy Chowdhury, Krzysztof Choromanski, Arijit Sehanobish, Avinava Dubey, Snigdha Chaturvedi,  ICLR 2025

[3] NegMerge: Sign-Consensual Weight Merging for Machine Unlearning. Kim, H. S., Han, D., and Choe, J. In Proceedings of the Forty-second International Conference on Machine Learning, 2025

### Questions
Could class-wise forgetting methods be applied to the random data forgetting setting? Because some unlearning approaches including  gradient ascent, minimization, and SalUn seem to applicable to both class-wise forgetting and random data forgetting setting. If so, shouldn't the paper compare against class-wise forgetting methods [4] under the random data forgetting setting?

[4] Novo: Unlearning-compliant vision transformers. Soumya Roy, Soumya Banerjee, Vinay Verma, Soumik Dasgupta, Deepak Gupta, and Piyush Rai., arXiv preprint arXiv:2507.03281, 2025.

### Soundness
2

### Presentation
2

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
This paper proposes a new approach for the problem of machine unlearning in ViTs. Specifically, this work focuses on random data forgetting, where only specific samples (rather than entire classes) need to be forgotten. The authors claim that masking high-attention patches in ViTs reduces sample memorization while preserving class-level recognition. Leveraging this, they propose LetheViT, a contrastive unlearning method that encourages ViTs to forget sample-specific details while retaining category-level structure. Experiments show that LetheViT successfully maintains accuracy while succeeding in yielding a low membership inference attack (MIA) success rates.

### Strengths
- The proposed approach is intuitive
- Experimental results seem to validate that the methodology is a good solution for the proposed problem
- The paper is clearly written and easy to follow, for the most part
- I appreciate the theoretical analysis on the convergence of the proposed approach

### Weaknesses
- Experiments are mainly on medium-scale datasets (CIFAR, Tiny-ImageNet). It is unclear how LetheViT performs on large-scale datasets like full ImageNet, which would be crucial for practical deployment.

- Performance is evaluated by comparing average metric values (e.g. average test accuracy) between the unlearning approach and the ideal retrain baseline. However, it seems like method that matches the retrain baseline in terms of the point wise measures doesn't fully validate that the model is behaving like the fully retrained baseline. Instead, including some kind of mutual information score between the predictions of the retrain model and the predictions of the unlearned models seems like it would be more informative?

- The "average gap" metric is the average difference between the performance for an unlearning method and the performance of the ideal retrained model. Specific, the average is computed over 3 accuracy metrics and 1 adversarial membership inference metric. This 3:1 ratio seems like it would bias the AG metric in favor of methods that maintain accuracy, and gives a lesser weight to the task of unlearning. Computing average gap over MIA and TA seems like it would be more reasonable, to me at least.

### Questions
- Please let me know if 3:1 ratio concern regarding the AG metric is based on a misunderstanding. Otherwise, a version of this metric that weights the adversarial metric the same as the sum of the 3 accuracy metrics would be nice to see.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This manuscript focuses on "random data forgetting" (i.e., sample-level) in Vision Transformers (ViTs). The authors propose LetheViT, a new two-stage unlearning pipeline for this problem. In forgetting stage, contrastive loss is applied to the forget set. This loss pushes the unlearned model's output away from the original model's output on the original image, and pulls it towards the original model's output on an attention-masked version of the image. In Retaining Stage: Standard cross-entropy fine-tuning is performed on the retain set. This manuscript claims that this method achieves state-of-the-art performance, measured by an "Average Gap" metric that compares the unlearned model to a "gold standard" retrained model across four metrics (FA, RA, TA, MIA).

### Strengths
1. The paper try to tackle random data forgetting (sample-level unlearning), which is a challenging and practical unlearning task.
2. The authors validate their method across a wide variety of models and datasets, providing a comprehensive evaluation.

### Weaknesses
1. The Core Contrastive Unlearning Objective (Eq. 5) is a heuristic solution. The goal of approximate unlearning is to produce an unlearned model $f_{\theta_u}$ that approximates the "gold standard" retrained model $f_{\theta_r}$ (trained from scratch on $D_r$). The proposed contrastive loss does not optimize towards this objective. The "positive" target for the unlearning model's output $Z = f_{\theta_u}(x)$ is $Z_p = f_{\theta_o}(x_m)$, which is the output of the original (pre-forget) model $f_{\theta_o}$ on a masked input $x_m$. The paper provides no theoretical or strong empirical justification for why $f_{\theta_o}(x_m)$ should be a valid proxy for $f_{\theta_r}(x)$ (the output of the true retrained model on the original forget sample). The method is optimizing $f_{\theta_u}$ to mimic the original model's behavior on a damaged input. This is a heuristic solution and the claim that this "enables the ViT model to forget the specific details... while retaining a general outline" is more like a post-hoc narrative.
2. The entire method hinges on the observation in Section 3 (Table 1) that masking 5% of top-attended patches "preserves recognition capability (TA increases by 0.01%) while significantly degrading memorization (MIA drops by 14.33%)." This "key observation" is unconvincing. This experiment is conducted on small datasets (like CIFAR-100), and a single forget ratio (10%). This is insufficient to build a convincing unlearning method upon. The slight increase in TA suggests the baseline model may have been overfitted, and masking acted as a simple regularizer. The MIA is performed on masked images against the retrain model. A drop in MIA success is the expected outcome when the input distribution is shifted (from original images to masked images). This does not prove that the model has "forgotten" the sample-specific traces of the original image $x$; it only proves that the model (and the MIA attacker) are confused by the noisy input $x_m$. The conclusion that this "lose[s] sample-specific memory traces" is overclaim and not supported by the experiment.

### Questions
1. The overall pipeline consists of 2 epochs of the contrastive loss (forget step) followed by 8 epochs of standard cross-entropy training on the retain set. This "retain step" is a powerful unlearning baseline (often called "fine-tuning"). The paper's own "fine-tuning" baseline is trained for 10 epochs. LetheViT also trains for 10 epochs total. Therefore, LetheViT is not a new method, but rather a modification of the "fine-tuning" baseline, where the first 2 epochs of retain-set-tuning are replaced with 2 epochs of a custom forget-set-loss. The paper provides no ablation to disentangle the effects of these two stages. What if only the 2-epoch contrastive loss step is performed? What if a 2-epoch standard "negative gradient" step is followed by the 8-epoch retain step? Without these ablations, it is impossible to conclude that the contrastive loss is responsible for the performance gains.
2. The primary metric, "Average Gap", equally weights the gaps in FA, RA, TA, and MIA. This is questionable, as it hides critical trade-offs. For example, in the ViT-B results from Table 2, LetheViT achieves a good AG (2.23). However, it does so at a severe cost to model utility, incurring a massive TA gap of 5.35, which is far worse than SalUn (0.80). People would likely prefer a model that retains its generalization performance (low TA gap) over one that simply scores well on an averaged metric. Could you explain more about this?

### Soundness
2

### Presentation
3

### Contribution
2
