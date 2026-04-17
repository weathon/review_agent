# Understanding the Learning Phases in Self-Supervised Learning via Critical Periods

- Decision: Accept (Poster)
- Scores: 6, 6, 8, 8, 6

## Abstract
Self-supervised learning (SSL) has emerged as a powerful pretraining strategy to learn transferable representations from unlabeled data. Yet, it remains unclear how long SSL models should be pretrained to yield such representations. Contrary to the prevailing heuristic that longer pretraining translates to better downstream performance, we observe a transferability trade-off: across diverse SSL settings, intermediate checkpoints can yield stronger out-of-domain (OOD) generalization, whereas additional pretraining primarily benefits in-domain (ID) performance. From this observation, we hypothesize that SSL progresses through learning phases that can be characterized via the lens of critical periods (CP). Prior work on CP has shown that supervised models exhibit an early phase of high plasticity, followed by a consolidation phase where adaptability declines but task-specific performance increases. Since traditional CP analysis was developed for supervised settings, we rethink it for SSL in two ways. First, we inject deficits to perturb the pretraining data and assess their lasting impact on representation quality via downstream tasks. Second, we compute the Fisher Information on pretext objectives to track plasticity, quantifying how sensitive model parameters are to the pretext task. Our experiments suggest that SSL models may exhibit their own CP, with CP closure coinciding with a sweet spot for broad downstream transferability. Leveraging these insights, we introduce CP-guided checkpoint selection as a strategy for selecting checkpoints that offer stronger OOD transferability. Finally, to balance the transferability trade-off, we present CP-guided self-distillation, which selectively distills layer representations from the intermediate checkpoint into their overspecialized counterparts in the final checkpoint.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates the temporal dynamics of self-supervised learning (SSL) and its effect on transferability. The authors identify a transferability trade-off: intermediate checkpoints during SSL pretraining often yield stronger out-of-domain (OOD) generalization, whereas longer training enhances in-domain (ID) accuracy. Authors characterize SSL pretraining into three phases—plasticity, consolidation, and overspecialization—tracked via Fisher Information dynamics and deficit injection experiments. They further propose 2 training scheme for improvement:
1. CP-guided Checkpoint Selection (CPCS) for identifying checkpoints near CP closure, improving OOD transfer;
2. CP-guided Self-Distillation (CPSD) to distill representations from CP checkpoints into later ones, mitigating overspecialization effects

### Strengths
1. Proposed that the learning process of SSL is stage-by-stage and provide evidences for verification.
2. Proposed 2 training improvement scheme CPCS and CPSD without uesag of extra label.
3. Propose novel explanation for OOD generalization decay at the end of training.

### Weaknesses
1. The paper defines Critical Period closure as the epoch when the Fisher Information (FI) curve stabilizes—operationalized as a near-zero slope. It better comes up with a formal metric consider many other variables like batch size, hyperparameters and even downstream tasks.
2. Need to provide more insights about why the FI dynamic performance differently among different SSL architectures. Are these differences come from the loss term or the contrastive and reconstruction-based SSL methods？

### Questions
1. Can Fisher Information be replaced with a more stable unsupervised proxy?
2. Do critical periods also exist in language or multi-modal SSL?
3. How the model capacity will influence the CP closure?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper studies how self-supervised learners evolve over training and argues that there exist “critical periods” (CP) during which representations are highly plastic and best for transfer. The authors (i) document a transferability trade-off: longer pretraining continues to improve in-domain accuracy while hurting out-of-domain generalization, (ii) operationalize CP analysis for SSL via two tools, and (iii) propose two practical mechanisms: CP-guided checkpoint selection and CP-guided self-distillation that transfers intermediate “sweet-spot” layer features into the final model to balance ID and OOD performance.

### Strengths
1. The idea of rethinking critical periods without labels by analyzing deficits and Fisher Information on the pretext loss is conceptually elegant and broadly applicable.
2. The CP-guided checkpoint selection provides a simple yet practical tuning mechanism. The selective self-distillation strategy offers a concrete approach to restore transferability while retaining late-stage in-domain performance gains.

### Weaknesses
1. The paper directly equates Fisher Information on the self-supervised pretext objective with parameter sensitivity to supervision signals to quantify plasticity, but it does not establish a theoretical connection between this proxy and transfer performance or generalization error.
2. The paper posits a trade-off between sustained in-domain improvement and degraded out-of-domain transferability, which motivates the critical-period perspective, but lacks experiments linking this phenomenon to representation drift or task specialization.
3. The use of broad phrases such as “across datasets or distributions” suggests generality, yet the paper neither defines nor categorizes the types of distribution shifts considered, nor decomposes which shift types benefit most, making the scope of conclusions unclear.
4. The abstract claims coverage “across multiple SSL methods, architectures, and datasets,” but omits explicit statements on systematic coverage and boundary conditions. Without clarifying the circumstances under which the findings fail, the general conclusions risk overextension.

### Questions
1. Why do different SSL methods in Section 2.2 exhibit distinct trends, and why do models such as SimCLR not show a clear “critical period” phase?
2. How is Fisher Information computed in practice? Do you use full, block-diagonal, or diagonal approximations, per layer or per parameter group, and how frequently is it recomputed during pretraining?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper studies the question of how long to train a visual self-supervised learning model, by looking at the critical periods happening throughout training. The paper presents metrics that allow to identify these periods, and studies how they correlate to downstream tasks. Finally, from these metrics the paper describes methods to reconcile optimal in-domain and out-of-domain performance on downstream tasks. Experiments are conducted on the ImageNet-1K and fMoW-RGB satellite imagery datasets, and several popular SSL methods are studied (SimCLR, VICReg, DINO and MAE).

### Strengths
- The paper highlights a very interesting phenomenon in SSL, the fact that there exists learning phases that correlate with different levels of performance in out-of-domain or in-domain tasks.  The paper calls this phenomenon, which is the counterpart of overfitting in supervised learning, but for self-supervised learning: “critical periods”. The paper clearly explains how they found these critical periods with precise metrics, how these metrics correlate to downstream tasks and how to leverage these metrics to derive SSL models with better generalization capabilities.

- There is existing large-scale evidence of this phenomenon in the SSL literature. Most SSL practitioners have already encountered these critical phases without putting a name on it.

- The paper is well-written, easy to follow and with good presentation, the message and finding is simple and presented with clear experiments.

### Weaknesses
- The paper focuses on two datasets: ImageNet, which make sense as a general pretraining datasets to learn visual representation, and fMoW-RGB, a satellite imagery datasets, so not generalist. But the use fMoW-RGB is not well motivated and does not help make the experiment convincing. Is the motivation to clearly identify what is ID and what is OOD ? I would appreciate more a focus on ImageNet data or even larger scale generalist data, with the objective to see if these critical periods are also observed in real-world or large-scale scenarios. There is some evidence in the literature that this is the case, in DINOv3, they use gram anchoring at the end of their pretraining to retrieve pic segmentation performance that the model has towards the beginning of training, this is very similar to what the authors describe line 78 “Intermediate checkpoints often achieve better out-of-domain (OOD) transfer than later checkpoints”.

- The paper does not study critical phases with the prism of data overfitting, or doing too many passes on the same data. What would happen in an infinite data regime ? Would we still observe the same behaviour ? It would be great to have more experiments controlling the data distribution and studying the impact on these critical periods.

- Other than the impact of data, the paper also misses the fact that a schedule is used on the learning rate and weight decay. Both are brought to 0 progressively throughout the training and that could be correlated with the critical period observed. I think some experiments with a fixed schedule should be conducted to remove this confounding factor.

- Other similar metrics that have the same objective of characterizing SSL features are not acknowledged. For example, RankMe or LiDAR.

- The introduction should say a little more about the experimental setup, in terms of dataset used and concrete results obtained.

### Questions
Is there a link between critical periods and double descent ?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper studies the training dynamics of self-supervised learning methods and how performance evolves both in domain and out of domain during training. To understand why OOD performance drops even though ID performance increases during training the authors people to study this behavior through the lens of critical periods. But using two different criteria, the authors are effectively able to detect when overspecialization starts to happen which helps select the best model for OOD performance, or a more balanced model.
Finally, the authors motivate and experiment with a distillation technique to help with overspecialization.

### Strengths
- The authors provide clear evidence of the studied problem, notably the drop in OOD performance during longer training

- The proposed metrics, in particular the one based on Fisher Information, correlate really well with this behavior. This leads to an effective method to perform early-stopping

- Experiments are performed at a good scale (R50/ViT-B, trained up to ImageNet for 1000 epochs) which adds to the relevance of the results

- The proposed distillation method is useful and very relevant to current SSL research. A similar problem was shown to be present in DINOv3[1], which concurrently proposed another solution based on earlier checkpoint distillation.

[1] Siméoni, Oriane, et al. "Dinov3." arXiv preprint arXiv:2508.10104 (2025).

### Weaknesses
1) Focus on Satellite dataset in the main paper. The same results as in the main paper are performed on ImageNet in the appendix but should be emphasised more in the main paper to appeal to a broader audience.

2) Throughout the paper, the considered evaluation is finetuning. However, the considered methods are more commonly used with lighter evaluations such as training a linear classifier. This would help make the results more relevant and may shed different insights.

### Questions
1) For Probe 1, how much do you think that the sensitivity is correlated with a lower learning rate later in training ?
2) Lines 254-255: when using noise during the deficit window, are data augmentation (jitter,crop,masking etc) still applied ? If not, how are the input image pairs constructed ?
3) Figure 4: All methods except VICReg have their Fisher Information drop to zero, do you have any intuition why it does not for VICReg ?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper aims to answer the following question: "how long should SSL models be pretrained?". They relate this problem to the notion of critical periods: where models exhibit high plasticity in early training stages, and go through a consolidation phase where OOD adaptability declines but ID performance improves. This phenomenon has been observed in supervised learning and this work claims to be the first systematic investigation of critical periods in SSL. They make three main contributions:
- how to do CP analyses for SSL without requiring labels
- how CP closure can guide the process of selecting intermediate checkpoints that show stronger OOD robustness
- propose a distillation technique that uses sweet spot CP checkpoints as teacher and overspecialized networks as student, which leads to improved OOD generalization while maintaining ID performance.

### Strengths
1. **Reformulation of CP for SSL**: This seems to be the main novelty of this work. To show critical periods in SSL, the authors have proposed two analyses techniques: (a) perturbations at different learning stages (b) FI matrix with respect to pretext tasks.
2. **Practical Impact**: Contributions in Section 4 (CPCS and CPSD) can potentially have good practical impact providing guidance to select checkpoints based on transferability tradeoff, and improving OOD robustness. 
3. **Experiments**: This work is backed well with strong experiments. They have considered two real-world datasets (IM-1K and fMoW-rgb) and four methods (SimCLR, VICReg, MAE, DINO). There're some DINOv2 results as well but I am not sure why it's only mentioned in Appendix.
4. **Reproducibility**: Hyperparameters and dataset details are well documented, and the methodology seems reproducible given the provided information.
5. **Quality & Clarity**: Overall, the paper is well-written and has a logical flow. The schematic in Figure 1 is particularly effective in summarizing the conceptual framework, which is later supported by quantitative results.

### Weaknesses
1. **Lack of Mathematical Rigor / Theoretical Depth**: The study is primarily empirical. While the FI metric offers some analytical grounding, the paper does not provide a principled explanation for why SSL exhibits critical periods or overspecialization. Some lightweight theoretical reasoning could strengthen the argument. 

2. **Issues with CP-guided self-distillation**:

    (a) L117: it is not explained why the authors chose to distill early layers only. It's only later in Section 4.3 where the rationale behind this choice is addressed. 

    (b) Figure 6 is not convincing enough to justify distilling early layers. I think it will be useful to verify their claims by distilling the entire network and comparing performance gains.

    (c) **Missing Ablations**: The number of layers distilled (L) and the distillation weight (λ) are unspecified and unexplored. Both likely affect results and reproducibility.

3. **Figure 6**: It is not clear what the authors imply by "stage". 
4. **SimCLR's behavior**: I appreciate the attempt to explain why SimCLR's critical period closes much later (L375-376). I think this section needs more clarification. What are the differences in objectives of all the methods taken into consideration that leads to this behavior?
5. **Definition of CP closure**: The paper briefly mentions detecting CP closure “L388- when the FI slope stabilizes (e.g., below a tolerance for p consecutive epochs)” (in Sec. 4.2). However, it is unclear whether this rule was actually used to determine the CP checkpoints reported in the experiments. For reproducibility and clarity, it would be valuable for the authors to specify the exact tolerance threshold and window length used in practice (if any).

### Questions
Please refer to weaknesses.

### Soundness
4

### Presentation
4

### Contribution
4
