# CAMDrop: Gradient-Guided Dynamic Feature Dropping for Multimodal Balanced Learning

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 2, 4, 4

## Abstract
Multimodal learning seeks to integrate complementary information from diverse modalities to enhance model performance. However, modality imbalance—where dominant modalities overshadow weaker ones—often hinders effective representation learning and generalization. Existing solutions are either gradient-based, which lack fine-grained control, or feature-based, which suffer from randomness and low interpretability. To address these challenges, we propose CAMDrop, a lightweight and plug-and-play strategy that suppresses dominant modality regions guided by class activation maps (CAMs). By leveraging GradCAM, CAMDrop identifies class-relevant spatial features and adaptively masks them based on instance-level importance, enabling semantically meaningful and dynamically adjusted suppression without altering model architecture or training objectives. Extensive experiments on three benchmarks, demonstrate that CAMDrop consistently improves accuracy and robustness, effectively mitigating modality imbalance while enhancing model interpretability. Beyond quantitative gains, CAMDrop provides qualitative insights into modality contributions, offering a transparent mechanism for balanced learning. We believe our method can serve as a practical component for future multimodal systems in applications such as emotion recognition, event localization, and human–computer interaction.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes CAMDrop, which balances multimodal learning by suppressing dominant modality regions using class activation maps (CAMs). This technique offers an interpretable and flexible intervention to imbalanced training and is shown to be effective on several audio-visual tasks across multiple training settings. Therefore, CAMDrop can be considered as a contribution to a generalizable and effective approach to mitigate modality imbalance.

### Strengths
- Paper is clearly written and easy to follow;
- Relatively extensive evaluation on audio-visual tasks with fair comparison to multiple existing methods and ablation on training settings, which shows convincing margins and thus effectiveness of the proposed approach;
- Insightful ablation study that shows how the CAMDrop balances the learning of individual modality throughout training;

### Weaknesses
- From a perspective of a general, practical approach to address modality imbalance, there should be more evaluation on the effectiveness of CAMDrop on other multimodal tasks instead of only focusing on audio-visual tasks; in particular, the paper is **missing evaluation on visual-language tasks** where there has been a wide body of work showing models (and especially large models) demonstrate strong bias towards the language modality with under-optimized visual learning;
- The usage of GradCAM, which establishes the importance attribution of features to the predicted labels, is only **weakly correlated with the motivation to emphasize the learning of the "underused but relevant" regions**. In particular, the attribution can only indicate areas that are "underused" but not necessarily "relevant". This may explain the drop in unimodal performance of the suppressed modality: if the model already learns the relevant regions, by masking out them in CAMDrop, it's possible that the model is forced to learn the underused but irrelevant ones;
- Incorrect citation format, please differentiate the use of \cite{...} and \citep{...};

### Questions
- The authors should include evaluations of CAMDrop on a set of more diverse multimodal tasks which include mainstream modalities like language, sensory data, etc. Also, current evaluations are only performed on small multimodal models. Does the same technique apply to large-scale models like VLMs, which are more widely used in applications?
- Does Figure 4 imply some kind of tradeoff in learning different modalities: by emphasizing the learning of the weak one, the model necessarily sacrifices its learning of the dominant one? If so, there should be a more comprehensive analysis of how the performance varies along the tradeoff frontiers to show that the technique can be leveraged to achieve optimal learning with certain imbalance ratio (in other words, the optimal learning is not necessarily achieved at perfectly balanced learning with ratio of 1.0, but may instead vary from tasks to tasks depending on domain knowledge).
- The paper mentions that CAMDrop "promotes multimodal synergy" but actually rebalancing the learning of different modalities is not very relevant to synergy, which defines the extra information that arises with both modality present and is missing with only one observed. In the case of this study, both modality are necessarily available, regardless of the use of CAMDrop, and there is no quantitative / qualitative evidence attributing the improved performance to synergy. Please avoid such over-claims.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
In this paper, the authors proposed CAMDrop, which aims to tackle the model imbalance problem in multimodal learning. CAMDrop works by first using GradCAM to identify regions within each modalities that matters most to the ground truth class, and then mask them based on a modality dominance ratio, and train the model with the partially-masked data. The proposed method can be applied on top of existing methods, and it is evaluated against baselines over 3 audio-visual datasets, where CAMDrop outperformed all baselines. Further analysis on hyperparameters and training settings are conducted, and the paper analyzed imbalance ratio over the modalities to show that CAMDrop indeed helps alleviate modal imbalance.

### Strengths
1. The proposed method is simple and straightforward, but manages to outperform all baselines across 3 audio-visual datasets.

2. The proposed method can be directly applied on top of existing methods/frameworks.

3. Further analysis demonstrates that the proposed method indeed reduces domain imbalance.

### Weaknesses
1. The generalizability of the proposed method is limited. While the title/abstract/introduction make it sounds like the method can be applied to various multimodal setting, .the proposed method's methodology was only defined with audio and visual modalities and.all experiments were conducted on 2-modal (audio+visual) tasks, so it is unclear if the proposed method is applicable at all to tasks with other combinations of modalities, or tasks that involve more than 2 modalities

2. The proposed method seems to be very sensitive to batch size increase, so when it is applied to larger datasets with more computing resources, either the training will be slow due to low batch size or the method's performance will suffer with larger batch sizes, so the method may not scale well with additional data and compute. The paper hinted that it is the ratio between learning rate and batch size that caused this issue, but the paper didn't include any empirical results to back up this claim. For example, if simply increasing batch size from 16 to 64 causes a large drop in performance, what would happen if you also increase learning rate by 4 times at the same time?

3. While the paper had an "ablation study" section, it only contained an analysis on the imbalance ratio between modalities, and didn't actually ablate on any of the method's design choices. Some design choices that needs to be ablated includes: (1) what if you apply masking to only one modality instead of both? (2) What if you performed GradCAM with the unimodal class logits instead of the multimodal ones? (3) What if dominance ratio is calculated independently for each data instance instead of averaged over a batch?

4. The paper claims that the proposed method is lightweight, but it does seem to add some additional computation overhead (e.g. the GradCam part and the masking resulting in 2 forward passes, as shown in Figure 2). The authors should include some computation time comparison to support the lightweight claim.

5. The authors left the ".git" folder inside the supplemental material, which contains information that could expose the authors' identities. Please be more careful in anonymizing the attached materials.

6. For some reason, the in-text citations within the manuscript do not have parenthesis around them, which makes reading a bit more difficult.

### Questions
1. How exactly are the unimodal classifiers (i.e. the linear layer that maps unimodal z to classification logits) trained? are they also trained with cross entropy loss jointly with the main multimodal loss?

2. In Eq 9, the dominance ratio calculation for the 2 modalities are not symmetric. What happens if you switch the two (i.e. first compute dominance ratio for audio as the arithmetic average over ratio of each data instance, then compute the visual ratio as the reciprocal)? Also, why not use geometric mean instead of arithmetic mean over the batch so that the computation of the two dominance ratios are actually symmetric?

3. Have you explored randomly masking out parts of the high-gradcam regions rather than always masking out all regions whose gradcam is higher than the threshold? This may serve as another layer of augmentation that could increase model generalizability.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes CAMDrop, a gradient-guided masking strategy to mitigate modality imbalance in multimodal learning. The method uses GradCAM to identify class-relevant dominant regions and suppresses them during training to encourage more balanced cross-modal learning. CAMDrop requires no modification to network architecture or loss function. The authors evaluate their approach on three emotion- and event-related audio-visual datasets (CREMA-D, AVE, and Kinetics-Sounds), demonstrating performance improvements over baseline methods.

### Strengths
+ The experiments are fairly comprehensive, including multiple datasets, comparisons across several baselines, and analyses of hyperparameters and fusion strategies.  
+ The paper is clearly written and well-structured, making it easy to follow the proposed approach and experimental design.

### Weaknesses
- The core methodological contribution is limited. The proposed approach mainly masks “important” regions identified by GradCAM, which conceptually resembles existing feature dropout or attention suppression methods. The novelty beyond prior work (e.g., OPM, OGM-GE) is incremental.  
- The paper lacks discussion and experiments on the balance between randomness and importance in masking. By deterministically removing high-importance regions, the model might overfit to complementary regions without verifying whether randomization could yield similar or better robustness.  
- Experimental validation is restricted to emotion and event recognition datasets within the audio-visual domain. There is no evidence of generalization to other modalities (e.g., text–vision), limiting claims about the method’s universality.  
- The baselines used for comparison are relatively outdated.
- While interpretability is mentioned as an advantage, qualitative analyses are limited to simple GradCAM visualizations without deeper evaluation of semantic consistency or stability.

### Questions
see weakness

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces CAMDrop, a lightweight, plug-and-play strategy designed to address modality imbalance in multimodal learning by semantically suppressing class-relevant regions from dominant modalities. CAMDrop leverages GradCAM to identify and adaptively mask spatial features that are overly influential for class predictions, encouraging the model to utilize under-represented modalities. The method requires no architectural or loss function changes, is validated across several audio-visual benchmarks, and claims improvements in performance, robustness, interpretability, and modality balance.

### Strengths
1. The paper's core motivation is clear and compelling. It correctly identifies a key limitation of prior feature-dropping methods like OPM —their randomness . The idea of replacing this randomness with _semantic guidance_ by using class activation maps (GradCAM) to decide _what_ to drop is a novel and intuitive contribution.
2. The method demonstrates strong and consistent performance gains. Table 1 shows that CAMDrop outperforms all baselines, including recent strong methods like OPM-GE, OPM , and PMR , across three different datasets (CREMA-D, AVE, KS) and three different fusion architectures (Concat, Sum, FiLM).
3.  The proposed method is conceptually simple and presented as a plug-and-play module that modifies the forward pass, requiring no changes to the loss function or network architecture.

### Weaknesses
1. The claim of CAMDrop being "lightweight"  appears to be inaccurate and is a major weakness. The method requires generating a GradCAM heatmap for each modality _on every training iteration. The paper completely omits any analysis (e.g., training time, FLOPs) of this significant overhead.
2. Figure 3 shows that the key hyperparameter $r_{max}$ is sensitive and its optimal value varies significantly across datasets.This reduces the "plug-and-play" nature of the method, as it requires careful tuning for each new task.
3. The most severe flaw in the paper is the complete absence of theoretical justification for why GradCAM-based masking should solve modality imbalance. The authors claim that "GradCAM identifies class-relevant spatial features and adaptively masks them based on instance-level importance," but provide no analysis showing how this directly addresses the core problem of modality imbalance.

### Questions
1. Could the authors should compare with the most recent and relevant methods ,provide a detailed analysis of the results, and explain why CAMDrop works better than existing methods? especially compared to the 2025 methods.
2. Could the authors provide a clear theoretical justification for why GradCAM-based masking addresses modality imbalance? Without this, the paper's core contribution is unfounded.
3. Could the authors please provide a quantitative analysis of the training overhead (e.g., wall-clock time per epoch, GFLOPs) introduced by CAMDrop? How do you reconcile the "lightweight" claim with the requirement of an extra backward pass per modality on every iteration?

### Soundness
2

### Presentation
2

### Contribution
3
