# BrainAlign: Leveraging EEG Foundation Models for Symmetric, Interpretable Alignment with Visual Representations

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 4, 4

## Abstract
Custom electroencephalography (EEG) encoders trained on limited, task-specific data have restricted ability to learn generalizable, brain-like representations. We propose a representation-first alternative, leveraging a large-scale pretrained EEG foundation model (CBraMod) to learn brain-aligned representations. We introduce BrainAlign, a contrastive learning framework that uses a brain-inspired projection network to align EEG features with those from image encoders. On the challenging 200-way zero-shot visual object classification task, BrainAlign, when paired with a CORNet-S encoder, achieves a top-1 accuracy of 14.2\% and a top-5 accuracy of 37.9\% for EEG-to-image retrieval, performing competitively to prior baselines while reducing training time by 70\%. This computational efficiency is particularly crucial for developing the subject-specific models vital for practical EEG decoding. Additionally, the framework learns a highly symmetric alignment, achieving a 23.2\% top-1 and 54.7\% top-5 accuracy in the reverse image-to-EEG retrieval task. We observe a time-averaged RSA correlation (r = 0.365) with the neuro-inspired CORNet-S model, consistent with a moderately high degree of representational similarity. A post-hoc CCA-INLP analysis isolates a subject-agnostic subspace and, together with a semantic similarity evaluation, shows meaningful category structure yet residual cross-subject variability. Collectively, these results in performance, efficiency, and biological plausibility provide support for our representation-first approach. The resulting robust and symmetric representations can potentially be applicable to demanding downstream applications such as object classification, high-fidelity image decoding directly from brain activity, and real-time object disambiguation.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper builds upon a large-scale pretrained EEG foundation model (CBraMod) to learn brain-aligned representations. The proposed method 
BrainAlign is  a contrastive learning framework that uses a projection network to align EEG features with those from image encoders.  Improved accuracy over the existing methods is presented as well as a reduced training time.

### Strengths
The paper integrates existing techniques and addresses training efficiency in aligning EEG features with those of image encoders. the paper addresses relevant problem and provides a meaningful discussion on how the presented work relates to the state of the art. The paper is clearly written and experimental evaluation supports claims on approved accuracy.

### Weaknesses
The paper states to be biologically inspired but fails to give a detailed explanation of what exactly is biologically inspired. It claims robust performance, but given that the focus is on subject-dependent models, robustness could have be explained better.  I understand the robustness from the perspective of individual studies but how the robustness is actually tested even on these is not well described. The results are on-pair or better than comparable  approaches but the training converges after 60 epochs and 70%decrease in training time is reported.

### Questions
What is the evidence of the approach being biologically inspired?
How is robustness of the method tested?

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
2

### Summary
In this paper the authors contribute BrainAlign, a framework to learn a shared representation space of image and EEG signals, leveraging pretrained image encoders, EEG foundation models, and trainable projection modules to the common latent space. The authors evaluate BrainAlign across cross-modal, zero shot classification tasks, highlighting that it outperforms standard multimodal approaches (BraVL, NICE, NICE-GA), requiring less training time to achieve competitive results. Furthermore, the authors present extensive evaluations on the interpretability of the model and the quality of the learned representations, highlighting the connections to visual processing in the brain and the bidirectionally of the framework. Finally, the authors discuss the extent of subject-specific and subject agnostic information within the learned representation.

### Strengths
- Overall, the work presented in this papers is quite substantial, and clearly shows the effort done by the authors. In particular, I thoroughly enjoyed the substantial (and insightful) evaluations over the learned representations, present both in the main paper and Appendix.

- The idea of employing EEG foundation models for image retrieval tasks, instead of training from scratch, appears to be novel (to the best of my knowledge). Moreover, the use of a self-supervised loss for aligning the image and EEG representations is a sound methodology. The bidirectionality of the framework is also a nice bonus, yet I believe it to be a byproduct of the self-supervised loss employed, and not particularly novel (see, for example, [1]).

- I believe this paper to be of some significance to the community. The successful demonstration of the use of EEG foundation models for image retrieval tasks, as shown in this paper, can lead to the development of the cross-modal capabilities of these models to other modalities (such as sound, video). Furthermore, the discussion over the properties of the multimodal representation learned by BrainAlign can lead to the development of improved fine-tuning techniques of these foundation models, that target participant-specific information.

- Finally, the paper is well-written, without any major typo. The images are also of high-quality (yet could be slightly bigger to facilitate the interpretation). The document is also well structured, one exception being Section 4.4 and Section 4.5. which could benefit from a reformulation (see weaknesses below).

**References**:
- [1] Rajabi, Nona, et al. "Human-Aligned Image Models Improve Visual Decoding from the Brain." Forty-second International Conference on Machine Learning. 2025.

### Weaknesses
- While the presented framework is sound, it does suffer from a lack of novelty: the use of a self-supervised loss to align EEG signals and image representations encoded from pretrained visual encoders has been extensively demonstrated in literature before (see [1,2], as well as other references discussed by the authors in Section 2). The novelty of the work lies in the replacement of the fully-trainable EEG encoder with a pre-trained EEG encoder. However, to achieve reasonable performance, the pre-trained EEG also requires fine-tuning. While the authors highlight that this leads to a "70% reduction in training time (Line 316)", EEG encoders are usually of a much smaller complexity (e.g., in terms of parameters) than the image encoders, and their training time is usually (comparitavely) neglectable. What could be more interesting is, instead, to evaluate: (i) how much data does the fine-tuning process require against the training from scratch procedure? (ii) how much of the original encoder can remain frozen, and how much it needs to be fine-tuned?

- Another apparent disadvantage of the model is that the best performing framework proposed by the authors still underperforms in EEG-to-image retrieval against the NICE-GA (as shown in Table 1). That is not necessarily a problem, if the authors show that their approach improves other measures in comparison with NICE-GA. However, NICE-GA is not a baseline comparison of the extensive evaluations present in Sections 4.3 and 4.4, so it is not clear how they compare.

- While the extensive evaluations of the learned representations in Section 4.3 and 4.4. are welcomed and insightful, the presentation of these results do a disservice to the work shown. I would recommend the authors to unify and streamline their evaluation section, clearly highlighting the main takeaways of the different experiments. Currently, Section 4.4. appears to contain extra irrelevant experiments (the title of the subsection is "Additional Analyses"), which is obviously not the case.

- Some of the Figures could be substantially improved: Figure 4 does not include a legend to identify what are each of the lines, what is the colored area (std, 95% CI?), etc... Figure 5 is also too small, making the text unreadable, which is surprising since the authors still have available space in the document.

- I also found it disappointing that the main paper does not include a dedicated section to discuss the ethical considerations of this work. While the use of this technology in the future can have benefits for certain populations, it's also obvious that it can be used for nefarious purposes, and this should be discussed in the context of the paper.

**References**:
- [1] Song, Yonghao, et al. "Decoding natural images from eeg for object recognition." arXiv preprint arXiv:2308.13234 (2023).
- [2] Rajabi, Nona, et al. "Human-Aligned Image Models Improve Visual Decoding from the Brain." Forty-second International Conference on Machine Learning. 2025.

### Questions
- Can the authors present the ablation study described in the first point of the weaknesses? Also, can the authors present the full training time comparisons of the different models to substantiate the statement in Line 316?

- Can the authors compare their approach to NICE-GA in the tasks presented in Sections 4.3 and 4.4?

- In Figure 2 the authors show that the fine-tuning procedure is fundamental to achieve a more biological plausible distribution of weights across the different brain regions. Why this difference? Wouldn't the pretrained EEG model also have learnt a biological plausible distribution, from the large-scale EEG datasets it was trained on? I understand there are subject-specific changes, but wouldn't the "average distribution" learned by the foundation model also produce a similar distribution to the fine-tuned one presented in Figure 2?

- While it is understandable the differences in the results of Figure 3, it would be clarifying if the authors could elaborate on the connection between these differences and the temporal processing of visual information in the brain. Currently there is no explanation in the main text, beyond the statement: "our aligned representations captured significant, dynamically evolving neural information, mirroring the know temporal progression of the visual" (Line 371). Can the authors elaborate on this?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper used pretrained EEG foundation model to enhance the constrastive alignment between visual stimuli and brain responses for visual decoding. It gave interesting attempts to finetune pretrained model for other tasks. Apart from the decoding performance, they claimed that the training time was significantly reduced by 70% with the help of pretrained EEG encoder. The demonstrated the performance with eeg-to-image retrieval and also image-to-eeg retreival, and observed notable correlation with neuro-inspired image model.

### Strengths
1. It's a foundamental question that how to use pretrained EEG model in down-stream task. This paper focues on the problem and gave a good evaluation in the visual decoding with contrastive learning framework.
2. The set bidirectional task to test the representation provided by finetuned foundation model.
3. They gave clear results comparison to show the performance of the model without a overly fit results.

### Weaknesses
1. It's not clear how the work applied finetuning of the CBraMod and the specific computation cost reduction with the finetuning compared to train a new model.
2. It would be benifit to give a clear description on how to implement EEG-to-image and image-to-EEG tasks, as well as the meaning of the visualization (Fig. 2 and 3). It's a little blurred that what visual information obtained by the model. 
3. A summary of the improvement taken by including the pretrained model would help to figure the main contribution of the work. Would that help achieve better prediction results, lower computational cost, clearer brain pattern extraction?

### Questions
1. Would different finetunes have impact on the overall performance?
2. How did you use CBraMod here adjusting to the channel settings of the dataset used in this work?
3. What concolsion was give by the CCA-INLP analysis?
4. I still curious about the brain patterns achiveved by the framework, including frozen and finetuned CBraMod model.

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
2

### Summary
The paper proposes BrainAlign, a contrastive learning framework that leverages a pretrained EEG foundation model to align EEG features with visual representations from various image encoders . The authors claim that their model achieves biologically plausible alignment, symmetric bidirectional mappings (EEG↔image), and competitive results on the THINGS-EEG2 dataset with reduced training cost.

### Strengths
The paper addresses an important problem: improving EEG-vision alignment using foundation models.

### Weaknesses
1. The argument for biological plausibility is rather vague. The paper states that fine-tuned models show "biologically plausible attention patterns”. However, I would say these observations should be framed as interpretability, which is different from biological plausibility. The same for the use of the “mechanistic interpretability”  term.
2.  For the EEG to image direction, the model’s advantage over state-of-the-art methods is not clearly established. A proper statistical test (e.g., Wilcoxon signed-rank) should be reported for these comparisons, not only for fine-tuning vs. frozen variants but also versus existing baselines. The small improvement may not justify fine-tuning a model per subject unless there is clear evidence of superior performance or interpretability.
3. There is no comparison with other models for the image to the EEG direction. Although the authors acknowledge this in their limitations section, the subject-dependent setup is a major limitation.
4. While fine-tuning improves performance, the gain is modest. 
5 It is unclear whether the results presented, particularly in Tables 1 and 2, are based on a single run or averaged across multiple random seeds.

### Questions
See weaknesses

### Soundness
2

### Presentation
2

### Contribution
2
