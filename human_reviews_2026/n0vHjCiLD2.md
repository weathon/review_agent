# You Point, I Learn: Online Adaptation of Interactive Segmentation Models for Handling Distribution Shifts in Medical Imaging

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 8, 4

## Abstract
Interactive segmentation uses real-time user inputs, such as mouse clicks, to iteratively refine model predictions. Although not originally designed to address distribution shifts, this paradigm naturally lends itself to such challenges. In medical imaging, where distribution shifts are common, interactive methods can use user inputs to guide models towards improved predictions.
Moreover, once a model is deployed, user corrections can be used to adapt the network parameters to the new data distribution, mitigating distribution shift. Based on these insights, we aim to develop a practical, effective method for improving the adaptive capabilities of interactive segmentation models to new data distributions in medical imaging.  Firstly, we found that strengthening the model's responsiveness to clicks is important for the initial training process. Moreover, we show that by treating the post-interaction user-refined model output as pseudo-ground-truth, we can design a lean, practical online adaptation method that enables a model to learn effectively across sequential test images. The framework includes two components: (i) a Post-Interaction adaptation process, updating the model after the user has completed interactive refinement of an image, and (ii) a Mid-Interaction adaptation process, updating incrementally after each %user 
click. Both processes include a Click-Centered Gaussian loss that strengthens the model's reaction to clicks and enhances focus on user-guided, clinically relevant regions.
Experiments on 5 fundus and 4 brain‑MRI databases show that our approach consistently outperforms existing methods under diverse distribution shifts, including unseen imaging modalities and pathologies.
Code: https://github.com/WenTXuL/OAIMS

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents an online adaptation framework for interactive medical image segmentation to address distribution shifts.
It introduces 2 types of adaptation: Post-Interaction and Mid-Interaction adaptation. A Click-Centered Gaussian loss is proposed to control how the model adapts to signals around user clicks.

### Strengths
The proposed framework, combining Post- and Mid-Interaction adaptation, along with the problem formulation, which integrates model updates with interactive segmentation, is interesting.

The CCG loss is a reasonable design choice and can effectively prevent the model from learning pixels' labels irrelevant to the click, focusing updates on relevant regions.

Experiments are comprehensive, covering fundus and brain MRI datasets. Experiment results are promising, demonstrating significant performance gains over existing methods, especially under large distribution shifts

### Weaknesses
It would be helpful if the authors could provide more implementation details on finetuning e.g., whether the entire U-Net is made trainable during adaptation or if some layers are frozen.

It would be helpful if the authors could explain why focal loss is used. Is the training data overwhelmed with easy-to-learn regions?

Mid-Interaction adaptation seems intimidating, because the user has to wait for the model to be updated before providing another click, which could introduce latency and disrupt workflow.

With only one gradient update step, it is possible that the updated model still predicts the wrong label for the user-clicked region. It would be helpful if the authors could provide some discussion.

The method's effectiveness relies on a reasonably good pseudo-ground-truth. In cases with very few clicks or extremely poor initial predictions, this assumption may not hold.

### Questions
Please refer to the weaknesses.

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
5

### Summary
This paper proposes an online adaptation framework for interactive medical image segmentation under distribution shift. The method, termed OAIMS, introduces two adaptation mechanisms, Post-Interaction (PI) and Mid-Interaction (MI), that update model weights during and after user interaction sessions. The approach employs a Click-Centered Gaussian (CCG) loss to strengthen responsiveness around user clicks and to improve model generalization across unseen modalities or pathologies. The authors evaluate their method on multiple fundus and brain MRI datasets, demonstrating superior performance over existing online adaptation methods such as IA+SA and TSCA.

### Strengths
1. The paper tackles a practically important problem, handling distribution shifts in medical image segmentation, by combining interactive segmentation with online adaptation.
2. The proposed OAIMS framework, integrating Post-Interaction and Mid-Interaction updates, is conceptually clear and technically well-structured.
3. The Click-Centered Gaussian (CCG) loss is an intuitive and effective design that enhances model responsiveness around user inputs.
4. Experiments cover a wide range of datasets (five fundus and four brain MRI), demonstrating consistent cross-domain generalization.
5. Comprehensive ablation studies verify the contribution of each component and show the robustness of the method.
6. The paper is clearly written, well-organized, and supported by informative visualizations that make the method easy to follow.

### Weaknesses
1. The method is only validated on a U-Net backbone; generality to Transformer-based or large segmentation models (e.g., SAM) is not demonstrated.
2. The experiments lack a practical interaction-efficiency metric (e.g., number of clicks to reach 80% IoU / 80% Dice), which is important for interactive workflows.
3. The pseudo–ground-truth assumption is strong, and if the model’s initial predictions are highly erroneous, online updates may accumulate errors and cause distributional drift.
4. The baseline comparisons omit the latest (2025) interactive segmentation methods; the novelty claim for CCG is overstated given related prior work (e.g., methods that weight click neighborhoods or modify attention such as AdaptiveClick (TNNLS, 2025)).
5. Computational cost and latency of online updates are not reported, leaving clinical feasibility unclear.

### Questions
1. How computationally expensive is the online adaptation per image? Could this realistically run in real-time clinical use?
2. Have the authors tested their method with actual human annotators rather than simulated clicks?
3. How sensitive is the method to noisy or inconsistent initial inputs?
4. Could the pseudo–ground-truth mechanism lead to error accumulation (catastrophic drift) during long adaptation sequences?
5. How does OAIMS behave under minor distribution shifts, does continual adaptation degrade performance on source-like data?
6. Compared to strong non-online interactive models (e.g., nnInteractive), does online learning provide a clear advantage?
7. The method focuses on point clicks, can CCG also adapt to other interaction modalities such as bounding boxes or scribbles?
8. The paper claims PI adaptation provides benefits even when MI has already been applied; however, if MI is trained on one modality, can MI alone be applied on a new modality without retraining PI and still be effective?

This is a solid paper, and if the authors can address these questions, I would consider increasing my score.

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
This paper proposes an online learning technique for interactive image segmentation in the medical domain, where they specifically propose it for domain adaptation (which is one of the main problems in medical imaging). In particular, the main innovation seems to be a Click-Centered Gaussian loss function, which is a loss which acts on interactive clicks and makes the model react more quickly to user-clicks than previous methods. This loss is applied in a pre-training phase (out-of-domain pre-training phase), a Mid-interaction phase (i.e. between user-clicks in a single image), and a Post-interaction phase (i.e. after a single image is segmented). A well designed set of experiments show that the proposed loss is important in all three training phases, that their effects are cumulative, and that their final method outperforms the state-of-the-art.

### Strengths
* The Click-Centered Gaussian loss function is simple: It takes a user click, does a Gaussian convolution, and uses this as weights for a cross-entropy loss. 
* The CCG loss makes sense: by focusing only on the area around the user click, it encourages the model to quickly respond to user interactions. In contrast to previous user-click losses which only operate on a single pixel, the Gaussian ensures that a larger portion of the image is taken into account which experiments demonstrate to be beneficial. Note that I prefer simple and effective methods, since these are more likely to get actual adoption in the field.
* The authors compare with two previous methods (IA+SA and TSCA) on six different datasets and convincingly show that their method works better.
* Stage two (page 4) of fine-tuning with multiple correction points is well explained how it avoids trivial network updates.
* There are a good amount of ablation studies to better understand the behavior of the proposed method.

### Weaknesses
* There is not a single study with a real human: typically a simulated user behaves differently from a real user. The paper would improve if there is at least on real human study on a single dataset where the authors show their method works. This would also entail having a variable number of user interactions during annotation. Ideally this should be compared to a similar experiment using the best competitive method TSCA.
* The sigma of the Gaussian is not ablated. Theoretically, a sigma close to zero goes back to having the loss only on a single point. A sweep over sigmas could help understand how much the ‘Gaussian’ part of the loss really contributes.
* There seems to be a contradiction in the writing: In stage 2 the authors argue that it is not possible to reuse the user’s original correction clicks to update the model as the model already yielded the final prediction using those corrections, leading to trivial updates. Then in stage 3 they seem to do exactly that: using the last click c_t to reinforce the resulting prediction p_t^initial. While I think both are valid, the way it is written now is confusing since it directly contradicts the previous paragraph. Also, it took me quite some time to figure out why p_t^initial has the ^initial modifier. Some careful rephrasing in both stage 2 and 3 are needed to solve this problem, but this should be easily doable.
* Mid-interaction and post-interaction learning are not new (e.g. IA+SA). But this method works better.
* Using Gaussians for points is also not new, but it is shown to be very effective in this context.

### Questions
* Is it possible to ablate the Sigma for the Gaussian points?
* Is it possible to do a small user study as described above?

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
This paper addresses distribution shifts in medical image segmentation by using interactive segmentation as an online adaptation mechanism. The authors propose a framework where user click inputs not only guide predictions but also serve as signals for adapting model parameters during deployment. The method consists of two adaptation components: Post-Interaction adaptation (updating after complete image refinement) and Mid-Interaction adaptation (updating after each individual click), both incorporating a Click-Centered Gaussian loss that enhances the model's responsiveness to user guidance. They state that the user-corrected outputs can be treated as pseudo-ground-truth labels, enabling the model to continuously learn from sequential test images without additional manual annotations. The framework strengthens click responsiveness during initial training and focuses learning on clinically relevant regions indicated by user interactions. Comprehensive experiments across various medical imaging datasets demonstrate consistent improvements over existing methods in handling distribution shifts, including unseen imaging modalities and novel pathologies. This work aims to provide a practical solution for deploying segmentation models in medical imaging environments where domain shifts are common and user interaction is already an integral part of the clinical workflow.

### Strengths
The paper’s direction is interesting, and treating the post‑interaction, user‑refined model as a pseudo ground truth is a thoughtful design choice.

The evaluation spans multiple levels of distribution shift and reflects realistic clinical scenarios in medical imaging.

### Weaknesses
The submission would benefit from greater technical depth, as the current contribution focuses primarily on the overall framework and the click‑centered Gaussian loss, which, on its own, is not fully convincing.

Moreover, I attempted to explore the advantages of the proposed method further. I have been trying to consider whether simply using an unsupervised loss based on the user’s inputs during inference would have improved the performance.

### Questions
The role and necessity of multiple correction points are unclear, especially if Stage‑1 user clicks are already assumed to be pseudo ground truth.

The introduction of artificial clicks risks injecting label noise during re‑training under the stated loss functions, and this potential degradation should be analyzed and controlled.

The click‑centered Gaussian loss lacks sufficient technical detail, and the paper does not provide empirical insights or ablations to explain its behavior and sensitivity.

The use of pretraining and its fidelity to real deployment are under‑specified, including whether the source model was trained with extensive click supervision and which loss functions were used.

A concise algorithmic description for both source training and test‑time inference with precise mathematical notation would significantly improve reproducibility.

Extending results to adaptation‑oriented segmentation benchmarks would strengthen claims of generality and applicability.

In Table 5 for BraTS, the inclusion of CCFL yields a large jump from 46.6 to 81.9, whereas similar gains are not observed elsewhere and are only marginal on BraTS T2, which warrants deeper analysis.

Two additional baselines would strengthen the study, namely standard cross‑entropy and entropy‑minimization losses applied at test time with pseudo labels or model predictions.

Clarification is needed on which network layers are updated during adaptation, given the combined image‑and‑click inputs and their contribution to re‑training effectiveness.

### Soundness
2

### Presentation
2

### Contribution
1
