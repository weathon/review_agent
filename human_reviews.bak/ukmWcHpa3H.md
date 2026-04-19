# Meta-weighted Diffusion Model for Reliable Online Surgical Phase Recognition

- Decision: Reject
- Scores: 6, 6, 5, 6

## Abstract
Surgical phase recognition has drawn great attention most recently thanks to its potential downstream applications closely related to human life and health. Despite deep network-based models have made significant advancement in capturing discriminative long-term dependency of surgical videos to achieve improved recognition, they seldom account for exploring and modeling uncertainty of surgical videos, which should be crucial for reliable surgical phase recognition. we categorize the sources of uncertainty into two types, imbalanced phase distribution and low-quality image acquisition, which are inevitable in surgical videos. To address this pivot issue, we introduce a meta-weighted diffusion model (MetaDiff) to take full advantages of meta-learning and deep generative model in tackling uncertainty. For uncertainty caused by image quality, we present a classifier-guided diffusion model to produce countable denoised recognition results, making it possible to measure uncertainty using statistical tools for each video frame. For uncertainty caused by phase distribution, we propose a meta-weighted objective function to optimize the classifier-guided diffusion model, making the classification boundary robust against surgical video uncertainty.
We demonstrate outstanding ability of our model through comprehensive benchmarks on Cholec80, AutoLaparo, M2Cai16, and CATARACTS. Experimental results reveal that MetaDiff significantly outperforms state-of-the-art methods, separately achieving accuracies of $95.3\%$, $85.8\%$, $92.2\%$, and $85.1\%$ on Cholec80, AutoLaparo, M2Cai16, and CATARACTS.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents Meta-weighted Diffusion Model (MetaDiff) for reliable online surgical phase recognition. MetaDiff aims to enhance phase recognition accuracy by addressing uncertainties inherent in surgical videos, primarily due to imbalanced phase distribution and variable image quality. The model integrates a Classifier-guided Diffusion Model (CDM) to manage noise in predictions, and a Meta-weighted Optimization Algorithm (MOA) for reweighting, thus focusing on minority phases that might otherwise be underrepresented. Experimental evaluations on four surgical datasets—Cholec80, AutoLaparo, M2Cai16, and CATARACTS—demonstrate MetaDiff’s strong performance in terms of accuracy and uncertainty estimation, outperforming existing models in accuracy metrics.

### Strengths
(1) The work is innovative in combining meta-learning and diffusion models to tackle uncertainties in surgical video recognition, particularly focusing on imbalanced data issues and varying image quality. 

(2) The paper is methodologically rigorous, with a thorough background in generative models, uncertainty quantification (e.g., Prediction Interval Width and Paired Two-Sample t-Test), and the use of meta-learning for handling imbalanced datasets. The proposed MetaDiff model and its evaluation are well-supported with extensive experiments on diverse surgical datasets, showing notable improvements over baseline methods.

### Weaknesses
(1) How does the classifier-guided diffusion process handle scenarios where phase boundaries are ambiguous, and does this affect MetaDiff’s stability across various surgical tasks?

(2) Could the meta-weight net introduce biases if specific phases dominate the meta dataset? If so, how is this mitigated?

(3) Given that uncertainty estimation was key to the model's objectives, would there be cases where the uncertainty metrics (PIW and PTST) lead to false negatives or positives?

(4) The author could try conducting experiments on a larger-scale and more fine-grained phase recognition dataset, such as OphNet
(https://github.com/minghu0830/OphNet-benchmark).

### Questions
Is there potential to generalize MetaDiff to other medical image tasks that face similar uncertainty issues, or are the diffusion model adjustments tailored specifically to the temporal structure of surgical videos?

### Soundness
4

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
3

### Summary
This paper identifies the unsolved problems of uncertainty caused by low-quality images and imbalanced phase distribution in surgical phase recognition. It introduces a meta-weighted diffusion model (MetaDiff) as a solution. MetaDiff integrates a classifier-guided diffusion model (CDM) that conditions the diffusion process on class predictions. Additionally, it employs a meta-weighted objective function (MOA) during optimization to learn weights that rebalance the training loss, effectively mitigating the impact of imbalanced phase distribution throughout training. Extensive experiments across four benchmarks demonstrate the promising performance of this approach.

### Strengths
* While previous works focus on learning strong and long-term spatial-temporal representations, this paper is the first to identify the uncertainty problem in surgical phase recognition.This challenge could have significant implications for deep learning applications in surgical settings.
* The application of meta-learning to rebalance class weights is an elegant approach.
* The results show that the proposed approach effectively improves the performance without increasing the training time.
* Experiments are extensive.

### Weaknesses
The paper's writing is often unclear, and clarity needs improvement to enhance readability and comprehension.

Some confusion and suggestions for the writing:
* In lines 38 and 39, mentioning the objective in the introduction in a mathematical format without defining the variables ($L$, $C$, $I$) is confusing.
* Figure titles should only describe the corresponding figure, while the objective is included in the title of Figure 1, which is irrelevant to the figure content. Additionally, the two images in the black solid boxes of each example are not explained.
* citation format makes the text a bit hard to read. It would be good if the authors could check the ICLR citation format and make updates to the paper.
* usages of notations are inconsistent, e.g., i in Eq. (4) and $i$ in the text (line 141).
* multiple grammar errors, e.g., line 157 " CDM is consisted of a ..." -> " CDM consists of ..."
* line 160-161 "Notably, ... which are **modestly collected** by the MOA used to learn parameters of CDM" is confusing. What does "**modestly**" mean in this context?
* line 207-208 "we use model the diffusion process endpoint through the incorporation of the conditional embedding.. " is unclear.
* in Figure 2, CDM and MOA should be illustrated clearly (put CDM's learning process in one box potentially?), and in the prediction process, replacing the DM component with "......" makes the whole process inconsistent. Also, the CDM learning process should be illustrated more clearly.
* in Section 3.2, video frames and clips are used interchangeably, leading to minor confusion.
* in Table 3, LoViT (Pr) on AutLapro is the best while not bolded, and the corresponding description in lines 354-355 is not accurate. 
* Table 6 title "DM" -> "CDM"

### Questions
* I don't see how the CDM addresses the uncertainty introduced by image quality. It appears that the conditioned diffusion process primarily aligns the class inference with the provided ground truth labels. Could you elaborate on this aspect further?
* Could you provide details on the training time and memory usage? Will the MOA and CDM significantly increase computation time?

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
5

### Summary
This paper addresses the challenge of uncertainty in surgical phase recognition by proposing a meta-weighted diffusion model (MetaDiff) that leverages meta-learning and classifier-guided diffusion for improved reliability. The model is designed to address issues caused by imbalanced phase distribution and low-quality image acquisition, common in surgical videos, which can lead to unreliable phase recognition outcomes. MetaDiff combines a classifier-guided diffusion model (CDM) with a meta-weighted optimization algorithm (MOA) to manage these uncertainties, presenting significant improvements in accuracy over previous methods on benchmark datasets, including Cholec80, AutoLaparo, M2Cai16, and CATARACTS.

### Strengths
1.	The integration of a diffusion model (specifically, a classifier-guided diffusion model) for handling uncertainty in predictions is innovative. By generating multiple denoised trajectories for each frame, the model can assess the confidence of predictions based on statistical measures like Prediction Interval Width (PIW) and the Paired Two-Sample t-Test (PTST). This enables more robust decision-making, especially in high-stakes scenarios like surgical phase recognition where reliability is critical.
2.	The use of a meta-weighted optimization algorithm (MOA) that dynamically re-weights the loss function for each video frame is an effective strategy to counteract data imbalance across phases.
3.	The paper uses a carefully chosen ConvNeXt backbone paired with LSTM to capture long-range dependencies in surgical video frames without leaking future information, crucial for real-time phase recognition. This pairing is technically sound as ConvNeXt focuses on spatial features, while LSTM efficiently captures temporal dynamics.

### Weaknesses
1.	The performance of MetaDiff is partially dependent on ConvNeXt+LSTM pre-training. For tasks beyond surgical videos, this dependence could limit the model’s adaptability, especially if ConvNeXt features do not generalize well.
2.	Although MetaDiff is well-evaluated on four datasets, these datasets cover similar surgical contexts. The model’s robustness to highly variable surgical environments (e.g., laparoscopic vs. robotic vs. open surgery) remains untested. 
3.	High dependency on accurate timestamp annotations. Surgical video data often contains inaccurate timestamp annotations, especially when manually labeled. And underexplored sensitivity to non-surgical contextual information, some frames may contain non-surgical context (e.g., camera adjustments or the organ scene is shown, but no procedure is performed).
4.	Some new but not necessarily SOTA works could also be included for comparison, such as：
[1] SurgPLAN++: Universal Surgical Phase Localization Network for Online and Offline Inference. https://arxiv.org/pdf/2409.12467
[2] SR-Mamba: Effective Surgical Phase Recognition with State Space Model. https://arxiv.org/pdf/2407.08333
[3] SPRMamba: Surgical Phase Recognition for Endoscopic Submucosal Dissection with Mamba. https://arxiv.org/abs/2409.12108

### Questions
please check the Weakness section

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper addresses a highly relevant problem in surgical phase recognition, emphasizing its importance due to the direct impact on human life and health. This underscores the paper's significance and the potential real-world impact of the proposed solutions. The authors effectively highlight two key sources of uncertainty—imbalanced phase distribution and low-quality image acquisition—which are critical but often overlooked issues in surgical video analysis. Recognizing these challenges and focusing on their mitigation shows a deep understanding of the domain's requirements. The introduction of a meta-weighted diffusion model (MetaDiff) is an innovative approach.

### Strengths
The use of meta-learning combined with a diffusion model to address uncertainties is unique and relevant, particularly for medical applications where reliability is crucial. The paper provides an extensive evaluation on four benchmark datasets (Cholec80, AutoLaparo, M2Cai16, and CATARACTS). This demonstrates the generalizability and robustness of the proposed model across a diverse range of surgical procedures, enhancing the credibility of the results. Experimental results indicate that MetaDiff outperforms state-of-the-art methods, achieving impressive accuracy improvements across all tested datasets. This supports the effectiveness of the proposed method and its potential for real-world applications.

### Weaknesses
1. The classifier-guided diffusion model is a core part of the approach, yet its description lacks sufficient technical depth. More explanation on how it specifically handles uncertainty caused by image quality, along with any potential limitations, would make the model easier to understand and evaluate.
2. Although the model performs well on surgical video benchmarks, there’s limited discussion on its adaptability to other medical video analysis tasks (e.g., surgical process planning) or datasets with differing visual and temporal characteristics. Expanding on this aspect could strengthen the broader applicability of MetaDiff.

### Questions
1.Guiding diffusion model training with classification is not a new approach; for example, in "Pdpp: Projected Diffusion for Procedure Planning in Instructional Videos and See" and "Predict, Plan: Diffusion for Procedure Planning in Robotic Surgical Videos", task and phase class information is used to guide planning task learning via diffusion models. Are there similarities or differences in using classifiers as guides for diffusion models between phase recognition tasks and planning tasks?
2. In Figure 2, the input consists of three consecutive frames. Are there significant differences among these frames? Given the typically long annotated duration for each phase, the visual information across different sets of three frames should vary considerably. Is there a mechanism to assess this variability?

### Soundness
3

### Presentation
2

### Contribution
3
