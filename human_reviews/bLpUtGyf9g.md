# Boundary Denoising for Video Activity Localization

- Decision: Accept (poster)
- Scores: 3, 8, 6, 6

## Abstract
Video activity localization aims at understanding the semantic content in long, untrimmed videos and retrieving actions of interest. The retrieved action with its start and end locations can be used for highlight generation, temporal action detection, etc. Unfortunately, learning the exact boundary location of activities is highly challenging because temporal activities are continuous in time, and there are often no clear-cut transitions between actions. Moreover, the definition of the start and end of events is subjective, which may confuse the model. To alleviate the boundary ambiguity, we propose to study the video activity localization problem from a denoising perspective. Specifically, we propose an encoder-decoder model named DenosieLoc. During training, a set of temporal spans is randomly generated from the ground truth with a controlled noise scale. Then, we attempt to reverse this process by boundary denoising, allowing the localizer to predict activities with precise boundaries and resulting in faster convergence speed. Experiments show that DenosieLoc advances 
several video activity understanding tasks. For example, we observe a gain of +12.36% average mAP on the QV-Highlights dataset.
Moreover, DenosieLoc achieves state-of-the-art performance on the MAD dataset but with much fewer predictions than others.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors propose a method for boundary denoising to tackle video activity localization. A model architecture DenoiseLoc is proposed, together with a boundary denoising training method. The authors argue that a single step denoising is better than the diffusion process with multiple steps. Experiments show some improvement over the previous state-of-the-art.

### Strengths
The experimental results show some improvements over the previous state-of-the-art.

### Weaknesses
Most importantly, the method part is not clear. It is rather hard to follow through most of the written parts. Figure 2’s caption also does not provide a clear overview of the proposed method and the novel aspects.


In Figure 2, it is rather confusing what the pipeline actually is. For instance, the ground truth span/noise injection part should definitely not be part of the inference pipeline. So, it is not clear what is done during the inference process.


Suddenly, in 3.2.2, the dynamic convolution is used without much explanation. Why is it important to the proposed design? What is the dynamic convolution exactly doing, and why no other design can be used? It is not well motivated.


The boundary denoising training part in 3.2.3 is not clear at all. How the method works, what loss is used, where the loss is applied, and why it is designed this way is not clear. Why do we need to divide into two different groups? How does the model use both of them during training?

Importantly, since boundary denoising has been widely explored, what are the further insights that make the proposed method more effective than previous works? This has not been clearly expressed.




Experimentally, there are also some parts not well established.

Most importantly, it is very strange to me why adding diffusion will lead to performance drops. Furthermore, the more steps used, the worse the performance seems to get. This is totally different from what is usually observed in many diffusion-based works (for generation and for prediction tasks). Usually, the benefit of using a single step is only for efficiency purposes. Furthermore, the given reason is also not convincing. It would be good if the authors provide a lot more details about how diffusion is used, and more qualitative/quantitative evidence to substantiate this claim, since it is quite a strong and counterintuitive claim.


It seems that more ablations are required for various designs, for example the various designs in denoising training. But, currently the method is too unclear for me to suggest concrete settings.







Note that there are some mistakes with the spelling/formatting. This does not affect the score. Some examples are:

Pg 2 bottom: citation formats are mixed up, and all the authors of the papers are listed in the citation.
Pg 1 and Pg 2: “QV-Hightlights”
Pg 9: “prediction the denoised proposal”, “more discussion”

Throughout, please standardize as DenoiseLoc (instead of denoiseloc at several parts).

### Questions
Apart from the above concerns, some other specific questions are below.


1)	Could the authors provide the time gains from using a single denoising step against multiple?
2)	Could the authors provide the model size and time gains as compared to previous state-of-the-art methods?
3)	In table 4, when the dB measure for noise is used, what exactly does it mean in this context?

### Soundness
1 poor

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposed an encoder-decoder model, namely DenoiseLoc, for video activity localization. DenoiseLoc introduces a boundary-denoising paradigm to address the challenge of uncertain action boundaries. DenoiseLoc leverages across modalities in the encoder and progressively refines learnable proposals and noisy ground truth spans in decoder layers. Extensive experiments on standard benchmarks demonstrate the effectiveness of the proposed DenoiseLoc.

### Strengths
- A novel boundary-denoising paradigm is proposed to address the challenge of uncertain action boundaries in video activity localization task.
- Extensive experiments on standard benchmarks demonstrate the effectiveness of the proposed DenoiseLoc.
- It is interesting to find that satisfactory performance can be achieved with very few proposals and very few denoising steps.

### Weaknesses
-  Lack of visual analysis. It would be helpful to understand the properties of the proposed method if some cases can be visually analyzed.
- "DenoiseLoc" and "denoiseloc" are used interchangeably, which confuses readers. It is recommended that all be changed to "DenoiseLoc".

### Questions
Please refer to Weaknesses for more details.

### Soundness
4 excellent

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper tackles an important and common challenge of boundary ambiguity in the video action localization task. The authors adopt the encoder-decoder framework as DETR for embedding video/caption features and predicting the boundary locations. The proposed denoiseloc aims at regress more precise boundaries by noise injection. Extensive experiments to demonstrate the effectiveness of denoiseloc.

### Strengths
+ The inspiration of boundary-denoising training approach has good novelty.
+ This paper is well-organized and the proposed method achieves good performance.

### Weaknesses
- This paper uses a complex symbol system, which makes it difficult to read. \epsilon presents the number of fixed consecutive frames and then it presents a vector of a span. n, which represents a quantitative index, is sometimes a subscript and sometimes a superscript.
- The process of denoising is unclear. Which loss function is used for boundary denoising? The core technology is a proposal augmentation strategy to obtain more candidate proposals for training?
- Missing related works of boundary ambiguity and temporal grounding.

Wang Z, Gao Z, Wang L, et al. Boundary-aware cascade networks for temporal action segmentation[C]. ECCV2020.

Xia K, Wang L, Zhou S, et al. Learning to refactor action and co-occurrence features for temporal action localization[C]. CVPR2022.
- Typo. L5 of Sec. 3.2.

### Questions
- What is the definition of an action span or temporal span?
- What do 0.25 and 0.75 mean in the Sec. 3.2.3? Negative proposal set is from the inside or outside of the ground truth?

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper tackles the problem of video activity localization, specifically given language descriptions. The main challenge of this task is boundary ambiguity caused by the annotator subjectivity and the smoothness of temporal events. To this end, the authors design a novel framework, named denoiseloc, aiming to progressively refine the moment predictions. To facilitate the model training, boundary-denoising training scheme is adopted, which encourages the decoder to reconstruct ground truths from the noisy moments. In the experiments on two benchmarks, MAD and QVHighlights, the effectiveness of the proposed method is validated.

### Strengths
+ The paper is well-written and easy to follow with good readability.
+ The figures well represent the proposed method, helping the understanding.
+ The proposed approach surpasses the strong competitors on both benchmarks.
+ The comparison between Denoiseloc and Diffuseloc is interesting, and brings valuable insights.

### Weaknesses
- Some important details of the method are missing. In its current form, the information about the model is insufficient in the manuscript.

(1) DETR-like approaches conventionally adopt the moment representation of (center, width). In contrast, the authors stated that they express a moment as start and end. In this case, the start position can be predicted as a larger value than the end position. How do the authors handle this?

(2) The details of temporal ROI alignment and DynamicConv layer are missing. I would like to suggest the authors to include graphical illustrations of these operations at least in the appendix for help the understanding.

(3) In the boundary-denoising process, the model generates two types of noise-injected proposals, i.e., positive and negative sets. To my understanding, the proposals in the positive set have their corresponding ground truths by design, so the model learns to recover the ground-truth moments from them. However, there is a lack of explanations about the role of the proposals in the negative set. Are they also used to recover ground truths? Or do they serve as negative samples for classification? If the former is the case, how is the matching performed? In addition, what happens if they overlap with ground truths? Will it disturb the training?

- The comparisons with existing DETR-approaches seem not fair. To my knowledge, the DETR-based approaches (e.g., Moment-DETR and UMT) leverage four encoder layers and two decoder layers with a total of 10 queries on QVHighlights. On the other hand, the proposed architecture utilizes (at most) four times more encoder/decoder layers than those of the competitors, and three times more moment queries than those of the competitors. This makes it unclear whether the performance gains come from increased parameters or the proposed algorithm, and it is highly encouraged to perform comparisons under the same setting. In addition, comparisons on the computational cost and the memory consumption will be beneficial. Meanwhile, one of the state-of-the-art method, QD-DETR [1], is missing in the comparison table. If included, the proposed method shows inferior performances even with more layers and more queries.

[1] Moon et al., “Query-Dependent Video Representation for Moment Retrieval and Highlight Detection”, CVPR, 2023.

### Questions
Please refer to the Weakness section.

### Soundness
3 good

### Presentation
4 excellent

### Contribution
2 fair
