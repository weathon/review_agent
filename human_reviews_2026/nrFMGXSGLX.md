# PCAInit: Training-Free Initialization for Image-Based Neural Representations

- Avg Score: 5.50
- Decision: Reject
- Scores: 8, 4, 6, 4

## Abstract
Implicit neural representations (INRs) have been widely used to model data as continuous functions parameterized by multi-layer perceptrons (MLPs).
However, the relationship between the weight space of INRs and the underlying data space remains underexplored. 
In this paper, using SIREN as a baseline architecture, we study this connection through the lens of video frame reconstruction, which serves as a controlled setting where principal component analysis (PCA) reveals a striking alignment between image space and weight space. 
Building on this observation, we introduce \textit{PCAInit}, a novel training-free initialization strategy.
We compare PCAInit with pretrained-based approaches that also offer higher reconstruction quality but come at the cost of additional training time: a meta-learned initialization and our two additional proposed methods.
We show that PCAInit achieves the best overall reconstruction quality without extra training time.
For example, on a representative DAVIS 2017 video (bear, 480p), PCAInit improves PSNR by up to +37.1\% over SIREN and +26.7\% over meta-learned initialization.
Furthermore, we show that PCAInit generalizes beyond video frames, achieving the best PSNR on collections of images as well.
Moreover, we demonstrate that PCAInit achieves high PSNR in additional evaluation tasks and exhibits strong universality through cross-video initialization experiments.
Our results reveal a promising research direction on the interplay between image space and weight space in INRs, opening new avenues for future research on efficient INRs with improved reconstruction quality and broader applicability.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes a method for INR initialization. The method uses PCA to align the initial weight trajectory over time to that of video frames over time. This can also be applied to a collection of images in the same way. The method shows improved reconstruction accuracy over the naive initialization of several prior works.

### Strengths
- The method is a simple, actionable, method for improving INR performance from better initialization

### Weaknesses
- Unclear if this affects bitrate
- More comparisons would be nice

### Questions
This paper provides a simple framework for improving INR results by using a better initialization. Overall the method is simple, straightforward to implement, and achieves clear improvements in results. It's the kind of thing that I can imagine becoming standard in the INR world. I do have some minor issues which I think should be clarified though:

1. How does the method perform on non-SIREN networks? I think the only evaluation was on SIREN but it could be applied to other methods as well. I expected to see a table comparing for several methods: "Naive init" vs "Our init". 
2. I think there should be some comparison to NIRVANA because they did some work on the initialization scheme as well

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper focuses on improving implicit neural representations (INRs) for video data, which are often limited by high training costs and suboptimal reconstruction quality. Using SIREN as the baseline, the authors explore three initialization strategies to accelerate INR training and enhance performance on large-scale video frames. Among them, the proposed PCAInit method is a training-free initialization derived from analyzing the relationship between video frame space and network weight space using principal component analysis (PCA)

### Strengths
1. The discovery of a similar PCA trajectory between the image space and the weight space is highly insightful and provides a valuable perspective for understanding the internal structure of INRs.

2. Building on this observation, the proposed PCA-based initialization (PCAInit) is conceptually sound and well-motivated, offering a clear rationale for improving INR performance in video reconstruction.

3. The experimental evaluation is comprehensive and convincing, comparing PCAInit with multiple classical and meta-learned initialization methods across diverse datasets, which strongly supports the validity and effectiveness of the proposed approach.

### Weaknesses
1. The method description lacks clarity. In lines 281–294, the phrase “use the PCA basis of ... weight space” is conceptually vague. The authors should present the method using clearly defined mathematical symbols and equations rather than verbal explanations. As written, it is difficult to understand the precise transformation steps between image and weight spaces.
2. The abstract and introduction should emphasize the discovery of similar PCA trajectories in image and weight spaces, which I consider to be the paper’s most valuable contribution. In contrast, the Previous Frame and First Frame initialization strategies are relatively minor contributions and do not represent the core insight of the work, yet the current abstract and introduction fail to highlight this key finding.
3. Although the paper defines its problem in the context of video representation, it uses SIREN, a model originally designed for static image representation. This design choice is unconvincing. The authors mention NeRV in the related works but incorrectly state that NeRV “focuses on compression at the cost of reconstruction quality.” In reality, NeRV provides faster and more accurate video representation, with compression as an added benefit. For representing videos, NeRV remains a more suitable and efficient choice than modeling each frame with separate MLPs.
4. The training time comparison in Figure 1 is somewhat misleading. The reported speedup relies heavily on parallelization, effectively trading memory for time rather than reducing total computation. In scenarios without sufficient parallel hardware, the training time would remain long. This limitation further weakens the argument for applying the proposed method to video representation, where efficiency is crucial. That's why NeRV is more suitable for video representation.

Overall, while the paper presents a valuable finding regarding the relationship between image and weight spaces, its practical applicability to video representation is limited due to methodological and architectural choices.

### Questions
1. The statement in line 196 is confusing. Given the initialized weights $\theta_t$, it is unclear why the model still needs to initialize $\theta_t$ again. This redundancy suggests unclear notation or an inconsistency in the description of initialization steps.
2. In line 305, for the “image collections” case, can the PCA be computed from a single image only, or does the method require multiple images to form the PCA basis?

### Soundness
3

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
3

### Summary
This paper proposes PCAInit, a training free initialization mechanism for INRs. The main
idea is to leverage PCA-based alignment between image space and weight space. This
approach is very insightful and explores the connection between weight space and image
space. I think this is the first work that explores this kind of connection for weight
initialization (haven’t checked the latest works). When it comes to results, it has shown
strong empirical results on the DAVIS dataset.

### Strengths
1). Paper is written very well and understandable. Further, the Figure1 helps to understand
what is going on with the approach (overall).
The proposed approach has significant improvement over the baseline.
2). The most interesting observation and the strength is this not only works for videos but also
for unrelated image collections. This is a huge plus.
3). When it comes to novelty, as far as I know, this is a novel approach.

### Weaknesses
1). I would like to know why authors specifically selected SIREN as the base architecture? Did authors attempt GAUSS, WIRE or any improved versions of SIREN (for instance FINER).

2). As this paper focuses on improving initialization for video frames, how does PCAInit compare to architectures explicitly designed for videos, such as the NeRV family or related INR models? Can PCAInit also be applied to those networks, or are there
structural limitations?

3). The evaluation is limited to 15 videos from DAVIS 2017, which are all natural videos. It would be valuable to understand how the approach performs on non-natural or AIgenerated videos, or on videos with abrupt scene transitions that break temporal
continuity. 

4). The approach relies on the assumption that image and weight manifolds are related through an orthogonal transform plus a linear projection. This is a strong assumption, but the paper provides only empirical evidence. I would like to know can authors include at least some theoretical or intuitive justification for why this correspondence arises.

5). Many of the new works on INRs have not been cited. specifically, video inr methods. 

6). For image-based evaluation, why PCAinit is not compared with meta-learning-based approach?

### Questions
Please see the weaknesses section.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
In this work the authors introduce a new method to initialize Video-INR systems to encode the videos much faster than the previous baselines. They introduce a novel training free PCA-based initialization method which outperforms MAML based methods and is also void of any sequential dependency. This work paves way towards more practical and faster encoding times for Video INRs.

### Strengths
- The paper is well written and easy to follow with most claims backed by theoretical/empirical analysis. 
- The idea of obtaining weight space initialization by using pseudo weight trajectories without any expensive MAML training is an important contribution to the field.

### Weaknesses
- Initialization strategies like re-using previous frame/first frame have been explored in prior works like [1] and the authors should refrain from claiming it as "main contributions" in their paper.  
- Impact on Compression: Apart from serving as general purpose representations, INRs can also be used for video compression. It would be interesting to see the impact of PCAInit on reducing the bits required for representing a video. 
- More analysis on how the content of the video used for calculating PCAinit influences the convergence speed/final quality would be helpful. That would help us answer few crucial questions like - does this transfer across datasets of videos? How does PCAInit fare when the initialization is derived from a completely different set and so on. 
- The paper restricts itself to 480p videos. Does it have the potential to scale ? or does that require substantial architectural changes? 
if it is the latter, then we might even need to revisit the assumptions made to derive the PCAinit weights. 


[1] https://arxiv.org/abs/2212.14593

### Questions
NA

### Soundness
3

### Presentation
3

### Contribution
3
