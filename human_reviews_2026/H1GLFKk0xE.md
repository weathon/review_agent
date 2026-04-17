# A Cognitive Process-Inspired Architecture for Subject-Agnostic Brain Visual Decoding

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
Subject-agnostic brain decoding, which aims to reconstruct continuous visual experiences from fMRI without subject-specific training, holds great potential for clinical applications. However, this direction remains underexplored due to challenges in cross-subject generalization and the complex nature of brain signals.
In this work, we propose Visual Cortex Flow Architecture (VCFlow), a novel hierarchical decoding framework  that explicitly models the ventral-dorsal architecture of the human visual system to learn multi-dimensional representations. By disentangling and leveraging features from early visual cortex, ventral, and dorsal streams, VCFlow captures diverse and complementary cognitive information essential for visual reconstruction.
Furthermore, we introduce a feature-level contrastive learning strategy to enhance the extraction of subject-invariant semantic representations, thereby enhancing subject-agnostic applicability to previously unseen subjects. 
Unlike conventional pipelines that need more than 12 hours of per-subject data and heavy computation, VCFlow sacrifices only 7\% accuracy on average yet generates each reconstructed video in 10 seconds without any retraining, offering a fast and clinically scalable solution.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes VCFLOW, a cognitive process–inspired architecture for cross-subject fmri-to-video decoding. By disentangling early visual, ventral, and dorsal pathways and introducing a SARA module, the method can effectively generalizes to unseen subjects without any finetuning.

### Strengths
1. This paper conducts comprehensive ablation studies, and the overall framework design is well visualized.
2. Compared with NEURONS, the proposed SARA module and fine-grained hierarchical alignment mechanism achieve good performance improvements in the cross-subject setting.

### Weaknesses
1. The dataset used only contains three subjects, which weakens the persuasiveness of the cross-subject evaluation. Future work could validate on datasets with larger subject pools, such as CineBrain[1].
2. The study only reports results in the cross-subject setting. Given that the overall performance on the fMRI-to-video task is still relatively limited, including single-subject results would better show whether the proposed architecture truly enhances decoding quality rather than merely improving transfer robustness.
3. The claimed 10s inference time appears only in the introduction and lack specific  configuration. The authors should clearly state the experimental conditions and provide comparative results with other methods.
4. Section 3.3 introduces multiple auxiliary tasks, but the paper lacks detailed explanations of how these are constructed and how multi-feature fusion is handled during inference.
5. In terms of neuroscientific interpretation, VCFLOW relies heavily on strong anatomical priors (early/ventral/dorsal separation). Thus, the model validates existing neuroscientific hypotheses rather than discovering such structure in a data-driven way, which limits novelty compared to NEURONS.

[1] CineBrain: A Large-Scale Multi-Modal Brain Dataset During Naturalistic Audiovisual Narrative Processing

### Questions
1. How exactly are the video processed and aligned with fMRI signals? More details should be clarified.
2. In Figure 3, what exactly is the “Brain Model” shown in the framework? Additionally, what are the input dimension of the early, ventral, and dorsal features, and what is the dimension of the shared projection space? It would be helpful if these details could be included in the appendix for clarity and reproducibility.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes the Visual Cortex Flow Architecture (VCFLOW), a hierarchical brain decoding framework capable of learning multi-dimensional representations from different brain regions (early visual cortex, ventral, and dorsal streams). It also introduces a feature-level contrastive learning strategy to help the model generalize to unseen subjects. The article achieves this by sacrificing a certain degree of accuracy, enabling the model to perform well on unseen test subjects.

### Strengths
1. The motivation of the article is very good. I completely agree that the brain decoding model should further focus on decoding for multiple individuals, especially for unseen subjects. Because we cannot expect users to undergo prolonged collection processes in practical use. Generalization among the subjects is a problem that must be addressed.

2. The article is well-written and clearly expressed. The experiment was also conducted thoroughly.

3. The model not only designs multiple different modules to extract features at different levels for alignment, but also adopts multiple tasks to conduct a more comprehensive alignment.

### Weaknesses
1. I agree that generalization to unseen subjects is very important. However, I do not fully agree that the reliability of the method can be demonstrated in the video reconstruction task. As is well known, current brain video reconstruction largely depends on the pre-trained Diffusion model to achieve its effects. Then, we calculate the indicators based on the generated videos. I believe the results are heavily influenced by the generation model rather than actual brain decoding. (In other words, brain decoding might only produce very high-dimensional and not detailed semantic information, which the diffusion model then uses to generate the video. In fact, the results shown in Figure 5 also reflect this point. Current brain decoding is likely to have provided only very vague information. We evaluate everything based on the generated video, and there are too many factors in between that can influence the outcome.)

2. Based on the article's indicators, the improvement is minimal. In the 50-way scenario, it only increased by 2% to 4% compared to other pre-training methods, resulting in just 2 to 4 more images correctly identified out of 100 samples. Additionally, the pixel-level metrics are very low, so they offer limited comparative value. Even a completely black image can reach a PSNR of around 10. 

3. The study had too few subjects. With only three participants involved, it is very difficult to rule out coincidental factors and establish a clear pattern. The works Clip-Mused [1] and TGBD [2] focus on retrieving image/video frames based on brain signals for multiple subjects. Particularly, TGBD is also designed for unseen subjects, utilizing the HCP[3] dataset (with over 170 participants). The main conclusion is that as the number of participants increased, the generalization ability on unseen participants significantly improved. I think TGBD could provide inspiration for this work and should be included by related works, as one of the few studies that focused on the unseen subjects. Maybe the author's method is limited by the number of participants. When the number of participants is sufficient, a much greater performance gap might be achieved.


[1] CLIP-MUSED: CLIP-guided multi-subject visual neural information semantic decoding, ICLR2024

[2] Toward Generalizing Visual Brain Decoding to Unseen Subjects, ICLR2025

[3] The WU-Minn Human Connectome Project: An Overview. Neuroimage, 2013.

### Questions
Please see Weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Visual Cortex Flow Architecture (VCFLOW), which is a subject-agnostic framework for reconstructing videos from fMRI.  One key idea of the pipeline is to extract hierarchical features from brain regions which are the early visual areas, ventral and dorsal stream areas and  align them with different types of visual representations to capture complementary information such as low-level, semantic and motion cues. Besides, the authors design a subject-agnostic model by proposing a module called SARA that projects subject-specific information into a shared space.

### Strengths
- The paper is well-written and mostly easy to follow (i.e up to the complexity of the presented method)
- The motivation for subject-agnostic model is clearly explained and detailed.
- Most fMRI-to-video decoding methods are subject-specific which is a significant bottleneck for clinical applications while this method addresses this fundamental challenge, making it a first important step towards better practicality.

### Weaknesses
- The method seems quite complex with many components while the ablation seems quite incomplete and doesn’t ensure all the elements are crucial to the final performance.
- Missing information on how they generate the final videos in less than 10 seconds (claim in abstract). Could they be more specific into which generative model they use and exactly which brain-predicted embeddings they give to it (as only stable diffusion is mentioned in figure 3 but without further details, maybe explained in fig 4 but didn’t really understand... )?
- Difficult to draw parallels between the letters/concepts introduced in method section 3 and drawing in figure 3. Could you represent on fig 3 at which stage E_early, E_ventral, E_dorsal and F_brain, F_ventral, T_sem, T_subj are present in the figure for instance ?
- The authors claim ‘A key innovation of our approach lies in the utilization of CLIP embeddings from multiple layers to achieve fine-grained semantic alignment with fMRI signals’ but it has already been done in previous brain-to-image papers.

### Questions
- Could the authors explain more clearly evaluation metrics ? N-way top k is cited for frame-level metrics but it seems to also be used for semantic accuracy in video metrics 
- Could you add an ablation on the different SARA/HED losses displayed in line 267 and line 298, on the effect of adjusting the loss coefficients in line 299
- Could you give more details on what is the image pretraining phase ? To which representations do they align brain features ? ( as i guess there is no motion representations for instance)
- Would be nice to compare to other baselines such as the recent work ANIMATE YOUR THOUGHTS: RECONSTRUCTION OF  DYNAMIC NATURAL VISION FROM HUMAN BRAIN ACTIVITY (ICLR25)

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
I have carefully read this manuscript. The research area of this paper is fMRI-to-video reconstruction, with a focus on the challenge of cross-subject transfer (generalization to new/unseen subjects). This is an important problem, as the authors describe in the introduction: in practical applications, it is impossible to collect large amounts of data from a single subject for training decoding models. Although the authors emphasize the cross-subject issue, only about one-third of the proposed methods (Section 3.2) are actually related to this problem, with the majority focusing on designing fMRI model architectures that better align with human visual mechanisms and on improving representation learning methods.

Furthermore, I believe the paper lacks certain methodological and evaluation details regarding cross-subject transfer; the authors do not explain how the model adapts when new subjects are introduced. Putting these issues aside, I think the contribution of the methods proposed in this paper to cross-subject visual decoding is still limited.

### Strengths
+ This paper provides a good overview of the work on fMRI-to-video reconstruction.

+ This paper focuses on an important issue.

### Weaknesses
1. The abstract and the first two paragraphs of the introduction strongly emphasize the issue of transfer/generalization to unseen subjects. However, in the third paragraph of the introduction (lines 73–94), the methods described by the authors have no relation to this issue. This creates a jarring shift in the writing. Similarly, after reading the Introduction, I think the authors should focus on how to achieve transfer to unseen subjects. Yet, in the methods section, only one design is actually related to this issue. Therefore, I believe the introduction of the manuscript requires substantial revision and should not solely emphasize the transfer to unseen subjects.

2. I am fairly familiar with the related work. Essentially, the token extension method proposed by the authors in Section 3.2 assigns subject-specific tokens to each participant. This approach has actually appeared previously in fMRI-to-Image studies [1]. However, it should be acknowledged that in the fMRI-to-Video task, the authors may indeed be the first to apply this method.

3. I believe that some of the methodological and experimental descriptions in this manuscript are not sufficiently clear, and I could not understand how the model adapts when a new subject is introduced; for details, please refer to the Questions section.

4. The citation format in the main text of this manuscript needs to be corrected.

5. According to Figure 5, the video reconstruction results are not very satisfactory.

[1] Zhou et al. CLIP-MUSED: CLIP-guided multi-subject visual neural information semantic decoding. ICLR 2024.

### Questions
1. In Equation 2, does 𝐹 refer to the fMRI voxels or the result obtained through the diffusion prior? According to the equation, it seems to be the former, but according to Figure 3, it appears to be the latter.

2. Equation 5 is puzzling—why is the InfoNCE loss computed using fMRI representations from two different subjects within the batch? How is the InfoNCE loss computed with only two samples?

3. What do 𝑧 and 𝑦 represent in Equation 6?

4. The authors state in line 101 that no training using new-subject data is needed. Does this mean the new subject’s test data is fed directly into the model? If so, where does the subject-specific extended token T_{subj} come from?

### Soundness
2

### Presentation
1

### Contribution
2
