# Animate Your Thoughts: Reconstruction of Dynamic Natural Vision from Human Brain Activity

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 8

## Abstract
Reconstructing human dynamic vision from brain activity is a challenging task with great scientific significance.  Although prior video reconstruction methods have made substantial progress, they still suffer from several limitations, including: (1) difficulty in simultaneously reconciling semantic (e.g. categorical descriptions), structure (e.g. size and color), and consistent motion information (e.g. order of frames); (2) low temporal resolution of fMRI, which poses a challenge in decoding multiple frames of video dynamics from a single fMRI frame; (3) reliance on video generation models, which introduces ambiguity regarding whether the dynamics observed in the reconstructed videos are genuinely derived from fMRI data or are hallucinations from generative model. To overcome these limitations,  we propose a two-stage model named Mind-Animator. During the fMRI-to-feature stage, we decouple semantic, structure, and motion features from fMRI. Specifically, we employ fMRI-vision-language tri-modal contrastive learning to decode semantic feature from fMRI and design a sparse causal attention mechanism for decoding multi-frame video motion features through a next-frame-prediction task. In the feature-to-video stage, these features are integrated into videos using an inflated Stable Diffusion, effectively eliminating external video data interference.  Extensive experiments on multiple video-fMRI datasets demonstrate that our model achieves state-of-the-art performance. Comprehensive visualization analyses further elucidate the interpretability of our model from a neurobiological perspective.  Project page: https://mind-animator-design.github.io/.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work addresses the problem of reconstructing high-quality video from fMRI data. The authors suggest that the key to achieving high-quality video reconstruction lies in decoupling and modeling semantic, structural, and motion information, also carefully handling the frequency discrepancy between fMRI data and videos. To this end, the authors develop a tri-modal contrastive learning scheme along with a next-frame prediction task. Lastly, to ensure that the generated videos are derived purely from the fMRI data, the input is fed into an untuned inflated Stable Diffusion model. Empirical evaluations show promising results and strong interpretability performance.

### Strengths
[Quality, Significance]:

The structure of this work is very easy to understand and follow, with sufficient context and supported citations. The model notations and figure presentations are excellent. 

The authors provides comprehensive empirical metric evaluations and compared their results with many other state-of-the-art approaches, and provide comprehensive ablation study and interoperability results.

[Novelty]: One particular point that I find this work novel is that the authors do not fine-tune the stable diffusion, ensuring us that the videos that we see are not coming from the overfitting.

### Weaknesses
I do not find this work particularly having major weaknesses.

### Questions
1. How did the author chose $\lambda_1$ and $\lambda_2$ in equation 4? Can the author provide the ablation study/ discussion on how these choices affect the downstream performance?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper addresses the challenge of video reconstruction from fMRI data, identifying three key components for an effective decoder: semantics, visual structure, and motion patterns. Building on prior works that successfully capture semantic content using CLIP and Stable Diffusion, the authors aim to improve fidelity to the visual structure and motion dynamics present in the original videos. They propose a decoupled, multi-step reconstruction approach, utilizing distinct decoders and specialized reconstruction criteria. Their method combines tri-modal CLIP (fMRI, image, text), single-frame Stable Diffusion (T2I), and a uniquely learned internal motion prior that avoids biases from external video datasets. Applied to three publicly available movie-fMRI datasets, the approach demonstrates moderate gains in some metrics across different configurations, with notable improvements in preserving visual structure.

### Strengths
The authors’ effort to achieve simultaneous reconstruction of visual structure, semantics, and motion patterns represents a significant and timely contribution. Many prior works prioritize visually appealing reconstructions or focus on semantic fidelity, often overlooking finer alignment with the ground truth video structure. Advancing fidelity in this regard is important; without it, decoders risk becoming general natural video generators with limited adherence to fMRI cues.

The primary improvement over previous methods appears in the preservation of visual structure, evidenced both qualitatively and quantitatively, though this improvement is moderate. The authors’ thorough analysis across three datasets, coupled with an extensive benchmarking on various metrics assessing semantics and visual structure, adds robustness to the work. Additionally, the website offers a helpful visual supplement to this complex work, making it more accessible and digestible.

### Weaknesses
The results, both qualitatively and quantitatively, lack compelling evidence of substantial improvement over prior work, particularly when compared to Chen et al. (2024). In fact, it appears that with proper statistical analysis (e.g., Table 2-4), there may be no significant gains. Methodologically, the advancement over Chen et al. (2024) seems minimal, with certain versions of the authors' approach even reintroducing external videos—the very bias this paper aims to avoid. Furthermore, the claim of improved motion pattern reconstruction is insufficiently supported, as the authors compare only against shuffled frames. A meaningful comparison should have been made with prior works, similar to the comparisons on semantic and visual structure fidelity.

Additional comments:
- [36-38] Statement is unclear; consider rephrasing.
- [54-59] The text appears to confuse BOLD integration duration with fMRI sampling rate. The BOLD signal integrates neural activity over a period greater than 10 seconds (~300 video frames), which is a major limitation in recovering any motion patterns potentially encoded in neural activity. The fMRI sampling rate is limited for other technical reasons but even if was sampling at a higher rate this would likely not resolve this fundamental issue.
- [100-101] The claim about enabling video reconstruction may be overstated, as prior methods have also achieved this to an extent.
- [102-104] Similar to above, there is a mix-up regarding temporal aspects. Clarification of this contribution would help.
- [104-106] The visualizations mentioned are not novel to this work; they have been used in prior studies and are reintroduced here rather than proposed anew.
- [107-110]  It remains unclear what metric is used to evaluate successful recovery of motion patterns. Detailing this metric is necessary for interpreting the results.

### Questions
1) Consider adding robust statistical analyses and uncertainty estimates for the primary results. This would clarify the significance of the observed gains, if any.

2) For the claim regarding improved recovery of motion patterns, stronger evidence is needed. As presented, the motion generator appears modest/limited in comparison to the tools used for recovering structure and semantics. Since improvement in motion fidelity is a key goal, it would be impactful to demonstrate and emphasize substantial gains in this area. If stronger evidence cannot be shown, it may be best to reconsider this claim.

3) The authors note that MT, an area associated with motion processing, shows significant activation for semantics, typically associated with the ventral stream. If this outcome supports the motion recovery, it could benefit from additional context. Some clarification on MT’s role in semantics would be helpful for the reader’s interpretation of these findings.

4) For Figure 8, consider enhancing its visual accessibility to make it easier to interpret. Improvements in layout or clarity could aid the reader in understanding the figure’s contribution.

### Soundness
3

### Presentation
3

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
Paper presents an approach the reconstruct video clips for fMRI brain recordings.
The reconstruction is broken to 3 streams: structure, semantic and motion.
First the fMRI signal is transformed to align with image embeddings, the fMRI embedding is used with pretrained image diffusion models to generate reconstructions.
The results are compared against multiple previous works on a variety of metrics.

### Strengths
Competitive results on multi evaluation metrics, the provided reconstruction look good visually.

### Weaknesses
- Full reconstruction of the video clips not provided, this would make the work more transparent, and would allow future works to easily compare on new metrics.
- No comparison to other methods is provided for retrieval metric (Which I think is one of the most objective/relevant metrics)
- Authors don't show results for sequences with actual motion, give that one of the focuses of the work is motion, it would make sense to show the ability to reconstruct motion. (for example the clip with the soldier)

### Questions
- Figure 7 color scheme is not consistent across plots(relevant to all similar plots) 
- I think it makes sense to put retrieval results in a more centric place in the paper. 
- The work " Kupershmidt22" provides a retrieval metric for itself and "Wen18", you should consider adding this results.
- It would be helpful to add standard error/ statistical tests to the comparisons between the models.

### Soundness
3

### Presentation
3

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
In this work, the authors present a novel method for video reconstruction from fMRI recordings. There are two core components to this novelty, (1) the method is interpretable and learns separate semantic, structural, and motion features from the fMRI - later used to generate video frames, and (2) the motion component of the generated video is solely based on the motion predicted from the fMRI because the video generator is an inflated image diffusion model. The videos are generated in two stages, fMRI-to-feature which has trainable components, and feature-to-video which is completely frozen. At training time of stage 1, a Semantic decoder using frozen CLIP encoders, a Structure decoder using a frozen VQ-VAE, and a custom transformer-based Consistency Motion Generator with a masked causal frame prediction task are trained to learn the semantic, structural, and motion features respectively. The performance of the reconstruction model is evaluated on 3 public fMRI datasets with 8 different (semantic, pixel, and spatiotemporal) evaluation metrics, compared against previous work, analyzed with ablation studies, and assessed under two different interpretability analyses. The model is found to exceed state-of-the-art both in quantitative and qualitative metrics, yield sensible component contributions in the ablation, and offer interpretability insights that include neurobiological plausibility.

### Strengths
This paper has good quality, with sound methodology and extensive experiments. It is an original, systematic, and rational approach that addresses important issues in previous work; namely the entanglement of different types of features that are decoded from fMRI (semantic, structural, motion), and the entanglement of motion information learned by external training of the video generator with the motion information derived from fMRI. The improvements brought forward by this approach are clearly indicated in the quantitative and qualitative results and consist significant contributions in the research aiming to recover, as well as understand, dynamic visual information from brain recordings.

### Weaknesses
**Presentation** could be more clear at many points throughout the paper. There are technical parts that require more explanation or are currently creating some confusion, which are outlined as follows in order from more major to more minor issues.
* In section 3.2, it is not mentioned at all how the ground truth text condition $\bf{c}_i$ is obtained - are these from the training dataset of the (frozen) inflated image generator or are these the same as the video captions $\bf{t}$? It would help to have an intuitive explanation of this in figure 3 as well ($\bf{c}$ is not shown at all in the training pipeline).
* In section 5.1, it is not clear why the videos are external in the “External Videos (EV)” case; This is described as “we further fine-tuned the image diffusion model using videos from the CC2017 dataset (with the training set)” which is still videos from within the video-fMRI dataset, already seen during training in earlier stages of the pipeline.
* In section 3.2, authors are not explicit enough in the description of the trainable modules Semantic & Structure decoder; From the notation it can be assumed that the Semantic decoder is just one trainable vector $\bf{f}$ initialized by the fMRI vector, and the Structure decoder is an MLP $D_{Structure}$ (of unknown number of layers) but it needs to be outlined explicitly.
* In the same section, it is also not explicit how the frames are aggregated to create $\bf{v}$ - is it the average of the CLIP visual embedding across all frames or another aggregation?
* In section 4.2, the description of End-Point-Error(EPE) is missing - the reader has no clue what it is, if it is something existing in the literature (missing citation) or introduced by the authors (missing formula). There are also missing citations for some of the other metrics (Hue-pcc, CLIP-pcc).
* In section 3.3, a mention to the VQ-VAE decoder depicted in the figure is missing - the reader is left to wonder about this. Additionally, this section would benefit from more information on the inflation process (e.g. at least an in-line equation).
* In section 5.2, there is no mention of which dataset the ablation study is on.
* In section 3.2, figure 4 is missing $\bf{E_{pos}}$ and would benefit from a more informative caption. Also, the abbreviation LDM used in the text is probably not familiar to all readers (either remove or expand it).
* In the rightmost box of the training pipeline in figure 3 (CMG), the bracket of $\bf{L_{consistency}}$ is placed in a misleading way - a suggestion would be to move the mask matrix to the bottom and have the input and output frames on the right, with $\bf{L_{consistency}}$ connecting the original (masked) future frames with the predicted future frames.
* In figure 2c the arrow for the noise would be more accurate if it pointed to the Motion instead of the Structure box.
* Prior work is sometimes too generically described e.g. “Subsequently, Han et al. (2019), Wen et al. (2018) and Wang et al. (2022) map brain responses to the feature spaces of deep neural network (DNN) to reconstruct video stimuli.” - what DNN?
* In section 6.2, where authors describe “visual cortices”, probably a more adept term is to say “visual cortical areas” or “areas of the visual cortex”.

**Soundness** related issues exist in parts of the methodology, or sometimes its interpretation.
* The reader is not yet completely convinced that motion information is from the fMRI and not from the training videos. It would help to see an ablation with the CMG trained without the fMRI entirely (output from temporal module is then the Q, K, and V of spatial module), and see if the next frames can be predicted at inference from the structural latent fMRI embedding of the first frame. If the CMG module is still good, then the motion information is not from the fMRI but from the videos in the training set.
* The reader finds several issues with the analysis in  6.1. First the y-axes should end at 1.0 as the maximum value of $\sum _i \delta _i$ is 100. Additionally the y-axes are different across the 3 panels which is also misleading. Second, it seems that although p-values are lower with the CMG, they are still very high and much higher than 0.05, meaning that the order of the generated frames does not matter significantly for these metrics. This is not commented on at all by the authors. Here, the reader’s suggestion for a better baseline to compare against (instead of the standard threshold of 0.05) is the p-value of the shuffle test with the ground-truth video, as it is not certain that even this would be below 0.05. Third, it is not clear why the authors are examining the structural metrics for the shuffle test instead of solely the spatio-temporal metrics, and why for the latter only CLIP-pcc is shown and not EPE. In the view of the reader, only CLIP-pcc and EPE are relevant and should be shown. Finally, the results are vastly different across the 3 subjects which is also not (and should be) commented on in the text. 
* In section 5.2, table 5, it is observed that the structure metric Hue-pcc is increasing significantly when the structure module is removed. This seems like an important inconsistency in the results, yet the authors do not comment on it. It is expected to provide some sort of explanation, perhaps based on how this specific metric works and on how the w/o Structure videos look, since the other pixel-level metrics seem to decrease.
* In the analysis in 6.2, it is noticeable that the weight proportion of V1 in the motion information is almost double that of the next highest-weighted areas (e.g. TPOJ, MT, V2, V3), which seems very significant and is not (and should be) commented on - this effect is hidden when the whole of LVC weights are added up.

### Questions
The reviewer would like the authors to address the points outlined in the “Weaknesses” section, in each case either by making the suggested change or another change that to the authors’ opinion fixes the issue better, or lastly by giving a sufficient (and convincing) explanation of why no change is needed. This way the reviewer’s opinion of the paper would be improved, to a rating of 6 or above.

**Update after rebuttal:** The score has been updated from an initial 5 to an 8, since all of this reviewer's points were addressed in a very careful, timely, and complete manner. More specifically, the crucial points outlined above in "Soundness" were fixed and the additional ablation experiments strengthen the paper and alleviate this reviewers concerns, which on its own brings the paper rating above acceptance threshold. On top of this, the paper's presentation was substantially improved, making it a consistent and clear read, and overall a good paper (8).

### Soundness
3

### Presentation
3

### Contribution
4
