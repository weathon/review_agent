# Point Prompting: Counterfactual Tracking with Video Diffusion Models

- Decision: Accept (Poster)
- Scores: 6, 6, 2, 4

## Abstract
Trackers and video generators solve closely related problems: the former analyze motion, while the latter synthesize it. We show that this connection enables pretrained video diffusion models to perform zero-shot point tracking by simply prompting them to visually mark points as they move over time. We place a distinctively colored marker at the query point, then regenerate the rest of the video from an intermediate noise level. This propagates the marker across frames, tracing the point's trajectory. To ensure that the marker remains visible in this counterfactual generation, despite such markers being unlikely in natural videos, we use the unedited initial frame as a negative prompt. Through experiments with multiple image-conditioned video diffusion models, we find that these "emergent" tracks outperform those of prior zero-shot methods and persist through occlusions, often obtaining performance that is competitive with specialized self-supervised models. Finally, we show that trajectories produced by pretrained generators can be distilled into a fast tracker with similar performance, serving as effective supervision for a tracking model.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Point Prompting, a novel method for zero-shot point tracking using pre-trained I2V diffusion models. It achieves strong performance without any training by adding a visual marker to the first frame and leverage the internal priors of the model to propagate the marker through the generated video.

### Strengths
1. Proposed method is novel and simple. It utilizes the existing priors of video diffusion models without requiring costly fine-tuning or specialized architectures.

2. The paper provides thorough ablation studies that validate the contribution of its different components.

3. This paper demonstrates the model-agnostic performance covering diffusion model (CogVideo X) and flow-based model (Wan 2.1 and 2.2)

### Weaknesses
1. A major practical limitation is the computational expense. As noted by the authors in L338-339, tracking a single point requires take 7 to 30 minutes. This makes the approach impractical and unfeasible for large-scale offline analysis.

2. It is unclear if the method can handle tracking multiple points simultaneously.

3. More analytical experiments are needed. For instance, the paper would benefit from an attention-based analysis to explore how the model focuses on the point, or an investigation into how tracking paths vary with different random seeds.

4. I wonder how the performance would be affected by using DDIM inversion for the video generation process, rather than the simpler noising approach used.

### Questions
1. The explanation of the evaluation metrics (Positional Accuracy, Occlusion Accuracy, and Average Jaccard) could be more detailed.

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
The paper explores repurposing large-scale pre-trained video generative models for object tracking. It leverages the rich spatio-temporal representations learned by these models and adapts them in a training-free manner to track targets across frames. The paper is well-written, logically structured, and presents clear quantitative results that complement the experimental tables.

### Strengths
* Clearly motivated and interesting idea of repurposing large-scale pre-trained video generative models for object tracking.
* Intuitive and easy-to-understand method.
* Comprehensive experiments with both quantitative metrics and qualitative results demonstrating the approach’s effectiveness.

### Weaknesses
* Runtime comparison of each component in the pipeline (i.e., the time cost of steps listed in Table 4)?
* What are the effects of different denoising steps? Was a simple Euler solver used, and what is the time schedule?
* Is the tracking performance evaluated on real-world data or synthetic data (e.g., self-generated clips)?
* Can you confirm whether the models are run in image-to-video mode?
* Can you compare with the latest works on flow-matching model editing? [1-2]


Ref:\
[1] Jiao, G., Huang, B., Wang, K.C. and Liao, R., 2025. Uniedit-flow: Unleashing inversion and editing in the era of flow models. arXiv preprint arXiv:2504.13109.\
[2] Kulikov, V., Kleiner, M., Huberman-Spiegelglas, I. and Michaeli, T., 2025. Flowedit: Inversion-free text-based editing using pre-trained flow models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 19721-19730).

### Questions
Please see the [Weakness] section.

### Soundness
4

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper investigates whether pretrained image-conditioned video diffusion models exhibit emergent point-tracking capabilities. The authors propose a “point prompting” technique in which a red dot is placed on the first frame of a real video, the video is regenerated using SDEdit, and the propagated dot is tracked via color-based detection. Several heuristics (color rebalancing, negative prompting, and inpainting refinement) are introduced to stabilize the dot’s visibility across frames. Experiments on TAP-Vid show improvements over image-based zero-shot tracking baselines, and the authors claim that these results indicate temporal reasoning and object permanence in video diffusion models.

### Strengths
- Limitations are clearly articulated, with helpful visual examples that make failure modes easy to interpret.
- Unlike some zero-shot correspondence approaches, the work attempts to handle occlusion.

### Weaknesses
### 1. Limited novelty and lack of conceptual advancement

The core claim that video diffusion models contain emergent temporal correspondences, has already been demonstrated by prior work such as DiffTrack [1]. The statements in lines 35–36 and 92–93 suggest novelty in analyzing emergent tracking in video diffusion models, but the conceptual contributions closely follow existing findings and provide little new insight into the temporal behavior of DiT-based models.


### 2. The method is not suitable for either analysis or tracking

(1) Misalignment with analysis goals

The methodology relies heavily on pixel-space operations rather than model-level signals. The pipeline includes removing all red pixels from the input, adjusting global color balance, reducing marker saturation after generation, performing inpainting refinement when the dot drifts. These operations substantially alter the video content and disconnect the analysis from the model’s inherent behavior. Because the method depends on multiple rounds of video regeneration, it does not capture the model’s natural temporal consistency. Table 4 further shows that performance sharply degrades without the heuristic refinements, indicating that consistency comes from the heuristics rather than from the generative model itself.

(2) Inefficiency as a point tracking method

The method tracks the dot purely based on pixel color, ignoring positional encoding and geometry, and failing on rapid motion. Because each point requires re-generating the entire video often more than once, the pipeline is extremely inefficient. 

Overall, the method behaves more like a handcrafted video-editing pipeline than a principled analysis of temporal correspondences.

### 3. Writing and presentation issues
- The definition of the tracking problem is unclear (pixel-level vs. semantic vs. object-level).
- Lines 36–38 contain vague phrasing (“high-level understanding tasks”) and ambiguous pronoun ("these capabilities") use.
- The Related Work section blends supervised, self-supervised, and counterfactual modeling approaches without clear structure.

### 4. Missing comparisons and limited generalization
- The most relevant baseline, DiffTrack [1], is not included in quantitative comparisons, which weakens the empirical evaluation.
- The method applies only to image-conditioned video diffusion models and cannot be used for text-to-video models, contradicting claims of architectural generality.
- Experiments do not analyze how point radius interacts with input resolution, potentially biasing comparisons across models.


[1] Nam, Jisu, et al. "Emergent Temporal Correspondences from Video Diffusion Transformers." (NeurIPS 2025)

### Questions
- Lines 107–108 claim that DINOv2 has been adapted for temporal correspondence. What specific prior work supports this claim? A citation is required.
- The paper reports (line 338-341) runtime for a single 50-frame generation, but the full pipeline requires at least two rounds (generation + refinement). What is the actual cost per tracked point?
- In Table 3, what exactly is the configuration represented by the second row (“DAVIS 256×256 up.”)?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a training-free method that adopts video diffusion models for point tracking.  It first puts a colored marker on the query point in the first frame, then asks the video diffusion models to propagate the marker across frames by generating new videos. To avoid the loss of the marker in video generation, this paper proposes to use the unedited video’s initial frame as a negative prompt. The proposed method is evaluated on the TAP-Vid benchmark.

### Strengths
1. The idea of using a colored marker to indicate the query point is interesting and insightful. 
2. It is also interesting to use an unedited video’s initial frame as a negative prompt to make the marker visible.
3. The paper is well-written and easy to follow.

### Weaknesses
1. This paper requires video generation to get the tracking results. What is the computational cost of this method for tracking one point, compared to methods that do not use diffusion models?

2. This approach requires generating a video for each tracked point, which is difficult to use in real applications. 

3. Previous work[1] already shows that video diffusion models have an inherent ability for point tracking. Simlilar observation is also proposed in [2].  

[1] Emergent Temporal Correspondences from Video Diffusion Transformers
[2] Track4Gen: Teaching Video Diffusion Models to Track Points Improves Video Generation.

4. The tracking performance is much lower than non-diffusion methods such as CoTracker3. Although it performs better than DIFT and SD-DINO, these two methods are not designed for point tracking in videos.  Considering the high computation cost, the tracking accuracy is not good enough.

5.  According to line 305, the proposed method allows the video diffusion models to 
>  generate only regions near the potential tracked point. 

What if there are occlusions or significant object motions in the videos？

6. The accuracy of the tracking method is limited by the generation ability of video diffusion models, which limits the effectiveness of the proposed method in diverse scenarios. This also raises my concern about the robustness of the proposed method in cases such as 
small objects,  corrupted videos, or videos with poor weather, and so on.

7. How does the method track points in long videos that exceed the maximum length supported by video diffusion models? 

8. This paper lacks the text “Under review as a conference paper at ICLR 2026” in the header, which is unexpected according to the official template.

### Questions
Please refer to the weakness part.

### Soundness
3

### Presentation
3

### Contribution
2
