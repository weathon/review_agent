# HoloGarment: 360$\degree$ Novel View Synthesis of In-the-Wild Garments

- Avg Score: 3.50
- Decision: Reject
- Scores: 6, 2, 4, 2

## Abstract
Novel view synthesis (NVS) of in-the-wild garments is a challenging task due significant occlusions, complex human poses, and cloth deformations. Prior methods rely on synthetic 3D training data consisting of mostly unoccluded and static objects, leading to poor generalization on real-world clothing. In this paper, we propose HoloGarment ($\textbf{Holo}$gram-$\textbf{Garment}$), a method that takes 1-3 images or a continuous video of a person wearing a garment and generates 360$\degree$ novel views of the garment in a canonical pose. Our key insight is to bridge the domain gap between real and synthetic data with a novel implicit training paradigm leveraging a combination of large-scale real video data and small-scale synthetic 3D data to optimize a shared garment embedding space. During inference, the shared embedding space further enables dynamic video-to-360$\degree$ NVS through the construction of a garment “atlas” representation by finetuning a garment embedding on a specific real-world video. The atlas captures garment-specific geometry and texture across all viewpoints, independent of body pose or motion. Extensive experiments show that HoloGarment achieves state-of-the-art performance on NVS of in-the-wild garments from images and videos. Notably, our method robustly handles challenging real-world artifacts -- such as wrinkling, pose variation, and occlusion -- while maintaining photorealism, view consistency, fine texture details, and accurate geometry.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a method to generate novel view image of a given garment.
The novelty lies in the use of a video diffusion model and real world video data to learn a shared garment embedding space.
The quality of the results demonstrated in the paper are better compared to previous works.

### Strengths
- The method cleverly leverage real-world garment videos and video generation task to enable multi-view garment generation without paired data training.
- The quality of the generated results are better than previous methods.
- The exposition of the paper is clear and the idea is not difficult to follow.

### Weaknesses
- While the method works nicely, the additional training effort might be large. And the computational requirements and performances are not explicitly discussed in the main paper.
- Regarding evaluation, I think it is better to have a small testing set with ground truth for easier comparison. For example, one can record a video of a person wearing a given garment and perform some actions. In this case, we can compare the reconstructed garment more explicitly.
- The comparison with methods that have explicit geometry, e.g., Garment3DGen, is a bit weird since the proposed method does not have a mesh geometry. That means, it is difficult to edit the result garments unlike those method with explicit geometry. I would suggest at least to mention the advantage of those methods.

### Questions
- The internal color and structure seems not inferred correctly, what might be the main reason for this? (see results in Fig. 6)
- What are the possibilities of enhancing symmetry, e.g., for results in Fig. 6?
- While the goal of the generated garment seems to the garment without wrinkle, but given some human body poses, the generated garment without wrinkles seem unnatural. I am curious about whether the goal is reasonable?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Taking 1-3 garment images or a video as input, this work aims to output its 360-degree novel view synthesis. To train such a model usually requires a large amount of data. However, 360-degree garment images are very hard to collect. And, to use synthetic data will also cause significant domain gap. This work considers to use large-scale real video data (with no 360-degree but owns diverse deformation and short-period view changing) and small-scale synthetic 3D data. Based on this, an implicit training mechanism is proposed. The key idea is to learn a shared garment embedding space between both domains.

### Strengths
- the motivation is clear and the target problem of briging real and synthetic gap is significant.

### Weaknesses
- In all, I think the experiments and analysis are not sufficient. The key contribution is the training paradigm, but it lacks in-depth analysis about why it works. There are two tasks, one is novel pose synthesis for real data and another one is novel view synthesis for 3D data, how do these two tasks benefit each other? 

- In Fig 5, it seems Ours_3d produces very bad quality. Is it because the limited data scale (only ~8000)? or the real-sim domain gap? It is unclear. I think it is not hard to have more synthetic 3D garment assets, for example, we can use image-to-3D models to do generation from images. 

- The visual comparisons are limited, only two examples are shown in Fig 5 and the result of the proposed method for the first example is not aligned with the input. This makes me doubt on the generalization ability of the proposed approach. 

- Another primary concern is from the limited contribution. It seems the key contribution is just a training strategy. At first, I really do not know why it is called "implicit". And, it is also hard to see if the proposed training strategy can be used for other tasks.

### Questions
No.

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
4

### Summary
The paper proposes a method to create a 360 degree novel view of garment in a canonical pose from few images or videos. Existing method fails in the occlusions or produce noisy rendering while the proposed method is robust. The method uses combination of real 2D data and synthetic 3D data to learn shared garment embedding for novel view synthesis. The results are quite good and the video is satisfactory.

### Strengths
The paper uses the Fashion VDM model well; the shared representation os garment seems a good idea.
The implicit training paradigm seems working for this problem quite well and the authors design the architecture elegantly.
The quantitative results are encouraging and the qualitative results are good.

### Weaknesses
It is not clear how the occlusion is solved? Which part of the network is responsible to solve this problem? Or is it by data?
Why Spinning is handled differently? Isnt spin a form of dynamic poses?
What is garment pose and how is it estimated?
If the garment has different texture (not symmetry) and they are visible in 1-3 views then how the method perform? Need a visual results for this.

### Questions
Please refer weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The authors in this paper presented a method for synthesizing state-of-the-art novel views of garments in real-world images and videos. An implicit training scheme is introduced to optimize a video diffusion model for real-world garment image-to-360° novel-view synthesis (NVS) using a combination of large-scale 2D garment data and limited synthetic 3D garment assets. 

However, the results are not impressive, and missing key experimental setups as discussed below.

### Strengths
Key strength of this paper is an implicit training scheme to optimize a video diffusion model for real-world garment image-to-360° novel-view synthesis (NVS) using a combination of large-scale 2D garment data and limited synthetic 3D garment assets.

### Weaknesses
- Video results are not impressive, there are many places where texture/pattern continuity across different views is broken. Results on clothes where some textual information is written etc could be a good example to validate this.
- There are many examples where shading effect in the input  images around folds and wrinkles were treated as part of textures and the same is reflected in the generated output which is wrong. Those folds and wrinkles must go away under certain body postures, but it is not happening. 
- Results on images where parts of garments are occluded with body parts are limited. Show more video examples with diverse loose garments, not on tight body hugging T-shirts etc. 
- Only one failure case is shown, whereas after seeing the supplementary video, I believe there could be more ?
- While on one side author motivated that there method leverages the limited 3D data and that is sufficient. On the contrary in the limitation section authors are saying "larger synthetic garment dataset may remedy such issues", which is contradicting the initial motivation.
- Speed is another concern, taking 30 minutes is a huge. Authors should add a detailed inference and training time analysis /profiling in the paper. Right now this information is missing in the paper.

### Questions
please see the weakness section

### Soundness
2

### Presentation
2

### Contribution
2
