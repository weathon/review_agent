# USB-NeRF: Unrolling Shutter Bundle Adjusted Neural Radiance Fields

- Avg Score: 5.50
- Decision: Accept (poster)
- Scores: 3, 8, 3, 8

## Abstract
Neural Radiance Fields (NeRF) has received much attention recently due to its impressive capability to represent 3D scene and synthesize novel view images. Existing works usually assume that the input images are captured by a global shutter camera. Thus, rolling shutter (RS) images cannot be trivially applied to an off-the-shelf NeRF algorithm for novel view synthesis. Rolling shutter effect would also affect the accuracy of the camera pose estimation (e.g. via COLMAP), which further prevents the success of NeRF algorithm with RS images.
In this paper, we propose Unrolling Shutter Bundle Adjusted Neural Radiance Fields (USB-NeRF). USB-NeRF is able to correct rolling shutter distortions and recover accurate camera motion trajectory simultaneously under the framework of NeRF, by modeling the physical image formation process of a RS camera.
Experimental results demonstrate that USB-NeRF achieves better performance compared to prior works, in terms of RS effect removal, novel view image synthesis as well as camera motion estimation. Furthermore, our algorithm can also be used to recover high-fidelity high frame-rate global shutter video from a sequence of RS images.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
# Summary
This paper integrates the rolling shutter model into NeRF training. The proposed method assumes the given images are rolling shutter images. Given the specific camera motion trajectory model, the rendered rolling shutter images can be computed from the global shutter images from the NeRF model. The camera trajectory and the NeRF model can be optimized by minimizing the loss between rendered rolling shutter images and the input images.

### Strengths
# Strength
- good performance. Since the experiments are done on the rolling shutter dataset, it is unsurprising that the proposed method outperforms previous methods like barf.
- The idea is straightforward, like some previous NeRF-based methods integrating another luminance/render/texture/deblur model, like BRDF.

### Weaknesses
# Weakness
- The contribution and the novelty are limited. The effect of the rolling shutter is well-known in 3D vision community, and many works have been trying to solve it in the past years. The proposed methods only contain the basic concept of modeling a rolling shutter.
- It seems like the proposed method uses the specific motion model of the camera. It might prevent the proposed method from working on the global shutter dataset(no experiment to prove it) and another dataset that cannot be modeled in bicubic motion(no experiment to prove it).
- The proposed method requires COLMAP to provide an initial camera pose, which is only mentioned in the footnote.
- To sum up, due to the abovementioned concerns, I cannot give a positive rating to the proposed method in the current version.

### Questions
A Possible Direction for improvement is to regard the camera motion model as the parameters in the Nerf model and solve it during optimization. To this end, the proposed method can be more general to handle global and rolling shutter datasets. It will be a benefit for the community further.

### Soundness
2 fair

### Presentation
3 good

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposes a method called Unrolling Shutter Bundle Adjusted Neural Radiance Fields (USB-NeRF) that corrects rolling shutter distortions and improves the accuracy of camera pose estimation. USB-NeRF models the physical image formation process of a rolling shutter camera and uses a bundle adjustment framework to optimize the camera poses and scene geometry. The technique unrolls the rolling shutter effect by modeling the exposure time of each pixel and correcting the time-varying motion of the camera. USB-NeRF also uses a neural radiance field to model the scene geometry and appearance, which allows for high-quality novel view synthesis. The paper includes tables and figures that show the quantitative and qualitative comparisons of USB-NeRF with other methods on synthetic and real-world datasets. The experimental results demonstrate that USB-NeRF achieves better performance compared to prior works in terms of RS effect removal, novel view image synthesis, and camera motion estimation.

### Strengths
1. This paper is very well written. I can easily understand the paper even though I'm not very familiar with the rolling-shutter camera.

2. The proposed method is simple yet effective, which uses the cubic B-Spline to interpolate between camera poses instead of linear interpolation.

3. The paper did exhaustive experiments to evaluate the effectiveness of their method on both the synthetic and real-world datasets. Though there is a lack of baseline methods for rolling-shutter NeRF, they compared with various methods that bundle-adjust rolling-shutter cameras.

### Weaknesses
1. I think the ATE (absolute trajectory error) in Table 3 is the same as the absolute translation error (I'm used to the term `translation` instead of `trajectory`). Therefore, only the translation errors are given and no rotation errors are provided. Moreover, the unit of the ATE is unclear (I think it is in meters).
2. The cubic B-Splines interpolation is suitable for complex camera trajectories, however, it can be worse than the linear interpolation method when the camera moves at a constant velocity.

### Questions
- From Fig. 6, BARF looks much worse than NeRF; and from other tables and figures, BARF performs almost the same as NeRF. Are there any explanations for this?

- Follow the question above. It is expected that BARF can fail under the rolling-shutter setting since each row of the image is recorded at different timestamps. My question is can we build a stronger baseline method that associates each row with a camera pose, then we can use BARF to optimize these camera poses and obtain better results? It can be time-consuming since an image can often have >400 rows, and then we have to optimize too many parameters. A simplified way is to split the image into R row blocks, where R <<< the width/height of that image. Then we have a simplified stronger baseline than BARF.

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes a method to handle rolling shutter problem in NeRF reconstruction. In particular, it proposes a method to rectify the input images caused by rolling shutter followed by NeRF reconstruction. Experimental results show that the proposed method can handle distortion caused by rolling shutter effectively.

### Strengths
The proposed method is evaluated on both synthetic and real world dataset and demonstrated the improvement of reconstruction with and without the rolling shutter correction.

### Weaknesses
I am not fully convinced that using rolling shutter camera is an effective way to capture a NeRF model. There is actually no motivation/benefits to use rolling shutter camera to capture a NeRF model.

Considering the case that using rolling shutter camera is necessary, the proposed solution is just a simple two-step approach with first rolling shutter correction followed by NeRF reconstruction. I do not see any connection between rolling shutter correction and NeRF reconstruction in the proposed method. Since there is no connection, the rolling shutter correction method is just a standard method which estimate motion trajectory followed by rectification. From the formulation, I do not see any technical novelty.

### Questions
Please try to correct me if I have made any mistakes on the evaluation of this submission.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes a method to model rolling shutter effect during NeRF training. The main idea is to model camera trajectory with a B-spline, which allows time interpolation so that each scanline in an image can be associated with a more accurate camera pose at the moment of when the line of pixel is captured.

### Strengths
1. The paper is well written and easy to follow. 
2. The proposed idea is novel and technically solid.
3. The evaluation is convincing and supports the main contribution well.

### Weaknesses
I don’t see any major weakness. 

It would be interesting to see more analysis/discussions on how much is the performance gap when modelling rolling shutter effect vs not modelling with datasets from modern DSLR cameras and smart phone cameras.

### Questions
See the weakness section.

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
3 good
