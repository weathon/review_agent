# Diff3DS: Generating View-Consistent 3D Sketch via Differentiable Curve Rendering

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 6

## Abstract
3D sketches are widely used for visually representing the 3D shape and structure of objects or scenes. However, the creation of 3D sketch often requires users to possess professional artistic skills. Existing research efforts primarily focus on enhancing the ability of interactive sketch generation in 3D virtual systems. In this work, we propose Diff3DS, a novel differentiable rendering framework for generating view-consistent 3D sketch by optimizing 3D parametric curves under various supervisions. Specifically, we perform perspective projection to render the 3D rational Bézier curves into 2D curves, which are subsequently converted to a 2D raster image via our customized differentiable rasterizer. Our framework bridges the domains of 3D sketch and raster image, achieving end-to-end optimization of 3D sketch through gradients computed in the 2D image domain. Our Diff3DS can enable a series of novel 3D sketch generation tasks, including text-to-3D sketch and image-to-3D sketch, supported by the popular distillation-based supervision, such as Score Distillation Sampling (SDS). Extensive experiments have yielded promising results and demonstrated the potential of our framework. Project: https://yiboz2001.github.io/Diff3DS/

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces Diff3DS, a novel approach for generating view-consistent 3D sketches from either text prompts or image inputs. It employs a differentiable rendering framework that enables the optimization of 3D curve parameters through fully differentiable rasterized images. The method first projects parametric 3D curve control points into 2D using perspective projection. It then applies differentiable rasterization, similar to DiffVG, while correctly handling depth during curve intersections. The framework leverages Score Distillation Sampling (SDS) for supervision, using either a pretrained text-to-image model or an image-guided model. Additional techniques, such as dynamic noise deletion and time annealing schedules, further enhance optimization and generation quality.
Experiments show that Diff3DS generates consistent 3D sketches from both text and image inputs, outperforming previous sketch synthesis methods in terms of CLIP visual similarity and user preference.

### Strengths
1. The paper introduces a novel framework for 3D sketch generation, which, to the best of my knowledge, is the first to support generation from both images and text prompts.

2. The proposed method employs differentiable rendering that supports perspective projection and occlusion-aware rasterization for 3D curves, distinguishing it from earlier approaches.

3. Experimental results demonstrate that Diff3DS outperforms prior methods in both semantic alignment and user-preferred quality.

4. The writing is clear and the structure of the paper is easy to follow.

### Weaknesses
Although the paper is well-written and the generated sketches align with the given image and text inputs, there are concerns about the sufficiency of the contribution and the depth of the experimental evaluation:

1. One of the claimed novelty is the use of perspective projection with depth-aware rasterization. However, it seems this approach is not entirely new. A similar perspective projection has already been discussed in 3Doodle (specifically in Appendix A.2), which seems to have no clear difference with the procedure proposed in this paper.

2. While fixing occlusions in 3D curves is a positive addition to the rasterizer, there is no conclusive experimental evidence showing how this improvement impacts the quality of generation. For instance, it remains unclear whether depth-aware rasterization significantly enhances the final results beyond the occlusion analysis in Figure 11.

3. Another novelty is the use of SDS loss supervision, which has not been employed by previous methods. However, applying SDS loss to rasterized images is relatively straightforward and does not present a significant technical difficulty, given its widespread use in NeRF-related tasks. This loss could be easily incorporated into prior approaches without altering their core components. Moreover, the experiments do not convincingly demonstrate the advantage of direct SDS supervision on 3D curves, which I elaborate below.

4. In the image-to-3D sketch task, Diff3DS is compared with 3Doodle and NEF, showing better CLIP scores and improved 3D consistency. However, the fairness of this comparison is questionable, as Diff3DS benefits from multi-view supervision via SDS loss, while 3Doodle and NEF are trained only from a single reference view. Naturally, multi-view supervision leads to better 3D consistency. I think a better comparison would be another baseline like this: first train a 3D representation such as NeRF using the text2image or image2image SDS loss, then use the multi-view renderings of this representation to provide multi-view supervision for training other 3D sketch reconstruction methods, followed by a comparison in terms of quality and efficiency to truly demonstrate the benefits of directly optimizing 3D curves with SDS loss.

Given above points, I believe the paper's contributions may not be substantial enough to justify its acceptance. However, I am open to raise my score if the authors can provide a convincing explaination to the above concerns.

### Questions
1. In some examples, such as the phoenix, the 3D curves appear insufficient to fully capture the object's shape, and the curves still seem somewhat disorganized. Can this be solved with a higher number of 3D curves?

2. The method initializes curves within a fixed-size sphere. How heavily does this initialization impact the final optimization?

3. Why not consider optimizing additional parameters, such as the width, color, or transparency of the 3D curves? Could optimizing these properties lead to significantly better results?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
Nowadays, there are a lot of works conducting optimization of 3D models from 2D diffusion priors with SDS loss. Insteading of optimizing 3D models, this work aims to output 3D sketch which is also an important 3D format in related areas. To do so, a novel differatiable rendering framework is proposed. Experimental results validate the effectiveness of the proposed methods. Compared with existing differenatiable curve rendering methods, such as 3Doodle, the proposed method can maintain depth ordering which is very helpful for the opimization.

### Strengths
- In general, the paper is very well-presented. 
- To make 3D sketch rendering differienable is important and challenge. The paper proposed a novel one which enables the maintaining of the depth ordering. I also believe such rendering framework could benefit some other tasks. 
- The results seem very good. It's far better than existing ones visually.

### Weaknesses
There are also some issues which can be improved. 
1) Disscussions on the practical usage of 3D sketches can be improved. After the first glance of this paper, my primary curiousity is from the usage of 3D sketches. In the paper, the authors just mentioned "widely used" but did not provide examples. For me, it is really hard to imagine the usage of such 3D format. When thinking this in deep, I think 3D sketch/line can be an abstract representation of 3D structure, which can be an intermediate representation for some tasks like 3D reconstruction. In this senario, the differentiable rendering will also be very important. More discussions on this are strongly encouraged to make the application senarios more clear. More specifically, I think a subsection is needed to discuss those applications of 3D sketches. 

2) The evaluation is somewhat weak. As mentioned in the title and intro, view-consistency is the target of the proposed method. Howerver, there are only one user study to measure the view-consistency which is weak. The authors is encouraged to present some numeric results of view-consistency. For example, given two adjacent views, we can do a warping based on the optical flow estimation and then calculate the alignment error. If two views are consistent, the error should be minor. This metric can be compared with some baselines such as zero-123. 

3) I also have a concern about the number of curves. It seems the number of curves is a parameter needs to be set manually. This will increase the burden of the users. If it is so, please include this discussion into the limitation part.

### Questions
No more questions.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a framework for generating 3D sketches from either a text prompt or a single image. It is based on differentiable rendering of 3D Bézier curves and applies supervision from pretrained image diffusion models from random viewpoints.

### Strengths
* Covers multiple aspects of differentiable rendering.
* Extends DiffVG, a differentiable rasterizer for 2D vector graphics, to 3D Bézier curves.
* Presents a relatively simple and elegant pipeline, clearly summarized in Figure 3.
* Includes small but useful techniques that are well explained and analyzed:
    * Time annealing for the Score Distillation Sampling (SDS).
    * Curve filtering to reduce noise.

### Weaknesses
* The z-ordering of Bézier curves in Section 4.3 is the most technically advanced contribution. But the motivation is unclear: all sketches consist of black strokes, and intersections are rare at the image scale. Why care about correctly ordering the curves? Figure 11’s ablation shows only very minor visual differences. The effect of this ordering on 3D sketch generation is also left unclear.
* In the related works section, the motivations against existing 3Doodle approaches seem weak:
    * Issues with perspective projections.
    * Use of z-ordering for handling colored intersections.
* The usefulness and relevance of 3D sketches as shape representation is not fully clarified.

### Questions
* Why care about z-ordering of the curves? In which situation is it relevant?
* Figure 4: The meaning is unclear. Is this supposed to represent a pseudo-ground truth image? How does it relate to gradient supervision?
* In which situation are 3D sketches useful?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors present Diff3DS, a framework for optimizing 3D strokes using text or image guidance. This work introduces a novel extension of DiffVG that allows users to differentiably rasterize 2D projections of 3D parametric curves with more accurate depths / occlusions based on the 3D location of the curves. Using this new rasterization technique, Diff3DS applies SDS supervision to generate view-consistent 3D sketches using text or image guidance. This work validates its approach through ablations of key components and comparisons to other similar works. Diff3DS shows improved performance over other methods in both automated metrics and a perceptual user study. The paper is well written and easy to follow. The method is explained in sufficient detail and design decisions are properly motivated. Informative figures clearly showcase the contributions of this work (i.e. Figure 2 nicely demonstrates this ability to have non-uniform ordering between curves). The provided experiments validate claims made in the paper.

### Strengths
Strengths:
- Through the use of score distillation sampling for supervision, this work is the first to enable text/image driven 3D stroke generation from scratch (as opposed to existing works which simplify / abstract / sketchify an existing 3D representation defined by multi-veiw images).
- The novel differentiable rasterization technique introduced by this work allows for more accurate depths when rasterizing 3D curves.
- The paper is well written and organized.

### Weaknesses
Weaknesses:
- The main contribution of this work appears to be the improved differentiable rasterization approach that enables more accurate depths during rasterization. However, it is not clear that this component is responsible for improved performance. Figure 11 ablates the impact of this rasterization technique on 3Doodle. However, the differences are quite minor and at this level of abstraction, the depths of these curves seem less important to the visual coherence of the object.
- The comparisons currently shown do not use strong enough / similar enough baselines. Some of the comparisons shown are to standard (non-sketch) generative approaches. Showing these is fine, especially given the lack of existing work on text driven 3D consistent sketch generation. However, it feels that using score distillation guidance + DiffVG should still produce strong results for text/image guided curve generation. This is an important baseline and should be compared to in the paper.

### Questions
It appears from Figure 6 that, while your approach produces more 3D consistent results than 3Doodle, it less closely captures the structure of the guiding image. Do you have a sense of why that might be? Is it due to the fact that Diff3DS is more 3D consistent and thus is less able to overfit to exact structural details? Can this be improved?

### Soundness
3

### Presentation
4

### Contribution
3
