# MoCtrl4D: Precise and Efficient Motion-Guided 4D Content Generation

- Avg Score: 2.50
- Decision: Reject
- Scores: 2, 2, 4, 2

## Abstract
Promptable 4D generation is a crucial task with broad applicability across industries, thus has recently gained tremendous interest in research community. However, existing works remain predominantly limited to image and text conditioning, which neglect the nuances of motion controllability. To address this, we propose to use dynamic motion prompt defined by any number of point trajectories. 
To translate user intention into this motion representation, we design a user-friendly interface that allows users to intuitively input motion trajectories, bringing images to life through direct interaction. Unlike prior works, in leveraging prior knowledge of a base reconstruction model, our method integrates prompts without added modules, maintaining scalability and data efficiency without overhead, achieving a full forward pass in under a second. Furthermore, instead of relying on existing appearance-focused learning frameworks, which suffers from poor motion fidelity, we design a novel physically inspired \textit{Vector Consistency Loss (VCL)} function for explicit motion learning. 
Our quantitative and qualitative results show significant improvement in spatiotemporally-precise and expressive control.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes MoCtrl4D, a pipeline for controllable 4D generation. Compared to previous works, which mainly rely on text to define motion control, this work controls motion through point trajectories. Given static images and motion trajectories, the method synthesizes 4D Gaussians that can be rendered from any viewpoint and timestep. The model is supervised with appearance and geometry losses separately.

### Strengths
- Important task: Motion-controllable 4D generation is an important and underexplored task.

### Weaknesses
- Weak results: The quantitative results are not very convincing, and the qualitative results lack much motion. The videos are generally super short and it is not clear why the motion is so limited, since Objaverse training data does have larger amount of motion.
- Weak result presentation: The supplementary video material, crucial for such 4D generation papers, is not very polished. I highly recommend a supplementary website with side-by-side comparisons with previous works. The results also do not have any full 4D visualizations, i.e., space-time renderings where both time and viewpoint are changing, as well as freeze-time and freeze-camera renderings. 
- Missing comparisons: The paper does not compare with motion-controllable 4D generation methods, e.g., MagicPose4D [1] or SP4D [2]. It mainly compares with L4GM, a feed-forward 4D reconstruction method that models each time separately and does not have any motion control.

[1] Zhang et al., MagicPose4D: Crafting Articulated Models with Appearance and Motion Control, TMLR 2025 \
[2] Zhang et al., Stable Part Diffusion 4D: Multi-View RGB and Kinematic Parts Video Generation, NeurIPS 2025

### Questions
I am convinced by the task the paper is trying to solve but the results are not convincing.
I would like authors to address following questions:
- Why is the motion is so limited?
- How does the method compare to MagicPose4D or SP4D, i.e., other motion-controllable 4D generation frameworks?

I am pretty negative and it is rather unlikely that I will raise my score to an accept but I am still happy to consider a rebuttal.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces MoCtrl4D, a feed‑forward, motion‑prompted 4D content generation framework that lifts a single image plus user‑drawn point‑trajectory prompts into dynamic 3D Gaussians with an intuitive UI and sub‑second forward pass claims. The core ideas are a trajectory‑image injection that avoids extra control modules and a Vector Consistency Loss (VCL) that complements ARAP to supervise per‑Gaussian motion more precisely than appearance‑only training.

### Strengths
1. Introduces a direct motion‑prompt mechanism using point trajectories that provides fine‑grained, expressive control beyond text/image prompts.​
2. Proposes “trajectory images” that integrate seamlessly into a base reconstruction model, avoiding ControlNet‑style module duplication and extra parameters.
3. Novel VCL loss for more specialised motion control.
4. Mitigates opacity‑driven degeneracies by predicting appearance only at the first frame while updating geometry over time, improving temporal consistency.

### Weaknesses
1. The presentation is extremely poor. There is no direction to understand the supplementary material videos. I suggest using a html page so that the reviewers understand what the videos indicate and what exactly to interpret. Authors have just poured in a bunch of videos in the suplementary material without proper direction to understand them. 

2. As a work on dynamics and 4D content generation, I would request atleast seeing 20-30 videos to see the fidelity of the work and not just assume that these are just cherry-picked results. Also few videos in the supplementary almost have no motion at all. It would also be good to see the same object undergoing different motion trajectories, like atleast 2-3 different motions to understand the generalisability of the model. So same object but different motion points to get different 4D gaussians and visualise their dynamics. 

3. The paper is also really poorly written. It is extremely difficult for the reviewer to understand what the author is trying to convey at several places like: Fig 2 caption: Figure caption should be extremely descriptive and should describe each module properly. In the figure, you have kept a bar graph of what I assume is Resblock, Multi-view attn and Temporal attn but what do the height signify? Please be more clear. Also in the input, I am really confused as to what exactly is the input? Why does the first row contain 4 images and then 2nd and 3rd row to have one point image and 3 rgb images and again 4th row contains different set of rgb images. Please properly describe the input and output space. 

4. The comparisons are very limited. I don’t see a single image except Fig 3, which has only comparison with one method on only 2 data samples. I would request the authors to provide more comparisons on more scenes and also comparisons with more baselines and previous works. This level of comparison is not acceptable for the standard of this conference. 

5. It would also be great to include a user-study done on atleast 10-15 scenes with varying object motions and view points. More photos of the UI could be shown atleast in the appendix to understand the flexibility of motion control in the UI. 

6. How does the model scale to real-world objects. All the results are shown on synthetic dataset which is good but what matters is results on real-world examples. It would be great if you could show results on few real-world objects. 

7. Also how does the model scale to longer motion trajectories. There are no discussions on failure cases which I recommend adding and explaining the limitations of the work.

### Questions
I have highlighted the major questions in the weaknesses already, and I am summing them up below(I have summarised them shortly so that it is easier for the reviewer to quote the exact question they are answering. Please refer to the Weakness for the detailed problem and question asked.

1. Could you organize the supplementary videos in a clearer way, perhaps using an HTML page, so that it’s easier for readers to understand what each video shows and what we’re supposed to interpret from them? Right now, the supplementary material feels like a random collection of videos without structure or explanation.

2. Why are there so few videos presented? Since this paper focuses on 4D dynamics and content generation, it would be good to see at least 20–30 videos to properly judge fidelity. With so few examples, it’s hard to tell if the results are representative or cherry-picked.

3. Some supplementary videos seem to have almost no motion at all. Could you explain why that is the case? Are those examples meant to show static scenes, or is the model struggling to generate realistic motion?

4. Could you include examples where the same object undergoes different motion trajectories, say two or three variations, to test whether the model generalizes well and to better visualize its learned 4D Gaussian dynamics?

5. In Fig. 2, could you clarify what the bar heights represent for the ResBlock, Multi-View Attention, and Temporal Attention modules? The caption and figure are not clear on what those heights signify.

6. What exactly is the input to the model? The figure is confusing because the first row shows four images, the second and third rows have one point image and three RGB images, and the fourth row has a completely different combination. Could you describe the input and output spaces more clearly?

7. Why are the comparisons so limited? Apart from Fig. 3, which only compares to one method on two samples, there are no other comparisons. Could you add more comparisons with other baselines and across a wider range of scenes?

8. Would it be possible to include a small user study on around 10–15 scenes with varying object motions and camera viewpoints? That would help quantify perceptual quality and realism.

9. Could you include more images of the UI (perhaps in the appendix) to show how flexible the motion control interface is?

10. How does the method perform on real-world scenes? Currently, all the results seem to be on synthetic data. It would strengthen the paper to show at least a few examples on real-world objects.

11. How does the method handle longer motion trajectories? There is no discussion of whether performance degrades or artifacts appear with extended motion.

12. Could you include a section discussing failure cases and the main limitations of your approach? This would help readers understand where the method works well and where it struggles.

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
MoCtrl4D is a motion-promptable 4D Gaussian generation framework that takes a single image + user-specified motion trajectories as input and outputs dynamic 4D Gaussians in a single forward pass. Instead of text prompts, it encodes explicit 2D point trajectories into a “trajectory image” and feeds it directly into an existing L4GM video-to-4D reconstruction model without adding any new control modules. To improve motion fidelity, it proposes a new Vector Consistency Loss (VCL) to complement ARAP and better enforce rigid yet directionally correct deformation. Experiments on Objaverse show improved motion controllability (EPE) over L4GM while maintaining similar appearance quality.

### Strengths
1. Using explicit 2D trajectories as motion prompt is a very reasonable and underexplored alternative.

2. The UI in Fig. 4 is intuitive and gives users precise, fine-grained control over where to move and how to move.

3. The trajectory prompt is natively injected via “trajectory image” encoding, without increasing model size. This is clean and lightweight design.

4. VCL loss is a well-justified improvement over vanilla ARAP, addressing the translation vs. rotation ambiguity that ARAP alone cannot disambiguate.

### Weaknesses
1. The output 4D quality is still visually limited. Motions are generally small, conservative, and low-energy (mostly local deformations, not full articulated or large trajectory motion).

2. Appearance fidelity is mediocre. Textures look relatively flat/synthetic / Objaverse-style, clearly lagging behind 4Real, Imagen-3D, DynamiCrafter, or even modern feed-forward Gaussian works. It feels more like reconstructed animation than genuinely generated cinematic 4D content.

3. Relies almost entirely on synthetic Objaverse rigged assets, lacking evidence of generalization to real images or real natural motion.

4. Core novelty is not architectural. It is fundamentally an extension of L4GM with trajectory conditioning + new motion supervision loss. The “no extra control module” aspect is good engineering, but not necessarily a strong research novelty — many will see this as “just encoding trajectories into RGB channels” rather than a deep innovation.

### Questions
1. How robust is your method to natural photos or messy in-the-wild images?

2. Does your method still work if the motion trajectories are large or non-rigid?
Can it handle non-linear / high-amplitude / self-occluding motion? Or does it break because the model was only trained on Objaverse’s mild rig motions?

3. How do you guarantee the generated motion is physically plausible?
The VCL loss enforces relative vector consistency, but is there any failure case where Gaussians drift or jitter unnaturally? A failure case visualization would be important to assess reliability.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces a promptable 4D generation framework that enables motion control through user-defined point trajectories. It provides an intuitive interface for users to create motion prompts and animates static images interactively. Unlike prior methods that add complex modules, this approach integrates prompts directly into a base reconstruction mod. Additionally, it proposes a Vector Consistency Loss (VCL) to improve motion fidelity based on physical principles, overcoming the limitations of appearance-focused methods.

### Strengths
•  The paper addresses an important problem in 4D generation.

•  Providing users with an interactive experience is very interesting.

•  The results demonstrate a diverse range of cases.

### Weaknesses
1.	How does this work differ from SC4D [1], which also uses sparse point control?
[1] SC4D: Sparse-Controlled Video-to-4D Generation and Motion Transfer

2.	From the demonstrations, it seems that users need to perform multiple operations and drag several points. On average, how much time does it take to manipulate one example during inference? How many keyframes need to be adjusted — only the first and last frames, or multiple ones? Would using more frames improve accuracy while increasing interaction time?

3.	Previous works mainly focused on video-to-4D methods. With the recent progress in controllable video generation and stronger video base models (e.g., Wan, Veo3), how should we evaluate the quality of control signals? My understanding is that video-to-4D approaches use a short front-view video as input, while your method relies on a single image and sparse points. How do you compare the advantages of these two settings?

4.	Compared with rigging-based methods, where motion sequences are either learned from data or derived from motion libraries, your point-dragging approach seems to require heavier user interaction. For complex actions such as lifting a leg or dancing, rigging methods tend to produce smoother and more natural motion. In contrast, I feel that manually controlling motion through sparse points may lead to less natural and less continuous results.

5.	According to Table 1, there appears to be little performance improvement. Could this be because the model initializes L4GM weights and uses the same datasets? Compared with L4GM, this work employs the same architecture with an additional input signal, which makes the contribution seem somewhat incremental.

6.	Due to the limitations of the L4GM architecture, the model can only generate very short clips — the supplementary materials show results of just about one second. The visual results are not particularly impressive, and it’s unclear what practical value a one-second 4D generation result could have.

7.	In the supplementary materials, the ablation experiments (Exp1–5) are not clearly explained — it’s hard to tell what each experiment corresponds to.

### Questions
Please refer to the weeknesses.

### Soundness
2

### Presentation
2

### Contribution
2
