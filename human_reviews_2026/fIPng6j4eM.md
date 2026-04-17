# Scaling Sequence-to-Sequence Generative Neural Rendering

- Decision: Accept (Poster)
- Scores: 2, 6, 6, 4

## Abstract
We present Kaleido, a family of generative models designed for photorealistic, unified object- and scene-level neural rendering. Kaleido is driven by the principle of treating 3D as a specialised sub-domain of video, which we formulate purely as a sequence-to-sequence image synthesis task. Through a systemic study of scaling sequence-to-sequence generative neural rendering, we introduce key architectural innovations that enable our model to: i) perform generative view synthesis without explicit 3D representations; ii) generate any number of 6-DoF target views conditioned on any number of reference views via a masked autoregressive framework; and iii) seamlessly unify 3D and video modelling within a single decoder-only rectified flow transformer. Within this unified framework, Kaleido leverages large-scale video data for pre-training, which significantly improves spatial consistency and reduces reliance on scarce, camera-labelled 3D datasets --- all without any architectural modifications. Kaleido sets a new state-of-the-art on a range of view synthesis benchmarks. Its zero-shot performance substantially outperforms other generative methods in few-view settings, and, for the first time, matches the quality of per-scene optimisation methods in many-view settings. For supplementary materials, including Kaleido's generated renderings and videos, please refer to our website: https://shikun.io/projects/kaleido.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes Kaleido, a diffusion transformer for neural rendering. The model is jointly trained on 2D video and 3D data to take image(s) and output images from novel viewpoints. The paper proposes a new positional encoding to facilitate joint training on 2D video and 3D data.

### Strengths
- New positional encoding layer: This work combines RoPE for 2D video and 3D to jointly train on both data modalities.

### Weaknesses
- Limited viewpoint changes: The synthesized videos don’t show very large viewpoint change or show any synthesized “new” content not available in the input. It seems like the method requires many images to increase the viewpoint change. Though, methods like CAT3D can synthesize large viewpoint changes from a single image. Moreover, recent camera-controlled video models can synthesize very long trajectories.
- Flat geometry: Most generated trajectories have very flat geometry without good depth.
- Missing comparisons with camera-controlled video diffusion: This work looks conceptually similar to camera-controlled video diffusion models but comparisons with those kind of works are missing and the key differences in the approach are not clear. Moreover, in that case it would be possible to just use MegaSaM [1] or ViPE [2] to annotate all 2D videos with camera poses and directly train on large-scale video data for novel view synthesis.
- Simple datasets: The model is trained with limited scale 3D datasets. Is it a current issue that the model is not cross-generalizing from the synthetic 3D datasets to the video dataset distribution?

[1] Li et al., MegaSaM: Accurate, Fast, and Robust Structure and Motion from Casual Dynamic Videos, CVPR 2025 \
[2] Huang et al., ViPE: Video Pose Engine for 3D Geometric Perception, arXiv 2025

### Questions
I am not very convinced by the paper. It makes a big deal out of jointly training on 2D and 3D data, but it is not clear what the advantage or novelty of this work is. There are many camera-controlled video diffusion models that can just be used for joint fine-tuning. The architecture of Kaleido looks very similar to that of a regular diffusion transformer.

I would like authors to address the following questions:

- How does this compare to camera-controlled video diffusion models qualitatively and quantitatively? And why would someone not just jointly fine-tune a video model on 2D video data and multi-view data?
- Why are the viewpoint changes limited so much?
- Why is the geometry often flat?

I am not very optimistic that I can be convinced to accept this paper, since it is missing critical comparisons and the results are not very convincing. Moreover, the storyline is not convincing to motivate why we need this from-scratch-trained model. But still happy to see a rebuttal.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presented a method to generate novel view images from input views using the sequence-to-sequence generative model. They conducted numerous validate experiments to design the final model, such as the positional encodding, model architecture, and video pretraining, which can achieve the certain consistent view synthesis from both sparse and many view conditions.

### Strengths
1. The model design of this paper is motivated by many ablation studies. This paper designed the model mainly from five aspects, including the positional encodding, activation function and so on, and most designs have the corresponding ablation studies.
2. This paper conducted both 3D reconstruction and novel view synthesize comparisons with existing methods and show its advantage.
3. The results with many-view setting show that it can achieve more consistent results compared with existing methods.

### Weaknesses
1. Adopting the generative model to synthesis the novel view images is a common solution and is just one posible solution to achieve the final reconstruction of 3D object or scene. It has clear advantage in challenging situation like generating the non-seen views, but on the contraty, it has clear shortcoming in the consistency rendering results and efficiency.
2. To my knowledge, many previous methods have treated the multi-view synthesis as the video generation, and they can also generative any number of target views conditioned by any number of input views. And the declared novelty in this field is not clear to me. 
3. In the many-view setting, seems like luck the experiments with long sequence where the inputs not just have numerous views but have long trajectory like SEVA.
4. Most of the technologies used may not seem novel, but the key is to choose the right combination.

### Questions
Missing the information of the memory and runtime comparisons with both previous generative methods and reconstruction methods like 3DGS, and how many input views can be processed at most and the corresponding runtime time?

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
This paper presents a sequence-to-sequence method to achieve multi-view image synthesis, and they construct the pipeline from the positional encoding to the video pretraining. The final results show that this model has advantage in both single or multiple view input.

### Strengths
Strengths:

- Proposed the unified positional encoding to seamlessly represent multi-view and temporal positions.

- The presentation structure is reasonable and most design modules have corresponding ablation studies.

- The results with single or multiple view inputs show the advantage in both 3D and novel view synthesis.

### Weaknesses
Weakness:

- Missing the discussion on the running time and memory consumption (especially with the classical reconstruction representations like 3DGS). This kind of generation methods has the advantage on the generation of the nun-seen part, but it still needs huge memory consumption and long running time.

- From the proposed demos, the generated results still have obvious ghosting and artifacts (e.g., the generation attempts at the same location is inconsistent), this is still the inherent disadvantage of this kind of generation methods.

- How about the results when the input views and the target views have large difference in perspective? Maybe this challenging situation can further prove the effectiveness.

### Questions
As mentioned in the Strengths and Weaknesses, I am inclined to give a borderline-accept score; however, I would be happy to raise it if the authors address these main concerns.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Method Kaleido solved the object- and scene- level synthesis through the sequence-to-sequence generation technology. And they mainly borrowed the video generation model to achieve this and assisted some structural designs like activation function to improve the performance.

### Strengths
- The results show that this method have advantage in both few- and many- view settings, and they can match the performance of classical 3D representations under the many-view setting.

- This paper explored many detailed model designs including the positional encoding and architecture designs through ablation studies.

### Weaknesses
- Adopting the video generation to achieve novel view synthesis is a common solution, and many methods still adopt the video pretrained model to improve the performance. So the contribution on the video pretrained model is unconvincing.

- Once this model unified the 3D and video, so how about the performance on the 4D scene.

- The demo results still have obvious inconsistency in the generated novel view.

### Questions
N/A

### Soundness
2

### Presentation
3

### Contribution
2
