# Positional Encoding Field

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 6, 2, 6

## Abstract
Diffusion Transformers (DiTs) have emerged as the dominant architecture for visual generation, powering state-of-the-art image and video models. By representing images as patch tokens with positional encodings (PEs), DiTs combine Transformer scalability with spatial and temporal inductive biases. In this work, we revisit how DiTs organize visual content and discover that patch tokens exhibit a surprising degree of independence: even when PEs are perturbed, DiTs still produce globally coherent outputs, indicating that spatial coherence is primarily governed by PEs. Motivated by this finding, we introduce the Positional Encoding Field (PE-Field), which extends positional encodings from the 2D plane to a structured 3D field. PE-Field incorporates depth-aware encodings for volumetric reasoning and hierarchical encodings for fine-grained sub-patch control, enabling DiTs to model geometry directly in 3D space. Our PE-Field–augmented DiT achieves state-of-the-art performance on single-image novel view synthesis and generalizes to controllable spatial image editing.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
In this paper, the authors find that global spatial coherence comes from positional encodings (PEs) rather than token–token coupling, in which perturbing PEs preserves semantics while reorganizing layout. Based on this insight, the authors investigate the effect of PEs in the image generation process. And a new and neat positional encoding method is proposed that combines a 3D encoding (additional depth information) and a fine-grained patch encoding. With the proposed positional encoding manner, the PE-Field–augmented DiT achieves state-of-the-art performance on single-image novel view synthesis.

### Strengths
1. Interesting Concept Finding. The authors find that PEs control the spatial structure coherence during image generating or reconstructing process, motivating a new potential way for visual editing or synthesis.

2. Novel and Effective Positional Encoding. The authors consider the depth-aware information and more fine-grained semantics in the patch, then propose a new positional encoding that locates both the mentioned information.

3. Good Synthesis Result. With the proposed positional encoding, the introduced PE-augmented DiT achieves SOTA performance on single-image novel view synthesis.

### Weaknesses
1. Over-dependence on Monocular Reconstruction. As shown in the main framework, the monocular reconstruction provides essential 3D positional information for new view generation. How about the model’s robustness to terrible or noisy 3D position info injection?  

2. Concerns about Multi-Level Positional Encoding. The head-layer match in fine-grained positional encoding is handcrafted in order.  More matching ways should be conducted to verify its robustness for fine-grained positional encoding.

3. More View Generation Scenes Discussion. 1) The visualization result from unseen view to seen view (i.e., view from left face to front face) should be reported. 2) Object-level or Region-level view synthesis. 

4. Computing Cost Discussion. Since the proposed framework relies on monocular reconstruction, it may incur additional cost.

### Questions
1. Does the text-editing prompt influence the 3D-Positional encoding for view synthesis based on Flux Kontext? 

2. Additionally, what is the instinct difference for text-control and PEs-control for view synthesis (visual editing and generation) during the training process?

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
3

### Summary
This paper presents a positional encoding field for novel view synthesis based on the DiT model. It is inspirted from the shuffling experiment and visualization of the position encoding for a standard model. The proposed method includes two parts: multi-level positional encodings for sub-patch detail modeling, and depth-aware extension to the rotary encoding. The proposed method has been evaluated on three datasets: Tanks-and-Temples, RE10K, and DL3DV. Both quantitative and qualitative experimental results demonstrate the effectiveness of the proposed method.

### Strengths
1.[effectiveness] The proposed method consistently outperforms the previous methods both qualitatively and quantitatively.

2.[motivation] The proposed method is well-motivated. It starts from an observation by shuffling the positional encodings. That helps the readers to better understand the motivation and design of the proposed method.

3.[ablations] Ablation studies has been carried out to demonstrate the effectiveness of individual components in the proposed method in section 4.3

4.[extension] The proposed method has probably more potential to be used in other spatial editing tasks as mentioned in section 4.4

### Weaknesses
1.[clarity] As described on L033-041, as well as figure 1, the proposed method is motivated by a position encoding shuffling experiment. What would be the worst case for a shuffled or re-ordered position encoding? How does that impact the model? If there is a threshold beyond which the generated output is completely messed up, the authors might have to rethink the connection between the position encoding shuffling expeirment and the proposed method.

2.[typesetting] The macro "\citep" or "\citet" should be used instead of a plain "\cite". The brackets are missing in manyplaces like the related works section, which makes it not easy to read.

3.[clarity] The caption of figure 1 is confusing. What does "perturbed" mean? The paper and the figure says "warp, shuffle, reorder". What is re-order again? Are they the same thing? Please use a consistent term to avoid confusion.

4.[clarity] On line 145, how are they re-ordered? Is it random?

5.[clarity] On line 246: how is the depth obtained during the training and testing process of the proposed method?

6.[design] The proposed method is only tested on the Flux model. Does it apply to other DiT-based models, such as the MMDiT (stable diffusion 3)? If so, the proposed method would be helpful for a wider community.

7.[fair comparison] In table 1, the proposed method outperforms the previous methods. But not every other method is built-on the Flux model. Therefore, it is unknown how much performance improvement originates from the Flux backbone itself. This would raise a fair comparison issue, and leave the real improvement of the proposed method in question.

### Questions
Generally this is a good paper. However the fair comparison issue makes it difficult to identify the real improvement made by the proposed method since it is mixed together with the powerful Flux backbone. Please see the questions in the weaknesses part.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper builds upon a very interesting idea that DiT attention pattern is very very local. Starting from this idea, the authors gained the intution of editting the position/strcture/viewpoints of the image through editting the positional encoding of the DiT. The author conducted simple experiments in Figure-1 to show this concept. 



Given these insights and polit study, the author developed more sophisticated designs for 3D-aware positional encodings. With this, the author builds upon flux-context (an open source image editting model) to implement his algorithm. The author finetuned the model on DL3DV MannequinChanllege with processed depth-map and camera poses.  After finetuning, the author showed impressive results with NVS on portaits images, object images and sceneray images.  Even though a simple call of the model can only generate relative view shifts, the author showed larger view-shift results by recurrsively applying the model.  When comparing with other methods, the author showed a higher PSNR.

### Strengths
1. The idea is quite neat. With polit study to build up intuitions, and interesting designs, and also good experiment design to validate the idea
2. The results are quite impressive! The visual quality is quite high. I do understand that each simple forward pass of the model can only edits a small view shifts. To address this, the author also showed recurssvely applied results, which is also quite cool. 
3. The overall presentation is quite clear, and I can easily follow the logics of the author.

### Weaknesses
I lean towards accept the paper. 

There are only several minor points, I don't think these are real weakness. 

1. This methods is hard to be applied to multi-view novel view synthesis.  Maybe it's still possible is to build one single reference images by merging patches from multiple input views?  
2. This demo would be even cooler if there are videos attached.  (Novel-view-synthesis videos with smoothly moving cameras.)

### Questions
Is it possible to build a training free baseline using the warpped 2D RoPE?  Cause from the intuition of Figure-1, it seems possible to build a training free baseline?  Correct me if I am wrong.

### Soundness
4

### Presentation
4

### Contribution
3
