# TINKER: Diffusion's Gift to 3D--Multi-View Consistent Editing From Sparse Inputs without Per-Scene Optimization

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 4, 6, 6

## Abstract
We introduce TINKER, a novel framework for high-fidelity 3D editing without any per-scene finetuning, where only a single edited image (one-shot) or a few edited images (few-shot) are required as input. Unlike prior techniques that demand extensive per-scene optimization to ensure multi-view consistency or to produce dozens of consistent edited input views, TINKER delivers robust, multi-view consistent edits from as few as one or two images. This capability stems from repurposing pretrained diffusion models, which unlocks their latent 3D awareness. To drive research in this space, we curate the first large-scale multi-view editing dataset and data pipeline, spanning diverse scenes and styles. Building on this dataset, we develop our framework capable of generating multi-view consistent edited views without per-scene training, which consists of two novel components: (1) Multi-view consistent editor: Enables precise, reference-driven edits that remain coherent across all viewpoints. (2) Any-view-to-video scene completion model : Leverages spatial-temporal priors from video diffusion to perform high-quality scene completion and novel-view generation even from sparse inputs. Through extensive experiments, TINKER significantly reduces the barrier to generalizable 3D content creation, achieving state-of-the-art performance on editing, novel-view synthesis, and rendering enhancement tasks, while also demonstrating strong potential for 4D editing. We believe that TINKER represents a key step towards truly scalable, zero-shot 3D and 4D editing.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes a feedforward 3D scene editing pipeline and introduces a multi-view consistent image editing dataset. The method adopts a state-of-the-art large image editing model to generate sparsely edited views, and then employs a finetuned video model to reconstruct the scene based on the estimated depth maps of the original scene and the sparse edited views. The results demonstrate strong multi-view consistency, and the approach does not require per-scene optimization.

### Strengths
1. The paper proposes a feedforward pipeline for 3D scene editing, which differs from previous per-scene optimization-based editing methods.
2. By leveraging the power of 2D editing models, Tinker preserves the identity and consistency of multi-view sparse images; moreover, by exploiting the capability of a video diffusion model, it achieves precise reconstruction by concatenating depth maps and sparse views with full attention.
3. The writing is clear and easy to follow.

### Weaknesses
The method appears to struggle with structural editing of scenes. In particular, it is difficult to perform large geometric changes or significant deformations. This limitation arises because, during the scene completion stage, the approach relies on the depth maps of the original videos. Such dependency introduces inconsistencies between the edited views and the original depth maps when large deformations occur, which likely leads to degraded reconstruction quality.

### Questions
Based on the hypothesis mentioned in the weakness section, could the authors provide further insights or explanations regarding this limitation? Specifically, can the proposed model support large geometric deformations? If not, what are the potential directions to improve or extend the framework to handle such cases? I would be happy to hear the authors’ thoughts on this.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces TINKER, a framework for 3D editing that eliminates the need for per-scene diffusion model optimization, making it highly efficient compared to traditional methods. TINKER achieves multi-view consistency from as few as one or two edited images, enabling precise edits across different viewpoints.

### Strengths
1. The visual results are impressive and demonstrate the effectiveness of the proposed method.
2. The use of a video model for 3D editing is an innovative approach. With video generative priors, TINKER is the first method capable of jointly editing both 3D and 4D scenes.

### Weaknesses
1. There is an over-claim of contributions. Many baselines, such as DGE and GaussCtrl, also do not require fine-tuning the diffusion model. Therefore, this should not be considered a unique contribution of TINKER.
2. The majority of the baselines use InstructNerf2Nerf or ControlNet as the base 2D editors, whereas TINKER utilizes the FLUX model. It is unclear where the true improvement lies: is it in the advanced 2D editing model, or is it in the proposed pipeline? What if these baselines were equipped with the FLUX model?
3. AIGC tasks, such as 3D editing, necessitate a comprehensive user study to assess its performance in terms of human preference and subjective quality.
4. More visualizations of the collected training dataset should be presented to give readers a clearer understanding of the data and its characteristics.

### Questions
As with weaknesses

### Soundness
2

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
This paper introduces a framework for 3D editing that operates without per-scene fine-tuning. The approach consists of two core components: a multi-view consistent editor, built upon a large-scale image editing model and a multi-view image editing dataset, and a scene completion model that generates dense edited views. The method is evaluated on the Mip-NeRF-360 and IN2N datasets with qualitative and quantitative results.

### Strengths
1. This paper introduces a one-shot or few-shot approach for 3D editing.
2. This paper proposes a generalizable pipeline for 3D editing.
3. The paper is well-structured and easy to follow.

### Weaknesses
1. For the multi-view image editing model, when dealing with views that have large variations, does the multi-view consistency of the edits decrease? If so, could these inconsistencies be further propagated and amplified by the subsequent scene completion model?
2. Since the scene completion model relies on geometric information like depth maps, in few-shot or even one-shot settings with large view variations, is it prone to introducing more hallucinations or geometric distortions to fill in the missing information?
3. Although the method eliminates per-scene finetuning, its overall editing time, as shown in Table 1, does not present a significant advantage over some baseline methods. This suggests that the inference cost of the models involved might be a bottleneck.
4. In the supplementary video "Edited_Novel_View_Rendering.mp4" for the IN2N person scene, noticeable artifacts can be observed around the edges of the edited object. What is the primary cause of these edge artifacts?

### Questions
Please refer to the weaknesses.

### Soundness
2

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
3

### Summary
This paper, Tinker, has proosed a general-purpose 3D editing framework. Given the reconstructed 3DGS, the proposed editing pipelien will perform video depth estimation, muti-view consistent editsing, and scene completion sequentially. The edited 3dgs is of high quality under comprehensive evaluation.

### Strengths
1. The proposed method is efficient, reasonable, and offers high quality.
2. The writing is good.
3. The proposed dataset will be very helpful to this field.

### Weaknesses
1. The main issue is that this method looks very complicated, though it is necessary to make the 3D editing feed-forward. Still, too many components are involved in this process.

Overall, I think this is a good paper and worth acceptance. I just encourage the authors to think of the next step and tackle this task in a more elegant way.

### Questions
1. Since VDM has made great progress recently, 3D-aware VDM like Lyra, Gen3C, and VIST3A has also shown good capability. I wonder whether the proposed pipeline can be radically replaced by the VDM-based pipeline.
2. Besides, 3D foundation models are getting better now. Rather than directly working on the 3DGS, I wonder whether the proposed pipeline can be improved to incorporate 3D VFMs like VGG-T / AnySplat to facilitate easier 3D reconstruction editing.

### Soundness
4

### Presentation
4

### Contribution
4
