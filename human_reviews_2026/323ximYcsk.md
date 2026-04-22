# DA$^{2}$: Depth Anything in Any Direction

- Avg Score: 6.40
- Decision: Accept (Poster)
- Scores: 8, 6, 6, 6, 6

## Abstract
Panorama has a full FoV (360$^\circ\times$180$^\circ$), offering a more complete visual description than perspective images.
Thanks to this characteristic, panoramic depth estimation is gaining increasing traction in 3D vision.
However, due to the scarcity of panoramic data, previous methods are often restricted to in-domain settings, leading to poor zero-shot generalization.
Furthermore, due to the spherical distortions inherent in panoramas, many approaches rely on perspective splitting (\textit{e.g.}, cubemaps),
which leads to suboptimal efficiency.
To address these challenges, we propose $\textbf{DA}$$^{\textbf{2}}$: $\textbf{D}$epth $\textbf{A}$nything in $\textbf{A}$ny $\textbf{D}$irection, an accurate, zero-shot generalizable, and fully end-to-end panoramic depth estimator.
Specifically, for scaling up panoramic data, we introduce a data curation engine for generating high-quality panoramic depth data from perspective, and create $\sim$543K panoramic RGB-depth pairs, bringing the total to $\sim$607K.
To further mitigate the spherical distortions, we present SphereViT, which explicitly leverages spherical coordinates to enforce the spherical geometric consistency in panoramic image features, yielding improved performance.
A comprehensive benchmark on multiple datasets clearly demonstrates DA$^{2}$'s SoTA performance, with an average 38\% improvement on AbsRel over the strongest zero-shot baseline.
Surprisingly, DA$^{2}$ even outperforms prior in-domain methods, highlighting its superior zero-shot generalization.
Moreover, as an end-to-end solution, DA$^{2}$ exhibits much higher efficiency over fusion-based approaches.
Both the code and the curated panoramic data have be released.
Project page: https://depth-any-in-any-dir.github.io/.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper proposes DA$^2$, an end-to-end, panoramic depth estimation model that achieves strong zero-shot generalization.

Previous methods face challenges in zero-shot panoramic depth estimation because:
1. [Data scarcity] Existing panoramic depth datasets are rare
2. [Spherical distortions] Common equirectangular projection of panoramas clearly distorts near poles.

Correspondingly, the authors addess this problems with two main contributions:
1. [Panoramic data curation engine] Considering the rich data in perspective depth estimation, this paper proposes a panoramic data curation engine. Firstly, the perspective RGB / depth images are transformed into spherical space via Perspective-to-Equirectangular (P2E) projection. Then, for the transformed RGB is panoramic out-painted to obtain the "full" panorama. This engine curates additional 540K data samples.
2. [SphereViT] A ViT network that explicitly incorporates spherical coordinates into latent feature via spherical embeddings, designed to mitigates the effect of spherical distortions.

To validate DA$^2$, the authors conduct a comprehensive comparison, including both panoramic depth estimation methods and perspective methods. The results clearly show the SOTA performance of DA$^2$.

### Strengths
1. Reasonable analysis of existing methods, and straightforward solutions targeting the existing limitations. The existing issues (Data scarcity, Spherical distortions) is critical in panoramic depth estimation. And the solutions (Panoramic data curation engine, SphereViT) effectively address the limitations, making this paper valuable in panoramic depth estimation.

2. In the experiments, the author built a comprehensive benchmark among various datasets, and compared DA$^2$ with both zero-shot / in-domain, panoramic / perspective approaches. The quantitative and qualitative results clearly show DA$^2$'s SOTA performance. The ablation studies clearly show the performance gained via scaling-up with perspective data, also the spherical embeddings in SphereViT.

3. The authors claimed that the code and the curated data will be open-sourced. This will be a valuable contribution to the research community.

4. The writing is clear and also the figures. The paper is easy to follow.

### Weaknesses
1. Performance concerns on real-world panoramas: The reviewer noticed that the curated panoramic data are basically come from synthetic perspective data. The domain gap between synthetic data and real-world data may decrease the model's performance on real-world panoramas.

2. While the data curation engine is effective, the depth label is largely missing, which is better obtained. The author should add more discussions on this matter.

3. More qualitative results: There are only 2 cases in this paper's qualitative comparison.

4. The reviewer also noticed left-right seams in the reconstructed 3D point clouds. The author should add more discussions about this problem, which can be critical in real-world applications.

5. Minors:

    (1). The arrow on the right figure of figure 2 is in opposite direction.

    (2). The citation of MVS-Synth in line 371, use \citep instead.

### Questions
Please refer to weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces DA2, a method for zero-shot, monocular panoramic depth estimation. The work tackles two fundamental challenges in this domain: the scarcity of large-scale panoramic training data and the geometric distortions inherent in the equirectangular projection.

To address data scarcity, the paper proposes a panoramic data curation engine. This engine leverages abundant perspective RGB-D datasets by first projecting a perspective image into a partial panorama and then using a image generative model (i.e. FLUX-I2P) to out-paint the partial panorama into a full one. This process increased their training set size by nearly tenfold, from ~63K to ~607K pairs.

To handle spherical distortions, the paper proposes SphereViT, a ViT-based architecture. Instead of using standard 2D positional embeddings, SphereViT creates a fixed Spherical Embedding derived from the spherical coordinates (azimuth and polar angles) of each image patch. It then uses cross-attention, with the image features as queries and the spherical embedding as keys and values, to explicitly inject distortion-awareness into the feature representations.

Through extensive experiments on standard benchmarks (Stanford2D3D, Matterport3D, PanoSUNCG), DA2 is shown to achieve state-of-the-art performance, significantly outperforming prior zero-shot methods and many specialized in-domain models with high inference efficiency.

### Strengths
The paper's primary strength is its two-fold innovation.

1) The data curation engine creatively uses a generative model (i.e. FLUX-I2P) to synthesize panoramic training data from abundant perspective data, the authors have effectively broken through the data barrier in the academic community that restricts a open source panoramic depth estimation comparable to commercial models. The resulting "scaling law" (Fig. 2) is a powerful demonstration of the data-centric approach.

2) The SphereViT architecture introduces a neat and elegant solution to the problem of spherical distortion. Its use of cross-attention with a fixed spherical embedding is a more principled and efficient approach than the fusion strategies common in prior work.

The resulting model, DA2, is not only accurate but also efficient. As a fully end-to-end method, it is significantly faster than fusion-based alternatives, making it practical for real-world applications, as mentioned in the appendix. The planned release of the code and the large-scale curated panoramic dataset is a big contribution in itself and will be a substantial asset to the research community.

The paper is easy to follow. The experimental validation is thorough, featuring a comprehensive benchmark against a large number of competing methods.

### Weaknesses
The paper is novel and effective, the following points are intended as constructive suggestions for further improvement.

1) About Out-painter

The quality of the curated data, and thus the performance of DA2, is intrinsically linked to the performance of the FLUX-I2P out-painter. Any biases or artifacts of the generative model could be implicitly learned by DA2. A brief discussion on the limitations of the out-painter, and how different out-painters affect depth estimation would add valuable context.

2) About SphereViT

The cross-attention mechanism in SphereViT is a key architectural novelty. The ablation study shows that including the spherical embedding is crucial. However, it does not compare the proposed cross-attention design against other plausible alternatives for incorporating this embedding. For instance, the standard ViT approach (adding the embedding to image features before self-attention) or simply concatenating the spherical coordinates as extra channels would provide stronger evidence that the specific cross-attention is optimal.

3) About Seam Artifact 

The paper acknowledges visible seams at the panorama's left-right boundary as a limitation. Given that SphereViT is designed with explicit knowledge of spherical coordinates, it is somewhat surprising that this issue persists. A brief discussion on why the current architecture does not fully resolve this and potential solutions would strengthen the paper.

### Questions
Thank you for this good work. Besides the points raised in the discussion of weakness, I have a few further questions to better understand the nuances of your contributions.

1) Current image generative methods are capable of out-painting depth information. If the perspective depth maps are also out-painted, would this lead to performance improvements?

2) As mentioned in the paper, the depth estimation model and data will be open-sourced. Will the FLUX-I2P data and model also be open-sourced? The paper does not provide sufficient details about the data used to train FLUX-I2P.

3) Please discuss the relevance to the method proposed in "S2Net: Accurate Panorama Depth Estimation on Spherical Surface". If relevant, please cite this work.

4) In Table 3, what would the results be if only Pano. out-painting is used without SphereViT and Normal loss?

5) In Section A.1, the authors mention that only translation is needed to align the 3D points from panoramas taken at different locations. Please explain why rotation is not needed?

6) Please proofread the manuscript again to identify and correct spelling errors—for example, "date" in Line 380.

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
This paper aims to create a universal, end-to-end depth estimation model that generalizes across domains and any directions for panoramic application.
Main contribution:
1. provide a synthetic panoramic data engine/pipeline mainly powered by utilizing FLUX-I2P out painting. And evaluated interest scaling law on that curated data.
2. Design a SphereViT which incorporates spherical embedding, can yield accurate geometrical estimation by distortion-aware image features.
3. Evaluation on zero-shot/in-domain benchmark shows strong quality.

### Strengths
1. Good work which represents a step toward universal depth perception across all directions.

2. It combines synthetic panoramic data and sphere-aware architecture could inspire like Geometry-consistent generative modeling.

3. Solid results on comprehensive panoramic benchmarks show good zero-shot performance, outperform in-domain models proved good generalizability.

### Weaknesses
1. The large-scale training set is created by FLUX I2P. This work tries to find a good way to utilize the strong FLUX base model capacity. However, the generation pipeline might have distribution gaps like lighting environment & geometry consistency may still have some gaps from real panoramic scenes.
2. Efficiency: this paper’s E2E pipeline is faster than fusion based method but remains at same level with other E2E methods like UniK3D. Both are 0.3s. The E2E pipeline seems not novel.
3. The data and ViT arch design are mainly focused on depth estimation quality improvement. Other methods like PanDa, DepthAnyCamera or UniK3D could also cover the tasks.

### Questions
1. What if we collected more real panoramic data (not limited in some domains) and use them to evaluate the scaling law? It should also have improvement for depth estimation task.
2. Can we also use video-generation model to generate some temporal panoramic data for temporal consistency? Single frame is easier to handle.
3. For the ablation study, spherical embedding didn’t show too much improvement (less than 1 point), is there any other better design for the specific utilization of spherical information?

### Soundness
3

### Presentation
4

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
This paper aims to build a strong zero-shot depth estimation model for panoramic images. The main contributions are:

- A data curation pipeline that maps perspective images into spherical coordinates followed by image out-painting, substantially expanding the training data scale.
- A customized transformer with angle-based positional embeddings and a cross-attention design.
- A comprehensive evaluation of zero-shot and in-domain depth estimation across multiple benchmarks.

### Strengths
1. The paper clearly identifies limitations in current panoramic depth estimation methods and significantly expands the data scale, which is crucial for the field’s progress.
2. Benefiting from the enlarged dataset, the model outperforms existing methods on standard benchmarks.
3. It provides a comprehensive benchmark, including perspective-based models, which clarifies the current landscape.
4. The paper is clearly written and contains rich technical details.

### Weaknesses
1. Limited ablations on the dataset curation process.
   - Because ground-truth depth is only available in the central regions of the curated data (due to the limited FOV of perspective images; Fig. 3), it is unclear how missing annotations at the borders affect final performance.
   - The model uses image out-painting to compensate for limited FOV, but the visual quality of the out-painted regions is not clearly evaluated.

2. Limited novelty in some architectural components.  
   The paper emphasizes angle-based positional embeddings (L269–299), cross-attention (L300–311), and a normal loss (Table 3, listed as a separate ablation item), but:
   - Positional embeddings: Improvements appear modest. In Table 3, removing $E_{\text{sphere}}$ yields only a small degradation (AbsRel 6.62 → 6.84) compared to gains from out-painting and the normal loss. It is also unclear whether the baseline here uses standard \(uv\) positional embeddings or no PE. The utility of the proposed PE seems smaller than claimed.
   - Cross-attention: The idea is not new and is under-ablated. Prior work (HUSH [1]) uses cross-attention to attend image features and SH features. Here, the method uses \(uv\) PE instead of SH and fixes image+feature as queries with PE as keys/values. Ablations should compare:  
     (a) Cross-attention vs. self-attention (with image+PE as inputs, matching L303), and  
     (b) Swapping queries/keys (PE vs. image features), following HUSH.  
   - Normal loss: It contributes notably (Table 3), but normal losses have been used previously (e.g., [1]).

### Questions
1. Center vs. border performance: Can you provide ablations comparing performance in central regions versus border regions on panoramic images, to clarify benefits and limitations of the proposed dataset (related to Weakness 1-1)?
2. Out-painting quality: Can you include visual quality comparisons (e.g., SSIM, LPIPS, PSNR) between the out-painted images and corresponding ground-truth images (Weakness 1-2)?
3. Attention ablations: Please provide ablations for self-attention as well as query/key-value choices in cross-attention (Weakness 2).
4. Ablation dataset: Which evaluation dataset is used for the ablation results? The reported scores do not seem to align with any dataset in Table 2.

---

Reference  
[1] HUSH: Holistic Panoramic 3D Scene Understanding using Spherical Harmonics, CVPR 2025.

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
DA2 trains an end-to-end panoramic depth (scale-invariant distance) model by first,  curating a large dataset containing panoramas from perspective RGB-D via P2E projection plus generative panoramic out-painting (FLUX-I2P), and develop a sphereViT backbone that cross-attends from image features to a fixed spherical angle embedding. On benchmarks, DA2 reports the best zero-shot results, beating strong zero-shot baselines like UniK3D.  Authors highlight better or comparable speed to end-to-end competitors while much faster than fusion pipelines.

### Strengths
1. Strong empirical gains. The proposed method demonstrates strong performance compared to both zero-shot fusion, zero-shot end-to-end and in-domain methods.
2. The proposed method scales well. The paper shows clear scaling-law-like improvements as more perspective data are "panoramaized"
3.  The proposed architecture is effective. The SphereVIT's fixed spherical embedding plus cross attention is a simple but effective way to inject spherical awareness, which shows strong results.

### Weaknesses
1. It seems there is a supervision mismatch introduced by the generative out-painting. The curation engine out paints RGB to a full panorama but supervised only the P2E covered part of depth (no out-painted depth due to acc concern). Hence, many pixels in the training panoramas lack ground-truth depth, yet their RGB content is model-generated. This opens risks of learned correlation with out-painter priors and biased geometry in unsupervised regions (especailly near poles and seams).  The paper shows gains with out-painting but it doesn't quantify how much of the test-time improvement stems from distribution mathcin got FLUX-I2P artifacts compared with real geometric learning. 

2. Injecting coordinates/angles and using attention to a fixed positional bank is not conceptually new. [1] mitigates ERP distortion via spherical tangent tokens and transformer design. [2] operates directly on spherical meshes, and several recent spherical ViTs or positional-encoding approaches aim at similar goals. 

3. Metric depth is not addressed when comparing to Unik3D. DA2 predicts scale-invariant distance, not metric depth. With Unik3D showing metric 3d across cameras (including panoramic), it is important to discuss scale recovery or to provide an optional scale head, since it is important for downstream applications like AR or robotics.


[1] Shen, Zhijie, et al. "PanoFormer: panorama transformer for indoor 360∘ depth estimation." European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2022.
[2] Yan, Qingsong, et al. "Spheredepth: Panorama depth estimation from spherical domain." 2022 International Conference on 3D Vision (3DV). IEEE, 2022.

### Questions
1. What fraction of pixels per training panorama actually have GT depth supervision, on average and by latitude?

2. Can we consider add a simple scale-head (or post-hoc scale regressor) and evaluating metric errors where ground truth exists. Compare to UniK3D on the same splits.

### Soundness
3

### Presentation
3

### Contribution
3
