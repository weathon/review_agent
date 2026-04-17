# ArtUV: Artist-style UV Unwrapping

- Decision: Accept (Poster)
- Scores: 8, 4, 6, 4

## Abstract
UV unwrapping is an essential task in computer graphics, enabling various visual editing operations in rendering pipelines. However, existing UV unwrapping methods struggle with time-consuming, fragmentation, lack of semanticity, and irregular UV islands, limiting their practical use. An artist-style UV map must not only satisfy fundamental criteria, such as overlap-free mapping and minimal distortion, but also uphold higher-level standards, including clean boundaries, efficient space utilization, and semantic coherence.
We introduce ArtUV, a fully automated, end-to-end method for generating artist-style UV unwrapping. We simulates the professional UV mapping process by dividing it into two stages: surface seam prediction and artist-style UV parameterization. In the seam prediction stage, SeamGPT is used to generate semantically meaningful cutting seams. Then, in the parameterization stage, a rough UV obtained from an optimization-based method, along with the mesh, is fed into an Auto-Encoder, which refines it into an artist-style UV map. Our method ensures semantic consistency and preserves topological structure, making the UV map ready for 2D editing. We evaluate ArtUV across multiple benchmarks and show that it serves as a versatile solution, functioning seamlessly as either a plug-in for professional rendering tools or as a standalone system for rapid, high-quality UV generation.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The authors address the challenge of artist-level UV unwrapping. To achieve this, they begin by reimplementing SeamGPT to partition the mesh into distinct, semantically meaningful islands. An initial UV unwrapping is performed using the Ministretch unwrap method. However, the results of this step often exhibit distortions in UV space. Skilled artists carefully arrange UVs “neater”. To address this, they propose learning offsets that distort the initial unwrapping to suit the preferences of artists. The process starts with the embedding of features on each vertex and the subsequent execution of a graph neural network. Subsequently, a PyramidED network is employed to enable global vertex interaction. The outcome is then 2D offsets that are applied to the initial unwrapping. Two novel losses are introduced for training this offset generation. Lastly, the individual islands are consolidated into a single atlas using a pre-existing method.

### Strengths
- The dataset release, along with the accompanying processing scripts, are highly valuable for future research.
- The introduction of the paper is well-written, particularly the section on the various approaches to UV unwrapping.
- Overall, the paper’s structure is clear and easy to follow.
- The two novel losses are well explained and motivated. They are also appropriately evaluated and fulfill their intended objectives as shown in ablations.
- The results presented are impressive, especially considering that the Artist-Level rating is surpassed by ArtUV even when compared to manual UV unwrapping.
- ArtUV has also a rather short run-time, particularly when applied to lower poly meshes. As mentioned in the Weakness section, SeamGPT may pose a potential limitation. However, the method remains applicable if more efficient cutting networks are discovered.

### Weaknesses
- Some weaknesses of the method are inherited from SeamGPT. Especially, that it requires significant amounts of memory and according to the original authors has a limit of roughly 20k vertices.

### Questions
- Why are Nuvo and FAM run in single chart mode? At least performing the Multi-Chart mode would result in slightly better results. Albeit not artist level.

### Soundness
4

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
2

### Summary
The paper presents a method to automatically perform UV unwrapping combining multiple state of the art methods. In particular, they use SeamGPT to infer cutting seams and then use them to run an optimisation procedure aimed at providing an initial UV map. Initial UV map and mesh info are then processed by a neural network using adaptive dimension mapping to modulate the importance of the different inputs (Res-M MLP), graph convolutions to locally fuse information (SAGE Conv) and an attention-based module (Pyramid ED) to faclitate global communication. This paper also introduces a curation of high-quality textured objects sourced from existing mesh datasets.

### Strengths
The paper presents an automated method for UV unwrapping which appears to surpass many state of the art methods. The pre-trained model and dataset could be of value for the community as well as for 3D artists. The paper is grammatically well written, but not entirely scientifically sound (see Weaknesses and Questions).

### Weaknesses
\- As later detailed, many citations are missing. Authors make extensive use of graphics software (e.g. Blender, Maya, 3DsMax), but fail to investigate the underlying techniques. This confers the paper an overall tone of superficiality and limited knowledge of the field. 

\- Technical novelty seems to be limited for the venue or authors fail to successfully highlight their core innovation. The main contribution appears to be the arbitrary combination of existing methods. While this could be an acceptable contribution the choice of the different "building-blocks" appears arbitrary. For instance, there is no written nor proven justification as to why SAGE Conv or Pyramid ED should be preferable to the many potential alternatives. An ablation of the model components and some benchmarks against potential alternatives would have been appreciated.

\- Mathematical notation is inconsistent and not rigorous. 

Some of these are further detailed in the questions section.

### Questions
After reviewing the paper I have several outstanding doubts and questions, which are listed below alongside a list of suggestions to improve the paper:



1\. I believe more context is needed in the introduction, especially for a venue whose main focus is not Graphics or 3D vision. Readers and participants are not necessarily familiar with the concept of UV mapping. I would suggest providing more details on what textures are and what they are used for. It would be also more comprehensive if the authors could mention methods that aim to skip this step and represent textures directly on the surface of objects (e.g., i, ii, iii, iv).



2\. Why should larger islands destabilise training?



3\. At line 161 authors claim that they manually selected UV islands. Were authors able to manually inspect 200k islands? Reading lines 162-163 I am under the impression that the selection was not manual, but mostly automated. 



4\. By using ministretch to assess the quality of UV textures during data curation, authors implicitly assume that this automatic procedure is capable of producing satisfactory UV maps. How can a presumably flawed method be used as rejection criteria?



5\. Authors state that "extensive empirical evidence" proved that directly learning the mapping from 3D is challenging (line 215). Where is this evidence? Please, cite relevant literature or report personal "extensive evidence".



6\. At line 217 authors state that a simple projection is not their goal and imply that this is a consequence of choosing meshes as input. I suppose that this was not the intended claim and I would encourage them to restructure the sentence as input meshes don't automatically translate into simple projections. 



7\. Please adopt a more rigorous notation. For instance:

&nbsp;  - line 221, I\_M is a set, as such it would be more appropriate to define the following: I\_M = {V, F, ...D}, I = I\_m U Q\_i. 

&nbsp;  - \\bar{p} and \\bar{q} were never formally defined. 

&nbsp;  - notation in Eq. 5 is inconsistent with Fig. 4

&nbsp;  - the rotation matrix computed in Eq. 4 should probably be part of Eq. 5 and 6



8\. Are Q and V sharing the same faces? Cutting seams should introduce vertex duplications to unwrap the surface. It is unclear to me which topological information is used and fed to the networks. Also, how is Res-M MLP processing faces and edges?



9\. Which specific technique was used for adaptive dimension mapping in the Res-M MLP? Briefly describe how it works. 



10\. Comparisons against Blender, Maya and 3DsMax are conceptually reasonable, but a better understanding of the underlying techniques is necessary. What these software implement today may differ from what they are going to use tomorrow. In addition, results for Maya and 3DsMax are rather similar. The main difference appears to be how islands are placed in the image rather than their shapes. It is very likely that they implement the same underlying method considering that they are both produced by Autodesk. The different island placement could be just a matter of some randomness in their procedure or hyperparameter choice.



11\. Also the following would benefit from additional details in terms of the inner workings and ideally be introduced with a one/two plain English sentences describing how they operate:

&nbsp;  - Blender ministretch (first occurrence at line 161), 

&nbsp;  - SSIM (line 162)

&nbsp;  - Conformal energy (line 376), please provide more details, there is a very extensive literature on conformal mapping.

&nbsp;  - UVPackMaster plugin (line 377)



12\. XAtlas appears to achieve lower distortion results in the qualitative results depicted in Fig. 7. It would be great to see more examples and better understand the distortion metric used to determine whether or not it can be considered reliable. 





Please note that given the weaknesses highlighted in the weaknesses section I am unlikely to increase my score above the acceptance threshold. I believe that too many details were omitted in the initial submission and even a good rebuttal may not be sufficient to ensure all the problems are properly addressed. Limited technical novelty and poor justification of methodological choices are my second major concern.





References:

\[i] Yu, X., Dai, P., Li, W., Ma, L., Liu, Z., \& Qi, X. (2023). Texture generation on 3d meshes with point-uv diffusion. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 4206-4216).

\[iii\\] Wei, J., Wang, H., Feng, J., Lin, G., \& Yap, K. H. (2023). Taps3d: Text-guided 3d textured shape generation from pseudo supervision. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 16805-16815).

\[iii] Foti, S., Zafeiriou, S., \& Birdal, T. (2024). Uv-free texture generation with denoising and geodesic heat diffusion. Advances in Neural Information Processing Systems, 37, 128053-128081.

\[iv] Oechsle, M., Mescheder, L., Niemeyer, M., Strauss, T., \& Geiger, A. (2019). Texture fields: Learning texture representations in function space. In Proceedings of the IEEE/CVF international conference on computer vision (pp. 4531-4540).

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies UV map refinement. The authors introduce a neural network (ArtUV) that learns to predict per-vertex offsets of an UV map using a weighted multi-term loss function. By adding the predicted offsets to the vertices, it moves the islands of an UV map into an artist style layout. The main strength of this paper is that it addresses a valuable practical problem in the 3D graphics industry. The proposed method makes a UV map easier for artists to manipulate and increases space utilization.The main weaknesses are 1) lack of academic novelty and 2) over-claimed contributions. Overall, I would rate this paper as Weak Accept (WA). Although its contribution lies in system-level innovation rather than academic novelty, it provides a practical solution for industry applications.

### Strengths
1.This paper is clearly written.

2.The proposed approach is valuable for applications.

### Weaknesses
1.Overstated contribution. The paper claims to present an end-to-end system, yet key components—SeamGPT for seam prediction and Ministretch-Unwrap for initialization—are borrowed from prior works. In fact, UV map refinement is a valuable problem in its own right; 

2.There is no need to bring SeamGPT here merely to inflate the contribution. Instead, provide an analysis of the core challenges in UV-map refinement, and explain how your network design (e.g., multiple inputs, SAGEConv, etc.) addresses each, would clarify the true contribution and strengthen the paper.

### Questions
1.Does UV map’s resolution affect performance? Please find three AI-GENERATED 3D models with simple, normal, complex geometry structure, and run your pipeline using UV resolutions of 256×256, 512×512, and 1024×1024.

2.Please investigate if ArtUV improves rendering quality.  Please find three AI-GENERATED 3D models with simple, normal, complex geometry structures, render them using Init Map and ArtUV’s output. Check if the rendered images’ quality is improved by ArtUV.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes ArtUV, a novel method for UV unwrapping that aims to generate artist-style UV maps. UV unwrapping is critical for tasks such as texture mapping and lightmap generation. The proposed method divides the process into two stages: surface seam prediction using SeamGPT, and UV parameterization using a specialized Auto-Encoder. ArtUV is designed to address the shortcomings of existing methods, such as fragmentation, lack of semantic coherence, and high computational demands.

### Strengths
- Structure looks interesting. ArtUV introduces a hybrid approach by combining SeamGPT for surface seam prediction and an Auto-Encoder for UV parameterization. This combination seems promising for addressing both the technical and artistic challenges of UV unwrapping.

- The method demonstrates significant improvements in UV utilization and distortion minimization when compared to professional software such as Blender, Maya, and 3DsMax, as shown in the experiments and visual results.

- The paper provides extensive experimental validation using both qualitative and quantitative measures, demonstrating the superiority of the method in terms of distortion, utilization, and runtime across various benchmarks

### Weaknesses
- The proposed method seems to work only on manifold meshes. However, artist-style meshes often go with arbitrary design patterns. The method may not generalize well to a wide range of 3D object types, particularly those that require non-traditional unwrapping methods.

- While SeamGPT provides semantic cutting predictions, it requires careful training and may struggle with particularly challenging meshes that deviate from typical models. The method could potentially fail for meshes that have unusual topologies or textures.

- The method is highly sensitive to the initial quality of surface cutting (seam prediction). Inaccurate or incomplete seams can lead to significant distortions during the UV initialization process, which may degrade the final result, especially for complex models

### Questions
- How does ArtUV perform when applied to non-standard or highly irregular meshes (e.g., organic models with non-manifold edges)? Are there specific types of models or topologies where the method might fail?

- While the method shows good results in benchmarks, how well does it scale when integrated into large-scale production pipelines with diverse and complex 3D assets? Could performance degrade with larger models or meshes with higher vertex counts?

- Given that SeamGPT’s accuracy is critical for ArtUV’s performance, how robust is it to small errors in seam prediction? Could slight inaccuracies in cutting lead to major issues in the final UV map, especially in intricate areas like facial features or other highly detailed parts?

- How does the method handle texture mapping for more complex or varied textures (e.g., procedural textures or dynamically generated ones)? Does it maintain its effectiveness for non-static textures?

- Please provide more justifications on how the proposed method generalizes to in-the-wild cases. This would help assess its ability to generalize across various 3D assets.

### Soundness
2

### Presentation
3

### Contribution
2
