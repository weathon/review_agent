# LumiTex: Towards High-Fidelity PBR Texture Generation with Illumination Context

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 8, 8, 2, 2

## Abstract
Physically-based rendering (PBR) provides a principled standard for realistic material–lighting interactions in computer graphics. Despite recent advances in generating PBR textures, existing methods fail to address two fundamental challenges: 1) materials decomposition from image prompts under limited illumination cues, and 2) seamless and view-consistent texture completion. To this end, we propose LumiTex, an end-to-end framework that comprises three key components: (1) a multi-branch generation scheme that disentangles albedo and metallic–roughness under shared illumination priors for robust material understanding, (2) a lighting-aware material attention mechanism that injects illumination context into the decoding process for physically grounded generation of albedo, metallic, and roughness maps, and (3) a geometry-guided inpainting module based on a large view synthesis model that enriches texture coverage and ensures seamless, view-consistent UV completion. Extensive experiments demonstrate that LumiTex achieves state-of-the-art performance in texture quality, surpassing both existing open-source and commercial methods. Project page: [https://lumitex.vercel.app](https://lumitex.vercel.app).

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces LumiTex, a PBR texture generation model that achieves SoTA quality for PBR texture painting for 3D assets. LumiTex strengthens the connection between illumination shading and PBR material decomposition in the context of 3D generative models. The texture completion via LVSM has also proved to be very effective.

### Strengths
* The integration of illumination context implicitly through attention layers is a wise design. Additional illumination prior can effectively reduce shading ambiguity in PBR tasks.
* A joint model for multi-view, multi-channel PBR materials is a good attempt. The results demonstrate its effectiveness in reducing accumulation errors compared to prior multi-stage approaches.
* LVSM for texture completion performs very well in ensuring global consistency and seamlessness of the final texture maps.

### Weaknesses
I don’t find any major weakness point. Below are a few minor weaknesses points. 
* It is unknown how MR is represented and decoded. Is MR in the orm convention or modeled separately? Is MR latents decoded with the same VAE used for regular RGB images?
* For the LVSM texture inpainting part, authors seem not to mention how decomposed PBR textures get inpainted. The example (Fig. 4) is on the rendered images.

### Questions
The LumiTex DiT is a native multi-view model, implying it should be capable of performing texture inpainting for additional views. I am curious why the authors opted for LVSM instead of leveraging the LumiTex DiT for this task and would appreciate some insights into this decision.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
Authors propose an end-to-end framework that, given a reference image and a mesh, generates textures with PBR materials.

Main contributions:
- Multi-branch design (one for generation of shaded images - to capture illumination context).
- 2-stage training of multi-view illumination context branch and material branch
- Lightning-aware material attention mechanism (directly attending to shaded tokens instead of using explicit intermediate images or optimization techniques)

### Strengths
To my knowledge, this is the first approach that is able to generate close-to-true PBR materials, without noticeable baked reflections or highlights. Given strong quantitative and qualitative evaluation, and the importance and complexity of the texturing task, I consider this work to be significant to the field.

The main contributions and strengths of the paper are clearly demonstrated and ablated in section 4.5, namely:
- Separate branch for shaded images prediction
- Single-stage generation (no use of explicit intermediate shaded images)
- (!) Multi-branch design (instead of multi-channel generation of albedo and MR)
This was very insightful to learn, and well-supported by visuals in Fig 9b.

It was also demonstrated that the pipeline generalizes well to real-world scenes which is helpful in practical applications.

### Weaknesses
- Impact and novelty of geometry-guided inpainting module is limited, although this is not claimed as a main contribution of the paper.

### Questions
1. Please clarify the novelty of the inpainting module. Is the main contribution that you added geometry guidance to LVSM?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a method to generate PBR textures for given objects. The core idea are two stages: from an input mesh + reference image, the goal is to generate N view-consistent PBR images. Here, the core idea is essentially a multi-view image generator with several conditions (they authors refer to this as an illumination-consistent base model); this is then frozen and a material branch is trained for the PBR part. The second stage is a geometry-guided LVSM to synthesize more viewpoints. This is a texture in-painting strategy based on LVSM which generates the extended views, hence, more complete textures. Training is done from 92K objects from Objaverse and Objaverse-XL – for each object, 30 views are rendered (albedo, metallic, roughness, and HDR images). The base model is FLUX.1-dev.

However, to be honest, the main technical exposition is quite confusing and it’s not easy to follow the exact details of the multi-view PBR generator (more details below).

### Strengths
- The authors tackle an important problem.

- The renderings of the shaded outputs look nice.

- I appreciate the re-lighting results in the video.

### Weaknesses
The presentation is confusing, and I'm having trouble understanding several of the technical details:

- The introduction mostly pitches the features but a coherent description of the core idea; e.g., how does the multi-view shaded image generator work is somewhat omitted – this makes it hard to read (e.g., first need to read the whole main section and even some of the results to understand which base models they were using)

- Fig 3 is a pipeline but the description of the multi-view illumination-consistent base model is missing, and the pipeline flow is confusing (e.g., where does the input mesh go, where do the reference images come into play).

- I’m confused about the term multi-modal DIT. Seems all input here are images… what are the modes you are referring to here?

- In the video, while the re-lighting looks great, I would’ve loved to see the actual PBR materials rather than the shaded versions. Also the shading in the video seems exhibit some temporal unstable artifacts which is confusing given that the underlying rendering should be just a mesh + PBR texture = this should be visualized.

- The main results are in Fig 6 in the main paper (the majority of visuals does not show the actual PBR results but the shaded versions). Unfortunately, this looks not that impressive. E.g., I would’ve loved to see the PBR textures of the objects in Fig 1 or Fig 5 instead of the shaded outputs.

There is a general confusing claim to  PBR textures but then consider environment lighting which would still mean that scene specific context is baked in. This is contrary to the definition of a PBR texture.

Figure 2 albedo looks poor. These results look shaded as there is lighting baked in – this is unfortunately not a PBR image / texture. Am I missing something here?

### Questions
See above in the weakness sections.

### Soundness
1

### Presentation
1

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
The paper proposes LumiTex, a PBR texture generation framework that addresses challenges like material decomposition with limited illumination cues and seamless texture completion. LumiTex combines a multi-branch generation scheme, a lighting-aware material attention mechanism, and a geometry-guided inpainting model to enhance texture quality, realism, and consistency across views. Extensive evaluations are delivered.

### Strengths
- First of all, the work proposes the multi-branch generation design and the lighting-aware attention mechanism offers a novel way of disentangling albedo and metallic-roughness (MR) while integrating illumination context. 


- Quantitative results (FID, CMMD, LPIPS) and qualitative evaluations indicate that LumiTex achieves competitive or superior performance compared to existing methods, particularly in terms of texture quality and relighting fidelity.

- The authors conducted a wide range of experiments, including comparisons to both open-source and commercial systems, as well as a user study on texture quality, demonstrating LumiTex’s practical advantages in real-world applications.

### Weaknesses
- While the framework employs multi-branch generation and illumination context, similar ideas have already been explored in other recent works. For example, the idea of using lighting priors for material generation is not very novel. The originality of LumiTex comes into question because the combination of multi-view consistency and lighting-guided material attention don't significantly advance the state of the art in a groundbreaking way.

- The method is computationally intensive and limited to generating textures at 768×768 resolution, which severely restricts its application in high-end industries that require 4K or 8K resolution for detailed textures (IMO this is more useful for texture generation for gaming applications or AAA filming). Although the authors propose potential avenues for scaling (e.g., multi-resolution models), scalability remains an unresolved bottleneck. Real-time applications (such as interactive design in AR/VR) would be severely constrained by the current training time of 106 GPU days. The paper does not adequately explore solutions to this scalability issue.


- The lack of support for transparent materials. Transparent materials, such as glass, water, and liquids, are commonly required in real-world 3D rendering, especially in architectural visualization and product design. The authors don’t provide a solid argument for why these materials are left out or when this limitation might be addressed. There already exist many works that investigate transparent image / video generation.

- LumiTex doesn’t show sufficient results in handling complex reflective materials or subsurface scattering. These properties are common in materials such as skin, water, and polished metals, which are common in visual effects and games. 

- The generalization to novel inputs seems to be limited. While the paper demonstrates robustness with real-world scanned meshes, the model still relies heavily on the type of training data (from Objaverse and Objaverse-XL). The authors did not provide convincing results showing how well LumiTex generalizes to extremely diverse or out-of-distribution inputs. This is a key issue for any system claiming real-world applicability in fields such as gaming or film, where content varies vastly.

- The paper presents failure cases (such as small printed text or transparent materials), but it does not delve deeply into why these failures occur or how they might be addressed in future work. For instance, the lack of alpha channel modeling for transparent materials is acknowledged, but the paper does not offer a roadmap for incorporating transparency modeling or refraction effects, making it seem like a major limitation without any plans for resolution.

- Despite claims of robustness under various illuminations, the framework’s real-world reliability under extreme lighting scenarios (e.g., highly reflective surfaces, strong backlighting) is not well-documented. The existing results are focused on typical lighting conditions, and there is a risk that the model might fail in edge cases involving extreme or uncontrolled lighting, common in cinematic and interactive applications. Without those, the contribution of this paper is doubtful.

### Questions
I would be grateful if the authors could address the above mentioned issues in weakness section and other reviewers' concerns.

### Soundness
3

### Presentation
3

### Contribution
2
