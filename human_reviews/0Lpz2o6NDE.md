# Tex4D: Zero-shot 4D Scene Texturing with Video Diffusion Models

- Avg Score: 5.00
- Decision: Reject
- Scores: 5, 5, 5, 5

## Abstract
3D meshes are widely used in computer vision and graphics because of their efficiency in animation and minimal memory footprint. They are extensively employed in movies, games, AR, and VR, leading to the creation of a vast number of mesh sequences. However, creating temporally consistent and realistic textures for these mesh sequences remains labor-intensive for professional artists. On the other hand, video diffusion models have demonstrated remarkable capabilities in text-driven video generation, enabling users to create countless video clips based solely on their imagination. Despite their strengths, these models often lack 3D geometry awareness and struggle with achieving multi-view consistent texturing for 3D mesh sequences. In this work, we present Tex4D, a zero-shot approach that integrates inherent 3D geometry knowledge from mesh sequences with the expressiveness of video diffusion models to produce multi-view and temporally consistent 4D textures. Given an untextured mesh sequence and a text prompt as inputs, our method enhances multi-view consistency by synchronizing the diffusion process across different views through latent aggregation in the UV space. To ensure temporal consistency, we leverage prior knowledge from a conditional video generation model for texture synthesis. However, straightforwardly combining the video diffusion model and the UV texture aggregation leads to blurry results. We analyze the underlying causes and propose a simple yet effective modification to the DDIM sampling process to address this issue. Additionally, we introduce a reference latent texture to strengthen the correlation between frames during the denoising process. To the best of our knowledge, Tex4D is the first method specifically designed for 4D scene texturing. Extensive experiments demonstrate its superiority in producing multi-view and multi-frame consistent videos based on untextured mesh sequences.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
# Summary

This paper focuses on creating temporally consistent and realistic textures for mesh sequences. The input is an untextured mesh sequence and a text prompt. To achieve this, a method named Tex4D is proposed, which is a zero-shot approach that integrates geometry information from mesh sequences with video diffusion models, specifically the depth-conditioned CTRL-Adapter (Lin et al., 2024), to produce multi-view and temporally consistent 4D textures. The model synchronizes the diffusion process across different views through latent aggregation in the UV space. Additionally, a reference latent texture is introduced to strengthen the correlation between frames during the denoising process.

### Strengths
# Strengths

- The paper is well-written, making it easy to understand and follow.
- The related works are sufficiently covered.
- Several experiments are conducted for demonstrating its effectiveness.

### Weaknesses
Regarding the contribution and motivation, I still wonder about the specific meaning of "4D texturing" for the object and how this differs from a 3D model that is first textured and then animated using skinning techniques. Even for dynamic textures, one could also generate the dynamic texture for a 3D model and then animate the character through skinning. This approach seems useful **if** the mesh is also significantly dynamic, such as with topology changes. Further, when it comes to the video background, I noticed in the supplementary video that there are some dynamic effects from the proposed method, but they are not that significant. Could one first perform 3D texturing and then render it with a generative background video?

I think the following suggestions could be helpful in justifying the significance of the setting in this paper.
- Provide specific examples or use cases where 4D texturing offers advantages over traditional 3D texturing and animation.
- Clarify how the proposed method handles dynamic meshes with topology changes, if applicable.
- Compare the method with a pipeline of 3D texturing followed by rendering with a generative background video, highlighting any benefits of the presented "integrated" approach.

I also have concerns about the novelty of the paper. The entire pipeline can be seen as a depth-conditioned CTRL-Adapter for mesh sequence texturing with UV space aggregation, which feels like a straightforward composition of existing models. I would prefer to see a simple yet effective method of tackling a critical problem. However, as I am still uncertain about the meaning/significance of "4D texture," this makes me somewhat skeptical about the proposed pipeline.

I think the authors could provide some arguments for the novelty of the proposed method:
- Highlight the key technical innovations in the pipeline beyond the composition of existing models.
- Explain how the introduced method addresses specific challenges in 4D texturing that are not easily solved by existing methods.
- Provide a clearer definition and motivation for "4D texturing" to help readers understand its significance in the context of their work (similar to the previous questions).

I understand the difficulty in evaluating the results, but it would be helpful and necessary to conduct an evaluation of Appearance Quality, Spatio-temporal Consistency, and Consistency with Prompt via a user study - The quantitative evaluation is insufficient.

The following action can be taken:
- Conduct a user study evaluating the specific aspects, e.g., Appearance Quality, Spatio-temporal Consistency, and Consistency with Prompt, and compare the proposed method with previous models.

Limitations and Future Works should be included. 
- For instance, the authors may discuss the current limitations of their approach, such as some failure cases or more specifically, the types of meshes or textures it struggles with, etc.
- Potential future improvements or extensions to the method.
- Broader implications or applications of this work in related fields.


# Minor Comments

- "To resolve this issue, we analyze the underlying causes and propose a simple yet effective modification to the DDIM (Song et al., 2020) sampling process." In the introduction section, it would be beneficial to briefly explain how you achieved this.

### Questions
Q: "While these methods produce multi-view consistent textures for static 3D objects, they do not address the challenge of generating temporally consistent textures for mesh sequences." I would appreciate further clarification on the motivation behind "generating temporally consistent textures" for mesh sequences. Could you provide examples where dynamic 4D texturing is essential and cannot be achieved through traditional methods?

Q: How does the model ensure robustness when dealing with varying UV mapping results?
- How sensitive is the method to different UV unwrapping techniques?
- Did the authors experiment with different UV mapping strategies, and if so, what were the results?
- Are there any limitations or best practices for UV mapping when using this method?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces 4D scene texturing to generate textures that are consistent both temporally and across multiple views for animated mesh sequences. Tex4D, uses 3D geometry to synchronize diffusion processes, incorporates video generation model insights for temporal consistency, and modifies the DDIM sampling process to improve texture clarity.

### Strengths
The paper is the first work to perform video generation based on animated mesh sequences, while its UV mapping strategy ensures multi-view consistency. The experimental results show significant advantages compared to some existing works.

### Weaknesses
Under the current pipeline, this work has yielded highly effective results. However, the importance of this pipeline should be further clarified, such as by comparing it with pipelines based on 2D poses or textured meshes. The paper should include more comprehensive comparisons to highlight the contribution of the pipeline. For example, is it a reasonable pipeline to first generate textured meshes and then use animated meshes for video generation?

### Questions
Overall, the experimental results are quite satisfactory; however, there is a lack of explanation regarding the advantages of the pipeline compared to other pipelines using 2D poses and textured meshes.

Animation can drive the mesh. Are the position and rotation of the mesh manually controlled, such as in the second example on the first page?

How is the animated mesh obtained? We observe some temporal changes in Figure 5. Is this one of the contributions of your paper? How do you distinguish between temporal changes and temporal inconsistencies, as there are some temporal inconsistencies in your results?

Would using textured meshes yield better outcomes? What are the advantages of generating videos with untextured meshes compared to textured meshes?

What is the difference between using a 2D pose and an animated mesh?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper proposed a 4D scene texturing approach by video diffusion models, a zero-shot pipeline for generating temporally and multi-view consistent textures. In order to aggregate multiview latents in UV space, they discovered the issue of "variance shift" caused by their aggregation, and proposed to modify DDIM sampling process to address the issue. By UV blending during denoising steps, the issue of self-occlusion is addressed and synchronized in invisible regions.

### Strengths
1. The authors assert that this is the first method developed specifically for 4D scene texturing.
2. The authors introduce a multi-frame consistent texture generation technique, demonstrating improved consistency in results compared to baseline methods.
3. The paper is fluent and well-written, contributing to its readability and overall clarity.

### Weaknesses
1. The generated textures do not blend seamlessly with the background, creating a disjointed appearance that resembles separate foreground and background elements stitched together.
2. Despite claims of multi-view consistency, flickering effects are observed across different views, indicating instability in rendering.
3. Some of the compared methods, such as TokenFlow and Text2Video-Zero, do not utilize mesh or depth inputs, making direct comparisons less equitable.

### Questions
1. In the paper, the authors mentioned that the mesh texture could be significantly influenced by the background, providing an example with a white background. I’m curious how the generated texture might look if a non-white background was used, especially one that contrasts strongly with the foreground object. How would such a background affect the consistency and quality of the generated texture?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces a novel framework for generating textures for mesh sequences. The authors utilize a depth-conditioned video diffusion model to ensure temporal consistency in videos generated from rendered mesh sequences for each predefined viewpoint. To achieve multi-view consistency, they adopt a UV space texture aggregation strategy. Additionally, they propose a modified sampling approach to address the issue of blurriness in the generated textures.

### Strengths
See below.

### Weaknesses
See below.

### Questions
Although the experiments provide some evidence of the proposed method’s effectiveness, several concerns remain:

1. Could the authors provide additional qualitative and quantitative comparisons in the ablation study? With only one demonstration, it is difficult to convincingly assess the effectiveness of the proposed method.

2. The authors suggest that video diffusion models struggle with multi-view consistent texturing for 3D mesh sequences due to a lack of 3D geometry awareness. However, the approach already uses a depth-aware video diffusion model, which inherently includes some geometric awareness. Why does this straightforward combination not achieve the desired consistency? Does this imply that depth-aware video diffusion models alone cannot guarantee multi-view consistency even with depth information? If so, could the authors provide performance metrics or visual comparisons showing results when using only the depth-conditioned video diffusion model as a prior? Additionally, for a single viewpoint, does the video diffusion model produce temporally consistent results? If not, visual examples would help clarify.

3. Since a mesh input is available, a straightforward approach could be to texture the mesh on the first frame using methods like Text2Tex or SceneTex, then animate the textured mesh. This method might improve efficiency and naturally maintain multi-view consistency across frames. How does this alternative approach compare in terms of both methodology and performance? An in-depth discussion of these differences would be beneficial.

4. The authors mention that for each predefined viewpoint, a sequence of K rendered meshes is used as input and individually textured by the depth-guided diffusion model. Could the authors clarify the motivation behind this setup? Since the videos are generated separately for each view, multi-view inconsistencies are expected. Why introduce this setup if it inherently leads to consistency issues at the start?

5. While using UV textures for each mesh can enhance multi-view consistency, this approach seems more like an averaging of multiple viewpoints to produce a smoother result. Can the authors elaborate on how this averaging mechanism ensures true multi-view consistency?

6. Given that the current method requires rendering V views for each mesh in the sequence, which may be computationally intensive, could the authors discuss the efficiency of the method? Details on the time required to process a sample would help assess the method's practicality.

7. It would be beneficial to include video visualizations or comparative examples to further illustrate the method's performance and effectiveness.

### Soundness
3

### Presentation
2

### Contribution
2
