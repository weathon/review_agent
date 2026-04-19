# AvatarGO: Zero-shot 4D Human-Object Interaction Generation and Animation

- Decision: Accept (Poster)
- Scores: 5, 8, 3, 6

## Abstract
Recent advancements in diffusion models have led to significant improvements in the generation and animation of 4D full-body human-object interactions (HOI). Nevertheless, existing methods primarily focus on SMPL-based motion generation, which is limited by the scarcity of realistic large-scale interaction data. This constraint affects their ability to create everyday HOI scenes. This paper addresses this challenge using a zero-shot approach with a pre-trained diffusion model. Despite this potential, achieving our goals is difficult due to the diffusion model's lack of understanding of ''where'' and ''how'' objects interact with the human body. To tackle these issues, we introduce **AvatarGO**, a novel framework designed to generate animatable 4D HOI scenes directly from textual inputs. Specifically, **1)** for the ''where'' challenge, we propose **LLM-guided contact retargeting**, which employs Lang-SAM to identify the contact body part from text prompts, ensuring precise representation of human-object spatial relations. **2)** For the ''how'' challenge, we introduce **correspondence-aware motion optimization** that constructs motion fields for both human and object models using the linear blend skinning function from SMPL-X.  Our framework not only generates coherent compositional motions, but also exhibits greater robustness in handling penetration issues. Extensive experiments with existing methods validate AvatarGO's superior generation and animation capabilities on a variety of human-object pairs and diverse poses. As the first attempt to synthesize 4D avatars with object interactions, we hope AvatarGO could open new doors for human-centric 4D content creation.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper introduces AvatarGO, a novel framework for generating realistic and animatable 4D human-object interaction (HOI) scenes directly from textual descriptions. Addressing the challenges of determining "where" and "how" objects interact with human bodies, the authors propose two key innovations:

1. **LLM-Guided Contact Retargeting**: Utilizing Lang-SAM to identify contact body parts from text prompts, ensuring precise human-object spatial relations.
2. **Correspondence-Aware Motion Optimization**: Constructing motion fields for both human and object models using the linear blend skinning (LBS) function from SMPL-X to maintain correspondence during animation, reducing penetration issues.

The proposed method operates in a zero-shot manner using pre-trained diffusion models, overcoming the scarcity of realistic large-scale interaction data. Extensive experiments demonstrate that AvatarGO outperforms existing methods in generating coherent compositional motions and handling various human-object pairs and poses.

### Strengths
- The paper tackles the challenging and less-explored problem of generating visually realistic and animated 4D human-object interactions directly from textual inputs, which is crucial for applications in AR/VR and gaming.
- To the best of my knowledge, using LLM-Guided Contact Retargeting and Correspondence-Aware Motion Optimization to improve HOI in 4D Gaussian Splatting generation is novel. The proposed methods clearly solve the “where” and “how” problems raised in the introduction.
- The proposed method operates in a zero-shot manner with pre-trained diffusion models. It avoids the need for scarce large-scale interaction datasets.

### Weaknesses
A significant problem of the paper is the lack of a substantial body of related work in HOI generation, particularly in 4D HOI generation methods. Important works such as Programmable Motion Generation, Chain-of-Contacts, CG-HOI, HSI, among others, are not discussed. These methods have contributed significantly to the field by generating SMPL and SMPL-X poses for HOI scenarios. 

While the paper focuses on generating realistic appearances using Gaussian splatting, which differs from the approaches of these methods, it is essential for the authors to analyze and position their work in the context of existing studies in 4D HOI generation. For instance, the use of large language models (LLMs) to generate contact-based constraints was first proposed in Chain-of-Contacts. By not acknowledging these relevant works, the paper misses the opportunity to highlight how AvatarGO builds upon or differs from established methods, potentially overlooking important insights and contributions from the existing literature.

Another weakness of the proposed method, as mentioned by the authors in the limitation section, is that it assumes a consistent geometric relationship between the interacting body part and the object. This prohibits it from generating complex manipulations such as tossing an object or passing an object between hands. 

### References

- Liu, Hanchao, et al. "Programmable Motion Generation for Open-Set Motion Control Tasks." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*. 2024.
- Xiao, Zeqi, et al. "Unified Human-Scene Interaction via Prompted Chain-of-Contacts." *The Twelfth International Conference on Learning Representations (ICLR)*.
- Diller, Christian, and Angela Dai. "Cg-hoi: Contact-guided 3d human-object interaction generation." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2024.
- Jiang, Nan, et al. "Scaling up dynamic human-scene interaction modeling." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2024.

### Questions
- How does AvatarGo compare to SMPL-based 4D HOI generation methods?
- Aside from the apparent appearance advantage, does AvatarGo achieve a similar interaction quality to the SMPL-based methods?
- What advantages does direct 4D Gaussian splatting optimization offer over a two-stage approach (SMPL generation followed by GS fitting)?
- Why do the results trend toward anime/cartoon styles? What technical modifications would enable more photorealistic outputs?
- How does runtime scale with sequence length and scene complexity?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces AvatarGO, a novel framework that utilizes diffusion models and language guidance to enable zero-shot generation of realistic, animatable 4D human-object interactions from text prompts.

### Strengths
1. The paper is well-written, with a clear motivation and an easy-to-follow presentation of the core idea. Related work is thoroughly covered, incorporating up-to-date research in the field.
2. Extensive experiments and ablation studies convincingly demonstrate the effectiveness of AvatarGO, showcasing appropriate human-object interactions and superior robustness against penetration issues.

### Weaknesses
I believe this paper is ready for publication, as I found no major weaknesses. My only question concerns the diversity of objects included in the text prompts used in the experiments. Additionally, how does your method perform across different types of objects?

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper proposes AvatarGO, a framework that generates 4D human-object interaction (HOI) to address the challenges of the lack of realistic large-scale interaction datasets. The authors generate 4D HOI samples in a zero-shot manner by leveraging a pre-trained diffusion model (Stable Diffusion). To mitigate the diffusion model's lack of understanding of "where" and "how" objects interact with humans, the authors propose 1) LLM-guided contact retargeting, which identifies the contact body part from the text prompt, and 2) correspondence-aware motion optimization, which models human and object deformation fields using linear blend skinning function. The authors argue that their extensive experiments validate the proposed method's superior generation capabilities.

### Strengths
1. The qualitative results of the proposed method seem more plausible than the compared methods in Figure 3. Moreover, the authors conduct user studies to compare the proposed method with comparable methods in diverse criteria, e.g., penetration, motion quality, and overall performance.

2. The paper is well-structured, with clear sections from introduction to methodology and experiments. The authors provide detailed explanations about their task with various citations, which helps the reviewer to understand the problem the authors focused on.

### Weaknesses
1. The reviewer wonders if the human motion is given as input or is generated by the proposed method. In Figure 2 caption (L185), the authors explain that correspondence-aware motion optimization jointly optimizes human and object animation, but in L345, it is written that the motion sequence is given. The reviewer requests clear explanations of what are the inputs and outputs of the proposed pipeline.

2. The reviewer thinks that the proposed method's novelty is limited. The objective function of the proposed generative model is spatial-aware SDS (L276), that is introduced by ComboVerse (Chen et al., 2024b), not newly proposed by the authors. Moreover, the authors propose LLM-guided contact retargeting, which identifies 3D Gaussians by unprojecting 2D segmentation masks. This process operates with a heuristically pre-defined threshold $\alpha$, and the reviewer wonders if this heuristic threshold $\alpha$ applies to various contact body parts, e.g., fingers, feet, head, and legs.

3. Since the proposed method only handles the composition of generated 3D humans and objects, the overall quality is prone to be bounded by the 3D asset generation module. Then, if the 3D objects and humans are generated in poor quality, how can the authors mitigate the error propagation in the proposed method?

4. The qualitative results in Figure 3 only show the interaction with human hands. The reviewer requests more visualization results of diverse objects and body parts, such as the head, feet, and legs.

### Questions
1. In L310, the reviewer wonders how the authors define the threshold $\alpha$.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces AvatarGO, a framework for generating and animating 4D human-object interactions (HOI) in a zero-shot manner using pre-trained diffusion models. The main contributions of the paper include: (1) LLM-guided ****contact ****retargeting, which addresses the challenge of determining where objects should interact with the human body by using large language models (LLMs) to identify appropriate contact areas from text prompts, and (2) correspondence-aware motion optimization, which ensures coherent motion between humans and objects while preventing issues like object penetration during animation. AvatarGO generates realistic 4D HOI scenes directly from textual inputs, overcoming limitations in existing methods that rely on SMPL-based models. Qualitative and quantitative results show that the AvatarGO outperforms existing methods in terms of generating plausible human-object interaction animation.

### Strengths
- The paper tackles the new task of zero-shot generation of 4d human-object interaction (HOI).
- The proposed LLM-guided contact retargeting helps initialize the object in reasonable 3D locations to help generate plausible HOI.
- The correspondence-aware motion optimization helps prevent penetration and maintain plausible interactions throughout the animation.
- Visually, the method outperforms previous methods in the quality of generated HOI. The higher CLIP score and user preference also support this.

### Weaknesses
- The overall appearance quality of generated humans and objects is still limited. The results are often blurry and lack realism. There is also still human-object penetration (see “Naruto in Naruto Series stepping on a football under his foot” on the webpage).
- The notations in object animation and correspondence-aware motion optimization are poorly explained. What is G_c and G_o? How are they computed? The paper mentioned they are derived from x_ci and x_oi, but it’s unclear.
- It seems the method requires training for each human-object pair, which can be quite inefficient for large-scale generation. Is there anyway for the method to train the same character interacting with multiple objects?

### Questions
- Is each HOI trained with one motion only? Can it generalize to multiple motions?
- Fig. 4 only shows one frame of animation, which is not animation but reposing. Multiple frames need to be shown.

### Soundness
3

### Presentation
3

### Contribution
2
