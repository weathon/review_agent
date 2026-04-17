# Unified Multi-Modal Interactive and Reactive 3D Motion Generation via Rectified Flow

- Decision: Accept (Poster)
- Scores: 4, 4, 4, 8

## Abstract
Generating realistic, context-aware two-person motion conditioned on diverse modalities remains a fundamental challenge for graphics, animation and embodied AI systems. Real-world applications such as VR/AR companions, social robotics and game agents require models capable of producing coordinated interpersonal behavior while flexibly switching between interactive and reactive generation. We introduce DualFlow, the first unified and efficient framework for multi-modal two-person motion generation. DualFlow conditions 3D motion generation on diverse inputs, including text, music, and prior motion sequences. Leveraging rectified flow, it achieves deterministic straight-line sampling paths between noise and data, reducing inference time and mitigating error accumulation common in diffusion-based models. To enhance semantic grounding, DualFlow employs a novel Retrieval-Augmented Generation (RAG) module for two-person motion that retrieves motion exemplars using music features and LLM-based text decompositions of spatial relations, body movements, and rhythmic patterns. We use contrastive rectified flow objective to further sharpen alignment with conditioning signals and add synchronization loss to improve inter-person temporal coordination. Extensive evaluations across interactive, reactive, and multi-modal benchmarks demonstrate that DualFlow consistently improves motion quality, responsiveness, and semantic fidelity. DualFlow achieves state-of-the-art performance in two-person multi-modal motion generation, producing coherent, expressive, and rhythmically synchronized motion.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents a unified framework for two-person motion generation from texts or/and music. DualFlow employs a retrieval-augmented generation module for semantic alignment. It also uses a contrastive rectified flow to achieve good sampling quality. Experiments on three datasets, InterHuman-AS, DD100, and MDD all shows DualFlow outperforms most baseline methods, especially in terms of the semantic alignment, showing the effectiveness of the proposed RAG, and contrastive learning object.

### Strengths
+ A new SOTA performance is achieved with the proposed method, with cutting-edge techniques, including RAG, contrastive rectified flow. It will serve as a better and stronger baseline in the field of two-person motion generation.
+ Experiments are extensive. Three benchmarks and ablation studies all shows the effectiveness of the proposed method.

### Weaknesses
- The paper looks very much like an "updated and modern version" of motion diffusion model, with some fancy new techniques from broader diffusion community to boost the performance, but is short of insights in motion generation itself, thus makes me feel lack of novelty.
- I did not see many insights in the "two-person" motion generation. Joint modeling of two-person motions and the synchronization loss are pretty standard in two-person motion generation. RAG and contrastive rectified flow are very general method and can be applied on single-person motion generation.

Typo:
- Line 295: L_{sync} => L_{inter}

### Questions
N/A

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
This paper proposes a framework for human-human interaction/reaction generation conditioned on text descriptions and music. The framework integrates multiple existing techniques, such as Rectified Flow and Retrieval-Augmented Generation. It demonstrates notable improvements over baselines across multiple datasets.

### Strengths
1. The figures and tables are clear and visually appealing, with an overall well-structured and comprehensible presentation.

2. The paper presents an engineering system of considerable complexity, integrating multiple relevant techniques.

3. The quantitative improvements over baselines are substantial.

### Weaknesses
1. **Outdated motivation regarding computational efficiency.** The paper claims that diffusion-based approaches like Motion Diffusion Model (MDM) [1] "remain limited by high computational cost" and "weak real-time reactivity" (Lines 55-57), and "require hundreds of denoising steps, making real-time use impractical" (Line 107). The proposed method, built on Rectified Flow, requires 200 steps to demonstrate efficiency. However, as early as 2024, the authors of MDM released MDM-50steps in their GitHub repository, which requires only 50 denoising steps without degrading FID and other metrics. Current state-of-the-art models commonly use 50 or fewer denoising steps, such as [2,3,4,5,6]. This significantly weakens the claimed advantage of the proposed approach.

2. **Insufficient motivation for adopting Rectified Flow.** A main contribution of the paper is the introduction of Rectified Flow, motivated by the claim that diffusion-based approaches have "constraints [that] hinder their applicability in interactive systems such as virtual avatars, social robotics, and dance simulation, where low-latency and seamless coordination are essential." Given the point above, this motivation appears questionable and may not hold.

3. **Lack of rigorous quantitative evidence for efficiency claims.** In Section 4.2, the paper supports its claim of "faster sampling and reduced latency" primarily by showing fewer inference steps compared to InterGen. This is insufficient. Given differences in network architecture and parameter count, a more rigorous evaluation should include Average Inference Time per Sentence (AITS) [6] or similar metrics to substantiate the efficiency claims.

4. **Limited validation and impact of RAG module.** The paper lacks validation of whether the LLM-generated descriptions (spatial relationship, body movement, rhythm) are consistent with actual motion content. Additionally, the ablation study (Table 4) shows that RAG provides only modest improvements, with some metrics in the reactive task showing marginal degradation. The practical value of this added complexity requires more thorough investigation.

5. **Concerns about evaluation reliability.** The evaluators achieve only 0.231 R-Prec@1 on ground truth, which is relatively low. While the paper includes user studies, it would be beneficial to discuss the limitations of these automatic evaluators and why ground truth performance is weak, helping readers better interpret the reported improvements.

6. **Limited methodological novelty.** The proposed framework is largely a combination of existing techniques. The key component (Rectified Flow) lacks strong motivation as discussed above, and the RAG module shows limited improvements. The paper would benefit from stronger empirical validation or clearer articulation of why this particular combination is necessary.

### Questions
1. Could you provide more details on how the positive and negative velocity samples are obtained?

2. InterGen [7] uses 8 blocks with 2 attention layers per block, while the proposed method uses 20 blocks with 4 attention layers per block. Does this imply that the proposed method has significantly more parameters than InterGen? Please provide a comparison of parameter counts.



**References:**

[1] Tevet G, Raab S, Gordon B, et al. Human motion diffusion model[J]. arXiv preprint arXiv:2209.14916, 2022.

[2] Dai W, Chen L H, Wang J, et al. Motionlcm: Real-time controllable motion generation via latent consistency model[C]//European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2024: 390-408.

[3] Huang Y, Yang H, Luo C, et al. Stablemofusion: Towards robust and efficient diffusion-based motion generation framework[C]//Proceedings of the 32nd ACM International Conference on Multimedia. 2024: 224-232.

[4] Hong S, Kim C, Yoon S, et al. SALAD: Skeleton-aware Latent Diffusion for Text-driven Motion Generation and Editing[C]//Proceedings of the Computer Vision and Pattern Recognition Conference. 2025: 7158-7168.

[5] Bae J, Hwang I, Lee Y Y, et al. Less is more: Improving motion diffusion models with sparse keyframes[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2025: 11069-11078.

[6] Wang Y, Wang S, Zhang J, et al. TIMotion: Temporal and Interactive Framework for Efficient Human-Human Motion Generation[C]//Proceedings of the Computer Vision and Pattern Recognition Conference. 2025: 7169-7178.

[7] Liang H, Zhang W, Li W, et al. Intergen: Diffusion-based multi-human motion generation under complex interactions[J]. International Journal of Computer Vision, 2024, 132(9): 3463-3483.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes DualFlow, a unified rectified flow-based framework for multi-modal 3D two-person motion generation, which supports both interactive and reactive motion generation tasks and can take text, music, and prior motion sequences as input conditions. Experiments demonstrate that DualFlow performes well on the MDD, InterHuman-AS, and DD100 datasets.

### Strengths
DualFlow seamlessly integrates interactive and reactive two-person 3D motion generation. The incorporation of Retrieval-Augmented Generation (RAG) with music features and LLM-based text decomposition effectively strengthens semantic grounding. Besides, leveraging contrastive flow matching improves motion quality, diversity, and alignment with conditioning signals.

### Weaknesses
1. The motivation is weak. Two-person motion generation is not a new task. But the author emphasize the limitations of single-person motion generation methods. 
2. The technical contributions are limited. I didn't find anything that really impressed me in this article. This paper is a combination of RAG, Contrastive Flow Matching. The authors are advised to claim why using RAG and what advantages RAG has in practical applications, since the motion generation doesn't have a large source of corpus that can be retrieved from.

### Questions
1. In Table 4, the Top3 of DualFlow (Spectral) is 477,which may be wrong.
2. The quantitative results of ReGenNet in Table 2 is under unconstrained setting, i.e., to generate instant reactions without knowing the intention of the actors. Thus the comparison of ReGenNet is not fair.
3. Two reference papers are not existed. It is recommended that the authors carefully review the entire manuscript to ensure correctness.
(1) Yuxuan Cao, Jiawei Ren, et al. Diffmotion: Conditional motion generation with diffusion models. In CVPR, 2023.
(2) Yilun Duan, Joshua B Tenenbaum, et al. Multi-modal motion generation for embodied ai via transformers. arXiv preprint arXiv:2204.08718, 2022.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces a novel framework DualFlow for synchronized two-persoon 3D motion generation. DualFlow unifies the tasks of interactive motion generation and reactive motion generation within a single rectified flow framework and a Retrieval-Augmented Generation module for semantic grounding from text and music. The authors also adopt contrastive learning for better motion quality and inter-person coordination. Experiments show the effectiveness of the proposed frame.

### Strengths
1. The unified transformer-based architecture for both interactive and reactive two-person motion generation is practical and novel.

2. The RAG module, guided by LLM-decomposed text cues (spatial relationship, body movement, rhythm) and music features, provides semantically rich conditioning that is well-motivated and improves generation quality and diversity. 
It's a good application. 

3. The adoption of Rectified Flow Matching is technically sound and is more effective than diffusion.

4. The paper is well-written and clearly structured. The architecture and methodology are described in sufficient detail, it would be reproducible if the codes and data were released.

5. The experiments are comprehensive on multiple two-person motion datasets (MDD, InterHuman-AS, DD100) and show better performance than baselines. The user study is helpful.

### Weaknesses
1. The overall ablation over RAG is limited, and the effectiveness of each retrieved feature (RS, RB, RR, RM) should be more precisely evaluated. 

2. How do you set the Look-Ahead parameter L? There should be an ablation study to evaluate the effectiveness of L, and analyze how this parameter affects the performance of reactive motion generation.

### Questions
Refer to weaknesses, and it would be better to show some failure cases.

### Soundness
3

### Presentation
3

### Contribution
3
