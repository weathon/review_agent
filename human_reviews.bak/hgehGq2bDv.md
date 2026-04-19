# GPAvatar: Generalizable and Precise Head Avatar from Image(s)

- Decision: Accept (poster)
- Scores: 6, 6, 6

## Abstract
Head avatar reconstruction, crucial for applications in virtual reality, online meetings, gaming, and film industries, has garnered substantial attention within the computer vision community. The fundamental objective of this field is to faithfully recreate the head avatar and precisely control expressions and postures. Existing methods, categorized into 2D-based warping, mesh-based, and neural rendering approaches, present challenges in maintaining multi-view consistency, incorporating non-facial information, and generalizing to new identities. In this paper, we propose a framework named GPAvatar that reconstructs 3D head avatars from one or several images in a single forward pass. The key idea of this work is to introduce a dynamic point-based expression field driven by a point cloud to precisely and effectively capture expressions. Furthermore, we use a Multi Tri-planes Attention (MTA) fusion module in tri-planes canonical field to leverage information from multiple input images. The proposed method achieves faithful identity reconstruction, precise expression control, and multi-view consistency, demonstrating promising results for free-viewpoint rendering and novel view synthesis.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The authors proposed a one-shot/few-shot NeRF-based talking face system. The contributions are twofold: (1) a point-based expression field (PEF) for 3D avatar animation; (2) a multi triplanes attention (MTA) that supports multiple images as input to handle hard cases like occlusion or closed eyes. I like some ideas in this paper. This is a overall well-written paper and is easy to follow. The experiment shows good performance over previous baselines. 

However, the identity similarity in the demo video is not as good as previous one-shot 3D talking face methods (such as HiDe-NeRF). Besides, I'm also curious about the performance of this method under large head poses, which is not revealed in the demo.

### Strengths
- I like the idea of PEF since it well utilizes the geometry prior of FLAME to help learn the avatar animation in the 3D space. 
- Also, it is the first one-shot 3D talking face paper that focuses on the few-shot setting.
- The paper is well-written and is easy to follow.

### Weaknesses
- the PEF could well handle the segment modeled by FLAME (such as head and torso), but it cannot handles other parts, such as hair and clothes. See question 1.
- The identity similarity in the demo video is worse than some baseline (HideNeRF).
- The image quality can be improved. For instance, in Figure 1, the predicted images in the second column seems blurry.

### Questions
- In  PEF, the expression feature of facial part can be queried from the FLAME mesh, but the non-facial part, such as hair, clothes, and background is not modeled by FLAME. The authors said "we instead search for neighboring points in the entire space", but it is not clear how it bundles the non-facial part in the 3D space with the learnable features.
- The identity similarity in the demo video is not as good as previous one-shot 3D talking face methods (such as HiDe-NeRF), what's the cause?
- The head movement in the provided video is quite gentle. Is there any demo where the head pose is larger (such as side view)? Since one of the biggest advantage of 3D methods over the traditional 2D methods is the good quality under a large view angle, I think this is necessary for the reviewer to assess the performance of a 3D-based work.
- Could you provide the visualization of the attention weights in the multi-reference setting? I'm also curious about the scalability of the MTA, how is the attention map looks like under the two-in/five-in/ten-in ?
- In the demo video, the avatars are driven by audio, it is better to illustrate the way used to obtain the facial expression.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a method that reconstructs 3D head avatars from images and synthesizes realistic talking head videos. It takes as input a single or a small number of face images and reconstructs 3D head avatars in a single forward pass. It extends the formulation of NERFS and proposes a dynamic point-based expression field that is driven by a point cloud, motivated by the need to have an accurate control of facial expressions. In addition, the proposed method adopts a Multi Tri-planes Attention (MTA) fusion module that facilitates the 3D representation of the scene and the incorporation of information from multiple input images. The proposed method is compared with several SOTA methods and achieves promising results.

### Strengths
+ The proposed method achieves promising results and the supplementary videos show that the videos synthesized by the proposed method are in general realistic and visually pleasing. 

+ The experimental evaluation is detailed and systematic. The proposed method is compared with several recent SOTA methods that solve the same problem.

### Weaknesses
- The presentation in several parts of the paper, especially in the methodology, is unclear and needs several clarifications. See detailed comments in Questions below. 

- The following paper is not cited, despite the fact that it is very closely related in terms of methodology:

Athar, S., Shu, Z. and Samaras, D., 2023, January. Flame-in-nerf: Neural control of radiance fields for free view face animation. In 2023 IEEE 17th International Conference on Automatic Face and Gesture Recognition (FG) (pp. 1-8). IEEE.

The above un-cited paper also uses a FLAME-based representation of the 3D face and extends the formulation of NERFS to achieve realistic face animation with expression control. The similarities and differences with the proposed method are not discussed and the Flame-in-nerf is not included in the comparisons of the experimental section.  This raises concerns in terms of the real novelty and contributions of the proposed method. 

- Furthermore, there are also several other closely related works that are not cited. For example: 

Jiaxiang Tang, Kaisiyuan Wang, Hang Zhou, Xiaokang Chen, Dongliang He, Tianshu Hu, Jingtuo Liu, Gang Zeng, and Jingdong Wang. Real-time neural radiance talking portrait synthesis via audio-spatial decomposition. arXiv preprint arXiv:2211.12368, 2022.

Yudong Guo, Keyu Chen, Sen Liang, Yong-Jin Liu, Hujun Bao, and Juyong Zhang. Ad-nerf: Audio driven neural radiance fields for talking head synthesis. In ICCV, pp. 5784–5794, 2021.

Yu, H., Niinuma, K. and Jeni, L.A., 2023, January. CoNFies: Controllable Neural Face Avatars. In 2023 IEEE 17th International Conference on Automatic Face and Gesture Recognition (FG) (pp. 1-8). IEEE.

### Questions
- Section 3.2: the paper fails to clearly explain how the expression information affects the process of building a point-based expression field.

- Section 3.3: the paper provides insufficient details about about how the canonical encoder is defined and built.

- Figure 4: For several columns with results, it is unclear which is the corresponding method. There is one column more than the number of methods in the caption and one column more than the columns of Figure 5. This creates confusions. 

- Equation (1): the definition of w_i does not seem to make sense. Is there a missing norm in the denominator?

- After Equation (2): N is refereed to as the "input number", but apparently it should be referred to as the number of input images

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
In this paper, a novel framework named GPAvatar is introduced, designed to reconstruct 3D head avatars seamlessly from one or multiple images in a single forward pass. The key novelty lies in the incorporation of a dynamic point-based expression field, guided by a point cloud, to intricately and efficiently capture facial expressions. The authors present the concept of a dynamic Point-based Expression Field (PEF), enabling accuracy and control of expressions across different identities. Additionally, they introduce a Multi Tri-planes Attention (MTA) fusion module, capable of handling a varied number of input images with precision.

### Strengths
- Overall, the paper is well-organized and easy to follow. The motivation is clear. The figures and tables are informative.

- Experimental results demonstrate that the proposed method achieves the most precise expression control and state-of-the-art synthesis quality (StyleHeat, ROME, OTAvatar, and Next3D) (based on NeRF and 3D generative models)n on multiple on VFHQ and HDTF benchmark datasets.

### Weaknesses
- The model proposed has overall more trainable parameters compared to baseline models, which could potentially bring in some unfairness during comparison with other works.
- No discussion about the limitations of the approach?

### Questions
- It is not clear to me how the model captures the subtle information such as closed eyes ? 
- why the normalization of the weight wi in the equation is required?
- Several terms in the equation (page 3) I_t= R (....) are not defined. What R means?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
