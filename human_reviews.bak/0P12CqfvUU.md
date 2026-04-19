# 3D Morphable Master Face Generation: Towards Controllable Wolf Attacks against 2D and 3D Face Recognition Systems

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 3, 5

## Abstract
Biometric authentication systems are facing increasing threats from Artificial Intelligence-Generated Content (AIGC). Previous research has revealed the vulnerability of face authentication systems against master face attacks. These attacks utilize generative models to create facial samples capable of matching multiple registered user templates in the database. In this paper, we present a systematic approach for generating master faces that can compromise both 2D and 3D face recognition systems. Notably, our approach is the first to enable morphable and controllable master face attacks on face authentication systems.

Our method generates these 3D master faces using the Latent Variable Evolution (LVE) algorithm with the 3D Face Morphable Model (3DMM). Through comprehensive simulations of simultaneous master face attacks in both white-box, gray-box, and black-box scenarios, we demonstrate the significant threat posed by these 3D master faces to mainstream face authentication systems. Furthermore, we explore the realms of face morphing and facial reenactment in our generated samples, enhancing the efficacy of the master face attack. Compared to existing methods, our approach exhibits superior attack success rates and advanced flexibility, highlighting the importance of defending against master face attacks.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
Summary: In this paper, the authors study the master attack on 3D face recognition systems. They use combined 2D and 3D face similarities as a loss functions, CMA-ES as an optimizer, 3DMM for controlling the face properties.

### Strengths
1.	The first several sections are well-presented. 
2.	The algorithm is simple and straightforward.

### Weaknesses
1.	Some consider that master attack is a real problem. This problem is existing when we run the system in identification mode, 1-to many matching, which is used in low security application. This attack is much less effective in verification mode, 1-to-1 matching. If a system is run on identification model, collecting data from one particularly registered user is very possible. Thus, target attack can be performed. 
2.	The proposed work is an integration of the existing models and tools. It affects it novelty. 
3.	This work can be considered as an extension of Friedlander et al 2022, which also generates 2D and 3D master faces. The proposed work uses better components to achieve better performance. It further weakens the novelty. 
4.	The presentation of the experimental session is not good enough. I read several times to try to understand the contents. The authors also use abbreviation defined in appendix, e.g., dev. The table numbers in the main text are not in order. Very significant effort is needed to improve the presentation.
5.	Most of the attacks are ineffective. Many zero FMR in Table 3. Most important, in black-box setting, the attack is totally ineffective. 
6.	The current study is in digital domain but 3D face is called in physical domain.
7.	Furthermore, all the previous related works are published in the biometric conference and biometric journals. Submitting this paper to e.g. IEEE Trans. On Biometrics, Behaviour and Identity Science or other biometric conferences, may be more suitable.

### Questions
See the comments in weaknesses.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper presents the vulnerability of biometric authentication systems to master face attacks, which utilize generative models to create facial samples that can match multiple registered user templates. The authors propose a systematic approach for generating 3D master faces that can compromise both 2D and 3D face recognition systems. They introduce a novel framework that uses a latent variable evolution (LVE) strategy with a 3D Morphable Face Model (3DMM) to generate these master faces. The authors demonstrate the significant threat posed by these 3D master faces through simulations in various attack scenarios. They also explore the application of facial reenactment and morphing in the generated samples to enhance the efficacy of the master face attack. The authors highlight the importance of defending against master face attacks and call for further research on strengthening face recognition systems.

### Strengths
-With the fast development of AIGC, the problem of master face attacks discussed in this paper will be an important issue for the face recognition system. Thus, the task of solving the master face attacks is meaningful. 
-The problem of generating 3D master faces for 3D face recognition systems may be firstly studied by the authors. 
-A latent variable evolution (LVE) strategy with a 3D Morphable Face Model (3DMM) is used to generate the 3D master faces, and the facial reenactment and morphing are further explored. 
-Experiments with different databases, deep face models, and attack scenarios has been demonstrated.

### Weaknesses
-The problem of master face attacks for 2D FR systems has been studied by Nguyen et al. (2020), Shmelkin et al. (2021) and Nguyen et al. (2022). The authors extend this issue to 2D and 3D FR systems, and thus the novelty of the proposed method is limited;

-The presentations and the layout of the whole paper is poor, the authors should re-organize the whole paper. Many information is not complete to the reader. The main part of the paper should be closed to itself, including the method and experiments. But there are many information are introduced in the APPENDIX;

- The comparisons to the baseline is not faithful to the reader. It should be also compare the proposed method with the previous method of generating master face attacks for 2D FR.

### Questions
-The experimetal setting for different databases should be clearly presentated. 

-Generally speaking, 3D face shapes generated by using the latent codes of 3DMM has inherent disadvantages of linearity, and looks very similar to each other (no identity information), thus I am wonder why the generated 3D face shapes can match many gallery faces with a pre-defined match threshold? Since we know that the threshold value of a given 3D face recognition system has been defined and fixed once the system has been used.

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes an approach for generating master faces that can compromise both 2D and 3D face recognition systems.3D master faces are generated using the Latent Variable Evolution (LVE) algorithm with the 3D Face Morphable Model (3DMM). Simultaneous master face attacks in both white-box, gray-box, and black-box scenarios are simulated in thee experiments showing the threat posed by these 3D master faces to mainstream face authentication systems. Face morphing and facial reenactment are also experimented on generated samples, enhancing the efficacy of master face attacks.

### Strengths
The main contributions are:
- a method is proposed that enhances the threat and usability of 3D master faces. 
- an evaluation is performed across diverse 3D face datasets and various face recognition systems, encompassing
both white-box, gray-box, and black-box attack scenarios to simulate real-world master face attacks.
- it is shown how a controllable master face can enhance potential attacks through facial reenactment and morphing.

### Weaknesses
- The contribution of the paper in terms of learning solutions and its relevance to the conference seem a bit below the standard for ICLR
- the 3D face recognition methods referred in the related work section are quite old (there are no methods cited after 2019)
- the 3DMM use is not well detalied, investigated and discussed. 
- experiements are not fully convincing

### Questions
- the used 3DMM it is not well explained. It is not clear which is the model used and its topology (from Figure 1 it seems Flame topology). It is the Flame model or the Basel Face Model? Did the authors consider non-linear 3DMM learned using deep neural networks? Which are the differences in using one 3DMM or a different one? Which are the data used to learn the 3DMM in the different experiments?
- 3DMM are not very good to reproduce fine shape details of the face. So one question here is how this impact in the proposed solution. Is it something desired this lck of detail or a more precise 3D face generation solution could be helpful?
- the 3D information is transformed to depth images. How does this solution affect the method? Would it be possible to work directly in the 3D domain? 
- the BU-3DFE and Texas3D datasets are quite low-resolution with smooth surfaces. So, it can be this lack of details can couple better with the used 3DMM. Authors should use higher-resolution 3D datasets. I strongly suggest to use the FRGC v2.0 dataset.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors aim at presenting a systematic approach for generating master faces that can compromise both 2D and 3D face recognition systems using the Latent Variable Evolution with the 3D Face Morphable Model.

### Strengths
The paper is well-written.
The motivation is clear.
The experimental setup is extensive.

### Weaknesses
Although the paper has a good potential, the paper does not provide sufficient novelty for publication. I was wondering if the authors could provide a summary of the major contributions of the paper.

### Questions
Please see above.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
