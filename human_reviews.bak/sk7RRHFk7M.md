# DisCo: Disentangled Control for Realistic Human Dance Generation

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 5, 5

## Abstract
Generative AI has made significant strides in computer vision, particularly in text-driven image/video synthesis (T2I/T2V). Despite the notable advancements, it remains challenging in human-centric content synthesis such as realistic dance generation. Current methodologies, primarily tailored for human motion transfer, encounter difficulties when confronted with real-world dance scenarios (e.g, social media dance) which require to generalize across a wide spectrum of poses and intricate human details. In this paper, we depart from the traditional paradigm of human motion transfer and emphasize two additional critical attributes for the synthesis of human dance content in social media contexts: (i) Generalizability: the model should be able to generalize beyond generic human viewpoints as well as unseen human subjects, backgrounds, and poses; (ii) Compositionality: it should allow for composition of seen/unseen subjects, backgrounds, and poses from different sources seamlessly. To address these challenges, we introduce \ourmodel, which includes a novel model architecture with disentangled control to improve the compositionality of dance synthesis, and an effective human attribute \pretraining for better generalizability to unseen humans. Extensive qualitative and quantitative results demonstrate that \ourmodel can generate high-quality human dance images and videos with diverse appearances and flexible motions.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors introduce a disentangled representation of dance motion that separates the content and style of the motion, allowing for more control over the generated motion. They use a combination of motion capture data and a variational auto to learn this representation and generate new dance motions. The results show that their method is able to generate diverse and realistic dance motions that are controllable in terms of content and style.

### Strengths
- Good quantitative results than baselines.

- Motivation is good and sufficient. 

- The usage of Grounded-SAM on person and background is technical sound.

### Weaknesses
Weakness:

- **Ethics issues. Ethics reviewers are required to review whether double-blind regulations have been violated.**
    - In code `config/__init__.py`, I found there exist Chinese characters and some codes like `MSRA PC Node`.  As we know, MSRA means Microsoft Research Asia. This would make the reviewer to infer the authors’ nationality and possibly Microsoft affiliation.
    - The authors did not discuss possible ethical issues with this study. Including issues such as race and gender bias.
- About Citation
    - It seems that authors are not clear on the difference between `\cite` and `\citep`. For example, “… ControlNet branch with the pre-trained U-Net weight following (Zhang & Agrawala, 2023).” should be“… ControlNet branch with the pre-trained U-Net weight following Zhang & Agrawala (2023).” There are more than one similar problem. During rebuttal, please list all of similar issues and revise them, I will check them one by one.
    - I would like to point out serious issues with the wrong citation on Grounded-SAM. The first implementation of Grounded-SAM is the https://github.com/IDEA-Research/Grounded-Segment-Anything, which has been accepted as an ICCV demo. Please cite it correctly. Besides, if authors used it, please cite Grounding DINO.
    - Instructpix2pix was accepted CVPR, not an arxiv paper. Please cite it correctly. There is more than one similar problem. During rebuttal, please list all similar issues and revise them, I will check them one by one.
    - Missing comma in the equation of Section 3.1.
    - When summarizing contributions, `To address this problem, ...` should be `To address these problems, ...`.
- Reproduction
    - When I try to run the code. I found it missing README and I cannot run the code correctly. I am not sure whether the codes are appropriate.
    - Missing video demo. This is very essential for me to check the results. And the video comparison with baselines is needed.
- Technical Design
    - It is hard for readers to check the technical designs in Figure 2. The CLIP feature is fed into TM module (ResBlock and TransBlock). How do the lines without arrows connect to the module? How do the lines with arrows connect to the module?
    - Do the features output by each layer need to be processed by BG ControlNet and Pose ControlNet and then added to the middle layer? If there are $X$ down layers, in the UNet middle layer, will features be added about $2X$ times?
- Why not compare with Follow-your-pose?

My main concern comes from ethics issues and writing issues. There is something confusing that makes readers hard to follow this work. I spent about 8 hours checking this submission and tried to run the codes (but failed). I plan to rate it as 4 but the system only supports 3 or 5. Therefore, I rate it as 3 now. After the rebuttal and reviewer discussions, I will revise my rating to provide a clear rating of accept or reject.

### Questions
see weakness.

### Soundness
3 good

### Presentation
1 poor

### Contribution
3 good

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper focuses on human motion transfer in real-world dance scenarios, and introduces a novel model called DISCO for better (i) Generalizability and (ii) Compositionality.
Specifically, DISCO applies the ControlNet of background and human pose to disentangle control signals.
Moreover, it is pre-trained in a proxy task to improve the generalizability to unseen humans.
A broadly various of evaluations demonstrate the effectiveness of proposed method.

### Strengths
See summary.

### Weaknesses
1. The technical contributions of this paper are limited. The proposed DISCO is mainly a combination of Image-variation Stable Diffusion, Background ControlNet, and Human pose ControlNet.
2. The authors only provide images or frames in paper, but not videos. Therefor, the temporal consistency of synthesized videos is unconvincing.

### Questions
See weaknesses.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper aims to generate high-fidelity dance videos given three single inputs, reference person image, target pose sketon (2D), target background. This task is similar to traditional pose transfer, while human dance is claimed to be a more challenging task. To this end, the authors propose a control-net based framework, where the foreground and background are taken as seperate inputs. To further strengthen disentanglement, a pretraining strategy is proposed which trains the model on a much larger image dataset. Extensive experiements demonstrate the effectiveness of the proposed method.

### Strengths
There are several merits in this work:
1. Using foreground and background segmentations as seperate inputs for pose transferring is new. It seems to provide a neat yet effective solution. I also appreciate the Human Attribute Pre-training. Firstly, it is simple, but effective as well. Secondly, it properly utilizes the large-scale meta data.
2. The quantitative evaluation results are significant. The proposed model outperforms other methods by a large margin.
3. The authors do have conducted sufficient experiments (e.g., comparisons, ablation), as well as exploring further applications such as fine-tuning on one person.
4. Detailed implementation details, and submitted code.
5. The image transfer results look promising.

### Weaknesses
I am not an expertise in video synthesis, here are my feelings about this work. 

First of all, the authors didn't provide the video demonstration for their work, which is supposed to largely decrease the validity of this work, since the whole highlight is about dance generation. 

1. Though the title is about "dance generation", I feel the emphysis of this work, including technical design, is more on pose-based image transfer. There is little thing about "sequence" modeling for dance.
2. I won't say the generated dances are realistic (as claimed in title). There are many temporal inconsistency and jittering, though I acknowledge the single image-based editting is realistic. I guess for better video modeling, we should pay more attention on the temporal consistency. (I found the video somewhere else.)
3. Upper-body video generation is kind of limited. Is there full-body dance dataset available?
4. Since in diffusion model, generating each image requires hundreds of steps, I am curious how long does it take to generate a dance video. Will that be a limitation of this work?
5. For video generation, it's necessary to have comparisons with baselines.
6. Some discussion about limitations are desirable.

### Questions
Please refer to the weakness, and may respond to these questions if applicable.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper presents a method to generate dance images or videos given reference images of people and background, and poses (or pose sequences, for videos) of the desired dance. The proposed learning model generates the target images or videos by combining attention-pooled CLIP image features for the reference image foregrounds (images of people), and latent features from a ControlNet architecture for the reference image backgrounds and the target pose skeletons. To improve the plausibility of the generated images or videos, the authors also propose a human attribute pre-training strategy to reconstruct the reference images from the foreground and the background features. The authors show the benefits of their proposed method through multiple quantitative and qualitative evaluations and ablation studies.

### Strengths
1. The proposed approach of separating the foreground and the background features from the reference images to learn their individual modifications given the target pose skeletons is technically sound.

2. The human attribute pre-training makes sense, particularly for the challenging scenario of images/videos with cropped or occluded humans.

3. The ablation studies highlight the benefits of the proposed network and training components.

### Weaknesses
1. Since the proposed task of the paper is generative, it requires a human evaluation of the perceived plausibility and overall generation quality. Quantitative metrics do not capture these aspects, and they are commonly covered through various types of user studies in related work. Without such an evaluation, it is hard to assess the impact of the proposed method fully.

2. Have the authors explored or encountered any incompatibilities between the reference images and the target poses? For example, any significant differences in the relative body shapes between the reference and the target, or backgrounds that may not realistically match the target poses? A discussion of such scenarios, or other potential limitations, is important to understand the full scope of the proposed method.

### Questions
1. What is the latency of the end-to-end generation pipeline during inference? Is there any component that takes significantly more time than the others, that is, becomes a bottleneck for efficiency?

### Soundness
4 excellent

### Presentation
3 good

### Contribution
2 fair
