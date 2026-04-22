# Egocentric Cross-Embodiment Video Editing via Dual Contrastive Representation Learning

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 6, 2

## Abstract
Learning robotic manipulation from human videos is a promising solution to the data bottleneck in robotics, but the distribution shift between humans and robots remains a critical challenge. Existing approaches often produce entangled representations, where task-relevant information is coupled with human-specific kinematics, limiting their adaptability. We propose a generative framework for cross-embodiment video editing that directly addresses this by learning explicitly disentangled task and embodiment representations. Our method factorizes a demonstration video into two orthogonal latent spaces by enforcing a dual contrastive objective: it minimizes mutual information between the spaces to ensure independence while maximizing intra-space consistency to create stable representations. A parameter-efficient adapter injects these latent codes into a frozen video diffusion model, enabling the synthesis of a coherent robot execution video from a single human demonstration, without requiring paired cross-embodiment data. Experiments show our approach generates temporally consistent and morphologically accurate robot demonstrations, offering a scalable solution to leverage internet-scale human video for robot learning.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a cross-embodiment video generation method. The main problem of using human videos for robot learning is the embodiment gap. This paper resolves the visual embodiment gap between human and robot by adopting a video generation approach. To disentangle task-centric information from embodiment-centric information, the authors propose a dual contrastive learning approach. Through the proposed objective function, a trainable adapter is trained while maintaining the ability of the frozen video generation backbone. Its generation quality outperforms previous baselines.

### Strengths
- The proposed objective function for dual contrastive learning, which incorporates the concept of disentangling task and embodiment information, is promising.

- The paper is well written, and the provided figures are helpful for understanding.

### Weaknesses
- **Insufficient explanation**
  - The paper provides limited information about the proposed method. For the masked videos, it is unclear what kind of method is used to create the masks. The training datasets used do not provide masks for each hand or robot end-effector. Therefore, the authors might have used an off-the-shelf method, which should be specified.
  - For the embodiment image, especially during inference, the authors do not state which embodiment image is used. Based on the provided figure, it seems to have a similar pose between the human and robot.
- **Limited analysis**
  - The main contribution of the paper is proposing a new objective function to disentangle task and embodiment information. However, the subsequent ablation study is not thoroughly executed. It only examines the removal of the entire objective function.
  - The authors should conduct an ablation study by removing each component of the objective function individually. For example, an ablation study on the mutual information loss could demonstrate the effectiveness of disentangling task and embodiment information. However, this paper lacks careful analysis of each component, even though it forms the core of the proposed method.
- **No robot experiment**
  - The main motivation of the paper is reducing the embodiment gap between human and robot for scalable robot training. However, the paper does not contain any robot downstream tasks.
  - The major distinction between computer vision and robotics lies in their application focus. The visual gap between humans and robots is a major bottleneck for robot learning, but various qualitative metrics such as FVD or LPIPS cannot guarantee effectiveness in real-world robot downstream tasks. For example, Phantom[1] demonstrates its effectiveness in reducing the embodiment gap by performing real-world robot tasks.
  - If the authors believe this paper falls under the category of generative methods, they should tone down the emphasis on robot learning and include further analysis regarding generative quality.
- **Missing citations**
  - H2R[2] is also a human-to-robot augmentation approach. It uses states such as hand poses and inpainted human video to generate robot datasets from human datasets.
  - UniSkill[3] is another learning from human approach. Its training objective ensures the extraction of embodiment-agnostic but task-centric latent skills from human videos. It also demonstrates cross-embodiment generation results from human to robot.
--- 
[1] Lepert, Marion, Jiaying Fang, and Jeannette Bohg. "Phantom: Training robots without robots using only human videos." arXiv preprint arXiv:2503.00779 (2025).

[2] Li, Guangrun, et al. "H2R: A Human-to-Robot Data Augmentation for Robot Pre-training from Videos." arXiv preprint arXiv:2505.11920 (2025).

[3] Kim, Hanjung, et al. "UniSkill: Imitating Human Videos via Cross-Embodiment Skill Representations." arXiv preprint arXiv:2505.08787 (2025).

### Questions
- How do you process the masked videos?
- For the self-embodiment image during inference, can it be any kind of robot embodiment image, or should it have a similar pose to the human?
- What's the effect of each objective function?
- What's the benefit of using a generative approach from human to robot? For the robot learning, rendering approaches like Phantom or H2R are sufficient for scalable robot learning.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a dual mutual information framework to disentangle embodiement representation and task representation. This disentangled representation learning is then used to edit human demonstration video into a robot demonstration video. Specifically, the two representations are encouraged to be maximally informative of each other within the modality, whereas they are encouraged to have minimal information across the modality. The authors used two different mutual information (MI) estimators. For intramodal MI maximization, InfoNCE estimator is used. For the intermodal MI bottlenecking, CLUB estimator is used. It should be noted that the this cross-embodiment generalization is an emergent result of this disentanglement, not a result of explicit supervised training. The training itself is done in an autoencoding manner without ground-truth cross-embodiment pair.

### Strengths
The proposed approach is well motivated and the information-theoretic approach is technically sound. While the idea dual contrastive learning is not new in the so-called multiview representation learning field, its adapation to embodiment-transferrable video robot policy is an important and timely contribution.

### Weaknesses
While the methodology itself makes a lot of sense, empirical results are insufficient to support its advantage over baseline methods or applicability to wide variety of videos and embodiments. Specifically, I have two major concerns.

### **[Concern 1]** Metrics used measure video quality, not the transferrability.
I understand that it is incredibly difficult to objectively measure the performance for this kind of problem. That said, the evaluation protocols seems inappropriate or hard to understand. 

For instance, the metrics in the Video Fidelity category requires target video sample (PSNR, LPIPS, SSIM) or distribution (FVD). However, there cannot be a ground truth target sample/distribution for this kind of cross-embodiment editing. As such, I believe the evaluation is done in an autoencoding fashion rather than with the unavailable GT cross-embodiment target. If the former, it contradicts the claim that the general purpose VACE performs better in-distribution but does not transfer to cross-embodiment editing. If the latter, it is important to explain how the GT pairs are obtained.

The VBench result is also counterintuitive. If the cross-embodiment transferrability is the key, then the metrics like Subject Consistency (SC) should be higher than VACE. However, the result is the opposite. The proposed method instead excels at more detailed consistency like temporal flickering or motion smoothness. By only looking at these results, I would be more inclined to believe that VACE actually transfers better. It would be helpful if the authors can provide video generated by VACE that corresponds to the one shown in Figure 3.

### **[Concern 2]** Diversity of evaluation settings is limited.
The provided qualitative results (supp. video and figures) are limited in diversity in terms of embodiment and background. These qualitative results are only shown for a single robot and a background. It is also unclear how diverse the quantitative evaludation settings are. Necessary details such as the number of embodiments, backgrounds, and tasks are missing.


As the methodology itself makes sense and the figures/videos looks convincing (although they are limited in diversity), I am willing to adjust my rating to more positive ones if these concerns are addressed.

### Questions
Please find the questions in the weakness section.

### Soundness
2

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
4

### Summary
This work proposes a representation learning framework for disentangling task and embodiment features in egocentric videos. The disentangled representations are then used for cross-embodiment video editing by replacing human hands with robot end-effectors using a video diffusion model. The model is trained with a flow-matching objective and a dual contrastive learning loss that minimizes mutual information between the task and embodiment representations while maximizing intra-space consistency. Experiments on the Physical Human-Humanoid Dataset ($PH^2D$) show that the proposed approach outperforms existing baselines in terms of visual quality and motion consistency metrics.

### Strengths
- Cross-embodiment video editing helps to mitigate the embodiment gap between human hands and robot end-effectors in pixel space.
- The proposed dual contrastive objective is effective at learning a disentangled latent space, as evident in Fig.4.
- Learning a disentangled representation for task & embodiment from unpaired cross-embodiment data has the potential to scale up robot learning from human videos.
- Quantitative results in Tab.1 show the benefits over existing baselines in terms of visual quality (FVD, LPIPS, PSNR, SSIM) & motion consistency (TS, MS, OC, TF) metrics.

### Weaknesses
- It'd be useful to have a simple baseline that finetunes an existing video editing model (using adapters) on the $PH^2D$ dataset. For example, training the proposed model with only the flow-matching objective from Eq.3. This would help understand if the representation learning framework is indeed effective compared to simpler alternatives like finetuning. 
- Tab.2 has an ablation without the dual contrastive loss. Which loss components from Eq.3 are removed in this ablation? It is important to ablate the different loss terms in Eq.3 (i.e., $L_{disentangle}$, $L_{taskcontrast}$, $L_{embcontrast}$) to understand the contribution of each term. Since the disentanglement component is applied only once every ten training steps for stability (L362-363), a systematic ablation would further clarify if this is indeed required.
- As per the CLUB paper, the upper bound on MI is valid if the conditional distribution is known. For unknown conditional distributions, the upper bound holds if a trained variational model can approximate the conditional distribution reasonably well (Theorem 3.2 in the CLUB paper). It'd be useful to clarify this since the proposed work trains a variation model $q_{\phi}$ to approximate the conditional distribution. Also, more details are required about this variation model and how it is trained. Is it a simple VAE or more complex neural network? Is it trained jointly with the cross-embodiment editing framework or separately?
- The text has several mentions of "physically plausible" (L196, L407) and "morphologically accurate/consistent" (L025-026, L083, L407-408, L481). However, there is no evaluation of physical plausibility or morphological consistency. The current evaluation metrics only focus on visual fidelity but do not capture kinematic structure or end-effector physics/dynamics (except in the pixel space). I'd suggest removing these terms from the text since the experiments do not support these claims.
- Visualizations in Fig.3,5,6,7 show that the VACE model often replaces the hand with either a black mask-like region or another hand. It'd be interesting to have some insights into this behavior. Is it because VACE has never seen robot end-effectors or any other reason?

### Questions
While learning a task-embodiment disentangled representation space is an elegant idea, I have three concerns (more details in the Weaknesses above):
- A simple finetuning baseline is required to verify if the proposed framework is indeed effective than simpler alternatives.
- Since a dual contrastive objective is a contribution, a system ablation of different loss terms is needed to understand the effectiveness of each term.
- Clarifications regarding the use of "physically plausible" and "morphologically accurate/consistent" terms.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a generative framework for cross-embodiment video editing, enabling robots to imitate human demonstrations by translating human egocentric videos into robot-execution videos. The method learns disentangled task and embodiment representations via a dual contrastive objective—minimizing mutual information between the two spaces while maximizing intra-space consistency.
By injecting these representations into a frozen video diffusion model, the system can synthesize temporally coherent and morphologically consistent robot videos from human examples without paired data. Experiments show improved visual quality compared to VACE and Phantom baselines.

### Strengths
Interesting and timely problem: Tackles the “embodiment gap” in robot learning from human videos — a core challenge in scalable imitation learning.

### Weaknesses
- **Evaluation metrics are weak and inconsistent.**: The paper relies primarily on video-generation metrics (FVD, SSIM, LPIPS), which are not well correlated with imitation quality or task performance. In particular, the reported improvements are numerically small and inconsistent across metrics—e.g., LPIPS barely changes, and VACE occasionally performs similarly or better on some sub-metrics. These differences are difficult to interpret and don’t convincingly demonstrate true progress in cross-embodiment understanding.

- **Missing downstream validation.**: Since the stated goal is to enable robot imitation from human video, the key question is whether the generated videos actually improve downstream imitation or policy learning. The paper stops at visual evaluation without measuring how close the generated robot demonstrations are to real robot executions (e.g., comparing to ground-truth robot videos or evaluating imitation performance when trained on generated vs. real data).
This is critical: even if the videos look realistic, they may not encode actionable trajectories or physical consistency necessary for control. A simple behavioral cloning or visual policy learning experiment [1] would make the paper significantly stronger. 

- **Quantitative results lack interpretability**: As this is a generative model, it’s understandable that numeric metrics are limited, but the current evaluation doesn’t connect to the end goal. Showing relative fidelity to real robot demonstrations (as an upper bound) would contextualize how useful the generated data actually is.

[1] Kim, Hanjung, et al. "UniSkill: Imitating Human Videos via Cross-Embodiment Skill Representations." arXiv preprint arXiv:2505.08787 (2025).

### Questions
Please see weaknesses.

### Soundness
1

### Presentation
2

### Contribution
1
