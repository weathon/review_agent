# Dual-Path Condition Alignment for Diffusion Transformers

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Denoising-based generative models have been significantly advanced by representation-alignment (REPA) loss, which leverages pre-trained visual encoders to guide intermediate network features. However, REPA's reliance on external visual encoders introduces two critical challenges: potential \textit{distribution mismatches} between the encoder's training data and the generation target, and the high \textit{computational costs} of pre-training. Inspired by the observation that REPA primarily aids early layers in capturing robust semantics, we propose an unsupervised alternative that avoids external visual encoder and the assumption of consistent data distribution. We introduce \textit{\textbf{DU}al-\textbf{P}ath condition \textbf{A}lignment} (\textbf{DUPA}), a novel self-alignment framework, which independently noises an image multiple times and processes these noisy latents through decoupled diffusion transformer, then aligns the derived conditions\textemdash low-frequency semantic features extracted from each path. Experiments demonstrate that DUPA achieves FID$=$1.46 on ImageNet 256$\times$256 with only 400 training epochs, outperforming all methods that do not rely on external supervision. DUPA is also model-agnostic and can be readily applied to any denoising-based generative model, showcasing its excellent scalability and generalizability. Code is available at https://github.com/PCH-gg/DUPA, https://openi.pcl.ac.cn/OpenAIDriving/DUPA.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper aims to accelerate the diffusion transformer training without an external pre-trained encoder by aligning latents from multiple noisy images through decoupled diffusion transformers. They show the proposed method, DUPA, can perform better than REPA without any external guidance.

### Strengths
1. The paper is well-motivated and easy-to-follow.

2. The proposed method, DUPA, consistently improves the baselines, and even perform better than REPA.

3. Extensive analysis demonstrates the effectiveness of each suggested component.

### Weaknesses
1. The authors point out the weakness of REPA in practical scenarios (i.e., beyond the ImageNet), e.g., out-of-distribution problem can arise, and additional pretraining is required. However, the authors do not handle such scenarios in the main experiments: They only conducted ImageNet generation results, which makes it difficult to argue whether the proposed method indeed addresses the problem of REPA. I think such a problem of REPA does not arise in the image generation problem, as (1) recent pretrained visual representations (e.g., DINOv3 and SigLIPv2) are trained on extremely large-scale datasets, and (2) we can easily use open-source image encoders.

2. The authors fixed the batch size at 256 for their experiments, but I think that increasing the sampling times K can have a similar effect to enlarging the batch size. In fact, a previous study [1] has shown that applying augmentations to the same batch can yield better performance with fewer iterations. Therefore, it is needed to verify how much of the improvement in the proposed method comes from this effect, e.g., by training SiT with a batch size of 512, or using sampling times K with only a flow matching loss.

[1] Hoffer et al., Augment Your Batch: Improving Generalization Through Instance Repetition, CVPR 2020

### Questions
Please answer the Weaknesses.

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
5

### Summary
This work (DUPA) first points out the internal issues brought by REPA: 
(1) Out of distribution. and (2) Huge additional computational costs. 
Then inspired by the idea of DDT, SD-DIT and Contrastive FM works, the authors propose DUPA to directly align  two noisy views of a single image without external supervision. And such self-alignment (unsupervised alignment)  can significantly accelerate the convergence of SiT.

### Strengths
1. The proposed DUPA is simple but effective, without the reliance of external ViTs, like dinov2.
2. The author’s writing is very straightforward and concise, without storytelling or beating around the bush.
3. Clear convergence of SiT is brought by DUPA.
4. Sufficient experiments .

### Weaknesses
1. More discussion and analysis about: why such self-supervised alignment could work for the convergence of SiT?
for example in SD-DiT[2], the choice of t=min (mostly close to pure image) is the most effective acceleration technique. And how about DUPA?
2.  Recently there are some works [1][2] like DUPA, focusing on the self-alignment DiT, please claim the difference/advantage/difsussion compared with DUPA  and add the corresponding reference.

[1] Jiang, Dengyang, et al. "No Other Representation Component Is Needed: Diffusion Transformers Can Provide Representation Guidance by Themselves." arXiv preprint arXiv:2505.02831 (2025).
[2] Zhu, Rui, et al. "Sd-dit: Unleashing the power of self-supervised discrimination in diffusion transformer." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.

### Questions
See weakness.

I am very willing to raise my ratings if you can provide sufficient discussions mentioned in Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper present DUPA, a self supervised approach for representation alignment for boosting the convergence speed of diffusion transformers. The core idea is to utilize representations from a parallel branch which has the input of the same image but at a different noise level Such a mode of training would force the model to learn noise robust features faster leading to faster convergence. Experiments show an improvement of training speed by 5x and the model achieving similar performance to REPA.

### Strengths
1. The idea of utilizing augmented versions of the same image as input and aligning their representations is a clever approach for representation alignment. Such a mode of training may scale better for text to image training when compared to a vision encoder that introduces an inductive bias
2. The idea is novel and solid and the authors have performed extensive experiments to find the suitable layers and hyperparameters for DUPA. 
3. The paper is well written and easy to follow.

### Weaknesses
1. Is there some contraints on the value of the independently sampled timesteps needed for better performance? As an example, assume that one timestep is sampled with maximum noise and the other at minimum noise, aligning their representations might be a case where training with DUPA loss might not leading to a meaningful solution. 
2. Would performing DUPA on multiple layers at the same time lead to a better performance? 
3. I think the claim regarding 10x inference speed may be a bit misleading. Usually diffusion models are utilized to obtain a few samples. I believe with the current setup, DUPA will portray similar sampling speeds to REPA. I would advise the authors to correct this wording.

### Questions
1. Aside from the cosine similarity loss similar to REPA, would a simple MSE loss work for DUPA? In the case of REPA the cosine similarity loss may make sense. But is it the same case here, since the features of the same network are aligned?
2. Could the authors provide a comparison between REPA and DUPA for text to image generation ? 
3. I'm rating the paper as borderline accept now, mainly because this approach seems scalable for text to image generation on the first look. ,but I'm willing to improve my rating if the authors can address the questions in a satisfactory way.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes DUPA (Dual-Path Condition Alignment), an unsupervised representation-alignment framework for training diffusion transformers. Building upon the decoupled diffusion transformer (DDT), DUPA introduces a condition alignment loss that aligns features of multiple noisy versions of the same image, effectively mimicking the self-supervised contrastive learning.
Experimental results on ImageNet 256×256 demonstrate that DUPA outperforms the reproduced DDT baseline both in terms of training and sampling efficiency.

### Strengths
- The proposed DUPA not only achieves better FID than DDT (reproduced) with the same number of training steps but also reduces the number of denoising steps for sampling. 
- The design of DUPA is simple and easy to integrate.

### Weaknesses
- The experiments are limited to ImageNet 256×256.
- While the paper is clear in structure, the prose is dense and overly formal in places, which introduces unnecessary friction. For example, in lines 192–194:
	- there are too many dependent clauses. "thereby generating... to be denoised". 
	- Unnecessary formality. "conduct multiple samplings to get different..." is verbose. "We sample multiple noises.." will be more natural. 
	- Grammar issues: "independent sampling times" -> "independent samples"

### Questions
- What's the main difference between DDT and the dual-path sampling in Table 4? Aren't they the same without DUPAlign loss? 
- Why does DUPA not integrate the architectural improvements of DDT? Would DUPA still retain its advantage if those were included?
- How does DUPA work on higher resolution like ImageNet512?

### Soundness
3

### Presentation
1

### Contribution
2
