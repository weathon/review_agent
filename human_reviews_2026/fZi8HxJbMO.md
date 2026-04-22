# InfVSR: Breaking Length Limits of Generic Video Super-Resolution

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 4, 4, 6

## Abstract
Real-world videos often extend over thousands of frames, posing unique demands far beyond current short benchmarks. Existing video super-resolution (VSR) approaches, however, face two persistent challenges when processing long sequences: (1) Efficiency due to the heavy cost of multi-step denoising for full-length sequences; and (2) Scalability hindered by temporal decomposition that causes artifacts and discontinuities. To break these limits, we propose InfVSR, which novelly reformulate VSR as an autoregressive-one-step-diffusion paradigm. This enables streaming inference while fully leveraging pre-trained video diffusion priors. First, we adapt the pre-trained DiT into a causal structure, maintaining  both local and global coherence via rolling KV-cache and joint visual guidance. Second, we distill diffusion process into a single step efficiently, with patch-wise pixel supervision and cross-chunk distribution matching. Together, these designs enable efficient and scalable VSR for unbounded-length videos. To fill the gap in long-form video evaluation, we build a new benchmark tailored for extended sequences, and further introduce semantic-level metrics to comprehensively assess temporal consistency. Our method pushes the frontier of long-form VSR, achieves state-of-the-art quality with enhanced semantic consistency, and delivers up to 58x speed-up over existing methods such as MGLD-VSR. Code will be released soon.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes InfVSR to tackle the difficulty of unbound-length real-world video sequences. By reformulating VSR as an autoregressive-one-step-diffusion paradigm, it supports efficient streaming inference. To maintain identity consistency across multiple chunks, it intrudces joint visual guidance and cross-chunk distribution matching to incorporate high-level semantic information. Also, it utilizes a two-stage curriculum learning paradigm for efficient training. Experiments demonstrate the effectiveness and efficiency of the proposed method.

### Strengths
* By drawing inspiration on current success in long video generation, it proposes an efficient streaming inference paradigm by autoregressive one-step diffusion model.
* It proposes joint visual guidance and cross-chunk distribution matching to maintain identity consistency across multiple video chunks.
* Detailed experiments demonstrate its effectiveness and efficiency.

### Weaknesses
* For a low-level vision task, I am skeptical about the actual effectiveness of the two high-level strategies (joint visual guidance and cross-chunk distribution matching) emphasized in the model. Authors need to supplement super-resolution results related to portrait or object identities to demonstrate their effect of maintaining consistency across multiple chunks.
* For the patch-wise pixel supervision technique in Sec 3.3, it is well known that directly decoding patch latent of smooth area to pixel space often leads to flickering results due to the reconstruction ability of VAE decoder. Have the authors observed the same phenomenon? If so, will the flickering output affect the optimization? 
* At the stage 2 of curriculum learning, low-resolution videos are used for training. Will the low-quality ground truth in stage 2 affect the overall VSR quality?
* InfVSR only utilizes 1K clips from REDS dataset for training. However, previous VSR methods often use 100K~1M video clips to improve its generalizability. Authors need to demonstrate the effectiveness on more diverse test sets or argue why this method is data-efficient.

### Questions
* Typo: ``loacl'' in line 182.
* Section layout in line 238, 476 and other places are largely changed, which is forbidden in ICLR rules.
* Authors are strongly recommended to provide a video demo or video files corresponding to the results presented in the paper. The absence of video results may leads to a lower score.

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
5

### Summary
This paper introduces InfVSR, an autoregressive-one-step diffusion (AR-OSD) framework for video super-resolution (VSR). Unlike prior diffusion-based VSR models that are limited to short clips due to high memory and computational cost, InfVSR reformulates VSR as a causal autoregressive process with single-step diffusion inference. Building upon the state-of-the-art T2I diffusion model, i.e., WAN2.1-.3B, 
The key modifications include:
- A causal DiT architecture featuring rolling KV-cache for local temporal smoothness and joint visual guidance for global coherence.
- A training scheme combining patch-wise pixel supervision (for efficient high-resolution detail recovery) and cross-chunk distribution matching (for long-range temporal consistency).
- A new MovieLQ benchmark of 1000-frame real-world videos and semantic-level temporal metrics (BC, SC, MS from VBench) for long-form consistency evaluation.

### Strengths
- The paper is easy to follow.
- The paper presents a successful practice to build an autoregressive VSR model based on a pretrained T2I model, which has not become a major trend for current diffusion-based VSR models.
- The proposed new benchmark, MovieLQ, may facilitate further evaluation for future works.

### Weaknesses
- The fundamental insights of this paper are somewhat incremental. It is more like an extension of existing technologies for VSR. Specifically, the key components used in the paper, including KV-cache, causal DiT,  DMD loss, multi-stage training, etc. These make the proposed method seem kind of trivial.
- The paper lacks an in-depth analysis of the proposed components. For example, it is well-known that casual attention may have its drawbacks compared with widely-used full attention, especially for large-scale generative models. While the paper presents positive numbers, it is unclear if the claimed improvement benefits from the proposed components or simply from the improvement of the used generative prior. After all, Wan1.3 is already much stronger than the previous generative priors used in the baselines. The theoretical analysis behind the claimed improvements is vague. Specifically, how does the autoregressive manner affect the performance of VSR compared with the non-autoregressive one? How does causal attention affect the performance of VSR?

### Questions
My concerns are as follows:

1. The two major weaknesses above.

2. In lines 251-260, if my understanding is correct, the paper proposed to calculate the loss between the decoded, cropped latent tensors and the cropped ground-truth in the pixel space. Given the padding operations as well as the receptive field of the CNN layers in the VAE, such supervision may introduce problems on the edges of the tensors.

3. The author should provide more details on the proposed test benchmark MovieLQ, including how the data is collected and why it is suitable to be a benchmark for VSR test.

4. It is unclear how sensitive the proposed autoregressive model is to the size of the KV-cache. Moreover, since the proposed method claims to target at very long videos, it is also unclear if the fixed-size memory can handle long-term content given an extra long video with frequent scene changes.

### Soundness
2

### Presentation
3

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
The authors propose a new VSR framework, InfVSR, to handle long videos efficiently. InfVSR uses an autoregressive-one-step-diffusion (AR-OSD) paradigm for temporal modeling.

### Strengths
### Motivation
- It is natural to introduce autoregressive sampling idea from video generation to generative VSR task, especially given that most of the VSR models are quite heavy to run.

### Method
- Rolling KV cache and joint visual guidance is a well-designed recipe for both local and global consistency. 
- DMD loss helps semantic consistency for long video clips.
- It is quite economic to train InfVSR as it only takes 4 A800-80G GPUs - much affordable compared to other baselines such as SeedVRs. 

### Experimental results
- The paper focuses its evaluation on MovieLQ, which consists of long videos with real-world degradations.
- InfVSR outperforms  previous methods on many metrics. 
- InfVSR shows good efficiency compared to diffusion-based methods. 

### Writing
The paper is well-written and easy to follow.

### Weaknesses
### Method
- Patch-wise supervision is common in many classic regression-based VSR papers, such as BasicVSR series. Although it is good to adopt it for latent space diffusion models, it is not quite clear to me to claim it as a novel technique. 
- The local temporal loss does not make sense to me as the local temporal dynamics between two adjacent frames could be very abrupt.  For example, a very large motion between two frames. Without a good alignment (e.g., flow-based warping), it might be harmful for the training. Please feel free to correct me.  

### Experimental results
- It is **extremely challenging** to see the performance of the proposed method, especially **temporal consistency**, without any video results shown in the supplementary results. Unfortunately at this moment I am inclined to reject because of this. 
- For VSR task, many commercial solutions prefer regression-based methods (e.g., RealBasicVSR and that line of works) for its fidelity and efficiency. How does the proposed InfVSR compare to those classic methods in terms of runtime?

### Questions
I'd strongly recommend authors to present some video results in the rebuttal stage, otherwise it is challenging to measure real performance of InfVSR.

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a novel method termed InfVSR, aiming to achieve efficient and temporally-scalable diffusion-based video super-resolution. InfVSR firstly adopts a pretrained DiT into causal structure and maintaining local and global coherence with rolling KV-cache, then distills the model with distribution matching to achieve one-step diffusion inference. This paper also proposes MovieLQ, a long-sequence video benchmark to evaluate the VSR in long-term semantic-level consistency, fidelity and efficiency.

### Strengths
1.	The idea is easy to follow. It adapts a pretrained T2V DiT-based diffusion model to causal form. Then applies distribution matching distillation to achieve one-step diffusion inference. This is effective and efficient for the model to perform unlimited length super-resolution video prediction together with the rolling KV-cache design.
2.	A good and straight-forward solution on long-sequence video super-resolution. Both semantic and pixel consistencies are maintained through DMD loss and pixel-level reconstruction loss.
3.	A new benchmark VideoLQ is proposed to evaluate the long-sequence video super-resolution task.

### Weaknesses
1.	The long-term consistency is not thoroughly discussed in the paper, e.g., what is the visual results and comparison between the SOTA methods after 10, 100 or 1000 frames? Also the visual results in the supplement seems not fidel to the GT in some text images, this seems to be brought by the generative instability. 
2.	The main idea of improving the efficiency of video model using DMD and causal structure is trivial since it has been adopted in many other video methods [1][2]. Some more discussions are needed to specify the novelty of this paper.
3.	The InfVSR is built upon the Wan T2V model, also there are models like SeeSR[3] using text to enhance the super-resolution results. Can model achieve better result with proper text prompt or guidance?
4.	The parameter compared with SOTA methods should be provided to better validate the efficiency of the proposed method.

[1] Self-forcing: bridging the train-test gap in autoregressive video diffusion. arXiv preprint arXiv:2506.08009
[2] Matrix-Game 2.0: An Open-Source, Real-Time, and Streaming Interactive World Model arXiv preprint arXiv:2508.13009
[3] Seesr: Towards semantics-aware real-world image super-resolution CVPR 2024

### Questions
Refer to the weaknesses. The visualization is not very persuasive. The novelty should be clearified with detailed discussion. How about using prompt to boost the SR performance and guide the semantic content?

### Soundness
3

### Presentation
3

### Contribution
3
