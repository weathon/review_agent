# Effortless Event-Augmented Latent Diffusion for Video Frame Interpolation

- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Latent Diffusion Models have advanced video frame interpolation by generating intermediate frames between input frames. However, effectively handling large temporal gaps and complex motion remains challenging, often leading to artifacts. We argue that event camera signals, with their ability to capture continuous motion at high temporal resolutions, are ideal for bridging these temporal gaps and enhancing interpolation precision. Given the impracticality of training an event-assisted model from scratch, we introduce a novel adapter-based framework that seamlessly and effortlessly integrates high-temporal-resolution cues from event cameras into pre-trained image-to-video models without modifying their underlying structure. Our method leverages Image Warped Events (IWEs) and bidirectional sparse optical flow for precise spatial and temporal alignment, significantly reducing artifacts and improving interpolation quality. Experimental results demonstrate that our event-enhanced interpolation achieves superior accuracy and temporal coherence compared to existing state-of-the-art methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper aims to integrate event camera signals into pre-trained Video Latent Diffusion Models for Video Frame Interpolation. This paper proposes a framework that could integrate event signals into pixel-based pre-trained video diffusion models. It consists of an IWE encoder, which encodes IWE into the latent diffusion model, and an Alignment Adapter to align latent features using flows. The IWE and optical flows are obtained from event streams using contrast maximization. The paper also provide a novel synthetic event-video dataset, EvPexels.

### Strengths
1. Integrating different form of data representation (event streams) into pre-trained diffusion models is interesting, and the paper successfully addressed the problem.
2. Strong results compared to baseline methods.

### Weaknesses
1. Discussion on how event stream helps in this framework. I figure that the proposed method successfully integrates event stream into pre-trained (latent) diffusion models. However, I did not understand how this could greatly help for the task, other than mentioning that there has been prior work that showed the effectiveness of event streams in frame interpolation. This paper does use event streams, but converts them into optical flows and warps the features with those. How would this differ from integrating optical flows, put aside the IWE insertion part? I think discussing the necessity of using event streams could have been very helpful. In the abstract, the authors mention that event camera signals are ideal due to their ability to capture continuous motion at high temporal resolution, but I could not understand where this strength shows its effect in the proposed framework. The ablation study in the Appendix quantitatively shows its advantage, but I believe it is also very important to explain theoretically how/where/why the proposed design is useful, along with the limits / reason for failure of existing methods in comparison.
2. Novelty. To begin with, I really like the approach and found it interesting. However, when it comes to dissection of the proposed components, I found it a little weak in terms of technical novelty. 1) Converting event streams to IWE and flows rely on contrast maximization, an existing method. 2) IWE encoder is a sequence of 3D convolutional layers, to encode and inject IWE signals. 3) Alignment Adapter sounds quite similar to feature warping and aligning which have been conducted in the field of VFI [1,2,3]. 4) Adapting LoRA into training for efficient tuning is a commonly used technique, especially when it comes to adapting a pre-trained model for different purposes [4], as in this paper’s scenario. Integrating existing techniques appropriately can be considered a contribution, but with respect, one could find it a little weak in terms of technical novelty. In case of any possible misunderstandings I have made, please point me out.

My main reason for the rating is W1, on discussing why this problem is important, and theoretical explanation / analysis on how their approach helps.

[1] Niklaus, Simon, and Feng Liu. "Context-aware synthesis for video frame interpolation." CVPR 2018

[2] Niklaus, Simon, and Feng Liu. "Softmax splatting for video frame interpolation." CVPR 2020

[3] Sim, H., Oh, J., and Kim, M. "Xvfi: extreme video frame interpolation." ICCV 2021

[4] Zhang, Lvmin, Anyi Rao, and Maneesh Agrawala. "Adding conditional control to text-to-image diffusion models." ICCV 2023.

### Questions
1. I did not quite understand how you actually add intermediate frames. Like, you get N frame latents, and how do you add the N x 23 intermediate frames’ latents to be generated? Is it simply adding Nx23xHxW tokens in between the existing frame tokens?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses video frame interpolation (VFI) using latent diffusion models, focusing on challenges like large temporal gaps and complex motion that cause artifacts. The authors propose integrating event camera signals, which capture continuous high-temporal-resolution motion, into a pre-trained Diffusion Transformer (DiT)-based video model. Their method extracts Image Warped Events (IWEs) and bidirectional sparse optical flow from event streams via contrast maximization, then injects these cues using lightweight adapters: an IWE encoder for spatial structure and a flow-based alignment-and-fusion adapter for temporal consistency. A synthetic event-video dataset (EvPexels) with 1,100 scenes is introduced. Experiments on real and synthetic datasets show improved interpolation quality over state-of-the-art methods in metrics like LPIPS, FID, and FVD.

### Strengths
-The motivation of incorporating event-derived signals (IWEs, optical flow) into a pre-trained DiT model via lightweight adapters is well explained. This is reasonable because the two extracted pieces of information directly correspond to the challenges in previous VFI methods (large temporal gaps and complex motion that cause artifacts).

-The proposed EvPexels dataset, with 1100 diverse motion-rich scenes, addresses the scarcity of large-scale paired event-video data and supports training and benchmarking. If the dataset and tools will be released, it will promote future research in event-driven video generation.

### Weaknesses
-I believe the paper lacks experiments that directly compare the proposed motion information extraction with other event‑based conditioning methods under the same baseline. I trust that the ablations in Table 4 show that the designed conditions work, but it is still necessary to explain their advantages over other comparable approaches to further justify the idea.

-The EvPexels dataset is introduced but lacks detailed analysis of its motion diversity, scene types, and potential biases, which are crucial for assessing generalization. Then, the dataset split (training/test) for EvPexels is not specified, raising concerns about evaluation fairness and overfitting. Since the authors present the data as a contribution, they should conduct a comprehensive analysis and evaluation of the proposed new dataset to demonstrate that it indeed has value for advancing research in this field.

### Questions
-I recommend using the experimental setting in Table 4 as the baseline and attempting to use other event based conditioning to guide DiT for VFI, comparing against the proposed approach. I also suggest that the authors provide a more detailed review of conditioning methods that may work, in order to highlight the originality of the proposed design.

-I hope the authors can provide more details about EvPexels, including the acquisition method, motion patterns, scene types, and dataset splits. It would be preferable to also include analyses showing why this dataset is important to the field and in what aspects it enables evaluation or training that other datasets cannot.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses video frame interpolation using event camera signals as control. It extracts both the IWE representation and optical flow from the event data. The IWE representation is concatenated with the noised video latents and fed into the diffusion denoising network, while the optical flow is used to warp features of neighboring frames, aligning them with the original DiT features. This design enables the model to incorporate explicit motion guidance for more accurate interpolation.

### Strengths
The paper tackles a well-defined problem—event-based video frame interpolation. Compared to earlier event based video frame interpolation,  this work uses both IWE representation and optical flow extracted from the event signal to adapt to the  video inbetweening diffusion model.

### Weaknesses
Compared to prior work, such as VDM-EVFI (Chen et al., 2024), the proposed method adopts a different video diffusion backbone and a new strategy for incorporating event representations into the network.

To fully demonstrate the effectiveness of the proposed approach, it would be fair to include a comparison with a version of VDM-EVFI adapted to the same video diffusion model (Wan 2.1 FLF2V) used in this paper. Without such a comparison, it is difficult to determine whether the performance improvement over VDM-EVFI stems from the new event representation or simply from using a stronger diffusion backbone.

I am also confused about the evaluate setting in the paper, please see Questions section.

### Questions
1. I did not fully understand why the method is evaluated on datasets such as DAVIS and Pexels, which do not contain event signals for interpolation. What is the evaluation setting when comparing on these datasets? Why not focus solely on event-based VFI datasets instead?
2.  It is also unclear why the ablation studies are conducted on Pexels and DAVIS rather than on BS_ERGB, which is an event-based interpolation dataset.
3. Which layers in the DiT architecture use the alignment adapter with optical flow? Did the authors conduct any ablation studies on this component?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes an adapter-based framework that integrates event camera signals into a pre-trained DiT-based latent diffusion model for video frame interpolation. The authors extract Image Warped Events (IWEs) and bidirectional sparse optical flow via contrast maximization and inject them into the latent space through two lightweight adapters: an IWE encoder for spatial structure and an alignment-and-fusion module for temporal consistency. Experimental results on several datasets (BS-ERGB, DAVIS, Pexels) show strong perceptual metrics and competitive reconstruction accuracy compared to prior diffusion-based and event-guided methods.

### Strengths
- The authors explore to inject event signal into pre-trained DiT-based model for video frame interpolation.
- The introduction of the EvPexels dataset could be a useful contribution for benchmarking event-based video tasks.
- The proposed method outperforms the baselines and achieves competitive visual results according to the experiments provided.

### Weaknesses
- Unfair comparison with prior event-based baselines. Most previous event-based generative methods, such as *VDM-EVFI*, are built upon the U-Net–based *Stable Video Diffusion (SVD)* backbone, whereas the proposed approach leverages the more advanced *Wan2.1* DiT architecture. This discrepancy makes the direct comparison somewhat unfair, as it is difficult to disentangle the performance improvements brought by the proposed event integration from those inherited from the stronger base model.
- Limited qualitative comparisons. The paper provides only a single qualitative example comparing the proposed method with prior works in Figure 3, and notably omits key baselines such as *VDM-EVFI*. Additional qualitative results would be valuable for demonstrating the visual benefits of the proposed method, especially under challenging motion scenarios.
- Lack of comparison with non-generative VFI methods. While traditional non-generative interpolation models may struggle with large temporal gaps, including one or two representative non-generative baselines (e.g., those used in *VDM-EVFI*) would help clarify the practical advantages and trade-offs of the proposed generative approach, making the evaluation more comprehensive and contextually grounded.

### Questions
- How do the authors ensure a fair comparison given that most event-based baselines (e.g., *VDM-EVFI*) are built upon the U-Net–based *Stable Video Diffusion (SVD)*, while the proposed method uses the more powerful *Wan2.1* DiT architecture?
- Have the authors conducted any ablation or control experiment using the same backbone (e.g., Wan2.1 without event integration) to isolate the contribution of the proposed event adapters?
- Why do the qualitative comparisons include only one example, and why is *VDM-EVFI*—a closely related event-guided diffusion baseline—excluded from the visual comparisons?
- Could the authors provide additional qualitative results on diverse or challenging motion scenes (e.g., occlusion, or high-speed motion) to better demonstrate the strengths and limitations of the proposed method?
- Have the authors considered including one or two representative non-generative VFI baselines for completeness, as the VDM-EVFI did?

### Soundness
2

### Presentation
2

### Contribution
2
