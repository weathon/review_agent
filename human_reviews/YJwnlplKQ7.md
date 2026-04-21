# MarDini: Masked Autoregressive Diffusion for Video Generation at Scale

- Avg Score: 5.50
- Decision: Reject
- Scores: 5, 6, 5, 6

## Abstract
We introduce MarDini, a new family of video diffusion models that integrate the advantages of masked auto-regression (MAR) into a unified diffusion model (DM) framework. Here, MAR handles temporal planning, while DM focuses on spatial generation in an asymmetric network design: i) a MAR-based planning model containing most of the parameters generates planning signals for each masked frame using low-resolution input; ii) a lightweight generation model uses these signals to produce high-resolution frames via diffusion de-noising. MarDini’s MAR enables video generation conditioned on any number of masked frames at any frame positions: a single model can handle video interpolation (e.g., masking middle frames), image-to-video generation (e.g., masking from the second frame onward), and video expansion (e.g., masking half the frames). The efficient design allocates most of the computational resources to the low-resolution planning model, making computationally expensive but important spatio-temporal attention feasible at scale. MarDini sets a new state-of-the-art for video interpolation; meanwhile, within few inference steps, it efficiently generates videos on par with those of much more expensive advanced image-to-video models.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper integrates masked auto-regression (MAR) and diffusion model (DM) in a ControlNet-like fashion, combining the efficient generation capability of MAR with the high-quality generation power of DM to achieve video generation conditioned on any number of masked frames. The network architecture adopts an asymmetric design, allocating most of the parameters to the MAR branch and using spatial-temporal attention. The MAR branch takes masked low-resolution frames as input and outputs conditioning signals for the DM network. The DM network, with fewer parameters and utilizing only spatial and temporal separated attention, generates masked high-resolution frames. Under the influence of the conditioning, the DM network generates the missing masked parts. This approach achieves state-of-the-art video interpolation and image-to-video performance.

### Strengths
1. This paper proposes a computationally balanced asymmetric network architecture. The MAR part has a larger number of parameters and higher computational complexity, but it requires fewer forward passes and takes low-resolution frames as input. In contrast, the DM part has fewer parameters and lower computational complexity, but requires more forward passes and models high-resolution frames. This is an ingenious approach.

2. The paper introduces Identity Attention, which cleverly addresses the issue of unstable training and theoretically reduces the computational load of the attention module (although it is unclear whether the authors actually implemented this in the code).

### Weaknesses
1. This paper does not provide sufficient ablation experiments to demonstrate the effectiveness of the progressive training strategy. At the very least, experiments are needed to show that under the same training cost, the combination of Joint-Model Stage + Joint-Task Stage outperforms the model with only the Joint-Task Stage.

2. In Table 3, the paper does not compare its results with SEINE [1], which, as far as I know, also focuses on video interpolation. 

3. Although the proposed method offers some advantages in inference latency, it introduces a large number of parameters compared to previous diffusion methods, which increases the memory requirements of the device. However, Table 3 lacks a direct comparison of memory usage.

4. Despite the introduction of numerous additional parameters, the image-to-video performance in Table 4 still lags behind previous work [2, 3].

[1] Xinyuan Chen, Yaohui Wang, Lingjun Zhang, Shaobin Zhuang, Xin Ma, Jiashuo Yu, Yali Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. Seine: Short-to-long video diffusion model for generative transition and prediction. In Proceedings of the International Conference on Learning Representations (ICLR), 2023.

[2] Jinbo Xing, Menghan Xia, Yong Zhang, Haoxin Chen, Xintao Wang, Tien-Tsin Wong, and Ying Shan. Dynamicrafter: Animating open-domain images with video diffusion priors. In Proceedings of the European Conference on Computer Vision (ECCV), 2024.

[3] Zuozhuo Dai, Zhenghao Zhang, Yao Yao, Bingxue Qiu, Siyu Zhu, Long Qin, and Weizhi Wang. Fine-grained open domain image animation with motion guidance. arXiv preprint arXiv:2311.12886, 2023b.

### Questions
1. Directly using a pre-trained VideoMAE to initialize the MAR branch may sound more elegant and efficient for several reasons:

    ① Leverage Pre-trained Features: VideoMAE is pre-trained on a large corpus of video data and has already learned spatiotemporal features. Initializing the MAR branch with VideoMAE would allow the model to start with a strong foundation, potentially leading to faster convergence and better performance, especially for tasks related to video generation.

    ② Reduced Training Time: By starting with a pre-trained model, the MAR branch wouldn't need to learn basic spatiotemporal representations from scratch, reducing the overall training time and computational cost.

2. Table 4 has typos in the last two rows: "MARDini-L/T-17" and "MARDini-S/T-17" should be corrected to "MarDini-L/T-17" and "MarDini-S/T-17" to maintain consistent naming conventions. And these two lines are missing the Latency metric.

[4] Zhan Tong and Yibing Song and Jue Wang and Limin Wang. VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training. Advances in Neural Information Processing Systems, 2022.

[5] Wang, Limin and Huang, Bingkun and Zhao, Zhiyu and Tong, Zhan and He, Yinan and Wang, Yi and Wang, Yali and Qiao, Yu. VideoMAE V2: Scaling Video Masked Autoencoders With Dual Masking. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023

### Soundness
2

### Presentation
3

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
The paper proposes a novel masked autoregressive diffusion model architecture that leverages a planning component which processes downsampled masked videos to learn high-level + long-term dependencies, coupled with a smaller diffusion model that learns high-resolution spatial dependencies. In addition, the authors a progressive training strategy for better training stability.

### Strengths
* The paper is generally well-written, and clear
* The architecture is simple, and a smaller diffusion head can help speed up inference by caching MAR outputs.
* Experimental results show better performance in various masked generation tasks, such as image-video generation and video interpolation, as well as ablations to show benefits of their model design

### Weaknesses
* I am not yet convinced of the scalability of this model. From Table (4), it seems that the “L” variants of MarDini perform worse than the “S” variants? Is this correct?
* Related to the above point, would this model have a harder time scaling to learn distributions that are more stochastic? Since a vast majority of parameters are in MAR and not in the diffusion model. To leverage the greater number of params in the MAR, the MAR would need to encode as many possible futures into its representation for the diffusion model to read from, which may be difficult if the complexity of the distribution grows (unconditional video generation where the MAR output is not useful / constant, or simultaneously predicting / interpolating longer videos).
* The training recipe is a little bit complicated, requiring a combination of stages such as separate / joint training, and different masking ratios.

### Questions
* Do you have video samples with more deformable motion (e.g. human running / doing stuff, or animals moving around)? Most examples shown in the supplementary show motion related to fire / water / clouds, camera translations, or object translation - of which are generally the easier things to learn for a video generation model.
* Instead of the initial training stage, would it possibly work if the model was jointly trained from scratch if the MAR simultaneously optimized the MSE loss? Could the diffusion model also be stabilized if cross attention was initialized to zero / a zero-init scaling term was learned (if that was not done already)?
* Is masked autoregression used during generation? E.g. similar to MaskGit, where in this case you would progressively sample subsets of frames following some schedule If not, what is the masked autoregression aspect of the model (what is autoregressive here)?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The authors propose a method for image animation, video interpolation and extrapolation. The novelty of the method lies in the use of a small diffusion model (DM) augmented by the features representations (planning signals) produced by a large masked autoregressive (MAR) model operating in low resolution. By allocating a large amount of compute to the MAR model, diffusion sampling can happen efficiently, as the MAR model needs to be executed only once. The proposed research question of improving the efficiency of DMs is of large interest to the community, though the current work only tackles image animation, video interpolation and extrapolation, without treating the most widespread text-to-video scenario. The paper is mostly clearly written. The high level method idea is sound, though several architectural and training recipe details are not fully justified, reducing soundness of the approach.

### Strengths
- The authors treat the relevant problem of improving efficiency of DM-based video generators
- The proposed idea of using a MAR model as a feature extractor is interesting
- The paper is well-written, though many choices are not fully justified
- Ablations show that the small DM benefits from access to MAR features, validating the design choice.

### Weaknesses
- Absolute quality of the generated videos as shown in the supplementary material is low
- Tab 1. presents evaluation of the method vs a DM with same compute and no MAR. A missing baseline would be comparing the full approach with a DM of a capacity rescaled to match the compute of DM + MAR. Without such baseline it is not possible to assert that the combination of DM + MAR outperforms DM under equal computational constraints.
- Several details in the architecture and training recipe are unclear or appear not sound (see questions)
- Fig 5 (c) shows bad convergence behavior, even after the proposed fixes. The behavior is justified by the use of warmup. Warmup is unlikely to generate a sharp transition at 6k steps as the one shown in the plot which appears to be more likely justified by optimization challenges for the current architecture, raising doubts on how scalable and user-friendly the proposed framework would be.
- In Tab. 3. The model underperforms LDMVFI and VIDIM in MidF-LPIPS and FID. A discussion in LL430 attributes this to defective metrics behavior with respect to blur. This appears unconvincing as, differently from PSNR and SSIM, LPIPS and FID would strongly be penalized by the presence of blur. In particular, FVD, would have a behavior substantially similar to FID, being both distribution based metrics. Lower performance under these metrics raises concerns about the quality of the method. The sample in Fig. 6 does not convincingly show the author's point. I would suggest showing multiple samples from regions containing large amounts of high frequency details where differences would appear more clearly.
- Text to video generation is not treated in this work
- (minor) The employed VAE does not perform temporal compression, contrary to modern T2V generators performing 4x (Cogvideo-X) to 8x (MovieGen) temporal compression. This appears as a lost opportunity for speedup. Note that temporal compression still allows for masking on each frame independently, by applying masks to the input video before encoding.

### Questions
- The model naturally lends itself to the tasks of video inpainting. I.e. considering frames with partial masks. Did the authors consider such scenario? Considering partial masks would enable model training on larger datasets of images that would improve performance.
- LL123-126 highlight using only videos without labels as a model advantage. The model however shows low quality video generations that could have been improved through joint image/video training and considering text captions that can be automatically generated for the training data. Could the authors discuss why the chosen approach is desirable?
- Why does the MAR model present a Spatial-only attention block? Modern T2V generators tend to use spatial blocks only as a way to preserve generation capabilities if the model was retrained on image-only data. Why would the MAR model benefit from Spatial-only attention blocks that would have been traded for more Spatio-Temporal blocks for which importance is advocated in the paper?
- Why does the Temporal attention block in the DM not have modulation applied to it?
- Why does modulation involve shifting? As modulation is applied after RMSNorm operations it would be natural to only consider scale operations. This would come at the benefit of reducing the amount of parameters devoted to modulation by 33%
- LL194 mention the use of Layer Normalization. Is layer normalization used to perform QK normalization? Why do the authors not employ RMSNorm for QK normalization as previous work (e.g. Stable Diffusion 3)? This would be a more logical choice given the use of RMSNorm layers as the main normalization mechanism.
- LL200-203 mention the use of [NEXT] tokens as Lumina-T2X (Gao 2024). In their follow up work Lumina-Next, the authors advocate for the removal of the mechanism in favor of 3D RoPE. Why was this approach adopted in this work?
- LL345 could the authors clarify how FSDP would contribute to inference speedups? FSDP would only allow sharing of model parameters to reduce memory usage, but would require additional parameter gathering overhead to achieve this. Please elaborate
- LL292 does not appear to be well justified. Could the authors describe why it would not be more desirable to train the DM from scratch with such architecture?
- In Fig. 7, the discussed training recipe for the "Initial Stage" of the DM comprises continuous changes in the set of training data (20M Clips -> 80M Clips -> 40M Clips) and FPS (8 -> 4 -> 8). The MAR is trained on 200M clips. Could the authors describe the rationale for adopting the chosen strategy and discuss how readers should instantiate their training recipe in practice?
- The proposed idea of using a MAR model as a feature extractor is interesting. The MAR model needs to only be run a single time during the whole DM sampling process. However, single-step distillation (see "SF-V: Single Forward Video Generation Model") of a large video DM also appears as viable route for achieving good quality-cost tradeoff with a single evaluation of a large model. Could the authors describe why MAR + small DM represent a more attractive solution than large video DM + single-step distillation?

### Soundness
2

### Presentation
3

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
This paper presents a new framework for video generation, MarDini, which integrates masked auto-regression (MAR) and diffusion model (DM). Specifically, it separates the low-res planning and high-res generation loads into a heavy-weight but efficient MAR and a lightweight but expensive DM, effectively balancing the computation of scaling video generation. To stabilize training, the paper also proposes Identity Attention, which preserves the input reference frames without attending to other tokens, and a multi-stage training strategy. The method has been evaluated for video interpolation, image-to-video generation, and video expansion. Visualization is provided in the supplemental materials.

### Strengths
1.	This paper presents a novel video generation framework that separates low-res temporal planning and high-res spatial generation via asymmetric MAR and DM models, effectively balancing the computation cost in this task. In general, I do agree with the three characteristics/strengths of MarDini listed in the Introduction, i.e., the flexibility, scalability, and efficiency advantages.
2.	The large flexibility of the proposed framework offers huge room for exploration, not only from the architecture side but also the applications (e.g., as shown in the Appendix, the model can potentially be generalized to other tasks such as zero-shot 3D view synthesis and video expansion). This paper can likely inspire future research in many domains.
3.	In-depth analyses are presented to reveal the model’s behavior and justify the design choices, nicely supporting the arguments.
4.	The paper has set new SoTAs on VBench for image-to-video generation and VIDIM-Bench for zero-shot video interpolation, further justifying the effectiveness of the proposed framework.
5.	Overall, I think this paper has been nicely written, methods, results, and visualizations are clearly presented.

### Weaknesses
Overall, I am happy with this paper, but considering the scale of experiments (256 H100 GPUs for training), which is very large, there are a few concerns I wish to point out:

1.	Video length. It seems that the method is only evaluated on very short videos, mostly generating 9 or 17 frames at low FPS. It is unclear that the method, in particular the temporal planning, can generalize well to longer videos.
2.	Conditioning. The paper claims that the method does not require image-based pre-training, but I guess this is mainly because the paper has chosen to address the video interpolation, video expansion, and image-to-video generation tasks. As mentioned in Future Works, if we want to enable the model to do text-conditioning, then we might have to introduce abundant image data to learn the rich semantics. It is unclear whether the model can still work well with text conditions (or condition signals in other modalities) and generate long videos, especially when the DM is very lightweight.
3.	Training difficulty. I am happy to see that the paper deeply investigates the training strategy and found feasible solutions. But this also makes me concerned about the stability and training difficulty of such a model, especially since this might be very unfriendly for future works that try to follow or extend this MarDini.
4.	Visualizations. I carefully reviewed all the videos provided in the Supplemental materials, but most of the videos are very short and static, and are scenery or single objects that are relatively easy to generate. For such a large-scale model (3.1B parameters, trained with 256 H100), I expect to see much more complex and successful examples such as high dynamics, complex scenes, or human activities. Additionally, I am not fully satisfied with the Figure 4 videos that compare with/without planning. In this (picked) example, buildings are much better for with planning, but the ferry still has obvious distortion.

### Questions
I don’t have any other questions. I hope that the authors can concisely respond to my concerns mentioned above.

### Soundness
3

### Presentation
4

### Contribution
3
