# GameSR: Real-Time Super-Resolution for Interactive Gaming

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 4, 4, 2

## Abstract
High-resolution gaming demands significant computational resources, with challenges further amplified by bandwidth and latency constraints in cloud gaming. Existing upscalers, such as NVIDIA DLSS and AMD FSR, reduce rendering costs but require engine integration, making them unavailable for most titles, especially those released before the introduction of upscalers. We present \textbf{GameSR}, a lightweight, engine-independent super-resolution model that operates directly on encoded game frames. The architecture of GameSR combines reparameterized convolutional blocks, PixelUnshuffle, and a lightweight ConvLSTM to deliver real-time upscaling with high perceptual quality. Extensive objective and subjective evaluations on popular games, such as \textit{Counter-Strike 2}, \textit{Overwatch 2}, and \textit{Team Fortress 2}, show that \system{} reduces cloud gaming bandwidth usage by 30--60\% while meeting target perceptual qualities, achieves real-time performance of up to 240~FPS, substantially outperforms existing super-resolution models in the literature, and reaches near-parity with DLSS and FSR \textit{without} accessing rendering engine data structures or modifying game source code, making \system{} a practical solution for upscaling both modern and legacy games with no additional development effort.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper introduces GameSR, a lightweight and engine-independent super-resolution (SR) model targeting both cloud and local gaming scenarios. The proposed architecture combines PixelUnshuffle for feature rearrangement, re-parameterized convolutions for efficient inference, and a ConvLSTM module to capture temporal consistency across frames. The system operates as a post-processing step between video decoding and display, allowing real-time upscaling of rendered frames without modifying game engine code. Experiments were conducted on three commercial games (CS2, OW2, TF2) comparing GameSR with DLSS, FSR, and several academic SR models. The results suggest that GameSR achieves perceptual quality and runs in real time.

### Strengths
1. The topic is interesting and relevant to practical gaming applications. 
2. The paper is evaluated on real commercial games instead of synthetic datasets, which adds realism. 
3. Results look good in both quality and runtime, and the paper is overall easy to read.

### Weaknesses
1. The problem scope is confusing. The paper frames the work around cloud gaming, but it is not clear whether the target is a cloud setting or local post-processing on a client. The deployment point and constraints specific to cloud gaming are not well defined.
2. The technical novelty is limited. The model mainly combines several existing techniques, such as PixelUnshuffle, ConvLSTM, and re-parameterization, and the connections between them are not clearly motivated. 
3. The claimed goals and the actual method do not really match. The paper emphasizes cloud deployment and lightweight design, but the model does not include any cloud-specific optimization, and the “lightweight” part is mostly limited to using PixelUnshuffle rather than any systematic efficiency strategy.

### Questions
1. The paper’s positioning is unclear. It claims to target cloud gaming, but the design does not include any cloud-specific optimization. At the same time, it suggests that the method also works locally, which essentially makes it a general SR approach without a clear focus. 
2. If the goal is indeed cloud deployment, the input format deserves more discussion. For example, is the model aware of or adapted to video compression structures such as key frames and delta frames? How is the streaming or codec aspect considered? 
3. The method appears to be a combination of existing techniques rather than a coherent new design. The components seem independent, with no evident co-design or joint optimization. 
4. The lightweight design lacks substantive innovation. PixelUnshuffle and re-parameterization reduce computation for sure, but these are standard tricks, not specific design choices tailored for this work. 
5. "In contrast, rendering at 540p and upscaling to 1080p using GameSR lowers utilization to ∼82%, reflecting reduced shading cost and lightweight inference." is not convincing. The reduction likely reflects idle cycles or a memory-bound condition rather than truly lighter computation. 
6. The evaluation uses three FPS games that are visually and structurally similar, which weakens the claim of generalization. 
7. The paper lacks comparison with the most recent game-oriented SR methods.

### Soundness
3

### Presentation
2

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
The paper proposes GameSR, a neural real-time super-resolution framework for video gaming. Compared to existing industry gaming SR frameworks like DLSS or FSR, the proposed GameSR network does not rely on access to engine-level data, so it does not require game developers to explicitly integrate with game engines and is more suitable for cloud-gaming settings. It leverages pixel shuffling for a lightweight feature learning and ConvLSTM for temporal learning. Results shows that the upsampling quality is on par with industry SR frameworks while the runtime performance is significantly faster than existing image/video SR networks.

### Strengths
1. Engine-data-free pipeline. Without relying on rendering engine data, the pipeline can be more broadly applicable across all kinds of games and cloud gaming scenarios. Especially for cloud gaming scenarios, it can greatly save the network bandwidth.
2. The proposed method achieves better runtime performance and on-par quality with SOTA methods.

### Weaknesses
## Major weaknesses
1. Missing citations. The real-time SR problem for gaming has been an important research direction in computer graphics literature, but the citation to several important works are missing here:
```
@inproceedings{zhong2023fusesr,
  title={Fusesr: Super resolution for real-time rendering through efficient multi-resolution fusion},
  author={Zhong, Zhihua and Zhu, Jingsen and Dai, Yuxin and Zheng, Chuankun and Chen, Guanlin and Huo, Yuchi and Bao, Hujun and Wang, Rui},
  booktitle={SIGGRAPH Asia 2023 Conference Papers},
  pages={1--10},
  year={2023}
}
@article{yang2023mnss,
  title={Mnss: Neural supersampling framework for real-time rendering on mobile devices},
  author={Yang, Sipeng and Zhao, Yunlu and Luo, Yuzhe and Wang, He and Sun, Hongyu and Li, Chen and Cai, Binghuang and Jin, Xiaogang},
  journal={IEEE transactions on visualization and computer graphics},
  volume={30},
  number={7},
  pages={4271--4284},
  year={2023},
  publisher={IEEE}
}
@inproceedings{zhang2024deep,
  title={Deep Fourier-based arbitrary-scale super-resolution for real-time rendering},
  author={Zhang, Haonan and Guo, Jie and Zhang, Jiawei and Qin, Haoyu and Feng, Zesen and Yang, Ming and Guo, Yanwen},
  booktitle={ACM SIGGRAPH 2024 Conference Papers},
  pages={1--11},
  year={2024}
}
```
2. Missing baselines. The paper only provides a limited comparison to industry SR in Tab 1, while in Tab 2 all baselines are not designated for real-time gaming but traditional image/video SR methods, which is unfair. The paper should also include a comprehensive comparison to real-time SR papers, such as Neural Supersampling (NSRR), FuseSR, Deep Fourier SR, Mob-FGSR, etc.
3. GPU utilization. The paper claims a lower GPU utilization rate to be an advantage over the industry's fully-utilized methods, which does not make sense. In the gaming industry, low GPU utilization rates typically mean a poor optimization in rendering, since game engines are supposed to fully utilize the GPU computation for a maximum frame rate boost. A low GPU utilization rate may indicate a performance bottleneck in the rendering pipeline, blocking the GPU from working fully, which is not desired.
4. Temporal consistency. The paper only evaluates image-based metrics (e.g., PSNR), but does not evaluate the temporal consistency between frames, which is also crucial for real-time gaming. In particular, even if each individual frame may have a high PSNR compared to the corresponding ground truth frames, when playing them together as a video, significant flickering artifacts may still exist. Result videos can be a convincing experimental result, but the author does not include them as supplementary material.

## Minor
1. Typo in Fig 2: The channels of the features after PixelShuffle in the ConvLSTM row show "H/SxW/Sx3(S*2)^2", is this a typo? Should it be "H/SxW/Sx3(S^2)^2"?
2. Higher resolution experiments. The paper mainly uses 1080P as the target high resolution in the experiments. However, in modern gaming, 2K and even 4K are becoming more and more common. Existing papers (e.g. FuseSR) already include experiments in 4K settings, it would be great if this paper could also include this to see the method's potential in modern ultra-high resolution gaming.

### Questions
1. Please address the questions raised in the "Weakness" section.
2. Discussion on whether or not to use engine-level data. The paper emphasizes that not using engine-level data is an advantage over existing methods. I can't say I disagree with this claim, but I think it is worth discussing. In practice, engine-level data such as G-buffers are widely used in industry, as they provide rich scene information that can greatly enhance SR quality with relatively modest integration effort. For AAA games, this integration is typically acceptable and even facilitated by official plugins (e.g., NVIDIA has official DLSS plugin on Unreal Engine). In contrast, non-AAA titles are often less photo-realistic and may not rely heavily on SR due to lower rendering demands. Therefore, the trade-off between integration effort and SR quality might be smaller than the paper suggests. That said, I agree that in cloud-based rendering or streaming scenarios, avoiding engine-level data is indeed advantageous, as it reduces network overhead and improves scalability.

### Soundness
3

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
This paper presents a real-time edge-side super-resolution solution applicable to game scenarios. Through some improvements in network structures, it achieves performance similar to that of DLSS and FSR without accessing rendering engine data structures or modifying game source code.

### Strengths
1.The motivation of this article is commendable, as it attempts to apply deep learning super-resolution technology to the edge side and achieve performance similar to that of DLSS and FSR.
2.Write clearly and understandably.

### Weaknesses
As a method attempting to achieve real-time super-resolution on the edge side, merely comparing it with the existing classical image super-resolution methods is insufficient. There is no comparison with some existing edge-side super-resolution methods [1][2] and some lightweight video super-resolution methods [3][4][5][6].

There is a lack of necessary analysis and experiments regarding the ablation of the network structure design. For instance, what advantages does the ConvLSTM have compared to the residual block-based RNN structure [3]? Meanwhile, the report on the additional consumption required for transmitting the hidden state has not been provided yet.  Moreover, it is unclear to what extent the FEB module contributes to the super-resolution performance. 
[1] edge–SR: Super–Resolution For The Masses. WACV 2022
[2] Super-Resolution by Predicting Offsets:An Ultra-Efficient Super-Resolution Network for Rasterized Images. ECCV 2022
[3] Efficient Video Super-Resolution through Recurrent Latent Space Propagation. ICCVW 19
[4] Stable Long-Term Recurrent Video Super-Resolution. CVPR 22
[5] Structured Sparsity Learning for Efficient Video Super-Resolution. CVPR 23
[6] Efficient Video Super-Resolution for Real-time Rendering with Decoupled G-buffer Guidance. CVPR 25

### Questions
Please refer to the weakness part.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes GameSR, an efficient approach for super-scaling video frames in cloud gaming. Unlike industrial solutions that rely on accessing in-game data such as motion vectors and depth maps, GameSR works without requiring direct access to the game engine. The framework consists of three main components: feature extraction, feature learning, and temporal learning. According to the evaluation, GameSR achieves visual quality comparable to industrial solutions while being significantly more efficient than existing CNN-based methods.

### Strengths
+ The paper explores an interesting and relevant problem domain in cloud gaming.
+ The writing and organization are clear and easy to follow.
+ GameSR demonstrates on-par or even superior quality and efficiency compared to both state-of-the-art research baselines and industrial solutions.

### Weaknesses
- Target Resolution: My major concern is on the experimental resolution setup. The paper targets high-resolution cloud gaming but only evaluates upscaling to 1080p. For desktop-level hardware such as an NVIDIA RTX A4000, 1080p is relatively low, especially given that high-end gaming commonly targets 2K or 4K resolutions. Even for handheld devices, prior work ([GameStreamSR: Enabling Neural-Augmented Game Streaming on Commodity Mobile Platforms](https://ieeexplore.ieee.org/abstract/document/10609685)) has already moved toward higher target resolutions (e.g., 1440p). Therefore, the chosen setting may not be representative and reasonable for real-world use cases, and the reported quality, frame rate, and memory usage might not be valid for the intended scenario. 

- Dataset Generalizability: The effectiveness of the method may heavily depend on the type of game and the way it is played. The authors seem to use a custom-collected dataset rather than a public benchmark. It is unclear if the captured gameplay scenarios are diverse enough or if they might unintentionally favor the proposed method.

- Scope of Evaluation: The paper only evaluates the super-resolution component itself. However, for cloud gaming, the end-to-end latency—including rendering, encoding, network transmission, decoding, and upscaling—is the key metric for real-time performance. Without this measurement, it is difficult to assess the practical benefit of GameSR.

- Limited Game Selection: GameSR is evaluated on only three games. This may be insufficient to demonstrate its robustness and general-purpose applicability across different game genres, art styles, and game engines.

### Questions
1. Could you please clarify the methodology for measuring the quality metrics (PSNR, SSIM, VMAF and LPIPS)? Specifically, what frames were used as the high-resolution ground truth reference, and what were the corresponding low-resolution inputs provided to GameSR?
2. The paper appears to target desktop-level hardware for super resolution inference, but the experimental setup could be stated more clearly. Could you clarify the intended hardware target(s)? How would you expect GameSR to perform on more resource-constrained handheld devices?
3. While the super-resolution performance result is detailed, what is the expected breakdown of latencies in a real-world, end-to-end deployment (i.e., including rendering, encoding, transmission, decoding, etc.)? Can the entire pipeline, with GameSR integrated, realistically achieve the real-time performance required for interactive high resolution cloud gaming?
4. How does GameSR's performance (in terms of both quality and processing speed) scale when targeting higher resolutions, such as 2K or 4K, which are more realistic for the described desktop gaming scenario?

### Soundness
2

### Presentation
3

### Contribution
2
