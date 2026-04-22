# Preserving Forgery Artifacts: AI-Generated Video Detection at Native Scale

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
The rapid advancement of video generation models has enabled the creation of highly realistic synthetic media, raising significant societal concerns regarding the spread of misinformation. However, current detection methods suffer from critical limitations. They rely on preprocessing operations like fixed-resolution resizing and cropping. These operations not only discard subtle, high-frequency forgery traces but also cause spatial distortion and significant information loss. Furthermore, existing methods are often trained and evaluated on outdated datasets that fail to capture the sophistication of modern generative models. To address these challenges, we introduce a comprehensive dataset and a novel detection framework. First, we curate a large-scale dataset of over 140K videos from 15 state-of-the-art open-source and commercial generators, along with Magic Videos benchmark designed specifically for evaluating ultra-realistic synthetic content. In addition, we propose a novel detection framework built on the Qwen2.5-VL Vision Transformer, which operates natively at variable spatial resolutions and temporal durations. This native-scale approach effectively preserves the high-frequency artifacts and spatiotemporal inconsistencies typically lost during conventional preprocessing. Extensive experiments demonstrate that our method achieves superior performance across multiple benchmarks, underscoring the critical importance of native-scale processing and establishing a robust new baseline for AI-generated video detection.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper explores AI-generated video detection. It curates a dataset consisting of videos created from 15 or 18 generative models, as well as a novel framework that processes videos at their native spatial resolution and temporal duration, avoiding destructive preprocessing like resizing and cropping that mess up subtle forgery artifacts. The framework is built on the Qwen2.5-VL Vision Transformer, which uses 3D patchification directly on the raw video tensor, preserving the high-frequency details essential for detection. The experiment result shows improvement compared to baseline methods.

### Strengths
1. AI-generated video detection is an important topic
2. Providing a curated dataset of Gen-AI videos is a good contribution
3. The proposed detection framework shows improvement compared to baseline methods.

### Weaknesses
1. The proposed detection methodology is simply a combination of existing approaches and does not present a novel contribution

Detail: The core detection framework is a direct application of the existing Qwen2.5-VL Vision Transformer using its native 3D patchification strategy, which the authors note is adopted from prior work (Bai et al., 2025). While applying this to the forgery detection task is a valid contribution, the paper presents it as a 'novel detection framework'  when the architectural novelty itself is minimal. The primary novelty seems to be the dataset and the hypothesis that native-scale processing is superior, rather than a new detection architecture.

2. The presence of artifacts is vaguely assumed but not explicitly defined or proven to exist.

Detail: The authors do not define, visualize, or analyze what these artifacts are. A significant weakness is the lack of any model explainability (e.g., activation maps or gradient-based analysis) to prove that the native-scale model is actually focusing on these "pixel-level artifacts", while the 224p-resized model is not. The entire premise, while intuitive, is treated as an assumption rather than a proven scientific finding.

3. The use of downgraded input by cropping or resizing is argued to be the main fault of previous works. However, it is not explained or experimented with in connection with the actual baseline used.

Detail: Since the proposed method also performs poorly when given low resolution input, it is not surprising that baseline methods perform poorly when given low resolution input. This makes it impossible to know if the proposed method is superior due to its architecture or simply because it's the only one allowed to see the high-resolution data. A fair comparison would require adapting baselines to also accept high-resolution inputs or, at a minimum, comparing all models at the same fixed resolutions (e.g., 224p, 480p, 720p).

Other issues:
- The proposed detection method is shown to have a much higher cost in training with memory and GPU hours.
- Discrepancy in the number of generative models used (abstract: 16; Introduction: 15 and 18; appendix: 15)

### Questions
1. Can baseline detection methods be retrofitted to use the high-resolution inputs for training and inference? 
2. Following the above question, if given high-resolution input in both training and inference, how does the baseline method compare with the proposed method?

3. Can the artifacts be identified or visualized?
4. Following the above question, can the artifacts be removed or interfered with so that the detection method fails?

### Soundness
2

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
The paper proposes an AI-generated video detector that (1) builds a  dataset and (2) trains a detector on native spatial resolution and temporal duration using Qwen2.5-VL ViT with 3D patchification, so that high-frequency, position-dependent artifacts aren’t destroyed. On three families of benchmarks (GenVideo, DVF, and their Magic Videos), it reports consistently higher performance than prior image-detectors, deepfake-detectors, and video backbones. The core narrative is: current detectors are undertrained on modern, high-quality video generators, and downsampling design.

### Strengths
1. Timely problem & data refresh. Most detectors are indeed lagging behind 2024–2025 video generators; this paper explicitly targets that gap and gathers content from new models, including commercial/API ones. That’s rare and useful.
2. Clear empirical story about resolution. The experiment results supports the authors' claim well and the motivation is clear.
3. The paper is well-written and easy to follow.

### Weaknesses
1. The paper addresses two real but relatively well-recognized issues — resolution-destructive preprocessing and stale generator distributions. The proposed solution mainly combines (i) a recent, high-capacity video/VL backbone and (ii) a refreshed multi-generator dataset. The detector architecture itself is largely standard, without forgery-specific inductive biases. Data curation process is also well defined in previous fake video detection work. Thus the contribution is more of an engineering consolidation.
2. Dataset release / licensing / reproducibility is unclear. Several of the listed video sources (Kuaishou, Luma, MovieGen) are API/commercial. The paper doesn’t yet make it clear what exactly will be released, how prompts will be shared, and how others can reproduce “Magic Videos”. For ICLR this is important.
3. No fairness breakdown. Real videos come from different sources than synthetic ones; there can be source, watermark, or codec biases. The paper doesn’t fully rule that out. It's important to demonstrate the performance on unseen real videos from very different source such as KITTI, etc. to demonstrate the generalizability of the proposed model.
4. Limited analysis on real-world perturbations. The core claim is about preprocessing destroying artifacts, but actual attackers / platforms will introduce their own compress-and-resize chains. It would be good to see: scale jitter & heavy H.264/HEVC compression and see whether “native-scale” still wins.
5. Incomplete discussion of most recent related work such as [1, 2]. 

Reference
1. Distinguish Any Fake Videos: Unleashing the Power of Large-scale Data and Motion Features. 2024
2. How Far are AI-generated Videos from Simulating the 3D Visual World: A Learned 3D Evaluation Approach. 2025

### Questions
1. How sensitive is the detector to platform compression? Your main argument is “don’t downsample to 224.” But if a platform already did that, can your model still outperform the older 224-trained models? A controlled experiment with platform-style compression would be convincing.
2. How do you prevent source leakage? Since real videos come from specific real video datasets and generated ones are from VBench/MovieGen/etc., detectors might be learning source signatures. Do you have a cross-source test where the real videos share encoding with the synthetic ones?
3. Happy to see more discussion with latest related work in the related area. This would help make the contribution clear.

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
3

### Summary
This paper investigates the limitations of AI-generated video detection, highlighting that fixed-resolution preprocessing removes high-frequency forgery artifacts and that outdated datasets struggle to represent modern models. The authors build a dataset of about 140K videos from 18 generative models and real sources such as Kinetics and MSVD, and design a realism-oriented test set called Magic Videos. They propose a detection framework based on Qwen2.5-VL ViT, which processes videos at their native spatial and temporal scales using 3D patch tokenization to preserve forgery details. Experiments are conducted on GenVideo, DVF, and Magic Videos.

### Strengths
Builds a large-scale dataset with over 140,000 videos from 18 generative models, covering mainstream AI video generation technologies and offering a research resource.

Conducts experiments on multiple benchmarks, including the self-built Magic Videos test set and public datasets, ensuring comparability and reliability.

Employs a dynamic resolution processing framework that preserves detection performance and demonstrates generalization ability across datasets.

Designs ablation experiments that are systematic and support the main conclusions.

Integrates Flash Attention and LoRA optimizations, reflecting concern for efficiency and practical deployment.

### Weaknesses
Lacks analysis of the model’s decision-making mechanism, failing to verify whether the model truly learns forgery-related cues rather than biased features in the data.

Provides insufficient discussion of computational efficiency, with no concrete results on inference speed or resource requirements for real-world deployment.

The training data source is limited, relying mainly on the VBench dataset, which may introduce hidden biases.

The failure case analysis is inadequate, lacking exploration of the scenarios and causes where the model fails.

### Questions
Regarding computational efficiency, the paper mentions optimization techniques such as Flash Attention, but lacks specific key metrics such as inference speed and memory usage. Could you provide detailed efficiency data under typical hardware configurations to assess the feasibility of this method in real-world scenarios?

For Magic Videos, how is the "indistinguishable to humans" claim verified?

Experimental results show significant performance differences across different generators (from 72.26% to 85.12% in Table 2). Have you analyzed the specific reasons for these differences? Are there any specific types of generated content or technical approaches that are blind spots for the current method?

Could you provide a more in-depth interpretability analysis, such as attention visualization or feature analysis, to demonstrate that the model truly learns meaningful forgery traces, rather than relying on other superficial features in the data?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper addresses the pressing need for effective detection methods for AI-generated videos, which is a highly valuable area of research. The authors provide an in-depth exploration of the current state of research, highlighting its shortcomings, including the development of video generation models, detection of generated images, and existing methods for detecting generated videos. Two major challenges are identified in the detection of AI-generated videos: 1) The fixed-resolution preprocessing through operations like cropping and downsampling leads to information loss and coarse-grained detection; 2) Current detection methods are typically trained on outdated synthetic data sources, which are insufficient for handling videos produced by the latest high-quality generative frameworks.
In response to these issues, the authors construct a high-quality and diverse dataset sourced from state-of-the-art generative models and propose a novel detection framework built upon Qwen2.5-VL, which processes videos in their native spatial resolution and temporal length. This approach preserves crucial forgery artifacts often lost in conventional preprocessing steps, such as resizing or cropping.
Through extensive experiments, the authors demonstrate the effectiveness of their method and reasonably discuss the remaining challenges. Their work significantly advances the field, providing a robust foundation for future AI-generated video detection efforts.

### Strengths
1.This paper addresses the significant limitations of existing methods for detecting AI-generated videos, highlighting the urgent need for effective detection strategies in light of the rapid development of video generation technologies.
2. The authors construct a novel, large-scale dataset by utilizing cutting-edge video generation tools, which ensures that the dataset is diverse and high-quality, effectively supporting the proposed detection framework.

### Weaknesses
1.The section on "3D Video Patchifying at Native Scale" lacks sufficient novelty. Although the paper claims that the model is trained at native resolution, this approach was already introduced in Qwen2.5-VL [1]. The focus of the method seems to be more on engineering optimization rather than presenting a fundamentally new contribution to the field. Additional exploration of novel techniques or improvements beyond existing methods would strengthen this section.
2.The paper lacks a more detailed analysis of the proposed method, particularly regarding how the model differentiates between real and generated content. While the authors claim the use of native resolution processing, there is little discussion on which specific parts of the video (e.g., temporal inconsistencies, high-frequency artifacts, or motion patterns) the model focuses on to distinguish between real and synthetic videos. A deeper exploration of the key features the model uses to make this distinction would help clarify the strengths of the approach and provide insights into its decision-making process.
3.While the use of dynamic resolution processing significantly enhances performance, it introduces additional computational overhead. This increased complexity may limit the model’s feasibility for real-world deployment, particularly on resource-constrained devices or in scenarios requiring real-time detection.
4.The proposed dataset structure is highly unreasonable; each class in the test set has very few samples, making the statistical results highly likely to be biased.
5. The experiments were insufficient and were not tested on the latest dataset GenVidBench.
[1]Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report, 2025.

### Questions
see the weakness

### Soundness
3

### Presentation
3

### Contribution
3
