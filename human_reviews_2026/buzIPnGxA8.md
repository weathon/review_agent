# DeAltHDR: Learning HDR Video Reconstruction from Degraded Alternating Exposure Sequences

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
High dynamic range (HDR) video can be reconstructed from low dynamic range (LDR) sequences with alternating exposures. However, most existing methods overlook the degradations (e.g., noise and blur) in LDR frames, focusing only on the brightness and position differences between them. To address this gap, we propose DeAltHDR, a novel framework for high-quality HDR video reconstruction from degraded sequences. Our framework addresses two key challenges. First, noisy and blurry content complicate inter-frame alignment. To tackle this, we propose a flow-guided masked attention mechanism that leverages optical flow for a dynamic sparse cross-attention computation, achieving superior performance while maintaining efficiency. Notably, its controllable attention ratio allows for adaptive inference costs. Second, the lack of real-world paired data hinders practical deployment. We overcome this with a two-stage training paradigm: the model is first pre-trained on our newly introduced synthetic paired dataset and subsequently fine-tuned on unlabeled real-world videos via a proposed self-supervised method. Experiments show our method outperforms state-of-the-art ones. Code and data will be available at https://zhang-shuohao.github.io/DeAltHDR/.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes DeAltHDR, a novel framework for high-quality HDR video reconstruction from degraded sequences. First, it proposes a flow-guided masked attention that leverages optical flow for a dynamic sparse cross-attention computation, achieving improved performance while maintaining efficiency. Its controllable attention ratio allows for adaptive inference costs. Second, to enable  practical deployment, it introduces a two-stage training approach: the model is first pre-trained on the newly introduced synthetic paired dataset and subsequently fine-tuned on unlabeled real-world videos via a proposed self-supervised method. Experiments show the proposed method outperforms state-of-the-art methods.

### Strengths
- A flow-guided masked attention for efficient inter-frame alignment, where the attention ratio can be dynamically adjustable for adaptive inference cost.
- A motion-enhanced self-supervised fine-tuning approach is proposed to improve the reconstruction effect on real-world videos.
- A synthetic and real-world datasets with rich scenes are constructed. The proposed method outperforms the state-of-the-art methods on multiple datasets.

### Weaknesses
- The paper shows many frame-level qualitative images but provides no video results or links to video demos to assess temporal consistency. For a video reconstruction paper, absence of video demos is a major omission. This omission reduces my confidence in the method’s claimed temporal stability and real-world applicability.
- The paper introduces both a synthetic and a real-world dataset but provides only limited visualizations and statistics . This makes it hard to judge how challenging the real captures are (exposure range, highlight clipping). It would be better add per-scene histograms, luminance range (DR) for the real-world set, sample raw/RGB patches at multiple exposure levels, and a table summarizing scene types and illumination conditions.  
- Moreover, the real dataset uses only one phone model (iPhone 16 Pro Max). That limits claims of real-world generalization across camera sensors (different saturations, noise profiles, ISP pipelines).

### Questions
- No supplementary video results are provided, making it difficult to assess temporal consistency. Please clarify.
- The contribution and effectiveness of the proposed dataset are insufficiently validated; stronger justification or ablation evidence is needed.

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
4

### Summary
This paper tackles the problem of reconstructing HDR videos from alternating-exposure LDR sequences, which is a realistic yet challenging task due to varying exposure, motion, and noise artifacts.
The proposed DeAltHDR is a framework featuring a flow-guided masked cross-attention module that selectively attends to motion-aligned regions across alternating frames.
This design aims to improve robustness to motion blur, ghosting, and exposure inconsistencies.
This paper  further introduces a controllable attention ratio to trade off performance and inference cost.
Experiments on both synthetic and real datasets demonstrate superior quantitative and qualitative HDR reconstruction results over several baselines.

### Strengths
1. Realistic Problem Setup: Addressing alternating-exposure HDR reconstruction is practically valuable and closely aligned with real camera capture settings.
2. Novel Attention Design: The flow-guided masked cross-attention module and controllable attention ratio are well-motivated and effectively balance accuracy vs. computational cost.

### Weaknesses
1. Evaluation Metrics: While the proposed method outperforms baselines on reconstruction metrics, it does not include temporal consistency metrics. For video HDR tasks, flicker suppression is crucial, making temporal consistency metrics essential.
2. Synthetic Dataset: The synthetic dataset uses RIFE for interpolation. However, existing interpolation methods are typically designed and trained for 8-bit videos, which may not be suitable for HDR video inputs, potentially resulting in inaccurate HDR frame interpolation.
3. Training Data for Comparison Methods: Since the proposed method is trained on a new dataset that includes blurriness and noise in the input LDR images, the comparison methods should also be trained on datasets with similar characteristics to ensure a fair comparison.
4. Absence of Video Results: The paper does not provide qualitative or quantitative video results. Since the method is designed for video HDR reconstruction, demonstrating video sequences with temporal consistency and flicker-free performance is crucial. Without video results, it is difficult to verify the practical effectiveness of the method in real-world scenarios.

### Questions
1. As shown in Fig. 5, our method outperforms the “w/o Adaption” baseline in terms of sharpness. Based on the definition of the Adaption module, it seems primarily designed to improve reconstruction consistency when different frames are used as input. However, it is unclear why this module also affects reconstruction sharpness.
2. Which specific component of the proposed method contributes most to handling the noise and blurriness in the input LDR images? The paper does not clearly describe any design specifically targeting in-frame blurriness or noise.

### Soundness
3

### Presentation
2

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
This paper proposes DeAltHDR, a method for reconstructing high-quality HDR videos from degraded LDR sequences with alternating exposures. It introduces flow-guided masked attention for efficient inter-frame alignment, and a two-stage training strategy that pre-trains on synthetic paired data and fine-tunes on real videos with self-supervision. Experiments show that DeAltHDR outperforms existing state-of-the-art methods on both synthetic and real-world datasets.

### Strengths
1. The paper has a strong motivation. The authors observe that previous methods, such as HDRFlow, overlook crucial degradations like noise and blur in HDR reconstruction. In contrast, this work explicitly addresses these issues, enabling the reconstruction of blur-free, high-quality HDR videos.

2. This work achieves state-of-the-art accuracy and efficiency. Specifically, it delivers higher reconstruction quality than previous methods like HDRFlow, while maintaining comparable efficiency to HDRFlow, the fastest existing method.

3. The paper is well-written and the experiments are comprehensive and thorough.

### Weaknesses
1. In the introduction, the authors mention that optical-flow-based alignment cannot handle long-exposure blur, while attention-based methods can. However, the paper does not provide a detailed explanation for this claim. I am curious why attention-based methods can restore blur as shown in Figures 3, 4, and 5.

2. The self-supervised fine-tuning is very similar to BracketIRE, which limits the novelty and contribution of this part of the work.

3. This paper aims to address noise and blur in HDR reconstruction, but most of the results mainly demonstrate the handling of blur, and there is a lack of results showing the effect on noise.

### Questions
1. The experiments show that s = ∞ achieves the best results, indicating that using only attention without optical flow gives the best performance. Then, why is the core method still flow-guided masked attention? It seems that not using flow performs better, which is confusing.

2. The content shown in Figure 1 is a bit confusing. Does increasing the attention ratio lead to higher FLOPs? Is the relationship directly proportional or inversely proportional?

### Soundness
3

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
This paper proposes DeAltHDR, a framework for HDR video reconstruction that robustly handles input degradations, such as blur and noise. A self-supervised fine-tuning strategy with randomized frame sampling enhances motion diversity and model performance. The authors also released two HDR video datasets with realistic degradation to support future studies.

### Strengths
1. The proposed FGMA mechanism effectively balances optical- and attention-based alignment, achieving a good trade-off between accuracy and efficiency.

2. The authors also released two HDR video datasets with realistic degradation to support future studies.

3. The paper is clearly written and well-organized.

### Weaknesses
1. Temporal consistency is a crucial aspect in video tasks; however, the authors did not provide corresponding video visualizations or quantitative evaluations. Additional analysis is recommended to better validate the temporal stability of the proposed method.

2. Although the dataset considers blur degradation, the model itself does not appear to include specific designs targeting blur removal. The authors should analyze why the proposed method can still handle blur effectively and discuss its generalization capability. As far as I know, traditional deblurring methods often suffer from limited generalization.

3. The visualization results are insufficient. It is suggested to provide more detailed analyses under different reference exposure settings: when the reference frame is long-exposure, the model should primarily address blur; when it is short-exposure, the model should focus on noise suppression. Please include corresponding visual and quantitative results to demonstrate the model’s robustness under varying exposure conditions.

4. While the paper discusses computational budgets, inference speed is also an important factor. The proposed method seems to be less efficient in terms of runtime. Please further analyze the reasons for this limitation.

### Questions
1. On which dataset was the Adaptation ablation study conducted, and what is the scale of the data used? Please clarify these details in the paper.
2. Will the authors make the dataset and code publicly available to facilitate reproducibility and future research?

### Soundness
3

### Presentation
3

### Contribution
3
