# GmNet: Revisiting Gating Mechanisms From A Frequency View

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 8, 6, 4

## Abstract
Lightweight neural networks, essential for on-device applications, often suffer from a low-frequency bias due to their constrained capacity and depth. 
This limits their ability to capture the fine-grained, high-frequency details (e.g., textures, edges) that are crucial for complex computer vision tasks. To address this fundamental limitation, we perform the first systematic analysis of gating mechanisms from a frequency perspective. 
Inspired by the convolution theorem, we show how the interplay between element-wise multiplication and non-linear activation functions within Gated Linear Units (GLUs) provides a powerful mechanism to selectively amplify high-frequency signals, thereby enriching the model's feature representations.
Based on these findings, we introduce the Gating Mechanism Network (GmNet), a simple yet highly effective architecture that incorporates our frequency-aware gating principles into a standard lightweight backbone. The efficacy of our approach is remarkable: without relying on complex training strategies or architectural search, GmNet achieves a new state-of-the-art for efficient models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper deals with lightweight networks for edge devices. The propose to use gating mechanisms like GLUs to directly counteract the low-frequency bias present in many efficient architectures. They analyse GLUs from a frequency perspective and present a lightweight architecture they call Gating Mechanism Network (GmNet) on top of MobileNetV2. The study different gating mechanisms and show that the simplest linear one performs best. They get state-of-the-art resuts on ImageNet

### Strengths
* The paper analyses GLUs from a frequency perspective and links their core operations to the ability to modulate a network’s spectral response. They use that to counteract a low-frequency bias in lightweioght architectures
* It is nice to see that linear gating seems to perform best 
* The paper reports strong results on ImageNet1k when basing GmNet on MobileNetV2 (2018). It also gets strong results on COCO for Object detection & segmentation.

### Weaknesses
1) The paper offers limited theoretical depth and bases the analysis on empirical Fourier observations. The analysis section (Sec 3) is weak in scope and effectively only studies how a smooth (Gelu) and "non-smooth" (ReLU6) activation function affect classification performance of a MobileNet durign training for low/high frequency components of the input image.

2) There is a lot of emphasis on results with low radii $r$. Although it is clear that the proposed method improves over other methods for high frequencies (Table 3), in terms of overall performance, the gains are small over the state-of-the-art (<1%). It is unclear to me how top-1 overall gain relates to the analysis in Sec 3 or results on low/high frequencies.

3) The paper would be strong if results on more datasets/tasks were presented. (E.g. Places or fine-grained classification datasets like CUB/Cars etc)

### Questions
Q1) Which dataset do Fig3/4 refer to? Is it all on ImageNet1k?

Q2) Why is the performance across epochs important in Fig3/4? what does the x-axis tell us over different epochs (eg vs re porting final performance)? It is hard for me to understand why the authors claim that "the convergence is faster" in the discussion, when the rightmost figure shows more or less equal convergence trends.

Q3) Could GmNet be adapted to recent transformer based architectures like RepVit or Lsnet?

Q4) "As shown in Fig. 4, we present the testing accuracy curves under varying frequency thresholds" - isn't the treshold (r) set to 10 everywhere? Apologies if i misunderstood.

Q5) What is the top-1 ImageNet1k performance on GMNet-S1/2/3/4 when simply changing the ReLu6 to simple ReLU (keeping the exact same architecture otherwise)? 

Q6) How do you justify that the simplest, linear gating mechanism works best (eg in Tab 4/8)?

Q7) Is there a  typo in the caption of Fig 1 where it mentions " retains the general shape of the frog"?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper introduces a theoretical study and experimental evaluation of Gated Linear Units (GLU) in CNNS from a Fourier perspective. In the theoretical part, the authors analyze the effects of non-linear activations of different variants of  ReLUs and GELUs and their properties in frequency space and their effect on the ability of networks to learn functions with high frequency components. Subsequently, the paper derives a theoretical analysis of GLUs in the same context by deriving the frequency mixing effect of element-wise multiplications in the frequency domain from the Convolution Theorem. This leads to the theoretical conclusion, that GLUs are suitable to allow networks to learn to preserve higher frequency information.

The paper then implements GMNet, a GLU enhanced version of MobileNetV2 and shows superior performance on ImageNet in comparison to SOTA efficient image classification networks. The authors conclude, that the effect of GLUs is most beneficial efficient network architectures (for mobile devices), which are known to have a stronger in low-frequency bias.

### Strengths
The paper tackles an important theoretical question: the properties of simple network components liker activation functions and element-wise multiplications (and a such GLUs) have not been studied even though these properties of other components (like convolution filters, padding and down-sampling units) have shown significant impact on the performance and robustness/generalization abilities of CNNs.

On the practical side, the introduced GMNet confirms the theoretical analysis and shows significant performance increases over efficient SOTA architectures.

### Weaknesses
N1: The introductory part is hard to read for someone who is not working on related problems. Especially the difference between the frequency representation of inputs (read data and feature maps in later layers) and the the frequency representation of the learned decision function is not well explained and leaves unclarities to which the the low-frequency bias is actually relating.

N2: Also the introductory and related work section somewhat neglect the bigger picture of similar studies conducted on other components (especially convolutions and down-sampling) - for example [1] and [2]. While some works are cited, the relation of these works are not put into perspective.
  
N3: as the authors mentioned, basing decisions increasing on high frequencies, can result in aliasing [3] and significant loss of robustness [2]. Here an ablation study would be very beneficial.

N4:  the experimental evaluation is limited to low resolution ImageNet data. This also directly limits the frequency representation. It would be good to see how the proposed approach behaves on high resolution data. 

**Minor:**
* Figure 1 is not matching it's caption: there are only two different frequency inputs (not three) and where is that frog? Also the schematic illustration of fig 1 does not actually show the effect of GLUs (on real data)-> it actually would be nice to have a plot like this: real data -> ReLU vs GELU vs GLU output in spatial and frequency domain.


[1] Durall et. al. "Watch your up-convolution: Cnn based generative deep neural networks are failing to reproduce spectral distributions." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020.

[2] Grabinski, et al. "Frequencylowcut pooling-plug and play against catastrophic overfitting." European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2022.

[3] Grabinski, Julia, Janis Keuper, and Margret Keuper. "Aliasing and adversarial robust generalization of cnns." Machine Learning 111.11 (2022): 3925-3951.

### Questions
Q1:  I'm curious what the GLUs are actually learning. Vitalization of the GLU weight by layer and the resulting changes in the frequency spectrum would be very informative. It would be very interesting to know in which parts of the network higher frequency information is relevant 

Q2: the evaluation focuses on efficient nets. it would be very interesting to see GLUs in large modern CNNs like ConvNext (v2). I would speculate, that the larger kernels in large CNNs (which are less band limited) mitigate the effects of GLUs. This would not harm the results of this paper, but would further increase the general insights. Have the authors tried this?

### Soundness
3

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
This paper reinterprets GLU gating from a frequency-domain standpoint and derives design rules that are instantiated in a compact backbone called GmNet. Concretely, the model uses a shared-representation gate, 7×7 depthwise convolutions to mix frequency bands, and favors ReLU6 for a more stable high/low-frequency trade-off. The authors also propose a DFT-based frequency-split evaluation protocol and run ablations on activation choice, gating variants, and latency–accuracy trade-offs. Experiments show GmNet achieves good performance on ImageNet-1K, with runtime comparisons against efficient baselines. Evidence is provided mainly on ImageNet classification. Overall, the primary contribution is an explanatory frequency view that motivates a modest architectural tweak for efficient vision models, supported by single-task experiments and ablations.

### Strengths
1. Recasts GLU with a frequency-domain hypothesis and condenses it into a single, drop-in GmNet block (shared-representation gate + flanking 7×7 depthwise convs, ReLU6), keeping parameters/latency overhead small and integration into lightweight CNNs straightforward.
2. On ImageNet-1K, reports SOTA/near-SOTA accuracy–latency trade-offs against strong efficient baselines under matched settings, with consistent gains across model scales.

### Weaknesses
1.	The frequency claim remains qualitative and no per-layer power spectra, gain/attenuation curves, or controlled sinusoid/wavelet probes; “high-frequency amplification” is hard to verify quantitatively.
2.	Architectural novelty is modest (GLU + 7×7 DWConv + ReLU6); gains may conflate with kernel size/width/depth/recipe. Lacks a contribution breakdown under strictly matched FLOPs/params and training budgets.
3.	Evidence is centered on ImageNet in the main text; downstream/cross-dataset/robustness results are relegated to the supplement and explored shallowly, limiting generality.
4.	Measurement/reporting gaps: mostly single-run top-1 without confidence intervals; latency lacks a standardized, reproducible protocol (hardware, batch size, warm-up, operator fusion). 
5.	Positioning vs contemporaneous gating/frequency methods is under-specified.

### Questions
1.	Would you be able to provide quantitative spectral evidence—e.g., per-layer power spectra before/after gating, high/low-frequency energy ratios, and responses to controlled sinusoid/wavelet probes—and also formulate a clear, testable proposition about the gate’s frequency response that could, in principle, be falsified by these experiments?
2.	Under strictly matched FLOPs, parameter counts, and training recipes, could you report a contribution breakdown for (a) baseline, (b) +7×7 depthwise conv only, (c) +gate only, (d) +ReLU6 only, and (e) full GmNet, including mean ± std over at least three random seeds?

### Soundness
2

### Presentation
3

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
This paper introduces GmNet, a new architecture for lightweight Convolutional Neural Networks (CNNs) that achieves state-of-the-art performance. The core idea is the adaptation of the Gated Linear Unit (GLU) mechanism, which has been highly successful in language models, to the domain of efficient computer vision. The authors present a systematic analysis of GLU from a frequency perspective, demonstrating that the element-wise multiplication within GLU acts as a convolution in the frequency domain, which can selectively amplify high-frequency signals. Based on these insights, they introduce the Gating Mechanism Network (GmNet), a simple and efficient architecture that incorporates these frequency-aware gating principles. Without complex training strategies, GmNet sets a new state-of-the-art for efficient models, with the GmNet-S4 variant achieving 81.5% top-1 accuracy on ImageNet-1K, validating the effectiveness of designing lightweight models that can learn from a full spectrum of frequencies.

### Strengths
- This paper shows not only the number of FLOPs and parameters, but also the model latency on both GPU and a mobile device, which demonstrates the effectiveness of the proposed lightweight model.
- The proposed GmNet block is simple and clean, integrating the gating unit effectively without excessive architectural complexity. But the state-of-the-art (SOTA) empirical results on the ImageNet benchmark show its effectiveness.
- The authors perform thorough ablation studies that validate their design choices. They compare different activation functions and different GLU designs.

### Weaknesses
- The novelty of the proposed method is limited. In the current LLM design, GLU variants are widely used in the gated MLP modules. Those LLM models already demonstrated the effectiveness of those designs.
- In CNNs, many "quasi-attention" methods, e.g., SENet[1], GCNet[2], have already explored the element-wise multiplication of the activation tensors, and their results show that the gated design in CNNs can help achieve better model accuracy, similar to the findings of this GmNet. It would be better to compare GmNet with those methods to demonstrate its advantages.
- The results in Table 6 cannot support the hypothesis in Section 3.2 (or Figure 4) that ReLU6 can be better for high-frequency data and not good for low-frequency data. The accuracy gain should mainly come from the activation function design, which leads to better model performance in general. This improvement doesn't seem to be related to how the activation function processes low or high frequencies.

[1] Jie Hu, et al. Gather-Excite: Exploiting Feature Context in Convolutional Neural Networks, NeurIPS, 2018

[2]Yue Cao, et al. GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond, ICCV workshop, 2019

### Questions
- The swiGLU and SiLU are commonly used in the gated MLP of current LLMs design, and many of those LLMs have demonstrated the effectiveness of these activation functions. It would be better to evaluate these two activations in CNN models to see whether they can also improve model accuracy.
- Typos:
	- L45: showed -> shown
	- L318: traini ng -> training
	- L427: Moverover -> Moreover
	- L86: duplicated sentences: 
		- Consequently, our approach is inherently more effective in preserving and enhancing high-frequency information. 
		- As a result, the former structure is inherently more effective in preserving and enhancing high-frequency information.

### Soundness
3

### Presentation
3

### Contribution
2
