# QuantDemoire: Quantization with Outlier Aware for Image Demoiréing

- Decision: Reject
- Scores: 2, 2, 4

## Abstract
Demoiréing aims to remove moiré artifacts that often occur in images. While recent deep learning–based methods have achieved promising results, they typically require substantial computational resources, limiting their deployment on edge devices. Model quantization offers a compelling solution with its advantages of compactness and efficiency. However, directly applying existing quantization methods to demoiréing models introduces severe performance degradation. The main reasons are distribution outliers and weakened representations in smooth regions. To address these issues, we propose QuantDemoire, a post-training quantization framework tailored to demoiréing. It contains two key components. **First**, we introduce an outlier-aware quantizer to reduce errors from outliers. It uses sampling-based range estimation to reduce activation outliers, and keeps a few extreme weights in FP16 with negligible cost. **Second**, we design a frequency-aware calibration strategy. It emphasizes low- and mid-frequency components during fine-tuning, which mitigates banding artifacts caused by low-bit quantization. Extensive experiments validate that our QuantDemoire achieves large reductions in parameters and computation while maintaining quality. Meanwhile, it outperforms existing quantization methods by over **4 dB** on W4A4. Code will be released.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors aim to improve quantization for ML-based demoireing methods on the edge. Their method consists of three components:
1. setting the activation quantizer grid through a minmax of a *sample* of the activations (thereby reducing the quantization range, as some outliers may not be sampled),
2. keeping the worst outlier entries of the weights in Float16,
3. calibrating the activation quantization boundaries (i.e. clipping) based on some "frequency-aware" reconstruction loss

### Strengths
1. The authors address a real-world application, demoireing on the edge.
2. The ablations are quite extensive (but see my questions)
3. The results suggest the method performs favorably compared to existing works.

### Weaknesses
**1. Method**

Unfortunately, at this point I do not recommend acceptance. I believe there are a few issues with the method that I urge the authors to seriously consider:

**a. Activation Quantization**. The idea behind Eq. 6, randomly sampling activation entries to avoid outliers, is not effective---despite the authors' claim "the proposed method effectively captures the typical distribution of activations" and "[this means we] naturally discard extreme outliers". Just subsampling the entries does not change the distribution that is sampled from, it only changes the number of samples that is used for estimating the max value. Moreover, there is a large variance in this procedure---sometimes bad outliers may be sampled, and sometimes not. For example, it is not clear why Figure 3 right should be preferred---the real outliers are actually worse here. It would be much better in my opinion to compute e.g. the 95% quantiles for setting the range---this is not cheap, but even an approximation would suffice (e.g. `x>mean+std*some_constant`). It would be even better to learn how much to clip on some calibration data, which doesn't need to be expensive.

**b. Weight quantization**. The mixed-precision scheme makes sense, but because the mixed-precision is completely unstructured, the matmul would not be parallelizable (e.g. not run on GPU/TPU/NPU). The authors do not address this. Using mixed-precision for weights is also not new, so in any case this cannot really be claimed as a contribution. 

**c. Calibration stage**. The authors' use of "quantization boundary optimization" (Section 3.3) is not clear to me. In 3.2, they propose the sampled activation quantizer approach, but now they say they optimize the boundary anyway.


**2. Results**

**a. stds**. The ablations do not have standard deviations. I assume these experiments are quite cheap, and hence std's would be feasible. They are valuable, because the sampling strategy of the activation quantization clearly has a high variance. The calibration approach also has some non-ignorable noise.

**b. Compression Ratio**. What do you mean with OPS? I assume the number of ops is the same (or slightly higher), but that the ops are cheaper in low-bit. In any case, these results should in my opinion be compared to Bfloat16, not Float32. Additionally, the effect of switching from INT to Float for some weight indices should not be underestimated and is not addressed by these results.

**c. Experimental details**. More experimental details would be desirable, see my two questions below.


## Minor

> [L.136] Although QAT achieves competitive performance (Nagel et al., 2021), its high training cost makes
it less suitable for deployment on edge devices.

Since training does not happen on the edge, this argument does not hold.

> [L. 213] Some existing post-training quantization methods (such as percentile (Li et al., 2019)) are specifically
designed to address the problems caused by outliers. However, these methods could lead to some
additional time overhead during the calibration stage. At the same time, because of their inflexible
design, these methods may lack robustness facing outliers in different distributions.

This could be explained better. Again it seems to me that calibration can happen not on the edge, so this is no issue. It is also not clear what is meant by the last sentence, "may lack robustness [due to their "inflexible design"]"

### Questions
1. Do all ablations use calibration of activation boundaries? E.g. also the "Original" and "Smooth (Raw)" baselines?

2. Can you explain why the percentile approach performs worse than your sampled approach?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces QuantDemoire: a post training quantization methodology designed specifically for the quantization of Image Demoiréing models. The methodology consist of an alternative sample-base range-setting methodology, which preserves a small percentage of weights in high precision, and a calibration method focused on preserving low and medium frequencies of the model output. An empirical evaluation demonstrates the effectiveness of the proposed methodology in extreme quantization settings, by comparing QuantDemoire against recent approaches developed in the context of image generation quantization literature.

### Strengths
* The paper introduces a novel method for effectively quantizing demoreing models, which is a novel application. The paper compares the demoreing performance against strong baselines used in Diffusion Transformer quantization. 

* Overall, the paper does a good job in describing all the components. The methodology section is overall quite simple and clear.

### Weaknesses
* Recent edge devices can handle models of up to a few billion parameters (fp16/bf16) hence it should be possible to deploy models consisting of million of parameters without aggressive quantization. The motivation is not entirely obvious from reading sections 1 and 2. 

* The sampling procedure described in section 3.2 seems like a high-variance estimate of the k-th percentile. However, the two methods are not compared theoretically, and the setting used for the empirical comparison are unclear. The computational cost of estimating distribution statistics is usually negligible since the calibration phase needs to be performed only once and yields lower variance estimate than the proposed sampling approach. Variance estimations are not reported in the experimental results in Table 1a. 

* Figure 4 is difficult to read because of the low density of the distribution tails. Please consider using a logarithmic scale.

### Questions
1. Can the authors elaborate on the statement “[edge devices] are also the most important application scenarios [for image demoiréing]” ? Although the relevance of the task is clear, this statement does not seem entirely obvious. Deploying models consisting of millions of parameters on the edge with recent hardware should also be possible without aggressive quantization. What is the ideal use case of QuantDemoire? 

2. What hyper-parameter ranges are compared to produce the results reported in Table 1a? What is the variance on the proposed sampling methodology when compared to the baseline? Can the authors further comment on the advantages of sampling vs percentile-based range-setting?

3. The paper proposes to keep the weight outliers in full precision instead of clipping them. Can the authors quantify the overhead introduced by $W_{outlier}$ matmul in terms of memory and latency? Is the bit width substantially lower than a less-aggressive quantization scheme (e.g. 5 bits) ? How is the sparse matmul performed efficiently? 

4. What is the effect of the two terms $\mathcal{L}_1$ and $\mathcal{L}_p$ on the loss? Is there a tunable weighting factor? I suspect that, due to the different number of dimensions the two terms might have a different scale.

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
3

### Summary
Image demoireing is an important vision task especially for edge devices such as smartphones, drones, and portable cameras.
Thus, model quantization is necessary to utilize high performance demoireing models in practical applications.
Existing quantization methods are not focused on demoireing so that they 1) overlook outliers in weights and activations, and 2) weaken representations in smooth regions, resulting in a huge performance loss.
QuantDemoire removes activation outliers through random sampling and keeps extreme weights in FP to mitigate the impact of outliers.
It also extracts mid- and low-frequency information by a recursive kernel to train the quantized model to preserve low-frequency features well.
Extensive experiments show that QuantDemoire outperforms existing methods.

### Strengths
* The work successfully identifies and addresses the frequency-related issue that is unique to the demoireing task.
* Effectively extracting low-frequency features through a simple convolution kernel.
* QuantDemoire achieves the state-of-the-art performance.

### Weaknesses
* Smoothing and clipping approach of outlier-aware quantizer strongly resembles existing methods such as OmniQuant [1].
* Preserving outliers in full-precision also resembles existing methods such as LLMint8 [2].
* Preserving arbitrary weight parameters in FP would induce computation inefficiencies because of hardware-unfriendly structure [3].
* From the ablation results in Table 3, the effect of frequency-aware calibration appears to be marginal.

[1] Shao, Wenqi, et al. "OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models." The Twelfth International Conference on Learning Representations.

[2] Dettmers, Tim, et al. "Gpt3. int8 (): 8-bit matrix multiplication for transformers at scale." Advances in neural information processing systems 35 (2022): 30318-30332.

[3] Lin, Ji, et al. "Awq: Activation-aware weight quantization for on-device llm compression and acceleration." Proceedings of machine learning and systems 6 (2024): 87-100.

### Questions
* Does sampling-based quantization occur online during inference or are scaling factors pre-computed during calibration?
* If it occurs online, how does it affect on throughput?
In my understanding, strength of smoothing based approach is that the smoothing factor is fused into weight so that the inference cost remains the same.
Is outlier-aware quantizer better than other online outlier-handling approaches such as OCS even without fusing?
* Could you further review related work and clarify how the proposed outlier-aware quantizer differs from existing methods?
* Could you provide additional ablation results under more diverse settings, such as different datasets or bit-widths?
* Does applying and training multiple kernels not lead to higher training cost or longer training time?

### Soundness
3

### Presentation
3

### Contribution
2
