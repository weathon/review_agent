# IntLoRA: Integral Low-rank Adaptation of Quantized Diffusion Models

- Decision: Reject
- Scores: 6, 6, 6, 5

## Abstract
Fine-tuning large-scale text-to-image diffusion models for various downstream tasks has yielded impressive results. However, the heavy computational burdens of tuning large models prevent personal customization. Recent advances have attempted to employ parameter-efficient fine-tuning (PEFT) techniques to adapt the floating-point (FP) or quantized pre-trained weights. Nonetheless, the adaptation parameters in existing works are still restricted to FP arithmetic, hindering hardware-friendly acceleration. In this work, we propose IntLoRA, to further push the efficiency limits by using integer type (INT) low-rank parameters to adapt the quantized diffusion models. By working in the integer arithmetic, our IntLoRA offers three key advantages: (i) for fine-tuning, the pre-trained weights are quantized, reducing memory usage; (ii) for storage, both pre-trained and low-rank weights are in INT which consumes less disk space; (iii) for inference, IntLoRA weights can be naturally merged into quantized pre-trained weights through efficient integer multiplication or bit-shifting, eliminating additional post-training quantization. Extensive experiments demonstrate that IntLoRA can achieve performance on par with or even superior to the vanilla LoRA, accompanied by significant efficiency improvements.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces IntLoRA, which uses integral low-rank parameters to adapt quantized diffusion models. First, Adaptation-quantization separation (AQS) is proposed to overcome the difficulty of vanilla zero-initialized LoRA parameters is not quantization-friendly. Then, Multiplicative low-rank adaptation is proposed to enable the fusion of quantized weights and LoRA parameters. Variance matching control is further proposed to make the distribution suitable for log2-quantization. Experimental results show that IntLoRA improves the efficiency of adaptation while maintaining performance.

### Strengths
1.	Reducing storage costs is significant for adaptation tasks.
2.	The idea of multiplicative low-rank adaptation is novel.

### Weaknesses
1.	The training costs are not reported in the paper.
2.	In section 5.2, the authors mentioned that “IntLoRA is the only one that achieves INT type adaptation weights”, which is not true since EfficientDM also uses quantized LoRA weights.

### Questions
See weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces IntLoRA, a pioneering method for adapting quantized diffusion models to optimize both training and inference efficiencies. It tackles the arithmetic inconsistency between low-rank parameters and pre-trained model weights by employing integer-based low-rank parameters. During training, the method applies quantization to the pre-trained model weights, and post-training, integrates these integer low-rank parameters seamlessly for efficient inference. To facilitate quantization-aware training, the authors propose the Adaptation-Quantization Separation (AQS), which allows for the coexistence of zero-initialized gradients and a quantization-friendly distribution. Additionally, a Variance Matching Control (VMC) mechanism is developed to finely tune the channel-aware variance, ensuring precise adaptation control.  Extensive experimental results indicate that IntLoRA not only competes with but often surpasses traditional methods in performance, significantly reducing memory usage and computational demands.

### Strengths
The paper demonstrates a strong and clear motivation by addressing a critical challenge in the domain of machine learning models—specifically, the adaptation of quantized diffusion models using integer low-rank parameters. This innovative approach effectively bridges the gap between maintaining precision and enhancing computational efficiency. Furthermore, the paper successfully resolves the prevalent issue of arithmetic inconsistency between pre-trained weights and adaptation parameters.

### Weaknesses
1.	The clarity of some explanations could be improved, particularly concerning why the all-zero distribution is not quantization-friendly. The text should elucidate why such a distribution necessitates a distinct quantizer design at the beginning of tuning. Providing a detailed example or analytical explanation on gradients would greatly enhance understanding, highlighting the specific challenges and potential impacts on the model’s performance during quantization.

2.	In the description of the VMC, the authors introduce a scalar $\alpha$ used as an exponent of $r$ to finely adjust the search for the optimal auxiliary matrix. However, the method for conducting this search remains unclear. It would be beneficial for the authors to specify the objective function used in this search process. Clarifying how $\alpha$ and $r$ are optimized, including the criteria and algorithms employed, would provide a more comprehensive understanding of the method’s effectiveness and implementation.

### Questions
1. The proposed method, IntLoRA, demonstrates substantial potential to enhance the performance of quantized models beyond simple adaptation. Integrating IntLoRA’s innovative techniques, such as integer low-rank parameters and network-wide reconstruction calibration, not only maintains but also significantly boosts the efficacy of models operating under quantization constraints.

2. The authors introduce the use of an auxiliary matrix in the AQS. It would be helpful to clarify whether the introduction of this auxiliary matrix introduces additional quantization difficulty like outliers to the pre-trained model weights.

3. The paper overlooks an important baseline in its experimental setup or discussion. The study “EfficientDM” by He et al., 2023, which similarly employs low-rank parameters for fine-tuning pre-trained models, is notably absent. Like IntLoRA, EfficientDM also allows for the merging of low-rank parameters into pre-trained model weights post-training. Including this baseline could enhance the comparative analysis, providing a clearer benchmark of IntLoRA’s performance against existing methodologies with similar strategies.

4. The authors merge integer low-rank parameters with quantized pre-trained model weights after training, a process that might lead to value overflow. It would be beneficial for the paper to address how this potential issue is managed. Specifically, detailing the mechanisms or techniques employed to prevent or mitigate overflow—such as scaling factors, normalization methods, or safeguards within the merging algorithm—would provide a clearer understanding of the reliability of the proposed method in handling data integrity during integration.

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
This paper introduces IntLoRA, a method that quantizes the LoRA branch to integers for parameter-efficient fine-tuning (PEFT) in diffusion models. Traditional approaches typically use 16-bit precision for LoRA branches and focus on quantizing only the base model during fine-tuning. However, for inference, these methods often require re-quantization when the LoRA branches are fused with the base model. Unlike prior methods, such as QLoRA, IntLoRA fine-tunes the LoRA branches in a quantization-aware manner. To maintain adapter performance, it incorporates techniques like Adaptation-Quantization Separation (AQS), Multiplicative Low-rank Adaptation (MLA), and Variance Matching Control (VMC), and supports both linear and log2 quantization. Extensive experiments across various downstream tasks—including subject-driven generation, controllable generation, and style customization—demonstrate the effectiveness of IntLoRA in both visual quality and model size reduction.

### Strengths
* The paper presents a strong motivation, addressing a key limitation in current quantization methods: the need for re-quantization after applying adapters. This limitation has prevented the wide adoption of quantization methods in diffusion models.
* The concepts of Adaptation-Quantization Separation (AQS), Multiplicative Low-rank Adaptation (MLA), and Variance Matching Control (VMC) are innovative and well-supported by mathematical analysis. Their effectiveness is further validated through comprehensive experimental results.
* The authors' support for log2 quantization, a challenging but highly hardware-efficient setting that further enhances the practical utility of IntLoRA.

### Weaknesses
* The paper exclusively focuses on weight-only quantization, where activations remain at 16-bit precision. Since diffusion models are primarily compute-bound, quantizing only the weights does not yield speedups during inference. Additionally, in a weight-only quantization setting, users can choose not to fuse adapters, thereby avoiding re-quantization. As a result, the primary benefit of IntLoRA is just a reduction in adapter size compared to conventional PEFT methods—a relatively minor contribution, given that LoRA adapters are already small.
* he proposed PEFT method does not reduce training memory usage compared to existing PEFT methods (e.g., QLoRA). Furthermore, quantization-aware training of adapters may slow down training and even increase memory usage. The paper should discuss the training speed and memory usage.
* Clarification on storage requirements for $\mathbf{R}$ is necessary. Storing $\mathbf{R}$ after training the adapters would require significant storage, which may limit the method's practicality. If $\mathbf{R}$ is not stored, the authors should explain how it is obtained during inference. Also, note that $\mathbf{R}$ should derive from $\mathbf{W}_{\text{round}}$ rather than $\mathbf{W}$, as only quantized weights are accessible during PEFT.
* Writing:
  * Using "FP" to refer to original LoRA training is ambiguous, as there are low-precision formats like FP8 or FP4. Consider using "16-bit" instead. Note also that QLoRA uses NF4 for weight representation, which is not integer-based. Therefore, Lines 190–202 need to be rewritten.
  * Line 156: The dimension of $\mathbf{B}$ should be $d \times C_{\text{in}}$.
  * Figure 2: The caption lacks detail, and the figure is difficult to interpret.
  * Line 229: Refer to Section 5.3 on how to determine $\mathbf{R}$.
  * Line 265: Need to clarify $\rho$ is the correlation coefficient.
  * Table 4: Correct "FP addtion" to "FP addition".
  * Figure 7: Add axis labels for clarity.

### Questions
* Line 252: Could you provide any insights or justification for the assumption that $\mathbf{A}\mathbf{B}$ is orders of magnitude smaller than $\mathbf{R}$?
* Line 260: What is the purpose of a zero-mean adaptation term, and why is it better?
* Table 4: Why does the storage reduction exceed $8\times$ (e.g., $9.8/1.2 > 8$ and $0.34/0.04 > 8$)? If the model is stored in 32 bits, the reduction should be slightly less than $8\times$ due to the additional overhead from zero points and scaling factors. If stored in 16 bits (the preferred format), the reduction should not exceed $4\times$.
* Figure 6: Could you clarify the difference between VMC $\sigma_{\mathbf{R}}$ and the appropriate $\sigma_{\mathbf{R}}$? Additionally, how can one observe the large quantization error from the second image in the second row of the figure?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The authors propose IntLoRA, which employes INT low-rank parameters to adapt the quantized diffusion models, to address the arithmetic inconsistency between the quantized pre-trained weights and adaptation weights. All the process will operate directly on INT arithmetic in training and merged weights at inference stage will also be on INT format. Apparently, it will speed up both the training and inference stage.

### Strengths
1) The presentation of challenges and contributions are clear to me. 
2) The presentation of methods and experiments are clear to me.

### Weaknesses
1) Recent works, e.g., EfficientDM and LoftQ, have worked out quantized diffusion models using LoRA. Different from this paper, EfficientDM focus on quantized merged weights at inference stage. Can you compare with it? BTW, I am doubting the importance of the speed-up for the training stage as the pretrained weights is fixed, only small adapters are updated.  Can you show how much speed-up as compared to full-precision? or at least theorotically show the improvement?

2) From my understanding, the proposed methods also can be used in NLP tasks. As the baselines have results on NLP tasks in their papers, why not verify the proposed methods on NLP tasks? Could you show some experimental results on NLP tasks?

3) For diffusion models, seems it have standard/common verification on some tasks, please check EfficientDM paper. As all baselines are implemented by authors, I just doubt the reliablity because of hyper-parameter tuning. Maybe it is better compare with SOTA results in paper.

### Questions
Have asked in Weakness section.

### Soundness
3

### Presentation
3

### Contribution
2
