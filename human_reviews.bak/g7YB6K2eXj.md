# Double Rounding Quantization for Flexible Deep Neural Network Compression

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 5, 5, 5

## Abstract
Model quantization is widely applied for compression and acceleration of deep neural networks, due to its simplification and adaptability. The quantization bit-width is typically predefined for quantizing a given neural network. However, the bit-width settings vary in different hardware and transmission demands, which will induce considerable training and storage costs. Therefore, the scheme of once-joint training for multiple bit-widths (multi-bit) is proposed to address this issue. In this paper, we propose a Double Rounding quantization method that can save the highest bit-width model instead of the full-precision counterpart and fully exploits the representation value range. Nevertheless, the performance during once-joint training degrades significantly due to inconsistent gradients between high-bit and low-bit quantization. To tackle this problem, we set the learning rate of multi-bit to proper values in an adaptive manner during training. We also apply our method for mixed-precision super-net and provide a novel training strategy with weighted probability. Experimental results demonstrate the proposed method outperforms the SOTA once-joint quantization-aware methods on ImageNet datasets. The code will be available soon.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper introduces a new method for multi-bit joint training and mixed-precision super-net training in deep neural networks. This method aims to make model compression more adaptable for varying hardware and storage needs. The authors propose a Double Rounding quantization technique, which only requires storing the highest-bit integer model, making it more storage-efficient. They address the challenge of inconsistent gradients in multi-bit training by adaptively adjusting learning rates for different bit-widths. Additionally, a weighted probability training strategy for mixed-precision super-nets is introduced, improving the method's versatility. The paper also presents a decision-making approach using integer linear programming to find the best bit-width combination for different model layers, targeting optimal solutions. Experiments on CIFAR-10 and ImageNet demonstrate the effectiveness of the proposed method.

### Strengths
The paper is well written and easy to follow. The proposed double rounding quantization technique, which offers a storage-efficient solution by only necessitating the storage of the highest-bit integer model.

### Weaknesses
The paper has several notable weaknesses, including lack of empirical evidence, ambiguities in methodology, unclear contributors to performance, missing Baseline, etc. Please see questions for details.

### Questions
1.	In Section 3.2, the authors attribute the notable divergence in convergence rates between the highest and lowest bit-widths during once-joint training of multi-bit models to the inconsistency in gradient updates between high-bit and low-bit quantization phases. However, this assertion lacks empirical evidence, as no experimental results are presented to highlight these inconsistent gradients. Additionally, the introduced Multi-LR approach, which adjusts learning rates based on different bit-widths, is heuristic in nature. It would be beneficial to understand if there is an underlying rationale or guiding principle for the selection of these learning rates to ensure they are both effective and justifiable.

2.	In Section 3.3, the authors introduce the use of weighted probability for the supernet training. However, the methodology behind computing the sampling probability for various bit-widths within a given layer is not explicitly explained. This omission can lead to ambiguities in replicating the approach and understanding its full implications. Providing a clearer, step-by-step computation process would enhance the reproducibility of the proposed method.

3.	In Section 3.3, the authors mention the capability of their method to swiftly produce multiple candidate configurations under specified constraints by adapting the ILP (Integer Linear Programming) algorithm post super-net training. Yet, this assertion lacks empirical backing as the section doesn not offer any associated experimental results. Providing tangible evidence or case studies would substantiate this claim.

4.	The authors propose double rounding quantization that only keeps the highest bitwidth model instead of the full-precision counterpart. However, the performance drop brought from double rounding quantization is not clearly investigated. A deeper dive into the proposed method on the model's accuracy and efficiency would have provided a more comprehensive understanding of its real-world applicability and limitations.

5.	In Figure 2, the authors present the gradient statistics of activation scales for ResNet20, offering insights into the network's behavior. However, a critical aspect that is not clarified is the initialization of these scales. For a fair and meaningful comparison, it is essential to ascertain whether all scales started from the same initialization point.

6.	In Tables 1 and 2, the proposed method shows a significant performance improvement over the state-of-the-art methods. However, the specific components of the proposed method that primarily drive this performance enhancement remain ambiguous. A breakdown or ablation study highlighting the individual contributions of each component would provide deeper insight into the key drivers behind the observed improvements.

7.	The experimental section appears to lack a crucial baseline comparison. It would be valuable to understand how the proposed method stacks up against an independent approach that trains different bit-widths separately. Such a comparison would shed light on the relative efficacy and advantages of the proposed joint training technique.

8.	A pivotal detail seems to be overlooked in the paper. It is unclear how many samples the authors utilized to compute the Hessian trace across various layers.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The key innovations in the study are as follows: "Double Rounding" is proposed to maintain integer-weight parameter storage without compromising representation values. "Multi-LR" introduces a training strategy for multi-bit models that effectively reduces the training convergence gap between high-precision and low-precision models. "Weighted Probability" determines the access probability of bit-width for each layer based on the layer's sensitivity, aligning with the subnetwork's decision-making process during inference. Experimental results on ImageNet datasets demonstrate that the proposed method surpasses state-of-the-art techniques across various mainstream network architectures.

### Strengths
1. The paper exhibits a well-defined structure, making it easy to navigate and understand.

2. The primary objective is to address the complex issue of multi-bit quantization and demonstrate notable performance improvements compared to similar methods.

### Weaknesses
1. I find the distinction between Adabits and the proposed Double Rounding in the figure somewhat minimal. It seems that the primary difference lies in the altered value range. Could these two methods essentially be equated with one having a zero point and a different scale value?

2. It may be beneficial to include the algorithm for weight probability in the main paper. This approach could reduce the volume of explanatory text, ultimately enhancing clarity for readers.

3. Evaluating the proposed method on large transformer models as well as tiny models, that are particularly susceptible to the effects of quantization, would provide valuable insights.

4. In my overall assessment, I believe that the three proposed techniques may fall short of meeting the publication standards.

### Questions
1. Could you provide further clarification regarding the distinction between Adabits and Double Rounding?

2. The results presented in Table 1 raise the question of whether Knowledge Distillation (KD) plays a more significant role than the three proposed techniques.

3. Table 1 exclusively presents uniform bit-width results. Is there a specific reason for not including mixed precision results in the table?

4. In Table 4, the epoch duration remains consistent, but there is a variance in training cost. Can you clarify the specific factor or factors that account for this difference?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
A multi-bit quantization framework (Double Quantization) is proposed, which quantizes a pre-trained model for once and enables inference with different pre-defined bit-width. To help convergence, a Multi-LR method is introduced to use seperate learning rate for each bit-width. Mixed-precision is also studied.

### Strengths
This paper deals with an important problem of network quantization, i.e., the multi-bit quantization problem. The proposed Multi-LR method seems to be useful for stable training. Experiments with various bit-widths and mix-precision results are provided.

### Weaknesses
The proposed multi-bit quantization framework consists of three main parts, i.e., the Double Rounding quantization scheme, the Multi-LR learning rate selection method, and the Weighted Probability mixed-precision method. However, these improvements seems to be a little bit incremental.
- I didn't see the difference between the double rounding quantization and adabit quantization. The adabit quantization can also represent with [−1, 1] but not limited to [0, 1]. 
- The multi-lr method is a hyper-parameter tuning, which is more like a tuning trick to me.
- Both mixed-precision quantization and mixed-precision based on multi-bit quantization have been widely studied in previous works. 

The improvements over Adabit are not quite significant if KD is not used.

### Questions
Does all baseline methods use the same pre-trained model in Table-1? The full-precision baseline accuracy should be reported.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors present a novel approach to mixed-precision quantization, allowing for post-training bit-width selection. This method uses quantization-aware training, with the central concept being the training of a model at the highest permitted bit-width and obtaining lower precision representations through bit shifting. The authors introduce a double rounding technique that allows switching between high and low precision configurations without necessitating retraining. To address the challenges associated with simultaneously training a model for varying bit-widths, the authors advocate the use of distinct learning rates for quantization scaling parameters across different configurations, where fewer bits correspond to a smaller learning rate. The proposed method builds upon the Bit-Mixer framework (Bulat & Tzimiropoulos, 2021) with the following key differences that enhance its performance:
1. The incorporation of double rounding for efficient switching between low and high precision via bit shifting.
2. The utilization of the trace of the Hessian information during the training phase to determine the bit precision for each layer separately, with lower trace values indicating lower precision.
3. The application of different learning rates for each bit-width configuration to mitigate training instability.
4. The use of probabilities that align with Hessian information instead of employing uniform probabilities.
5. The adoption of an Integer Linear Programming (ILP) approach to determine the optimal configuration while adhering to specified constraints (e.g., FLOPs, storage).

Empirical validation on various models applied to ImageNet and CIFAR-10 datasets demonstrates the superior accuracy achieved by the proposed algorithm while using fewer or equivalent bit-widths.

### Strengths
1. The method allows for training models capable of dynamically adjusting their precision levels, offering adaptability for deployment on diverse edge devices.
2. Leveraging the Hessian information of each layer during bit-width assignment in the training phase enables an estimation of the number of bits required for each layer.

### Weaknesses
1. The introduction of additional $\mathcal{O}(n^2)$ batch normalization layers, although a minor concern, should be noted, as it may lead to additional storage costs. Nonetheless, it's worth highlighting that the size of batch normalization layers is typically smaller than that of Linear or Convolutional layers.
2. The paper could benefit from more detailed explanations of key techniques, such as the use of the Hessian trace, the precise formulation of the ILP problem, and the weighted probability method.
3. The use of Integer Linear Programming (ILP) to find the optimal bit-width configuration may be computationally intensive due to the NP-completeness of the problem. Depending on the problem size, achieving convergence to the optimal configuration may require a substantial amount of time.

### Questions
N/A

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
