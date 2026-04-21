# Overcoming Distribution Mismatch in Quantizing Image Super-Resolution Networks

- Avg Score: 5.25
- Decision: Reject
- Scores: 6, 3, 6, 6

## Abstract
Quantization is a promising approach to reduce the high computational complexity of image super-resolution (SR) networks. However, compared to high-level tasks like image classification, low-bit quantization leads to severe accuracy loss in SR networks. This is because feature distributions of SR networks are significantly divergent for each channel or input image, and is thus difficult to determine a quantization range. Existing SR quantization works approach this distribution mismatch problem by dynamically adapting quantization ranges to the variant distributions during test time. However, such dynamic adaptation incurs additional computational costs that limit the benefits of quantization. Instead, we propose a new quantization-aware training framework that effectively Overcomes the Distribution Mismatch problem in SR networks without the need for dynamic adaptation. Intuitively, the mismatch can be reduced by directly regularizing the variance in features during training. However, we observe that variance regularization can collide with the reconstruction loss during training and adversely impact SR accuracy. Thus, we avoid the conflict between two losses by regularizing the variance only when the gradients of variance regularization are cooperative with that of reconstruction. Additionally, to further reduce the distribution mismatch, we introduce selective distribution offsets to layers with a significant mismatch, which selectively scales or shifts channel-wise features. Our proposed algorithm, called ODM, effectively reduces the mismatch in distributions with minimal computational overhead. Experimental results show that ODM effectively outperforms existing SR quantization approaches with similar or fewer computations, demonstrating the importance of reducing the distribution mismatch problem.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper explores how to overcome the distribution mismatch in quantizing SR. Specifically, the authors intuitively reduce the distribution mismatch by directly regularizing the variance of features when the gradients of variance regularization are cooperative with that of reconstruction. In addition, the authors introduce selective distribution offsets to layers with a significant mismatch, which selectively scales or shifts channel-wise features. The extensive experiments demonstrate the effectiveness of the proposed method.

### Strengths
1.	I enjoy the analyses of the distribution mismatch and conflict of the existing SR network. These observations are non-trivial and are critical for practical quantizing SR. 
2.	The proposed method is novel and reasonable. The idea is simple yet effective. The authors have comprehensively demonstrated the proposed methods from the perspective of optimization.
3.	The paper is well-written and is easy to follow.

### Weaknesses
1.	It would be better if the authors can provide more details about Gradient conflict ratio.

2.	In addition to the variance regularization, the selective distribution offsets also employ a learnable parameter about the standard deviation of the features. It would be better if the author could provide more discussions about the relation of these two terms. 

3.	Does the selective distribution offsets are learned on the features not processed in cooperative variance regularization?

### Questions
See the above weakness.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposes a new quantization-aware training technique that relieves the mismatch problem via distribution optimization. Specifically, the authors use variance regularization loss, cooperative variance regularization and selective distribution offsets to reduce such mismatch. Experiments demonstrate the effectiveness of the proposed method.

### Strengths
This paper proposes a quantization framework to address the distribution mismatch problem in SR networks without dynamic modules. The proposed method achieves state-of-the-art performance with similar or less computations.

### Weaknesses
The experiment section can be improved. Please refer to the details below.

### Questions
1. The main motivation is that feature distributions of SR networks are significantly divergent for each channel or input image. In Figure 1, does the distribution mismatch only occur in the SR network? Does such a distribution mismatch occur in other networks? In addition, could you show the distribution after quantization?

2. In the experiments, the authors mainly use EDSR, EDN and SRResNet. However, these methods are very old. Could you compare the new SOTA SR networks, e.g., SwinIR?

3. In Table 1, for Bit=2, the results of EDSR-DAQ do not correspond to the original results of the DAQ paper. Could you discuss these results?

4. The experiments only address the scale of 4. It would be better to conduct more experiments on other scales and put the results in supplementary.

5. In the ablation study, could you conduct an experiment with only the cooperative variance regularization? In addition, the network with only the selective distribution offsets is comparable with Coop.+Var. Reg.+Sel. Off. This result demonstrates that Coop.+Var. Reg. are not  important.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper aims to deal with the inherent distribution mismatch of the features in quantizing image super-resolution. To this end, the paper introduces a variance regularization loss, which can cooperate well with the reconstruction loss by computing the signs of gradients. Furthermore, the paper proposes to apply shifting/scaling offsets to layers with a large mean/deviation. The proposed quantization framework ODM is evaluated on three representative SR models in the main paper. ODM exhibits better performance over competitors using a small storage size and low BitOPs.

### Strengths
The paper proposes the variance regularization loss, which can regularize the distribution diversity beforehand and cooperate well with the reconstruction loss. The selective distribution offsets further reduce the variance distribution. The proposed methods are based on analyses and observations. The experimental results are competitive by achieving high performance and reducing computation overhead. The writing is easy to follow.

### Weaknesses
The comparisons seem not fair in terms of training epochs, and the proposed method does not reduce BitOPs compared to the previous method, i.e., DDTB, which contradicts the motivation of the method.

### Questions
1. The authors reproduce the results of other methods using the same training epochs. Does the number of epochs influent the performance of other methods? Why not use their optimal training epochs for comparisons?
2. Compared to competitors, the authors use seemingly complicated methods to address the quantizing problem beforehand. Will the proposed method increase the training time?
3. We can observe from Tab. 4 that the proposed method achieves a better tradeoff between the storage size and BitOPs, to be precise. What makes the ODM need higher storage space than DAQ.
4. The verb is missing in the sentence after Eq.4.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This manuscript focuses on quantizing super-resolution (SR) networks. Authors discover that the difficulty of quantizing SR networks is because of the fluctuation in activation distribution, which is significantly different in each channel. They propose ODM, a QAT framework to overcome the distribution mismatch problem by regularizing the variance in features using a new loss term. It mainly includes two contributions. First, it regularizes the gradients to ensure the losses are not in conflict. Second, it introduces a channel-wise offset that reduces the distribution mismatch.

### Strengths
The motivation is clear and strong. The authors design a new loss term and channel-wise quantization factors to regularize the activation to make the network easy to quantize. The plug-in module can be introduced and gain improvements in other networks and tasks that has variance feature distribution. 

The experimental results are comprehensive. The proposed methods show consistent improvements on various SR networks and datasets (But the improvements are not that significant). 

The figures and illustrations are easy to understand and targeted to the problem. And the overall writing is easy to follow.

### Weaknesses
There are lots of quantization methods that adopt channel-wise scaling and offsets. Although the channel-wise feature variance seems to be more severe in super-resolution networks, the channel-wise quantization factor is not novel. 

The paper mainly solves one problem with two strategies. I wonder if they are repeated. The regularization loss makes the activation variance smaller in each channel, which is easy to quantize. And the channel-wise quantization factor quantizes the features in a channel-wise manner that will not be affected by the value differences between channels. The experimental results in the ablation study also show that the two methods are not orthogonal. Combining the two methods together can only outperform a little compared with solely using one of them. 

The proposed methods may be too simple and need more insightful analysis and discoveries.

### Questions
x_i in Eq. (2) denotes the feature (activation). However, I wonder if it will lead to the homogenization of features since they are expected to have a low standard deviation. Did the authors try to minimize the difference in the mean of each channel?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good
