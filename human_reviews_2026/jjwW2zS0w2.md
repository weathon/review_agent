# Soft Quantization Activation Functions For Deep Learning

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 2, 8, 6

## Abstract
Activation functions (AFs) are a cornerstone of deep learning, providing the crucial nonlinearity needed for network expressiveness. However, widely used AFs like ReLU and GELU are fixed and non-adaptive, offering limited nonlinearity and often necessitating larger, more complex architectures to capture intricate functions. This paper introduces a new family of trainable, architecture-agnostic AFs called Soft Quantization Activation Functions (SQUAFs). We show theoretically that SQUAFs can approximate any continuous nonlinear one-dimensional function with arbitrary precision. Our extensive experiments demonstrate that networks equipped with SQUAFs consistently outperform their counterparts using existing AFs across diverse tasks. Specifically, we achieve orders-of-magnitude error reduction in function fitting, up to 25.27 dB gain in image fitting, and significant accuracy improvements in image classification and large language model (LLM) fine-tuning. Moreover, SQUAFs (1) enable smaller models to surpass larger ones trained with conventional AFs, and (2) can reduce the inter-device communication cost in model-parallel settings by up to 9-fold while still improving accuracy. These results highlight SQUAFs as a simple yet powerful drop-in replacement for standard AFs, offering both theoretical expressiveness and practical performance gains.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The proposed activation function (AF) is an impressive work highlighting the issues with simple AF such as ReLU which needs deeper models to capture non linearity and trainable AFs like PReLU, swish etc which can capture non linearity more effectively but by adding extra parameter makes distributed training more complicated (communication between different nodes to share updated parameters). Trainable, architecture-agnostic AF, Soft Quantization Activation Functions (SQUAFs) proposed by Authors can approximate any continuous nonlinear one-dimensional function with arbitrary precision. SQUAFs models are claimed to be better than counterpart bigger models trained using existing AFs. Additionally, it is shown to save more than 39x communication cost compared to models using trainable parameters in distributed computing setting. Results of four different tasks shows consistent improvement in performance relavie the the baseline models. 

This paper targets an important issue and the formulation of the problem and solution is interesting but the improvement in accuracy is not significant (given the the impact of new AFs in training is also now known) and limitation of 1D approximation may limit model expressivity. Some of the good results are on very small datasets which does not convey the real impact of this method. This is a weak reject and the score can be changed based on Author’s explanation as indicated in the limitations section.

### Strengths
- Authors have proposed a new Activation function which is trainable (but with partial derivatives),  adaptable to quantization and it is architecture agnostic to improve the performance of four different class of models
- Mathematical formulation of SQUAF is interesting and intuitive and it focuses on overcoming the limitations of existing AF.
- SQUAF models consistently have high accuracy compared to baseline models
- Designed as a plug-and-play replacement for existing activations in any architecture.
- Detailed results comparing with other activation functions such as GELU, PRELY, SWISH and few more.

### Weaknesses
- Table 4 : ResNet18 baseline performance is using ReLU (69.76) but if you replace ReLU with slightly more complex AF (SiLU), it will improve accuracy to almost similar to what can be achieved by SQUAF. Authors are encouraged to check those results and discuss the merit of using SQUAF in this context.
- Authors are also encouraged to check the same for models trained on CIFAR as it can be trained much faster compared to ImageNet.
- Table 7 results are impressive but CIFAR100 is too small and non complex dataset so showing a competitive results on larger datasets will make their claim stronger.
- Overall the improvement in accuracy does not seem to be significantly high.
- Trainable activation parameters could introduce extra computational or tuning cost, especially in large models. it will also be interesting to see the impact of these AFs on training complexity/training time etc.
- It’s not clear how much gain comes purely from adaptability versus other design factors.
- The approximation proof is restricted to one-dimensional functions and generalization to high-dimensional activations may be non-trivial. This can result in expressivity gap. Without extending the theory, it’s unclear if SQUAF’s adaptivity at the neuron level translates into provably higher expressivity at the layer or network level.

### Questions
- Do Authors think that for resnet18/imagenet results achieved by SQUAF will be better compared to just replacing ReLU with SiLU 
- What are the training overheads? can it be quantified?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper sets out to study more sophisticated activation functions than those currently employed in DNNs.   The paper proposes a family of activation functions, SQUAFs, that are based upon probability distributions combined over intervals (soft quantization).  There is some analysis of the representation ability of these in the appendices.  Empirical evaluation shows the activation functions can represent some test functions better.  Finally, the activation functions are applied to more standard DNN settings and improvements are claimed.

### Strengths
The paper highlights an interesting problem, how the design of activation functions can be considered an alternative to wider or deeper networks.

### Weaknesses
Making the activation function more complex increases computational costs.  There is some evaluation of the impact on inference latency in the appendix but the main paper did not seem to discuss the cost of the new activation functions and I didn't see any analysis on the potential impact on training times.  Similarly, the increased complexity of the activation function seems to come at the cost of increasing the number of parameters (the $y_i$'s and $L$) of the model and it is unclear if the performance comparisons are fair comparisons in terms of number of (modifiable) parameters.

The motivation on the first page seems to be trading off activation function complexity versus increased number of weights, but this tradeoff does not appear to be quantified in the results section.  

It was unclear what the motivation was for evaluating how closely a function can be approximated.  

It was also unclear what the particular motivation was for using quantization (Section 2.2 and 2.3) for the activation function.

The claim that KD using SQUAFs can increase accuracy above the teacher seems highly suspect and as if some form of overfitting is occurring.

### Questions
What is the motivation for introducing (soft) quantization in this paper? 

Are the parameters introduced by SQUAF (Equation 5 to 7) meant to be trained or are they meant to be fixed?  If they are fixed how were they chosen in your experiments?   If they are trainable, what is the increase in trainable parameters in the overall model?

What is the relationship of i and j in P2 on page 5?

Line 316: "We consider fitting two sinusoidal functions of increasing complexity" -- Why do this experiment?

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces a new activation function called Soft Quantization Activation Functions (SQUAFs). The activation function is specific to a layer, and its shape is controlled by three parameters. The authors show that SQUAF can approximate any 1-dimension function to an arbitrary precision. Experimental results are promising and show the approach outperforming multiple existing AF on varied benchmarks ranging from regression to LLMs.

### Strengths
1. The experimental results are convincing. The authors create models for regression, CIFAR-100 classification, LLM-finetuning and show that SQUAF outperforms other popular AF, including RELU, GELU, and SiLU. 
2. They provide a proof for approximating a 1-D function using SQUAF.

### Weaknesses
1. Figure 1 results look impressive, but are not very convincing; the model might be memorizing things and might not generalize better. This concern is removed later with training and test results on other datasets. 
2. The 1-D approximation proof is limited and not extended to the general multidimensional case. 
3. I am not convinced about reducing communication costs with SQUAF-P. The other results are sufficiently impressive, so I do not see a need to include this here. 
4. The proposed AF does lead to some slowdown, as shown in Table 11.

### Questions
It would be great if the authors could remove the half-baked claims from the paper. Otherwise, I do not see any major issues that need fixing.

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
The paper introduces Soft Quantization Activation Functions that are family of trainable functions that can approximate any continous 1D function with any arbitrary precision. The activation functions can replace fixed activation functions like ReLU and GELU and the paper demonstrates accuracy improvements across MLPs, CNNs and transformers. The proposed activation functions help reduce communication costs.

### Strengths
1. Proposed activation functions have theoretically proven universal approximation and differentiability.
2. Consistent performance gains across diverse tasks and architectures.
3. Reduces inter-device communication cost while improving accuracy.
4. Paper demonstrates improvements in fine-tuning, classification etc.

### Weaknesses
1. The proposed activation functions add additional trainable params which might hinder the optimzation process.
2. Generalization of this method across larger models is unknown.

### Questions
1. How sensitive are results to initialization of the quantization parameters (y, z, α)?
2. Do you have more results on more realistic LLM workloads (large dataset, models (1B-8B scale)) on different benchmarks?

### Soundness
3

### Presentation
3

### Contribution
3
