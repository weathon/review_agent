# Compensating for Nonlinear Reduction with Linear Computations in Private Inference

- Decision: Reject
- Scores: 1, 5, 6

## Abstract
Increasingly serious data privacy concerns and strict regulations have recently posed significant challenges to machine learning, a field that hinges on high-performance processing of massive user data.
Consequently, privacy-preserving machine learning (PPML) has emerged to securely execute machine learning tasks without violating privacy.
Unfortunately, the computational cost to securely execute nonlinear computations in PPML models remains significant, calling for new neural architecture designs with fewer nonlinear operations.
We propose Seesaw, a novel neural architecture search method tailored for PPML. 
Seesaw exploits a previously unexplored opportunity to leverage more linear computations and nonlinear result reuse, in order to compensate for the accuracy loss due to nonlinear reduction.
It also incorporates specifically designed pruning and search strategies to efficiently handle the much larger design space including both nonlinear and linear operators.
Compared to the previous state-of-the-art PPML for image classification on ImageNet, Seesaw achieves $1.68\times$ less latency at 71\% iso-accuracy, or 4.59\% higher accuracy at iso-latency of 1000K ReLU operations.

## Human Reviews

## Human Reviewer 1

### Rating
1: strong reject

### Rating Number
1

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper introduces approaches to improve the performance of cryptographically secure private inference by improving the network's ReLU efficiency. The authors leverage neural architecture search to compensate for the reduced number of ReLU activations by augmenting the network's linear operations (FLOPs). The achieved performance is on par with state-of-the-art ReLU-pruning methods.

### Strengths
$\bullet$  Experimental results on Imagnet dataset. 

$\bullet$ The ReLU-Accuracy performance of the proposed baseline model is on par with existing ReLU-pruning methods. 

$\bullet$  Ablation studies are presented well to show the efficacy of the pruning method and the merits of reusing ReLUs.


$\bullet$ Proposed methods are very well presented and easy to understand.

### Weaknesses
$\bullet$ **Novelty of the method:** The proposed approaches for improving ReLU efficiency at the expense of FLOPs count are not novel at all. Previous work, such as CryptoNAS and Sphynx, has already explored this concept by maintaining a constant ReLU count per layer, which in turn increases the FLOPs count (however, these studies did not provide detailed FLOPs count information). Furthermore, this paper fails to introduce any fresh insights or observations that shed light on how networks can trade ReLUs for FLOPs. The insights presented in Section 4.4 of the paper are already well-established.

$\bullet$ **FLOPs-cost are ignore:** This is the core-issue with the paper. The authors presented the results with online latency and ignored their impact on end-to-end latency, a trend also seen in works like SENet, SNL, Sphynx, DeepReDuce, and CryptoNAS. Nonetheless, these prior work on PI improved the ReLU efficiency of the given baseline networks (mostly ResNets and WideResNets) *without increasing their FLOPs counts* (Although WidResNets have 4x to 5x higher FLOPs than ResNets). 

Delphi assumes that no matter how many FLOPs are there in networks, they can be processed offline. However, in real-world scenarios, private inference requests arrive at non-zero rates. Even at low arrival rates, processing the entire FLOPs offline becomes impractical due to limited computing resources, storage, communication bandwidth between server and client, and time constraints arising from the non-zero request-arrival rate. Consequently, offline costs are no longer truly offline, and FLOPs start affecting real-time performance, as illustrated in Figure 7 of the paper [1]. This effect can be exacerbated by networks with higher FLOP counts, as proposed by the authors.


**The argument of online vs. offline latency is only valid if the optimization is performed for a single private inference in isolation.** FLOP penalties can only be disregarded when there are no inference arrivals or when an accelerator offering more than 1000x speedup is employed. Even with complete FLOP parallelization, such as using LPHE in [1], end-to-end performance improves but does not eliminate FLOP costs.

The authors cite [1] to support the claim that ReLU is 300x costlier than FLOPs, but this argument is valid only for online overhead. When considering end-to-end latency, FLOPs are shown to be 4.8x more expensive than ReLUs, as demonstrated in Table 1 of [1]. Therefore, the authors should have provided a FLOPs comparison with ResNet18, WRN22x8, and CryptoNAS.

In summary, *this paper does not contribute any new perspectives on private inference and fails to advance the understanding of current gaps in the field, rendering it less relevant for the ICLR audience.*


[1] Garimella et al., "Characterizing and Optimizing End-to-End Systems for Private Inference," ASPLOS 2023.

### Questions
See the points in weakness.

Additionally, in line with the proposed method, increasing the network's width has been shown to  [2] [3] (however,  at the expense of FLOPs). How does the proposed method's trade-off between ReLUs and FLOPs counts compare to the straightforward approach of widening the baseline networks, such as ResNet18? 



[2] Dollár et al., Fast and accurate model scaling. CVPR'21. 


[3] Lee et al.,  Wide neural networks of any depth evolve as linear models under gradient descent. NeurIPS'19.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper addresses the issue of accuracy loss due to approximating the activation function during the implementation of a privacy-preserving artificial intelligence model. It proposes a method for modifying the model to add convolution operation blocks or reuse activation results to restore accuracy. In the process of modifying this model, it suggests utilizing Neural Architecture Search to automate the model adjustments.

### Strengths
When implementing information security artificial intelligence models using homomorphic encryption or multi-party computation, the issue of accuracy loss due to activation function approximation has been extensively discussed in many papers and is one of the fundamental problems to address. The approach of solving this problem by designing a Neural Architecture Search (NAS) with meaningful exploration directions, such as the addition of linear operations or activation reuse, is considered a novel method. While papers using NAS algorithms have existed before, I believe there is a novelty in setting the exploration directions. I consider this a valuable method that can be effectively used in subsequent papers to design information security artificial intelligence models.

### Weaknesses
When reviewing the experimental results using this technology, I found it ambiguous whether the corresponding privacy-preserving machine learning model were actually implemented with the homomorphic encryption and the multiparty computation. If homomorphic encryption and multi-party computation were used, specific encryption parameters and communication amounts should be provided, but such information was not clearly presented. While the NAS algorithm itself is novel, the lack of thorough discussion on security during its implementation and validation makes this paper appear incomplete as a paper on privacy-preserving AI. Simply presenting a good algorithm may not be sufficient for approval if it's not clear whether actual cryptographic algorithms were used in the implementation.

### Questions
1) Please all of the details about the cryptographic parameters in your implementation.
2) Please give the communication costs for each result.
3) Did you implement your models with homomorphic encryption and multiparty computation? or did you only compute the expected runtime for your results without the implementation of the privacy-preserving machine learning system?

My rating will be changed with these questions.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper Seesaw presents a new approach in Privacy-preserving machine learning which overcomes state-of-the-art issues on the decrease of accuracy when reducing the amount of non linear operators for a given architecture. The authors designed a new Neural architecture search where they consider to use more linear computations and to reuse the results from non linear operators and thus compensate from having less of them. The introduced design also enables more representational capability of the model with the increase on the linear operators. The proposed design outperforms SOTA especially on Imagenet with better accuracy with fewer non linear operations.

### Strengths
- The paper is well presented with understandable figures on the design of Seesaw
- The proposed approach gives competitive results compared to SOTA. Especially on Imagenet, we observe a big improvement

### Weaknesses
- The authors do not provide an analysis on the weighting parameters used in the loss function regarding the pruning of the linear branches and non linear operators.
- The results on CIFA100 are less significant compared to the one on Imagenet
- It would have been nice to have an additional dataset for the evaluation to see the result tendency compared to CIFAR100 and Imagenet

### Questions
- Is there a reason which explains why the results on accuracies on Imagenet and CIFAR100 are different with respect to SENet? We observe better results of Seesaw compared to SENet on Imagenet
- Section 4.1, you say that with abundant ReLu budget, Seesaw can outperform Resnet models, do you prove this statement somewhere? or It is just the assumption knowing that you have more linear operators.

### Soundness
2 fair

### Presentation
3 good

### Contribution
3 good
