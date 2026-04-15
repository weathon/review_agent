# Neural Architecture Search for TinyML with Reinforcement Learning

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 5, 3, 5, 5

## Abstract
Deploying Deep Neural Networks (DNNs) on microcontrollers (TinyML) is a common trend to process the increasing amount of sensor data generated at the edge, but in practice, resource and latency constraints make it difficult to find optimal DNN candidates. Neural Architecture Search (NAS) is an excellent approach to automate this search and can easily be combined with DNN compression techniques commonly used in TinyML. However, many NAS techniques are not only computationally expensive, especially hyperparameter optimization (HPO), but also often focus on optimizing only a single objective, e.g., maximizing accuracy, without considering additional objectives such as memory consumption or computational complexity of a model, which are key to making deployment at the edge feasible. In this paper we propose a novel NAS strategy for TinyML based on multi-objective Bayesian optimization (MOBOpt) and an ensemble of competing parametric policies trained using Augmented Random Search (ARS) Reinforcement Learning (RL) agents. Our methodology aims at efficiently finding tradeoffs between a DNN's predictive accuracy, memory consumption on a given target system, and computational complexity. Our experiments show that we outperform existing MOBOpt approaches consistently on different data sets and architectures such as ResNet-18 and MobileNetV3.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces a new NAS strategy based on multi-objective Bayesian optimization and an ensemble of competing parametric ARS RL agents. Experiments are conducted on two TinyML use cases, including image classification and time-series classification.

### Strengths
1. Multi-objective optimization is critical for improving the performance/efficiency of deep learning on microcontrollers. 
2. The studied problem has many practical applications.

### Weaknesses
1. The technical contribution of the proposed method is a bit limited. MOBOpt and ARS RL are existing techniques. Combining them and applying them to TinyML is a bit straightforward. 

2. This paper lacks experiments on more challenging tasks, such as ImageNet classification on MCU [1] and Tiny Object Detection [2]. 

3. In addition, this paper also lacks direct comparisons with stronger baselines [1,2].

[1] Lin, Ji, et al. "Mcunet: Tiny deep learning on iot devices." Advances in Neural Information Processing Systems 33 (2020): 11711-11722.

[2] Lin, Ji, et al. "Memory-efficient patch-based inference for tiny deep learning." Advances in Neural Information Processing Systems 34 (2021): 2346-2358.

### Questions
See comments above

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
To optimize DNN for deployment on microcontrollers, the paper proposes a NAS approach that combines Augmented Random Search with Reinforcement learning for Bayesian Optimization. It employs an ensemble of competing polices to identify optimal DNN architectures and also strike a balance among accuracy, memory and computational complexity. Experimental results demonstrate the proposed approach outperforms traditional multi-objective Bayesian optimization methods across various datasets and architectures.

### Strengths
1. The idea of combining ARS and RL for Bayesian Optimization is novel and interesting.
2. The paper is well-written and clearly presented.

### Weaknesses
1. The superiority of the proposed method is not convincing based on the existing experiments. There lacks comparison with NAS methods.
2. It is hard to duplicate the results based on the details provided in the paper.

### Questions
1. As a NAS strategy for microcontrollers, why is the proposed method only compared to Bayesian Optimization methods instead of related NAS methods (especially those for microcontrollers)? Also, more recent methods should be compared with instead of those quite traditional ones.
2. Why choose only ParEGO (proposed in 2006) for comparison in Table 1? Also, it seems that the superiority of the proposed method is not quite impressive compared to ParEGO.

### Soundness
2 fair

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper uses reinforcement learning to perform multi-objective hyper parameter optimization for pruning a given architecture. The hyperparameters identified by the search algorithm are then used by the pruning pipeline to prune and then evaluate the model on the various objectives specified.
 The Augmented Random Search (ARS) agents are trained to identify the next most promising candidates. They circumvent the computationally expensive step of training the candidates to obtain the reward model by employing a gaussian process as a surrogate model to predict the reward. The GP prior is initialized by evaluating candidates sampled using Latin-Hypercube sampling. During every iteration of the search, multiple ARS agents are trained and the GP's posterior is updated by evaluating the best candidates proposed by all the agents. The various objectives used are accuracy, RAM and ROM memory consumption and FLOPS.
  
  They performed pruning on MobileNetV3 trained on DaLiac , ResNet trained on Cifar10. They also evaluated their algorithm on timeseries classification where the architecture used is CNN. In addition to this, on a synthetic dataset, they also showed that their algorithm is able to find a good minima when compared to ParEgo even during poor initialization where all the samples are in a region with no gradients.

### Strengths
1. It outperforms all the ParEgo multi-objective bayesian optimization baseline.
2. ARS agent in conjunction with GP as a surrogate avoids expensive evaluations to obtain the reward model. It would be good to highlight this and report the time taken by all the algorithms.

### Weaknesses
1. In the past, other multi-objective algorithms have used Knowledge distillation (KD) rather than pruning to obtain smaller well performing architectures. Please adapt yours to perform KD and compare against the following baselines [5], [6], [7]. It is also essential to indicate the drop in accuracy of the searched architecture when compared to the original architecture. Given that these algorithms also employ reinforcement learning and Bayesian optimization, please highlight how your algorithm is different from theirs and also demonstrate that yours performs better than these baselines.

2. Please cite other multi-objective NAS algorithms such as  MnasNet [1], once-for-all [2], NSGA-NET [3], Dppnet [4] in the related work. 
3. Given that your algorithm is performing HPO, please include other HPO solvers that use multi-objective algorithms such as NSGA-II, Multi-Objective Particle Swarm Optimization etc.

[1] Once-for-All: Train One Network and Specialize it for Efficient Deployment, Cai et al.
[2] MnasNet: Platform-Aware Neural Architecture Search for Mobile, Tan et al.
[3] NSGA-Net: Neural Architecture Search using Multi-Objective Genetic Algorithm, Lu et al.
[4] DPP-Net: Device-aware Progressive Search for Pareto-optimal Neural Architectures, Dong et al.
[5] Learnable embedding space for efficient neural architecture compression., Can et al.
[6] AMC: automl for model compression and acceleration on mobile devices., He et al.
[7] N2N learning: Network to network compression via policy gradient reinforcement learning., Ashok et al.

### Questions
1. How can a weighted sum reward yield various Pareto fronts? Given that the final objective is a weighted sum, one must vary the weights to obtain various fronts. This is not a Pareto-optimal solution. Algorithms such as NSGA-II yield pareto-optimal solution. 
2. How can k-means ensure the diversity of the solutions in Pareto-front? K-means selects solutions from each centroid which would imply that it is the most representative sample in that cluster. To ensure diversity, we want to sample solutions from region where the number of solutions in the neighborhood is sparse.  NSGA-II uses crowding distance, Lemonade uses kernel density estimator and in both cases they select solutions from less crowded region.

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
This study merges the efficiency of Gaussian Processes (GPs) with the effectiveness of Reinforcement Learning (RL) agents to facilitate the efficient exploration of Deep Neural Network (DNN) architectures that can be directly implemented on microcontrollers. Unlike previous research on RL for Multi-Objective Optimization (MOOpt), it doesn't apply RL directly to the MOOpt problem's search space. Instead, it employs RL to enhance the optimization process conducted on GPs. To tackle the issue of resource-intensive RL agent training, it suggests training a group of competing policies within MOBOpt. These policies utilize a straightforward Augmented Random Search (ARS) method for candidate sampling.

### Strengths
1. The research introduces an innovative approach by uniting the sampling efficiency of Gaussian Processes (GPs) with the expressive capabilities of Reinforcement Learning (RL) agents. Additionally, the incorporation of Augmented Random Search represents a novel enhancement.

2. The study demonstrates performance improvements when compared to alternative Bayesian optimization strategies.

### Weaknesses
1. From a technical standpoint, I haven't identified a direct link between the proposed method and TinyML, and this study hasn't delved deeply into the realm of TinyML models, and some important related works [1] [2]  are missing. 

2. The comparative methods in this study are restricted to Bayesian optimization strategies, omitting comparisons with other Neural Architecture Search (NAS) approaches such as zero-shot methods, optimization-based methods, or predictor-based methods.

3. The mentioned validated architectures are limited to ResNet18 and MobileNetV3, which are somewhat dated in the context of contemporary developments.

[1] MCUNet: Tiny Deep Learning on IoT Devices
[2] MCUNetV2: Memory-Efficient Patch-based Inference for Tiny Deep Learning

### Questions
1. The main performance comparison didn't conclude for some architectural performance but mainly focused on the comparison with other optimization methods. How about the performance comparison with MCUNetV1[1], and MCUNetV2 [2]?
2. What is the search cost and efficiency compared with other types of NAS methods? For example, it would be fair to compare with other zero-shot NAS methods or preditor-based methods？

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 5

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work proposes a NAS strategy specifically for TinyML using multi-objective Bayesian optimization (MOBOpt) and an ensemble of competing parametric policies trained using Augmented Random Search (ARS) Reinforcement Learning (RL) agents. This approach can be utilized to explore the design tradeoffs between a DNN’s predictive accuracy, memory consumption on a given target system, and computational complexity. The experiments show competitive results compared to prior optimization strategies. The paper is well organized in general, but my major concern is that the proposed optimization approaches such as MOBOpt are generic and not specific to TinyML. Particularly, the authors argue that it is expensive to evaluate the accuracy and memory consumption for TinyML system, but accuracy evaluation is required for NAS despite the targeted computing platform. Although memory (RAM and ROM) consumption is a specific requirement for TinyML system, the evaluation is similar or even less expensive than accuracy evaluation. Hence, more discussion about why tinyML makes NAS challenging and necessary is expected.

### Strengths
The proposed NAS approach produces competitive models for MCU platform. The paper is well organized.

### Weaknesses
The proposed NAS optimization seems to be generic and not specific for TinyML platform, which makes the research problem not quite convincing.

### Questions
Could you illustrate why tinyML makes NAS challenging and unqiue?
How the proposed NAS optimization applies specifically for tinyML rather than general network design?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
