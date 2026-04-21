# DiffusionNAG: Predictor-guided Neural Architecture Generation with Diffusion Models

- Avg Score: 5.75
- Decision: Accept (poster)
- Scores: 6, 6, 6, 5

## Abstract
Existing NAS methods suffer from either an excessive amount of time for repetitive sampling and training of many task-irrelevant architectures. To tackle such limitations of existing NAS methods, we propose a paradigm shift from NAS to a novel conditional Neural Architecture Generation (NAG) framework based on diffusion models, dubbed DiffusionNAG. Specifically, we consider the neural architectures as directed graphs and propose a graph diffusion model for generating them. Moreover, with the guidance of parameterized predictors, DiffusionNAG can flexibly generate task-optimal architectures with the desired properties for diverse tasks, by sampling from a region that is more likely to satisfy the properties. This conditional NAG scheme is significantly more efficient than previous NAS schemes which sample the architectures and filter them using the property predictors. We validate the effectiveness of DiffusionNAG through extensive experiments in two predictor-based NAS scenarios: Transferable NAS and Bayesian Optimization (BO)-based NAS. DiffusionNAG achieves superior performance with speedups of up to 35$\times$ when compared to the baselines on Transferable NAS benchmarks. Furthermore, when integrated into a BO-based algorithm, DiffusionNAG outperforms existing BO-based NAS approaches, particularly in the large MobileNetV3 search space on the ImageNet 1K dataset. Code is available at https://github.com/CownowAn/DiffusionNAG.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes a method to perform neural architecture search using diffusion models. More precisely, they model the neural architecture as a stochastic process and model it using diffusion. Then at inference with appropriate initialization the reverse process samples a new neural architecture to train a model.

### Strengths
1. The use of diffusion models to neural architecture search is in an interesting idea. 
2. Modeling NAS as a generative problem enables creating multiple architectures starting with random noise, compared to exhaustively searching the entire space. This enables reducing the NAS cost. 
3. Empirically the method outperforms the baselines considered in the paper.

### Weaknesses
1. Even though the motivation to the use of diffusion models to neural architecture search is interesting, it seems more appropriate to use discrete diffusion models for discrete random variables instead of using a continuous variables. For instance [1] is an interesting approach which seems to be more correlated to the desired problem instead of continuous diffusion. 
2. The use of diffusion models for NAS seems to be very incremental, as the proposed approach is a direct application of the standard available diffusion based approaches. What are some of the major hurdles faced by the authors in making this work for NAS?
3. For conditional NAG, the authors can consider convert A_t to A_0 using the appropriate equations depending on the choice of diffusion sampler used and then apply the pre-trained predictor. This can truly enable using a pre-trained predictor which is just a function of A_0 instead of A_t.

[1]Vector Quantized Diffusion Model for Text-to-Image Synthesis(https://arxiv.org/abs/2111.14822)

### Questions
Can such an approach be used to find optimal architectures for diffusion models? Instead of the standard classification problems. 
One can start with discrete NAR transformers for instance (like MaskGIT) which can model the NAS process and token predictions, and then the produced model will be used for generation which is another variant of cross entropy minimization problem. Since the foundation models are increasing in size day by day it will be interesting to see if their training methods can be used to find optimal sparse architectures.

[1]MaskGIT: Masked Generative Image Transformer (https://arxiv.org/abs/2202.04200)

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes DiffusionNAG, a novel conditional Neural Architecture Generation framework based on diffusion models, which considers the neural architectures as directed graphs. The proposed method is shown to be significantly more efficient than previous NAS schemes. Empirically, the authors demonstrate it can provide speedups up to 20x, while being superior in performance.

I would also like to note that I am no expert in Neural Architecture Search, which could affect my judgments below.

### Strengths
* The proposed method, which uses diffusion models to essentially generate the entire search space and then utilizes the gradient of property predictor to guide the generation towards the more potent candidate, is novel to my knowledge.
* Diffusion model sampling is essentially treated as a search algorithm, which enables more aggressive searches that result in faster procedures.
* The proposed framework could utilize different predictors dependent on the tasks, so the base diffusion model only needs to be trained once.
* The paper also explores diffusion models suitable for generating directed acyclic graphs, which could be useful for other fields as well.
* Empirically, the DiffusionNAG is shown to be scalable and effective, providing speedups while improving the performances.
* The paper is well-written and easy to follow for the most parts.

### Weaknesses
* I think some parts could be explained a bit more, even if they are from prior works, e.g. the meta-learned dataset-aware predictor in Sec.2.3. This, along with the diffusion background, could be covered in a dedicated background section.

### Questions
* I would not say that I have the best time understanding the parts about "Score Network for Neural Architectures": what exactly is being pursued here? if our adjacency matrix is defined to be upper triangular, should it not already be a DAG? What is the goal of the proposed positional embedding? A bit more explanation is welcomed here.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors propose a diffusion based model for neural architecture generation. By using the guidance from the predictor, DiffusionNAG can generate task-specific architectures. They verify the effectiveness of DiffusionNAG on Transferable NAS and Bayesian Optimization-based NAS.

### Strengths
This paper is well-motivated. Previous work only uses the diffusion model to model undirected graphs for neural architecture generation. Using the diffusion model for modeling the directed graph is needed. Moreover, classifier guidance is well suited for the Transferable NAS and substitute step 3) in BO-based NAS approaches, where the dataset and the pre-trained classifier are given as the setup.

### Weaknesses
1."Whether diffusion model for neural architecture generation is more efficient than Mutation + Random" is a good question. It is important to ask "when they are better".

* The strength of the predictor guidance is an important hyperparameter. In this paper, a good performance can be achieved by cross-validation. Could the author provide some ablation studies of how this hyperparameter affects the performance? How could we effectively define the search range for cross-validation?

* Will using the diffusion model for neural architecture generation have some model collapse phenomenon? Does the Diffusion model only generate some similar architecture, and achieve the better "worse case accuracy" by sacrificing the diversity? If that is the case, any explore and exploit framework for it?

### Questions
Please see the Weaknesses part.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The researchers are introducing a new approach called DiffusionNAG to overcome limitations in Neural Architecture Search (NAS) methods. Instead of traditional NAS, they propose a conditional Neural Architecture Generation (NAG) framework based on diffusion models. This framework, utilizing graph diffusion models, allows for efficient generation of task-optimal neural architectures with desired properties. DiffusionNAG incorporates parameterized predictors to guide the generation process, making it more efficient than previous NAS methods. The effectiveness of DiffusionNAG is demonstrated through experiments in Transferable NAS and Bayesian Optimization-based NAS scenarios.

### Strengths
1. This work treats neural architectures as directed graphs and introducing a graph diffusion model. It enables flexible generation of task-optimal architectures guided by parameterized predictors, surpassing the efficiency of traditional NAS methods.

2. DiffusionNAG outperforms baselines in Transferable NAS, achieving up to 20x speedups. Additionally, when integrated into a BO-based algorithm, it outperforms existing approaches, especially in the extensive MobileNetV3 search space on the ImageNet 1K dataset.

### Weaknesses
1. This work only reports results on two NAS benchmark search spaces. It is unclear whether it can achieve similar performance on much larger search spaces, such as the ones proposed in [1] and [2], which are widely used in previous works.

2. Fig. 3 and 4 compare results on existing AO strategies and various acquisition functions. However, there is no comparison to other methods on ImageNet, as shown in Tab. 1 and 2.

3. In Tab. 5, the authors include robust accuracy against the APGD attack, which is good. However, the corruption evaluation is insufficient, considering only glass blur. It would be better to include more comprehensive corruptions, such as those in ImageNet-C or CIFAR-10-C.

[1] Learning Transferable Architectures for Scalable Image Recognition, CVPR 2018.

[2] DARTS: Differentiable Architecture Search, ICLR 2019.

### Questions
Please address questions in "Weaknesses".

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
