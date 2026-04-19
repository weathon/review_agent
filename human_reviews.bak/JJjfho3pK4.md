# Everyone Counts: Fair and Accurate Heterogeneous Federated Learning with Resource-Adaptive Model Modulation

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 5, 5

## Abstract
In the practical implementation of federated learning (FL), a major challenge arises from the presence of diverse and heterogeneous edge devices in real-world scenarios, each equipped with varying computational resources. The conventional FL approaches, operating under the assumption of uniform model capacity, face a dilemma. They can opt for a large global model, but this may not be feasible on resource-constrained devices, resulting in issues of fairness and training biases. Conversely, they can choose a small global model, but this compromises its ability to represent complex patterns due to limited capacity. In this paper, we present a novel approach called Dynamic Federated Learning (DynamicFL). It employs structural re-parameterization to achieve adaptable local model modulation and seamless knowledge transfer across a diverse set of heterogeneous models. DynamicFL ensures equitable treatment of all clients, empowering them to actively participate in the learning process with their full computational potential, thereby fostering sustainability within the FL ecosystem. Extensive experimental results validate that DynamicFL surpasses state-of-the-art techniques, including knowledge distillation and network pruning-based methods, in achieving significantly higher test accuracy in the context of heterogeneous FL.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work considers an approach in which resource-constrained devices participating in FL training can achieve seamless knowledge transfer between diverse heterogeneous models. The proposed Dynamic FL method does structural re-parameterization of the local model per device compute abilities. Experimental results show performance improvement as compared with standard baselines, such as HeteroFL and KD-based methods.

### Strengths
This work employs structural re-parameterization to enable adaptive modulation of local models, which is better for a practical FL setting where the collaborating nodes have heterogeneous computing resources. It has been shown this approach enables fair participation of clients with improved model accuracy (as compared to known methods to address heterogeneous FL, like KD and pruning). The authors have theoretically provided a convergence guarantee for their method.

### Weaknesses
I have concerns regarding the technical aspects of this work, as follows:

1. Several aspects are not clearly defined and discussed in the paper. For instance, after (2), what is r_i, and how is it defined?
2. As the structural re-parameterization is done each round (in my understanding), it certainly adds computational burden during training on the already constrained nodes. Then, the question arises: what is the cost of executing (4)?
3. The manuscript is not clear in justifying how, through (5), to assess the contribution of branches to the global aggregation. I feel the authors missed (in 2.2.2) building a clear connection between re-parameterization and local training operation during model training. For instance, how "contribution" is quantified per se, and is this only with the adaptation in the number of layers, and also the number of parameters per layer, and so on.  
4. Is the number of epoch E fixed for each client? If so, why? because one may argue clients can be adaptive in choosing the number of epochs during local training.
5. It is unclear why the aggregation scheme, as in (11), was used. Also, can the authors please justify further their claim: "operations in re-parameterization guarantee lossless knowledge transfer across heterogeneous local models"?
6. While the experiments cover a broad range of (relevant) baselines (which is commendable), however, I am not sure the comparison is sufficient to demonstrate improvement. For instance, the authors have not mentioned why the number of local epochs was set (fixed) to 3 (for all clients); the model capacity is only abstracted without particular practical details - how do make these changes in the computing abilities; no convergence analysis (experimental) is provided, which can be added in my understanding; training time-complexity analysis is missing (also, what is the training efficiency), and so on. In fairness analysis, how is the accuracy reported?

Minor:
- Figure 1 is unclear in its current form, and the description is insufficient. How to interpret this? The representative Diverse Branch Block (DBB), as in [Ding et al., 2021a,b], is unclear to the readers unfamiliar for readers.
- notations are a bit difficult to follow.

### Questions
Please kindly see the comments in the weakness section.

### Soundness
3 good

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
The authors introduce what they call DynamicFL: the idea is to make federated learning practical for heterogeneous set of devices by dynamically adjusting the model architecture (re-parameterization) during local training. Each client will then train a slightly different architecture of the model, depending on the available resources. Afterwards, the model will be scaled back and shared for aggregation.

### Strengths
- Heterogeneous FL is important, especially when we need to train larger models on a wide range of end-clients 
- The paper is well presented and written
- The idea of structural re-parameterization is interesting contribution

### Weaknesses
The paper is interesting to read, but there are some areas to improve:

- The fact that the model is always scaled up (on the clients) and then scaled down before aggregating is a bit counter intuitive. One would expect that the aggregated (server-side) model would need additional capacity to learn from the wide distribution of data, rather than the device submodes. This might restrict the overall capacity of the final aggregated model. Maybe the authors can further motivate this approach. 

- Similarly, this paper focuses on convolutions, limiting a bit the applicability on other architectures 

- The authors tackle a heterogeneous environment with device of different capabilities. In most such use-cases devices typically are sampled and are likely to participate for just a few rounds (in some use-cases just once) during the FL training. I was wondering what would be the impact of having clients participating only for 1 or K (where K is small) rounds. 

- The evaluation does not provide any insights on energy, memory footprint, transfer data volume, convergence speed. This is important as this method is focusing towards performance on a heterogeneous set of devices.

### Questions
- The fact that we scale up the model on-device, does this mean that the largest possible model we can support for inference is determined by the weakest devices ? 

- In the experiments, the authors show that other methods (HeteroFL , Split Mix etc) have worse accuracy. But a benefit of these approaches is that they can scale *down* the model (I.e., submode training). Whereas dynamicFL practically scales up. I was wondering if the authors used a larger initial (aggregated) architecture for these models when compared to their method. Essentially, a fair comparison would be having the other methods provision for the large devices at the aggregator level (I.e., having a larger initial model).

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper presents Dynamic Federated Learning (DynamicFL), addressing the challenge of heterogeneous edge devices in federated learning. DynamicFL employs a structural re-parameterization to adapt local models, ensuring a fair client participation while achieving a high test accuracy. It outperforms existing methods, including knowledge distillation and network pruning, in heterogeneous FL scenarios.

### Strengths
1. The paper is easy to follow.
2. The authors conducted a series of comparative experiments involving several distinct methods.

### Weaknesses
1. The major issue of this paper is about the technical contribution. The techniques employed in this paper, specifically the 'REF' and 'DYMM,' appear to be exsiting ones proposed in prior works, which could potentially diminish the originality and contribution of this article.
2. The proposed theoretical framework focuses primarily on the analysis of convergence rates, yet the experimental part lacks the presentation of any convergence curves.

### Questions
Can the authors provide an analysis that quantifies the additional computational and storage resources consumed by the introduced operations 'REF' and 'DYMM'?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
