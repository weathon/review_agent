# Speed Up Federated Learning in Heterogeneous Environment: A Dynamic Tiering Approach

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 5, 3

## Abstract
Federated learning (FL) enables collaboratively training a model while keeping the training data decentralized and private. However, one significant impediment to training a model using FL, especially large models, is the resource constraints of devices with heterogeneous computation and communication capacities as well as varying task sizes. 
Such heterogeneity would render significant variations in the training time of clients, resulting in a longer overall training time as well as a waste of resources in faster clients. To tackle these heterogeneity issues, we propose the Dynamic Tiering-based Federated Learning (DTFL) system where slower clients dynamically offload part of the model to the server to alleviate resource constrains and speed up training. 
By leveraging the concept of Split Learning, DTFL offloads different portions of the global model to clients in different tiers and enables each client to update the models in parallel via local-loss-based training. This helps reduce the computation and communication demand on resource-constrained devices and thus mitigates the straggler problem. 
DTFL introduces a dynamic tier scheduler that uses tier profiling to estimate the expected training time of each client, based on their historical training time, communication speed, and dataset size. The dynamic tier scheduler assigns clients to suitable tiers to minimize the overall training time in each round.
We first theoretically prove the convergence properties of DTFL. We then train large models (ResNet-56 and ResNet-110) on popular image datasets (CIFAR-10, CIFAR-100, CINIC-10, and HAM10000) under both IID and non-IID systems. Extensive experimental results show that compared with state-of-the-art FL methods, DTFL can significantly reduce the training time while maintaining model accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper proposes Dynamic Tiering-based Federated Learning (DTFL), which aims to speed up FL via dynamically allocating training load to heterogeneous resource FL in different tiers. DTFL introduced a dynamic tier scheduler to cluster FL local clients into tiers and then leverage split learning to split different portions of the global model and deploy on local clients based on their tier. Additionally, the paper provides theoretical proof of the convergence properties of DTFL.

### Strengths
1. DTFL leverages local-loss-based training and split learning, which enables dynamic offloading training workload for local clients with different resource capacities.

2. DTFL introduced dynamic tier scheduling components to adaptively cluster local clients to different resource tiers and hence speed up training. 

3. The paper provides theoretical convergence analysis.

### Weaknesses
**1.** The experiment looks weak and can not support the arguments proposed in the paper.

**2.** The tier scheduling metrics are unreliable. DTFL uses training time, communication time, and training time of the server-side model to profile the tier. However, using `actually time`  is unreliable, in computational devices (especially edge devices), many factors can affect the execution time for the same program, such as temperature, IO thread, etc. More standard metrics might be considered.

Additionally, in the experiments, the evaluation metric for training speed is unfair. The authors use total training **time in second** to evaluate the training speed of federated learning. However, simply tracking the training time is hard to avoid hardware and network traffic effects. Instead, more standard evaluation metrics should be used, such as **FLOPs, MACs, GPU Hours, electricity usage, total #trainable parameter**s, etc.

**3.** The experiments is simulated on CPU and GPUs raising further concern on point 2 above.

**4.** Delay on dynamic Tier Scheduling. DTFL uses total training time in previous communication rounds to tiering clients, it may not accurately reflect computational status in the current round.

**5.** No experiments reflect the communication cost of the proposed method.

### Questions
Please kindly address the concerns I list in the weakness section.

Overall, the experiments are incomprehensive and lack the evidence to support the argument of the paper. I'll change my mind if the authors add further fairness evaluation results.
For instance, use more fairness metrics to evaluate the speed, consider more system heterogeneity settings with more diverse resource profiles, etc.

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
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper presents a new tiered-based split federated learning to handle heterogeneous environments where the resources of clients change over time. Specifically, the authors propose dynamic tier scheduling which operates through tier profiling and tier scheduling. Tier profiling tracks the training times of clients, which change over time, and based on this, the server estimates the current training times for each client in all tiers using EMA. Tier scheduling assigns clients to a tier according to their estimated time for efficient training. The authors provide theoretical convergence bound for both convex and non-convex loss functions. The empirical results show that the proposed method has fast convergence speeds over various baselines.

### Strengths
- The authors propose a new tiered-based split learning which can efficiently train large models depending on clients’ resources in heterogeneous environments.
- The convergence behavior of the proposed method is theoretically established
- The proposed method significantly reduces the training time compared to existing works.

### Weaknesses
- It seems that this work only considers a scenario where computational and communication resources of all clients change over time. However, it’s not clear that this is a reasonable scenario in practice. I think there might be more cases where the resources of only a portion of all clients change in real-world scenarios. To demonstrate the effectiveness of the proposed algorithm in various practical scenarios, it would be beneficial for the authors to conduct additional experiments where they vary the proportion of the devices whose training times change over time.
- What are some practical applications in which the resources of each client changes over time? It would be helpful to describe some examples of such applications to emphasize the importance of addressing the targeted problem. 
- Ablation studies should be performed to confirm the effect of each component. First, a comparison with local-loss based SFL [1] (not tiered-based) should be considered. Secondly, using local-loss based SFL, a comparison between static tiered methods and the proposed dynamic tiered method seems necessary. Finally, to see the effect of EMA in tier profiling, the author should compare the results of tier profiling with and without EMA. 
- Overall, the technical novelty of the proposed method seems limited. I feel that tier profiling and scheduling are straightforward approaches based on the previous works. 

[1] Han et al., "Accelerating federated learning with split learning on locally generated losses." In ICML 2021 Workshop on Federated Learning for User Privacy and Data Confidentiality. ICML Board, 2021.

### Questions
See Weaknesses and,

- The main results only provide the training time required to achieve the target accuracy. What is the maximum accuracy that each method can achieve? 
- There is a type: cross-solo -> cross-silo

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper presents an approach to address the challenges posed by heterogeneity in Federated Learning (FL) systems. The proposed Dynamic Tiering-based Federated Learning (DTFL) system leverages the concept of Split Learning to dynamically offload portions of the global model to different tiers of clients, thereby mitigating the straggler problem and reducing the computation and communication demand on resource-constrained devices.

### Strengths
1. The paper is well-written and easy to follow.

2. The extensive experiments validate the proposed method.

### Weaknesses
1. The primary question I have regarding this paper pertains to its motivation. While there is a wealth of prior work on heterogeneous computation and communication capacities in the Federated Learning (FL) setting—such as clients training heterogeneous models, clients performing partial training based on their individual abilities, asynchronous updates, and lightweight training with pre-trained models—the proposed method introduces a requirement for the server to update its model with labels, which raises privacy concerns. Therefore, it is crucial to understand the advantages of the proposed method.

2. It is good that the authors have included a convergence analysis. However, the convergence rate presented in Theorem 1 appears to be suboptimal compared to classical FL settings that have been studied previously.

3. The authors are encouraged to provide more results in non-IID settings, similar to the approach demonstrated in Figure 2. Additionally, it would be beneficial if Figure 2 could display the entire coverage process, as it is currently truncated.

### Questions
Please see the weakness section above.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
