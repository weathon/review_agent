# Task-Agnostic Federated Continual Learning via Replay-Free Gradient Projection

- Decision: Reject
- Scores: 2, 4, 2, 4

## Abstract
Federated continual learning (FCL) enables distributed client devices to learn from streaming data across diverse and evolving tasks. A major challenge to continual learning, catastrophic forgetting, is exacerbated in decentralized settings by the data heterogeneity, constrained communication and privacy concerns. We propose Federated gradient Projection-based Continual Learning with Task Identity Prediction (FedProTIP), a novel FCL framework that mitigates forgetting by projecting client updates onto the orthogonal complement of the subspace spanned by previously learned representations of the global model. This projection reduces interference with earlier tasks and preserves performance across the task sequence. To further address the challenge of task-agnostic inference, we incorporate a lightweight mechanism that leverages core bases from prior tasks to predict task identity and dynamically adjust the global model's outputs. Extensive experiments across standard FCL benchmarks demonstrate that FedProTIP significantly outperforms state-of-the-art methods in average accuracy, particularly in settings where task identities are a priori unknown.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper provides a orthogonal gradient projection for federated continual learning to mitigates the catastrophic forgetting.

### Strengths
The paper conducts extensive experiments on various challenging datasets, e.g., DomainNet, ImageNet-1K.
The paper's contribution is reasonable and show some potentials. However, there are some parts in the paper are not well justified, thus, limiting the paper from showing their novelty. 
I'm willing to listen to the authors' responses and give update to the reviews.

### Weaknesses
1. The introduction and abstract should have more significant improvement to emphasize the paper's novelty and contributions. For example, the authors claimed that they propose FedProTIP to mitigate the catastrophic forgetting. However, it is not clear what is the problem that FedProTIP can solve while others can not. 
2. The authors claimed that the paper is a replay-free methods. However, it is shown in the paper that the work is replay-based. Furthermore, from my limited understandings, the paper requires a lot of memory usage to store the exemplars. The authors please refer to the questions below to discuss about the works. 
3. Some statements seem to be not correct in terms of technical, which make the paper not too reliable (e.g., saying GPM is a replay-free CL, actually GPM store orthogonal gradient bases to support the training).  
4. Some technical issues needed to be discussed and well-explained to make the contributions more clear.

### Questions
1. It is not clear why FOT incurs significant communication overhead and raises potential privacy concerns? 
2. The $\Phi$ needs to be saved for all tasks. Thus, it is questioned how are the size of the $\Phi$? 
3. The authors claimed that they stored the last layer activations only. However, as I checked the code, the authors used the full function of update_GPM. Furthermore, no smaller subset are used in the code as shown in L242. The authors please discuss about this?
3. The function (4) requires all of the $Phi^t$ are orthonormal for all task to work well. However, at the second task, there are first task only? How do the authors guarantee the $Phi^0$ to be orthonormal? 
4. Furthermore, the $\Phi^{[0:t-1]}$ needs all of the bases to be orthonormal. It is reasonable that the entire gradients are orthonormal as used in GPM. However, by using the last activation layer, can we guarantee the orthonormal characteristics? 
5. Why do we only use local core bases only? we lack the V and $\Sigma$? Can we preserve the orthonormal of the $\Phi$? 
6. In L242, this is resevoir sampling. However, this one seems not proved the significant performance. Furthermore, the data transmission seems not to be justified. This one needed to be discussed carefully in the experimental evaluations. 
7. How (10) can show the task alignment? This part need justification in the paper. 
8. The authors claimed that FedProTIP achieves the fastest training. However, due to the questions above, it is unclear how FedProTIP can achieve the fastest computation time. The authors please discuss about this. 
9. The core bases in memory usage in 5.4 should be explained why we can achieve this (refers to the previous questions). 
10. The GPU usage seems not properly shown the memory efficiency of the replay memory. It is how samples are stored in the GPU for the training. The authors please consider about this carefully.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes an approach for FCL to mitigate the catastrophic forgetting problem. The paper extends the ideas introduced in GPM to federated setting where gradients are projected onto the orthogonal complement of the subspace spanned by prior task activations. 

    1. Replay based:  Store old examples, risking privacy or exceeding storage limits
    2. Generation based: require server-side generative models, slowing aggregation
    3. Regularization based: more computational overhead

### Strengths
The paper is a straightforward extension of the gradient projection memory technique that was proposed for centralized continual learning.

The paper is clearly written. The gaps in existing FCL approaches were clearly identified, motivation is justified as to why a federated way to do GPM makes sense. Solution developed is explained well, claims not overstated. Experimental results are comprehensive and covers the usual FCL setting very well.

### Weaknesses
- The calculation of the global core base is done sequentially. Since projection is not commutative, this can favor certain clients over the others. Especially, since client 1 is always chosen first, it can be seriously advantaged over other clients.
        - My second concern is that suppressing gradient updates along core bases would likely slow down convergence when data heterogeneity is low. Specifically, new updates may not contribute much if their data distributions are similar. Therefore, will the solution be applied to only when the data distribution is expected to change across clients?
        - The paper argues that sharing the raw activations leaks privacy (absolutely true). However, there is non negligible privacy leakage from sharing the orthonormal bases from each client. In the least, it helps in identifying which client has shared and not shared bases and may even help in inferring data statistics.  Refer to -> “Subspace Leakage in Federated Representation Learning” (ICML 2022 Workshop)”, “Federated PCA with Privacy” (NeurIPS 2020).
        - Some aspects such as each client experiencing same sequence of tasks, while used in other prior art, is somewhat unrealistic
        - The idea in the paper is sound when compared to other federated continual learning, especially in not causing a significant additional overhead that exist in other baselines. However, the novelty seems to be incremental when compared to the GPM technique. The key contribution seems to be federating the computation of the core bases and merging them. Merging sequentially has the above 2 challenges, which are not addressed in the convergence analysis. Privacy issues might still remain (albeit reduced significantly).

### Questions
Can you clarify how the sequence of calculating the global core bases affect fairness (favoring some tasks over others purely based on the update sequence)?

Clarify whether the gradient direction suppression only happens when there is task heterogeneity.

Privacy impact (see comments above). The paper argues that existing work leaks privacy (e.g., sharing raw activations), but ends up sharing the orthonormal bases from each client. So please comment on the non-negligible privacy leakage for the proposed approach.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
FedProTIP is a federated continual learning framework that mitigates catastrophic forgetting through subspace-based gradient projection and task identity prediction. It extends gradient projection methods like GPM to the federated setting without requiring task IDs, replay buffers, or generative models. The approach is validated on CIFAR100, ImageNet-R, and DomainNet, showing gains over existing FCL baselines.

### Strengths
* Introduces a subspace projection mechanism that operates locally on clients, avoiding privacy leaks and communication overheads associated with transmitting activations (like in FOT)
* Task-agnostic inference: a task identity prediction component that uses subspace relevance vectors.
* Avoids replay, generative modeling, and task labels, which are common barriers in existing CL techniques.

### Weaknesses
The comments are provided in the decreasing order of importance:
1. Novelty: The novelty in this work is very low. Almost the entire technique resembles FOT, except for two steps towards the end. FOT transmits activations to the server, while here, they only transmit the subspaces. Thus, theres a fundamental tradeoff with complexity. There is also task-agnosity during the inference time, which is novel compared to FOT. 
2. The paper claims task-agnostic federated continual learning, but this claim holds only at the inference stage, not during training. During training, task boundaries and task identifiers are known. Each new task is introduced sequentially, and the model (both FedProTIP and baselines) is informed that it is now training on a particular task ID. This means the system is task-aware during optimization, subspace extraction, and gradient projection. True task-agnostic continual learning would require the model to detect task boundaries automatically or operate under mixed unlabeled task streams, which is not the case here (e.g. BGD is a task agnostic setting). Most of the baselines (GPM, FOT, LANDER) were originally designed for task-incremental settings, where both training and inference use task IDs to select the appropriate classifier head. In this paper, those methods appear to have been trained (required more clarity regarding this) in their standard task-incremental mode (each task having its own head, with task ID known during training), but then evaluated in a task-agnostic manner at test time, i.e., without providing the task ID and by concatenating all task heads into a single unified classifier. This introduces a training-testing mismatch for the baselines: During training, they optimized separate heads per task, learning disjoint class boundaries. During testing, those separate heads are simply merged, without retraining tasks. It forces task-incremental models, which were never trained for shared label spaces, to operate in a task-agnostic evaluation regime for which they were not designed. In short, I think the evaluation setup disadvantages baselines, as they are tested outside their intended framework. I suggest the baselines be trained over a more consistent setup, or the proposed method be compared against truly task-agnostic algorithms. Some recent baselines with theoretical guarantees (CFLAG, AISTATS 2025) is not cited.
3. Another drawback is that there is no rigorous convergence theorem under FL conditions; no explicit assumptions are stated about client sampling, learning rates, or optimizer. Projection changes the geometry of local updates; it may reduce alignment across clients and remove useful gradient components, both affect global convergence. The method aims to bound forgetting, but there’s no formal theoretical bound on final loss or forgetting. Without formal bounds or at least derivations, it’s unclear when/why local projected updates converge.
4. Task identity inference vulnerable to client heterogeneity and small-task regimes: TIP uses client-stored reference vectors and majority votes at the server. But (a) under high non-IID (very different client data) or when tasks are highly similar (e.g., many small classes per task), reference vectors may not generalize; (b) majority voting can be biased if only a subset of clients are representative or if malicious clients exist. The experiments hint at weaker TIP performance in 20-split ImageNet-R (they report TIP less effective there); this shows TIP may break when tasks are small or under severe heterogeneity. The paper does not analyze failure modes or robustness of TIP to skewed client distributions or adversarial/malicious voters.
5. The task-identity inference mechanism assumes that reference vectors for different tasks are distinguishable. If two tasks yield highly similar subspace relevance patterns, clients may produce nearly identical reference vectors, leading to incorrect task routing. No mechanism or experiment tests this case.
6.  Repeatedly appending orthogonal bases across many tasks can exhaust the representational capacity as the orthogonal complement shrinks. The paper lacks experiments or mechanisms showing how capacity is preserved, and experiments on sensitivity to small local datasets, which can lead to bad SVD estimates.
7. Comments on experimental section:Partial participation experiments are missing. Uses pretrained ResNeT-18 backbones and 32x32 inputs for ImageNet-R which simplifies the problem - do the baselines also use pretrained networks?  For large models (e.g., ViTs), subspace dimensionality may explode; this scalability aspect is not explored. Ablation on task identity prediction accuracy will be useful to gain more insights into the strategy. (eg. confusion matrix or precision of the predicted task IDs.)
8. Communication & Computational Complexity Not Properly Quantified - The claim of “compact core bases” lacks concrete quantification. There is no detailed accounting of how communication cost scales with model depth L, task count T, or clients K. Each client transmits (a) model updates, (b) layer-wise subspace bases, and (c) reference vectors, which increases uplink rounds beyond standard FL. Additionally, randomized SVD per layer is computationally expensive on edge devices, yet no runtime or energy analysis is provided.
9. Extreme Client Heterogeneity Not Explored: Experiments include only moderate non-IID settings. The paper does not test FedProTIP under severe heterogeneity (e.g., highly skewed label or feature distributions) or when only a subset of clients participates per task. Since subspace alignment and voting rely on similarity across clients, extreme heterogeneity could significantly degrade performance.

### Questions
The authors must clarify the points provided in weaknesses. In addition, these are the questions:

-How sensitive is FedProTIP’s performance to the quality and dimensionality of the extracted subspaces (from randomized SVD)? If subspace estimation is inaccurate (e.g., due to small local datasets or noisy activations), could projections remove useful gradient components?
-As the number of tasks grows, the orthogonal complement shrinks, reducing the feasible gradient space. How does FedProTIP prevent capacity exhaustion or representation stagnation over many tasks? Does the model ever discard or merge old subspaces to recover capacity?
-Projecting gradients onto orthogonal subspaces can introduce optimization instability if the projection matrix is ill-conditioned or changes abruptly between rounds. Is there any smoothing or regularization applied to mitigate oscillations between projected and unprojected directions? Does the projection step conflict with local optimizer dynamics (e.g., Adam momentum or adaptive learning rates)?
-The method mentions layer-wise subspace projection. Does this apply equally to all layers? Lower layers might benefit from more stability, while upper layers might require more flexibility. Is there an adaptive layer weighting strategy?
-While FedProTIP reduces communication by sending compact subspace bases, how much compression is achieved in practice compared to FOT’s embedding uploads? Are there quantitative results showing communication savings?
-How does FedProTIP ensure cross-client alignment of subspaces learned from heterogeneous data distributions?
-Since the server constructs projection matrices, does this not reintroduce partial centralization of feature statistics, similar in spirit to FOT, though lower-dimensional? What happens if subspace information itself leaks sensitive feature structure?
-As more subspaces are accumulated, task identity prediction could become computationally expensive (multiple alignments per test sample). What is the complexity per inference? Is it feasible for edge devices?
-The paper claims strong task-agnostic performance,  but do the benchmarks include truly overlapping or ambiguous task boundaries (e.g., same-class drift or domain shift)? If not, the effectiveness in real-world task-agnostic FL might be overstated.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper focuses on FedProTIP, a federated continual learning framework for improving the performance on task-agnostic scenario. 

The basic idea is twofold: (1) use gradient projection method for mitigating forgetting, and (2) predict the task identity and adjst the global model's outputs accordingly, in order to handle the task-agnostic inference.

### Strengths
This paper focuses on an important and timely problem of task-agnostic FCL.

### Weaknesses
* The idea of gradient projection is nothing new. There are several related works using the gradient projection idea, as pointed out by the authors.

* It is unclear whether there exist recent works that have proposed some methods for task-agnostic scenario. 

* No theoretical guarantee on the success of FedProTIP.

### Questions
None

### Soundness
2

### Presentation
2

### Contribution
2
