# Fed3+2p: Training different parts of neural network with two-phase strategy

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3, 3

## Abstract
In federated learning, the  non-identically distributed data affects both global and local performance, while clients with small data volumes may also suffer from overfitting issues. To address these challenges, we propose a federated learning framework called Fed3+2p. In Fed3+2p, we divide the client neural network into three parts: a feature extractor, a filter, and classification heads, and to train these parts, we present two types of coordinators to train client sets with a two-phase training strategy. In the first phase, each Type-A coordinator trains the feature extractor of partial clients, whose joint data distribution is similar to the global data distribution. In the second phase, each Type-B coordinator trains the filter and classification heads of partial clients, whose data distributions are similar to each other. We conduct empirical studies on three datasets: FMNIST and CIFAR-10/100, and the results show that Fed3+2p surpasses the state-of-the-art methods in both global and local performance across all tested datasets.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper presents a two-stage training scheme for personalized federated learning, using coordinators to manage client groupings. While the paper is well-presented with thorough experiments, there are concerns about novelty and areas requiring further clarification.

### Strengths
1. The paper is clearly written and well-structured.
2. The experimental evaluation is comprehensive, comparing multiple baselines across different datasets.

### Weaknesses
1. **The novelty of the approach is not sufficiently articulated.** It's unclear whether: (a) The use of coordinators for client grouping is novel. Have similar dataset/client splitting approaches been tried before? (b) The two-phase training scheme is not novel as it is frequently seen in personalized federated learning (pFL). In terms of progressive training, the idea has been explored in "FedBug: A Bottom-Up Gradual Unfreezing Framework for Federated Learning"), and a direct comparison would be helpful to better understand the novelty.

2. **Empirical validation.** (a) The paper lacks experiments in IID scenarios, which would help demonstrate the method's robustness across different data distributions. (b) The learning dynamics (e.g., train/test loss profiles) may be helpful to understand the method's behavior better.

### Questions
1. Why are different divergence measures used in different stages (KL divergence in the first phase, JS divergence in the second)? Theoretical or empirical validation for these choices would strengthen the paper.
2. Could the coordinator setup introduce any privacy concerns?
3. Can FED3+2P be combined with other federated learning techniques like FedDyn, FedProx, or FedExp? This could potentially enhance its performance further.
4. Can the authors clarify the statement "we turn off the Type-B coordinator in the second phase but retained the management functionality of the Type-A coordinator". Does this mean there is no second phase in this ablation? Also, additional experiments with random Type-B assignment would be informative.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper aims to improve both global federated learning and personalized feature learning. It first decomposes the model into three parts: a feature extractor, a filter, and classification heads. Then, the clients are divided into different groups according to their data distribution. The training process employs two types of coordinators within a two-phase training strategy to train different parts of models. Experiments indicate that the proposed Fed3+2p method surpasses existing state-of-the-art approaches in both global and local performance.

### Strengths
1. Propose a new framework to enhance both global and local FL performance.
2. Experiments indicate that the proposed Fed3+2p method surpasses existing state-of-the-art approaches in both global and local performance.

### Weaknesses
**Method:**

1. The proposed filter seems to be the key innovation in this paper. However, the ablation experiments presented in Table 2 indicate that removing the filter results in only minor performance declines of 0.3, 0.4, and 0.1 on FashionMNIST, CIFAR-10, and CIFAR-100, respectively. This suggests that the filter may offer limited utility.
2. This method involves partitioning clients into distinct groups. I hypothesize that this requires clients to upload their data distributions to the server, which presents a potential privacy risk.

**Experiment:**

3. Could the author provide a comparison with FedETF [1]? This method also aims to enhance both personal and global performance.
4. The experiments are conducted on small ConvNet architectures consisting of 2 or 3 convolutional layers. I am curious about the performance on larger networks, such as ResNet.

**Writing:**

5. The paper's writing requires improvement. Additionally, the notation is inaccurate. For instance, high-dimensional tensors, such as data $x$,  should be bolded as $\mathbf{x}$ in eq. (1).
6. In Section 2.1, citations should be included when discussing methods related to categories.
7. Regarding the methods, it is advisable to include an algorithm.

[1] Li, Zexi, et al. "No fear of classifier biases: Neural collapse inspired federated learning with synthetic and fixed classifier." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023.

### Questions
1. The proposed filter seems the key innovation in this paper. Could the authors provide a more detailed explanation of the filter's motivation? 
2. The filter's structure is unclear. Could the authors elaborate on its design and whether its implementation incurs significant computational overhead?
3. Could the author provide a comparison with FedETF [1]? This method also seeks to enhance both personal and global performance.
4. The experiments are conducted on small ConvNet architectures consisting of 2 or 3 convolutional layers. I am curious about the performance on larger networks, such as ResNet.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
The paper proposes a new federated learning framework called Fed3+2p, which aims to address the impact of non-iid data on global and local performance, as well as the overfitting issues that small-data clients may encounter. The framework divides the client neural network into three parts: feature extractor, filter, and classifier head, and trains these parts using a two-stage strategy with two types of coordinators. Experimental results show that Fed3+2p outperforms existing methods on the FMNIST and CIFAR-10/100 datasets.

### Strengths
The design idea of dividing the client neural network into three parts and training them using a two-stage strategy has some practical application value.

### Weaknesses
1. The setup of the paper is not clear. In general, in federated learning, training should protect the privacy of each client, and the clients selected for training in each round should be random. The method proposed in this paper requires obtaining the class distribution of each client, which to some extent violates the principle of privacy protection in federated learning. Additionally, controlling the clients participating in training in each round contradicts the standard setup of federated learning.
2. The method proposed in the paper is too simple, and I did not find any unique or innovative aspects.
3. The notation in the paper is confusing and the descriptions are unclear, making it difficult to read. For example, C is used to represent both coordinators and categories.
4. The experiments are insufficient:
-  The authors seem to have shown only one set of experimental results under a single setting, which has a high degree of randomness and is not sufficiently comprehensive.
- Important settings such as how many clients the data was divided into and how many clients were randomly selected for each communication are missing from the paper. Additionally, crucial details like how the number of coordinators should be determined are also absent.
- Many papers[1][2][3] that consider class imbalance in federated learning are not included in the comparison methods.

[1] On Bridging Generic and Personalized Federated Learning for Image Classification.

[2] No Fear of Classifier Biases: Neural Collapse Inspired Federated Learning with Synthetic and Fixed Classifier.

[3] Aligning model outputs for class imbalanced non-IID federated learning.

### Questions
Refer to the weakness.

### Soundness
2

### Presentation
1

### Contribution
1
