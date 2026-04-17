# HebbGate: Local Reward‑Modulated Gating for Continual Learning

- Decision: Reject
- Scores: 2, 4, 4

## Abstract
Neural networks that learn continually should acquire new tasks without revisiting old data and with small per-task overhead.  In parameter-isolation CL, existing approaches typically learn dense task masks via backpropagation, which couples mask learning to the backbone optimiser, adds training compute, and inflates memory with extra mask parameters.
We introduce HebbGate, a parameter-isolation method for continual learning that uses local, reward-modulated gates in place of backpropagated masks. Crucially, each task adds just one scalar per channel (not per weight), keeping memory growth tiny and the masks interpretable. A utilisation penalty discourages reuse of over-popular channels, and a $\kappa$-decay capacity warm-up lets new tasks explore larger masks before annealing to the target sparsity, mitigating order bias and improving forward transfer.
On CIFAR-100, Tiny-ImageNet-200, and ImageNet-100 with ResNet-18, HebbGate achieves best-known exemplar-free Class-IL final accuracy $A_{\text{last}}$ while a variant with task-specific BatchNorm further improves both $A_{\text{last}}$ and incremental accuracy $A_{\text{inc}}$ at the cost of only two additional scalars per channel per task. Additional experiments on Permuted-MNIST, Split-CIFAR-10, and lower-capacity backbones confirm that HebbGate’s gains extend beyond a single architecture or dataset. Overall, HebbGate offers a lightweight, transparent alternative for exemplar-free, single-head continual learning.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a gating method for parameters based on a reward modulated Hebbian rule, in which one scalar is added per neuron instead of per weight. This helps keep memory growth sublinear and mask interpretable.

### Strengths
1. HebbGate is a sensible algorithm satisfying realistic desiderata in class-incremental learning.
2. Extensive theoretical and empirical analysis are presented.

### Weaknesses
1. No Mention or consideration of recent related works in continual learning, e.g., most prior work in Section 2 is from 2017-2019.
2. The experimental setup is somewhat outdated. The experiments use CIFAR 10 and CIFAR 100 with AlexNet and ResNet.
3. No recent continual learning baselines included (see Table 1).
4. The empirical results are mixed, especially (1) EFC (only recent method included in baselines) outperforms HebbGate, (2) context (e.g., computational/memory overheard) for each baselines and HebbGate not presented.
5. Only average accuracy is presented in the main paper whereas other metrics critical to continual learning such as backward transfer should be presented. This can be visually inferred from Figure 3 but only roughly.
6. Conceptually, no discussion on how HebbGate applies to more recent transformer models.

### Questions
1. How does HebbGate apply to transformers?
2. Can HebbGate scale to more recent and challenging datasets, e.g., MTIL?

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper addresses the challenge of continual learning with a parameter-isolation approach. The authors first identified several issues with existing parameter-isolation strategies, such as dependence on backpropagation, memory overhead, and task bias. To address these issues, the authors proposed HebbGate, which applied a scalar mask to each channel. The update strategy of the mask takes activation energy, usage penalty, and margin reward, which do not depend on backpropagation. The authors then validated their proposed method on a few backbone models across different evaluation benchmarks, showing the improved performance.

### Strengths
1. The paper is well motivated

It is very interesting to consider parameter isolation without backpropagation and heavy memory overhead.

2. The presentation is clear

I appreciate the efforts of the authors to clearly explain the details in every section: motivation, method, implementation and experiment.

3. The proposed method is lightweight and effective

### Weaknesses
1. The motivation needs to be further justified

2. The contribution for forward transfer is not clear

3. The applicability to deeper networks is not clear

4. The presentation can be improved

Please see the Question section below for details.

### Questions
1. The motivation needs to be further justified

One of the drawbacks of existing parameter-isolation approaches, as claimed by the authors, is the bias of channels for early tasks (lines 48, 107-108). However, this is not well presented in the paper. Therefore, it is not a solid support to motivate the proposed method.

2. The contribution for forward and backward transfer is not clear

The authors claimed that the proposed method improves transfer (lines 100, 114, 131,217). However, this is not supported by a quantified evaluation. There are metrics for evaluating forward and backward transfer. I would recommend that the authors include this metric if such a point is the main emphasis in the paper.

3. The applicability to deeper networks is not clear

Current experiments are with shallow neural networks. I believe parameter isolation is more interesting when the model's capacity is much larger and the complexity of the tasks grows as well. 


4. The presentation can be improved

The authors refer to many tables and figures in the appendix in the experiment section. While it is not forbidden to do so, it largely reduces the readability of the paper.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces HebbGate, a local, reward-modulated gating mechanism for continual learning based on a three-factor Hebbian update. Instead of the dense, backprop-trained task masks often found in parameter-isolation approaches, HebbGate uses one scalar per channel (rather than per weight), updating via local activation, a global margin reward, and a utilization-aware penalty. The method incorporates a k-decay schedule allowing new tasks to initially explore greater capacity before annealing to a target sparsity, helping balance transfer and forgetting. Experiments on Permuted-MNIST, Split-CIFAR-10, and Split-CIFAR-100, across several architectures, demonstrate competitive or superior performance to state-of-the-art exemplar-free methods, with lower memory and computational overhead.

### Strengths
1. Parameter Efficiency: Each task adds only one scalar per channel, versus dense masks per parameter, resulting in negligible memory growth and making the approach scalable. 
2. Forward Transfer and Order Robustness: The k-decay schedule and utilization-aware gate initialization directly target the early-task monopoly problem seen in prior work, supporting more equitable channel budgets and improved forward transfer.

### Weaknesses
1. Limited comparison with recent methods: Only one method from 2024 has been compared with, rest of the compared methods are older. This reflects poorly on the claim of achieving state-of-the-art performance.
2. Poor Performance on task incremental learning for the CIFAR-100 dataset
3. Poor Performance on ResNet for class-incremental learning on CIFAR-100. The approach seems to perform poorly on ResNet but better on AlexNet, which seems strange.
4. The proposed method shows better performance on CIFAR-10. However, the total classes being 10, makes it a weak case for an incremental learning setup.

### Questions
Is there any reason why the authors chose not to give results on more established ImageNet-100 and ImageNet-1000 datasets for class-incremental learning?
Does the proposed method work better, when the number of classes are less?

### Soundness
2

### Presentation
2

### Contribution
2
