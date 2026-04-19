# Sparse Cocktail: Every Sparse Pattern Every Sparse Ratio All At Once

- Decision: Reject
- Scores: 6, 8, 3

## Abstract
Sparse Neural Networks (SNNs) have received voluminous attention for mitigating the explosion in computational costs and memory footprints of modern deep neural networks. Despite their popularity, most state-of-the-art training approaches seek to find a single high-quality sparse subnetwork with a preset sparsity pattern and ratio, making them inadequate to satiate platform and resource variability. Recently proposed approaches attempt to jointly train multiple subnetworks (we term as ``sparse co-training") with a \ul{fixed sparsity pattern}, to allow switching sparsity ratios subject to resource requirements. In this work, we take one more step forward and expand the scope of sparse co-training to cover \underline{diverse sparsity patterns} and \underline{multiple sparsity ratios} \textit{at once}. We introduce \textbf{Sparse Cocktail}, the \underline{first} sparse co-training framework that co-trains a suite of sparsity patterns simultaneously, loaded with multiple sparsity ratios which facilitate harmonious switch across various sparsity patterns and ratios at inference depending on the hardware availability. More specifically, Sparse Cocktail alternatively trains subnetworks generated from different sparsity patterns with a gradual increase in sparsity ratios across patterns and relies on an \textit{unified mask generation process} and the \textit{Dense Pivot Co-training}  to ensure the subnetworks of different patterns orchestrate their shared parameters without canceling each other’s performance. Experiment results on image classification, object detection and instance segmentation illustrate the favorable effectiveness and flexibility of Sparse Cocktail, pointing to a promising direction for sparse co-training. Codes will be released.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
Sparse Cocktail is a novel sparse co-training framework that can concurrently produce multiple sparse subnetworks across a spectrum of sparsity patterns and ratios, in addition to a dense model.

### Strengths
Key technical contributions include:

(S1) Simultaneously co-trains diverse sparsity patterns (unstructured, channel-wise, N:M) each with multiple sparsity ratios. The well-articulate problem is an important strength.
(S2) Uses iterative pruning with weight rewinding to segregate subnetworks of different sparsity ratios
(S3) Proposes a Unified Mask Generation technique to jointly produce masks of different patterns
(S4) Employs Dense Pivot Co-training to align optimization of diverse sparse subnetworks 
(S5) Performs Sparse Network Interpolation to further boost performance (relatively old trick)

Key experimental strengths include:

(S6) Sparse Cocktail achieves comparable or better performance than SOTA sparse co-training methods that focus on single patterns only. It generalizes previous methods while producing more subnetworks at once. Its performance can be on par with or even better than strong baselines such as AST and MutualNet. 
(S7) Besides evaluation on CIFAR10/ImageNet with ResNet/VGG, it also transfers effectively to object detection and instance segmentation tasks.
(S8) In ablation studies, key components like weight rewinding, network interpolation, Unified Mask Generation and Dense Pivot Co-training are shown to contribute to Sparse Cocktail's performance

### Weaknesses
(W1) The whole pipeline looks like a huge ensemble of existing techniques, such as the "Dense Pivot Co-training" stage from USNet and BigNAS, the "Sparse Network Interpolation" stage from AutoSlim and LotteryPool … However, the author did not make meaningful discussions in each stage, on their differences from prior arts. I would like to hear the authors clarify.

(W2) I would like to see some more relevant metrics such as training time, memory savings, or inference speed ups if any. Without those, it is hard or meaningless to fetch any real benefit of training with sparsity. 

(W3) Is Dense Pivot Co-training just weight rewinding (which is a pretty standard trick), or are they different (in which way)?

(W4) Why the three mask generations in Section 3.4. are called “unified”?

### Questions
See W1-W4

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposed a new joint sparse training algorithm called “Sparse Cocktail”, that allows for the selection of the desired sparsity pattern and ratio at inference. The benefits of using Sparse Cocktail for training sparse neural networks include the ability to produce a diverse set of sparse subnetworks with various sparsity patterns and ratios at once, making it easier to switch between them depending on hardware availability.

### Strengths
Overall, Sparse Cocktail can effectively generalize and encapsulate previous sparse co-training methods. Experiment results look promising, and paper writing is clear to follow (plus a lovely title :)
In more details:

-	Sparse Cocktail differs from other sparse co-training approaches in that it can produce multiple sparse subnetworks across a spectrum of sparsity patterns and ratios simultaneously, while previous approaches only focus on one or two types of sparsity patterns and/or with different sparsity ratios. 
-	The approach alternates between various sparsity pattern training phases, incrementally raising the sparsity ratio across these phases. Underlying the multi-phase training is a unified mask generation process that allows seamless phase transitions without performance breakdown. 
-	The authors also complement a dense pivot co-training strategy augmented with dynamic distillation, aligning the optimization trajectories of diverse sparse subnetworks. In the end, all sparse subnetworks share weights from the dense network, culminating in a "cocktail" of dense and sparse models, offering a highly storage-efficient ensemble. 
-	The paper shows that Sparse Cocktail achieves great parameter efficiency and comparable Pareto-optimal trade-off individually achieved by other sparse co-training methods. Sparse Cocktail achieves comparable or even better performance compared to the state-of-the-art sparse co-training methods that only focus on one sparsity pattern per model. Additionally, Sparse Cocktail avoids the need for co-training multiple dense/sparse network pairs, making it a more storage-efficient ensemble.

### Weaknesses
•	No discussion of training time cost. The proposed joint/switchable training appears to take much longer time than any single sparse training method. Please report the details and provide a fair discussion on training cost.
•	Hyperparameter setting was missed in Appendix C (empty - though mentioned multiple times in the main paper)!! This paper has so many moving widgets and it seems challenging to get all the hyper-parameters and settings right in practice.

### Questions
Overall the paper is clear, but several important pieces of information were missed, as pointed out in the weakness part.

### Soundness
3 good

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
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper aims at performing sparse cotraining to obtain multiple sparse networks at once with different sparsity ratios and sparsity types (unstructured, structured or N:M). The authors propose to use a combination of iterative magnitude pruning, unifying masks and interspersed dense training in order to obtain multiple subnetworks within the same network for different sparsity ratios and sparsity types.

### Strengths
The authors present a sparse cotraining method that can obtain subnetworks of different sparsity ratios and sparsity types at once.

### Weaknesses
I am concerned about the novel contributions of this paper, and the results presented in this paper are the combination of existing works with little novelty of its own.

1. The results are shown on different sparse subnetworks obtained from multiple sparse masks. However, it is likely that the performance of these sparse subnetworks is stable merely because of the relatively low sparsity reported in the paper. In order to see the effectiveness of the method, I would like to see the performance of the subnetworks with higher sparsity (> 90%) especially for unstuctured sparsity patterns.

2. The algorithm is not entirely clear from the Figure and methodology section. For example, how many sparsities is each sparse pattern trained for, what are the performances of each sparsity pattern and how does a subnetwork’s performance improve after merging (if it does).

3. The author’s don’t comment on the loss landscape of each of the subnetworks obtained during training. From previous work by Paul et al [1] I would expect each of the obtained subnetworks to lie in the same loss basin. In order to assess the effectiveness of the dynamic distillation step I would expect to look at the Hessian or the linear mode connectivity between the subnetworks obtained.

4. Additionally, the performance of the proposed method on ImageNet is poorer than AC/DC (in Table 1) which is a well established method. 

Overall my primary concern is that the novelty of this paper is limited as the authors have put together multiple existing methods (AST, AC/DC) in order to obtain multiple subnetworks at once. 
However, the attained subnetworks themselves have not been confirmed to be effective at higher sparsities.

[1] Paul, Mansheej, et al. "Unmasking the Lottery Ticket Hypothesis: What's Encoded in a Winning Ticket's Mask?." International Conference on Learning Representations 2022.

### Questions
1. How does Network Interpolation help, and at what stage of training is it used. It seems to be similar to the implementation of Lottery Pools [1].

2. Its not made clear how the N:M network and Unstructured networks obtained from IMP are kept similar to each other such that their weights can be interpolated. 

3. It is not clear to me why the authors choose to generate a total of 24 subnetworks by restricting the unstrcutured and structured sparse networks to 10 each. Is this a hyperparameter and why not choose additional networks at higher sparsity ratios?

[1] Yin, Lu, et al. "Lottery pools: Winning more by interpolating tickets without increasing training or inference cost." Proceedings of the AAAI Conference on Artificial Intelligence 2023.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
