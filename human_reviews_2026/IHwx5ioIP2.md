# Composable Sparse Subnetworks via Maximum-Entropy Principle

- Decision: Accept (Poster)
- Scores: 4, 8, 4, 6

## Abstract
Neural networks implicitly learn class-specific functional modules. In this work, we ask: Can such modules be isolated and recombined? We introduce a method for training sparse networks that accurately classify only a designated subset of classes while remaining deliberately uncertain on all others, functioning as class-specific subnetworks. A novel KL-divergence-based loss trains only the functional module for the assigned set, and an iterative magnitude pruning procedure removes irrelevant weights. Across multiple datasets (MNIST, FMNIST, CIFAR-10, CIFAR-100, tabular and text classification data) and architectures (MLPs, CNNs, ResNet, VGG), we show that these subnetworks achieve high accuracy on their target classes with minimal leakage to others. When combined via weight summation or logit averaging, these specialized subnetworks act as functional modules of a composite model that often recovers generalist performance. For simpler models and datasets, we experimentally confirm that the resulting modules are mode-connected, which justifies summing their weights. Our approach offers a new pathway toward building modular, composable deep networks with interpretable functional structure.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a method for training sparse networks that accurately classify only a designated subset of classes while remaining deliberately uncertain on all others, functioning as class-specific subnetworks.

### Strengths
The topic this paper focused on is very interesting.

### Weaknesses
1. The motivation of using Iterative Magnitude Pruning is not clear enough. Authors should explain clearly why using such pruning is a necessity.

2. Authors only conduct experiments on simple networks such as MLP and LeNet, making results not convincing enough. Can the proposed method be applied to other widely-used DNNs, such as ResNets, transformer-based models, VIT?

3. Can the proposed method be applied to large-scale and complex image dataset, such as imagenet or CUB200? MNIST is too simple.

4. Can the proposed method be generalized to other tasks, such as text classification and other tasks rather than classification?

5. What is the pratical usage of the proposed method? Authors can add a discussion for this.

### Questions
Please refer to the weaknesses, especially more experiments on larger models (e.g., ResNet, ViT) and more complex tasks (e.g., CIFAR, ImageNet).
If all my problems are addressed, I will raise my rating.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The proposed method creates modular NNs for classification by training each subnetwork to recognize only a subset of the classes. When the input is from another class the network is trained to output a uniform probability vector. Sparsity is achieved by iterative pruning. Subnetworks are then combined by weight summation, which is justified by a modified mode-connectivity analysis. Experiment results show the subnetworks learn their tasks, the merged models succeed at the combined tasks, and pruning leads to better merging.

### Strengths
Clever training objective to make each subnet an expert on its classes but to be agnostic (i.e., not interfere) on other classes. The combined loss is elegant, relative entropy from either a one-hot or a uniform. Experiment results show the method works as intended.

### Weaknesses
The promise of modular and interpretable networks makes sense, but the paper would be stronger with some direct demonstrations of downstream applications that could not be done with a standard monolithic network.

The mode connectivity analysis is not well justified since (as the paper notes) it evaluates a combined loss that neither model was trained on. The results in fig 5 are not surprising, because each separate model does terribly on the other model’s task. The more important question is how the combined model’s loss for classes in $\mathcal{R}\_{\rm A}$ compares to model A’s loss on those classes. In the ideal case, where summing parameters results in summing the logits, these losses will be equal because model B’s logits will be uniform for items in $\mathcal{R}\_{\rm A}$. This is the rationale for the method, but how close to you come to that?

### Questions
Would there be a benefit of end-to-end fine-tuning after merging? Fine-tuning just the last linear layer could reduce interference between subnets and wouldn't impact modularity or interpretability.

The Quasi-MaxEnt loss will bias predictions of merged models for the reason explained in the final paragraph of sec 3.1. As an alternative baseline, what about giving each subnetwork an “other class” output for $\mathcal{C}\setminus\mathcal{R}$ which is then discarded in the merged model?

Combining the above two comments, I bet last-layer fine-tuning would eliminate the difference between QME and ME, since the targets of these two loss functions are linear combinations of each other (in logit space). The same should be true of one-vs-all formulations. If the primary question is whether an effective modular representation can be learned then it seems fair to free up the final weight layer, which is just an affine map from this representation to logits.

Merging models requires a choice of correspondence between neurons (or filters) because of permutation invariance. The naive approach of identifying neurons in different models that have the same nominal index amounts to a random permutation. Can you do better by choosing a permutation, for example to minimize overlap in weights?

Rather than merging models, you could compute each model separately and sum their logits. This would avoid questions of mode connectivity. In fact you’re effectively doing this already when pruning is high (e.g., the 99% pruning for the MLPs).

L193: I'm not certain what is meant by the first term on the RHS. Is it $\mathcal{L}(f\_{\theta\_1}(\mathcal{D}))$ or is $(1-t)$ included? Why the asymmetry, basing the threshold only on model 1?

Table 1: It would help to state the number of classes for each dataset (I had to look up HAR (6) and yeast (10)). Why is entropy so low for yeast when it is near maximum for the other datasets? The dataset is imbalanced (I calculated entropy 1.729) but the target should still be uniform implying entropy 2.302. Also it would help to state $|\mathcal{R}|=1$ since that is given only later in the text.

Fig 2: what are the units of the entries? I would have guessed they’re predicted probability summed over the test set, but the MNIST test set is balanced and the row sums here differ across rows (though they look to be consistent across tables).

There’s a claim that merged models maintain near-maximal entropy for non-rewarded classes (line 308) but Fig 2 suggests a significant bias toward rewarded classes. I calculated the entropy for classes 1 and 2 in fig 2a and got 1.97 and 1.78, much lower than the reported 2.28. Is this an atypical example?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper examines whether independently trained class-specialized sub-networks can be combined into a single network that accurately identifies all classes.

The authors propose a method combining the maximum entropy principle with iterative magnitude pruning. The approach defines an objective function that uses KL divergence during module training to identify specific class subsets, while enforcing uniform distribution predictions for samples from other classes. The method employs iterative magnitude pruning with weight rewinding to enhance both functional specialization and module composability.

The authors evaluates the method using two image classification and two tabular classification datasets. The evaluation encompasses three architectures: a shallow MLP, a deep MLP, and a CNN. Results demonstrate that the specialized sub-network modules successfully identify specific class subsets and perform well when combined to handle all classes. The authors also conduct mode-connectivity analysis, revealing that the modules are mode-connected with a local minimum at the linear path's midpoint.

### Strengths
- The paper presents its hypotheses, questions, details, and conclusions with exceptional clarity.
- The proposed approach offers a novel and well-justified contribution.
- The results convincingly demonstrate the method's advantages over traditional training approaches, with mode-connectivity analysis providing valuable insights into local minima and composability outcomes.

### Weaknesses
- The method's application is restricted to classification tasks.
- The approach only identifies end-to-end modules and requires manual class assignment, limiting its utility for complex tasks where classification is just one component and intermediate labels may be unavailable.
- The experiments are confined to small-scale settings, and composability deteriorates with increased network depth, raising concerns about the method's scalability to deeper networks.

### Questions
- What factors contribute to the degradation of composability in deeper networks? This issue requires further investigation and discussion to understand its implications.
- Why does iterative magnitude pruning show greater effectiveness in deeper networks compared to shallow MLPs? Additional insights would be valuable.
- How does this work relate to broader research trends addressing increasingly complex tasks beyond classification?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors describe a method for composing independently trained sparse neural subnetworks into a single model without joint training or fine-tuning. Each subnetwork (“module”) is trained on a subset of classes called the rewarded set, using a modified loss based on the maximum-entropy principle: the model is encouraged to make confident predictions only for its rewarded classes and uniform (high-entropy) predictions for all others. Sparsity is induced through IMP to limit parameter overlap between modules. Once trained, these subnetworks (each specialized to different classes) are combined by simple weight summation, producing a merged model that can classify across all classes while preserving much of each specialist’s performance.

The authors empirically evaluate this approach on MNIST, Fashion-MNIST, and two tabular datasets, using small MLPs and a LeNet-style CNN. They report that modules trained with the MaxEnt objective achieve functional isolation, and that merged networks retain high per-class accuracy and high entropy on non-rewarded classes, indicating low interference. Mode-connectivity analyses further suggest that module parameters lie in nearby basins, explaining why their linear combination yields coherent behavior. The work claims that enforcing entropy separation, combined with sparsity, enables modular composition of neural networks in weight space.

### Strengths
- simple, interpretable composition mechanism, avoiding complex re-alignment or joint optimization.
- clear/principled loss function that explicitly enforces confident predictions on rewarded classes and uniform predictions elsewhere, promoting functional isolation
- demonstrates that iterative pruning improves mergeability, showing a plausible connection between sparsity and reduced interference across modules
- mode-connectivity analysis indicating that isolated modules occupy nearby low-loss basins, lending some theoretical (but limited) support for weight-space composition
- related to modular and compositional learning. In my opinion, this is an important and promising direction in DL these days.

### Weaknesses
Major issue: experiments are limited to simple datasets and small architectures (MNIST, Fashion-MNIST, shallow MLPs, LeNet)
- The scalability and effectivenes to realistic settings (e.g., CIFAR-10/100, ImageNet, Transformers) is unclear (probably it would not work in my opinion)

- no comparison to strong baselines such as model soups, Fisher-weighted merging etc
- The interaction between pruning and the MaxEnt loss is not disentangled. It is unclear which component drives functional isolation and mergeability
- Summation without alignment or normalization can be tricky to work
- the MaxEnt objective may suppress shared features and degrade generalization when classes share structure, limiting applicability beyond disjoint one-vs-all setups
- Claims of “modular structure” are not supported by structural analysis. There is no measurement of mask overlap, activation similarity, or connectivity between class modules
- The method lacks quantitative ablations and uncertainty metrics (e.g., calibration, leakage, loss barriers vs. number of merged modules).
- The writing and figure clarity varies a bunch. Some tables lack legends and experiment protocols (e.g., dataset reuse and merge order) are not sufficiently described.

### Questions
-  Please, if there is enough time in the rebuttal phase, compare the MaxEnt loss directly to label smoothing and confidence-penalty objectives, tuned so that non-rewarded classes approach a uniform distribution. This is essential to verify that MaxEnt provides more than a re-parameterization of existing regularization methods
- Conduct a 2×2 ablation (MaxEnt vs. standard cross-entropy) × (with vs. without IMP) while keeping final sparsity fixed. This would clarify whether isolation arises mainly from the loss, from pruning, or from their interaction.
- can the method compose modules on datasets with shared features, such as CIFAR-10 or CIFAR-100? Can you have some results on a small ResNet at similar sparsity?
- As you know weight summation can amplify parameter norms. Did you test weight rescaling, re-centering, etc before merging? Please report any normalization strategy (or justify why none is needed?)
- Quantify inter-module overlap. for example Jaccard similarity of masks, cosine similarity of activations, or participation coefficients. This will help support your claim of isolated modules.
- in the real world we often have overlapping or hierarchical class groups. Can this approach handle partially overlapping rewarded sets without large interference?
- How would the method extend to architectures with BatchNorm or LayerNorm? Did you attempt merging such models?
- beyond non-rewarded entropy, report per-class confusion or leakage rates, calibration error and loss-barrier heights as quantitative measures of interference after merging
- I would love to see some comparisons against model soups, Fisher-weighted merging, and Git Re-Basin under the same initialization and pruning regime to position your approach within current weight-space composition literature.
- Add legends and unit definitions in Table 1. specify the dataset split and permutation protocol in complete-merge experiments. plz give implementation details for IMP (rounds, pruning ratios, weight resets) and temperature settings for all architectures

### Soundness
3

### Presentation
3

### Contribution
3
