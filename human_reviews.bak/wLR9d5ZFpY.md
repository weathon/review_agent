# No Training Data, No Cry: Model Editing  without Training Data or Fine-tuning

- Decision: Reject
- Scores: 6, 3, 3

## Abstract
Model Editing(ME)--such as classwise unlearning and structured pruning--is a nascent field that deals with identifying editable components that, when modified, significantly change the model's behaviour, typically requiring fine-tuning to regain performance.
The challenge of model editing increases when dealing with multi-branch networks(e.g. ResNets) in the data-free regime, where the training data and the loss function are not available.
Identifying editable components is more difficult in multi-branch networks due to the coupling of individual components across layers through skip connections. 
This paper addresses these issues through the following contributions.
First, we hypothesize that in a well-trained model, there exists a small set of channels, which we call HiFi channels, whose input contributions strongly correlate with the output feature map of that layer.
Finding such subsets can be naturally posed as an expected reconstruction error problem. To solve this, we provide an efficient heuristic called RowSum.
Second, to understand how to regain accuracy after editing, we prove, for the first time, an upper bound on the loss function post-editing in terms of the change in the stored BatchNorm(BN) statistics.  With this result, we derive BNFix, a simple algorithm to restore accuracy by updating the BN statistics using distributional access to the data distribution.
With these insights, we propose retraining free algorithms for structured pruning and classwise unlearning, CoBRA-P and CoBRA-U, that identify HiFi components and retains(structured pruning) or discards(classwise unlearning) them. CoBRA-P achieves at least 50% larger reduction in FLOPS and at least 10% larger reduction in parameters for similar drop in accuracy in the training free regime. In the training regime, for ImageNet, it achieves 60% larger parameter reduction. CoBRA-U achieves, on average, a 94% reduction in forget-class accuracy with a minimal drop in remaining class accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper deals with the finetuning-free model editing of ResNet models without accessing the original training data. The authors hypothesize that High Fidelity (HiFi) components of the model take charge of overall performance retainment and propose determining the pruning parts from a model based on the reconstruction score. The authors further provide a novel theoretical analysis of the batch normalization statistic to characterize the model performance after editing. Evaluation was performed over model pruning and class-level unlearning tasks.

### Strengths
* This paper provide a novel theoretical analysis on batch normalization statistics to discuss post-edited model performance

### Weaknesses
* **Limited applicability of the proposed method**
  * Although ResNet models are still popular in some cases, given that Vision Transformer (ViT) or other transformer-based models are dominant in many applications, the aim of this study limits its impact compared to previous work on model editing [1].
  * Could the insights provided in this work have some implications for the transformer-style models?
* **Limited validation scope**
  * Although this paper provides some theoretical insights, the empirical validation is too weak in terms of 
 the number of baseline methods, datasets, and experimental settings.
  * Could more baseline methods for the unlearning task be considered? Either data-free [2] or not [3].
  * Could more datasets be considered here for the unlearning task?
* **Insufficient empirical advantage**
  * The authors claim that the proposed method achieves a good trade-off between accuracy and efficiency. However, the proposed method actually could not achieve good accuracy compared to baseline methods, and the benefits of enhanced efficiency are also not so strong on both pruning and unlearning tasks.
* **Reliance on external data (through distributional access)**
  * Although the proposed method does not use an explicit training dataset on which the mode is trained, it still requires some samples from a similar distribution. This weakens the practical usefulness of the proposed method compared with truly data-free methods such as task arithmetic-based unlearning [2]
  * Could the authors provide an ablation study for the size of the external dataset used for proposals?
* **Bad presentation quality**
  * In the introduction and experiment section, the author does not insert space between paragraphs, which makes the reading hard.
  * The quality of the figure and table is so bad in terms of font size and resolution.
  * There is incorrect labeling of assumption 5 in line 328
  * Notations are complex beyond need and somewhat unclear. One example is lines 177-178.



> Reference
1. Decomposing and Editing Predictions by Modeling Model Computation, Shah et al. 2024
2. Editing Models with Task Arithmetic, Ilharco et al. 2024
3. Decoupling the Class Label and the Target Concept in Machine Unlearning, Zhu et al. 2024

### Questions
See the weakness section.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper mainly focuses on the model editing task, emphasizing the setting without training data or loss functions.
To detour access to the data or loss functions, the authors investigate the 'distributional' behavior of network layer outputs, which is not a 'sample-wise' behavior. Based on the finding that a very limited number of components of networks contribute to the learned outputs (called **HiFi** components), the authors have proposed to freeze the HiFi components and adjust the batch normalization to compensate for the changes in the distributional behavior. To verify their approaches, they have provided two types of tasks, i.e., pruning and unlearning.

### Strengths
**Strength 1:** The main strength of this paper is that the authors' viewpoint to scrutinize the distributional behavior of networks rather than the sample-wise network sensitivity can be a key strategy to control or edit the learned models.
- The strategy seems to be widely applied to various long-aged problems across multiple related societies, e.g., continual learning, explainability, and pruning or unlearning, which are tested in this paper.

### Weaknesses
**Weakness 1:** Limited understanding of how the learned knowledge relates to the distributional behaviors of models
- The main weakness of this paper is the limited understanding of how keeping the HiFi part results in keeping the knowledge of learned models. Otherwise, how tuning the HiFi part results in forgetting the specific learned knowledge.
- At the conceptual level of understanding, it is quite convincing that the components showing similar distributional behaviors with the layer outputs are probably the crucial parts of the knowledge. However, it is not guaranteed theoretically. 

**Weakness 2:** Insufficient quality of presentation and writing
- I strongly believe this venue requires the highest presentation and writing quality. However, the submitted version contains too many grammar errors, unpolished sentences, and low-clarity visualizations, as follows:
- At line 47: a missing full name of 'CNN'
- At many parts: add a whitespace between text and '('
- At many parts: for citations, the form is inconsistent, e.g., at line 166, "behavior (Jia...; Shah et al., (2024))." is correct.
- At line 178: missing comma after i.e.
- At line 185: missing whitespace before "While"
- Figure 2: The size is too small to recognize the plots, formulations, and texts.
- Equation 3: it is better to keep the length within the text width of the page.
- At line 269: keep the name "HiFi"
- At line 328: It seems "Assumption 5" means A1 and A2 at the right upper part. The labeling of assumptions is not matched.
- At line 469: missing punctuation after "Training Details"
- At line 529: "loss" rather than "Loss"
- Figure 5 (in Appendix): The size is too small to recognize the contents.
- I strongly feel that the level of presentations and writing is not reaching the level of this venue.

**Weakness 3:** Limited comparison with other related works
- Although the authors have provided the 'Related Work' part in the Appendix, it seems insufficient to provide deep insights into this work beyond others.
- For instance, beyond the technically similar model editing methods, in-depth analysis of the prior works investigating the importance of weights or sensitivity measures of weights should be considered. I think that HiFi is another viewpoint to measure the importance of weights so that it has the potential to show further impact on continual learning (also without data of the past tasks) and explainability.

### Questions
**Question 1:** unclear notations in equations
- In the "What is Model Editing" part on line 176, to my understanding, 'B' is the number of components (not an individual weight, but a group of weights) in the model. Therefore, the equation, $\|\theta\|-B$, looks wrong because $\|\theta\|$ is commonly used for the number of weights, not components. Would you clarify the equations?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper addresses the problem of model editing (specifically, structured pruning and class unlearning) for deep neural networks when training data is not inaccessible. The authors propose the concept of "HiFi components", which are identified as a small subset of channels in each layer being responsible for the model's output. Detecting "HiFi components" could be solved by measuring the reconstruction error of these channels. However, due to the unavailable training data, the authors propose a heuristic "RowSum" to identify the similarity between distributions of input contribution and output feature map in a layer. Then HiFi components are the components having a high correlation(/similarity) between input channel contributions and the output feature map. To restore the model's accuracy after editing, the authors derive an algorithm called "BNFix" to update BN's statistics using only distributional access to the data distribution. Two algorithms COBRA-P and COBRA-U are proposed to find whether retaining or discarding HiFi components in pruning and unlearning, respectively. Empirical evaluations on CIFAR-10/100 and ImageNet datasets show the effectiveness of their approach in maintaining competitive accuracy.

### Strengths
1. The paper tackles the problem of model editing without accessible training data for the circumstances of structure pruning and class unlearning. 

2. Identifying the HiFi component with the proposed correlation measure is interesting to me.

### Weaknesses
1. While the concept of HiFi components is interesting, the technical novelty of the RowSum heuristic and BNFix algorithm appears limited. There are many papers proposing to update BN's parameters, a similar strategy to the one in this paper. 

2. The theoretical analysis focuses on providing upper bounds on the loss function, however, K is the largest eigenvalue of the hessian, which might not be tight enough as a guarantee. 

3. The overall writing and organization of the paper could be improved significantly. The presentation of the main framework and the transition between different concepts in sections should be intuitive.

### Questions
1. In Section Introduction, How do photos from a personal device constitute samples of a large collection of photos having similar distributions?

2. "In Figure 2, we show the relative reconstruction error after removing filters from a selection of layers of a ResNet50 trained on CIFAR10". Could you explain how to get Fig. 2 in detail, which is a key assumption in this paper? 

3. The introduction of HiFi components and the section of BN fix seem disjointed. Could you provide a clearer connection between these two concepts and explain why only BN's statistics are fixed?
 
4. Is there anything additional information that needs to be stored during training time for the proposed methods to work? 
 
5. The empirical evidence is primarily based on CIFAR-10/100 and ImageNet datasets, and it would be beneficial to evaluate the methods on more datasets and tasks.

6. There are many citation errors in the text. Please carefully check. The font of the figures is really tiny,  making them very difficult to read.

### Soundness
3

### Presentation
1

### Contribution
2
