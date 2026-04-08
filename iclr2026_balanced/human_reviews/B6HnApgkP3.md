## Human Reviewer 1

### Summary
This paper proposes an adaptation of the Feature Selection Layer (training process) to the context of post-hoc feature attribution for tabular data. The proposed method consists of freezing the neural network to be explained and placing a linear layer between the input and the first layer of the frozen architecture, where each neuron has a 1-to-1 correspondence with an input feature. The augmented network is trained to replicate the original network’s outputs while keeping the original network frozen (i.e., only the added layer is trained). The idea is that, at the end of training, the weights between the input and this layer can be used to express the importance of each individual feature. The method is evaluated using metrics related to pure performance (e.g., accuracy and F1) against the original model and the FSL method and, in terms of stability and separability against FSL and post hoc feature attribution methods, finding that the proposed method is in general able to identify relevant features outperforming competitors in separability and visual metrics.

### Strengths
- While the addition of a layer at the end of a pre-trained model is common in explainability literature, adding a layer at the beginning of the network to achieve the same goal is interesting.

### Weaknesses
In general, the paper suffers from a duality between feature selection and global feature attribution. The paper states that the goal of the proposed method is to “highlight the features the original model considers most important,” which falls under global feature attribution and interpretability. This differs from feature selection, which would require a different setup and comparison. The following review is based on this assumption, and the method is reviewed as a post hoc global feature attribution approach.

- There is a **mismatch between the claims and what is currently demonstrated** in the paper. **This method is not post hoc**. Post hoc methods do not modify the network’s decision process. While the method keeps the original model’s weights frozen, the network’s response (i.e., the activations of the frozen part) will differ from the original model. This can be verified by comparing the activations of the frozen network before and after training the added layer. If they differ, even marginally, then the decision process is different. In this context, the authors’ statement that the layer *“enhances interpretability and potentially improves predictive performance”* implicitly confirms that the method cannot be considered post hoc, since performance can change.

- **Weak evaluation setup: almost all metrics applied are better suited to feature selection rather than feature attribution**. For some of them, it is not clear how they are applied to feature attribution methods (see below). Feature attribution is a major area in explainability, and there are many metrics and procedures for global feature attribution methods. For example, visual separability metrics (employed by the authors) are appropriate for feature selection, since the goal is to select informative features that capture most of the state space. Conversely, in feature attribution, separability is not a goal, since the model could consider just a couple of highly correlated features as important due to shortcut learning. There are also concerns related to “ground-truth rankings”. While effective for feature selection, in explainability contexts the model may learn different rankings (which could explain the stability issues observed by the authors), highlighting that these two fields require different evaluation setups. In this context, the fact that the proposed method outperforms feature attribution methods only on separability metrics is an additional factor that limits the paper’s significance and contribution.
- **Limited novelty**: as stated by the authors, the paper does not modify the original FSL but proposes a slightly different training paradigm. The difference, from my understanding, lies in the initialization of the weights (to 1), the use of ReLU as the activation function, and keeping the main network frozen. In this context, the novelty seems very limited.
- **Missing details**: there are several missing details on how global feature attributions are computed for competitors (e.g., for stability). For example, are attributions computed as a sum over the full dataset or as an average? What is the value of K in K-fold (I could have missed it)?
- The **“Related Work” section is limited**. Only five methods (those used for comparison) are briefly cited, and the most recent among them is from five years ago. The paper should better contextualize the method within current and recent literature.


Minors: the post hoc methods used as comparisons are better characterized as baselines rather than state-of-the-art methods, given advances in feature attribution over the last five years.

### Questions
see weaknesses

### Soundness
1

### Presentation
1

### Contribution
1

### Rating
2

### Confidence
4

---

## Human Reviewer 2

### Summary
The paper aims at introducing an updated post-hoc version of Feature Selection Layer. The proposed algorithm inserts a series of independent weights at the input data and learns these weights by freezing the pre-trained model with L1 regularization loss. These weights are then the representations of the importance of a feature in a specific dataset. The model is tested on a series of tabular classification datasets and across different metrics.

### Strengths
The method provided is fairly simple and according to the authors results seems to work in a simple environment.

### Weaknesses
1. The algorithm, as highlighted by the author only works in simple, tabular binary classification problems and fails in multi-class where it is unable to identify local and class specific feature’s relationships. 
2. The algorithm produces some dataset level feature explanations (this is due by the fact that the weights are trained once and remain constants for all the samples). While most of the compared algorithms produce sample wise feature explanations. These two might be similar only in fairly consistent dataset and explains why the algorithm doesn’t work on more complex dataset or non tabular data. 
3. The paper overlooks a lot of details, some of these make the paper irreproducible and leaving the reader with a lot of doubts (the author states that the code will be made available if accepted, nonetheless a lot of crucial details are missing to better understand and reproduce the paper results from the paper):
    * The dataset splits are not highlighted and it is not clear on what the model is finetuned.
    * The architecture of the pretrained model is not specified. Although at line 355 the authors talks about a “complement” to their experiments by using a pretrained model (TabPFN), this doesn’t seem to be the architecture they used on the previous results.
    * The model seems to be tested on a single trained model and a single pretrained one (this is not enough for post-hoc methods)
    * It seems like the pretrained model is trained by the authors but it is not clear what hyperparameter they used to do so or what kind of split, optimizer, number of epochs, etc.
    * Similar applies to the finetuning of the weights.
    * Training and finetuning curves and graphs are missing
4. A few minor errors:
    * Multiple time the author talks about the FSL (both post-hoc and original) as a dense layer, this is not the case as they are simple elements-wise weights for the input or it can be seen as a diagonal matrix (this misconception is present both in the related works at lines 83-84 and in the proposed implementation line 121-122)
    * Equation 2 at lines 164 applies the activation function only to the weight and not at the element-wise multiplication of the weights and the input features x.
    * Figure 1 could be represented following the standard neural network representations with weights on the edges, resulting in a clearer description.

### Questions
These are more curiosity or possible reflection points for improvements.
1. What about using different architectures (both pretrained and to train)?
2. Have you tried to normalize the learnable weights so that they sum to one?
3. why not directly using the first layer of the pretrained network and combining the weights for each input? 
4. Why not for each sample freezing everything beside the first layer and temporarily finetune on a single sample and then merging the weights? (Going something along this line of direction would resolve the problem of computing dataset vs samples feature explanation)

### Soundness
1

### Presentation
1

### Contribution
1

### Rating
2

### Confidence
5

---

## Human Reviewer 3

### Summary
This work reinterprets the previously proposed "feature selection layer", which was designed to give partially interpretable medical diagnoses, as a generic blackbox explainer approach which can be attached to the input of any model as a set of sparse linear weights.  Recovery of important features is demonstrated on simple synthetic datasets

### Strengths
Sparse linear weights as an easily applied method for quickly understanding important inputs is seemingly novel and is generally applicable to any model.

Results show that sparse selection demonstrates recovery of important features on small synthetic datasets

### Weaknesses
Although recovery results are established on easy synthetic datasets, a comparison of the benefits and disadvantages with respect to well-established explainability approaches is not explored in great detail.

### Questions
- What are the pros and cons of FSL compared with other existing explainability approaches?
- How long does FSL take to train in comparison with other posthoc XAI approaches?
- What is the purpose of the t-SNE plots and silhouette score?
- What is the intuition for what the silhouette score is measuring and why is it being used instead of more typical metrics like inclusion/removal curves or visualization of important features?

### Soundness
2

### Presentation
1

### Contribution
1

### Rating
2

### Confidence
4