# Feature Learning in Attention Mechanisms Is More Compact and Stable Than in Convolution

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 5, 3, 1, 6

## Abstract
Robustness is a crucial attribute of machine learning models, A robust model ensures consistent performance under input corruptions, adversarial attacks, and out-of-distribution data. While the Wasserstein distance is widely used for assessing robustness by quantifying geometric discrepancies between distributions, its application to layer-wise analysis is limited since computing the Wasserstein distance usually involves dimensionality reduction, which is not suitable for models like CNNs that have layers with diverse output dimensions. To address this, we propose $\textit{TopoLip}$, a novel metric that facilitates layer-wise robustness analysis. TopoLip enables theoretical and empirical evaluation of robustness, providing insights into how model parameters influence performance. By comparing Transformers and ResNets, we demonstrate that Transformers are more robust in both theoretical settings and experimental evaluations, particularly in handling corrupted and out-of-distribution data.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper compares variance, stability, and intrinsic dimensionality of the representations learned via attention or convolution in transformers and CNNs. The authors first study this in a theoretical setting imposing assumptions on the input and formulation of the models, deriving the variance and Lipschitz constants for each of attention and convolution. They then provide arguments about intrinsic dimensionality and how they believe different components of the models (e.g., MLP or batch norm) impact these properties. In the end, they provide experimental observations on variance and stability of two attention and two convolution models as well as ViT and ResNet.

### Strengths
Understanding various aspects of success of commonly-used and successful models such as CNNs and transformers is important. This paper focuses on the stability of these models, which could shed light on another factor that plays a role in the remarkable success of transformers. Theoretical results in a clear setup are helpful, despite (some inevitable) shortcomings.

Some examples of interesting and insightful observations/findings provided in the paper:
- Dependency of variance of the outputs in the attention on dimensionality, while this variance only depends on the input variance in convolution (line 285).
- The implications of Theorem 4, showing that the Lipschitz constant of the Attn with practically-relevant values of the variables ends up being much smaller than that of Conv (line 346).
- Potentially intriguing arguments provided in 3.4 for intrinsic dimensionality,
- The experiments are ample for a theoretical paper (but not if the authors believe their main results come from the experiments) and the experimental setup seems suitable to investigate the main questions posed by the paper.

### Weaknesses
**The key weakness:** The main motivation of the paper seems lost. The paper does not motivate why a comparison on the variance of stability of transformers and CNNs is a question worth a paper, especially given the limited scope of the setup. Unless the introduction motivates this question, the paper essentially seems like it could be a section of a more thorough study, e.g., on the stability of transformers or on the comparison between CNNs and transformers. While it is not impossible that I am missing something, I do not see a way that this paper could be prepared for publication at ICLR without a major reconsideration of the content and the presentation.

**Other major weaknesses:**

1. Related to the key weakness above: the second paragraph of the introduction seems to jump from some general statements about transformers and CNNs (first paragraph) to questions the authors pose, without explaining why the questions are raised or why their answers matter. Same issue applies to the contributions listed in the introduction.

2. The literature review seems insufficient and shallow. The literature reviewed in the Related Work section is both old, and not closely related to what the authors study. While there are many recent theoretical works, both on feature learning, and on the behavior of transformers (see, e.g. [1-3]), not many prior works related to the topic of the paper are referenced. I believe this is yet another reason the manuscript does not motivate the main questions.

3. Some claims and statements seem unsubstantiated. I presume the authors do know why those statements hold, but the arguments are not communicated well. E.g., in line 295, the authors claim that they “*prove* attention can have a lower activation variance” through the analysis that follows. However, the analysis is on the Lipschitz constant, and the experiments do not seem to actually *prove* that (even in the sense of a “strong empirical evidence”, which still should not be referred to as a “proof”, especially in a paper that presents itself as a theoretical study). 

4. The experiments do not show convincing evidence of the theory. While some match, some do not, and the experiments in the appendix seem inconsistent. Moreover, the inconsistencies are not properly discussed.


**Other weaknesses:**

a. The Preliminaries section is unnecessarily long. Batch normalization or MLP for instance, are basic concepts that could be stated in one line and the definitions could be moved to the appendix.

b. How did the authors use persistence homology or TDA in the paper at all? The abstract claims the use of TDA, and there is a subsection on persistence homology, but there seems to be no analysis based on TDA.

c. Some basic definitions and concepts are provided in the core theoretical results of the paper, where they do not belong. e.g., the definition of the Wasserstein distance, or Theorem 5, are too basic to be stated in detail in section 3. They should be either in the preliminaries or in the appendix.

d. While intuitive, the arguments in section 3.4 lack rigor. The authors later claim (line 485) that they “prove” that attention leads to a lower intrinsic dimensionality, while I do not see anything close to a proof in section 3.4 (and, again, experiments are not a proof, especially if the evidence is not thorough and strong). 

e. There results in the appendix are not properly referenced, nor are they discussed in the main paper.

f. The authors claim that their analysis of the stability “has nothing to do with the model performance”) line 383, while there is often a tradeoff between expressivity (hence performance) and stability.

g. Arguments and explanations in lines 451-456 are not clear.

h. The discussion of Figure 2 seems unclear, with statements that do not clearly reference the evidence from the experiments.

i. This is a minor point, but I would not call Theorem 1 and 2 “theorem”, but rather a “proposition”, since they are direct results of the assumption that the input is a 0-mean Gaussian.



**References:**

[1] Von Oswald, J., Niklasson, E., Randazzo, E., Sacramento, J., Mordvintsev, A., Zhmoginov, A., & Vladymyrov, M. (2023, July). Transformers learn in-context by gradient descent. In International Conference on Machine Learning (pp. 35151-35174). PMLR.

[2] Abbe, E., Bengio, S., Boix-Adsera, E., Littwin, E., & Susskind, J. (2024). Transformers learn through gradual rank increase. Advances in Neural Information Processing Systems, 36.

[3] Radhakrishnan, A., Beaglehole, D., Pandit, P., & Belkin, M. (2024). Mechanism for feature learning in neural networks and backpropagation-free machine learning models. Science, 383(6690), 1461-1467.

### Questions
My main questions are implicit in the weaknesses mentioned. But two specific questions:

- Line 474: “We observe that while the distances in Attn become more variable after training, the distances in ViTs remain stable”. Where is this observed? (appendix? If so, why not reference or discuss?)

- How do you estimate the intrinsic dimension using kNN (line 476)? Perhaps this is standard, but I am not familiar with it, and the readers might not be either.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper provides an analysis on the feature map variance between basic convnets and attention only networks. The author then expands this analysis to known architectures using the same building blocks such as the ResNet18 and the Vision Transformer. The author finds that their attention networks feature embeddings have a lower variance than those produced by convolutional nets, however this does not hold when expanding the attn network to a vision transformer architecture.

### Strengths
- The paper provides an interesting result, showing that attention mechanisms learn feature maps with lower variance than convolutional networks.
- The paper is mathematically correct from my understanding

### Weaknesses
- I am struggling to understand what is different between the attn -> vit models as well as the conv -> resnet. Is it the exclusion of residual connections and normalisation layers?
- There is no indication of the performance of the networks in accuracy, assuming they are at convergence I would expect the accuracies to be in the low to mid 90's for cifar-10, but the paper does not indicate whether the networks reached this performance or not. In my opinion, this is an important detail as otherwise we may be comparing underfitted networks which presumably do not have the correct feature space configuration for optimal accuracy.
- The author presents the statistics, but does little analysis into what this means. Do we want low variance or high variance in our feature space?
- (Minor) The figures 11-17 in the appendix are interesting visualisations of the feature space, but would be more informative if we zoomed in on the blobs of features to see if there is separation within the blobs, even if it is small.
- (Minor) To add onto the above point, visualisations of the class specific latent representations would be more interesting than the entire space to determine whether despite high or low variance of features, is our feature space semantic.

### Questions
I would like to state that currently I am very borderline on this paper as I struggle to find a clear takeaway other than statistics of the variance between different architectures feature spaces. I would appreciate if the author could address the points that I have made in the weaknesses section as well as clarifying what the takeaway of the paper is. I will always agree that our field needs more papers that provide methodologies to analyse the internals of our networks, but it is important that we also provide insight into what these analyses mean.

### Soundness
3

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
3

### Summary
The submission investigates the output variance and smoothness of non-residual (i.e., without skip connections) Attention and Convolutional layers. The authors demonstrate that attention is more compact and stable than convolution. Specifically, after presenting background on Transformers, Convolutions, and ResNets, the submission reviews recent results on the smoothness properties of attention in a mean-field framework. In Theorems 1 and 2, the input variance of the activations is derived, while Theorems 3 and 4 provide an upper bound on the Lipschitz constant of attention and convolution. Finally, experimental results on toy models (Conv and Attn), ResNet18, and a small ViT are presented, where activation variance, Wasserstein distances, and intrinsic dimensions are recorded across layers during training.

### Strengths
- The paper addresses an interesting question: understanding the theoretical differences in smoothness between attention-based and convolutional models, two widely used components of modern deep learning.
- The authors acknowledge that the theoretical findings do not transfer well to models with skip connections and normalization.
- The theoretical results appear rigorously proved.

### Weaknesses
In my view, this submission is not yet ready for publication at ICLR.

*In terms of writing:*
- The paper is challenging to follow. For instance, while the abstract and introduction repeatedly reference "feature learning," this term is not mentioned again in the rest of the paper. Another example is the mention of masked attention (l. 361), whereas no mask is used in ViTs. The motivation for using the $W_2$ metric, in my opinion, is not well explained (l. 307).
- The paper consists mostly of background material until page 5.
- The persistent homology part discussed in Section 2.5 is not referenced again in the rest of the submission.
- The related work section is very brief, with insufficient discussion of prior work. For instance, [2] also discusses the Lipschitz constant of self-attention.

*Regarding the theoretical contributions:*
- For attention, I am unsure about the novelty relative to previous work on the smoothness of attention, particularly Theorem 3.5 from Castin et al. (2024) and computations by Geshkovski et al. (2024).
- Theorem 5 should be replaced by a concrete result concerning deep ConvNets and Transformers rather than the actual theorem followed by a discussion (which I personally find unclear).

*Regarding the experimental contributions:*
- I am not convinced that the empirical results validate the tightness of the bounds in the theorems, or that they are sufficiently related to the rest of the paper.
- Additional experimental details in Section 4 are needed. How is activation variance computed? Over how many samples? Are these training or test samples?
- Generally, stacking attention layers without residual connections is very poor practice, as shown in [1] (which, in my opinion, should be discussed in the paper).

### Questions
- Could you clarify the differences between your proof technique and those of Geshkovski et al. and Castin et al.? What are the additional contributions?
- I would be very interested in seeing the accuracy of both models on CIFAR. Could the authors provide these numbers? I would especially expect the attention model without residual connections to perform poorly, as suggested by [1].
-The measure formulation of attention is also valid a finite number of tokens, correct?
- From what I understand, the empirical observations diverge from the theoretical results for real-world models (with residual connections and normalization) but align with the results for deep models without residual connections or normalization. Is this correct?

[1] Dong, Y., Cordonnier, J. B., & Loukas, A. (2021, July). *Attention is Not All You Need: Pure Attention Loses Rank Doubly Exponentially with Depth.* In International Conference on Machine Learning (pp. 2793-2803). PMLR.  
[2] Kim, H., Papamakarios, G., & Mnih, A. (2021, July). *The Lipschitz Constant of Self-Attention.* In International Conference on Machine Learning (pp. 5562-5571). PMLR.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
1

### Rating Number
1

### Confidence
4

### Summary
The authors propose a mean-field regime study of attention and convolution, and argue the attention mechanism is more robust to variations in input data distributions, enabling more stable feature learning. They observe such conditions do not actually hold in ViTs, which are more aligned with ResNets in terms of behaviour. They demonstrate lower intrisinc dimensionality in feature learning of attention mechanisms wrt convolutional ones, however these characteristics do not persist in comparisons of ViTs and ResNets.

### Strengths
Strengths: 
- Originality: I do not know other papers performing the same type of analyses. The authors develop or apply the theory in novel ways to draw some conclusions about the attention mechanism and convolutional layers. Therefore the work could be considered novel. 
- Quality: the theoretical developments look sound with respect to the assumptions that are made ...
- Clarity: the paper is clearly written and easy to follow. 
- Significance: unclear, not a strength point ...

### Weaknesses
I start the list of weaknessess with a complementary comment to the strenghts. 

- Quality: ... however the experimental results are limited and undermine the usefulness of the theory. 
- Significance: the significance or importance of the paper is not particularly clear. There does not seem to be any useful consequence of the theory developed. It is not clear what point the authors are trying to make because the lower variance and Wasserstein-lipschitz condition impact on possible applications (e.g. training stability and convergence, data efficiency, robustness, generalization, differential privacy etc.) are not mentioned or discussed (and if they are, it's more to state their irrelevance for real applications.  The authors should strongly motivate the utility of their study and how it produces insights that can lead to future useful developments. For instance, the authors could consider Differentially Private (DP) SGD training ,where batch normalization (BN) is not allowed and the lipschitzness (of the per-sample gradients in this case) has a strong impact on the training accuracy. If the authors could find a relationship between their wasserstein-lipschitzness and the ones of the per-sample gradients, it could have some practical impact on DP. Similarly, it would be interesting if the authors could find at least some toy practical applications in which their findings could show their possible impact.


- The CIFAR-10 experiments are conducted on extremely small models, it's unclear whether the findings generalise to both larger scale datasets or larger models. Furthermore, the training of transformers (and of CNNs too) is strongly regularised with augmentations and training tricks. It's not so clear whether the findings hold under such forms of regularization (see question about training details)
- Seveal works compare different aspects of transformers robustness and generalization. Many works have found that the claimed ability of attention mechanisms to focus on the whole of an input has little to no impact on the robustness of the learnt features, outlining that training tricks like pre-training and training procedures have larger impact than the inductive biases of the convolutional/attention mechanisms  [1,2,3]

[1] https://arxiv.org/abs/2207.11347
[2] https://arxiv.org/pdf/2310.16764
[3] https://arxiv.org/abs/2310.19909

### Questions
- How were the models trained? VITs, for how small, should not converge easily if trained from scratch on CIFAR-10. Did the authors use pre-trained models? Could you please provide full details about all the training details and procedure? Furthermore the absolute accuracies and losses of the models should be reported and compared (we are talking of learning dynamics, if the models do not show similar accuracies of lossess comparisons between models that have learnt completely different things, or haven't learnt anything at all don't make sense).

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper presents a theoretical and empirical comparison of attention mechanisms and convolutional layers, focusing on their feature learning properties, including Lipschitz continuity, intrinsic dimensionality, and stability. It claims that attention mechanisms yield more stable and compact representations than convolutional layers and validates this through theoretical bounds and experiments on various architectures (Vision Transformers (ViTs) and ResNets).

### Strengths
* The paper provides a rigorous theoretical analysis of the feature learning characteristics of attention versus convolution

### Weaknesses
* Since attention and convolutional architectures can be combined in practice, it would have been useful to explore hybrid models or discuss scenarios where attention layers supplement convolutional layers, as is common in many architectures.

* Training on CIFAR-10 may not yield strong performance for ViTs, as they typically require large amounts of training data. Could this limitation have impacted the results?

### Questions
Look at the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
