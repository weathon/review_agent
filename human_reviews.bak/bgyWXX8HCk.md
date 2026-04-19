# Trustless Audits without Revealing Data or Models

- Decision: Reject
- Scores: 6, 8, 6

## Abstract
There is an increasing conflict between business incentives to hide models and data as trade secrets, and the societal need for algorithmic transparency. For example, a rightsholder who currently wishes to know whether their copyrighted works have been used during training must convince the model provider to allow a third party to audit the model and data. Finding a mutually agreeable third party is difficult, and the associated costs often make this approach impractical.

In this work, we show that it is possible to simultaneously allow model providers to keep their models and data secret while allowing other parties to trustlessly audit properties of the model and data. We do this by designing a protocol called ZkAudit in which model providers publish cryptographic commitments of datasets and model weights, alongside a zero-knowledge proof (ZKP) certifying that published commitments are derived from training the model. Model providers can then respond to audit requests by privately computing any function F of the dataset (or model) and releasing the output of F alongside another ZKP certifying the correct execution of F. To enable ZkAudit, we develop new methods of computing ZKPs for SGD on modern neural nets for recommender systems and image classification models capable of high accuracies on ImageNet. Empirically, we show it is possible to provide trustless audits of DNNs, including copyright, censorship, and counterfactual audits with little to no loss in accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a method for privately auditing machine learning models as well as their training data based on the use of cryptographic commitments and zero-knowledge proofs. More precisely, the audit framework proposed is composed of two phases: one in which the training data as well as the model weights are cryptographically committed and one in which an audit function F is computed on the data along with a zero-knowledge proof of the result. To realize this, a novel zero-knowledge protocol for computing the backward pass of the training of a neural network is also proposed.

### Strengths
The paper proposes an interesting approach to be able to audit machine learning models such as neural networks in a privacy-preserving manner. In addition, the use of ZK-SNARKS provides strong security and privacy properties. 

One of the main novelty of the paper is the design of an approach for computing the backward pass of a stochastic gradient descent algorithm based on ZK-SNARK. Several optimisation tricks are also proposed to be able to make the approach more efficient and control the computation-utility trade-off. Overall, this enables to prove the training in privacy-preserving manner while previous works were only focusing on inference.

Overall, the paper is well-written although the addition of an outline at the end of the introduction would help to clarify its structure.

### Weaknesses
In the introduction, there is a bit of confusion between the issue of training the model privately using secure multiparty computation vs performing an audit in a collaborative manner. The description of ZK-SNARKs in Section 2 is also a bit too concise for a reader that is not already familiar with this concept. I suggest to add a few concrete examples of what x and w could be, in particular within the context of privacy-preserving machine learning. Similarly, an intuitive example would help to understand what information or properties could be encoded in the 2D grid of arithmetic intermediate representations. Similarly more details are needed to understand the concepts of KZG commitments, inner-products arguments or structured-reference strings. Otherwise, the paper is going to be difficult to follow for a reader that does not have already a strong cryptographic background

The limit on which functions can (or cannot) be audited with the proposed approach is not clear. For instance, it is not clear for me if the proposed approach could be used to prove fairness properties about the model. A generic discussion on the applicability but also limits of the method would help.

The experiments are conducted on a limited number of datasets and architectures and thus it is not clear how the approach would scale. For instance, I suggest to the authors to provide additional experiments with classical datasets such as MNIST and CIFAR and architecture such as Resnet. In addition, one shortcoming of the current experiments is that they only report the training accuracy but not the test accuracy. The cost should be also reported in terms of computational time in addition of the monetary cost to be more meaningful.

### Questions
-The notion of trust should be more specifically defined in the paper, in particular as the focus of the paper is to argue that the proposed approach is « trusttless ».
-How does a traversal ordering attacks works? 
-What are the test accuracies obtained for the different conducted experiments?
-How does the use of salt to prevent dictionary attacks when hashing the weights combine with the ZK-SNARKS?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper introduces a new method, ZkAudit, for enabling auditing of deep learning models without revealing data or weights using zero-knowledge proofs. The method is separated into two parts: ZkAudit-T, which can be used to prove that a particular set of weights was the result of training with SGD on a particular dataset; and ZkAudit-I, which can be used to verifiably compute an audit function on the datasets and weights. The methods are all based on ZK-SNARKs. The authors implement several optimizations to make the method work with SGD: rounded division, an improved softmax function, and lowered precision throughout the network. In benchmarking on both an image classification task and a recommender system task, the authors show that ZkAudit can be implemented with relatively low cost compared to other methods for audit questions such as copyright censorship detection.

### Strengths
The paper’s primary strength is in its identification of an important problem - the reluctance of model owners to share their models for auditing purposes - and the optimizations that the authors implement to improve the method’s performance over existing work. The evaluation on two different kinds of tasks also shows the promise of the method overall.

### Weaknesses
The primary weakness of the paper is that, while it shows that the technique has some promise, it is difficult to tell how practical the implementation would be in a real-world setting. The paper shows improvements in cost and performance over existing methods, but there is no comparison to non ZK-based methods. The paper could be improved by showing how far away ZkAudit is from the cost/time it would take if the auditors had access to the model and dataset or even if they were in an SMPC setting as the authors cite early in the paper. While I expect that ZkAudit will be slower or more costly than these other scenarios, it would be useful to know by how much - is it several orders of magnitude? Additionally, I believe the paper needs more discussion of the limitations of the method and how that would implement the practicality of the method. In the next questions section, I have some questions that could help the authors expand on this discussion.

### Questions
In the results sections, the authors show the cost of various experiments they ran. Can the paper provide more reference points for how these compare to prior work and alternative methods?

The fact that the model architecture cannot be shared seems like a major limitation. How does this restrict the types of audit questions and functions that can be shared?

The method is set up to not reveal the training data and weights, but it does reveal the output of the audit function $F$. Are there ways to limit the amount of information that the audit function reveals? For example, could I pass an audit function that would leak information that the owner would not want to share? Or does the fact that $F$ needs to be implemented as a ZK-SNARK prevent that?

One very common audit task is to detect demographic disparities in model performance. I think answering this question would be precluded by the fact that model architecture is not shared, but I am curious to hear more discussion on this task and how it fits in with ZK-Audit.

How would an output privacy method like differential privacy stack up against ZK-Audit in terms of its capabilities and tradeoffs? It would be nice to see some discussion of what ZK-Audit offers over those methods (for example, I imagine that there is no way to verify that a particular model was trained on a particular dataset with differential privacy, even if you released noisy weights and data).

### Soundness
4 excellent

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The work describes methods with which a model can be trained in a manner that can later be
scrutinized in a zero-knowledge manner. That is, the training (and inferences) results in a proof
with which the trainer can prove to verifiers (presumably users who want to perform inferences but
cannot have the model itself) any desired property about the data, training, (and inference) they
might be interested in like presence of undesired data in the training dataset, without revealing
anything else about the training data or model parameters. Importantly, there are no trust
assumptions with regards to the prover. To achieve this (with reasonable performance), the work
introduces number representations with configurable precision/scale to be used in the ZK-SNARK
approach which are required for the backwards network passes in the training process. The paper
also describes a performant means of computing softmax.

Experimental evaluations using relatively small recommender and image classification models are
then presented showing: (1) scalability of the approach in terms of the numerical representation
precision, (2) accuracy/cost tradeoffs where the cost is a function of numerical precision, and (3)
relative loss of precision as compared to typical 32-bit floating point models (i.e. baselines).
Three types of queries or "audits" are also demonstrated alongside their estimated costs which are
said to be within "reason" though I cannot tell what sorts of costs are to be expected.

### Strengths
+ Work makes it possible to audit both training and inference as well as any audit-relevant
  property of data and the trained model without revealing either.

+ Quite general auditing capabilities. All three auditing examples are compelling.

### Weaknesses
- Novelty/contribution may be limited (or unclearly stated). Backwards pass work includes numerical
  representation adjustments and a better implementation of softmax. The structure of the paper
  thus makes it looks like these two bits of work were the missing pieces of zk-SNARK work that
  prevented them from being realistically applied to SGD. These are presumed to be "optimizations"
  as in the related work statement:

    "All of the work we are aware of focuses on optimizing inference, so all of this work omits the
     softmax computation and does not optimize the backward pass. In this work, we leverage ideas
     from inference to optimize the forward pass but show how to compute full SGD in ZK-SNARKs and
     optimize the softmax."

  This suggests that while other works can perform SGD, its backwards pass, and/or softmax, but
  just not efficiently. The title of the work and other statements, however, suggests that this is
  the first work that can handle these computations at all.

  Further, the contributions in the numerical representations do not seem significant and softmax
  may be even omitted from model training (or inference) without significant changes to training or
  inference. There thus appears some incongruity with the stated contribution and methods achieving
  those contributions.

  Suggestion: clarify whether the work presents the first verification systems that includes
  data/model training and that to achieve this within the frameworks from prior works, only changes
  to numerical representation (and softmax) were needed. Alternatively, if there was more necessary
  to achieve the goal, describe them in more detail. Alternatively, it may be that prior works
  already achieved the goal with less performant system in which case the title of the work and
  contributions need to be clarified. In this last case, comparisons of accuracy and efficiency
  need to be provided (as in Figures 1, 2).

Smaller thing / suggestions.

- Model parameters are not revealed to verifiers but model architectures are. This is noted but the
  abstract is unqualified, please adjust the abstract to make this clear.

- Some tabular results do not state which operation they are measuring and whether it is of a
  single instance of that operation (for inference) or multiple / entire dataset. Please include
  this.

- For prior techniques which do not handle softmax at all or well, consider including versions of
  models without softmax alongside the softmax ones. By this I don't mean to merely use an existing
  softmax implementation as done in Section 5.3, but instead remove the entire softmax operation
  from the model.

- Table 3: include units in table or description.

- Some results that depend on dataset size could be better presented in terms of cost per instance
  (Copyright audit is one example). The currently noted "$108" is not

- Please include a bit more how operations are implemented via AIR in its background section.

### Questions
- Question A: With regards to the weakness noted in the weaknesses section, please provide noted
  clarifications.

- Question B: How does this work compare: Efficient Representation of Numerical Optimization
  Problems for {SNARKs}. Angel et al. 2022. I also see several papers in federated learning that
  leverage SNARKs for enhancing trust. If models involving softmax or similar steps to the
  presently-experimented ones were used, I imagine they would have similar accuracy problems. Are
  their solutions applicable?

- Question C: Can the numerical representation work described in the paper by applied to other
  verified computation problems beyond model training?

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair
