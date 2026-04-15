# Automated Search-Space Generation Neural Architecture Search

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 5, 5

## Abstract
To search an optimal sub-network within a general deep neural network (DNN), existing neural architecture search (NAS) methods typically rely on handcrafting a search space beforehand. Such requirements make it challenging to extend them onto general scenarios without significant human expertise and manual intervention. To overcome the limitations, we propose Automated Search-Space Generation
Neural Architecture Search (ASGNAS), perhaps the first automated system to train general DNNs that cover all candidate connections and operations and produce high-performing sub-networks in the one shot manner. Technologically, ASGNAS delivers three noticeable contributions to minimize human efforts: (i) automated search space generation for general DNNs; (ii) a Hierarchical Half-Space Projected
Gradient (H2SPG ) that leverages the hierarchy and dependency within generated search space to ensure the network validity during optimization, and reliably produces a solution with both high performance and hierarchical group sparsity; and (iii) automated sub-network construction upon the H2SPG solution. Numerically, we demonstrate the effectiveness of ASGNAS on a variety of general DNNs, including RegNet, StackedUnets, SuperResNet, and DARTS, over benchmark datasets such as CIFAR10, Fashion-MNIST, ImageNet, STL-10 , and SVNH. The sub-networks computed by ASGNAS achieve competitive even superior performance compared to the starting full DNNs and other state-of-the-arts.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper focuses on automatically generating  the search space for neural architecture search, based on a predefined super-network. It also proposes "Hierarchical Half-Space Projected Gradient (H2SPG)", a NAS method that aims at leveraging the identified hierarchy and dependencies within generated search space. The authors benchmark the proposed procedures for different image classification benchmarks with different super-networks.

### Strengths
+ Addressing the problem of automated search space generation is an important topic and deserves further attention. Approaching this via tracing possible paths in a super-network is a reasonable approach.
 + The proposed procedure is evaluated on relevant datasets and performs competitive when compared to strong baselines. The authors also demonstrate that H2SPG can be applied in a settings where DHSPG (Chen et al., 2023) is not applicable.
 + The paper comes with helpful illustrations of the proposed procedure (Figure 2, 3, 4)

### Weaknesses
The paper in its current form severely lacks clarity. A lot of technical jargon is used but not properly introduced and defined. That unfortunately also prohibits understanding the paper in the necessary depth to write a more constructive review. I list a few examples of not defined terms below:
  * It is often stated that the "remaining network [need to] continues to function normally", "remains valid", " it is risky to remove such candidates" etc. However, it is never defined what defines a valid/normal network - I assume it must be a directed acyclic graph where every node needs to be reachable from the root not. But this needs a formal definition
 * The "saliency score" (sometimes also called salience in the paper) of a graph is used for ordering candidate graphs.But it remains unclear how this score is determined.
 * "By default, we consider both the cosine similarity between negative gradient −[∇f(x)]g and the projection direction −[x]g as well as the average variable magnitude.": how is this similarity considered?
 * Further terms that lack a solid definition "joint vertex", "trace graph", "modular proxy", "redundant groups", "half-space projection"

In summary, reviewing this paper would be mostly guessing what the authors actually try to say. This is a pity since the work seems to have potential but the severe lack of clarity prevents any reproducibility. As a suggestion: the entire Section 3 only contains a single equation, but several of the terms listed above should be amenable to a formal definition. In a future revision, rewriting this Section to be more formal can be helpful.

A further drawback is that the proposed procedure still requires the definition of a super-network in the first place; it is thus not really automated.It seems that more open-ended approaches would be desirable that are not constrained by the original super-network.

The related work discussion on NAS is relatively shallow, covering with one exception only works from 2020 or earlier. It also lacks a discussion of prior works on automated search space generation.

Please note that the confidence of my review is caused by the lack of clarity of the exposition. However, I am highly confident that Section 3 requires a major revision to reach the required level of clarity for an acceptance at this venue.

### Questions
How does the work relate to earlier works on automated search space design?

Are there options to lift the requirement of pre-specifying a super-network or design a super-network also in an automated fashion?

### Soundness
3 good

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes ASGNAS, an approach to firstly automate the search space generation for NAS, and then use a hierarchical half-space projected gradient approach to search for the desired final architectures. The proposed approach has been evaluated on a number of benchmarks and practical NAS search spaces.

### Strengths
* The idea of automate the process of NAS search space generation/design makes a lot of sense. 
* The proposed approach has been evaluated in a number of different settings. 
* The structure of the paper is easy to follow, and many detailed examples have been presented in the appendix.

### Weaknesses
* The actual technical contributions are a bit misleading. It seems the paper tries to claim that the automated generation of the search space is an important step, but it is not very clear how this is achieved. For instance, what do you mean by "general DNN"? What do you mean by the "search space" of a general DNN? All these terms are used in a quite arbitrary way, and reads very vague. It seems more like an approach to prune a network, rather than designing an actual NAS search space. 

* Missing important related works. Given that the paper claims to contribute a new approach to automate NAS search space, it is very surprising to me that many existing works in this area are missing: not to mention the new work like GPT-NAS, but even earlier works like RegNet or AutoSpace are not in the related work (some of them do appear in the Appendix). I think it is very difficult to position this work without explicitly comparing with that line of work. 

* The results are limited, and the performance of the proposed approach is not very convincing. For instance, when showing the C10 results we can see approaches such as TENAS, but when it comes to ImageNet results, it is very strange that many newer (and better) approaches are missing. For instance, what about SGAS or ZenNAS? It looks like the results were very much cherry-picked.

### Questions
Please see the above.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper highlights the constraints of traditional Neural Architecture Search (NAS) in DNN design, particularly the need for a pre-defined search space. In response, they introduce the Automated Search-Space Generation Neural Architecture Search (ASGNAS) system. ASGNAS aims to automatedly generate the search spaces and uses a graph algorithm combined with an optimizer named Hierarchical Half-Space Projected Gradient (H2SPG) to pinpoint and eliminate redundant structures in DNNs. Through this method, ASGNAS constructs a streamlined sub-network. The research includes tests of ASGNAS on various DNN architectures and datasets to demonstrate its functionality and potential benefits.

### Strengths
1.	The authors introduce the Hierarchical Half-Space Projected Gradient (H2SPG) algorithm, which optimizes network sparsity while considering constraints.

2.	The ASGNAS algorithm proposed by the authors doesn't require pre-training of the input DNN and the final output network doesn't necessitate re-training. Comparative experiments show ASGNAS has a search efficiency edge.

3.	The method introduced by the authors incorporates sparsity as a constraint, potentially allowing users to freely choose the desired level of sparsity in the resulting network.

### Weaknesses
1.	The paper introduces an 'Automated Search Space Generation' method, yet it seems to depend on a predefined super-network. In their context, the definition of the search space is unconventional. The search space is defined as a set of removal structures from the provided super-network, and this super-network is presumed to encompass all operation and connection candidates. While the authors have streamlined the problem, crafting such a general DNN still demands careful and time-consuming design by experts. Specifically, searching under the same scenario with different super-networks might lead to vastly different architectures. This implies that the proposed neural architecture search system isn't wholly automated, as important elements like the design of the super-network rely on human intervention. It would be better if the authors could provide further discussion.

2.	Rather than exploring new architectures, the proposed method appears to serve as a paradigm that optimizes and compresses a given architecture from a structural standpoint. With this in mind, it is advisable for the authors to engage in a discussion or comparison with NAO[1] and NAT[2].

3.	The ablation study presented needs improving. To demonstrate the effectiveness of H2SPG in handling constraints, the authors compare it against DHSPG. However, the optimization problem input to DHSPG omits constraints, while H2SPG incorporates them. This doesn't serve as a valid comparison to affirm H2SPG's efficiency. A more logical comparison would be between H2SPG and a version of DHSPG that incorporates standard constraint handling methods, such as penalty methods, rather than comparing it to an optimizer that neglects constraints.

4.	In the 'Segment Graph Construction' section, the details of how the trace graph is produced is missing. From Figure 2, the generation of the segment graph and the variable partitioning process seems a typical post-processing of computation graphs in deep learning frameworks (e.g., PyTorch), such as function tracing and operator fusion. It would be helpful if the authors could clarify or provide more information. 

5.	To bolster the credibility and demonstrate the effectiveness of the proposed method, it would be advantageous to apply it to architectures that have already been searched and have achieved state-of-the-art (SOTA) performance. Examples include OFA[3] (which is available on GitHub under the model name ofa-note10-64, boasting a top-1 accuracy of 80.2% on ImageNet) and NAT-M4 [4] (with a top-1 accuracy of 80.5% on ImageNet). If the proposed method succeeds in further enhancing and compressing these models, it would provide strong support for its effectiveness and superiority.

6.	The paper could benefit from a hyperparameter analysis. Specifically, in the 'Hierarchical Search Phase', the authors propose a method that requires the use of SGD for preliminary optimization of the network. It raises the question of how sensitive the results might be to the SGD hyperparameters, such as the number of iterations and the learning rate. Are there significant variations in the search outcomes based on these parameters? Additionally, the need for SGD-based network initialization raises the question of whether the method still depends on an input pre-trained network, and if so, how this aligns with the paper's objectives.

7.	The constraint handling approach introduced by the authors and the adopted HSPG optimizer seem to operate as two independent modules. This raises the question of why the authors chose to utilize HSPG specifically over other potential optimizers. It would be insightful if the authors could provide an additional ablation study.

### Questions
See Weaknesses.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The authors proposed ASGNAS, which is essentially a network pruning method. ASGNAS analyzes the architecture, identifies the removable components, and applies a greedy method to iteratively remove some of the removable components according to the saliency scores and the validity constraints, until the number of removed components reaches a certain value.

### Strengths
The proposed H2SPG extends HSPG to consider the hierarchy and the validity.

### Weaknesses
My main concern of this work is its performance compared with some SoTA methods. On ImageNet, the authors applied ASGNAS to the DARTS search space. As shown in Table 3, ASGNAS does not show improvement agains some DARTS based methods. As the authors also mentioned, it may be caused by the limitation of the proposed formulation. In addition, works like EfficientNet, MixNet and OFA can already achieve much higher performance with comparable or smaller FLOPs. 

Second, fine-grained search space like channel numbers of conv and linear projection is an important need for NAS. In practice, we often need to do some fine-grained pruning given an initial model and maintain its performance. In the paper, ASGNAS is operating at layer/operation level. I'm wondering if it can be applied to do fine-grained NAS.

### Questions
I'd like to see the authors' response on the performance issue. As there are many NAS works, tackling the search problem with various formulations and strategies. At the end, performance is one of the most critical factor.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
