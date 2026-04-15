# Effective and Efficient Federated Tree Learning on Hybrid Data

- Decision: Accept (poster)
- Scores: 8, 5, 5, 6

## Abstract
Federated learning has emerged as a promising distributed learning paradigm that facilitates collaborative learning among multiple parties without transferring raw data. However, most existing federated learning studies focus on either horizontal or vertical data settings, where the data of different parties are assumed to be from the same feature or sample space. In practice, a common scenario is the hybrid data setting, where data from different parties may differ both in the features and samples. To address this, we propose HybridTree, a novel federated learning approach that enables federated tree learning on hybrid data. We observe the existence of consistent split rules in trees. With the help of these split rules, we theoretically show that the knowledge of parties can be incorporated into the lower layers of a tree. Based on our theoretical analysis, we propose a layer-level solution that does not need frequent communication traffic to train a tree. Our experiments demonstrate that HybridTree can achieve comparable accuracy to the centralized setting with low computational and communication overhead. HybridTree can achieve up to 8 times speedup compared with the other baselines.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper addresses federated learning with hybrid tabular data, where data from different parties may differ both in the features
and samples. The common framework for this kind of settings is federated learning using gradient boosting decision trees (GBDT). However, existing solutions suffer from high communication and computation overhead. The paper introduces HybridTree, a GBDT-based approach with some clever design. The higher layers of a HybridTree follows a standard GBDT design, exploring host features. The lower layers of the HybridTree are distributed among the guests. This is a form of bagging where each guest contributes one tree to the bag, utilising only the local data of the guest. At each HybridTree training iteration, there are $k$ bags corresponding to $k$ endpoints of the host (sub)tree.
Experiments on simulated and natural hybrid federated datasets show that the HybridTree can be up to 8 times faster than existing approaches, while maintaining the level of accuracy close to when all data are not protected by privacy and low communication overhead.

### Strengths
The paper has good motivation and good overview. The first part of the paper consists of a demonstration of the existence and abundance of meta rules in federated learning using GBDT, which provides clues about where to improve upon existing approaches, and the theoretical contribution, i.e. theorem 2 and 3, albeit being rather simple and straightforward from the mathematical point of view, provides an interesting view on how the following HybridTree algorithm tackles the inefficiency of existing approaches.

The second part of the paper, the introduction of the HybridTree algorithm, in my view is a clever and efficient use of boosting and bagging under the context of privacy preserving of federated learning. HybridTree is like 2D boosting with 1 dimension resembling GBDT addressing global (host) data, and with the other dimension spanning across local (guest) data, eliminating guest-guest communications and mitigating guest-host communications.

Experiments on both simulated and hybrid federated datasets are appear adequate for the problem that the paper addresses, with results favouring HybridTree with reasonable gaps over existing approaches.

### Weaknesses
I only have a few minor complaints about presentation, which makes it difficult to follow in the first read:

1. Definition 1 needs revising. At the beginning $S$ is defined as an intersection of split conditions but at the end you have split conditions not in $S$. It is intuitively understandable but it is mathematically incorrect.

2. In Figure 2b, you could have mentioned that $(L'_1, L'_2, L'_3, L'_4)$ can be $(L_1, L_2, L_1, L_3)$ as an example, and the audience would understand Theorem 2 better.

3. I did not understand theorem 3 at first. How does a meta rule that "ended by $F_g$ differ from a meta rule where $F_g$ is the last layer? Aren't they the same thing? Does the intersection operator here suggest ordering as well? After reading the proof of theorem 3, I understood what the author(s) meant. In any path from the root node to a leaf node, there is a possibility that a guest split condition stays above a host split condition, you just want transform the tree so that such possibility does not exist, is that right? If so, I suggest to restate theorem 3 in a way that is more understandable.

4. What has caught me a surprise is that theorem 3 is not used in a conventional way. Instead of using theorem 3 to improve upon a given tree, the authors redesign a new tree where guest split conditions are always below host split conditions. This is a good point that appears to have been presented somewhat lightly in the paper.

5. In my view, the flow of the paper could better if the authors presented the HybridTree inference before presenting the HybridTree training. Once the audience knows how inference works, the training components make much more sense.

### Questions
I do not have any major question.

### Soundness
3 good

### Presentation
2 fair

### Contribution
4 excellent

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper introduces HybridTree, a novel federated learning approach tailored for hybrid data environments, where data from different parties vary in both features and samples.  HybridTree facilitates federated tree learning by capitalizing on consistent split rules observed in trees, allowing the integration of knowledge from different parties into the lower layers of a tree. Using these insights, the paper proposes a layer-level solution to train trees without the need for heavy communication. Experimentally, HybridTree has shown comparable accuracy to centralized models while substantially reducing computational and communication overhead. In comparison to other baselines, it can achieve up to an 8-fold speedup.

### Strengths
1. The problem studied in this paper is interesting and important. This work proposes federated tree models on hybrid data, which expands the scope of current FL frameworks and makes an important contribution to the FL community.
2. This paper is original and technically sound. The main claims regarding the proposed setting are well supported in the methodology and experimental parts.

### Weaknesses
1. The paper is not easy to follow.
2. It is better to provide more analysis or explanation on the training process in section 4.1 to let readers well understand how HybridTree handles hybrid data and makes their contribution to the improvement.
3. Although there is a relatively thorough literature review in the related work part, I prefer to see a discussion on the relation of this work, especially the specific methods.
4. Some minor errors, see below.

### Questions
1. "Eq 2" —> "Eq (2)"
2. $D_{G_i}= \{ x \} (x \in R^{ d_{g_i}})$ —> $D_{G_i}= \{ x | x \in R^{ d_{g_i}}\} $,  $D_H= { x, y } (x \in {R^{d_h}})$ —> $D_H= \{ x, y | x \in R^{d_h}\} $
3. "Last, guests update the following lower layers of the tree using their local features and received encrypted gradients and send back the encrypted prediction values." —> "Last, guests update the following lower layers of the tree using their local features and **receive** encrypted gradients**,** and send back the encrypted prediction values."
4. "Then, the gain of the split value is defined by the loss reduction after split, which is" —>"Then, the gain of the split value is defined by the loss reduction after **the split**, which is" 
5. "The best-split point is selected among these split candidates. When reaching the maximum depth or the gain is always negative, the current node will become a leaf node." —> "The **best split** point is selected from these candidates. If the tree reaches the maximum depth or **if** the gain remains negative, the current node becomes a leaf node."
6. "FL on hybrid data is rarely exploited in the current literature. Zhang et al. (2020) proposes to…. Liu et al. (2020) applies transfer" —> "Zhang et al. (2020) **propose** to…. Liu et al. (2020) **apply** transfer"

7. “For simplicity, we start from a single-host with multi-guest setting” —>  “For simplicity, we start from a single-host with multi-guest setting” —> "For simplicity, we start from a scenario involving a single host with multiple guests."

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This work proposed a novel tree learning protocol for the hybrid FL settings where data are distributed both by features and by samples, a scenario that is less explored. The proposed method does not need frequent communication between parties and exhibits comparable accuracy to the centralized setting.

### Strengths
1. This work targets at a hybrid FL setting, where vertical FL setting integrates with horizontal setting. The setting has practical applications in many industrial areas but are less studied in research.

2. The observation of the existence of meta-rule is novel and leads to the development of a communication efficiency tree-based FL algorithm.

### Weaknesses
1. The paper is not very well-presented and is hard to follow. First of all,  it is unclear in the hybrid setting considered, what are the relative relations of the guest parties? In the introduction, it appears that they share the same feature space but have different sample IDs, however, in 3.1 they appear to have different dimensions and unclear alignment. It is suggested that the paper properly define the problem setting. A figure on how data is partitioned by different parties would also help.  Secondly, the algorithm is very messy with many undefined variables and notations, making it also hard to process. It is also not clear how features from different guests are used in a collaborative manner in the approach due to the above issues on presentation. 

2. The experimental results are biased. 1) The paper compares its model performance using data from multiple guests against FedTree and Pivot using data from one guest, therefore the model performance gap is largely due to the utilization of additional data parties, not to the benefits of the algorithm. To be fair, the paper should compare them under the same settings, that is, multiple guests and host. 2) Important baselines such as Secureboost[1] is missing.  3) the model performance of the proposed methods still appear to be 
a little inferior to the centralized setting, not exactly "comparable" as claimed. It is important to understand whether the proposed method is "lossless" or "lossy" and why. I think more detailed examinations and explanations are needed here. 

[1] Cheng, Tao Fan, Yilun Jin, Yang Liu, Tianjian Chen, Dimitrios Papadopoulos, and Qiang Yang. Secureboost: A lossless federated learning framework, 2019

### Questions
1. Does a "meta-rule" have to include the last layer of a tree? Fig 1& 2 both show that they include the last layer. However, Definition 1 shows it has additional layers F_k, which seems to be inconsistent. 

2. How does the algorithm deal with trees that have mixed splits from both host party and guest party? Figure 3 only demonstrates a simple example that the guest splits appear in the bottom layers, but what if guest splits appear near the root, followed by intervened and alternative host and guest splits? More complicated examples will help to demonstrate how the framework works.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper presents HybridTree, an innovative algorithm tailored for federated learning within the framework of Gradient Boosting Decision Trees (GBDT) in a hybrid data context. It introduces a novel tree transformation technique that can reorder split features based on insights from meta-rules. This transformation is key to the development of the hybrid tree learning algorithm, which is capable of incorporating knowledge from different sources (guests) by appending new layers to the decision trees.

### Strengths
1. The main innovation of this work lies in the development of a tree transformation strategy that can reorder split features to accommodate a federated learning environment. This is particularly relevant for scenarios where data privacy and distribution are concerns.

2. By introducing a new layer-level training algorithm, HybridTree, they address the integration of knowledge from multiple participants (referred to as "guests") in the federated model without compromising on data privacy.

3. The authors conduct extensive experiments on multiple datasets. The empirical results show that the proposed method outperforms the baselines significantly in model performance.

### Weaknesses
Even though the proposed method can handle hybrid features, the features have to be tabular data. This might not be the constraint of this paper but rather the limitation of the tree-based methods. However, maybe the authors can consider the scenarios where clients have multi-modal data, where the data modalities are hybrid across clients.

### Questions
Please see the weakness.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
