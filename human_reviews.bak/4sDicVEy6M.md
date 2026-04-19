# What Do You See in Common? Learning Hierarchical Prototypes over Tree-of-Life to Discover Evolutionary Traits

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6, 6

## Abstract
A grand challenge in biology is to discover evolutionary traits---features of organisms common to a group of species with a shared ancestor in the tree of life (also referred to as phylogenetic tree). With the growing availability of image repositories in biology, there is a tremendous opportunity to discover evolutionary traits directly from images in the form of a hierarchy of prototypes. However, current prototype-based methods are mostly designed to operate over a flat structure of classes and face several challenges in discovering hierarchical prototypes, including the issue of learning over-specific prototypes at internal nodes. To overcome these challenges, we introduce the framework of Hierarchy aligned Commonality through Prototypical Networks (HComP-Net). The key novelties in HComP-Net include a novel over-specificity loss to avoid learning over-specific prototypes, a novel discriminative loss to ensure prototypes at an internal node are absent in the contrasting set of species with different ancestry, and a novel masking module to allow for the exclusion of over-specific prototypes at higher levels of the tree without hampering classification performance.  We empirically show that HComP-Net learns prototypes that are accurate, semantically consistent, and generalizable to unseen species in comparison to baselines. Our code is publicly accessible at Imageomics Institute Github site: https://github.com/Imageomics/HComPNet.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper addresses a scientific problem: discovering evolutionary traits in biology from images. To tackle this, it proposes the Hierarchy-Aligned Commonality through Prototypical Network (HComP-Net). The primary objective is to reduce over-specific prototypes that lose common features expected to be observed in all species sharing a common ancestor. To achieve this, the approach applies contrastive, orthogonality, and discriminative losses to the prototypes, and introduces over-specificity loss and masking to mitigate over-specific prototypes.

### Strengths
The method details are well-demonstrated, with a clear overview and examples that make the content easy to follow.

The study presents a novel approach to improving the interpretability of prototypical networks, allowing for more accurate representation of parent-child relationships.

It enhances the specialization of deep networks applied in scientific discovery, particularly in phylogenetic analysis.

### Weaknesses
Two limits of experimental demonstration. 
1. The primary contribution lies in incorporating parent-child relationships to reduce over-specific prototypes within the contrastive losses, while the architecture does not appear to include specific structures for establishing a prototype hierarchy.
In my understanding, proposing a hierarchical structure is a contribution of HPNet, not HCompPNet. This point should be clarified in title and method description. 
2. The exclusion of certain related works leaves me a question about novelty on practical impact. For instance, PIPNet, a recent and closely related method employing self-supervised learning (2023), is not included in the comparison. It only uses HPNet from 2019. However, semantic gaps has relevance to over-specificity, and authors also mention strong motivation from PIPNet. The reason why the network is excluded needs more detailed explanation or comparison in experiments.

### Questions
Why does the over-specificity loss adopt a specific log-tanh loss form? Doesn’t it simply establish an arbitrary criterion for identifying overly specific prototypes? This choice requires further explanation and discussion.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work presents an extended approach for hierarchical prototype learning by training a neural network to discover hierarchical prototypes using structured information from a phylogenetic tree. Building upon HPNet (Haze et al., 2019), this approach, termed HComP-Net, contrasts with traditional models that use flat structures by incorporating loss functions such as over-specificity loss and discriminative loss. These functions enable HComP-Net to learn prototypes that align with the hierarchical structure of the tree, enhancing interpretability and consistency. Empirical results highlight HComP-Net’s ability to produce accurate, semantically coherent prototypes transferable to unseen species. Tested on a dataset of 190 bird species and additional organisms, the model performs better than baseline models. Additionally, HComP-Net has been shown to generate visual hypotheses on evolutionary traits, offering insight into traits across various levels of the phylogenetic tree. This research underscores the potential of interpretable representation learning using structured hierarchical prior knowledge.

### Strengths
Originality: The research builds on the problem initially formulated in HPNet (Haze et al., 2019), specifically addressing the challenge of over-specific prototypes. The authors introduced over-specificity and discriminative loss functions, enabling HComP-Net to learn prototypes that adhere to the hierarchical structure of a phylogenetic tree. This approach enhances interpretability and brings a fresh perspective to prototype learning.

Quality: Given the identified problem of over-specific prototypes, the authors demonstrate the efficacy of their methods across fine-grained classification tasks, also showing advancements in interpretability. The proposed model performs consistently well in these tasks, validating its quality and effectiveness.

Clarity: The paper is well-organized, with clear explanations of the background, literature, methodology, experimental setup, and results. This clarity enhances the readability and accessibility of the research, making its contributions understandable and well-supported.

Significance: This work highlights the potential of interpretable representation learning driven by structured hierarchical knowledge. By using a phylogenetic tree for guidance, the model provides insights into trait evolution and aligns closely with the biology-inspired data structure, marking a contribution at the intersection of AI and biology.

### Weaknesses
1) On the Need for a Separate Masking Module
The fact that an additional masking module is required suggests that the initial contributions, particularly the over-specificity and discriminative loss functions, may not fully address the issue of over-specific prototypes. This need points to a limitation in the proposed loss functions' effectiveness in preventing prototypes from becoming overly tailored to specific species. Ideally, a more robust solution would directly manage prototype specificity through the loss functions alone, reducing reliance on extra modules that could complicate the model and potentially impact interpretability or scalability.

2) On the Role of the Phylogenetic Tree in Classification
In section 5.1, the authors suggest that achieving high classification accuracy is not the primary goal. However, providing the model with additional information, like a phylogenetic tree during training and inference, could indeed aid classification performance. The tree may allow the model to leverage hierarchical relationships, which could enhance classification accuracy by using shared traits among related species. Thus, there seems to be a disconnect between the claim of not prioritizing classification accuracy and the model's design, which inherently includes information that could enhance it.

### Questions
1) The CNN backbone used was ConvNeXt-tiny architecture; why the fine-grained accuracy from this architecture is not included in Table 1? Will it be better/worse when compared to other methods on the Table?

2) In terms of semantically meaningful prototypes, could you discuss the possibility of obtaining similar findings using Explainable AI techniques?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces a new framework that can be used for learning biological prototypes within hierarchies while avoiding the learning
of over-specific features at internal nodes of the genetic tree. The authors perform tests with different datasets including mostly the pictures of birds, fishes and butterflies. The authors focus on the quantitative and qulitative evaluation.

### Strengths
- The idea presented in the paper is interesting and original. The idea is quite simple (which is a plus). I liked that the authors aimed to present a method that can help to facilitate the analysis of different species in biology. 
- The paper is well-written, and also presents a lot of nice and clean graphics and examples that make the content understandable.

### Weaknesses
- The motivation of the paper says that there are many image datasets in the biology, so ML can be used to provide some new visual suggestions for common traits in species belonging to a common group. Nevertheless, such suggestions seem to be more important for some newly discovered species (and not the ones that are already well-known), and for such species there is a possibility of not having so many images. This can decrease the practicality of the method. Statements such as “Furthermore, HComP-Net demonstrates a unique ability to generate novel hypotheses about evolutionary traits, showcasing its potential in advancing our understanding of evolution” are too bold in my opinion.
- Another issue with a possible practical use of the method is that the proposed solution to provide semantically meaningful information requires human annotation, which can be very subjective. The authors mention this limitation in the appendix, however they do not give any solution for a mitigation. 
- minor: some typos/grammatical mistakes can be found in the paper, e.g. “that are not just shared across all it descendant species but are also” -> “that are not just shared across all ITS descendant species but are also”

### Questions
- The paper discusses many similarities with ProtoPNet and refers to it often, e.g. “For HPnet, we used the same hyperparameter settings and training strategy as used by ProtoPNet for the CUB-200-2011 dataset”, “We follow the same training strategy as provided by ProtoPNet for the CUB-200-2011 dataset.”. Why ProtoPNet is not used in the comparative experiments (also other non-hierarchical models are used there)?
- How can the paper help to “advance our understanding of evolution”? E.g. There is a high chance that for the common species (whose pictures are available in large-scale datasets) the same features that were already used to make such a classification will be highlighted. Could the authors present a use case, in which the solution could be successfully used in practice to contribute to the understanding of evolution?
- How are the representative images (also could be perceived as prototypical images) chosen from the dataset to allow for proper explanations (some images can be taken e.g. from a non-typical angle and do not show the prototypical features well and make the explanations difficult)?
- Is it possible (and if yes, how) to use this solution in other hierarchical problems (not in the area of biology)?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The work a novel framework called Hierarchy aligned Commonality through Prototypical Networks (HComP-Net) aimed at discovering evolutionary traits among species by learning hierarchical prototypes over the tree of life. It addresses the challenges of existing prototype-based methods that often produce over-specific prototypes at internal nodes, which can hinder the identification of common traits shared by descendant species. HComP-Net employs a unique over-specificity loss, a discriminative loss to ensure prototypes are absent in contrasting species, and a masking module to maintain classification performance. Through empirical analysis on various datasets, including birds, fishes, turtles, and butterflies, the authors demonstrate that HComP-Net effectively learns accurate, semantically consistent, and generalizable prototypes.

### Strengths
This paper investigates a highly intriguing task: identifying visual features preserved during species evolution. I believe this task is inherently challenging due to the limited and often insufficient quality of training data, making it difficult to obtain stable, semantically interpretable visual features. In this work, the design of the loss function and the associated explanations are intuitive and easy to understand.

### Weaknesses
1. The first comment is related to the choice of using visual data to identify common features in species evolution. Given the scarcity of high-quality biological images, especially in the context of vast evolutionary networks, and the inherent issues of imbalance and interference in such images, I wonder if it might be more precise to analyze common traits directly from textual descriptions or anatomical data. Could you please elaborate on the rationale behind prioritizing visual data for this task?

2. The paper introduces several loss functions aimed at ensuring the diversity and effectiveness of the learned prototypes. I would be very interested to see ablation studies on these loss functions to better understand their individual impact.

3. In Figure 4, when comparing the part consistency between HComP-Net and HPNet, different bird images are used. I am curious to know if this choice is justified and, if so, what the reasoning behind it is. Would it not be more appropriate to use the same images for a clearer comparison?

### Questions
Please see the weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper applies deep learning techniques to uncover evolutionary traits in biological data. It leverages contrastive and orthogonality losses to facilitate hierarchical prototype learning. Additionally, the paper introduces over-specificity and discriminative losses to guide and constrain model training. The proposed method demonstrates improved performance over baseline methods across multiple benchmark datasets.

### Strengths
1. The motivation is interesting and well-justified. The hierarchical structure of biological data presents significant challenges for distinguishing species and identifying evolutionary traits.
2. The techniques employed in HComP-Net appear technically feasible. The use of contrastive loss for learning clustered features has proven effective in self-supervised learning, while orthogonality loss helps capture diverse features.
3. The visualization results are clear and impressive.
4. The paper is well-structured and easy to follow.

### Weaknesses
1. It seems that the over-specificity and discriminative losses play opposing roles in the direction of model optimization, which raises the question of whether these two losses might interact and lead to abnormal model convergence. It would be beneficial if the authors could provide some theoretical or experimental analysis on this issue.
2. This is a minor point, but the overall method appears to be a combination of multiple techniques, making the flowchart somewhat complex and redundant.

### Questions
Please kindly refer the the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
3
