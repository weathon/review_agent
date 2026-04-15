# Bridging the Domain Gap by Clustering-based Image-Text Graph Matching

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 5, 5

## Abstract
Learning domain-invariant representations is important to train a model that can generalize well to unseen target task domains. Text descriptions inherently contain semantic structures of concepts, and such auxiliary semantic cues can be used as effective pivot embedding for domain generalization problems. Here, we want to use (image-text) multimodal graph representations to get domain-invariant pivot embeddings by considering the inherent semantic structure between local images and text descriptors. Specifically, we aim to learn domain invariant features by (i) representing the image and text descriptions with graphs, and by (ii) clustering and matching the graph-based image node features into textual graphs simultaneously. We experiment with large-scale public datasets, such as CUB-DG and DomainBed, and our model achieves matched or better state-of-the-art performance on these datasets. Our code will be publicly available upon publication.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposes a multimodal graph representation with textual semantic cues to obtain domain-invariant pivot embeddings for the domain generalization problem. The authors utilize graph neural network method to separately represent image and text descriptions, and cluster and match graph node features to learn domain-invariant features. The experiments verify the effectiveness of the proposed method.

### Strengths
1)	This paper proposes a simple and effective image-text graph matching method to solve the domain generalization problem.

2)	Experimental results show that the proposed method achieves state-of-the-art performance on domain generalization tasks

### Weaknesses
1)	The novelty of the proposed method is limited. Compared with GVRT [1], the proposed method only changes the manner of image-text alignment (adding graph), and thus provides incremental novelty. Moreover, graph-based image-text matching has been broadly studied [2-4], and the proposed technique contributes very poorly.

[1] Min S, Park N, Kim S, et al. Grounding visual representations with texts for domain generalization. In ECCV, 2022
[2] Liu C, Mao Z, Zhang T, et al. Graph structured network for image-text matching. In CVPR, 2020
[3] Wang S, Wang R, Yao Z, et al. Cross-modal scene graph matching for relationship-aware image-text retrieval. In WACV, 2020
[4] Li Y, Zhang D, Mu Y. Visual-semantic matching by exploring high-order attention and distraction. In CVPR, 2020

2)	The construction mechanism of the two graphs (visual and textual graphs) is unclear, especially the connection relations between graph edges. 

3)	Why graph matching contribute to learning domain-invariant features? The authors should provide detailed explanations of the theoretical basis of the proposed framework and provide more insights.

4)	There are many hyper-parameters (more than six), which need to be reported empirically via curve charts.

5)	The comparison of methods is not particularly sufficient. The latest comparative literature is from 2022, and there is a lack of comparison of related work as recently as 2023.

### Questions
Please see the weakness.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper exploits the multimodal graph representations to generate domain-invariant pivot embeddings for domain generalization. They use the graph to represent text descriptions and employ the graph matching scheme to align the embeddings of images and text. Some experiments validate the performance of the proposed method by comparing it to others.

### Strengths
The authors build the domain generalization problem on image datasets with a graph neural network. The overall structure of this paper includes a text graph encoder, an image graph encoder, and clustering-based graph alignment. Experimental results and visualizations on CUB-DG and DomainBed show the effectiveness of the proposed method.

### Weaknesses
The visual and text graph construction and the clustering-based alignment have been well and intensively studied in the current cross-modal learning, although the authors claim this is the first work for domain generalization problems.

The authors claim they suggest a novel graph neural network, but the overall graph learning is simply built on the conventional GCN or simple modification.

To me, this paper is more like a cross-modal learning or multi-modal learning algorithm rather than a domain generalization approach. The whole learning scheme mainly focuses on single model graph construction and local and global graph feature alignment. 

If I vote for a cross-modal learning task, I'd like to offer a borderline acceptance, but for a new task like domain generalization, the reviewer does not see its potential for new practical tasks.

### Questions
What are the differences between the proposed cross-modal or multi-modal feature alignment? What are the relations between the proposed method to your final task? How could we conduct the performance evaluation?

For the global graph alignment, this is a simple metric or numerical alignment? How could we take the adversarial or another distribution alignment into account? How about the performance or novelty?

For the local alignment, we can simply take it as a cluster-level or category-level alignment? Actually, the effects will greatly overlap with the global ones. How could you figure out the local and global ones? How could we avoid the redundancy in model construction?

There are too many hyper-parameters, which are hard to deploy in practical systems, let alone special domain generalization problems.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a graph-based domain generalization method to encode domain-invariant visual representations. It leverages text information and aligns the image and text from the global and local levels. Experiments on CUB-DG and PACS datasets show the effectiveness of the proposed approach.

### Strengths
1. This paper is well-organized and well-written.

### Weaknesses
1. The novelty of the proposed approach is limited. Utilizing graph models to represent the visual and text information and perform graph matching is normal in image-text retrieval task. It is unclear what is the novelty.
2. The proposed method introduces a novel modality(text information) to align different domains, which is unfair compared with those methods that only use image information. Moreover, it does not consider the relations among different domains.
3. The proposed method is unclear. Does it use the same backbone to extract features of all domains? The formulation only reflects the alignment of image and text modality, without showing different domains.
4. The authors do not compare with the art method, since the latest method in Table 1 is in 2022. 
5. There is little improvement in the performance. In Table 5, the performance of using image and text information is similar to that of only using image information. Moreover, why the compared methods on the two datasets are different in Table 1? The improvements in PACS are also limited.

### Questions
see the weakness

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
