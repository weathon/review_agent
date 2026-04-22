# Weight Space Representation Learning on Diverse NeRF Architectures

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6

## Abstract
Neural Radiance Fields (NeRFs) have emerged as a groundbreaking paradigm for representing 3D objects and scenes by encoding shape and appearance information into the weights of a neural network. Recent studies have demonstrated that these weights can be used as input for frameworks designed to address deep learning tasks; however, such frameworks require NeRFs to adhere to a specific, predefined architecture. In this paper, we introduce the first framework capable of processing NeRFs with diverse architectures and performing inference on architectures unseen at training time. We achieve this by training a Graph Meta-Network within an unsupervised representation learning framework, and show that a contrastive objective is conducive to obtaining an architecture-agnostic latent space. In experiments conducted across 13 NeRF architectures belonging to three families (MLPs, tri-planes, and, for the first time, hash tables), our approach demonstrates robust performance in classification, retrieval, and language tasks involving multiple architectures, even unseen at training time, while also matching or exceeding the results of existing frameworks limited to single architectures. Our code and data are available at [this https URL](https://cvlab-unibo.github.io/gmnerf).

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work addresses the challenge of learning on Neural Radiance Fields (NeRF), treating NeRF as a data format. The key question is: how can we effectively conduct learning across various NeRF formats? Given the existence of multiple NeRF formats, the learning framework varies accordingly. To accommodate these different formats—such as MLP, Triplane, and Hashgrid—this approach first converts them into a graph representation. Then, a Graph Meta-Network, equipped with an unsupervised representation learning framework, is implemented. Results indicate that employing a contrastive objective significantly enhances generalization abilities for unseen NeRF formats, as demonstrated through classification and retrieval tasks.

### Strengths
1. The research question is likely to attract interest from communities that utilize NeRF as a data format. The proposed technique enables direct retrieval and classification within this format.
   
2. The graph conversion method is novel, making it applicable to a wide range of NeRF data formats.

3. Performance results are impressive. In single architecture tasks, the method surpasses previous techniques. In multiple architecture tasks, the addition of the contrastive loss notably boosts accuracy.

### Weaknesses
1. While the problem is intriguing, the paper does not adequately highlight the practical uses and advantages of learning directly on various NeRF formats. In many scenarios, a single NeRF format should be sufficient for each platform. Furthermore, when it comes to classification and retrieval, the question arises: why not simply store a few images (e.g., three orthogonal views) for each NeRF and perform learning on those images? The paper mentions that rendering takes time, but this argument feels weak. Although using the complete NeRF may benefit certain dense prediction tasks, the paper does not demonstrate such complex tasks.

2. The experimental tasks performed undermines the claims made. Although the input NeRF formats are unified, the encoder architecture is limited to a Graph Network with global pooling to derive the embedding. It remains unclear how such an adaptation could extend to more complex tasks like 3D generation or 3D segmentation.

### Questions
Q1. Can the learned embedding be used to reconstruct the NeRF? What is the Peak Signal-to-Noise Ratio (PSNR) for the learned embedding when using the decoder?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a method for encoding different NeRF architectures to embeddings for the purpose of classification and retrieval.

In terms of architectures, MLP, triplane, and hash table-based NeRF architectures are being used as inputs for encoding.
In terms of loss, the rendering and the contrastive loss are being used.

For the encoding and decoding, a GMN-like and a nf2vec-like encoder and decoder are being used, respectively.

ShapenetRenderer and Objaverse datasets are being used for evaluation.

The authors noted the first time a hash table based spatial grid can be processed.

### Strengths
1. The paper is well written, the explanations are thorough, and the figures are clear. The equations (for the loss) are clearly written. And the figures demonstrating the graph representation (of MLP, triplane and hashtable), though do require background reading, but is understandable once with the proper context.
2. Supporting the MLP, triplane and hash table-based NeRFs cover most of the NeRF representation, if not all. This makes it a relatively general method for its own purpose.
3. Both the quantitative and qualitative evaluations of the method itself are thorough. The retrieval does demonstrate similar content is being retrieved.

### Weaknesses
1. All of the losses and the encoder/decoder (GMN/nf2vec) used in this work are existing components. Making the novelty in those domains minimal (though the author didn't claim particular novelty in that domain and faithfully discussed the original works).
2. The authors discussed its application to hash table. However, I have particular concern for this, both in terms of the novelty and soundness. Which I describe further in the question.
3. The results are mostly compared within the works' own variants with different losses. With minimal comparison with existing works, even for the MLP or triplane-based solutions.

### Questions
1. Does the author have additional discussion or ideas in losses/encoder/decoder?
2. hash table + GMN soundness: GMN is invariant/equivariant to standard neural network node permutation. Further, it respects the spatial content of a spatial grid (tri-plane like). Both these features are what the authors leveraged for the MLP and triplane-based NeRF in this current work. However, using a slightly different prime number for the hash function will lead to a completely different set of hash table features. I think this makes the hash table features inherently unsuitable for processing with the GMN. What's the authors' opinion on this? What features are stable, and what are unstable?
3. hash table + GMN novelty: If I didn't miss a particular major point, this component is a major one in the pipeline that didn't directly reuse existing works' components. However, I think the hash table + GMN combination maybe inherently insufficient. Or at least warrant further analysis, improvement or modifications which the authors didn't mention. Does the authors have further points on why hash table can be combined with GMN this way, what components may require more careful analysis?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work proposes a new framework aiming at performing downstream tasks on NeRF weights with diverse architectures. The authors claim that the proposed model can handle multiple NeRF-related architectures, including multi-resolution hash tables, and validate the effectiveness of their proposed components.

### Strengths
1.	This paper is well-written.
2.	It focuses on a commonly overlooked problem regarding downstream tasks on trained NeRFs.
3.	The proposed framework can handle various NeRF-related models, especially multi-resolution hash tables.

### Weaknesses
1.	The paper needs to provide a sufficient description of experimental details, such as training and testing datasets, as well as benchmarks.
2.	The paper claims to be able to classify different NeRF architectures, but all experiments focus on object-centric NeRFs, lacking discussion of scene-based models.
3.	Due to incomplete experimental details, it is difficult to evaluate the effectiveness and generalizability of the proposed method.
4.	The experimental section lacks discussion of some hyperparameters. Additionally, it lacks analysis of the experimental results, especially regarding the performance differences of the proposed method across different NeRF variants.

### Questions
How many different categories can the proposed model handle?

### Soundness
2

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
3

### Summary
This paper proposes an architecture-agnostic framework for weight-space representation learning on NeRFs. It encodes diverse NeRF architectures into a shared latent space using a Graph Meta-Network encoder plus an nf2vec-style decoder, trained with a rendering loss and a contrastive loss to align cross-architecture embeddings. Experiments over 13 architectures show strong performance on classification and instance-level retrieval, including unseen-architecture and cross-dataset scenarios.

### Strengths
- The paper tackles an underexplored problem: processing NeRFs directly in weight space across heterogeneous architectures. Prior art (nf2vec; tri-plane-specific encoders) is bound to a single architecture; this work demonstrates a unified encoder that also handles hash-table NeRFs

- The study spans 13 variants across three families, with both single- and multi-architecture training, and evaluates seen vs. unseen configurations. Results include detailed classification tables and retrieval

- A unified weight-space interface for NeRFs could enable future foundation-style models for NeRF manipulation, indexing, and understanding without rendering, saving compute and simplifying pipelines.

### Weaknesses
- Most training/evaluation is on synthetic datasets (ShapeNetRender), with generalization probed only via limited Objaverse tests. It remains unclear how robust the encoder is to real-world NeRFs (capture noise, exposure variation, backgrounds) or to tasks where material/lighting realism matters. A larger cross-dataset study (e.g., forward-train on Objaverse; test on different captured NeRF sets) would strengthen claims. 

- The paper focuses on classification and retrieval. These are important, but practical NeRF workflows also require detection/segmentation in NeRF space, quality assessment, editability guidance, or content moderation. Demonstrating that the embeddings transfer to such tasks (even with light adapters) would better justify the "general weight-space representation" claim.

### Questions
Under explicit weight permutations (e.g., permuting hidden units or hash entries), does the embedding remain stable? Any empirical test or theoretical discussion?

What are typical graph sizes and embedding latencies for MLP/TRI/HASH, and how do they scale with resolution, channels, and table size?

### Soundness
3

### Presentation
3

### Contribution
3
