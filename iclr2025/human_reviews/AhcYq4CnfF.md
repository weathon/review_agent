## Human Reviewer 1

### Summary
The paper presents a graph anomaly detection method, DiffGAD, which aims to capture critical discriminative content in reconstruction. By leveraging diffusion sampling, DiffGAD infuses the latent space with discriminative content. To evaluate its effectiveness, the authors conduct experiments on seven datasets and compare its performance against other anomaly detection methods.

### Strengths
1. This paper presents a clear motivation.
2. The logical structure of the paper is clear.

### Weaknesses
1. The paper states that ‘the latent space constructed by the AE-based method (Ding et al., 2019) tends to represent all samples for the Books dataset (Sánchez et al., 2013) into the same point’ and 'VAE constructs the latent space within a constrained distribution (e.g., the Gaussian distribution), leading to a uniform latent distribution.' This argument should be further supported through experimental validation.
2. Authors should add the latest baselines to demonstrate the effectiveness of DiffGAD.
3. Sec. 4.3 should further analyze why DIFFGAD does not achieve optimal performance.

### Questions
1. The GAE model detects anomalies by calculating reconstruction error. Wouldn’t it be more effective to focus on reconstructing common attributes rather than distinguishable features, as anomalies are typically identified by their deviations from common patterns?
2. Why add noise to the features? Can’t the original features capture the general patterns?
3. In the visualization results, normal nodes and abnormal nodes are not clearly distinguished.

### Soundness
3

### Presentation
2

### Contribution
2

### Rating
5

### Confidence
5

---

## Human Reviewer 2

### Summary
This paper proposes DiffGAD, an unsupervised graph anomaly detector based on diffusion models, designed to address the limited capability of capturing discriminative content in graph anomaly detection.

The main contributions include:

Pioneering the application of diffusion models in graph anomaly detection by adapting them from generative tasks, presenting DiffGAD to enhance the model's discriminative ability.
Proposing a generative paradigm guided by discriminative content to extract and refine discriminative features into the latent space, and designing a content preservation strategy to improve the reliability of the guidance process.
Demonstrating the effectiveness of DiffGAD through extensive experiments on 6 real-world datasets and 13 baseline methods.

### Strengths
1、The authors innovatively adapt diffusion models from generative tasks to the field of graph anomaly detection, introducing DiffGAD, which brings new perspectives to graph anomaly detection.

2、A novel latent space learning paradigm is introduced by combining unconditional and conditional diffusion models to capture and refine discriminative content, offering a fresh approach to address the challenge of extracting discriminative features in graph anomaly detection.

3、When addressing anomaly detection, the method incorporates diffusion sampling and a content preservation mechanism, effectively injecting and retaining discriminative content across different scales.

4、In the experimental section, six different real-world and large-scale datasets are used for evaluation, covering various types of data scenarios, such as social networks (Weibo, Reddit), commercial networks (Disney, Books, Enron), and large-scale financial networks (Dgraph), ensuring the reliability and generalizability of the experimental results.

### Weaknesses
1、Although the paper mentions that graph encoders (e.g., GCN in Graph AE) may limit the model's ability to represent complex graph structures and relationships, it only briefly notes this issue without analyzing how this limitation impacts DiffGAD’s performance in specific experiments or real-world scenarios. For example, when handling highly heterogeneous or dynamically changing graph structures, the encoder may fail to accurately capture critical information, potentially leading to reduced anomaly detection accuracy.

2、For the key hyperparameter λ in the model, simply showing the impact of different values on performance through experiments is insufficient. There is a lack of theoretical justification explaining why the optimal value differs across datasets and the intrinsic connection between these values and data characteristics.

3、In the introduction, the logical connection between the research motivation and the subsequent presentation of DiffGAD's innovations could be clarified. It would be helpful to explicitly highlight the specific limitations of traditional methods in capturing discriminative content and how DiffGAD’s unique design (such as the generative paradigm guided by discriminative content and content preservation strategy) directly addresses these issues.

### Questions
The paper mentions that graph encoders (such as GCN in Graph AE) might limit the model’s ability to represent complex graph structures and relationships, but it does not detail how this limitation manifests in practical applications or to what extent it impacts DiffGAD’s performance. For instance, when processing graphs with highly heterogeneous node attributes or complex topologies (e.g., multi-layer nested structures, frequently dynamic graphs), would the current encoder lead to critical information loss? How does this information loss impact the accurate extraction of discriminative content and, subsequently, anomaly detection accuracy? The authors could quantify this impact by designing targeted experiments. For example, they could construct synthetic graph datasets with varying levels of heterogeneity and topological complexity to compare DiffGAD’s performance when using the current encoder versus a hypothetical ideal encoder (with stronger representational power). Metrics such as anomaly detection accuracy and recall could provide insight into performance differences.

### Soundness
3

### Presentation
2

### Contribution
2

### Rating
6

### Confidence
3

---

## Human Reviewer 3

### Summary
This article presents DiffGAD, a new method based on Diffusion Models (DM), for dealing with the problem of anomaly detection in graph data. A research framework based on Diffusion Models is constructed, which encodes the graph data into the latent space via an encoder, then adds noise to preserve the general content and samples from unconditional and conditional diffusion models, and finally transforms the reconstructed embedding back into the graph space via a decoder to compute the reconstruction error. Finally, through experiments on multiple real-world datasets, the authors conclude the effectiveness of DiffGAD on graph anomaly detection tasks.

### Strengths
1、The method combines graph neural networks (GNNs) with diffusion models, an innovative fusion of techniques that takes advantage of both the strengths of GNNs in graph structural analysis and the sophistication of diffusion models in generative modeling.
2、 DiffGAD uses both conditional and unconditional diffusion models to reconstruct the graph, and this combination improves the model's sensitivity and ability to recognize anomalies.

### Weaknesses
1、the experimental part of the dataset settings, the dataset selected in this paper are relatively small feature dimensions, have you considered the use of high-dimensional features of the dataset for the experiment?
2、Comparison methods on although there are based on deep learning, but the latest is 2022, it is recommended to increase the comparison experiments, and the current advanced methods in the field, to show the advantages of the project method.

### Questions
1、How do the two DMs achieve concurrent sampling without adding extra parameters?
2. In 3.3, the minimum perturbation is used to change the potential embedding z0 to zt. How is this “minimum perturbation” defined and quantified?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
3

---

## Human Reviewer 4

### Summary
The paper introduces DiffGAD, a diffusion based model designed for graph anomaly detection (GAD). It employs a latent space learning paradigm and incorporates discriminative content to enhance profiency. The authors present experimental results on six large real world datasets to demonstrate model performance. Code is made available as well.

### Strengths
1. The use of diffusion models for GAD is an innovative application
2. A generous number of meaningful datasets are used for experimentation
3. The figures used in the paper manage to illustrate the problem and approach well.

### Weaknesses
1. A specific graph autoencoder (AE) is used by the model. This may limit the adaptability of the model. (See Q1)
2. Hyperparameters like λ affect the performance of DiffGAD. If this turns in to an "art" to get the best performance, then it may be problematic for real world scenarios. Clarification from the authors about some guidelines to selecting λ could be helpful.

### Questions
1. Did the authors explore any alternatives to the currently used AE?
2. Follow-up: a less comlex alternative could give an efficiency boost at some performance cost. Would be interesting to see. (not required for acceptance)
3. Are there any particular type of datasets that are more "conducive" to DiffGAD than others?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
3

---

## Human Reviewer 5

### Summary
The paper proposes a Diffusion-based Graph Anomaly Detector for unsupervised identifying abnormal entities in networks. The contributions can be summarized as follows: 1: The author made the first attempt to transfer the diffusion models to the graph anomaly diffusion tasks. The model consists of an auto-encoder framework and two diffusion models to conduct unconditional diffusion and conditional diffusion. 2: To guide the training of diffusion models in latent space, the authors propose to use a discriminative content-guided generation paradigm to distill the discriminative content in latent space; and a content-preservation strategy to enhance the confidence of the guidance process. 3: The authors conduct experiments on 7 real-world datasets and make comparisons with 13 baseline methods.

### Strengths
- The first attempt to transfer the generative diffusion models to the GAD task is a great try, bringing new perspectives to the anomaly detection community.
- The idea of distilling the discriminative content based on a linear combination of the two different diffusion models is interesting and easy to follow.

### Weaknesses
- Some parts of the paper need further explanation. For example, in the introduction, the authors mention that some researchers utilize encoders to map graph data into a latent space, but there is a lack of essential discussion about why they are doing that.
- The motivation behind some of the model designs is ambiguous. For instance, in the selection of the encoder and decoder, the conditions that a good encoder and decoder should satisfy are unclear. Instead of directly utilizing the GAE framework, it seems more important to discuss the criteria for selecting a good encoder and decoder and add related experiments to support your claims. I did not see any discussions related to that.

### Questions
- Why did you choose to use GAE as the encoder and decoder instead of considering other models, such as VGAE, graph transformers, etc.? Is it possible to use these models as the encoder and decoder?
- What are the conditions that a good latent space should satisfy? Or, what constitutes a good latent space for performing the diffusion process to conduct anomaly detection tasks?
- Regarding the training process, it looks like you first train a satisfactory encoder and decoder, then fix the parameters, and train the diffusion models. Why did you not choose to construct a global loss and train the GAE and diffusion models together?

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
3