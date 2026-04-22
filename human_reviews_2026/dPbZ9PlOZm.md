# In-Context Clustering with Large Language Models

- Avg Score: 2.50
- Decision: Reject
- Scores: 2, 2, 4, 2

## Abstract
We propose In-Context Clustering (ICC), a flexible LLM-based procedure for clustering data from diverse distributions. Unlike traditional clustering algorithms constrained by predefined similarity measures, ICC flexibly captures complex relationships among inputs through an attention mechanism. We show that pretrained LLMs exhibit impressive zero-shot clustering capabilities on text-encoded numeric data, with attention matrices showing salient cluster patterns. Spectral clustering using attention matrices offers surprisingly competitive performance. We further enhance the clustering capabilities of LLMs on numeric and image data through fine-tuning using the Next Token Prediction (NTP) loss. Moreover, the flexibility of LLM prompting enables text-conditioned image clustering, a capability that classical clustering methods lack. Our work extends in-context learning to an unsupervised setting, showcasing the effectiveness and flexibility of LLMs for clustering.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This manuscript proposes In-Context Clustering (ICC), an LLM-based method that performs clustering without predefined similarity functions. ICC uses the attention mechanism of pretrained LLMs to capture context-dependent relationships among inputs across modalities. The authors show that LLMs or multimodal LLMs exhibit zero-shot clustering ability in numercal and visual data. With additional fine-tuning using LoRA with a Next Token Prediction loss, the experiments showed that ICC achieved improved performance on both numeric and image datasets. Moreover, ICC supports text-conditioned image clustering that allows prompt-based control on the clustering process.

### Strengths
Major Strengths:
- The writing and organization of this manuscript is clear and easy to follow (yet, it's better to add necessary details to make the main paper self-contained; jumping to figures in appendix from main paper is not very enjoable)

- The experiment that visualizes the attention allocation of input data and generated cluster labels at an intermediate layer is very interesting and innovative. This will give a good support to the clustering mechanism behind LLMs when in-context clustering is used.

- The presentation and visualization of this manuscript is clear and visually enjoyable.

### Weaknesses
Major Weakness:

- **My primary concern about this manuscript lies in the validity of its central claim** that “in-context clustering with large language models (LLMs) performs as well as or better than” traditional clustering methods such as K-means, spectral clustering, or DBSCAN, or other related methods. The rationale is as follows: clustering algorithms are designed to handle and explore unlabeled, unseen, and novel data—such as new concepts, observations, or protein structures—across diverse modalities. In contrast, the proposed “in-context clustering with LLMs” method fundamentally depends on pre-trained LLMs or multimodal LLMs and, consequently, on the massive datasets and implicit clustering criteria, and of course, clustering centroids, these models have already encountered during training. Therefore, it is unclear how well the proposed method would perform when the data is genuinely novel, unseen or out-of-distribution. This data coverage limitation also raises concerns about the validity of the claimed “zero-shot” setting in the experiments.  In contrast, traditional clustering methods such as kmeans can be easily adapted to truly novel and unseen data.

- **The reviewer found the following statement to be an overclaim:** At Line 73, the authors state: “We believe that this ability to change the way clustering is done based on different prompts makes ICC, and this research direction, particularly compelling.”  **In fact, text-conditioned or prompt-steered clustering using LLMs paradigm was first proposed by IC|TC [1], and subsequently explored in [2,3,4]. Moreover, [5] further extended this line of work by enabling automatic discovery of clustering conditions from data using LLMs.** So, such innovation and capability has already been proposed and studied by the community recently. **The authors should appropriately acknowledge prior research contributions rather than implying that this innovation originates solely from their proposed ICC method.**

- Several highly relevant studies, including [3, 4, 5], are missing from the literature review and discussion.  

- Compared to IC|TC [1] and [2], the novelty of the proposed ICC method is quite limited, as it mainly adds an additional fine-tuning component.

- **Regarding the “Zero-shot In-context Clustering” experiments in Section 3.1: are they truly zero-shot? The prompt template (Lines 144–146) explicitly provides the number of clusters to the model.** This information constitutes a strong prior about the data structure, meaning the model already **knows how many ground-truth groups exist in the dataset**. With such prior knowledge given, the zero-shot nature of the setting is questionable. In real-world zero-shot scenarios, the number of clusters is often *unknown*.  

- **The experimental setup and baseline comparisons in Section 4.2 (Table 2) and Section 5 (Table 3) appear to be unfair.** The IC|TC baseline is training-free and uses LLMs directly without fine-tuning (e.g., GPT-3.5-turbo). In contrast, the proposed ICC method  either (1) uses GPT-4o, which is a much stronger model, or (2) llava-interleave-qwen-7b-hf includes further fine-tuning on data drawn from a similar distribution. Comparing the GPT-4o model and fine-tuned llava-interleave-qwen-7b-hf to a training-free GPT-3.5-turbo baseline is not fair, as it conflates improvements due to model scale and additional tuning. Consequently, the conclusions drawn from this comparison are not well supported.  

- Further, the reviewer questions how ICC would compare against traditional clustering methods using strong vision features. **For example, what is the clustering performance of K-means when using features extracted from DINOv3-ViT-7B/16?** Would ICC—relying on a significantly larger model—still outperform DINOv3-based clustering under comparable settings?  


[1] Kwon, Sehyun, et al. "Image clustering conditioned on text criteria." In ICLR, 2024.

[2] Luo, Yulin, et al. "Llm as dataset analyst: Subpopulation structure discovery with large language model." In ECCV, 2024.

[3] Yao, Jiawei, Qi Qian, and Juhua Hu. "Customized multiple clustering via multi-modal subspace proxy learning." In NeurIPS, 2024.

[4] Yao, Jiawei, Qi Qian, and Juhua Hu. "Multi-modal proxy learning towards personalized visual multiple clustering." In CVPR, 2024.

[5] Liu, Mingxuan, et al. "Organizing unstructured image collections using natural language." Arxiv preprint, 2024.

### Questions
Minor questions are described in the following:
- The authors claim that prior similarity-based clustering methods cannot capture “context.” However, no proof, reference, or experimental evidence is provided to support this claim in either the textual or visual modality. In fact, many earlier methods in text clustering, including classical probabilistic models such as LDA [6], explicitly aim to model contextual information to group documents by topic. Similarly, in the vision domain, when images are represented through learned or encoded features, it is unclear to the reviewer why such representations would be inherently incapable of capturing context.

- Regarding the evaluation metrics in Section 3.1: while it is standard practice to compute clustering accuracy using the Hungarian linear assignment, this metric can be easily biased due to its matching paradigm. Since LLMs are capable of producing textual labels for each cluster, the authors could consider an alternative approach that approximates classification accuracy for a more direct comparison.  

- At Line 189, the authors state: “We also observe that instruction tuning improves the overall accuracy.” However, the dataset used for instruction tuning, and the details of how this tuning was performed, are not specified in the paper. Without this information, the result cannot be properly interpreted or reproduced.  

- Please explain what is “df” (degree of freedom) in Section 3.1. It is not explained in the manuscript.

[6] Blei, David M., Andrew Y. Ng, and Michael I. Jordan. "Latent dirichlet allocation." Journal of machine Learning research 3.Jan (2003): 993-1022.

### Soundness
1

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces In-Context Clustering (ICC), a novel method that leverages LLMs for clustering of images and numerical data. The authors show that the LLM’s attention mechanism captures complex relationships between inputs that can be used for clustering with spectral clustering. Further improvements are obtained through fine-tuning with next-token prediction, extending the method to numeric and image data. Additionally, text-conditioned image clustering is demonstrated where multiple different clusterings can be extracted based on the design of the prompt.

### Strengths
- The paper addresses a highly relevant research area, presenting an approach that enables more user-guided clustering through prompt-based interactions with LLMs
- I found it interesting that the method works well with numerical data as input for clustering via LLMs, although this capability appears to be constrained to lower-dimensional data
- Employing the attention matrix derived from the LLM as input for spectral clustering is an interesting insight
- Experiments show benefits across different datasets and modalities

### Weaknesses
## Soundness 

The authors limit their comparison to a single classical clustering algorithm, namely K-Means, which serves already as a strong baseline in Table 2 and 3. Based on this I am missing the comparison to different classical algorithms like Expectation-Maximization Clustering, DBSCAN or its popular extension HDBSCAN to see if "simpler" baselines can outperform the proposed method.
 

## Novelty

My main concern with this paper lies in its limited novelty. Several recent works have already explored closely related ideas, particularly the use of prompting and multimodal representations to induce or control clustering behavior, but the authors only compare to IC|TC. For example, prior studies such as

- **Jiawei et al. "Multi-modal proxy learning towards personalized visual multiple clustering." CVPR 2024.**

- **Jiawei et al. "Customized multiple clustering via multi-modal subspace proxy learning." NeurIPS (2024): 82705-82725.**

- **Stephan et al. Text-Guided Image Clustering. EACL (1) 2024: 2960-2976**

- **Stephan et al (2024). Text-Guided Alternative Image Clustering. In Proceedings of the 9th Workshop on Representation Learning for NLP (RepL4NLP-2024) (pp. 177-190).**

already use prompt-based approaches to obtain one or multiple clusterings conditioned on different attributes or textual guidance.  

Moreover, recent works such as 

- **Gadetsky et al: Large (Vision) Language Models are Unsupervised In-Context Learners. ICLR 2025**

- **Gadetsky et al: Let Go of Your Labels with Unsupervised Transfer. ICML 2024**

demonstrate already that large (vision) language models can perform unsupervised in-context learning and clustering without explicit supervision.  

Taken together, these prior works already explore the use of LLMs and in-context mechanisms for unsupervised or text-guided clustering, which significantly overlaps with the proposed In-Context Clustering (ICC) framework. The authors should therefore clearly differentiate their method from these existing methods and compare to them in benchmarking experiments. Further, a dedicated discussion of what is conceptually and technically novel about ICC compared to these earlier contributions is missing.

### Questions
- In what key methodological ways does your approach differ from the prior works referenced in the weaknesses section? What are the key contributions of your method?
- The current comparison is limited to k-Means and IC|TC. How does your algorithm perform relative to other recently proposed methods mentioned above?
- How do other classical clustering algorithms compare to your approach? Are there scenarios in which simpler baselines outperform ICC, and if so, under what circumstances? More broadly, when might traditional methods be sufficient compared to LLM-guided clustering?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The article introduces In-Context Clustering (ICC), which extends the in-context learning paradigm to unsupervised clustering tasks. The authors demonstrate that large language models (LLMs) can perform zero-shot clustering on text-encoded numeric data and images by leveraging their attention mechanisms to capture complex relationships between inputs. The work introduces fine-tuning strategies using next token prediction (NTP) loss to enhance clustering performance, particularly for heavy-tailed distributions and semantically rich data. Additionally, ICC enables text-conditioned image clustering, allowing users to specify clustering criteria through natural language prompts.

### Strengths
- The paper is clear about its motivation with sufficient significance and quality.
- The paper makes a compelling case for extending in-context learning to unsupervised settings. The ability to perform clustering without predefined similarity measures through prompting is innovative and addresses real limitations of classical methods.
- The zero-shot clustering results on t-distributed data with varying degrees of freedom convincingly demonstrate that LLMs can outperform k-means when Gaussian assumptions are violated. The performance gains are particularly striking for heavy-tailed distributions.
- The visualization and analysis of attention matrices revealing emergent cluster structures (Section 3.2) provides valuable insights into the internal mechanisms. The finding that spectral clustering on attention matrices achieves 85% accuracy before fine-tuning while generation only reaches 74% is particularly intriguing.

### Weaknesses
- The paper insufficiently addresses the computational limitations for practical deployment. With O(n²) attention complexity and token limits, how does ICC handle datasets with thousands of points? The average pooling for images seems like a band-aid solution that could lose critical fine-grained information.
- While the empirical results are good, the paper lacks theoretical analysis of when and why ICC works. What properties of the attention mechanism enable clustering? Under what conditions might ICC fail? 
- This is the most critical weakness of the paper. For image clustering, the comparison is limited to k-means and IC|TC. Missing comparisons with modern deep clustering methods (e.g., SCAN, NNM, SwAV, or other self-supervised approaches) makes it difficult to assess the true performance gains.
- The fine-tuning data generation process using t-distributions with random parameters seems arbitrary. How sensitive is performance to this choice?
- No ablation studies on key design choices (e.g., impact of different pooling strategies, prompt variations)
- Figure quality could be improved - some attention visualizations are difficult to interpret
- The related work section could better position this work relative to recent advances in foundation models for clustering

### Questions
- How does performance degrade as the number of data points approaches context limits? Have you experimented with hierarchical clustering or other strategies to handle larger datasets?
- How robust is ICC to prompt variations? The paper uses a simple template "Cluster the following data into {k} clusters" - have you tested more sophisticated prompting strategies or chain-of-thought reasoning?
- Can you provide any theoretical justification for why attention patterns correspond to cluster structure? Is there a connection to graph-based clustering methods or spectral theory?
- What types of clustering problems does ICC struggle with? For instance, how does it handle clusters with varying densities, non-convex shapes, or hierarchical structures?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes and tests using LLMs for “in-context clustering”, where (multimodal) LLMs are presented sequences and are tasked with assigning cluster labels (conditioned on a priori cluster count) to each element. The experiments include (1) synthetic numerical clustering, where points sampled from mixtures of low-dimensional t-distributions with varying degrees of freedom; (2) attention-based analysis, where token-level attention maps are treated as affinity matrices for spectral clustering; (3) LoRA fine-tuning, where the model is trained via next-token prediction on synthetic text prompts containing sample–label pairs; and (4) image experiments, where images and captions from ImageNet are clustered through textual prompts. All experiments report Hungarian-aligned accuracy against true labels and compare only to simple baselines like k-means.

### Strengths
- The experiments are reproducible and clearly presented
- Analysis of attention affinity matrices is interesting

### Weaknesses
- Performance gains are unsurprising: the fine-tuned models are trained on synthetic mixtures drawn from the same/very similar distribution family as the evaluation sets, so improvement simply reflects distributional overlap, not generalizable clustering ability. 
- "Classical methods often rely on predefined measures" - I don't agree with this. For example, embedding models trained with contrastive learning transform data onto a low dimensional manifold where local distance meaningfully represents semantic difference. 
- Minimal novelty. The pipeline and evaluation duplicate prior IC|TC work, with only superficial framing changes (“in-context” language).
- Accuracy via Hungarian matching inflates scores and hides near-chance performance. Consider using ARI/NMI as well.

### Questions
- Does attention-spectral remain strong under permutation of item order and different prompt formats? Show stability across layers/heads with an automatic selection rule.
- How do CLIP/DINOv2 features + spectral/DBSCAN/GMM compare under the same data, including conditional setups via text features?

### Soundness
1

### Presentation
2

### Contribution
1
