# Recursive Cleaning for Large-scale Protein Data via Multimodal Learning

- Decision: Reject
- Scores: 5, 6, 5, 5, 5

## Abstract
Reliable datasets and high-performance models work together to drive significant advancements in protein representation learning in the era of Artificial Intelligence. The size of protein models and datasets has grown exponentially in recent years. However, the quality of protein knowledge and model training has suffered from the lack of accurate and efficient data annotation and cleaning methods.
To address this challenge, we introduce ProtAC, which corrects large Protein datasets with a scalable Automatic Cleaning framework that leverages both sequence and functional information through multimodal learning. To fulfill data cleaning, we propose the Sequence-Annotation Matching (SAM) module in the model, which filters the functional annotations that are more suitable for the corresponding sequences. Our approach is a cyclic process consisting of three stages: first pretraining the model on a large noisy dataset, then finetuning the model on a small manually annotated dataset, and finally cleaning the noisy dataset using the finetuned model. Through multiple rounds of “train-finetune-clean” cycles, we observe progressive improvement in protein function prediction and sequence-
annotation matching. As a result, we achieve (1) a state-of-the-art (SOTA) model that outperforms competitors with fewer than 100M parameters, evaluated on multiple function-related downstream tasks, and (2) a cleaned UniRef50 dataset containing ∼50M proteins with well-annotated functions. Performing extensive biological analysis on a cleaned protein dataset, we demonstrate that our model is able to understand the relationships between different functional annotations in proteins and that proposed functional annotation revisions are reasonable.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper presents ProtAC, an innovative framework for recursive cleaning of large-scale protein datasets. ProtAC employs multimodal learning to enhance annotation accuracy by integrating sequence and functional data. The proposed Sequence-Annotation Matching (SAM) module refines annotations iteratively, using a pretrain-finetune-clean cycle to reduce noise and enhance dataset quality. This cycle leverages multimodal embeddings to identify and correct unreliable annotations in large datasets. The study claims state-of-the-art performance in protein function prediction across multiple benchmarks, achieving efficient and biologically meaningful annotations with models of relatively modest parameter sizes.

### Strengths
1. ProtAC introduces a recursive, multimodal cleaning framework for large-scale protein data, which combines sequence and functional annotation data in an iterative process. This recursive approach significantly improves data quality by progressively refining protein annotations, resulting in cleaner datasets and more accurate models.
2. By integrating sequence and functional data, ProtAC enhances annotation accuracy and functional prediction, addressing a common limitation in protein datasets. The multimodal Sequence-Annotation Matching (SAM) module offers a practical solution for aligning sequence data with functional information.
3. ProtAC achieves a state-of-the-art (SOTA) model with fewer than 100M parameters is achieved, outperforming competitors on multiple function-related downstream tasks. Additionally, it produces a cleaned version of the UniRef50 dataset, containing approximately 50 million accurately annotated protein sequences.
4. The study includes a detailed biological analysis of the cleaned dataset, showing that ProtAC’s modifications align well with established biological principles. This analysis underscores ProtAC’s potential for providing biologically meaningful insights, especially in the context of functional annotation.

### Weaknesses
While ProtAC introduces a promising strategy for improving dataset quality in protein bioinformatics, there are four weaknesses listed below that can be addressed to enhance its performance, generalizability, and applicability across diverse datasets and research objectives.

1. Despite emphasizing multimodal learning, ProtAC does not incorporate protein structure data, which is fundamental for accurate protein function prediction. By not incorporating structural data, the framework limits its understanding of functional nuances that are strongly influenced by three-dimensional protein conformations. This is critical for predicting protein functionality beyond simple sequence-based correlations, especially in cases where structural context dictates function (e.g., enzyme active sites). Integrating structural data would elevate the biological fidelity of predictions, improving ProtAC’s applicability in diverse biological and therapeutic contexts.
2. The pretrain-finetune-clean cycles require extensive computational resources and time, raising questions about the scalability of this approach to datasets beyond UniRef50，such as UniProt. Without scaling adaptations, ProtAC’s utility may be restricted to smaller databases, potentially underrepresenting protein diversity and thus limiting the reach of its cleaned annotations.
3. The approach heavily relies on a small, manually annotated dataset (SwissProt) for finetuning, which may limit generalization to diverse or rare protein functions. Relying on a manually curated SwissProt dataset introduces bias toward well-studied proteins, hindering generalization to underrepresented or novel sequences. This limitation affects ProtAC’s ability to accurately annotate functions in proteins from less-characterized species or environmental samples, reducing the utility of cleaned data in exploratory proteomics.
4. While competitive with larger models, ProtAC’s effectiveness may be further constrained by the limited model parameters (sub-100M), especially in highly complex functional predictions. The choice to use models under 100M parameters constrains ProtAC’s ability to capture complex biological interactions that may be better addressed by larger, more expressive models. Although it provides competitive performance on standard tasks, a parameter increase could unlock deeper representations that facilitate more accurate, granular annotations, particularly in multifunctional proteins.

### Questions
1. How do you plan to extend ProtAC to include structural data in future work?
2. Could you please discuss potential strategies for improving the efficiency of the proposed approach, and to provide estimates of the computational requirements for scaling to larger datasets like UniProt? How can the recursive cleaning process be optimized to balance effectiveness and computational cost?
3. Could you please explain how to address potential biases introduced by relying on SwissProt, and suggest ways to incorporate more diverse protein data in the finetuning process?
4. Could you please explain the rationale for choosing a parameter range of sub-100M? Whether you have explored larger models in future work?
5. In Table 1, OntoProtein shows substantial improvement over the original ProtBERT by leveraging knowledge graphs, effectively utilizing high-quality data. Could you elaborate on the unique advantages ProtAC offers in comparison to OntoProtein?
6. Table 2 indicates only a slight improvement in keyword prediction for the 8M-cleaned model compared to the 8M-uncleaned model. Could you provide results for the 35M-uncleaned model to offer a broader basis for comparison?
7. The ablation study results in Table 3 show inconsistent performance across the four tasks. Could you clarify whether these variations stem from the data-splitting method or the inherent differences between tasks?
8. In Figure S2, the clustering outcome for GO:0005576 on the cleaned dataset appears improved compared to the original, while the clustering for other terms shows limited improvement. Could you clarify why this effect is more evident for GO:0005576 and provide any additional insights or visualizations to highlight improvements for other terms?

### Soundness
2

### Presentation
2

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
This paper introduces a novel method called ProtAC, which applies a scalable automatic cleaning framework leverage both Protein sequence and functional information through multimodal learning to correct large protein datasets. Then, it conducts experiments on multiple function-related downstream tasks and publish a cleaned UniRef50 dataset containing ~50M proteins with well-annotated functions. Furthermore, it also presents experiment that demonstrate biological meaningful significance.

### Strengths
1. This methods propose the Sequence-Annotation Matching module and a cyclic process which consists of three stages：”train-finetune-clean” cycles. In this way, it may clean the large protein datasets and obtain an effective dataset, which provide a novel method for other researchers to clean the dataset they need.
2. In Table 1, some protein function prediction task performance shows that ProtAc+backbone surpassing models with less more than 100 million parameters.

### Weaknesses
Some technical details should be explained, please refer to Question part.

### Questions
1. In Table 1, the ProtAC-PB, does the pharse PB refer to ProtBert in your baseline？In my view, we find that the performance of ProtAC-PB is lower than ProtBert. Could you explain what caused this? And the baseline that you choose is not novel, you are supposed to compare with the latest baseline. Does your method have been tested on SaProt [1]? if not, you are supposed to test on SaProt as a way to strengthen the evaluation.
2. In Table 2, as you described in Section 4.2 “Furthermore, as illustrated in Tab 2 comparisons between the 8M-cleaned and PB-cleaned version reveal that, pretrained models consistently outperform their from-scratch counterparts”, whether different parameters size cause this phenomenon？, we find that lacking the ProtAC-ESM2-35M-cleaned models, why don’t you compare their performance？You could include results for the in ProtAC-ESM2-35M-cleaned models.
3. During the process of cleaning the large protein datasets, does this process cause additional noise to influence the quality of cleaned datasets. You are supposed to explain how to reduce additional noise, or quantify the quality/noise level of the cleaned datasets compared to the original.
4. In addition, authors should provide access to code and datasets so that other researchers can reproduce the results.

[1] Su J, Han C, Zhou Y, et al. SaProt: Protein Language Modeling with Structure-aware Vocabulary[C]//The Twelfth International Conference on Learning Representations.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper presents ProtAC, an automatic cleaning framework for large protein datasets using multimodal learning with sequence and functional information. The core SAM module ensures proper sequence-annotation matching. ProtAC follows a cyclic process of pretraining, finetuning, and cleaning, progressively improving protein function prediction.

### Strengths
The proposed multi-round training strategy allows the model to progressively clean noisy datasets by predicting and selecting credible protein functional information, leading to mutual improvement in both dataset quality and model performance.

### Weaknesses
One concern with this paper is the unclear source of the noisy data. Is the noise stemming from the functional labels (e.g., GO annotations) or from unreliable textual information? GO annotations are typically well-supported by literature, so the rationale for annotation errors is not sufficiently explained. Lines 88-99 lack citations to support the claim of noisy data, making the subsequent discussion difficult to follow.

Additionally, the paper's characterization of OntoProtein is inaccurate. OntoProtein leverages biomedical text by incorporating it into a knowledge graph, contrary to the authors' description that it does not utilize biomedical text-related information.

The paper should also address why its results do not outperform knowledge graph-based methods like OntoProtein. Furthermore, it would be beneficial to compare ProtAC with models such as ProtST, which also utilizes protein sequences and biomedical texts, and ProtT3, a protein-text generation method, both of which could provide meaningful points of comparison for representation learning.

### Questions
Nan

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper introduces a novel approach leveraging knowledge distillation techniques for cleaning large protein datasets, aimed at improving protein function prediction. The authors propose a multimodal learning model, ProtAC, that incorporates sequence and functional annotations to enhance dataset quality and model training through an iterative train-finetune-clean cycle.

### Strengths
1. Applying knowledge distillation to clean large protein datasets is a novel and promising approach.
2. The resulting pre-trained model shows improvement over the baseline model, ESM2, suggesting potential in the proposed method.

### Weaknesses
1. Importance "Performant PLM Baselines" like ProtST is missing. Given ProtST outperforms OntoProtein by a large margin, the performance in Tab. 1 remains to be justified. This omission also raises questions about the comparative effectiveness of the proposed approach. Also see Questions
2. The reliability of the automatically annotated dataset is mainly accessed by case study, which may lead to doubt in applying the dataset further in future.

### Questions
1. High-performing baselines like ProtST is missing for experiments.
2. In Fig. 4 and Tab. 2, ProtAC-ESM2-35M-Uncleaned is missing.  Can authors provide these missing comparisons?
3. For Fig. 4d, the comparison may be unfair due to different computational resources used. For Scratch-epoch4 and Scratch-original, the pre-training only lasts for 1 epoch. But for ProtAC-ProteinBERT-cleaned, it experienced 4 epochs staged pre-training. 
4. Could authors further explain "Among the 20 sampled clusters, clusters are no longer present in UniRef50"? How does this relate to the prediction of transmembrane regions with the Phobius tool?
5. In ablation, why does removing MLM loss lead to better performance for KW prediction?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The authors propose a novel method to recursively clean the large-scale protein dataset, namely UniRef50 and Swiss-Prot in the paper's context. The authors develop a multi-step cleaning procedure alone with a joint model for predicting the annotation as well as judging whether the annotation is correct. Experiments show the efficiency of the proposed methodology.

### Strengths
1. The manuscript is generally well-written, along with good figures that can clearly explain the details.
2. The analysis authors conduct are solid and the topic of cleaning protein datasets can be helpful to the wide community of AI4protein.

### Weaknesses
1.It is a solid idea to clean the existing protein datasets and strengthen their quality. However, the data cleaning procedure is not novel since this is a well-explored topic in natural language processing. So in my opinion, the major contribution of the paper is on the cleaned dataset itself, not on the methodology, while the main body of the paper is devoted to it.

2.The authors specifically emphasize on the efficiency of their methodology. But efficiency should not be the primary concern since you only need to produce the cleaned dataset once. The performance should be dominantly important instead. However, the large gap between esm2 and protbert seems to show that the performance mainly relies on the pretrained language model, not on the procedure itself.

### Questions
1. The figure4, 5 and 6 have too small fonts. Please make them larger.

### Soundness
2

### Presentation
3

### Contribution
2
