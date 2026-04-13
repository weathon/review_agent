## Human Reviewer 1

### Summary
This paper presents ProtEx, a retrieval-augmented model designed to enhance protein function prediction by leveraging exemplar sequences from a reference database. By conditioning predictions on retrieved exemplars, the model aims to improve accuracy and generalization, especially for underrepresented classes. ProtEx is applied to various functional annotation tasks, such as predicting EC numbers, GO terms, and Pfam families, where it reportedly achieves state-of-the-art performance. The proposed approach includes a unique multi-sequence pretraining task and a targeted fine-tuning strategy to maximize the benefit of exemplar retrieval.

### Strengths
- The integration of exemplars into protein function prediction is a novel direction that could address challenges with generalization and robustness, particularly for underrepresented classes in protein datasets. This retrieval-augmented approach could be valuable in instances where limited data is available for specific protein functions during training.
-  ProtEx is tested across diverse protein function prediction tasks, including EC, GO, and Pfam annotations. This extensive evaluation provides insights into the method’s versatility and potential applicability to various protein annotation domains.
- The authors provide proper ablation studies and analyses showing ProtEx’s capability to improve performance on classes with limited representation in the training data, a common challenge in protein function prediction.
- The research is well motivated and of good biological significance.

### Weaknesses
- While retrieval-augmented models are relatively new in the protein domain, the approach is already widely used in other areas, such as natural language processing. For broader readers,  the authors may consider use a few illustrative figures or tables to justify the significance of retrieval-augmentation for protein function prediction. (I am not questioning this, I believe in the importance of similarity-based prediction for protein, and think this will further enhance the quality of manuscript for readers who are not familiar)
- The performance of ProtEx heavily depends on the quality and comprehensiveness of the reference database. In cases where exemplars are sparse or unavailable for certain protein functions, the retrieval mechanism may underperform, potentially limiting the model’s utility in practical applications with limited exemplar coverage. Only BLAST is tested for similarity search, which may weaken the soundness of this study.
- Although ProtEx aims to improve generalization, its effectiveness for predicting completely novel functions is not well-supported in the experiments. The evaluation focuses primarily on tasks with annotated classes, leaving uncertainty about the model’s robustness in generalizing to truly unseen or hypothetical functions, a critical aspect of protein function prediction. As a matter of fact, a lot of EC classes, for example, are labeled by expert largely based on the sequence similarity and that explains why the BLAST baseline alone yields already good results.
- The paper primarily compares ProtEx against few retrieval-based and BLAST baselines but does not provide extensive comparison with other models (especially, **under their data train-test split for comparison**) that may be equally effective for protein function prediction. This makes it difficult to assess the value of ProtEx. For example, in one of the mentioned papers, ProtIR, I saw the authors there have benchmarked across different types of models on specific data splits and their BLAST results have a distinguishable mismatch with the one shown in this paper. Add such comparison can improve the soundness of this paper.

### Questions
- Have you explored using other sequence-based / structure-based alignment or search engines, potentially with different retrieval mechanisms, to assess if ProtEx’s performance varies based on the retrieval engine? For example, foldseek, mmseqs, and even neural-based models. Such experiments could clarify the model’s dependency on specific retrieval tools. 
- Have you considered applying ProtEx (as an encoder) as an augmentation layer on top of existing protein embeddings rather than directly on raw sequences? Testing ProtEx in this way might provide insights into whether it could benefit from the structural and functional information encoded in precomputed protein embeddings.
- Could you provide an analysis of how ProtEx performs when fewer exemplar sequences are available, or even when limited to a small number of retrieved sequences? For example the orphan protein. This could help in understanding ProtEx’s robustness and effectiveness when retrieval options are sparse.

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

## Human Reviewer 2

### Summary
The paper introduces ProtEx, a retrieval-augmented approach for protein function prediction, emphasizing the importance of conditioning predictions on exemplar sequences, particularly for less represented classes and sequences in the training data. The approach is novel in integrating retrieval into protein prediction tasks, improving generalization across diverse protein classes, and effectively addressing the data scarcity for certain sequences.

### Strengths
1. The empirical performance for ProtEx outperforms other reported baselines on essential protein function prediction tasks.
2. The introduction of a retrieval-augmented approach to protein function prediction is innovative, particularly the use of exemplar conditioning which helps in handling underrepresented data effectively.

### Weaknesses
1. The increased computational complexity due to the conditioning on labels is a drawback, particularly if scalability or application to datasets with larger classes is considered.

2. Some experimental details are not clear. See questions.

### Questions
1. In Sec. 3, the proposed method is conditioned on label and labels are enumerated. What's the advantage for the conditioning?

2. Since ProtEx uses BLAST to produce the pre-training and fine-tuning datasets for exampler, it would be reasonable for ProtEx to outperform BLAST. Could the integration of BLAST results with ProtEx be considered to enhance performance? If not, maybe ProtEx totally absorbs the knowledge from BLAST

3. Also, when not using examplars for fine-tuning, i.e., not using BLAST for help, ProtEx perform even much worse than BLAST.  In this sense, the largest constribution for ProtEx seems to be incorporating BLAST for downstream tasks, instead of its pre-training or fine-tuning strategies. How do the authors justify the innovation in their approach given the above thinking?

4. In Tab. 6, if the candidate labels are not incorporated, how the prediction is done, since the model cannot take it as input?

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
5

### Confidence
3

---

## Human Reviewer 3

### Summary
The authors propose ProtEx, a novel approach to use retrieval-augmentation to enhance protein language models in protein function prediction tasks. The retrieval part is done by a devoted language model to determine the similarity between two protein sequences. The information of retrieved sequences are then integrated into query sequence to determine the prediction labels. The author conduct extensive experiments to show the validity of their approach, and conduct several ablation study.

### Strengths
1. The paper is well-written and clearly structured. The details are clearly explained in training and experiments part.
2. The idea of retrieval-augmentation for protein function prediction is nice, though not novel. However, the paper is devoted to its methodology part, and the experiments show that their method outperforms current SOTA.
3. I really like the various analysis and ablation studies in experiment part. They are quite convincing and helpful to the community.

### Weaknesses
1. It is to be discussed whether the paper is novel enough for a machine learning conference. pro: the task of protein function prediction with retrieval-augmentation is not well-explored today, and the paper provides meaningful discussion. con: the methodology itself is not novel. I can see that some parts are well designed, like the old-BERT-style pretraining loss, but I don't think they are contributions in a machine learning conference.

### Questions
The paper is well written to its claimed scope. I have no further questions.

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