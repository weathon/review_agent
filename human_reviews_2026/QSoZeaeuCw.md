# HyperHELM: Hyperbolic Hierarchy Encoding for mRNA Language Modeling

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 2, 6, 4

## Abstract
Language models are increasingly applied to biological sequences like proteins and mRNA, yet their default Euclidean geometry may mismatch the hierarchical structures inherent to biological data. While hyperbolic geometry provides a better alternative for accommodating hierarchical data, it has yet to find a way into language modeling for mRNA sequences. In this work, we introduce HyperHELM, a framework that implements masked language model pre-training in hyperbolic space for mRNA sequences. Using a hybrid design with hyperbolic layers atop Euclidean backbone, HyperHELM aligns learned representations with the biological hierarchy defined by the relationship between mRNA and amino acids. Across multiple multi-species datasets, it outperforms Euclidean baselines on 9 out of 10 tasks involving property prediction, with 10\% improvement on average, and excels in out-of-distribution generalization to long and low-GC content sequences; for antibody region annotation, it surpasses hierarchy-aware Euclidean models by 3\% in annotation accuracy. Our results highlight hyperbolic geometry as an effective inductive bias for hierarchical language modeling of mRNA sequences.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper extends the work presented in “HELM: Hierarchical Encoding for mRNA Language Modeling” [1] by replacing the hierarchical encoding in Euclidean space with one in hyperbolic space. Hyperbolic geometry is leveraged as an inductive bias for hierarchical language modeling of mRNA sequences. Following the architecture described in [1], the proposed HyperHELM model employs a Euclidean backbone, with only the final layer replaced by a hyperbolic prototypical classifier. This design preserves hardware efficiency while exploiting the representational advantages of hyperbolic geometry. The paper investigates three approaches to hyperbolic learning: hyperbolic multinomial logistic regression (MLR), distance-to-prototype learning, and prototype classifiers based on hyperbolic entailment cones. It is the first work to explore hyperbolic language modeling for mRNA sequence data, demonstrating clear benefits over Euclidean space-based models across multiple downstream tasks.

[1] Mehdi Yazdani-Jahromi, Mangal Prakash, Tommaso Mansi, Artem Moskalev, and Rui Liao. HELM: Hierarchical Encoding for mRNA Language Modeling. In *International Conference on Learning Representations*, 2025.

### Strengths
- Overall, the paper is well written, with a clear central idea and an effective flow that guides the reader through the motivation and methodology.
- The paper presents **HyperHELM**, the first mRNA language model pretrained using masked language modeling in hyperbolic space. While the concept of hierarchical encoding for mRNA sequence data was previously introduced in [1], the application of hyperbolic geometry to RNA sequence modeling is novel.
- HyperHELM is evaluated on ten diverse downstream tasks spanning multiple organisms and covering various mRNA-related properties, such as protein production, expression levels, degradation, and abundance. It achieves the best performance on nine out of ten datasets, underscoring the advantages of hierarchy-aware learning in hyperbolic space.
- Unlike the Euclidean space-based HELM, which struggles with longer RNA transcripts, HyperHELM maintains or even improves its performance as sequence length increases. This finding is consistent with prior reports of hyperbolic models in other domains and is particularly important, as longer mRNA sequences are often underrepresented in datasets and pose challenges for existing models.

### Weaknesses
- **Novelty:** While the idea of using hyperbolic geometry has not previously been applied to mRNA sequence modeling, it has been successfully explored across many areas of machine learning. The application of hyperbolic geometry to mRNA sequence encoding is well motivated; however, the theoretical framework used in the paper does not contribute new insights to the broader machine learning field. As such, the paper’s novelty lies primarily in its application rather than in methodological innovation.
- **Reproducibility:** The paper does not mention whether the code or pretrained model weights will be made publicly available. For application-oriented work, this should be considered essential. Moreover, the dataset splits used for downstream tasks are not provided. This omission is particularly concerning, given that labeled mRNA datasets are typically small (often fewer than 5,000 samples). Repeating experiments with different random splits can lead to significant variation in performance.
- **Downstream task results**: Table 1 reports Spearman’s rank correlation and accuracy for ten tasks, but only single numerical values are provided, without clarification on whether these represent mean results or single runs. Combined with the lack of details on dataset splits, this makes the results potentially unreliable. For such small datasets, it is standard practice to perform cross-validation or multiple runs with different splits, reporting both mean and standard deviation to ensure robustness and statistical validity.
- **Baselines**: HyperHELM is compared to five baselines on four downstream tasks, but only two baselines on the remaining six. The chosen baselines include RNA-FM, SpliceBERT, and CodonBERT. Since the authors use a simple probing setup (with frozen language model parameters), comparisons to RNA-FM, a model pretrained exclusively on non-coding RNAs, are not particularly informative. Similarly, SpliceBERT was pretrained on pre-mRNAs, making mRNA data effectively out-of-distribution. While these models may perform well when fine-tuned on new RNA types, such comparisons are not meaningful under a probing-only setup. Additionally, simpler baselines such as linear or logistic regression are missing and would help contextualize the gains of the proposed approach.
- **References**: Several references appear unrelated to the core topic (for example, *Xu et al.* in line 131). There also appears to be an overrepresentation of citations from a single research group, some of which are only tangentially related and not essential for understanding the paper. A more balanced and focused selection of references would strengthen the scholarly framing.

I recommend **rejecting** the paper.

1. The paper lacks methodological novelty in machine learning, as it mainly extends an existing framework (HELM) by replacing Euclidean space with hyperbolic space.
2. Reproducibility is limited due to the absence of publicly available code, pretrained weights, and dataset splits, raising concerns about result reliability.

While the paper is clearly written and the motivation to use hyperbolic geometry for mRNA sequence modeling is well justified, the contribution is primarily applied rather than methodological. The theoretical and algorithmic components of hyperbolic learning are already well established in prior literature, and the paper does not introduce new learning principles or modeling techniques.

Empirically, the reported results on downstream tasks are promising - HyperHELM outperforms Euclidean baselines on 9 out of 10 datasets and shows robustness to longer RNA sequences. However, the absence of code, pretrained weights, and dataset splits makes it difficult to verify or reproduce these findings. Moreover, the use of small datasets without clear information on data splits or repeated trials reduces the statistical reliability of the reported improvements.

Overall, the work is a solid application of existing hyperbolic methods to a new biological domain, but it falls short of the novelty and reproducibility standards expected for acceptance at a top-tier conference. The paper, in this form, is more suitable for presentation at a workshop than a main conference.

### Questions
- **Availability:** Will the code, pretrained model weights, and downstream task dataset splits be made publicly available if the paper is accepted?
- **Reproducibility:** Please comment on the reproducibility of the work. Are sufficient details provided to replicate the experiments?
- **Downstream task results:** In Table 1, please clarify what the reported numbers represent. Are these results from single runs, or do they correspond to mean values across multiple experiments or cross-validation folds?
- **Baselines**: Can you compare HyperHELM to simpler baselines? I think having proper baselines would improve the overall presentation of the results.
- **HXE loss parameter:** Have you experimented with different values of the hyperparameter $ \alpha $ in the HXE loss? It appears to be an important factor, yet it was fixed at 0.2 for all experiments. Please elaborate on the reasoning behind this choice and whether sensitivity analysis was performed.

### Soundness
3

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
3

### Summary
The authors propose HyperHELM, a framework for representing mRNA sequences in hyperbolic space to better capture hierarchical structure that may be present in the data. HyperHELM uses a Euclidean transformer backbone paired with several heads for MLM prediction: hyperbolic multinomial logistic regression, and two prototype-based classifiers that use either geodesic or entailment cone energy as similarity functions. The codon-amino acid hierarchy is used to generate fixed prototypes for the latter options. HyperHELM compares against several RNA foundation models, and against a fully Euclidean model trained on the same data. The authors find performance benefits on 9/10 mRNA property prediction tasks when both the prototype and hyperbolic representation learning methods are used in conjunction. The authors conclude by analyzing the performance of HyperHELM as a function of sequence length and GC content on several downstream tasks.

### Strengths
- To my knowledge, this work is the first implementation of hyperbolic learning into mRNA language modelling. The adaptation of new learning approaches to mRNA biology is an area of high significance.
- The use of prototype learning to encode the codon-amino acid hierarchy has strong biological relevance and offers an interpretable inductive bias for mRNA representation learning in the CDS region.
- The evaluation across sequence length and GC content distributions reveals biologically meaningful insights about model generalization.
- Presentation of background and methodology on hyperbolic learning is written with high clarity.

### Weaknesses
- My primary concern lies with biological validity of the tokenization strategy underlying the prototype learning. While using trinucleotide tokenization in the CDS region makes sense, the paper does not explain how the reading frame is selected. Without proper alignment to the annotated start codon, the tokenization could be out-of-frame, preventing the codon-amino acid structure from being leveraged. Furthermore, this strategy has no biological validity in an mRNA's UTR regions. For example, the presence of upstream AUGs [1] strongly affect mRNA translational efficiency, but the proposed tokenization strategy could completely fail to represent them due to misalignment.

- Pre-training dataset description is unavailable without referencing an external paper, hurting self-contained clarity. More critically, if I am not mistaken, the pre-training database primarily contains antibody-encoding sequences. Given that this corpus is relatively specialized, it introduces a confound for the antibody-related downstream tasks, as domain-matching the pre-training data may drive a significant portion of the model improvement over language model baselines.

Experimental evaluation is relatively weak. 
- Choice of baseline foundation models is incomplete. Several models identified in the related works are not evaluated, in particular models which are actually suited for mRNA biology. Current selection include models trained on ncRNA (RNA-FM) and pre-mRNA (SpliceBERT) while models such as mRNA-FM, AIDO.RNA-CDS, Orthrus, Evo2, Helix-mRNA, etc. all provide stronger and more appropriate baselines.
- Selected tasks almost exclusively focus on quantifying protein expression, while mRNA biology encompasses many additional properties. If HyperHELM is meant to only focus on expression, the scope of the claims should be restricted.

Incomplete related work coverage:
- Critically, the paper should make specific comparison to "Hyperbolic Genome Embeddings" [2], which previously explored hyperbolic learning in genomic sequences. It may be appropriate to restrict the scope of the originality claims and suggested future directions in response to this citation.
- The related works section contains an incomplete overview of methods for mRNA representation learning. Several of the strongest models such as AIDO.RNA-CDS [3], Orthrus [4], and LoRNASH [5] are not referenced. Supervised approaches that are directly applicable to the downstream tasks such as RiboNN [6] and Optimus 5' [1] are not mentioned.

[1]: https://www.nature.com/articles/s41587-019-0164-5
[2]: https://arxiv.org/abs/2507.21648
[3]: https://www.biorxiv.org/content/10.1101/2024.11.28.625345v1
[4]: https://www.biorxiv.org/content/10.1101/2024.10.10.617658v3
[5]: https://www.biorxiv.org/content/10.1101/2024.08.26.609813v2
[6]: https://pubmed.ncbi.nlm.nih.gov/39149337/

### Questions
I would appreciate a rebuttal to the above concerns, particularly a detailed description of how mRNA transcripts were processed for pre-training, and how the reading frame problem was tackled. Specifically, my questions are:
- Are pre-training sequences full transcript, or CDS-only?
- How was the reading frame determined?
- How are UTRs handled?

Furthermore, I am curious whether the proposed downstream tasks empirically contain hyperbolicity. I would appreciate the authors considering whether an evaluation of the downstream tasks as proposed in Hyperbolic Genome Embeddings would be suitable here.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces HyperHELM, a framework for modeling mRNA sequences. The authors argue that the inherent hierarchical structure of mRNA is not well represented by the Euclidean geometry used in standard language models. They propose a hybrid model that combines an efficient Euclidean transformer backbone with a hyperbolic prediction head. This model is pre-trained using a masked language modeling objective directly in hyperbolic space. The hyperbolic representations are guided by prototypes derived from the known codon hierarchy. The paper evaluates two types of prototype classifiers, one based on distance and another on entailment cones. The results show that HyperHELM outperforms Euclidean baselines. The model also shows better out-of-distribution generalization to long sequences and sequences with low GC content.

### Strengths
1. Novel and well motivated application: The paper applies hyperbolic geometry to mRNA language modeling. This is a very natural and well justified choice. 

2. The choice of a hybrid architecture is pragmatic. The model retains the hardware efficiency of a standard Euclidean transformer for most of its computations. It only projects into hyperbolic space at the final layer. This design gains the benefits of hyperbolic geometry without incurring the full computational cost of a purely hyperbolic network.

3. The experiments are well designed. The authors compare HyperHELM not just to standard baselines but also to HELM, a state of the art Euclidean model that is also hierarchy aware. This comparison effectively isolates the benefit of using hyperbolic geometry rather than just using the hierarchy itself.

### Weaknesses
1. The prototypes used in the hyperbolic head are fixed. They are generated once from the codon hierarchy using an embedding method and are not updated during training. It seems like a potentially missed opportunity. Allowing the prototypes to be learnable might permit the model to fine tune the hierarchical relationships based on the data. 

2.  The paper focuses exclusively on masked language modeling (MLM). While this is a standard objective, it would be interesting to see if the benefits of the hyperbolic hierarchy encoding also apply to generative tasks, such as those using a causal language modeling objective.

3.  The HyperHELM MLR variant performed very poorly, even worse than the Euclidean baselines. The paper could benefit from a more detailed discussion of why this specific formulation failed, while the prototype based methods were so successful. It strongly suggests that the prototype learning aspect is just as critical as the hyperbolic space itself.

### Questions
1. Could the authors elaborate on the poor performance of the HyperHELM MLR variant? What is the intuition for why this method struggles so much, given that the prototype models demonstrate a clear benefit from the hyperbolic space?

2. The model architecture involves a projection from the Euclidean backbone's output to the hyperbolic space. Did the authors study the sensitivity of the model to this projection step? For instance, how much does the choice of projection function (detailed in Equation 9) matter for the final performance?

3. The paper uses the HS DTE method to embed the codon hierarchy and generate the fixed prototypes. How critical is this specific embedding choice to the model's success? Have the authors considered or briefly tested other, perhaps simpler, tree embedding methods?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces HyperHELM, a new framework for modeling mRNA sequences that leverages hyperbolic geometry to better capture the data's inherent hierarchical structure.

### Strengths
-The model's core idea is to use a geometry (hyperbolic) that naturally matches the data's inherent structure (hierarchical).

-The MLR variant performed very poorly, even worse than the Euclidean models. This serves as a powerful ablation study, demonstrating that simply using hyperbolic space is not enough. The performance gain comes from the combination of hyperbolic geometry with an explicit, hierarchy-aware prototype learning scheme

### Weaknesses
-The model's prototypes are explicitly fixed. They are generated by an external method, HS-DTE, before training begins. This presents a significant technical bottleneck. The entire performance of the model is permanently limited by the quality of this one-time embedding.

-The model appears to be "double-dipping" on the hierarchical information, which could be redundant or even create conflicting signals

-This method involves an exponential map centered at the origin ($exp_0^c$). This is a very specific and strong assumption, treating every output token from the Transformer as a tangent vector at the origin. This step could be a significant information bottleneck, and it's not clear if this single-point-of-origin mapping is the optimal way to bridge the two geometric spaces without losing a
lot of the rich information captured by the Euclidean backbone.

### Questions
-Why did you choose to keep the prototypes fixed? Do you think making them learnable would allow the model to capture more subtle, data-driven relationships beyond the canonical codon hierarchy, or would it risk "overfitting" and losing the strong biological prior?

-The HyperHELM MLR variant (which used hyperbolic space but no prototypes) performed very poorly. Does this imply that hyperbolic geometry alone is insufficient and that the explicit "prototype-based" classifier is the real source of the performance gain?

-What specific geometric property of the hyperbolic space do you believe enables the model to generalize so much better to long sequences, where the Euclidean model fails?

-why there are missing values for CodonBERT on some tasks that are in the CodonBERT paper?

-why not comparing with EVO 2 - https://www.biorxiv.org/content/10.1101/2025.02.18.638918v1.abstract


-would like to increase my score if the above concerns are well addressed

### Soundness
3

### Presentation
2

### Contribution
2
