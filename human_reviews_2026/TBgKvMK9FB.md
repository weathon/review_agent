# TIGER: Text-Informed Generalized Enzyme-Reaction Retrieval

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 2, 6

## Abstract
Enzyme–reaction retrieval is a fundamental problem in computational biology, underpinning enzyme characterization, reaction mechanism elucidation, and the rational design of metabolic pathways and biocatalysts. As a bidirectional task, it entails both enzyme-to-reaction and reaction-to-enzyme mapping. However, existing approaches suffer from poor generalization across tasks and distributions, with performance highly sensitive to dataset splits and substantial asymmetry between retrieval directions. To address these challenges, we present TIGER, a Text-Informed Generalized Enzyme-Reaction Retrieval framework that leverages protein-to-text generation models to distill textual semantic knowledge from enzyme sequences, providing a generalized representation that bridges enzymes and biochemical reactions. To ensure the quality and reliability of textual semantics, we design a Dynamic Gating Network that adaptively fuses text-derived knowledge with sequence features, enabling more consistent and informative enzyme representations, while a Structure-Shared Feature Projector aligns enzyme and reaction representations within a unified latent space. Extensive experiments demonstrate that, under bidirectional retrieval supervision, TIGER significantly outperforms state-of-the-art baselines across diverse distributions and exhibits strong robustness and transferability across tasks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper addresses the bidirectional enzyme–reaction retrieval problem, highlighting fundamental challenges around asymmetric retrieval directions and poor generalization under distribution shifts in current approaches. The authors introduce TIGER, a Text-Informed Generalized Enzyme-Reaction Retrieval framework, which leverages protein-to-text generation models to augment enzyme representations with knowledge-rich textual descriptions. This information is adaptively integrated via a Dynamic Gating Network to mitigate textual noise, while a Structure-Shared Feature Projector aligns enzyme and reaction representations into a unified latent space. The method is trained with a symmetric contrastive objective. Extensive experiments on the ReactZyme benchmark show that TIGER substantially outperforms strong baselines across several challenging evaluation splits, with in-depth ablation studies supporting each architectural component.

### Strengths
1. The experimental evaluation is robust and thorough. The authors split the dataset based on time, enzyme similarity, and reaction similarity to rigorously test for generalization, and this is supported by abundant quantitative evidence and detailed documentation in both the main text and the appendix. I particularly appreciate this data-splitting methodology.
2. Clear problem motivation and positioning. The authors articulate the limitations of existing contrastive retrieval frameworks (directional asymmetry, lack of robustness to data splits) and motivate the introduction of textual knowledge with references to related successful paradigms in multimodal learning.
3. Coherent integration of textual and structural protein information. The method employs protein-to-text generation (ESM2Text) to introduce explicit functional and mechanistic information, combined with amino acid sequence features. The Dynamic Gating Network (DGN) adaptively weighs these sources, responding intelligently to the quality of generated text. The effect of DGN is thoroughly analyzed with concrete performance metrics

### Weaknesses
1. The theoretical novelty is limited. The model follows a standard contrastive framework, despite the inclusion of equations for its gating mechanism, embedding projection, and loss. The paper lacks new theoretical findings or a deeper analytical investigation into the system's learning properties.
2. The handling of negative sampling and training details is ambiguous. The contrastive loss formula relies on in-batch negatives, but the paper provides no details on the sampling strategy, the temperature parameter initialization, or the method for addressing potential class imbalance.
3. The paper lacks a comparison with more recent methods. Newer models, such as DeepEnzyme and ProtDETR, already predict enzyme function or substrates, and their training objectives could be modified to apply them to reaction prediction. The absence of a relevant comparison weakens the claim that TIGER achieves state-of-the-art performance.
4. Some of the equations are not numbered.

### Questions
1. From the model architecture in Figure 2, the training process for TIGER is clear, but its inference procedure is not. Based on the task formulation, it seems the model maps a single enzyme to a single reaction (and vice versa). How does TIGER handle cases where an enzyme is associated with multiple reactions, or a reaction can be catalyzed by multiple enzymes?
2. Does the textual description of an enzyme generated by ESM2Text potentially contain information about its catalytic function? Is there a risk of target reaction leakage? An analysis of the enzyme's textual description is required.
3. The paper compares "AI text" vs. "human text" but does not test intermediate text quality controls (e.g., fine-tuning ESM2Text on SwissProt, filtering low-similarity text before fusion). This makes it unclear whether TIGER’s DGN is a necessary solution or a workaround for unoptimized text generation, especially since human text still outperforms AI text.
4. The results section lacks a more detailed analysis. The paper does not analyze performance on low-data subsets (e.g., enzymes with <5 associated reactions, or reactions catalyzed by <3 enzymes).  Real-world enzyme-reaction retrieval often involves rare biocatalysts, but it has not yet been validated whether TIGER can alleviate the problem of data sparsity.
5. What are the batch size, temperature initialization, and hard/soft negative mining settings for the contrastive loss? Is class imbalance meaningfully addressed, given the pronounced asymmetry in enzyme vs. reaction cardinalities?
6. Ablations on Structure-Shared Feature Projector. Is its contribution demonstrably better than standard one-layer MLP or simplified attention mechanisms? Can the authors provide a controlled experiment to show the effects of this component alone?
7. Why are several enzyme-related methods, such as DeepEnzyme and ProtDETR, omitted from the main comparison tables and the discussion? What prevented their experimental inclusion or, at a minimum, a deeper conceptual positioning of your work in comparison to them?

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
5

### Summary
The authors proposed a new text-informed generalized enzyme-reaction retrieval, which using a well-established text-protein interface with a new and popular task enzyme-reaction retrieval. With dynamic gating network, the model is able to learn a combination of representation for enzyme and its text description. So actually, the model use a protein2text model to enforce the ability of enzyme-reaction retrieval.

### Strengths
Usually, researcher consider using text2protein or protein2text as a seperated task, using this kind of model to enhance the performance in other tasks is not well explored before. Since enzyme-reaction retrieval is an extremely hard task in enzyme-related deep learning models, using text to inform better enzyme understanding is an interesting trial to help this task, or even future enzyme-related or protein-related hard tasks.

### Weaknesses
In the enzyme-reaction retrieval task, since there are a lot of protein related model using both or either protein sequence, structure, surface, function, etc. But the reaction or molecule side is comparably not well-explored. Currently most methods using only molecule-pretrained models, but not understanding the reaction or the description of the reaction. If we are able to use texts to enforce the reaction side, it would be more impactful in this field.

### Questions
1. In the performance in Table 1 and Figure 4, when there is no text information, the performance is still better than the baselines, can you explain why this is better than them since there is no model improvement as we know.
2. Using SwissProt text for comparison might contains potential data leakage, since the text in SwissProt might contain the chemical name of the target substrate and product, what did authors do to prevent such leakage.
3. The generated text information might contain the reaction information of the enzyme, is it possible to report the accuracy using only the protein to text model (extract the reaction-related information from the predicted text)
4. Can you report the accuracy of the Hit@20 which is used in Reactzyme? (Has at least 1 correct hit in top 20 prediction)

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a text-informed enzyme-reaction retrieval framework. It considers both enzyme to reaction and reaction to enzyme directions and applies a contrastive learning algorithm to learn the aligned enzyme and reaction representations to achieve retrieve.

### Strengths
The paper is introduced in a clear logic and it applies a multitask learning framework to simultaneously learn the enzyme to reaction and reaction to enzyme mappings to achieve better aligned representations.

### Weaknesses
The meain weakness of this paper are listed as follows:

1. The claim in this paper sounds a little bit incorrect to me. In line 59-61, the paper mentioned "previous methods demonstrate cross-directional asymmetry, where the retrieval accuracy from enzymes to reactions substantially diverges from the reverse direction." Why would the author assume the accuracies of the two directions should be the same? The reaction types are around 8400 classes in EC tree, but the enzyme space is much larger (20^L).  The two tasks have different difficulty. Also, in Table1, the paper didn't achieve similar performance on the two tasks, like in Enzyme Similarity-based Split, E-> R is much higher than R->E. Their performance didn't match with their motivation and claim. According to the method in this paper, I would say the method is just more like a multitask learning.

2. In line 244, the author mentioned the enzyme sequence shows  structural information. However, just using enzyme sequence can't explicitly demonstrate the structural information.

3. Another major limitation the paper mentioned about previous method is "these models show a high sensitivity to dataset splits". I think if the paper would show higher robustness of their method, it should be tested on more dataset instead of just one, like EnzymeMap and CARE.

[1] EnzymeMap: curation, validation and data-driven prediction of enzymatic reactions.

[2] CARE: a Benchmark Suite for the Classification and Retrieval of Enzymes.

4. The paper should compare to more well-developed baseline models instead of just methods set up by themselves, like [3].

[3] Enzyme function prediction using contrastive learning.

5. For the R->E, what is the candidate pool of the potential enzymes? Are they just the whole training enzyme sequences or the whole testing sequences? If just training sequences, it means this method will always retrieve the seen sequences in training. If it's all testing sequences, I don't think it's a fair setting since in reality you don't have a "ground truth" pool to choose from. If it's the whole Swiss-Prot or even Uniprot, it would make more since to me.  Additionally, how to calculate the R-> E hit@1 rate? In a more reasonable setting, enzymes belonging to the same EC class should be regarded as correct. If only exactly the same sequence is regarded as correct, I don't think it's a reasonable setting.

6. I'm actually not quite sure the practical application of R->E. E->R can be used to predict the function of a given enzyme, which is clear. But retrieving an existing enzyme without considering any zero-shot settings makes no sense to me.

Some small issues are:

1. Some citations are not correct, like line 58 and line 60.

### Questions
1. To calculate the fused enzyme representations, how about just concatenating the $s_{attn}$ and $t_{attn}$ to compute $f_E^{Fused}$ since $f_{gated}$ are from $s_{attn}$ and $t_{attn}$?

2. How to get the 3D structures of the substrates and products?

3. To calculate the reaction representations, how about just using the reaction smiles like EnzymeMap as SMILES also includes the molecule structure information? Since the paper doesn't consider the enzyme-substrate complex, I don't think using smiles or sole molecule structure would lead to difference.

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
4

### Summary
The paper presents TIGER, a framework that integrates protein sequence representations with textual semantics for bidirectional enzyme–reaction retrieval. On one hand, the authors employ ESM-2 to extract sequence embeddings and utilize PubMedBERT to derive text-based semantic representations from protein-to-text generation. A Dynamic Gating Network is designed to adaptively fuse the textual and sequence-derived features, while a Structure-Shared Feature Projector aligns enzyme and reaction embeddings within a unified latent space. On the other hand, UniMol-3D is leveraged to generate reaction embeddings. The model is then trained via contrastive learning to achieve bidirectional retrieval between enzymes and reactions. Extensive experiments conducted across multiple benchmarks and cross-distribution settings demonstrate that TIGER achieves superior retrieval accuracy and robustness compared to existing baselines.

### Strengths
1. Multimodal innovation: Incorporating protein→text semantic information into enzyme representation is an interesting and promising idea, effectively injecting literature-level semantic knowledge into sequence-based embeddings.

2. Reasonable bidirectional retrieval setting: The simultaneous study of both enzyme→reaction and reaction→enzyme directions is well-motivated and covers realistic application scenarios.

3. Comprehensive experiments: The authors conduct extensive comparative experiments under various data splits and cross-distribution settings, providing convincing empirical evidence for the effectiveness of the proposed method.

### Weaknesses
1. Limited fine-grained semantic capture: As the authors themselves note, the processing of textual descriptions is relatively coarse-grained. It would be beneficial to supplement the analysis with a discussion or experiment on how fine-grained reaction details (e.g., catalytic residues, reactive centers) influence the matching process.

2. Implicit and insufficient text quality evaluation: The paper mentions text generation via ESM2Text and comparison with human-written SwissProt data, but the evaluation of text quality remains implicit. It would strengthen the study to include additional AI-generated texts (e.g., from other protein–text generation models) as baselines for comparison.

3. Weak biological validation: While the retrieval performance is empirically validated, the biological interpretability and practical applicability are less explored. Including case studies—such as retrieved candidate enzymes confirmed in literature or databases—would make the work more convincing.

4. Data handling and many-to-many relationships: Since enzyme–reaction mappings are inherently many-to-many, the contrastive learning setup might mistakenly treat some true positive pairs as negatives. The manuscript does not appear to discuss how this issue is handled in data preprocessing; clarification or additional details are needed.

### Questions
1. How does the model handle the many-to-many nature of enzyme–reaction relationships to avoid false negatives in contrastive learning?

2. Have the authors considered incorporating fine-grained textual cues (e.g., catalytic mechanisms or reaction site descriptions) to enhance semantic alignment?

3. Could additional AI-generated protein descriptions be included to test the robustness of the Dynamic Gating Network to varying text quality?

4. Are there specific biological case studies (e.g., retrieved enzymes validated by experimental or database evidence) that can illustrate the model’s real-world usefulness?

5. How sensitive is the TIGER framework to noisy or partially incorrect textual descriptions in protein–text embeddings?

### Soundness
3

### Presentation
3

### Contribution
3
