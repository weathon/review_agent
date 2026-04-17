# Drug-few: A few-shot learning method for target-specific drug virtual screening with interaction-informed adaptation

- Decision: Reject
- Scores: 2, 4, 4, 2

## Abstract
Few-shot virtual screening (VS), which aims to identify active molecules for a target with only a few known ligands, is crucial for accelerating drug discovery in cases where experimental data are extremely limited. Traditional ligand-based few-shot learning methods rely on molecular similarity or latent embeddings to generalize from these limited examples, but they often fail to capture target-ligand interactions, leading to overlooked active molecules. Here, we present Drug-few, the first few-shot learning framework designed for strict target-specific VS that explicitly incorporates binding-relevant information. Drug-few introduces prompt tokens for rapid target-specific adaptation and incorporates lightweight adapter modules to refine pocket-ligand representations. These components are combined in a Gated Prompt Adapter (GPA), where the contribution of prompt tokens is dynamically modulated by interaction-aware signals. Extensive experiments on three benchmark datasets show that Drug-few consistently outperforms the zero-shot baseline, maintaining strong retrieval performance even when the target active molecules are dissimilar to the known ligands, demonstrating its ability to generalize to novel molecules.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces Drug-few, a few-shot learning framework designed for target-specific drug virtual screening. The framework incorporates two modules: the Gated Prompt Adapter and Token-Level Mutual Interaction, which adapt the model to specific targets using a few known ligands. Drug-few demonstrates improved performance on the DUD-E, LIT-PCBA, and DEKOIS 2.0 datasets.

### Strengths
- This paper introduces Drug-few, a few-shot learning framework tailored for target-specific drug virtual screening, a relatively underexplored area.
- The integration of interaction-aware adaptation modules—specifically, the Gated Prompt Adapter (GPA) and Token-Level Mutual Interaction (TMI)—marks a significant advancement in few-shot learning for drug virtual screening.

### Weaknesses
- This paper lacks novelty because the GPA and TMI modules have been applied in few-shot scenarios in NLP and CV, as demonstrated by Conv-Adapter [1], Ta-Adapter [2], and MICL [3].
- The paper lacks a crucial baseline comparison, specifically with Drug-TTA [4]. Additionally, Drug-TTA's zero-shot learning performance surpasses that of Drug-few, which employs a few-shot learning approach with 16 samples. This raises concerns about the performance of Drug-few.
- Further evaluations on additional datasets, such as CASF-2016, are needed.

[1] Conv-Adapter: Exploring Parameter Efficient Transfer Learning for ConvNets.

[2] Ta-Adapter: Enhancing few-shot CLIP with task-aware encoders.

[3] Few-shot intent detection with mutual information and contrastive learning.

[4] Drug-TTA: Test-Time Adaptation for Drug Virtual Screening via Multi-task Meta-Auxiliary Learning.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents Drug-few, a few-shot learning framework for drug activity classification. The authors utilize a Gated Prompt Adapter (GPA) to encode information about the query molecules and information about the target protein, and an interaction module to refine the encodings based on the support set and its 3D interactions. A DrugCLIP-like model is then used to score similarities between compounds and targets, which is then used to predict if a drug will bind to the target or not.

### Strengths
1. Paper is well-written with clear figures and justification for each model component
2. Problem setting is convincing, and I agree there are many cases where a target structure and a few known compounds with activity are known
3. The model is overall well-constructed and makes many reasonable design choices, including the GPA and TMI modules
4. As far as I can tell, the authors are correct that no few-shot learning methods have been developed to predict drug-target interaction using the full structures of the pocket and associated drug interactions
5. Results are somewhat convincing, although without a baseline comparison it is hard to see if the improvement over zero-shot is strong or not.

### Weaknesses
1. While the authors introduce a new problem setting, a key consideration is whether incorporating structure can actually lead to better predictions. Since the paper does not compare to any previous methods that do not utilize the structure, it's hard to know if the method the authors introduce is effective
2. Similar to above, the authors should compare to classical drug-target interaction (DTI) methods besides DrugCLIP. For example, a comparison with Boltz-2's drug interaction probability score would be informative, as well as more traditional DTI models like [1].
3. The authors should also compare with other few-shot learning methods, even if they don't incorporate structure, for example [2], [3], and [4]. 
4. In my opinion, the problem setting introduced in [4], where there is a known protein structure but the structures of each individual drug-target interaction are unknown, is more realistic than the one used in this paper. This is because drug campaigns will probably not produce crystal structures for each compound they have that binds (e.g. hits from a HTS), and will only have one or a few structures of the protein determined. If the authors could justify their problem setting a bit more, that would be helpful.

Because there are no other baselines compared with besides DrugCLIP, I'm not convinced that the extra structural data utilized by Drug-few is critical for accurately predicting drug-target interactions. The results as they stand are not convincing, because there is no point of comparison for the performance improvement observed when giving the model multiple support examples.

[1] Lu et al. "DTIAM: a unified framework for predicting drug-target interactions, binding affinities and drug mechanisms." Nature Communications 2025.
[2] Altae-Tran et al. "Low data drug discovery with one-shot learning." ACS Central Science 2017.
[3] Eckmann et al. "Ligand-Based Compound Activity Prediction via Few-Shot Learning." Journal of Chemical Information and Modeling 2024.
[4] Paggi et al. "Leveraging nonstructural data to predict structures and affinities of protein–ligand complexes." PNAS 2021.

### Questions
N/A

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
5

### Summary
The paper proposes Drug-Few, a few-shot learning framework for molecular property prediction. It uses meta-learning and graph neural networks to transfer knowledge from large molecular datasets to new tasks with limited data. Experiments show that Drug-Few performs competitively under low-data conditions, demonstrating improved generalization for drug discovery. The authors note limitations in data quality and generalization to novel compounds

### Strengths
The idea of improving DrugCLIP for few-shot learning is interesting and relevant. The paper is clearly written, the model design appears reasonable, and the experiments are adequate, with fair comparisons to DrugCLIP. Overall, I find the work solid, though I will outline my main concerns below.

### Weaknesses
My main concern is that, since the method introduces information from active ligands during the inference stage, this information might be heavily incorporated into the final pocket representation, effectively turning the retrieval process into a ligand-based virtual screening. This raises a key question: is the performance improvement driven by a better ligand representation or by a better modeling of interactions? I believe this issue should be addressed through specifically designed experiments.

### Questions
my main concern please see the Weaknesses.

moreover, does batch size or batch composition sensitive to TMI module (or its perfromance) ?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors present a few-shot learning approach for drug discovery. Unlike many recent methods, the proposed model operates on 3D atomic conformations rather than SMILES strings or 2D graph representations, taking both the ligand and the protein pocket as inputs. The approach builds upon DrugCLIP, introducing two key modifications that enable (a) information sharing between ligand and pocket representations and (b) mutual updates of these representations based on the shared information. The authors evaluate their method against a naive zero-shot baseline across three benchmark datasets and conduct an ablation study to analyze the contribution of each component.

### Strengths
* **(S1 - relevance, novelty) - Few-shot models operating on 3D conformations represent a logical next step for improving SOTA approaches.** The authors correctly point out that current few-shot models typically operate on simplified representations, where both ligands and targets are expressed as SMILES strings (or equivalently as 2D molecular graphs). Instead of explicitly leveraging pocket information, targets are usually represented by the support set itself (as in similarity-based few-shot approaches). Ignoring known conformational knowledge in this context indeed overlooks valuable structural information.

* **(S2 - significance) - Few-shot adaptation improves performance, and the introduced modifications contribute to this gain.** The experiments demonstrate that the few-shot–adapted model consistently outperforms the naive zero-shot baseline. The inclusion of error bars supports the statistical significance of these performance improvements.

* **(S3 - relevance) - The introduced modifications are conceptually reasonable.** Enabling information sharing between ligand and protein pocket representations, and allowing mutual updates based on shared information, makes sense and the ablation study shows that they are helpful (however, see (W3)).

### Weaknesses
* **(W1 - quality) - Statement about target-specific virtual screening.** The authors claim to be the first to perform target-specific virtual screening. I would argue that this statement is incorrect, as several state-of-the-art (SOTA) models already employ target-specific prediction functions for a given target. MAML-based methods achieve this by explicitly fine-tuning their models on a target, deriving a target-specific bioactivity predictor within a few update steps. Similarly, similarity-based few-shot models represent the target using the support set itself, effectively taking it as a second model input. When fixing the target (i.e., the support set), similarity-based models derive a target-specific function analogous — and in many cases equivalent — to hypernetworks, where part of the model parameters are conditioned on processing this second input.

* **(W2 - quality, relevance) - Missing baselines.** In the absence of other few-shot models operating directly on 3D conformations, the authors should include drug–target interaction (DTI) baselines, which typically use amino acid sequence representations for proteins. Notably, several DTI models — such as GNN-DTI, DrugVQA, HyperPCM, and GanDTI — report zero-shot performance on DUD-E around 0.96, surpassing the proposed approach. Furthermore, comparisons to standard few-shot drug discovery models are missing. While it seems plausible that a 3D pocket conformation provides a more informative target representation than using a support set of molecules with SMILES-based features, it remains unclear how well such a 3D pocket representation generalizes across different ligands.

* **(W3 - clarity) - The proposed mechanism for information sharing and representation updates appears ad hoc.** I believe the suggested mechanism may work to some extent, since the pairwise multiplication step allows the integration of ligand and pocket information, and the injected tokens are populated with features derived from this interaction. However, the mechanism appears nonstandard. For instance, wouldn’t a more conventional approach be to feed all token representations into a transformer block, which naturally enables information sharing and representation updates via self-attention? The authors should either reference prior work that adopts a similar strategy or justify why their proposed mechanism is preferable to more established alternatives.

* **(W4 - quality) - Potential risk of data leakage.** As far as I am aware, DrugCLIP was pre-trained on PDBbind, which shares partial data overlap with DUD-E. Consequently, the few-shot setup might be compromised if the base model has already seen a substantial portion of the evaluation data during pretraining. The authors should comment on this potential overlap and clarify whether any filtering was applied to avoid data leakage.

### Questions
* I assume the pocket conformations are derived from crystal structures with bound ligands. However, for many targets, binding pockets only form in the presence of a ligand, and pocket conformations can vary substantially across different docked ligands. Did the authors examine, for the considered protein targets, how much the pocket structures actually vary in reality?

* If such variations are significant, would one still expect the model to identify new hits whose true pocket conformations differ from those represented in the support set? Could a SMILES-based approach potentially be more robust in this case, as it avoids incorporating the bias introduced by an inaccurate or nonrepresentative pocket conformation?

### Soundness
2

### Presentation
2

### Contribution
2
