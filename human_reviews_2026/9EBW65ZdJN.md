# Structure-Aligned Protein Language Model

- Decision: Reject
- Scores: 6, 2, 6, 2

## Abstract
Protein language models (pLMs) pre-trained on vast protein sequence databases excel at various downstream tasks but lack the structural knowledge essential for many biological applications. To address this, we integrate structural insights from pre-trained protein graph neural networks (pGNNs) into pLMs through a latent-level contrastive learning task. This task aligns residue representations from pLMs with those from pGNNs across multiple proteins, enriching pLMs with inter-protein structural knowledge. Additionally, we incorporate a physical-level task that infuses intra-protein structural knowledge by optimizing pLMs to predict structural tokens. The proposed \textit{dual-task framework} effectively incorporates both inter-protein and intra-protein structural knowledge into pLMs. Given the variability in the quality of protein structures in PDB, we further introduce a \textit{residue loss selection} module, which uses a small model trained on high-quality structures to select reliable yet challenging residue losses for the pLM to learn. Applying our structure alignment method to the state-of-the-art ESM2 and AMPLIFY results in notable performance gains across a wide range of tasks, including a $12.7\%$ increase in ESM2 contact prediction. The data, code, and resulting SaESM2 and SaAMPLIFY models will be released on Hugging Face.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses a well-known limitation of large-scale protein language models (pLMs): while they excel at sequence-based tasks, they lack explicit knowledge of 3D protein structure, which is critical for many biological applications.

The authors propose a dual-task framework to finetune sequence-only pLMs with structural information. The two tasks are:
* Latent-level task: A contrastive learning objective that aligns the latent residue representations from the pLM with corresponding representations from a pre-trained, frozen protein graph neural network (pGNN), specifically GearNet. This is designed to distill *inter-protein* (dataset-level) structural knowledge into the pLM.
* Physical-level task: A prediction task where the pLM's residue embeddings are used to predict discrete structural tokens from Foldseek, reinforcing *intra-protein* (local) structural context.

In addition, the authors also propose a *residue loss selection* module mitigate the impact of low-quality structures in the PDB: they train a smaller reference model on a high-quality subset and then prioritize training the main pLM on residues with a high "excess loss" (i.e., residues that are both reliable/learnable and challenging for the current model).

The authors apply this framework to ESM2 and AMPLIFY, creating "SaESM2" and "SaAMPLIFY." Performance improvements are observed across a wide range of downstream tasks, including structure prediction, mutation effect prediction, and property prediction.

I think the paper is of high quality in terms of conceptualization, experiment design and presentation, but there are certain concerns that need to be addressed.

### Strengths
* The ablation studies are comprehensive. Comparisons are made with different base models, ablated loss components, ablated residue loss selection module, different structure latents and different structure tokens.
* The performance is strong, especially for the contact prediction task.
* The paper is written in an easy-to-understand language. I believe most readers of ICLR can follow the paper easily.

### Weaknesses
* The method is a bit intuitive and is potentially in lack of novelty. Using contrastive losses to enhance representation learning is a standard technique today. The residue loss selection module has certain novelty, but the performance improvement caused by it is marginal (comparing "SaESM2" with "full" in Table 5).
* Some terminologies could be misleading, such as "inter-protein" and "intra-protein".

### Questions
* How do you compare your method with SaProt, in which the structural information is directly modeled in the vocabulary?
* Have you tried using the learned protein representations in more downstream tasks, such as protein-ligand binding prediction?

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
4

### Summary
This paper proposes a contrastive learning-based framework to align protein language models with structural data. They introduce a protein-level and a residue-level CLIP-style post training task. To measure the improvement in 6 structure-related tasks, 2 mutation effect prediction tasks, and 9 property prediction benchmarks; they observe consistent improvements in those tasks.

### Strengths
Most claims are well-supported, particularly when it comes to the performance on the many downstream tasks which is consistent, albeit marginal. Overall the paper is easy to understand - the method is presented in a straightforward way,

### Weaknesses
Because of this marginal gain observed in the downstream tasks, subsamples could be used to obtain variances of the metric estimates to show that those increases are still substantial, and ideally statistically significant. Additionally, stating that the embeddings show better separation as a result of structural alignment based on UMAP projections need to be supplemented with more quantitative methods measuring separability. Also, Figure 2 needs to have some kind of statistical test to show the increase in pseudo-perplexity. There seem to indeed be some increase visually but this needs to be quantified more concretely. 

Presentation: Major missing work includes: BioCLIP https://arxiv.org/abs/2311.18803, S-PLM https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.202404212. Similar ideas are discussed here: https://openreview.net/pdf?id=xDcTugulVV.

In terms of presentation: the embeddings presented are not very concrete and waste considerable space. Comparatively, the other parts of Figure 1 are too small and hard to read. 

A non-exhaustive collection of language improvements:

- The footnote on page 1 is confusing - one could consider rephrasing to avoid biological connotations entirely.
- "structural insights" is a vague term which could be made more precise; one could consdier "embeddings derived from pGNNs"
- Abstract: "including a 12.7% increase" lacks a metric name in which this increase is observed.
- "the common types" in 2.1 is a vague phrase. Consider making this more explicit
- 4.6 "does increases"
- In the introduction, some sentences need to be broken up to improve flow. The paper overall is hard to read in its current state. 
- Ensure all acronyms are explicitly defined on first use.
- There is inconsistent capitalization of e.g. protein language models. 
- R-free needs to be defined, either in a footnote or supplementary material,

The paper makes overall very incremental contributions to the field - the idea of using contrastive learning on proteins is not new. More generally, I wonder if considering the structure and the sequence of the protein as different 'views' of the same underlying manifold is warranted, given one can be predicted from the other as per Anfinsen's principle (and as shown by AlphaFold, ESMFold, SimpleFold, etc.). Additionally, making pLMs "structure-aware" is known to bring advantages but mostly at a small parameter count, as larger models inherently are able to capture underlying structural features. This confines the benefits of structural alignment for pLMs primarily to models at the lower end of the size spectrum.

### Questions
- Given ESM-2 has many model sizes, can we see the effect of structure alignment on multiple model sizes and see if the benefits of structural alignments mostly benefit smaller models? 
- Does the pGNN benefit from being post-trained in this way? 
- Have the authors tried other pGNNs than GearNet? 
- Can the authors clarify if the compute costs associated to performing structure alignment outperform finetuning of a base pLM on a given task with the same compute budget?
- Can the authors comment on the effects of structure alignment on larger ESM-2 model sizes like the 3B and 15B models? I would expect those effects to be more marginal.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors retrain a protein language model (PLM) so that its embeddings receive structural signals. To do this, they train a PLM such as ESM and a frozen structure-based protein embedding model (pGNN) on a few different objectives. These include aligning the sequence and structure-based embeddings, predicting structure tokens from tokenizers such as 3Di, and a specialized loss for low resolution residues. The embedding model is tested on various tasks including mutation effect prediction, contact prediction, and various protein property prediction tasks with solid results. There is also an interesting analysis on the tradeoff between perplexity and the degree of influence from structural information.

### Strengths
* Model architecture is well-motivated, the resolution-based loss is novel and ablations are thorough.
* Improved PLMs have very high application potential.

### Weaknesses
* There is a related work which could bear including in the baseline [1]
* The authors should explore a direct feature fusion approach. It is not clear what is left once even the dual loss is removed and is probably not the strongest form of this ablation. [2] 


1. Chen, Dexiong, et al. "Endowing protein language models with structural knowledge." arXiv preprint arXiv:2401.14819 (2024).
2. Dai, Yimian, et al. "Attentional feature fusion." Proceedings of the IEEE/CVF winter conference on applications of computer vision. 2021.

### Questions
see weaknesses

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
Proteins are sequences of amino acids that fold to a 3d structure. As such, representation learning can be done on the sequence space, for example by masking and predicting parts of the sequence, or in the structure space, by designing a self-supervised task around the 3d-structure of the protein. The authors argue that protein language models trained on the sequence space lack structural knowledge and can be improved by a contrastive learning task using pretrained graph neural networks trained on the structure domain. They call this structure alignment. They use a contrastive method toe align protein language models with structure information and show that these methods perform better than sequence only methods.

### Strengths
- The paper is well written and the methodology is clear.
- The authors tackle an important problem of creating better protein sequence representations that account for the multimodal nature of proteins.
- The specific method of computing the loss seems novel.

### Weaknesses
There are several methods that combine sequence and structure including contrastive learning 

e.g: 
[1] CCPL: Cross Modal Contrastive Protein Learning https://arxiv.org/abs/2303.11783, 

[2] S-PLM: Structure-aware Protein Language Model via Contrastive Learning between Sequence and Structure https://pubmed.ncbi.nlm.nih.gov/37609352/ 

[3] BioCLIP - Contrasting Sequence with Structure: Pre-training Graph Representations with PLMs https://www.biorxiv.org/content/10.1101/2023.12.01.569611v1 

[4] ProTrek that performs contrastive learning across sequence structure and function https://www.biorxiv.org/content/10.1101/2024.05.30.596740v1.full.pdf)

[5] Cross-modality and self-supervised protein embedding for compound–protein affinity and contact prediction https://pmc.ncbi.nlm.nih.gov/articles/PMC9486597

The authors does not distinguish how their work differs from all this prior work and how this is significant. The evaluation also does not consider these works.

### Questions
What is the main difference between this work and all the prior work on structure aware protein language models. How does it compare in terms of methodology and performance?
Could you please look at the tasks covered in these papers (and many others), and make the baselines more comprehensive.

### Soundness
2

### Presentation
3

### Contribution
2
