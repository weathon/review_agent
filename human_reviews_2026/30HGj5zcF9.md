# GenDrugCLIP: A Generation-Augmented Framework for ContrastiveDrug-Target Representation Learning

- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Virtual screening (VS) has become an indispensable component of early drug discovery, aiming to identify potential ligands for a given protein target. While CLIP-style methods (e.g., DrugCLIP) have emerged as a powerful solution by enabling efficient compound retrieval through drug-target representation alignment, current models face two fundamental challenges: (1) the scarcity of true binding data for training limits coverage of diverse binding modes, and (2) the use of trivial negatives—molecules binding to other pockets—leads to a significant train-test domain gap. To address these challenges, we introduce GenDrugCLIP, a novel generation-augmented framework that repositions structure-based drug design (SBDD) models as controllable data engines. GenDrugCLIP implements a Generate-Filter-Score-Select pipeline to construct target-aware pseudo positives and hard negatives for triplet contrastive learning. Our approach not only expands the chemical space but also prevents the model from relying solely on trivial negatives. Extensive experiments on three benchmarks demonstrate that GenDrugCLIP achieves state-of-the-art performance, outperforming DrugCLIP by +7.66% in BEDROC and +7.45 in early enrichment on the DUD-E benchmark. Our work highlights the untapped potential of SBDD models as powerful data engines for representation learning, opening a new paradigm for data-efficient drug discovery.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces GenDrugCLIP, an extension of DrugCLIP, which incorporates a structure-based drug design (SBDD) model, MolCRAFT, to generate molecule samples conditioned on the target. These generated samples are then filtered and used as pseudo positives and negatives for data augmentation, aiming to enhance the training of DrugCLIP. While the method is conceptually reasonable, it achieves only marginal improvements on virtual screening benchmarks.

### Strengths
1. The idea of using a generative model to create pseudo hard negative samples for virtual screening is both interesting and conceptually sound. 
2. The filtering strategy is thoughtfully designed and appears to be well-justified.
3. The proposed method leads to some kind of improvements over baselines.
4. The improvement over the “with trivial negatives” setting in the ablation study is encouraging, suggesting that generating negatives via a generative model is more effective than using random samples, although more details should be included.

### Weaknesses
1. The major concern of this paper is the relatively limited overall performance improvement. Both the enrichment factor and BEDROC show less than 10% gain, which may not be sufficient to convincingly demonstrate the effectiveness of the proposed method.
2. The overall techniqual depth and contribution of this paper is limited. It is based on an existing model for virtual screening, and use an existing  SBDD model for data augmentation. 
3. The paper currently explores only one SBDD model: MolCRAFT. It is better to try more generative models and compare the performance.
4. The ablation study should be more detailed and thorough. For example, It is unclear whether the “trivial negatives” were also passed through the same filtering process, which may be a key factor behind the method’s success. If the random negatives were not filtered similarly, the comparison could be misleading. Clarifying this would strengthen the credibility of the analysis.
5. While the inclusion of visualizations is appreciated, Figure 3b provides limited insight. It only shows one case where GenDrugCLIP outperforms the original DrugCLIP on a specific target, without offering any analysis of the underlying causes. Although cherry-picking cases can illustrate potential improvements, a deeper explanation of why the improvement occurs is needed to enhance the informativeness of the result.

### Questions
see weakness

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a generation augmented contrastive learning framework tGenDrugCLIP, to address two key limitations of CLIP-style virtual screening: (1) sparse true binding data and (2) reliance on trivial negatives. By repurposing SBDD models as controllable data engines, it introduces a Generate-Filter-Score-Select pipeline to create target-aware pseudo positives and hard negatives for triplet contrastive learning.

### Strengths
1. This study successfully applies Structure-Based Drug Design (SBDD) to new practical scenarios(Virtual Screening), yielding positive effects.

2. The paper is well structured and easy to follow.

3. The paper evaluates on three benchmarks: DUD-E, DEKOIS 2.0, and LIT-PCBA. On DUD-E, GenDrugCLIPVina achieves 45.42 % BEDROC (+7.66 % over DrugCLIP) and EF0.5 % of 35.63 (+7.45 over DrugCLIP). It also sets new state-of-the-art on DEKOIS 2.0 (BEDROC 49.12 % ) and improves over DrugCLIP on the challenging LIT-PCBA dataset (BEDROC 5.51 % vs. 3.78 %).

### Weaknesses
Concerns:  

1. AutoDock Vina’s known biases (e.g., molecular-weight preference) may propagate into the pseudo-labels and cap generalization.  

2. The paper employs two scoring schemes DrugCLIP similarity and Vina affinity, but provides insufficient analysis of why, in different settings, DC performs better (DEKOIS 2.0) while Vina prevails (LIT-PCBA).

3. The optimal mixing ratio of generated data must be carefully tuned—more is not always better, limiting the overall gain.  

4. SBDD model choice is limited to MolCRAFT; no ablation on alternatives (DiffSBDD, Pocket2Mol, etc.) or sensitivity analysis is provided. 
 
5. Core contribution is standard data-augmentation via generation, with incremental novelty.

### Questions
Refer to the weakness.

### Soundness
2

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
This paper proposes a data augmentation method for protein pocket-small molecule contrastive learning called GenDrugCLIP, using the generative ability of SBDD models like MolCRAFT. The method addresses challenges such as the scarcity of true binding data and the use of trivial negative samples by generating target-aware pseudo positives and hard negatives. Experimental results show that GenDrugCLIP outperforms DrugCLIP on multiple benchmarks.

### Strengths
1. The idea of using generated model to generated synthetic data for VS task is a great idea, as the real data is hard to get because of the expense of wet lab exps.
2. This paper mentioned a important problem is that the negative sample need to be hard to push the model learn the really important pricinples.

### Weaknesses
The main contribution of this paper is focused on the Data Augmentation method, which may not be a very broad topic, and whether it is sufficient for publication at ICLR might be worth discussing. However, I think it’s still quite ok, as data augmentation methods are currently much needed in this field.

For others, see the questions.

### Questions
1.	In your motivation, you mentioned that a key challenge is the limited number of true active ligands for each target. Why is this considered a critical issue when modeling pocket-ligand interactions?
2.Have you experimented with combining both the DC score and Vina score filters? If applied simultaneously, do you think this would improve the results?

3.	The baseline results on DUD-E seem to differ from those reported in previous papers, such as DrugCLIP. Is this due to dataset deduplication based on UniProt IDs or another factor?
4.	MolCRAFT is trained using CrossDocked, which is not deduplicated from the test sets. Could using MolCRAFT to generate training data result in data leakage?
5.	It’s not clear how much useful information is provided by MolCRAFT versus the multiple filters. I recommend adding an ablation study where random molecules are first sampled (with or without matching ligand atom numbers to the reference ligand), then subjected to Vina docking to get their poses. Afterward, apply the same property and DC or Vina score filters to select the synthetic data. Would this approach potentially enhance DrugCLIP’s performance?
6.	The distribution of the generated molecules differs from that of real molecules. Could this pose a problem for your approach?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper introduces GenDrugCLIP, a generation-augmented framework for contrastive drug–target representation learning. The key idea is to repurpose structure-based drug design models as controllable data engines that can generate target-aware molecules. GenDrugCLIP employs a Generate–Filter–Score–Select pipeline to produce pseudo positives (high-scoring molecules) and hard negatives (low-scoring molecules) for triplet-based contrastive learning. Experiments on three benchmarks show that GenDrugCLIP outperforms existing methods such as DrugCLIP.

### Strengths
- Creative reuse of SBDD models as generative data engines, bridging generative modeling and contrastive learning.  
- Demonstrated strong empirical gains over DrugCLIP across multiple benchmarks.  
- Conceptually broadens the role of generative models in drug discovery from candidate generation to data augmentation for representation learning.

### Weaknesses
- Compared to DrugCLIP, the methodological novelty is limited. The main contribution appears to be the generation of an augmented dataset rather than a fundamentally new learning framework.  
- Only one SBDD method is used for data augmentation, and it is unclear why this specific method was chosen or how robust the approach would be if alternative SBDD generators were applied.  
- ground truth positive compounds might not even pass a strict filter like the one the authors implemented here, potentially introducing a data distribution shift between the augmented and original datasets.  
- The paper does not discuss how sensitive or robust the model is to imperfect or less curated filtering procedures, which could affect generalization in practical scenarios.

### Questions
- GenDrugCLIP consistently outperforms both DrugCLIP and EquiScore across most metrics, but the differences in AUROC are relatively small. Does this indicate that GenDrugCLIP primarily improves early enrichment rather than overall ranking performance?

### Soundness
3

### Presentation
3

### Contribution
2
