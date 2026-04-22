# Greater than the Sum of Its Parts:  Building Substructure into Protein Encoding Models

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Protein representation learning has achieved major advances using large sequence and structure datasets, yet current models primarily operate at the level of individual residues or entire proteins. This overlooks a critical aspect of protein biology: proteins are composed of recurrent, evolutionarily conserved substructures that mediate core molecular functions. Despite decades of curated biological knowledge, these substructures remain largely unexploited in modern protein models. We introduce Magneton, an integrated environment for developing substructure-aware protein models. Magneton provides (1) a large-scale dataset of 530,601 proteins annotated with over 1.7 million substructures spanning 13,075 types, (2) a training framework for incorporating substructures into existing models, and (3) a benchmark suite of 13 tasks probing residue-, substructure-, and protein-level representations. Using Magneton, we develop substructure-tuning, a supervised fine-tuning method that distills substructural knowledge into pretrained protein models. Across state-of-the-art sequence- and structure-based models, substructure-tuning improves function-related tasks while revealing that substructural signals are complementary to global structural information. 
The Magneton environment, datasets, and substructure-tuned models are all openly available.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces Magneton, an integrated environment for developing substructure-aware protein representation models. Magneton comprises three components:

(1) a large-scale dataset of 530 k proteins annotated with 1.7 M substructures spanning 13 k types;

(2) a training framework that incorporates these substructures into existing sequence- or structure-based encoders; and

(3) a benchmark suite of 13 tasks covering residue-, substructure-, protein-, and interaction-level evaluations.

Building upon this environment, the authors propose substructure-tuning, a supervised fine-tuning strategy that distills substructural information into pretrained protein models. Experiments across six state-of-the-art encoders (150 M–650 M parameters) show consistent ≈5 % improvements in function-related tasks (e.g., EC and GO prediction) and demonstrate that substructural signals complement global structural information.

### Strengths
1. The paper identifies an under-explored gap—modern protein language models neglect conserved substructures, despite their well-known biological importance. The proposed Magneton environment and “substructure-tuning” paradigm offer a novel, systematic way to integrate decades of curated biological knowledge into modern representation learning.

2. The dataset construction and methodology are rigorous. The authors curate and filter substructures from major biological databases (Pfam, InterPro, DSSP, SwissProt), perform consistent train/validation/test splitting, and benchmark on 13 tasks. The experimental setup spans multiple model architectures and modalities, lending credibility to the reported gains.

3. The manuscript is well-written and logically structured. Figures 1–3 clearly illustrate the motivation, architecture, and effects of substructure-tuning.

4. The work lays a foundation for multi-scale protein modeling by linking residue-, motif-, and domain-level representations. This contribution is both scientifically meaningful and practically useful, likely to influence future developments in protein representation learning, function prediction, and design.

### Weaknesses
1. Substructure information is only injected through supervised fine-tuning; the model architecture itself remains unchanged. As the authors note, this approach can be brittle under task-specific fine-tuning and may not fully exploit hierarchical dependencies. Exploring architectural biases (e.g., graph-hierarchical encoders) would strengthen the contribution.

2. The dataset filters out substructures with < 75 occurrences. While this simplifies training, it biases learning toward frequent structural motifs and limits generalization to rare but functionally critical substructures.

3. Substructure-tuning improves global functional prediction but sometimes degrades residue-level metrics. This trade-off warrants deeper analysis—e.g., are low-level features being overwritten, or are substructure objectives misaligned with certain downstream signals?

4. The paper could better contextualize gains against concurrent structure-aware fine-tuning strategies (e.g., S-PLM, ESM-S). A direct quantitative comparison or ablation with those would clarify the incremental benefit of substructure-tuning.

5. While performance tables are comprehensive, confidence intervals or variance over multiple runs are missing, which is important given modest absolute improvements.

### Questions
1. Could the authors discuss whether hierarchical or graph-based encoders (e.g., multi-resolution GNNs) might integrate substructures more stably than fine-tuning alone? A comparison would help isolate the source of gains.

2. Have the authors evaluated whether the tuned models can generalize to rare or unseen motifs not present in the training subset? Such analysis would demonstrate the learned “substructure awareness” beyond supervised classes.

3. Since different substructure classes use separate heads, how correlated are their learned embeddings? Could joint attention or shared bottlenecks yield better cross-scale consistency?

4. One potential advantage of substructure-aware representations is improved interpretability. Can the authors provide examples where substructure-tuned embeddings correlate with known functional motifs or mechanistic insight?

### Soundness
3

### Presentation
3

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
Protein models are typically supervised with individual residues or full structures. However, there are various labeled substructures of active sites, binding sites, etc. that have not been previously used to train models. This paper introduces a dataset and framework to train sequence or structure-sequence models on these datasets. They show that the fine-tuned models improve performance.

### Strengths
Substructures are intuitive abstractions that captures various levels of protein modeling that are not immediately apparent from either structure or sequence.
The dataset developed by processing Swissprot is large and diverse, and is often overlooked in training protein models.

### Weaknesses
While there is some analysis of the dataset in Table 1, it is insufficient for a paper which proposes a dataset.
The results are impressive but raises concerns as to how much of the test set's labels are related or derived from the SwissProt annotations.
The authors finetune the model using elastic weight consolidation (EWC). However, it is not clear empirically how much EWC affects downstream performance.

### Questions
1. Beyond Table 1, what does the dataset consist of? What do the unique types consist of?
2. How do the authors address potential data leakage between the training data present in the Magneton substructure dataset and the evaluation datasets? Are there sequence similarity cutoffs to filter the training set to ensure that the training and testing proteins are sufficiently different?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents Magneton, a framework that brings protein substructure knowledge into representation learning in a systematic way. The authors build a large annotated dataset with over half a million proteins and around 1.7 million substructures from six categories. They propose a substructure-tuning method that fine-tunes pretrained protein language models on multi-scale substructure classification and test its impact on 13 downstream tasks at different levels, from residues to protein interactions. The results show consistent improvements on function-related tasks such as EC, GO-MF, and thermostability, while the effects on localization and residue-level predictions are neutral or slightly negative. Overall, the work shows that substructure information complements global structural signals and offers a reproducible benchmark for studying hierarchical protein representations.

### Strengths
The paper takes a fresh approach to protein representation by explicitly modeling mid-level substructures that connect sequence and overall structure. While earlier work has explored structural supervision, few have looked at substructure information at this scale or systematically tested its impact. The authors run solid and wide-ranging experiments across several model families, including ESM2, SaProt, Cambrian, and ProSST, covering six types of substructures and thirteen downstream tasks. The dataset is clearly described and easy to reproduce, and the figures and tables make the framework intuitive to follow.

Overall, the work is clearly written and well-executed. It provides a practical and scalable benchmark that shows how substructure-level fine-tuning can bring meaningful gains on function-related tasks, even for models that already use 3D information. The results highlight the value of adding explicit substructure knowledge to protein models and point toward new directions for improving interpretability and representation learning in protein science.

### Weaknesses
Lack of theoretical motivation:
The paper shows that substructure-tuning works, but it doesn’t really explain why. There’s no clear reasoning or framework to describe why it helps with function prediction but hurts residue-level performance. The discussion feels more like observation than real analysis.

Unbalanced and oversimplified supervision:
The six types of substructures are not only uneven in number—some, like catalytic or binding sites, are much rarer—but also differ in biological meaning. Many of these categories, such as catalytic sites, cannot be treated as simple binary labels because they involve diverse biochemical functions rather than just “active vs inactive.” Similarly, tasks like PPI prediction or contact prediction represent complex relational signals that go beyond binary classification. The paper filters and simplifies these cases, but it doesn’t assess how this affects the model’s ability to capture functional diversity or relational structure.

### Questions
Mechanistic validation:
Could the authors include attention-map or embedding-similarity analyses (for example, t-SNE or clustering of substructure embeddings) to check whether substructure-tuning actually makes the model group residues or motifs by function? Comparing pre- and post-tuning representations on catalytic or binding site residues would make the mechanism more convincing.


Generalization and transferability:
To show robustness, the authors could run few-shot or zero-shot transfer experiments — for instance, testing on unseen protein families or on new substructure types not used during tuning. Measuring how performance drops as family divergence increases would make the generalization claims stronger.


Functional specificity of catalytic sites:
Instead of treating all catalytic sites as one binary label, the authors could run an additional experiment focusing on specific catalytic functions (for example, hydrolases vs oxidoreductases). This would test whether substructure-tuning helps the model distinguish between different catalytic mechanisms rather than just recognizing “catalytic vs non-catalytic” regions.

### Soundness
3

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
The paper introduces Magneton, a new environment that incorporates protein substructures into protein language models. It contains (1) a large-scale dataset annotated with million-scale substructure annotations, (2) a suite of evaluation tasks,  and (3) a training method called substructure-tuning, a supervised fine-tuning method for distilling substructural information into pretrained models. Substructure-tuning shows substantial improvement on models’ ability to represent protein function.

### Strengths
1. Compiling curated substructure annotations at scale and providing packaging code + benchmark tasks is a valuable community resource. 

2. The suite of 13 tasks spans from residue, substructure, protein, and interaction levels. 

3. The paper reports consistent improvements on function-related tasks after substructure tuning, suggesting substructural priors are useful.

### Weaknesses
1. The authors restrict to substructures that occur at least 75 times in the SwissProt dataset, corresponding to retaining only the top 10% most frequently occurring domains. Although the histogram in Appendix A supports this, more experiments on this may be beneficial. Also, this practice might limit claims about the general utility of rare/novel substructures.

2. Have you tried other structure pooling techniques more than the mean pool?  E.g. learnable aggregation, or attention?

### Questions
Please refer to the weakness section.

### Soundness
3

### Presentation
3

### Contribution
3
