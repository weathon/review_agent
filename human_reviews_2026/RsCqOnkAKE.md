# The Limits of Fairness Gains Under Scaling in Vision Models

- Avg Score: 3.00
- Decision: Reject
- Scores: 4, 4, 2, 2

## Abstract
Recent advances in computer vision indicate that increasing dataset size and model parameters substantially enhance model performance. Scaling laws derived from these observations provide valuable guidance for the design and optimization of large vision models. However, the impact of scaling on fairness within these models has yet to be systematically investigated. Here we empirically show that scaling model parameters and dataset size can improve fairness for certain protected attributes in downstream tasks. Our results demonstrate that, when using a loss function that jointly optimizes for utility and fairness, there exists a critical threshold in scaling beyond which fairness gains plateau. While scaling enhances fairness for some attributes, it does not eliminate disparities. These results emphasize that fairness in vision models requires more than scaling. Fairness techniques must be incorporated early in model development to address structural disparities and improve outcomes for all groups. This is especially crucial in sensitive domains such as medical imaging, where achieving equal representation and unbiased performance across diverse populations is essential for ethical and effective deployment.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates the relationship between scaling (of model parameters and dataset size) and fairness in vision models. The authors empirically test the hypothesis that the "scaling laws" which lead to improved model performance (utility) will also lead to improved fairness. Using three distinct datasets (CelebA, HAM10000, and CheXpert) and a family of Hierarchical Vision Transformers (Hiera), they find that scaling alone is not a sufficient strategy for mitigating bias.

### Strengths
1. The topic is timing, as more data and larger models are adopted currently.

2. The experiments are extensive.

3. The overall presentation is clear and easy to follow.

### Weaknesses
1. The paper introduces a joint loss function $\mathcal{L}_{total}=\mathcal{L}_{ce}+\lambda\cdot\mathcal{L}_{fair}$. But the specific $\mathcal{L}_{fair}$ used (penalizing the p-norm of the violation vector) is just one of many possible fairness interventions. 

2. A primary limitation of the current study is its focus on scaling the downstream dataset size instead of the pre-training dataset scale.  The paper would be strengthened by considering pre-trained models and datasets, rather than just sub-sampling the downstream task data.

3. The paper's reliance on a single, primary fairness metric limits the scope of its conclusions. Please include more widely adopted fairness notions.

4. The paper correctly notes that model scaling produces divergent fairness trends across datasets (e.g., improving 'Age' fairness on CheXpert while having no effect on HAM10000 , and even worsening 'Mustache' fairness on CelebA ). The primary explanation offered hinges on the concept of "dataset alignment" and a distinction between "local" and "global" representations. This explanation feels incomplete and not convincing enough.

5. Regarding data scaling, the results depend heavily on how the data is sampled. Different sampling methods would create different data distributions, which could easily change the outcome. This makes it hard to know if the reported effects are from scaling or just different data distributions.

### Questions
1. The models were pre-trained on ImageNet-1k. How do the known biases within ImageNet-1k (e.g., geographic, demographic) interact with the downstream tasks? Would they influence the experiment results?

2. The paper suggests unfairness arises from "local representations" while "global representations" are fairer. This is an interesting but feels somewhat speculative and could be defined more concretely. What properties of the model or data make one attribute "local" and another "global"?

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
3

### Summary
This work investigates how scaling, increasing model size and dataset volume, affects fairness in vision models across medical and natural imaging domains. Using Hierarchical Vision Transformers (Hiera) pretrained on ImageNet-1k and evaluated via linear probing on CelebA, CheXpert, and HAM10000, the authors measure fairness through AUROC disparity ratios across demographic groups (e.g., age, gender). They find that scaling improves fairness only up to a point, with diminishing returns beyond a critical threshold, especially when a fairness-aware loss (combining cross-entropy and a fairness penalty) is used. Fairness gains are highly dependent on the protected attribute and dataset domain: gender disparities are consistently low, while age-related disparities vary significantly and are harder to mitigate due to their localized representational nature. Crucially, the study demonstrates that scaling alone cannot eliminate bias, underscoring the need to integrate fairness considerations, particularly during pre-training, to address structural inequities.

### Strengths
1. The paper presents a systematic and empirical investigation into the relationship between scaling, both in model size and dataset volume, and fairness in vision models, addressing a critical gap in the literature.

2. This work provides nuanced analyses of fairness across different protected attributes and data domains.

### Weaknesses
1. One notable weakness of the study is its reliance on linear probing with frozen pretrained backbones, which limits the ability to fully assess how scaling interacts with fairness when models are allowed to adapt more deeply to downstream tasks. While linear probing isolates representation quality from fine-tuning effects, it may underestimate the potential of larger models to mitigate bias through end-to-end adaptation, especially in domains like medical imaging where domain-specific features are critical. This design choice restricts the generalizability of the findings to scenarios involving full fine-tuning or task-specific architectural modifications.

2. The work evaluates fairness primarily through AUROC disparity ratios across predefined demographic groups, which, while common, may overlook other important fairness notions such as equalized odds, demographic parity, or calibration across subgroups. Moreover, the binary treatment of gender (male/female) fails to account for non-binary or transgender identities, reflecting a limitation inherent in the datasets but not critically interrogated by the authors. This narrow framing of protected attributes could mask more nuanced forms of bias, especially in intersectional contexts where multiple identities interact.

3. All models are pretrained exclusively on ImageNet-1k, a natural-image dataset with known biases and limited relevance to medical domains. While this setup helps control for pretraining confounders, it also means the observed fairness behaviors may not generalize to models pretrained on more diverse or domain-aligned data (e.g., large-scale medical image corpora). The study’s conclusions about the insufficiency of scaling might be specific to this particular pretraining regime and may not hold for foundation models trained on broader, more representative datasets.

4. The fairness loss introduced in the second set of experiments is applied only during the downstream linear probing phase, not during pretraining. This raises questions about whether the observed plateau in fairness gains is a fundamental limit of scaling or simply a consequence of applying fairness constraints too late in the pipeline. The paper acknowledges that fairness should be incorporated earlier but does not empirically test this hypothesis, leaving a gap between its recommendations and demonstrated evidence.

### Questions
Please refer to the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper investigates how scaling vision models and datasets affects algorithmic fairness, particularly in medical and natural image domains. Using Hierarchical Vision Transformers of varying sizes and datasets including CelebA, HAM10000, and CheXpert, the authors systematically examine fairness disparities measured by AUROC gaps across protected groups such as age and gender. They find that while scaling improves fairness for some attributes, especially gender, it does not universally mitigate bias, and fairness gains plateau beyond a critical scale even as overall accuracy continues to improve. Incorporating a fairness loss into the objective helps reduce disparities but also exhibits diminishing returns, highlighting that scaling alone is insufficient to achieve equitable performance. The results emphasize the need for fairness-aware objectives during pre-training to address structural and representational biases, particularly for localized attributes like age in medical imaging contexts.

### Strengths
* Strong motivation and relevance: The paper addresses an important and timely question — whether scaling up vision models and datasets inherently improves fairness. It challenges the common assumption that “bigger models are fairer,” providing a necessary empirical correction to this belief.
* Systematic empirical analysis: Scaling laws have rarely been examined from a fairness perspective in vision models, especially in medical imaging. 
* Cross-domain evaluation: By comparing both natural and medical imaging domains (CelebA, HAM10000, CheXpert), the paper reveals that fairness behaviors are dataset- and domain-dependent.

### Weaknesses
* Limited coverage of experimental scope: The analysis is confined to linear probing with frozen backbones, which cannot capture fairness trends that may emerge under fine-tuning. Many modern downstream pipelines (e.g., LoRA, full fine-tuning) could exhibit different scaling–fairness dynamics, so the conclusions drawn from fixed-feature experiments are incomplete.
* Narrow exploration of fairness mechanisms: The study evaluates only one fairness-regularized loss. Given the variety of fairness interventions (e.g., adversarial debiasing, distributionally robust optimization, reweighting), it is unclear whether the reported “fairness plateau” is a general phenomenon or specific to this formulation.
* Weak connection to prior work on bias transfer: Although related studies have analyzed how bias propagates during pre-training and transfer (e.g., Lee et al., Continual Learning in the Presence of Spurious Correlations), this paper lacks a thorough discussion situating its findings within that literature. The related work section should explicitly link its motivation to those prior theoretical and empirical results.
* Scaling range too small for reliable laws: The changes in both model size (from 27M to 671M parameters) and dataset size (10–25 examples per class) are insufficient to meaningfully fit or verify scaling laws, which typically require orders-of-magnitude variation. The limited scale weakens the claim that observed effects represent “scaling laws.”

### Questions
* The authors claim that fairness gains plateau with scaling, suggesting the need for fairness-aware pretraining. However, since all experiments use frozen representations, how would the observed fairness–scaling relationship change under finetuning-based adaptation (e.g., LoRA or full finetuning)?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Paper evaluates 5 models on 3 binary tasks, looking at fairness as measured by ROCAUC ratio between groups and overall performance.

They explore adding a fairness loss that is never defined (eq. 3), which "penalizes the p-norm (*what is value is p?*) of the violation vector (*which definition are you using for this?*)".

The conclusion reached is that fairness is somewhat improved by scaling, but only up to a point with no improvement seen beyond that, and that this point differs from task to task.

### Strengths
The conclusion is likely true, but some familiarity with the fairness literature and existing papers, particularly "Why Is My Classifier Discriminatory?" by Irene Chen, Fredrik D. Johansson, David Sontag 2018 could provide some theoretical analysis of what's going on.

The writing is to a good standard.

### Weaknesses
There's two major concerns here. 

1. The practical experiments are needlessly minimal. Each of the three datasets comes with a wide range of binary labels that are likely to have different fairness properties, and because the approach fixes the backbone and only updates a linear head, a model for every label could have been easily trained for roughly the same computational budget. It's very likely that different labels will have different fairness properties. Particularly on celbA, there's large amounts of label noise for the label "earrings" on men, but not on women, and this will lead to fairness concerns that can not be resolved by better model generalization. 
2. There's no meaningful analysis of the results that tries to show what's going on. This is where looking at the literature, particularly works such as: "Why Is My Classifier Discriminatory?" and coming up with additional experiments/analysis to identify possible causes of the discrimination would go a long way.

Beyond this, the use of auc ratio as a fairness measure is non-standard and can mask a wide range of unfairness. For example, arbitrary equal opportunity or demographic parity violations can occur when two groups have the same AUC curve. Moreover, AUC is pretty unhelpful in deciding if a classifier works well when the label distribution is unbalanced (high AUC can mask a scenario where no threshold results in a classifier with acceptable recall and precision).

### Questions
What are you actually optimizing in the fairness loss? How was this chosen over all of the standard losses in the literature?

Can you justify the use of AUC ratio for fairness?

### Soundness
2

### Presentation
2

### Contribution
1
