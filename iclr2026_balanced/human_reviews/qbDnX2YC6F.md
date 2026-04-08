## Human Reviewer 1

### Summary
This paper systematically analyzes Open-Set Recognition (OSR) as a modular, two-stage framework combining Representation Learning (RL) and Postprocessing (PP). Its central contribution is the discovery of "magnitude collapse," a failure mode where popular magnitude-manipulating (MM) methods like Outlier Exposure fail at scale. The authors show this occurs when high similarity between known and auxiliary data causes the model to irreversibly destroy feature magnitude information. They contrast this with the simple, non-MM AddON ($K+1$ classifier), which remains robust. The paper's method is to study the "interaction effects" between different RL and PP components, concluding that small-scale benchmarks are misleading and that robust performance comes from the correct combination of methods (e.g., AddON + PostMax), not a single "best" component.

### Strengths
- Key Insight on Feature Magnitude: The paper's primary strength is its insightful diagnosis of the "magnitude collapse" failure mode. This provides a clear, testable explanation for why popular methods that excel on small-scale benchmarks (like OE) fail on large-scale ones, linking the failure to high semantic similarity between known and auxiliary data.

 - Marginal SOTA Improvement: While the paper successfully identifies a robust combination (AddON + PostMax), the resulting performance gains over other strong, existing combinations (e.g., CE + GHOST or ARPL + GHOST) are often marginal. For instance, on the $P_3$ benchmark (Figure 2, AUOSCR), the proposed AddON+PostMax (79.7) is only a minor improvement over ARPL+GHOST (79.2), suggesting the baseline was already very strong.

### Weaknesses
- Limited Compatibility of AddON: The AddON method's objective creates a representation that is incompatible with many existing OOD detectors. AddON trains the model to produce high-magnitude signals for unknowns (at its $K+1$ node), which directly contradicts the core assumption of many feature-norm-based detectors that expect low-magnitude signals for unknowns. This limits the "modular" combinations to only those PP methods that can be adapted to AddON's specific logic.

 - Oversimplification of the Unknown Space: The AddON method relies on a $K+1$ classifier, which fundamentally models the entire, infinitely diverse "unknown" space using a single prototype vector (the weights for that node). This is a significant oversimplification that likely only works on benchmarks where the unknown classes have limited diversity or happen to be well-represented by the specific auxiliary data used.

- Entangled Evaluation Metrics: The paper's main metrics for OSR performance (AUOSCR and OOSA) entangle unknown detection performance and closed-set classification accuracy into a single score. This can be misleading, as a method could improve in one aspect while regressing in the other. For example, the paper's results in Figure 8 show that AddON has a slightly worse closed-set accuracy than ARPL on ImageNet $P_3$, but its detection (AUROC) is stronger. The final AUOSCR score obscures this trade-off.

### Questions
Please refer to the above weaknesses

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
4

### Confidence
4

---

## Human Reviewer 2

### Summary
This paper presents the first systematic study of the interaction effects between Representation Learning (RL) and Post-Processing (PP) methods in Open-Set Recognition (OSR). The authors introduce a modular, two-stage framework to analyze these combinations, identifying a key failure mode termed "magnitude collapse" that affects certain RL methods at large scale. They propose a simple yet effective baseline (AddON) to mitigate this issue and provide actionable guidelines for combining RL and PP methods.

### Strengths
1. The core contribution—systematically studying the interaction between RL and PP—is highly novel and impactful for the OSR field.
2. The experimental setup is rigorous and thorough. 
3. The (re-)introduction and thorough evaluation of AddON as a powerful and simple baseline is a significant contribution.

### Weaknesses
1. While the paper's title and thesis revolve around "modular gains," the quantitative evidence for the practical significance of these gains is somewhat lacking. 
2. While the paper compares to canonical RL/PP methods (e.g., OE, OpenMax), it omits recent state-of-the-art OSR approaches that may interact differently with PP methods.

### Questions
See the weakness.

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
4

### Confidence
2

---

## Human Reviewer 3

### Summary
The authors propose analyzing the interactions between representation
learning (RL) and post processing (PP, post-hoc methods to add
open-set capabilities) in Open Set Recognition (OSR) in a modular
2-component structure.  Particularly they analyze (feature) magnitude
manipulation (MM) in RL and magnitude-aware (MA) methods in PP.  Some
RL methods use auxiliary data/classes to representation samples of
unknown classes.

For the analysis, they use 5 existing RL (3 used auxiliary data) and 5
existing PP methods (3 are MA) over 2 datasets.  With large-scale
data, they find using auxiliary data does not improve performance.
However, with small-scale data, using auxiliary data generally improves
performance.

To understand why MM methods with auxiliary data degrade in
performance, they analyze magnitude vs performance and find positive
correlation.  Also, increasing similarity (via the P1 to P3 protocols)
between known and auxiliary samples, MM methods learn stronger
relationships.  They find that high similarity between auxiliary and
known classes can degrade the performance of MM methods.  They call
the phenomenon magnitude collapse.  To reduce magnitude collapse, they
find Additional Output Node (AddON) for the auxiliary data is
beneficial.

Without using auxiliary data, they find that RL and PP are independent
and hence any methods from RL and PP can be paired without the
contributions from one being degraded by another.

### Strengths
1.  Investigating the interactions between representation learning
(RL) and post processing (PP) in Open Set Recognition (OSR) in a
modular 2-component structure is interesting.

2.  The analysis indicates that with auxiliary data, similarity between
auxiliary and known classes can degrade Magnitude Manipulation (MM)
methods.  Using AddON with MM can reduce the issue.

3.  Also, the analysis indicates that without auxiliary data, RP and PP
are independent and can be paired without interference.

### Weaknesses
1.  The auxiliary data are intended to represent the unknown classes,
so high similarity to known classes is generally not desirable.
Consequently, the findings are not surprising.

2.  While MM methods degrade when auxiliary data are similar to known
classes, how AddON can reduce the issue is not clear.

3.  Existing methods are analyzed, but new methods are not introduced.

### Questions
1.  Why does AddON reduce magnitude collapse?

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
4

### Confidence
3

---

## Human Reviewer 4

### Summary
The paper studies interaction effects between representation learning (RL) and post-processing (PP) in open-set recognition (OSR) using a modular two-stage framework (RL+PP). Across small-scale (CIFAR+N) and large-scale (ImageNet P1–P3) protocols, it shows that auxiliary-data RL methods, which manipulate feature magnitudes (e.g., OE, ObjectoSphere), can degrade at scale due to a newly identified failure mode, magnitude collapse, feature norms of some known classes shrink toward the origin when auxiliary and known classes are semantically similar, yielding imbalanced class-wise CCR and poorer OSR despite gains on small datasets. Conversely, non-magnitude-manipulating RL (notably AddON, i.e., a K+1 background class) synergizes with magnitude-aware PP (e.g., PostMax, GHOST) to produce additive gains. 

The key contributions are: (i) the first systematic analysis of RL-PP modular interactions, (ii) discovery and analysis of magnitude collapse, and (iii) practical guidance showing AddON + MA-PP as a robust recipe and that small-scale auxiliary-based evaluations are not predictive of large-scale performance.

### Strengths
Regarding originality, this is the first systematic analysis of interaction effects between RL and PP for OSR, framing OSR as a modular two-stage pipeline and introducing the magnitude-collapse failure mode, which explains why magnitude-manipulating RL degrades at scale when auxiliary and known classes are semantically similar.

In terms of quality, the study design is meticulous: five RL × five PP methods are combined, trained (mostly) from scratch to prevent leakage, and evaluated across both small-scale CIFAR+N and large-scale ImageNet P1–P3 protocols, using appropriate OSR metrics and a clear decomposition of RL versus PP gains, enabling fair attribution of effects.

On clarity, the paper clearly formalizes the RL+PP decomposition, defines decision rules, and provides intuitive visual and quantitative evidence (feature-magnitude distributions, regression linking class-wise CCR to feature norms, and heatmaps of AUOSCR/OOSA) that make the interaction story understandable. On significance, the results provide actionable guidance: non-MM RL, such as AddON combined with magnitude-aware PP (PostMax, GHOST), gives additive gains, whereas MM RL paired with MA PP should be avoided at high similarity; moreover, small-scale auxiliary-based wins do not predict large-scale behavior, which has immediate implications for benchmarking and deployment practices in OSR.

### Weaknesses
Although the authors provide a thorough modular evaluation, all large-scale experiments utilize a single ResNet backbone, leaving uncertainty about whether the identified interaction effects and magnitudes persist across other architectures, such as transformers, ConvNeXt, or contrastive self-supervised encoders. Incorporating these architectures would clarify if the observed norm-related behavior stems from the backbone’s feature geometry or from the learning principle itself. 

 Moreover, statistical rigor is lacking: most large-scale results appear single-seeded without confidence intervals or significance tests, and several reported gains fall within plausible noise margins. Multi-seed averages, confidence intervals, and effect-size reporting would strengthen reliability. 

The analysis of magnitude collapse, while intuitively presented, remains purely correlational; providing a geometric or probabilistic explanation of how auxiliary similarity drives norm shrinkage would deepen insight. Additionally, the work could explore hyperparameter sensitivity and mitigation strategies for the collapse phenomenon by systematically varying λ and ξ in OE/OS and visualizing stability regions. Baseline fairness could be improved through compute-normalized comparisons and consistent tuning across RL/PP methods. 

Finally, the practical impact would benefit from an automatic diagnostic that detects early signs of magnitude collapse or misaligned RL–PP pairings during training.

### Questions
1. Do the RL–PP interaction patterns and magnitude behavior persist with modern backbones beyond ResNet (e.g., ViT/DeiT, ConvNeXt, Swin) and with contrastive self-supervised encoders (e.g., MoCo-v3, DINOv2)?
   
2. Are the reported improvements stable across random seeds and training noise, and which results remain significant after controlling for multiple comparisons across the RL×PP grid?

3. What concrete mechanism links auxiliary-known semantic similarity to feature-norm shrinkage and class-wise CCR imbalance, beyond observed correlations?
   
4. Can collapse be prevented or reduced by tuning OE/OS hyperparameters (λ, ξ) or by simple regularizers (norm floors, margin constraints, temperature scaling)?
   
5. Are baseline methods trained and tuned under comparable compute, data budgets, and augmentation/search spaces, and could budget asymmetries explain small gains?
   
6.  Can practitioners detect early during training when a given RL–PP pairing is at risk of magnitude collapse or harmful interaction?

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