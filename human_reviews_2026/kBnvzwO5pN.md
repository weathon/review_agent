# FaLW: A Forgetting-aware Loss Reweighting for Long-tailed Unlearning

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Machine unlearning, which aims to efficiently remove the influence of specific data from trained models, is crucial for upholding data privacy regulations like the ``right to be forgotten". However, existing research predominantly evaluates unlearning methods on relatively balanced forget sets. This overlooks a common real-world scenario where data to be forgotten, such as a user's activity records, follows a long-tailed distribution. Our work is the first to investigate this critical research gap. We find that in such long-tailed settings, existing methods suffer from two key issues: Heterogeneous Unlearning Deviation and Skewed Unlearning Deviation. To address these challenges, we propose FaLW, a plug-and-play, instance-wise dynamic loss reweighting method. FaLW innovatively assesses the unlearning state of each sample by comparing its predictive probability to the distribution of unseen data from the same class. Based on this, it uses a forgetting-aware reweighting scheme, modulated by a balancing factor, to adaptively adjust the unlearning intensity for each sample. Extensive experiments demonstrate that FaLW achieves superior performance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
FaLW addresses machine unlearning when forget sets are long-tailed. It quantifies instance-level unlearning deviation by comparing a sample’s confidence to unseen-class probability distributions, then reweights forgetting loss with a class-aware balance factor, mitigating over-forgetting and under-forgetting and improving accuracy and MIA performance.

### Strengths
Strengths:

1. This paper first works on the long-tailed forgetting data unlearning
2. This paper proposes a new plug-and-play instance-wise unlearning method

### Weaknesses
Weaknesses:

1. This paper lacks an explanation of the differences between the unlearning on the general datasets and the long-tailed datasets.
2. This paper assumes that the class-conditional prediction is a Gaussian distribution, which is too strong. This paper lacks the validation of this key assumption
3. This paper does not conduct an effective ablation study for each component in Eq. 3.
4. The results do not contain std
5. This paper only uses VGG16 for experiments. More models should be tried.

### Questions
Please refer to the weaknesses

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
4

### Summary
The paper introduces FaLW (Forgetting-aware Loss Weighting), a novel loss design intended to improve the reliability and controllability of machine unlearning. The key idea is to dynamically reweight forgetting and retaining objectives based on the model’s forgetting confidence and feature-space similarity, thus preventing over-forgetting and instability observed in prior works. The authors derive an adaptive weighting function grounded in information-theoretic uncertainty and validate their approach on several image classification benchmarks using standard unlearning baselines.

### Strengths
1.	The paper is, to the best of my knowledge, the first to explicitly formulate long-tailed forget sets (not long-tailed training data) and to show that existing approximate unlearning methods exhibit heterogeneous and skewed unlearning deviations under this realistic setting. This is an underexplored but practical scenario.
2.	The proposed FaLW is conceptually simple, instance-wise, and orthogonal to most gradient-based unlearning pipelines. It can be adopted with minor code changes.
3.	The direction-aware weighting derived from per-class unseen distributions provides a principled way to decide whether to increase or decrease forgetting pressure for each sample, which directly matches the identified deviation phenomena.
4.	The paper is well-written and logically consistent, with clear motivation method experiment alignment.

### Weaknesses
1.	Limited theoretical justification – while the adaptive weighting function is motivated by uncertainty, the derivation remains heuristic. The paper lacks formal analysis or convergence guarantees explaining why the proposed weighting yields more reliable unlearning.
2.	Ablation insufficiency – although the paper reports a few ablations, it does not disentangle the specific contributions of the uncertainty term versus the similarity term in the weighting function.
3.	Lack of comparison with recent conformal or calibration-based approaches – given the growing body of work, FaLW should also be compared in terms of uncertainty calibration and reliability metrics to position itself clearly.
4.	Potential instability in extreme regimes – adaptive weighting can introduce oscillations or under-forgetting when uncertainty estimates are unreliable, but the paper does not report sensitivity or failure cases.

### Questions
See the comments above

### Soundness
3

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
2

### Summary
This paper addresses the long-tailed nature of the forgotten data distribution, which is an important and underexplored problem in machine unlearning. The authors identify two key issues in existing methods: Heterogeneous Unlearning Deviation (HUD) and Skewed Unlearning Deviation (SUD), which leads to biased forgettiing and uneven performance across samples. To mitigate these challegnes, they propose a dynamic loss reweighting strategy that adaptively adjusts learning signals based on forgetting difficulty. Experiments on multiple benchmarks demonstrate that the proposed method effectively reduces unlearning bias while maintaining model utility.

### Strengths
1. The paper highlights an under-explored but practically important phenomenon in machine unlearning that the forgotten data often follows a long-tailed distribution. The problem is important and the motivation of the work is clear. 

2. The formulation of Heterogeneous Unlearning Deviation (HUD) and Skewed Unlearning Deviation (SUD) provides a structured way to analyze performance degradation in unlearning systems, which offers a useful framing for future work.

3. The proposed FaLW is simple but effective. The paper evaluates multiple metrics and the results consistently show that the proposed achieves better balance between unlearning completeness and retained-task performance.

4. The paper is well-organized and easy to follow.

### Weaknesses
1. Lack of empirical validation for plug-and-play claim: Although the proposed FaLW (Forgetting-Aware Loss Reweighting) is described as a plug-and-play solution, the paper only evaluates FaLW as a standalone framework. There are no experiments demonstrating its integration into other existing unlearning methods.

2. Limited analysis of the identified issues HUD and SUD: The paper identified two important issues: Heterogeneous Unlearning Deviation (HUD) and Skewed Unlearning Deviation (SUD) as key motivating factors. Howerver, these notions closely resemble exiting sideas such as sample difficulty bias and class imbalance bias from the broader learning literature. The paper does not sufficiently differentiate its definitions from these established concepts, nor provides diagnostics or ablations to examine HUD and SUD independently.

3. Lack of isolated analysis for HUD and SUD: It remains unclear whether HUD and SUD always co-occur or can arise independently. The experiments treat them as jointly existing phenomena under the long-tailed setting, but no analysis is presented to determine their individual effects on unlearning performance.

4. Lack of discussion on design choices and hyperparameter sensitivity: While the method introduces a dynamic loss reweighting mechanism, the paper provides limited justification for specific design choices, like the form of the weighting function,or the dynamic adjustment rule. A more detailed discussion on hyperparameter selection, or sensitivity analysis would strengthen the paper.

### Questions
1. How easily can FaLW be incorporated into other unlearning frameworks? Would additional tuning or architecture changes be required, or can it indeed serve as a simple plug-and-play loss modification?

2. How do the proposed HUD and SUD differ formally from sample difficulty bias and class imbalance bias that have been widely discussed in prior works? Are there conditions under which HUD/SUD reduce to these known phenomena?

3. Have the authors considered running controlled experiments to isolate HUD and SUD individually (e.g., a balanced dataset for HUD-only, or uniform sample difficulty for SUD-only) to better understand their respective contributions?

4. What is the rationale for the specific dynamic weighting formulation and update rule? Have alternative forms been tested or theoretically compared?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper tackles the problem of machine unlearning under long-tailed data distributions in the forget set. It demonstrates that in real-world scenarios, the data to be forgotten can be highly skewed. It identifies two interesting  phenomena while unlearning in this situation, Heterogeneous Unlearning Deviation and Skewed Unlearning Deviation and demonstrates them.  To address these issues, the paper  propose FaLW (Forgetting-aware Loss Weighting), an instance-wise dynamic loss reweighting method. The weighting considers model’s current prediction confidence for a sample to the distribution of prediction confidences on unseen data of the same class.  The authors  conduct experiments on multiple image classification benchmarks (CIFAR-10, CIFAR-100, Tiny-ImageNet) demonstrate the effectiveness of FaLW.

### Strengths
The paper studies a novel problem arising in the context of unlearning and proposes a novel solution to address this. The problem addressed is quite relevant and practical. 

The paper  demonstrates empirically the unlearning deviation problem under long tailed distribution setups,  and  defines  the problem clearly, proposing two kinds of unlearning deviation.   

The proposed FaLW is sound, addressing the identified problem to the extent possible. 

Srong empirical results on several real world data sets to demonstrate the effectiveness of the proposed methodology.  

The presentation of the paper is clear and well written.

### Weaknesses
The methodology addresses the problem to a good extend but suffers from some drawbacks

1. The requirement to have unseen data points from the same class might be impractical.-  in practice such auxulliary data may not be available
2. FaLW does not provide a formal guarantee or certification that the influence of the forget set is removed 
3. The definition of unlearning deviation in the paper involves a threshold $\tau_i$, but in the proposed weighing scheme the paper seems to have ignored this. 
4.  The choice of Nornal distribution to model the distribution of predicitive probabailities is not clear.  Why not use a distribution with support in [0,1] which is more appropriate to model distributions.

### Questions
1.  How does this approach scale to a setup where we want to unlearn a particular class rather than unlearning a particular point from a given class.  
2. The paper describes FaLW as plug-and-play, but seems to have demonsrated it using only one specific unlearning approach. how can  FaLW can be used in practice an any  generic unlearning approach ?

### Soundness
3

### Presentation
3

### Contribution
3
