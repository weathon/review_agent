# Revisiting Meta-Learning with Noisy Labels: Reweighting Dynamics and Theoretical Guarantees

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 4, 0, 6

## Abstract
Learning with noisy labels remains challenging because over-parameterized networks memorize corrupted supervision. Meta-learning–based sample reweighting mitigates this by using a small clean subset to guide training, yet its behavior and training dynamics lack theoretical understanding. We provide a rigorous theoretical analysis of meta-reweighting under label noise and show that its training trajectory unfolds in three phases: (i) an alignment phase that amplifies examples consistent with a clean subset and suppresses conflicting ones; (ii) a filtering phase driving noisy example weights toward zero until the clean subset loss plateaus; and (iii) a post-filtering phase in which noise filtration becomes perturbation-sensitive. The mechanism is a similarity-weighted coupling between training and clean subset signals together with clean subset training loss contraction; in the post-filtering where the clean-subset loss is sufficiently small, the coupling term vanishes and meta-reweighting loses discriminatory power. Guided by this analysis, we propose a lightweight surrogate for meta reweighting that integrates mean-centering, row shifting, and label-signed modulation, yielding more stable performance while avoiding expensive bi-level optimization. Across synthetic and real noisy-label benchmarks, our method consistently outperforms strong reweighting/selection baselines.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies meta-learning based sample reweighting approaches for learning with noisy labels. The authors note that overparameterized networks easily memorize corrupted labels, and while meta-reweighting methods (which use a small clean validation set to guide weights for training samples) are empirically effective, their behaviour and theoretical dynamics remain under-explored. They provide a rigorous theoretical analysis of the training trajectory of meta-reweighting under label noise, showing it decomposes into three phases: (i) alignment – clean examples are amplified and conflicting noisy examples suppressed; (ii) filtering – weights of noisy examples go to near zero until the clean-subset loss plateaus; and (iii) post-filtering – when the clean subset loss is very small, the coupling term vanishes and discrimination becomes perturbation-sensitive. Based on these insights, they propose a lightweight surrogate algorithm for meta-reweighting (incorporating mean-centering, row-shifting, label-signed modulation) that avoids expensive bilevel optimization yet achieves improved robustness. They validate on synthetic and real noisy-label benchmarks and show superior performance over baselines.

### Strengths
Deep theoretical insight: The authors bridge the gap between empirical success of meta-reweighting and rigorous understanding of its dynamics.

Practical algorithmic outcome: Deriving a surrogate algorithm from theory is valuable for practitioners dealing with noisy data.

Clear experiments: The paper presents empirical results validating the theoretical claims and demonstrating improvement over baselines.

### Weaknesses
Empirical scope: More diverse architectures (e.g., large scale deep networks), more varied noise types (instance‐dependent noise, real-world label errors), and more explicit ablation of assumption breakdown would enhance confidence.

Practical cost/complexity: While the surrogate avoids bilevel optimization, the paper could provide more detail on computational cost, scalability, and comparison to simpler baselines (e.g., early stopping, loss clipping).

Clarification of limitations: More explicit discussion of when and why the method might fail or degrade (e.g., extremely high noise rates, unbalanced classes, small clean subset) would benefit readers.

### Questions
How sensitive is the surrogate algorithm to the size of the clean subset (meta-validation set)? If the clean subset is very small or itself noisy, how does performance degrade?

Could you provide more detailed runtime/compute cost comparisons between the surrogate algorithm and full bilevel meta-reweighting, across different network sizes and noise regimes?

How does the method perform under more challenging noise regimes (e.g., instance‐dependent noise, highly imbalanced labels, open‐set noise) beyond the synthetic or standard benchmarks?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
In this paper, the authors conduct a detailed theoretical analysis to reveal the mechanism of the meta-reweighting-based methods for training with label noise. As shown in the results, the training process can be mainly divided into three stages: an alignment stage, a filtering stage, and a post-filtering stage. Based on the theoretical study, the authors propose a lightweight surrogate for meta reweighting and demonstrate its effectiveness on synthetic and real noise datasets.

### Strengths
1. The theoretical analysis in Section 3 provides a reasonable explanation for the training mechanism of the meta-reweighting framework for the first time.

2. The proposed method, FBR, is a lightweight surrogate for the meta reweighting update, reducing the computational complexity.

### Weaknesses
1. Though the theoretical analysis is good under the settings the authors focus on, it is not comprehensive enough regarding the following aspects:

- The analysis only focuses on the squared loss, but the cross-entropy loss is more widely used for the classification task, which has not been discussed in this work.

- The main tool used for the analysis is the neural tangent kernel (NTK). However, NTK approximation requires strong assumptions about the deep models, which might not hold in practice.

- It seems that the analysis is mainly for the L2RW method without an explicit weighting function, and it is unclear whether the results still hold for MW-Net.

2. The experiments have some inefficiencies:

- There are no studies on the convergence behaviors of the proposed method.

- As mentioned before, the NTK approximation might deviate from the neural networks in practice, and thus it would be useful to test networks with different hyperparameters.

### Questions
The authors should address the issues mentioned in the "Weaknesses".

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
The paper studies training with noisy labels and finds that validation-based sample reweighting overfits late in training: noisy samples get higher weights. It analyzes why this happens and proposes a lightweight surrogate that avoids bilevel optimization to mitigate the issue. The method is evaluated on several noisy-label benchmark datasets.

### Strengths
- The writing and structure are relatively clear.
- The focus on the increasing weights of noisy samples in the later stages of training is interesting and practically relevant.

### Weaknesses
- Unrealistic and unjustified theoretical setup: The paper conducts its theoretical analysis in a binary classification setting and adopts the mean squared error (MSE) as the loss function. This choice is questionable: MSE is not suitable for classification tasks. More importantly, the experiments are performed under multi class classification with cross entropy loss, which makes the theoretical results difficult to generalize or connect to the empirical part. The authors provide no justification for this setup and make no attempt to relate the theory to the experimental conditions.
- Insufficient rigor and validation in explaining the key mechanism: The authors attribute late stage overfitting to “approximation error.” While this explanation is intuitively plausible, the paper lacks any systematic analysis of when and under what conditions this phenomenon occurs. For instance, overfitting of this kind is often not observed under large learning rates—can the proposed theory explain this behavior? The current discussion is coarse and lacks a quantitative relationship between key variables (e.g., learning rate, training stage) and various sources of error(e.g. approximation error, estimation error). No empirical verification is provided, such as curves of approximation error over time or sensitivity to learning rate. As a result, the core mechanism remains speculative and weakly supported.
- Main claims are not supported by direct evidence: The paper claims that the proposed method effectively avoids late stage overfitting, yet it only shows weight dynamics of other methods (L2W) and omits those of the proposed approach. If the ability to avoid weight overfitting is the main contribution, this missing evidence is critical. Furthermore, the method is described as a lightweight surrogate of L2W, but no complete L2W baseline is included in the experiments. It is thus unclear whether the reported improvement truly results from avoiding overfitting or from other unreported strategies or hyperparameter adjustments. If additional techniques are involved, ablation studies are essential to verify which component actually contributes to the improvement.
- Empirical evaluation is far below current community standards：The experimental evaluation is highly limited. The paper only tests on a few datasets and includes very few baselines. It does not consider instance dependent noise, nor does it evaluate on large scale real noisy datasets such as WebVision. The comparison set includes only a few classic baselines and omits recent SOTA methods such as Modified-DivideMix[1], L2B[2] and DCD[3]. Consequently, the experimental evidence is not convincing and would not meet the current standards in the noisy label learning or meta reweighting literature.
The phenomenon addressed in this paper is practically relevant, but the theoretical setup is oversimplified and disconnected from the experiments, the main explanation lacks rigorous validation, and the empirical evaluation falls well below the expectations of the field. Overall, the work does not provide sufficient evidence to support its claims.

[1] Yuan S, Li X, Miao Y, et al. Combating noisy labels by alleviating the memorization of dnns to noisy labels[J]. IEEE Transactions on Multimedia, 2024. 

[2] Zhou Y, Li X, Liu F, et al. L2B: Learning to bootstrap robust models for combating label noise[C]. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024: 23523-23533. 

[3] Mu C, Qu Y, Yan J, et al. Meta-Learning Dynamic Center Distance: Hard Sample Mining for Learning with Noisy Labels[C]. Proceedings of the IEEE/CVF International Conference on Computer Vision. 2025: 415-425.

### Questions
See the weakness

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper is about a theoretical analysis of meta learning sample reweighting under label noise. The paper provides also a lightweight surrogate for meta reweighting that integrates mean centering, row shifting, and label signed modulation. The analysis show that training is   characterized by three distinct phases, i) an alignment phase where clean samples are upweighted and noisy samples downweighted; ii) a filtering phase where weights polarize and the validation loss converges; iii) a post filtering phase where the discriminatory signal weakens. Based on the insights, the paper reports a lightweight alternative called Feature Based Reweighting that avoids expensive bilevel optimization while maintaining the essential mechanism of signed similarity weighted aggregation.

### Strengths
+ a good theoretical contribution where a rigorous analysis of meta reweighting dynamics using neural tangent kernel theory is reported. The three phase characterization offers insights into when and why meta reweighting succeeds or fails, addressing a significant gap in understanding these methods. The proposed algorithm is well motivated by the theoretical analysis.
+ The experiments are comprehensive and cover standard synthetic and realistic scenarios, with consistent improvements.
+ Well presented.

### Weaknesses
- Assumption 3 (and even relaxed assumption 4) requires strong kernel separation between classes with specific sign patterns. While the paper provides empirical evidence on binary MNIST, the assumption may not hold for complex, multi class datasets.
- While the theoretical analysis is novel, the algorithmic components are relatively standard techniques.
- The analysis is performed using binary classes, but the practical algorithm is on multi class classification. The paper does not discuss with sufficient bridging the two parts.
- Experiments are missing hyperparameters analysis

### Questions
- Can you further discuss the bridge between the two parts of the paper?
- How does the method perform with class imbalance, a common co-occurring issue with label noise?

### Soundness
3

### Presentation
3

### Contribution
3
