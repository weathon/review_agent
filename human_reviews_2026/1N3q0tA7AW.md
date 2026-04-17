# Sequential Subspace Noise Injection Prevents Accuracy Collapse in Certified Unlearning

- Decision: Reject
- Scores: 6, 4, 6, 4

## Abstract
Certified unlearning based on differential privacy offers strong guarantees but remains largely impractical:
the noisy fine-tuning approaches proposed so far achieve these guarantees but severely reduce model accuracy.
We propose sequential noise scheduling, which distributes the noise budget across orthogonal subspaces of the parameter space instead of injecting it all at once. This simple modification mitigates the destructive effect of noise while preserving the original certification guarantees. We extend the analysis of noisy fine-tuning to the subspace setting, proving that the same $(\varepsilon,\delta)$ privacy budget is retained. Empirical results on image classification benchmarks show that our approach substantially improves accuracy after unlearning while remaining robust to membership inference attacks. These results show that certified unlearning can achieve both rigorous guarantees and practical utility.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces Sequential Subspace Noise Injection to preserve formal certified unlearning guarantees while limiting utility loss. To do this SSNI partitions the model parameter space into orthogonal subspaces and applies noisy fine-tuning sequentially across these blocks. This reduces per-step distortion compared to injecting noise globally. The authors extend the certification framework analysis of noisy fine-tuning to the block-wise setting. And in experiments, SSNI reduces post-unlearning accuracy drop compared to certified baselines for both random and class-wise deletions on MNIST and CIFAR-10 while being robust against membership inference attacks.

### Strengths
* SSNI partitions the parameter space into orthogonal subspaces and injects noise sequentially. This reduces per-step distortion and avoids the accuracy collapse from inserting noise all at once to all parameters.

* Empirical results show that SSNI maintains significantly higher post-unlearning accuracy compared to baseline certified methods

* The paper extends prior theory to the block-wise setting and retains the same privacy budget as standard noisy fine-tuning.

### Weaknesses
* The method may be memory intensive, but the issue can be reduced by 1) splitting the orthogonal basis layer-wise and 2) using a permutation matrix instead (still layer wise).

### Questions
* In practice models with many layers (such as LLMs) may require smaller blocks to reduce memory for the orthogonal subspace decomposition. How does the unlearning - utility tradeoff behave as $k$ gets very large or the number of layers increases dramatically compared to current tested models?

* Are there characteristics of data where SSNI outperforms / underperforms in unlearning compared to the prior unlearning algorithms?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper aims to address the utility loss of certified unlearning methods based on differential privacy noisy fine tuning. The paper proposes a sequential subspace noise injection scheme that performs block-wise noisy fine-tuning. In particular, it partitions model parameters into orthogonal blocks and, at each stage, unlearn by updating a single block with Gaussian noise before going to the next. Standard fine-tuning is performed at the end.

Based on the shifted-Renyi framework of prior work, this paper extends the analysis to a block-aware setting and claim that composing per-block Renyi guarantees preserves teh overall privacy budget. 

The paper also presents a per-step noise lower bound for noisy fine-tuning, which they use to calibrate noise. In the proposed algorithm, this calibration replaces the initial-clipping threshold $C_{0}$ with a term $\Delta(\rho)$. 

They provide experiments on image classification benchmarks. The results show that the block-wise scheme can mitigate accuracy collapse after unlearning while maintraining resistance to membership inference attacks.

### Strengths
This paper is easy to follow and the algorithm is lised clearly.

The proposed sequential subspace noisy fine-tuning approach for certified unlearning is original and theoretically solid. The theoretical guarantee makes this approach very promising and have valuable potentials. The authors provide theoretical results to validate the reliability of the proposed approach. They also provide theoretical analysis to explain utility collapse in over-parameterized models and motivates the subspace schedule. 

Empirically, on standard image classification benchmarks, the proposed method mitigates accuracy loss after unlearning while maintaining strong resistance to membership inference attacks.

### Weaknesses
My main concern is about the unvalidated $\Delta(\rho)$.


The paper's certificates replace the unconditional model-clipping threshold $C_{0}$ from Koloskova et al. with a high-probability proximity $\Delta(\rho)$ between the full-data model and the retained-only retrain, and then calibrate noise by substituting $C_{0}$ by $\Delta(\rho)/2$. 

$\Delta(\rho)$ is the key component of the paper's guarantee and calibration. It replaces the standard clipping thresold $C_{0}$ everywhere in the analysis and in the algorithm itself, so the certified noise and step counts scale directly with $\Delta(\rho)$. 

The paper states replacing $C_{0}$ by $\Delta(\rho)$ tightens calibration. But the authors explicitly state that this "obliges us to estimate or bound $\Delta(\rho)$ in advance". However, in the experiments, they treat $\Delta(\rho)$ as a tunable parameter and do not estimate $\rho$, and ask readers to interpret results under this assumption. 

As a result, the experimental results do not establish the claimed privacy budget $(\epsilon, \delta)$ in general. The claimed results are conditional on an unvalidated proximity $\Delta(\rho)$.


Without a reasonable and/or conservative way to obtain or bound $\Delta(\rho)$ (with error analysis), the empirical section effectively reduces to demonstrating a block-wise noise schedule. This schedule by itself is a noise distribution chocie rather than a usable certified-unlearning method. That is, without $\Delta(\rho)$, it is not a novel and operational contribution beyond existing noise-allocation scheduling ideas. 

Without specifying a way to estimate a reliable/reasonable $\Delta(\rho)$, the practical advantage collapses to noise scheduling rather than an operational certified-unlearning method.

### Questions
Please refer to Weaknesses.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
In this paper, the authors investigate a block-wise variant of noisy fine-tuning for certified unlearning. They observe that injecting noise into all parameters simultaneously leads to a sharp drop in test accuracy, which only gradually recovers to the retrained model’s performance. To address this, they propose partitioning the parameter space into orthogonal subspaces and injecting noise progressively across these blocks. This strategy preserves the $(\epsilon, \delta)$-certifiability of unlearning while maintaining test accuracy significantly better than standard noisy fine-tuning approaches.

### Strengths
1. The paper is very well written, with a clear and thoughtful analytical presentation.

2. The theoretical results are rigorously proved, and the authors carefully address limitations in prior theorem assumptions.

3. The supporting experiments are well designed and effectively validate the theoretical insights presented in the analysis.

### Weaknesses
1. The experiments are conducted primarily on smaller models such as ResNet, raising concerns about the scalability of the proposed method. It remains unclear how well the approach would translate to large-scale, billion-parameter models, or how practical it would be to perform layer-wise orthogonal decomposition across complex architectures.

2. While the empirical results show minimal accuracy loss for block-wise noisy fine-tuning (Block-NFT), it is not clearly established whether the total fine-tuning cost $kT \ll T_{\text{retrain}}$ is guaranteed in practice. Since achieving minimal loss may require increasing $k$, the efficiency advantage over full retraining becomes less certain.

3. In Table 1, only class 5 from CIFAR is deleted. A more comprehensive evaluation would involve deleting multiple classes and reporting the average performance, which would provide a clearer picture of the method’s overall effectiveness and robustness.

### Questions
1. Can the proposed orthogonal decomposition be applied to transformer-based architectures, particularly within the attention blocks? If so, how computationally expensive would this process be compared to its implementation in convolutional networks?

2. Is the Chebyshev inequality (Eqn 3) applied in any of the theorems? I might have overlooked its usage if it was implicitly incorporated in the analysis.

3. In Table 1, the method Sa1UN shows a retraining accuracy (RA) closest to the full retrain, and although its runtime is somewhat higher than the proposed method, it is still reasonable. However, its test accuracy is also notably high — does the test set include samples from the forget set, potentially inflating the reported accuracy?

4. Regarding the Membership Inference Attack (MIA) metric — how exactly is it computed in this paper? Ideally, shouldn’t the MIA accuracy be close to 50% (indicating that the attacker cannot reliably distinguish between samples from the forget set and the test set), rather than approaching 100%?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the practical limitations of certified unlearning approaches based on differential privacy. The core insight is partitioning the parameter space into multiple orthogonal subspaces and sequentially applying noise injection and fine-tuning within each subspace. Empirical validation on the datasets demonstrates the effectiveness of the proposed approach.

### Strengths
1.	The paper identifies the limitations of current certified unlearning based on differential privacy and validates the root cause through both theoretical analyses.
2.	The idea of decomposing the noise injection process into orthogonal subspaces and applying it sequentially can preserve the theoretical guarantees of certified unlearning while alleviating the accuracy degradation issue.
3.	Experiments on standard image classification benchmarks consistently demonstrate the efficacy of the proposed approach.

### Weaknesses
1. The paper assumes that the parameter space can be partitioned into strictly orthogonal subspaces. Please explain how to correctly distinguish orthogonal parameter subspaces, and why exactly K subspaces can always be partitioned?
2. The parameter space of deep neural networks exhibits high non-convexity and strong coupling characteristics. Could you explain how strong assumptions impact the practical scenario?
3. The proposed approach critically depends on knowing the upper bound $\Delta(\rho)$ of the distance between the initial model and the retrained model. In practical unlearning scenarios, this bound is fundamentally unavailable.
4. In Figure 1, why is there a performance gap between the proposed scheme and the baseline (NFT) unlearning initializations? A similar phenomenon can be observed in Figure 3.
5. In the experiment, options such as K=2, 4, and 10 were used. Is the final choice of K better when larger, smaller, or is there a trade-off? Alternatively, what external factors might influence the selection of K?

### Questions
I would appreciate the authors’ responses to the five weaknesses outlined above.

### Soundness
2

### Presentation
3

### Contribution
3
