# Drop-Muon: Update Less, Converge Faster

- Decision: Reject
- Scores: 2, 2, 4, 8

## Abstract
Conventional wisdom in deep learning optimization dictates updating all layers at every step--a principle followed by all recent state-of-the-art optimizers such as Muon. In this work, we challenge this assumption, showing that full-network updates can be fundamentally suboptimal, both in theory and in practice. We introduce a non-Euclidean Randomized Progressive Training method--Drop-Muon--a simple yet powerful framework that updates only a subset of layers per step according to a randomized schedule, combining the efficiency of progressive training with layer-specific non-Euclidean updates for top-tier performance. We provide rigorous convergence guarantees under both layer-wise smoothness and layer-wise $(L^0, L^1)$-smoothness, covering deterministic and stochastic gradient settings, marking the first such results for progressive training in the stochastic and non-smooth regime. Our cost analysis further reveals that full-network updates are not optimal unless a very specific relationship between layer smoothness constants holds.
Through experiments on NanoGPT-124M, a ResNet50 image-classification task, and small controlled CNN studies, we empirically demonstrate that Drop-Muon consistently outperforms full-network Muon, achieving the same accuracy up to $1.4\times$ faster in wall-clock time.
Together, our results suggest a shift in how large-scale models can be efficiently trained, challenging the status quo and offering a highly efficient, theoretically grounded alternative to full-network updates.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
In this work, the authors challenge the assumption of all-layer udpates, showing that fullnetwork updates can be fundamentally suboptimal, both in theory and in practice. They introduce a non-Euclidean Randomized Progressive Training method—DropMuon—a framework that updates only a subset of layers per step according to a randomized schedule, combining the efficiency of progressive training with layer-specific non-Euclidean updates for top-tier performance. They provide rigorous convergence guarantees under both layer-wise smoothness and layer-wise (L0 , L1 )-smoothness. They also do CNN experiments on MNIST or CIFAR, and show that Drop-Muon consistently outperforms standard full-network Muon.

### Strengths
MuonDrop samples a random subset of layers according to a user-defined distribution D and updates only the parameters of the selected layers , keeping all other layers frozen. This is an interesting idea.

Establishes convergence guarantees for Drop-Muon under two regimes: layer-wise smoothness (Theorem 4.1) and layer-wise (L0 , L1 )–smoothness (Theorem 4.2). And the authors claim that provide the first convergence guarantees for progressive training-style methods in the non-smooth setting.

Did CNN experiments on MNIST or CIFAR, and show that Drop-Muon consistently outperforms standard full-network Muon.

### Weaknesses
(1) Some parts of the writing is not friendly to non-theory background people like me, such as line 099 "admits two equivalent formulations" or line 130/131 "when ||.||_(i)=||.||_2….". And I don't quite get the motivation of only updating a subset of layers. And naturally, it is hard for me to grasp section 4.

(2) The empirical result is conducted on relatively small datasets (MINIST CIFAR), with CNN. It would be nice to try larger dataset, or adding language model experiments.

(3) Only MOUN is compared as baseline, it would be nice to add more baseline optimization methods.

### Questions
In addition to weakness (1), what do you mean by non-smooth setting in line 192?

Did you also search for hyper parameter for the baseline? (line 3220)

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces Drop-Muon, a new randomized layer-wise optimizer that updates only a subset of layers at each iteration rather than the full network. The method extends the Muon optimizer framework by incorporating Randomized Progressive Training (RPT), sampling a random minimal index $s_{k}$ and updating all layers from $s_{k}$ to $b$, thereby reducing computational cost while maintaining convergence guarantees.

The authors make three key claims:
* Full-network updates are not theoretically optimal unless layer smoothness constants satisfy a rare equality.
* Theoretical results include convergence guarantees under both layer-wise smoothness and generalized $(L_{0}, L_{1})$-smoothness, for both deterministic and stochastic settings.
* Empirical results on CNNs (MNIST, Fashion-MNIST, CIFAR-10) show Drop-Muon matches Muon’s accuracy while converging up to 1.4× faster in wall-clock time.

### Strengths
* Novel perspective: The work challenges a long-standing assumption that updating all layers per step is necessary for optimal convergence, providing a fresh theoretical and empirical lens.

* Rigorous theoretical analysis: The paper presents detailed convergence proofs across multiple smoothness regimes and establishes a cost model linking layer sampling distribution to compute efficiency.

* Practical implementation: The method is conceptually simple, compatible with existing architectures, and demonstrated to reduce wall-clock time without loss of accuracy.

* Theoretical-experimental coherence: The experiments, though small, confirm the theoretical insight that selective layer updates can outperform full-network training.

### Weaknesses
$\textbf{Weaknesses in theoretical results}$
* $\textbf{Layer-wise optimal tuning.}$ Likewise, most theoretical results count for optimally tuning the learning rate as some problem property-dependent values. It makes sense if a single learning rate is the only (or dominant) term that requires such tuning. 
  * Theorem 4.1 assumes layer-wise optimal learning rates, which depend on unknown smoothness constants. This assumption is rarely satisfied in practice and undermines the practical interpretability of the theoretical results. Moreover, layer-wise learning rate tuning greatly reduces usability compared to single global tuning.
* $\textbf{No clear theoretical advantage for layer-wise randomization.}$ 
   * The results in Theorem 4.1–4.3 effectively decompose into independent layer analyses with double summations $(K,b)$, suggesting the benefits come from parallel or reduced-cost computation rather than a deeper coupling among layers.
   * The proof structure reads more like “an analysis per-layer, concatenated,” rather than showing true synergy among layers.

---

$\textbf{Cost-optimality justification, in section 4.1.1 COST OPTIMIZATION, is fragile.}$
* I believe a faithful result of such a claim, “full-network updates are not in general optimal”, is significant. However, the justification in this study is rather fragile.
* The claim is only supported by an idealized cost model. While it proves non-optimality theoretically, it does not guarantee that the proposed RPT sampling is superior.
* The argument seems to restate that the largest smoothness constant limits the global learning rate, a well-known result, without clearly showing why randomization improves optimization.
   * For me, it seems to restate that "ill-conditioned problems, e.g., quadratic with Hessian $H$, the largest learning rate is limited by the largest $L_{\max}$, i.e., $LR = 1/L_{\max} = 1/\lambda_{\max}(H)$". Here, the "cost" is also limited by the largest $L$. 

---

$\textbf{Empirical Verification}$
* I fully understand that evaluating a new optimizer is inherently challenging. However, a reasonably comprehensive assessment helps prevent superficial or unconvincing claims and demonstrates respect for the reviewers. Moreover, a study that presents rigorous theoretical results deserves an equally thorough empirical verification.
   * The experiments are limited to small CNNs on MNIST, Fashion-MNIST, and CIFAR-10, which are far from the common deep learning setups (e.g., ViT, BERT, GPT-2), motivating the paper. The current benchmarks are too lightweight to demonstrate practical scalability.
   * I am not asking for months of training, but rather for experiments that extend beyond the few hours reported in this work, ideally to a few days, to better reflect realistic training scenarios.

---

$\textbf{Conceptual Weaknesses}$
* The justification of the “randomized progressive” sampling rule (updating all layers from $s_{k}$ to $b$ appears primarily motivated by backprop efficiency, not theoretical necessity. A completely random subset could better illustrate the claimed generality of the framework.
* The paper’s theoretical innovation largely builds on the existing Muon and RPT frameworks with minimal new structural design.

---

Overall, this paper presents an interesting theoretical argument and a clean framework questioning full-network updates, with rigorous proofs and a well-structured manuscript. However, the practical implications and empirical validation are too weak for such a bold claim. 

With stronger empirical demonstrations and a clearer justification for how randomization improves beyond computational convenience, this work could become an impactful contribution to the optimizer design literature.

### Questions
* In Randomized Progressive Training, the expected number of updated layers is roughly $b/2$. Is this ratio fixed or tunable? Would different expected update ratios affect convergence and cost trade-off?

* Why does Drop-Muon require updating from the sampled $s_{k}$ to the last layer? Would fully random subset updates (non-contiguous) provide stronger or weaker convergence in practice?

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
4

### Summary
This paper claims that optimizers don't need to update all network layers at every step, arguing this is computationally suboptimal for modern non-Euclidean optimizers like Muon. The authors propose Drop-Muon, a framework that instead updates a random subset of layers. Their core contribution is Randomized Progressive Training (RPT), a computationally efficient strategy that samples a "cutoff" layer and only updates layers from the cutoff layer to the end, which aligns with the backpropagation algorithm. The paper provides rigorous theoretical convergence guarantees for this method under generalized layer-wise $(L^0, L^1)$-smoothness and shows that full-network updates are theoretically inefficient. Empirically, Drop-Muon shows wall-clock time speedup over standard Muon on shallow CNNs trained on MNIST, Fashion-MNIST, and CIFAR-10.

### Strengths
1. The paper notices a simple but fundamental question about a new class of optimizers (Muon, Gluon, Scion). Questioning the "dense update" assumption is a valid and interesting line of inquiry.
2. The paper introduces progressive training into non-Euclidean optimizers, that's an interesting idea.
3. The paper provides the first convergence guarantees for a progressive, non-Euclidean, stochastic optimizer under generalized $(L^0, L^1)$-smoothness, which is a significant technical achievement.
4. The paper provides empirical results showing Drop-Muon can show speedup in wall-clock time.

### Weaknesses
1. The paper uses Randomized Progressive Training (RPT) method for frozen layer selection, but lacks discussion about the effect of RPT to performance. I understand that it's mainly for efficiency. However, not updating model evenly may lead to performance sacrifice. As shown in Appendix G.1, updating deeper layers with higher probability shows worse performance. More discussion is necessary.
2. There is a gap between the theoretical conclusion and experimental setups. Although the authors show optimal probability for Progressive Training, they just use random version in experiments. 
3. My main concern is that we can't see any performance improvement (final accuracy) empirically and Drop-Muon is even worse than Muon in most cases. This is a bit opposite with theoretical conclusion -- subset-network updating can achieve optimal while full-network updating cannot.  
4. Experimental setup is limited. Section 6 only shows results of 3-layer CNN on simple tasks. It doesn't show more realistic results for modern model architectures.
5. Experiments is limited in image tasks. Why not show results on language tasks as well? As the authors claimed in line 333, results on nanoGPT have shown that full-network updates is hard to achieve optimal. Why not test if Drop-Muon can help in this case?
6. Why do all experimental results report training accuracy rather than validation/test accuracy that is a more standard and promising metric?

### Questions
1. The paper only mentions Gluon when discussing about $(L^0,L^1)$-smoothness for Muon/Scion. I notice another previous paper name ClippedScion [1] also contributes on the same non-Euclidean $(L^0, L^1)$-smoothness framework. It's better to discuss this work as well.


[1] Pethick, Thomas, Wanyun Xie, Mete Erdogan, Kimon Antonakopoulos, Tony Silveti-Falls, and Volkan Cevher. "Generalized Gradient Norm Clipping & Non-Euclidean $(L_0, L_1)$-Smoothness." arXiv preprint arXiv:2506.01913 (2025).

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes a novel claim that updating only part of neural network results in better computation complexity compared to updating the entire model under most scenarios when optimizing a neural network with Muon (or in general LMO-type updates), and provides both theoretical and empirical justifications. Instead of considering only the iteration complexity of an optimizer, this paper focuses on optimizing the computation complexity by carefully analyzing the total computation cost, which is the expected computation cost per iteration times expected number of iteration for convergence. Moreover, this paper proposes Drop-Muon, a general framework that only updates a random sub-model using Muon (or LMO-type) optimizers. While the random sub-model can be sampled from any arbitrary distributions, this paper provides theoretical analysis of a certain distribution, randomized progressive training (RPT), which samples a random index $s^k\in [b]$ in iteration $k$ and only updates the $s^k,\ldots,b$-th layer (where $b$ denotes the total number of layers) while keeping the first $s^k-1$ layers frozen. Combining Drop-Muon with RPT, this paper provides convergence analysis of the resulting optimizer under both deterministic and stochastic loss scenarios under various smoothness assumptions. Specifically, the convergence rates match the optimal rate in corresponding settings. Notably, the convergence analysis and the computation cost optimization analysis together suggest that updating the entire model every step is only optimal when certain smoothness condition is met, which is unlikely in practice. Finally, this paper also provides experiment evidence to show that partially updating the model is indeed faster in computation compared to updating the entire model.

### Strengths
For me, the major strength of this paper lies in its novelty. It studies an practically important questions of how to accelerate training computation while maintaining the same performance, and it proposes one novel possible solution--partially updating neural networks. Moreover, this paper provides a systematic justification, both theoretically and empirically, to support this idea. For example, it provides theoretical analysis of Drop-Muon with RPT sampling under both deterministic and stochastic gradient settings and under different smoothness assumptions. It also provides alternative sampling distributions to choose submodels. Overall, I find the theoretical analysis of this paper very concrete. Furthermore, the empirical experiments also provide strong evidence that Drop-Muon is not only theoretically better but also outperforms full model update in practice.

### Weaknesses
- Although theoretically Drop-Muon achieves the same theoretical iteration complexity (i.e., dependence on total iteration $K$) as other optimizers without partial updates, I noticed from experiments (e.g. Figure 1 and 3) that the iteration complexity of Drop-Muon is worse than Muon in practice. This suggests a potential limitation of the tradeoff between iteration complexity (or data complexity) and computation cost. Specifically, in certain scenarios where the data size is the limiting constraint, Drop-Muon might have worse performance than full update Muon.
- The experiment section only includes rather simple tasks and dataset such as MNIST and fashion-MNIST. Experiments on more complicated tasks such as pre-training language models will be a stronger evidence to support the empirical performance of Drop-Muon. I personally think some relatively small models such as nano-GPT would be sufficient for this goal.
- A minor opinion that might improve the readability: could authors elaborate a bit further on the layerwise smoothness, especially since such constants are essential in both the convergence rate and the sampling distribution. For example, I think providing an example of layerwise smoothness set of a $b$-layer dense neural network will help me better understand it.

### Questions
- Could the authors elaborate how the activation of the first $s^k-1$ layers can be cached? If I understand correctly, each iterations samples a fresh data batch $\xi_k$, so the forward pass requires re-computing all activations from layer 1 to b. For example, in iteration $k-1$, the first layer computed activation $X^{(1)}\_{k-1}\xi\_{k-1}$, but it's required to compute $X^{(1)}_k\xi_k$ in iteration $k$ and $\xi\_{k-1}\ne \xi_k$.

### Soundness
3

### Presentation
3

### Contribution
4
