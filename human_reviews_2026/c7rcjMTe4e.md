# Boosting Adversarial Robustness and Generalization with Dictionary Structure

- Avg Score: 4.67
- Decision: Reject
- Scores: 2, 6, 6

## Abstract
This work investigates a 
novel approach to boost adversarial robustness and generalization by incorporating structural prior into the design of deep learning models.
Specifically, our study surprisingly reveals that existing dictionary learning-inspired convolutional neural networks (CNNs) provide a false sense of security against adversarial attacks. To address this, we propose Elastic Dictionary Learning Networks (EDLNets), a novel ResNet architecture that significantly enhances adversarial robustness and generalization. 
Extensive and reliable experiments demonstrate consistent and significant performance improvement on open robustness leaderboards such as RobustBench, surpassing state-of-the-art baselines. To the best of our knowledge, this is the first work to discover and validate that 
dictionary structure can reliably enhance deep learning robustness under strong adaptive attacks, unveiling a promising direction for future research.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper revisits dictionary learning as a potential structural prior to improve adversarial robustness. The authors first show that prior dictionary-learning-based CNNs exhibit a false sense of security. Then, this paper proposes Elastic Dictionary Learning (EDL) to balance nature and robust performance.

### Strengths
+ Theory is consistent with good logic.
+ The RISTA optimization is technically sound and provides convergence guarantees.

### Weaknesses
This paper has drawbacks in both writing and experiments.

The presentation is unclear. I had to read the Introduction multiple times to understand what problem the paper is actually trying to formulate.
+ The introduction mixes multiple ideas, such as robustness plateau, reliance on generative data, robust overfitting, and dictionary priors, but lacks a coherent logical flow. I recommend authors to take a look of C.A.R.S.[1] to improve the Introduction part.
+ The phrase 'a false sense of security' is used with different meanings. I can understand that in the experiment section, it refers to gradient obfuscation. But in Section 3.2, what does this refer to (line 150)? Authors should include the citation of  [2].
+ In line 145, what is the setting of this kind of adaptive attack? It's necessary to identify the adaptive attack's settings based on [3].

Experiments:
+ All experiments are conducted on CIFAR-10/100 and Tiny-ImageNet with small ResNet backbones, which limits the generality of the conclusions.
+ Based on Table 7, the proposed Elastic DL layer introduces roughly 50% additional inference cost compared with standard CNNs, which is not practical to implement.
+ Whether this method can be extended to Vision Transformers or other attention-based models?

[1] Swales, John. "Create a research space (CARS) model of research introductions." Writing about writing: A college reader (2014): 12-15.

[2] Athalye, Anish, Nicholas Carlini, and David Wagner. "Obfuscated gradients give a false sense of security: Circumventing defenses to adversarial examples." International conference on machine learning. PMLR, 2018.

[3] Tramer, Florian, et al. "On adaptive attacks to adversarial example defenses." Advances in neural information processing systems 33 (2020): 1633-1645.

### Questions
Please see weaknesses.

### Soundness
2

### Presentation
1

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
The paper argues that prior dictionary-learning-inspired CNNs (e.g., SDNet) give a false sense of robustness: they handle random corruptions but collapse under adaptive attacks. It proposes Elastic Dictionary Learning (EDL) layers that replace convolutions. Each EDL layer solves, for a feature tensor $x$, a mixed $\ell_2-\ell_1$ reconstruction with sparsity,

$$\min_z \frac{\beta}{2}\left\|x-A^{\star}(z)\right\|_2^{2}+\frac{1-\beta}{2}\left\|x-A^*(z)\right\|_1+\lambda\|z\|_1,$$

and is unrolled with a reweighted ISTA (RISTA) update that uses per-iteration weights $w=1 /\left(2\left|x-A^*(z)\right|\right)$. Layer-wise $\beta$ is learned. The authors claim substantial gains in robust accuracy across CIFAR-10/100 and Tiny-ImageNet, often when combining EDL with adversarial training baselines (PGD-AT, TRADES, AWP, HAT, PORT). They present mitigation of robust overfitting (Table 3), strong AutoAttack numbers (Tables 4-6), multidataset/backbone ablations (Tables 8-10), certified robustness via randomized smoothing (Fig. 6), and simple checks intended to rule out gradient obfuscation (transfer attacks, a zero-order gradient match in Table 12).

### Strengths
1- Clear diagnosis + constructive fix. The paper empirically shows Vanilla DL (SDNet) ails under PGD even when good against random noise (Table 1, p. 3) and then designs EDL to trade off $\ell_2$ and $\ell_1$ fidelity (Alg. $1+$ Eq. (6), p. 5).

2- Simple, modular layer that plugs into ResNets. Fig. 1 (p. 4) and Fig. 12 (p. 15) clearly show the drop-in EDL layer and the end-to-end training pipeline (Fig. 11, p. 14). This architectural modularity is attractive for adoption.

3- Broad empirical sweep. Results cover three datasets and four backbones, multiple norms (ℓ∞/ℓ2/ℓ1), and combinations with diverse AT baselines. In particular, Table 4 (p. 7) shows sizable AutoAttack gains when adding EDL to PGD-AT/TRADES/HAT/PORT; Tables 5–6 (p. 7) compare to RobustBench-style leaderboards at multiple budgets.

4- Overfitting mitigation. Table 3 (p. 6) and Fig. 2 (p. 7) show that EDL reduces the BEST–FINAL gap and lifts final robust accuracy compared with popular regularizers and data augments.

5- Interpretability signals. The paper provides embedding-difference profiles (Fig. 5, p. 8) and detailed hidden-state visualizations (Figs. 19–20, pp. 27–28), which suggest attacks affect EDL representations less.

6- Some robustness hygiene. The authors include transfer-attack comparisons (Fig. 7, p. 8) and a zero-order vs autograd gradient agreement (Table 12, p. 26), and they plot RISTA convergence of the unrolled layer (Fig. 8, p. 9).

7- Runtime disclosure. Table 7 (p. 9) reports inference costs as EDL layers are stacked; overhead is 1–3× vs plain ResNets and only modest vs Vanilla DL.

### Weaknesses
1- Evaluation transparency gaps (AA settings, AT details).
AutoAttack. While AA results are reported (Tables 4–6), the exact AA configuration (version, components enabled, checks for catastrophic overstatement) is not fully specified in the main text. Precise ε-schedules, per-attack budgets, and restarters for PGD are scattered; PGD settings for the leaderboard comparison are summarized but could still allow optimistic numbers if not standardized.

2- Potential confounding from training protocol. The key robustness curves switch from Vanilla DL pretraining to EDL fine-tuning at epoch 150 (Fig. 2 and Fig. 14, pp. 7, 21). It is unclear whether the gains come from the structural prior itself or from regularization effects of unrolling, reset-like dynamics, or extra optimization steps. A controlled study that trains EDL from scratch under the same schedule as baselines is missing.

3- Theory is promising but incomplete.
Convergence/conditioning. The RISTA update depends on weights $w=1 /(2 \mid x- \left.A^*(z) \mid\right)$. There is no regularization floor to prevent blow-up when the residual is near zero, and step-size/Lipschitz conditions for global convergence of the nonsmooth, reweighted problem are not provided. Lemma 4.1 (p. 4) gives a local quadratic upper bound but end-to-end convergence guarantees (with learned $A, \beta$ ) are absent.

4- Obfuscation checks are not exhaustive. The zero-order match (Table 12, p. 26) shows local gradient agreement, and transfer attacks are included (Fig. 7), but EOT for any stochasticity, black-box query-budget tests, step-size sensitivity, and multi-restart PGD sweeps are not systematically presented. Given the large reported gains, a fuller obfuscation checklist is warranted.

5- Accounting for capacity \& parameters. Replacing convolutions with EDL (unrolled iterations, extra tensors $w$, learnable $\beta$ ) likely changes parameter counts and memory footprint. Only runtime is quantified (Table 7); parameter and activation memory overheads, training wall-time, and throughput are not. Fairness of comparisons-especially to SOTA AT-needs these numbers.

6- Certified robustness methodology is under-specified. Fig. 6 (p. 8) shows better certified accuracy with randomized smoothing, but the noise level 𝜎, base classifier, sample counts, and confidence computation are not detailed, making it hard to assess comparability to standard smoothing reports.

7- Generality beyond small images. All results are on CIFAR-10/100 and Tiny-ImageNet; no ImageNet-1k training, detection/segmentation, or non-vision tasks. The approach might scale, but evidence is missing. (A few ImageNet visualization examples appear in Fig. 22, p. 30, but not full training.)

8- Ablations on $\beta$ and unrolling depth. The layer-wise learnable $\beta$ is central, yet there is no distributional analysis of learned $\beta$ across depth, no freezing vs learning ablation, and limited study of T (number of ISTA steps), step-sizes $t$, or the shrinkage schedule $\lambda_t$.

9- Reconstruction claims need stronger quantification. The recovered-noise analysis (Table 13, p. 29; Figs. 21–24) is interesting but uses aggregate norms vs “adaptive noise.” It does not show per-example causal links between better noise recovery and AA success/failure, nor how this behaves under distribution shift.

### Questions
Please resolve the aforementioned weaknesses

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes combining Convolutional Neural Networks (CNNs) with Dictionary Learning (DL) to improve model performance and robustness. By integrating a DL module into a CNN, the network learns features that are both discriminative for classification and sparse and reconstructable, making them more stable under noise or adversarial attacks. This approach enhances the model’s ability to capture meaningful structures in data while maintaining strong accuracy.

### Strengths
The paper includes extensive experiments to support its main idea, including ablation studies, visualizations, and comparisons with baseline models. These experiments demonstrate how each component of the proposed method contributes to improved performance and help illustrate the interpretability and robustness of the learned features.

### Weaknesses
Dictionary Learning (DL) models require more complex and computationally intensive calculations compared to regular neural networks. Even with unrolled inference, the EDL layer adds approximately 2–3× more computation than a standard convolution block. However, their overall performance is generally lower, as traditional DL methods are difficult to train end-to-end and struggle to match the efficiency and scalability of standard deep neural networks.

### Questions
How does it compare with a regular CNN?

### Soundness
3

### Presentation
3

### Contribution
3
