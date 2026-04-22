# Random Projection Against Gradient Leakage: Privacy-Preserving in Federated Learning

- Avg Score: 2.00
- Decision: Reject
- Scores: 2, 2, 2, 2

## Abstract
Federated learning has emerged as an effective paradigm for collaborative model training under strict data-privacy constraints. However, conventional federated learning remains vulnerable to gradient-based reconstruction attacks, which can expose sensitive information. Existing privacy-preserving methods often incur significant performance degradation. To address this challenge, we propose a algorithm that balances privacy and utility by combining Random Projection Filters (RPFs) with controlled data perturbations. Specifically, during local training, each client first injects optimized noise into the data that minimally affects model performance to enhance privacy. Simultaneously, a subset of convolutional kernels is replaced with random projection filters, which structurally randomize the network and reduce the risk of sensitive data being revealed through gradients. This combination effectively strengthens privacy while inducing only a marginal drop in model accuracy. Extensive experiments demonstrate that our approach substantially improves privacy protection while maintaining high model performance, providing a practical solution for mitigating privacy risks in federated learning.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes fedRPF, a client-side algorithm to defend against gradient inversion attacks in federated learning.
The proposed fedRPF algorithm introduces two mechanisms: Random Projection Filters and Optimized Noise Injection.
The authors conduct experiments on MNIST, FMNIST, CIFAR-10, and CIFAR-100, comparing fedRPF against FedAvg and DP-Laplace, DP-Gaussian, FedNFL, and ALDP. The evaluation demonstrates that fedRPF achieves a superior trade-off between model utility and privacy.

### Strengths
- The combination of an RPF and an optimized noise injection is interesting.

- The idea of finding a perturbation that minimizes the loss is interesting.

### Weaknesses
- The paper's privacy claims are based exclusively on defense against the DLG attack. However, DLG attack is a very early gradient inversion attack. It is not known whether the proposed method is effective against subsequent attacks in this field.

- The literature review in Section 2.1 is not sufficient. There is a line of gradient inversion attacks not in discussion, and attribute inference and membership inference attacks are not relevant to the paper, as the proposed method targets the gradient leakage.

- The literature review in Section 2.2 is not sufficient either. Many defenses against gradient inversion attacks are not introduced.

- Many defenses against gradient inversion attacks are not evaluated as baselines either.

- Table 4 presents the computation cost of the method and the baselines. The computation cost for FedRPF is significantly higher than that of others, making the method impractical. There is no discussion on how to improve this.

- The authors evaluate fairness by comparing the validation accuracy of different clients (C0-C2) in Figure 5. They conclude that since the client accuracy curves "largely overlap," the proposed fedRPF method "does not introduce unfairness among clients". This part seems a little off topic, and this is not the way to evaluate fairness. Algorithm fairness should be assessed by fairness metrics like Equalized Odds, Disparate Impact, or Demographic Parity.

### Questions
See comments.

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
5

### Summary
The paper proposes fedRPF, a privacy–utility trade-off method for federated learning that (i) replaces a fraction of convolutional kernels with Random Projection Filters (RPFs) whose weights are resampled from a zero-mean Gaussian at every forward pass, and (ii) injects optimized (task-aware) input noise via multi-step PGD within a bounded box constraint. The goal is to reduce gradient-inversion leakage with minimal accuracy loss.

### Strengths
1. Simple, architecture-level defense: Swapping a subset of conv kernels with resampled random filters is easy to implement and model-agnostic, adding structured stochasticity that is plausibly harder to invert.

2. Empirical coverage and clarity: Multiple datasets, multiple baselines, and a clear threat model (semi-honest server observing gradients) make the setting explicit; figures visualize reconstructions and trade-offs.

### Weaknesses
1. Limited attack coverage.
The paper only evaluates defense under the classic DLG-style single-round gradient inversion. It totally ignores temporal gradient inversion attacks, where the attacker collects gradients from multiple rounds and optimizes them jointly. In real federated learning, this is actually more realistic — the server sees many rounds of updates. Without testing on those, it’s hard to know if fedRPF really protects privacy over time.
[1] Temporal Gradient Inversion Attacks with Robust Optimization

2. No consideration of adaptive attackers.
The current threat model assumes the attacker doesn’t know about the random projection mechanism (RPF). That’s a pretty strong assumption. In practice, an attacker can easily know or guess the defense method — for example, they might know which layers use RPF or the Gaussian sampling distribution — and adapt the inversion process accordingly (e.g., via robust optimization or meta-gradients). So the method might rely too much on “security through obscurity.”

3. No theoretical or provable privacy argument.
The paper doesn’t provide any formal analysis of how RPF or optimized noise ensures privacy. The link between the noise magnitude, random projection ratio, and privacy leakage is purely empirical. It would be nice to see at least some intuition or bound connecting these factors.

4. Heavy computation cost and scalability.
Since RPFs are re-sampled every forward pass and the PGD-based noise optimization runs multiple iterations per batch, the runtime is much higher than baselines, especially on CIFAR-100. This raises concerns about feasibility in larger or real-world deployments.

### Questions
How would fedRPF work when applied to Transformer-based federated foundation models, where gradients are huge, parameter sharing is structured (attention layers, embeddings), and random projection might interfere with self-attention stability?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes fedRPF to defend against gradient-based reconstruction attacks. The algorithm aims to balance privacy and model utility by combining two techniques at the client level: 1) replacing a subset of convolutional kernels with non-trainable Random Projection Filters (RPFs) that are resampled during each forward pass, and 2) injecting optimized noise into the local input data. This noise is generated through a Projection Gradient Descent (PGD) process designed to minimize the model's loss on perturbed inputs. The authors claim that extensive experiments on MNIST, Fashion-MNIST, CIFAR-10, and CIFAR-100 show that fedRPF achieves a superior privacy-utility trade-off compared to several baseline methods.

### Strengths
1. The idea of combining architectural randomization (RPFs) and data-level perturbation (optimized noise)  to defend against gradient inversion attacks seems interesting.

### Weaknesses
1. It is claimed that "optimized noise" protects model utility more effectively than “random noise”, but the results in Table 2 (0.7712 < 0.7752) show the opposite conclusion.
2. The paper's core assumption—that minimizing loss can defend against gradient attacks—is counterintuitive, and there is no theoretical justification for the proposed method.
3. The DLG attack evaluation in the paper skipped the most critical (first 8 rounds) training phase, making its privacy protection conclusions likely unreliable. Moreover, there have been many more advanced gradient inversion attacks besides DLG, which should be evaluated.
4. The critical experimental parameter (local training epoch) is described in a contradictory manner in the paper (“30 local epochs” vs. “one local update”). Please clarify the concepts of local epoch, local update, and global epoch.
5. The proposed method seems ineffective on complex datasets and is more than an order of magnitude slower than standard FL (>14 times), rendering it impractical in real-world applications.
6. The manuscript is riddled with undefined symbols ($r$...), typographical errors ($T-1$), citation format errors (figure 3), and typos (AIDP).

### Questions
1. Regarding the 'optimized noise' mechanism, please provide the theoretical justification for why minimizing the loss (Equation 3) can defend against gradient attacks, as this premise seems counterintuitive. 
2. Table 2 shows that the accuracy of “optimization noise” is lower than that of “random noise,” which directly contradicts your statement in Section 4.3. Could you please explain? Furthermore, the “optimized noise” condition in Table 2 appears unnecessary. Compared to the no-noise baseline (Experiment 2), accuracy drops significantly from 89.6% to 77.12% without delivering any essential privacy enhancement (as shown in Figure 3 for the MNIST dataset, where PSNR=8.52 indicates visually imperceptible image reconstruction).
3. Why is the DLG attack launched on rounds 11-13? 
4. How many local training rounds does the client actually execute per round?
5. Why is the work by RPF (Dong & Xu, 2023) not discussed in the “Related Work” section (Section 2)?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes FedRPF, a defense mechanism against gradient inversion attacks in federated learning. FedRPF employs a combination of Random Projection Filters (RPFs) and an optimized noise injection method to mitigate gradient leakage attacks. The authors conducted some experiments to examine the performance of FedRPF.

### Strengths
* The overall organization of the paper is good. The proposed mechanism is generally straightforward for the reader to follow.

* The empirical evaluations demonstrates that FedRPF achieves a favorable privacy-utility trade-off when compared against several baseline methods.

### Weaknesses
* The primary effect of RPF is to introduce randomness into the network architecture, which may just require more iterations to recover private data, rather than providing a meaningful privacy safeguard. Moreover, the use of random projections as a defense mechanism is not new [2]. The paper does not evaluate the defense against adaptive adversaries[1] that are aware of the RPF mechanism and could potentially tailor their attack to account for the structured randomness.

* The formulation of the objective function in Equation (3) is confusing and appears to contain a fundamental logical flaw within the context of privacy-preserving training. The inner minimization is defined as the process for obtaining the optimized noise. However, optimizing $\delta$ to minimize the loss function $\mathcal{L}$ on the perturbed input $X+\delta$ suggests that $\delta$ is being optimized to make the model perform better on that specific data point. If $\delta$ already makes $X+\delta$ optimal for the current model $w$, how is the subsequent model update step effectively learning?

* The experimental section lacks comparison against recent attacks and defenses [3,4].

[1] Bayesian Framework for Gradient Leakage

[2] Precode - a generic model extension to preventdeep gradient leakage

[3] See through gradients: Image batch recovery via gradinversion

[4] Refiner: Data refining against gradient leakage attacks in federated learning

### Questions
See Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
