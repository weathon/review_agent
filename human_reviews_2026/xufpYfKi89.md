# Sparse training of neural networks based on multilevel mirror descent

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
We introduce a dynamic sparse training algorithm based on linearized Bregman iterations / mirror descent that exploits the naturally incurred sparsity by alternating between periods of static and dynamic sparsity pattern updates.
The key idea is to combine sparsity-inducing Bregman iterations with adaptive freezing of the network structure to enable efficient exploration of the sparse parameter space while maintaining sparsity.
We provide convergence guaranties by embedding our method in a multilevel optimization framework.
Furthermore, we empirically show that our algorithm can produce highly sparse and accurate models on standard benchmarks.
We also show that the theoretical number of FLOPs compared to SGD training can be reduced from 38\% for standard Bregman iterations to 6\% for our method while maintaining test accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The goal of the work is to improve efficient training of deep neural networks. To this end it proposes to use mirror descent to train sparse neural networks combined with switching between coarse and fine phases to reduce the amount of updates needed. The pruning is done with soft thresholding. The method proposed is ML LinBreg or Multilevel LinBreg and is evaluated on standard vision benchmarks against another mirror (Bregman) based method LinBreg and standard pruning. The method is competitive or performs better then LinBreg.

### Strengths
The work is well motivated by the goal of improving efficiency of training. The combination of switching between regimes is interesting in the context of mirror descent applied to sparsity. This can combine the benefits of dynamics sparse training methods like RiGL with geometric optimization principles. Providing theory for convergence could help formalize sparse training algorithms. The empirical evaluation on standard vision benchmarks further supports the utility of the approach.

### Weaknesses
The main weakness of this work is that it does not compare with related baselines in sparse training such as RiGL.
Moreover, the current experiments are of small scale, extending the experiments to ResNet50 on ImageNet for example would make it also easier to compare with other methods. 


Another limitation is the limited discussion of the literature on pruning and sparse training. Some works that seem to be relevant based on type of algorithm proposed:

Sparse training strategy based on proximal formulation:

[1] Kusupati, Aditya et al. “Soft Threshold Weight Reparameterization for Learnable Sparsity.” International Conference on Machine Learning (2020).


Mirror flow formulation:

[2] Jacobs, Tom and Rebekka Burkholz. “Mask in the Mirror: Implicit Sparsification.” ArXiv abs/2408.09966 (2024): n. pag.


Alternating between dense and sparse phase:

[3] Peste, Alexandra et al. “AC/DC: Alternating Compressed/DeCompressed Training of Deep Neural Networks.” Neural Information Processing Systems (2021).


Using Fisher information:

[4] Singh, Sidak Pal and Dan Alistarh. “WoodFisher: Efficient second-order approximations for model compression.” ArXiv abs/2004.14340 (2020): n. pag.

To summarize, this work proposes an interesting modification of LinBreg for sparse training, however the limited comparison with other methods and scale of the experiments limits relevancy of the work. Given this I would vote for marginally reject.

### Questions
Can the experimental results be improved by training less long in the coarse regime? (If possible plot this on a curve)
Another consideration for this type of algorithm is the effect hyperparameters such as weight decay and learning rate can this be ablated and expanded upon in the appendix on CIFAR10, ResNet18 for example? [1,2]

See weaknesses for additional questions.

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
The paper proposes a dynamic sparse training algorithm that combines linearized Bregman iterations with periodic freezing of the network structure. Training alternates between full updates and phases where only currently nonzero parameters are updated, which exploits emerging sparsity while exploring the sparse parameter space.

### Strengths
+Clear optimizer view that unifies sparse training with mirror descent and Bregman iterations, including explicit proximal updates and a principled route to sparsity.

+Multilevel design that freezes structure and updates only active parameters, with formal handling of restriction and prolongation and a stated convergence result.

+Theoretical FLOP reductions relative to standard Bregman iterations are articulated and support the motivation for the freezing schedule.

### Weaknesses
-Convergence relies on relative smoothness and a PL-type inequality and is stated for exact gradients, so the guarantees do not directly cover the stochastic setting used in practice.
-The paper discusses potential computational savings but does not provide wall clock training speedups, energy, or memory profiles on hardware.
-The current evaluation focuses only on small-image vision classification tasks, which limits the generality of the findings. Such experiments may not capture the behavior of the proposed sparse training method on larger-scale datasets or different architectures like transformers.
-Recent related works [1][2][3][4][5][6][7] should be compared and discussed.
[1] Sparse Training via Boosting Pruning Plasticity with Neuroregeneration, NeurIPS 2021
[2] Top-kast: Top-k always sparse training NeurIPS 2020
[3] AC/DC: Alternating Compressed/DeCompressed Training of Deep Neural Networks, NeurIPS 2021
[4] Dynamic Sparsity Is Channel-Level Sparsity Learner, NeurIPS 2023
[5] Advancing Dynamic Sparse Training by Exploring Optimization Opportunities, ICML 2024
[6] NeurRev: Train Better Sparse Neural Network Practically via Neuron Revitalization, ICLR 2024
[7] A Single-Step, Sharpness-Aware Minimization is All You Need to Achieve Efficient and Accurate Sparse Training, NeurIPS 2024

### Questions
1.Can you report end to end training measurements versus SGD and LinBreg, including wall clock time, GPU utilization, and energy across several sparsity targets and freezing schedules.
2. How should the coarse update frequency and the restriction criterion be selected in practice, and how sensitive are results to these hyperparameters. 
3. How does the method compare to dynamic sparse training baselines such as SET, RigL, 
and DSR under matched sparsity and training budgets.

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
3

### Summary
The paper propose a DST algorithm, "Multilevel LinBreg" (ML LinBreg), which is based on linearized Bregman iterations (or mirror descent). The key novelty of the method is to alternate between two phases: a "fix" phase, where the network's sparsity pattern is frozen (only non-zero weights are trained), and a "explore" phase, where a single, full-level mirror descent step is taken, allowing the sparsity pattern to evolve. This fix-and-explore cycle is designed to drastically reduce computational costs, as the expensive, full-gradient "explore" step is performed only for a portion of the training time. Empirical results on CIFAR-10 and TinyImageNet demonstrate that ML LinBreg can produce models that are sparser and more accurate than the standard LinBreg algorithm.

### Strengths
1) The core idea of alternating between a "fix" sparse-training phase (the m coarse steps) and a "explore" phase (the 1 fine step) is simple, elegant, and well-motivated.

2) The paper's primary selling point is the massive theoretical reduction in FLOPs as shown in Appendix B. The authors demonstrate the source of the savings: standard LinBreg must compute a dense gradient at every step, while ML LinBreg only does so 1 out of every m+1 steps.

3) The proposed ML LinBreg consistently outperforms its most logical baseline, standard LinBreg, by achieving better accuracy at similar sparsity levels, or higher sparsity at similar accuracy. Furthermore, it demonstrates competitive or superior performance compared to a standard prune-and-fine-tune (PFT) baseline, without the need to first train a dense model.

4)By embedding the algorithm into a multilevel optimization framework, the authors connect it to a rich body of existing literature.

### Weaknesses
1) I think the paper can be made better by explaining the mismatch between the theoretical analysis and the practical implementation. The convergence guarantee (Theorem 1) is provided for the deterministic (exact gradient) setting. However, the algorithm is implemented using mini-batch "unbiased estimators" (Algorithm 1), which is a stochastic setting.

2) The headline-grabbing FLOPs reduction (to 6%) is entirely theoretical. This calculation assumes that unstructured sparsity (e.g., from $l_1$ shrinkage) translates directly to a proportional decrease in wall-clock time. This is notoriously false on modern hardware like GPUs, which are optimized for dense, contiguous matrix operations.

3) The FLOPs calculation for the baseline LinBreg method ($2f_S + f_D$ in Appendix B) is confusing and inadequately explained. The update rule for LinBreg (Eq. 4a) requires the full gradient $\nabla\mathcal{L}(\theta^{(k)})$, which is dense. It is not clear how this can be computed with a sparse forward pass ($f_S$) and a single dense term ($f_D$). A dense gradient typically requires a dense backward pass, which itself is often preceded by a dense forward pass. The paper needs to be much clearer on the assumptions that lead to the $2f_S + f_D$ formula, as the entire 38% vs. 6% comparison hinges on it.

### Questions
1) The main theoretical result is for the deterministic case. Can you please elaborate and provide the reader confidence that the stochastic version (which was implemented) converges or finds a good solution? Can the existing analysis be extended to the stochastic setting, perhaps under certain assumptions about variance reduction?

2) The theoretical FLOPs claims are strong. Could you please provide any practical wall-clock time results (e.g., total training time on a V100 or A100 GPU) for SGD, LinBreg, and ML LinBreg to demonstrate whether these theoretical savings translate to real-world speedups?

3) Could you please provide a more detailed derivation for the LinBreg baseline's FLOPs cost ($2f_S + f_D$)? Specifically, how is a full, dense gradient $\nabla\mathcal{L}(\theta^{(k)})$ computed at a cost that is not at least $f_D$ (forward) + $2f_D$ (backward) = $3f_D$?

4) The choice of m=99 seems to be a fixed, simplifying assumption. How sensitive is the final model's accuracy to this choice? Could a smaller m (e.g., m=10) with a different $\lambda$ achieve a similar result? A 2D-ablation plot showing final accuracy/sparsity for different pairs of (m, $\lambda$) would greatly improve the paper's practical utility. (Please point me to the explanation if I missed it in Appendix)

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Multilevel LinBreg, a sparse training algorithm derived from linearized Bregman iterations combined with multilevel optimization. The method alternates between fine and coarse updates. Experiments on CIFAR-10 and TinyImageNet show that the method achieves higher sparsity and comparable or better accuracy than LinBreg and pruning baselines, while theoretically reducing FLOPs to as low as 6% of SGD training.

### Strengths
1. Offers a convergence proof adapting results from ML-BPGD and relative smoothness theory
2. Empirical results on CIFAR-10 and TinyImageNet (ResNet-18, VGG-16, WideResNet-28-10) consistently support claims of sparsity and efficiency.

### Weaknesses
1. Limited to small-scale image datasets, lacks evaluation on large-scale or transformer architectures to demonstrate generality.
2. FLOP analysis is theoretical. It would be good to have experimental results.
3. Missing recent dynamic sparse training baselines and structured-pruning comparisons.

### Questions
1. Could the authors report actual training time and energy savings to complement the theoretical FLOP reduction?
2. How sensitive is convergence to the switching frequency and the choice of regularizer?

### Soundness
3

### Presentation
3

### Contribution
2
