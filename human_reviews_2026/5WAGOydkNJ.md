# Transformers with RL or SFT  Provably Learn Sparse Boolean Functions, But Differently

- Avg Score: 3.00
- Decision: Reject
- Scores: 2, 2, 4, 4

## Abstract
Transformers can acquire chain-of-thought (CoT) capabilities to solve complex reasoning tasks through fine-tuning. Reinforcement learning (RL) and supervised fine-tuning (SFT) are two primary approaches to this end, yet their underlying mechanisms and differences remain theoretically unclear. In this work, we examine these aspects specifically for learning $k$-sparse Boolean functions with a one-layer transformer and intermediate supervision that is akin to CoT. In particular, we analyze the learning dynamics of fine-tuning the transformer via either RL or SFT with CoT to identify sufficient conditions for it to provably learn $k$-sparse Boolean functions. We verify that these conditions hold for three basic instances, including $k$-PARITY, $k$-AND, and $k$-OR, thus demonstrating the learnability of both approaches. Notably, we reveal that RL and SFT exhibit distinct learning behaviors: RL learns the whole CoT chain simultaneously, whereas SFT learns the CoT chain step-by-step. Overall, our findings provide theoretical insights into the underlying mechanisms of RL and SFT as well as how they differ in triggering the CoT capabilities of transformers.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper studies the learnability of $k$-sparse boolean functions (parity, AND, OR) using a one-layer transformer model by generating intermediate steps in CoT fashion. For fine-tuning via RL and policy gradient, given access to immediate rewards, it is shown that transformers can learn certain functions in one gradient step. For SFT, the model can also learn these functions but in a stepwise manner, requiring compute equal to chain length.

### Strengths
* The setup is very similar to the CoT-based parity learning formulation introduced in [KS25] but makes some technical changes which allow for improved analysis. The intermediate tokens are generated via sampling as $p(y=1|z)=model(z)$ rather than simply setting $y=model(z)$, which better captures the autoregressive nature of CoT.
* This also makes error analysis easier. Compared to [WLS23,KS25], the paper does not require teacher forcing nor self-verification/filtering to control error amplification for the SFT learning result.
* The task can be more general than parity. A general condition is identified (critical gradient component) which implies learnability.

[WLS23] Noam Wies, Yoav Levine, and Amnon Shashua. Sub-task decomposition enables learning in sequence to sequence tasks, 2023.

[KS25] Juno Kim and Taiji Suzuki. Transformers provably solve parity efficiently with chain of thought. ICLR, 2025.

### Weaknesses
The main weakness is novelty, the setup and results are quite similar to [KS25]. While comparisons are given in the paper, I feel the differences are mostly technical and not significant from either an SQ learning perspective or helping to understanding post-training.
* RL with immediate rewards is very similar to teacher forcing in that each step decouples into its own problem, and so one GD step solves all steps simultaneously. While RL in this paper does not require intermediate states during generation as in [KS25], this is because exact process reward is available for the computation of each step regardless of the data from the previous step. This is an extremely strong assumption which sidesteps all the aspects of RL which make it challenging/interesting (e.g. how to obtain a process reward model, how to deal with imperfect rewards, how to do exploration, how to do credit assignment).
* SFT learning is very similar to the no teacher forcing result in [KS25]. While the paper does a cleaner setup/analysis and does not require the augmented data trick in [KS25] (which forces curriculum learning), the overall message is the same: step-by-step learning arises from training on CoT loss. Also, a similar result was proven for k-hop composition tasks by [WNBD+25] without teacher forcing or data augmentation. Hence the result does not seem to give particularly new insights into SFT.
* Besides the dynamical analysis, the results do not help us understand the differences of models fine-tuned via RL v.s. SFT since the converged solutions are the same. For example, there is no finite-sample or generalization analysis. Moreover, "pretraining" here simply refers to adding the step-wise causal mask and does not seem to have connection to real-world pretraining.
* Wording throughout the paper makes it seem as the results apply to general $k$-sparse boolean functions $\Phi_k$, but the results only apply to functions which can be seen as repeated composition of a fixed 2-sparse function $\phi_2$ (this is a much smaller class due to the standard circuit counting argument). This restriction is not made explicit in the abstract, introduction, and even the problem setup, which is misleading; $\phi_2$ is first mentioned in eq.(1) long after the definition of $k$-sparse function is given, and even the wording there does not make it clear that $\phi_2$ must be fixed across nodes (since the paragraph is talking about how to decompose general $k$-sparse functions into binary trees). This must be made clear in the introduction and Definition 2.1.

[WNBD+25] Learning Compositional Functions with Transformers from Easy-to-Hard Data. Zixuan Wang, Eshaan Nichani, Alberto Bietti, Alex Damian, Daniel Hsu, Jason D Lee, Denny Wu. COLT 2025.

### Questions
* Can the result be generalized to more general $k$-sparse functions?
* Can the setup be modified to obtain length or sparsity generalization results for the finetuned models?

### Soundness
4

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper is concerned with learnability of k-sparse Boolean functions by transformers with chain of thought.

A k-sparse Boolean function is a function that depends on k input bits. For example, a k-sparse parity on d bits is a function that is equal to the parity of inputs from some k-element subset of coordinates B.

The paper considers transformers with the following chain of thought. The first step produces k tokens. The second step produces k/2 tokens. The third produces k/4, and so on. Each token is supposed to attend exactly two tokens from the previous step (and, in case of parity, to compute their 2-wise parity). After log k steps, the final token is supposed to give the output of the function.

The paper shows results about learnability of this architecture via reinforcement learning and supervised
fine-tuning approaches.

### Strengths
The paper proposes potential approaches to circumvent limitations of learnability of k-sparse parities, established in

### Weaknesses
My major concern with the paper is that the set of k-coordinates, B, on which the function depends, seems to be known to the architecture. At least at the line 152, the lowest k tokens of the chain of thought are directly attending the corresponding indices of the set B. In this case, the role of the other coordinates is not clear (so that we are left with simply parity instead of k-sparse parity).

Now, in the interesting setting when B is not known in advance, I don't see how the proposed RL reward, for example, takes into account which coordinates belong to B and which are not. More generally, it is not clear how is the optimal solution W^* in Theorem 3.1, for example, for the initial problem of learning the parity of an unknown set of inputs.

### Questions
Is the set of k coordinates, where, say, the parity is computed, is known to the architecture?  If not, then how is it reflected, how well the architecture learns B? In particular, how is r_t  defined in equation 6 for t = 1? Is it possible to obtain some bounds on how well are optimal solutions W^* in Theorems 3.1 an 3.2 for the initial problem of computing a k-sparse parity?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper analyzes how a one‑layer transformer acquires chain‑of‑thought (CoT) style reasoning when fine‑tuned either with reinforcement learning (RL; sign policy‑gradient) or supervised fine‑tuning (SFT; hinge loss, no teacher forcing). It derives sufficient “gradient separation” conditions under which both procedures provably learn k-sparse Boolean functions, and verifies them for parity, AND, and OR. A key takeaway is a qualitative difference in learning dynamics: RL can learn the whole CoT chain at once, whereas SFT learns it step‑by‑step, needing one update per reasoning step. Closed‑form attention dynamics are given for parity (Eq. 15). Experiments in the appendix support the theory.

### Strengths
The paper cleanly formalizes why RL can propagate credit to all CoT steps while SFT induces an inductive, stagewise trajectory, yielding an intuitive and provable contrast between the two regimes.

### Weaknesses
1. Limited realism of the learning setting. Results hold for a one‑layer transformer with designed activations and structured masking aligned to the recursive CoT decomposition; this constrains architectural/general modeling fidelity to modern LLMs.
2. Population‑gradient analysis and simplified optimizers. The theory is for population (infinite‑data) gradients and sign policy‑gradient/SGD dynamics; practicality for finite‑sample training and commonly used RL (e.g., PPO with KL control) is not addressed.
3. Practical significance is unclear. While the paper sheds light on mechanisms in toy settings, the path to improved real‑world reasoning is not demonstrated.

### Questions
See weaknesses

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies the task of learning $k$ sparse Boolean functions given intermediate chain of thought data with both RL and SFT (without teacher forcing). They study a simplified 1-layer masked transformer and show that RL succeeds in a single step of sign-GD under a separation condition on a quantity they call the critical gradient component. They show that under the same condition, SFT without teacher forcing succeeds in $T = \log_2(k)$ steps.

### Strengths
- End to end analysis of sign-GD in both policy gradient and an SFT setting without teacher forcing. In addition, the SFT analysis is able to analyze multiple steps of sign-GD.
- Without making any direct assumptions on the distribution of the data, the paper identifies a criterion in terms of the critical gradient component which determines whether one gradient step is sufficient for learnability.
- The results are written with some generality and easily extend to k-parity, k-AND, and k-OR.

### Weaknesses
- The RL analysis is restricted to taking one very large step of sign-GD to immediately saturate the softmax, which significantly simplifies the analysis.
- Throughout the paper, the authors use SFT to refer to SFT without teacher forcing which changes the main conclusion. As the authors acknowledge, GD succeeds in one step in the more standard setting of SFT with teacher forcing.
- It may be more accurate to refer to the "pre-trained" transformer as a simplified model intended to ease the analysis. In particular, the hard-coded causal masks seem like a reasonable theoretical starting point but it's not clear to me how they would arise from pretraining. In addition, the activation function is required to depend on the target function, which limits the scope of the analysis.
- The final guarantees are on the L1 error of the attention patterns, which doesn't guarantee low loss.
- The paper does not analyze the sample complexity of learning $k$-parity. For example, even under random inputs and without chain of thought supervision, parity is learnable in a single gradient descent step using $d^k$ samples [Barak et al. 2022 Hidden Progress in Deep Learning: SGD Learns Parities Near the Computational Limit]. To argue for the benefit of chain of thought, you would need to prove that parity can be learned with dimension independent sample complexity.
- There is no lower bound that demonstrates that $T$ steps of sign-GD are actually necessary, only that $T$ steps are sufficient which doesn't immediately imply a separation (although from the argument it seems clear that $T$ steps really are sufficient due to the lack of teacher forcing).

### Questions
- Why is it necessary to go from equation 7 to 8 for the policy gradient? Is this a modeling decision (e.g. equation 8 is easier to analyze) or are these mathematically equivalent in some sense? The paper only mentions that the $r_t$ term is "considerably more important."
- Is there any intuition for the critical gradient component? Equation 9 is particularly hard to read because $\xi$ depends on $\sigma$ which depends on the transformer's weights $W$. Since this paper focuses on the initial step of gradient descent, it may be informative to write out a simpler expression for the critical gradient component for parity/k-AND/k-OR when $W$ is $0$ so that $\sigma$ is constant.
- In Proposition 3.1, why does this variance bound imply a lower bound for learnability via gradient descent?
- Why is the hinge loss important in section 3.2? If logistic loss were used instead, I think that the initial gradient step would be the same. Is the issue controlling the following $T-1$ gradient steps?

### Soundness
3

### Presentation
2

### Contribution
2
