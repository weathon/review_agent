# On the Interaction of Compressibility and Adversarial Robustness

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
As demands for resource efficiency and safety in modern neural networks intensify, substantial research effort has gone into model compression and adversarial robustness. Yet despite progress on each in isolation, a systematic understanding of how compressibility shapes robustness remains elusive. In this paper, we develop a principled framework to analyze how different forms of structured compressibility - such as neuron-level and spectral compressibility - affect adversarial robustness. We show that structured compressibility can induce a small number of highly sensitive directions in the representation space, which adversaries can exploit to construct effective perturbations. Our analysis yields a robustness bound that reveals how neuron and spectral compressibility impact $\ell_\infty$ and $\ell_2$ robustness via their effects on the learned representations. Crucially, the vulnerabilities we identify arise irrespective of how compressibility is achieved - whether via regularization, architectural bias, or learning dynamics. Through empirical evaluations across synthetic and realistic tasks, we confirm our theoretical predictions, and further demonstrate that these vulnerabilities persist under adversarial training and transfer learning, and contribute to the emergence of universal adversarial examples. Our findings show a fundamental tension between structured compressibility and robustness and highlight new pathways for designing models that are efficient and safe.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
To understand the impact of model compression on adversarial robustness, this paper presents an adversarial robustness bound that interprets how structured and spectral compression induce adversarial vulnerability through their effects on the model's Lipschitz constant. Extensive experiments further demonstrate the detrimental impact of model compression on adversarial robustness.

### Strengths
**Valuable insight into model compressibility in terms of adversarial robustness**

This work provides a unified theoretical analysis that helps to clarify the relationship between model compressibility and adversarial vulnerability without being constrained to a specific norm-based perturbation.

**Extensive experiments across various settings**

This paper presents extensive experiments across different model architectures, learning mechanisms, and adversarial example generation methods, offering a thorough and comprehensive analysis of the relationship between structured compressibility and adversarial robustness.

### Weaknesses
**W1: The scenario of generating model compression is unclear.**

Model compression can be achieved either through a fine-tuning procedure using the full or partial training dataset, or in a data-free manner. In particular, for the former case, model compressibility is closely related to the feature representations of the training dataset in terms of model capacity. I would assume that the considered compressibility is restricted to a fine-tuning procedure; however, a clearer introduction to the model compression setup is needed before the discussion of compressibility.

**W2: Representation of sensitivity along a small number of directions.**

The authors claim that structured compression induces high sensitivity to adversarial perturbations along a small number of directions. However, this conclusion cannot be directly drawn merely from the observed correlation between compressibility and adversarial robustness. Lines 216–223 explain that the potent attack directions are determined by interlayer alignment; however, these directions are neither visualized nor formally defined.

**W3: Unclear explanation for the amplification of adversarial attacks in the representation space.**

The authors attribute the high adversarial vulnerability of compressed models to amplification along certain sensitive directions. However, it remains unclear how the adversarial attack is amplified and what mechanisms cause this amplification to occur.

**W4: Lack of analysis on robustness-aware model pruning techniques.**

This work discusses compressibility mainly after model training. However, many prior approaches achieve better preservation of both standard accuracy and adversarial robustness simultaneously during structured model pruning [1,2,3]. Despite the observed proportional relationship between compressibility and adversarial vulnerability, an additional analysis is needed to investigate the impact of adversarially robust model pruning.

---
### References

[1] Zhao and Wressnegger, "Holistic Adversarially Robust Pruning", ICLR 2023.  
[2] Sehweg et al., "Hydra: Pruning Adversarially Robust Neural Networks," NeurIPS 2020.  
[3] Ye et al., "Adversarial robustness vs. model compression, or both?" ICCV 2019.

### Questions
**Q1: The rationale of interlayer alignment**

I acknowledge the importance of the alignment between consecutive layers. However, in Lines 216–236, it is not clear what *interlayer alignment* exactly refers to. Moreover, I am curious about the rationale behind the formal definitions of $A^{\ast}_{\inf}$ and $A^{\ast}_2$. How do these two terms represent interlayer alignment?

**Q2: The impact on robustness improvement with small compressibility**

For CNNs (Figure 5) and Transformers (Figure 6), a relatively small spectral compressibility can result in a slight improvement in adversarial robustness. How can this positive effect be explained according to the proposed theory?

### Soundness
2

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
4

### Summary
This paper proposes a systematic theoretical framework for analyzing how structured compressibility — specifically,  neuron-level compressibility and spectral compressibility, affect model’s adversarial robustness. The authors propose to characterize l∞ and l2 operator norms of the parameters by an upper bound that decomposes into (compressibility × Frobenius norm) terms. Building on this formulation, they further derive and analyze an upper bound on the network’s overall Lipschitz constant. They show that compression introduces a few highly sensitive directions that can significantly amplify perturbations, which can then be exploited by attackers, ultimately leading to degraded robustness. The experimental section covers a wide range of architectures, such as FCNs, CNNs and Transformers, validating the theoretical predictions that models with higher structured compressibility are more vulnerable to adversarial perturbations. The results also demonstrate that the vulnerabilities induced by compression persists even under adversarial training and transfer learning, and facilitates the emergence of universal adversarial perturbations.

### Strengths
1. Provides a unified norm-based framework connecting structured compressibility and adversarial robustness.
2. Characterizing the l∞ and l2 operator norms of the parameters by decomposing the effects into compressibility and Frobenius norm terms, thereby further formalizing an upper bound on the model’s Lipschitz constant.
3. The analysis shows that the impact of compressibility on robustness persists in adversarial training and transfer learning, and it can facilitate the emergence of universal adversarial perturbations.
4. The theoretical analysis, mathematical derivation, and experimental process are relatively complete.

### Weaknesses
Although the theoretical analysis is mathematically sound and logically consistent with prior robustness theory, the overall reasoning builds on well-known intuitions (ideas such as model structured compression concentrates sensitivity along a small number of directions in representation space, which in turn results in decreased robustness), and the upper bound mainly formalize this intuition rather than uncover new mechanisms. The experiments, though thorough, largely confirm expected behaviors without surprising counterexamples or deeper causal probes. The theoretical results appear to just extensions, restating known connections in a more formalized way. In addition, the two interventions mentioned in the paper for improving robustness also seem to have been studied.

### Questions
1. The main theoretical results seem to formalize known connections between compressibility, operator norms, and robustness. Could the authors further clarify the new theoretical insights or perspectives provided by their analysis beyond existing results?
2. While the paper formalizes an upper bound on adversarial robustness in terms of compressibility and Frobenius norm, the bound appears relatively loose—being an order of magnitude above the empirical robustness gap as the authors describe in appendix. As such, its practical utility for predicting or guiding robust model design seems limited.
3. The theoretical analysis appears focused on ℓ₂ and ℓ∞ perturbations. Does the same framework extend to other robustness notions (e.g., ℓ₁, distributional robustness, or label noise)?

### Soundness
4

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
4

### Summary
The paper investigates the fundamental relationship between network compressibility and adversarial robustness. The claim is that their interaction remains poorly understood. They show theoritical bounds and also empirical evaluation on architectures (FCNs, CNNs, VIT) and multiple datasets. Results show that Increased neuron or spectral compressibility consistently reduces adversarial robustness, even under adversarial training.

### Strengths
- Paper is well motivated and well written
- Provides a well-explained theoretical contribution between compressibility and adversarial robustness, tying together concepts from pruning, low-rankness, and Lipschitz theory.
- Empirical analysis covers diverse architectures and datasets, including FCN, convolutional and transformer families, and multiple attack settings

### Weaknesses
- The theory uses global operator norm–based Lipschitz bounds.
and The bounds rely on scale-normalized parameters (using ∥W∥_F) and strict (q, k, ε)-compressibility. Can this reflect practical training dynamics with normalization layers or adaptive scaling or deep non-linear networks.
- How does it position itself with other works exploring the same paradigm [1][2], As some works claim that some sparsity helps robustness 
- The claim that compressibility fosters universal adversarial examples is intriguing but briefly demonstrated

[1] Lipschitz Constant Meets Condition Number: Learning Robust and Compact Deep Neural Networks
[2] Robust low-rank training via approximate orthonormal constraints

### Questions
- Why focus exclusively on structured compressibility (neuron/spectral)? Would unstructured or other forms behave differently?
- In Fig1,2  did not understand how to interpret these new directions? How were they visualized?
- The alignment equation notations could be explain bit better
- Line 325, how does Fig4 prove the dominant singular directions claim?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper develops a framework to investigate the effect of structured sparsity on adversarial robustness through its effect on parameter norms and the network's Lipschitz constant. Compressibility can induce a set of highly sensitive directions in the representation space.

### Strengths
1. This paper is in general well-written and presents results clearly.
2. The motivating hypothesis is very interesting and described clearly in Figure 2.
3. Abundant numerical results are provided to testify the paper's theoretical results.

### Weaknesses
1. One central claim of this paper is that the compressibility may result to a few potent direction that increases the sensitivity to perturbations, and the adversarial attacks might exploit these directions. However, I cannot picture when and how the advesaries might be able to figure out these directions. Is the neural network and the compressibility totally white-box to the adversaries, which could hardly happen?
2. The evaluation of adversarially robustness of NN models seems to be dependent on the attack itself.

### Questions
1. Can you please offer a motivation example of how compressibility might be taken advantage of by adversaries? Especially how would adversaries figure out the "adversarial directions"?

### Soundness
3

### Presentation
3

### Contribution
2
