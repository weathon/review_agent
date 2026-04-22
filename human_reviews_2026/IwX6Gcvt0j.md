# Bridging Kolmogorov Complexity and Deep Learning: Asymptotically Optimal Description Length Objectives for Transformers

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 8, 2, 6, 6

## Abstract
The Minimum Description Length (MDL) principle offers a formal framework for applying Occam's razor in machine learning. However, its application to neural networks such as Transformers is challenging due to the lack of a principled, universal measure for model complexity. This paper introduces the theoretical notion of asymptotically optimal description length objectives, grounded in the theory of Kolmogorov complexity. We establish that a minimizer of such an objective achieves optimal compression, for any dataset, up to an additive constant, in the limit as model resource bounds increase. We prove that asymptotically optimal objectives exist for Transformers, building on a new demonstration of their computational universality. We further show that such objectives can be tractable and differentiable by constructing and analyzing a variational objective based on an adaptive Gaussian mixture prior. Our empirical analysis shows that this variational objective selects for a low-complexity solution with strong generalization on an algorithmic task, but standard optimizers fail to find such solutions from a random initialization, highlighting key optimization challenges. More broadly, by providing a theoretical framework for identifying description length objectives with strong asymptotic guarantees, we outline a potential path towards training neural networks that achieve greater compression and generalization.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This work suggests theoretical bridge between Kolmogorov complexity and deep learning. The authors demonstrate that transformer models can be trained using objectives derived from asymptotically optimal description length. They prove that minimizing these objectives leads to representations that are optimally compressed, with any deviation from the theoretical limit being negligible. The paper also confirms the existence of transformer architectures capable of achieving this asymptotic optimality. While the theory is promising, the authors note that the practical optimization of these objectives remains a significant challenge, even as their empirical results show measurable gains in model generalization.

### Strengths
This paper establishes rigorous theoretical foundations connecting Kolmogorov complexity to deep learning, proving the existence of asymptotically optimal description length objectives for transformers. The novel construction using the ALTA compiler to map Turing machines to transformer weights provides a concrete bridge between computation theory and neural networks. The paper offers comprehensive analysis of both two-part and variational codes with detailed proofs, while the gaussian mixture model implementation presents a practical, differentiable approach to these theoretically-grounded objectives.

### Weaknesses
While theoretically sound, the work faces significant practical limitations. The proposed method encounters substantial optimization difficulties - standard training procedures fail to discover the low-complexity solutions that the theory guarantees exist. This optimization challenge undermines immediate practical utility. The empirical validation remains quite narrow, relying primarily on synthetic tasks like parity computation without broader benchmarking. The Turing machine emulation approach, while elegant theoretically, appears computationally inefficient for real-world applications. Furthermore, the paper would benefit from direct comparisons with contemporary compression techniques to better position its contributions within the current landscape of this area.

Minor Issues:

1. The notation for additive constants ($\stackrel{+}{=}$, $\stackrel{+}{\leq}$) while standard in information theory, may be unfamiliar to some machine learning readers and could benefit from additional explanation.
2. Some figures (particularly in appendices) have small font sizes that reduce readability.
3. The relationship between the theoretical bounds and practical performance could be more clearly articulated.

### Questions
1. Could the optimization challenges be addressed through better initialization schemes or alternative optimization algorithms specifically designed for these objectives?
2. How does the performance of the GMM-based variational code compare to state-of-the-art compression methods like quantization-aware training or low-rank adaptation?
3. Are there theoretical guarantees about the convergence rate when optimizing these objectives, or bounds on the gap between the found solution and the theoretical optimum?
4. How scalable is the ALTA-based construction to large-scale transformers used in practice? Are there complexity bounds on the compilation process?
5. How do these objectives perform when applied to large-scale datasets and models, beyond presented synthetic tasks?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a framework for analyzing asymptotic bounds for description length objectives based on the Minimum Description Length (MDL) principle and the universality of Kolmogorov complexity. The authors prove the existence of asymptotically optimal description length objectives for Transformers. Empirically, they evaluate their derived variational objective on a parity task to demonstrate out-of-distribution (OOD) generalization ability.

### Strengths
1. The motivation for designing practical description length objectives for deep learning models is clear and well-articulated.

2. The notion of asymptotically optimal codes and their connection to the MDL principle is well-motivated and explained.

3. The paper is overall well-written and structured, offering a useful background on Kolmogorov complexity, source bounds, and the invariance theorem, along with a solid problem statement and mathematically detailed analysis.

### Weaknesses
**Major Concerns**:

1. **Limited practical significance**: While the paper formally establishes the existence of asymptotically optimal codes for Transformers (Theorem 1), it is hard to see what tangible impact or practical value this result provides to Transformer practitioners. The connection between the theoretical guarantees and practical model performance is weak.

2. **Reliance on heuristics**: Although Section 4 presents seemingly rigorous derivations, the practical heuristics (e.g., line 310) are mostly informal, intuitive, and insufficiently justified. This raises concerns about the soundness of analysis and whether the theoretical principles or the heuristics themselves are driving the results. The transition from theory to practice is not well grounded.

3. **Insufficient empirical validation**: The only empirical evaluation (Table 1) focuses on a toy parity task. This experiment is too limited to substantiate the theoretical claims or to demonstrate the effectiveness on realistic settings or tasks. The theory currently appears preliminary and largely hypothetical.

4. **Ambiguous scope**: The authors implement their framework on Transformers, yet do not make any architecture-specific analysis. It remains unclear whether the theory is meant to be architecture-agnostic or specific to Transformers, and why Transformers were chosen as the primary focus.

### Questions
1. In your statement, “Any regularity in the data that is useful for prediction is also useful for compression, and vice versa,” how do you formally define regularities? For readers without a background in computational complexity or algorithmic information theory, a more intuitive explanation would greatly enhance the readability.

2. Is the proposed description length objective specific to Transformers, or can it be generalized to other neural architectures such as fully connected or convolutional networks? If it is general, why is the empirical focus exclusively on Transformers? It is quite strange not to start with simpler fully connected networks. If it is not, what makes Transformers particularly suitable or unique in this context?

3. How does your objective compare to previous complexity measures such as path norms or weight norms investigated in [1] and other compression-based methods as surveyed in Appendix D.3? These metrics also serve as regularizers tied to model complexity and generalization. What conceptual or empirical distinctions set your formulation apart?

[1] Fantastic Generalization Measures and Where to Find Them, ICLR 2020.

### Soundness
2

### Presentation
2

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
This theoretical paper connects Kolmogorov complexity with deep learning by creating optimal compression goals for Transformers, based on the idea that the best model is the one that compresses the data the most. It shows that Transformers can simulate any computation, allowing them to achieve near-optimal compression as they grow larger. The paper also provides a practical version of this goal that can be trained with gradients, but finds that standard training methods struggle to find these well-compressed solutions on their own.

### Strengths
The paper makes significant theoretical contributions by rigorously connecting Kolmogorov complexity to neural network training through MDL, providing detailed proofs for the existence of universal two-part codes and asymptotically optimal families for Transformers.

### Weaknesses
1) The paper admits that standard optimizers fail to find solutions of low complexity via random initialization as also shown in Table 1. This might lead to practical difficulties when optimizing the objective if one doesn’t choose careful initialization or specialized methods.

2) The asymptotically optimal code relies on emulating a Turing machine via prompt tokens, which the authors acknowledge is "inefficient" and "overly restricting" (Section 4.2). This construction is primarily an existence proof and does not leverage the parallel or numerical strengths of Transformers, making it unclear for me how to apply it in practice reliably.

3) I would encourage the authors to explore asymptotically quasi-optimal codes (Appendix B.11) more deeply in future work, as they might offer better trade-offs between theory and practice.

### Questions
Is the optimization difficulty fundamental to MDL objectives in neural networks and what do you think would work best to resolve it?

### Soundness
3

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
3

### Summary
The authors introduce a theoretical bridge between Kolmogorov complexity and transformers by defining asymptotically optimal description length objectives. The authors show that under certain resource bounds a minimizer of this quantity achieves optimal compression up to an additive constant. They demonstrate that these objectives exist for Transformers, using the computational universality of Transformers. They then develop a variational objective based on adaptive Gaussian mixture prior. Empirically, the proposed objtective encourages lower-complexity solutions with strong generalization on algorithmic tasks, though optimization challenges remain.

### Strengths
- The paper provides a rigorous, original connection between the Kolmogorov complexity and neural network training objectives, extending minimum description length to modern architectures. 
- The mathematical derivations are sound, explained thoroughly, and presented in a way that makes them easy to parse. 
- The idea of asymptotically optimal description length objectives is novel and opens an the door towards a theory that guides model optimization in practice.

### Weaknesses
- The experimental section is narrow and doesn’t convincingly show that the proposed objectives can scale or improve over baselines in realistic settings. In particular, the OOD Acc gain is relatively small when random initialization is used and does not achieve the intended purpose of bridging the theory of MDL to practical value through new training objectives. 
- It is hard to do random initialization at larger scales and for more complex tasks, hence the method would not be useful for such settings. 
- There’s no quantitative comparison to other compression-aware training approaches.

### Questions
- Would the proposed objective lead to tighter generalization bounds? It seems like the natural objective to use for those, as it balances the compressed size and empirical performance. Could the authors try computing such bounds?
- Would using optimizers other than Adam prevent the collapse? SGD is known to generalize better than Adam, and there are now other pseudo second-order optimizers that work well in practice for Transformers. 
- How doable is it to optimize the asymptotically optimal objective in large-scale training? 
- Are there other prior distributions that could be theoretically sound in this context and could improve the optimization?

### Soundness
3

### Presentation
3

### Contribution
2
