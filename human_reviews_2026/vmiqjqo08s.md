# Robust Bidirectional Associative Memory via Regularization Inspired by the Subspace Rotation Algorithm

- Avg Score: 4.40
- Decision: Reject
- Scores: 4, 6, 4, 6, 2

## Abstract
Bidirectional Associative Memory (BAM) trained by Bidirectional Backpropagation (B-BP) suffer from poor robustness and sensitivity to noise and adversarial attacks. To address it, we propose a novel gradient-free training algorithm, the Bidirectional Subspace Rotation Algorithm (B-SRA), designed to improve the robustness and convergence behavior of BAM. Through comprehensive experiments, two key principles, orthogonal weight matrices (OWM) and gradient-pattern alignment (GPA), are identified as central to enhancing the robustness of BAM. Motivated by these insights, new regularization strategies are introduced into B-BP, yielding models with significantly improved resistance to corruption and adversarial perturbations. We conduct an ablation study across different training strategies to determine which approach achieves a more robust BAM. Additionally, we evaluate the robustness of BAM under various attack scenarios and across increasing memory capacities, including the association of 50, 100, and 200 pattern pairs. Among all strategies, the SAME configuration—which combines OWM and GPA—achieves the highest resilience. Our findings suggest that B-SRA and carefully designed regularization strategies lead to more reliable associative memories and open new directions for building resilient neural architectures.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper extends the Subspace Rotation Algorithm to Bidirectional Associative Memory (BAM), proposing B-SRA, and integrates both OWM and GPA regularization methods to enhance the training process. The proposed approach effectively improves the robustness of BAM against noise and adversarial attacks.

### Strengths
The proposed improvements are effective, with clear methodological explanations and comprehensive experimental results that strongly support the claims.

### Weaknesses
1. While the paper utilizes two well-established regularization methods, OWM and GPA, their direct application without significant adaptation or novel integration strategy does not constitute a strong innovative contribution.
2. The research focus of this paper, Bidirectional Associative Memory (BAM), appears to be a relatively niche area. Based on the introduction provided, it seems to have attracted limited research attention in recent years. Furthermore, the experimental results presented do not sufficiently demonstrate strong practical potential for BAM. It is necessary to provide a more detailed justification of BAM's research significance and application prospects.
3. The organization of the paper deviates from conventional structure, with the experiments section occupying a disproportionately large portion of the content, while the methodology section lacks sufficient detail. It is recommended to condense the definition of BAM and the stability analysis, possibly relocating them to an appendix. Additionally, the source of the stability analysis in section 2.1 should be clearly stated—whether it is an original contribution or derived from existing work.

### Questions
See weakness.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes a novel algorithm for robust training of Bidirectional Associative Memory (BAM), named B-SRA (Bidirectional Subspace Rotation Algorithm), which is gradient-free and inspired by recent work on subspace rotation in Hopfield networks. The authors identify two principles, orthogonal weight matrices (OWM) and gradient-pattern alignment (GPA), as critical to robustness and incorporate them as regularizers into the original gradient-based BAM training (B-BP). Experimental results across multiple datasets and attack scenarios demonstrate clear gains in robustness, particularly when both regularizers are applied (SAME configuration).

### Strengths
Clarity: The algorithmic description and core concepts are clearly presented and easy to follow, particularly for the new training method.

Originality: B-SRA is a compelling alternative to traditional B-BP, offering better robustness and convergence without the need for gradient-based optimization.

Empirical Validation: The experimental setup is extensive, with clear improvements shown in both adversarial robustness and noise resilience. The ablation study is particularly helpful in dissecting the roles of different regularization components.

### Weaknesses
Theoretical Scope Limitation: While the linear-case analysis is insightful, the paper would be stronger with theoretical justification or approximation results for more commonly used nonlinear BAM architectures. The lack of formal results in such settings limits the generality of the claims.

Clarity in Experimental Tables: The tables could benefit from clearer formatting. It is not always obvious which values represent robustness performance or regularization metrics, and which direction (higher/lower) is better. A clearer legend or visual emphasis on best results would improve readability.

Redundancy in Experimental Presentation: The inclusion of many variations and datasets is thorough, but at times excessive. A more concise presentation—e.g., one table summarizing robustness across all methods and datasets—would help the reader focus on key comparisons.

### Questions
1. Why does the regularized B-BP (SAME) sometimes outperform B-SRA? I believe a more elaborate discussion of this phenomenon would be a valuable addition. More broadly, a deeper reflection on the experimental results could greatly strengthen the narrative, e.g., what they suggest about the nature of robustness in BAM, and how each method contributes.

2. Could B-SRA be used as an initialization for B-BP? This might combine the robustness and fast convergence properties of B-SRA with the adaptability of gradient-based optimization. Was this hybrid approach considered or tested?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper tackles the problem of improving the robustness and stability of Bidirectional Associative Memory (BAM) networks, which are designed to learn two-way associations between paired patterns.  It extends the Subspace Rotation Algorithm (SRA)—previously used for Restricted Hopfield Networks—to Bidirectional Associative Memory (BAM), yielding a gradient-free training method
They introduce two regularization techniques for B-BP through the usage of Orthogonal Weight Matrix (OWM) to encourage orthogonal weights to preserve signal norms and suppress noise, and Gradient Pattern Alignment to align gradients with data patterns in order to make learning more stable and resistant to perturbations. They run experiments under gaussian noise and different adversarial attacks to demonstrate the resilience of the proposed method.

### Strengths
- While previous works introduced the Subspace Rotation Algorithm (SRA) for Restricted Hopfield Networks (RHN), this applies SRA to BAMs. 
- The algorithm is well explained and easy to implement with pseudo code 
- The authors also propose gradient pattern alignment (GPA) for aligning the gradient of the loss with the stored input patterns. Previous works do not apply GPA to associative memory training. The authors jointly apply Orthogonal Weight Matrix (OWM) regularization and GPA. - Evaluations are done both accuracy and bitwise error under perturbations

### Weaknesses
- Positioning  - Need more clarity on the contribution. The work is an adaptation of SRA to BAM
Orthogonality and gradient-input alignment style terms exist in broader literature; using them for training of BAM is reasonable but also incremental.
- No direct comparisons to Dense Associative Memories / Modern Hopfield Networks or to orthogonality-promoting training in neural networks. 
- The authors claim that B-SRA enhances the robustness and convergence speed. But, do not provide any timing or iteration count plots to back this claim.

### Questions
- See Weakness

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the poor robustness of Bidirectional Associative Memory (BAM) networks trained with standard Bidirectional Backpropagation (B-BP). The authors first introduce a novel, robust, gradient-free trainer, the Bidirectional Subspace Rotation Algorithm (B-SRA), which demonstrates inherent resilience to noise and adversarial attacks. By analyzing B-SRA, they identify two key principles responsible for this robustness: maintaining Orthogonal Weight Matrices (OWM) and achieving Gradient-Pattern Alignment (GPA). The authors then propose these principles as novel regularization terms for the standard B-BP algorithm. Extensive experiments on pattern association tasks (MNIST, Chinese script) under various noise and adversarial attacks (FGSM, PGD) demonstrate that B-BP with both OWM and GPA regularizers (the "SAME" strategy) achieves the highest level of robustness, significantly outperforming standard B-BP and even the B-SRA method that inspired it, especially at larger memory capacities.

### Strengths
1. The paper's strongest contribution is its scientific method. It proposes a robust gradient-free algorithm (B-SRA), performs a root-cause analysis to determine why it's robust (OWM + GPA), and then successfully ports those principles to fix the vulnerable B-BP algorithm.
2. The ablation in Sec 4.3.2 is excellent. It cleanly isolates the individual contributions of OWM (the ORTH strategy) and GPA (the ALIGN strategy) and demonstrates that both are required for full robustness (the SAME strategy). Table 1, which measures the OWM and GPA values for each strategy, provides a direct link between the model's properties and its performance.
3. The proposed "SAME" strategy (B-BP + OWM + GPA) is shown to be highly resilient. It achieves near-perfect retrieval under strong masking, noise, and adversarial attacks (FGSM, PGD) where the baseline B-BP and even the ALIGN-only models fail completely.
4. The paper shows that the "SAME" strategy scales well with increased memory capacity (from 50 to 200 pattern pairs) and network depth (to 5 layers). In fact, its robustness improves with scale, outperforming B-SRA, which degrades as capacity increases.

### Weaknesses
1. The paper focuses exclusively on Bidirectional Associative Memory (BAM), which is a classic but relatively niche architecture. The authors state an intent to apply these principles to Transformers and modern Hopfield networks as future work, but the paper presents no evidence that these findings will transfer.
2. The experiments use low-resolution, bipolarized images (MNIST, Chinese script) . While standard for testing associative memory, this is far from the complex, high-dimensional data where robustness is a critical issue today (e.g., in computer vision or language modeling).
3. Algorithm 1 is explicitly for a 3-layer BAM. The 200-pair experiment uses a 5-layer BAM. The paper never explains how B-SRA's SVD update (Algorithm 1) or the OWM/GPA regularizers are applied in this deeper, multi-layer setting. This is a significant methodological omission.

### Questions
1. Algorithm 1 is for a 3-layer BAM. How were the B-SRA algorithm and, more importantly, the OWM and GPA regularizers adapted for the 5-layer BAM used in the 200-pattern capacity test?
2. The "SAME" strategy (B-BP+OWM+GPA) was the most robust. What were the $\lambda_{ortho}$ and $\lambda_{align}$ (Appendix A.1) values used in the experiments? How sensitive is the model's robustness to these new hyperparameters?
3. Why did the "DIFF" strategy (OWM + opposing GPA) perform so much worse than "SAME"? Table 1 shows its OWM/GPA metrics look decent, but Figure 2 shows it fails under noise. This implies the direction of the GPA is critical, which is a key finding that seems under-emphasized.
4. Your conclusion suggests applying OWM and GPA to Transformers. Have you performed any preliminary experiments to see if these principles hold; e.g., does enforcing OWM on the FFN layers or Q/K/V matrices in an attention block improve its adversarial robustness?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents the Bidirectional Subspace Rotation algorithm (B-SRA), a gradient-free method for training Bidirectional Associative Memories (BAMs). B-SRA extends the Subspace Rotation Algorithm (SRA) from Restricted Hopfield Networks (RHN) to Bidirectional Associative memories. The claims that B-SRA improves the robustness and convergence behavior of BAMs relative to Bidirectional Backpropagation (B-BP). It mentions that a set comprehensive experiments show that orthogonal weight matrices (OWM) and gradient pattern alignment (GPA) are key to the robustness of BAMs. Based on this, the paper claims to introduced regularization techniques to that significantly improved B-BP's resistance to corruption and adversarial perturbation.

### Strengths
The paper tries to improve the robustness of BAMs. This topic is significant because of BAM's suitability for modular neuromorphic hardware design and robust learning. These and other potential benefits of BAMs have led to an increase in interest within the AI research community.

### Weaknesses
1) Unsupported claim about Bidirectional Backpropagation (B-BP): The paper makes the strong claim, in the abstract and introduction, that "B-BP suffers from poor robustness and sensitivity to noise and adversarial attacks". But the paper cites the Lin et. al 2024 paper to support this claim even though the Lin et. al 2024 paper does not mention B-BP at all, it instead discusses unrelated associative memories.  So the criticisms of B-BP are lack support and the author(s) appear to confuse B-BP with BAM.
Further, typing the string "Noise bidirectional backpropagation" in Google Scholar produces the 2019 paper in the journal Neural Networks titled "Noise-boosted bidirectional backpropagation and adversarial learning". This 2019 B-BP paper demonstrates not just that B-BP is robust to noise, but actually shows how B-BP benefits from blind and non-blind noise injections, see, for instance, the noise plots in Figure 4 and related noise summaries in Tables 8-10. The authors have simply mischaracterized B-BP and provide no support for their central claim.

2) Further unsupported claims about the paper findings: The authors claim to have conducted "comprehensive experiments" and "multiple experiments",  without presenting this claimed data that they have introduced new "regularization strategies ..." into B-BP without producing a mathematical description of "regularization" of B-BP. Regularization is a form of penalized or constrained optimization. The paper does not state any such optimization. Again, going to Google Scholar, one finds at least one paper on B-BP Regularization with Hidden Bayesian Priors: "Hidden Priors for Bayesian Bidirectional Backpropagation", and the 2023 proceeding of the IEEE SMC. So, again, the authors fail to support their claim, or even clearly define it.

3) Insufficient information on OWM and GPA: There is insufficient information about these two principles. It is important to clarify what they mean in the context of BAM training because they are central to the design of the new regularization strategies in this paper.

### Questions
1)  Could you clarify your claim about Bidirectional Backpropagation (B-BP) ?  (See number 1 under weaknesses)

2)  Please respond to the point on B-BP regularization?  (See number 2 under weaknesses)

3). Add more information about key components of the paper: SRA, OWM, and GPA.

4) Are there experimental results illustrating the impact of OWM and GPA on the robustness of BAMs in settings other than B-BP?

### Soundness
1

### Presentation
2

### Contribution
1
