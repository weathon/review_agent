# Provable Benefits of Sinusoidal Activation for Modular Addition

- Decision: Reject
- Scores: 6, 6, 4, 4

## Abstract
This paper studies the role of activation functions in learning modular addition with two-layer neural networks. We first establish a sharp expressivity gap: sine MLPs admit width-$2$ exact realizations for any fixed length $m$ and, with bias, width-$2$ exact realizations uniformly over all lengths. In contrast, the width of ReLU networks must scale linearly with $m$ to interpolate, and they cannot simultaneously fit two lengths with different residues modulo $p$. We then provide a novel Natarajan-dimension generalization bound for sine networks, yielding nearly optimal sample complexity $\widetilde{\mathcal{O}}(p)$ for ERM over constant-width sine networks. We also derive width‑independent, margin-based generalization for sine networks in the overparametrized regime and validate it. Empirically, sine networks generalize consistently better than ReLU networks across regimes and exhibit strong length extrapolation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates the role of activation functions in learning modular addition with two-layer neural networks, comparing sinusoidal (sine) activations against standard ReLU. The authors demonstrate that sine networks achieve superior expressiveness - requiring only width 2 to interpolate modular addition, while ReLU networks need width scaling linearly with the number of summands m. They provide Natarajan-dimension-based generalization bounds showing $\tilde{O}(p)$ sample complexity for constant-width sine networks and margin-based generalization guarantees in the overparameterized regime. Experiments validate that sine networks generalize better than ReLU on modular addition tasks in both underparameterized and overparameterized settings.

### Strengths
- Addresses an interesting and important research question. The paper tackles a fundamental question about inductive bias in neural networks - whether periodic structure in the activation can help learn periodic tasks. The choice of modular addition as a testbed is well-motivated since it's a canonical algorithmic reasoning task that exhibits grokking and has been extensively studied mechanistically. The theoretical formalization of this intuition is novel and valuable.
- Impressive and comprehensive theoretical results. The paper provides a complete theoretical picture spanning multiple regimes. The expressivity separation (Theorem 4.1 vs 4.2) cleanly shows sine activation needs only $d=2$ while ReLU requires $d \geq m/p - 1$, which is striking. The Natarajan-dimension analysis (Theorem 5.9) is technically sophisticated, covering piecewise-polynomial, trigonometric-polynomial, and rational-exponential activations in a unified framework. The margin-based bounds (Theorems 6.2-6.3) properly account for optimizer-induced geometry, and the explicit constructions in Appendix F demonstrate that the bounds are tight.
- Well-designed experiments that directly validate the theory. The experimental setup carefully controls for confounds by matching architectures, datasets, and training budgets between sine and ReLU.

### Weaknesses
- The paper slightly overclaims its findings. The introduction states, "On periodic tasks, periodic bias increases expressivity and makes learning provably easier" as a general hypothesis, but this is only validated for one highly specific task - modular addition. There's no evidence this extends to other periodic problems like learning trigonometric functions or circular convolutions. The paper doesn't even establish whether 2-layer sine MLPs are universal approximators on standard domains, which limits the generality of the claims. 
- Insufficient discussion of practical limitations that limit broader applicability. The paper doesn't adequately address why sinusoidal activations are rarely used in practice despite being proposed decades ago. Almost all successful modern activations (ReLU, GELU, SiLU, Swish, Leaky ReLU) are monotonic or nearly monotonic, which appears important for gradient-based optimization. The sine construction in Theorem 4.1 requires precise weight initialization that may be fragile - small perturbations could destroy the solution. The paper mentions initialization briefly in Appendix B but doesn't analyze sensitivity or provide guidance for practitioners. There's no analysis of optimization dynamics, gradient flow, or trainability that would explain when/why sine networks can actually be learned in practice.
- Some technical gaps and unclear statements. The paper claims "nearly optimal" sample complexity in Corollary 5.11 based on the PAC lower bound in Theorem D.6, but that lower bound assumes a uniformly random label permutation that the learner doesn't know - this is a much harder problem than the standard realizable setting. The connection between this hardness and the ERM upper bound is not clearly established.

### Questions
Why do you think monotonic activations dominate in practice if periodic activations have such strong theoretical advantages on periodic tasks? Is there a fundamental tradeoff between expressiveness and trainability that your analysis misses?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper provides a rigorous theoretical and empirical investigation of why sinusoidal activations outperform standard nonlinearities such as ReLU on periodic algorithmic tasks, using modular addition as a central test case. The authors first construct a width-2 sine MLP that exactly computes modular addition under a shared input embedding, and they prove that any ReLU MLP realizing the same function must have width growing at least linearly with the number of summands $m$. They then extend this analysis to generalization, deriving Natarajan-dimension bounds for a wide family of activation functions—including piecewise-polynomial, trigonometric, and rational-exponential classes—and showing that for constant-width sine networks, the sample complexity can be as low as$\tilde O(p)$. In the overparameterized setting, they establish width-independent generalization guarantees, proving that sine networks achieve population error $\tilde O(p/\sqrt{n})$ when the normalized margin remains constant. Finally, empirical results confirm that sine networks train faster and generalize better than ReLU networks under matched setups, with normalized margins closely tracking test accuracy.  


Overall, the paper offers a clear theoretical link between periodic inductive bias and learnability, demonstrating that sinusoidal activations make modular arithmetic both expressively simpler and statistically more efficient.

### Strengths
The paper presents clear and elegant theoretical results—the constructive proof that a width-2 sine MLP can exactly compute modular addition is both strong and intuitive. The generalization analysis is technically solid: the Natarajan-dimension and margin results are well-executed and connect naturally to established learning-theory tools. The empirical findings are also consistent with the theory, as the experiments replicate the same assumptions (shared embeddings, identical optimizers and data) and demonstrate a close match between margin growth and generalization behavior. Finally, the work is highly relevant to mechanistic studies, as it formalizes the periodic patterns often observed in neural networks trained on modular arithmetic and ties them to an explicit architectural bias toward periodic structure.

### Weaknesses
- The experimental and modeling setup is fairly restricted. The model operates only on bag-of-tokens count vectors without sequence order and is limited to a two-layer MLP. This design isolates periodicity effectively but omits much of the structure that real-world models, such as Transformers, rely on. As a result, it remains unclear how the theoretical advantages would carry over when tokens are contextualized or order-dependent.

- The study also focuses exclusively on modular addition. While this is a clean and interpretable benchmark, the conclusions would be more compelling if validated on other periodic or quasi-periodic tasks such as parity, sine-wave regression, or modular multiplication, to assess the robustness of the theoretical claims.

- The interpretation of the “constant-width” result could be clarified further. The $\tilde O(p)$ sample complexity bound depends on the assumptions of realizability and a discrete bounded input space. It would be useful to discuss how these guarantees scale under more realistic conditions, such as continuous-valued or noisy inputs.

- Finally, the paper stops short of discussing broader implications. Although it convincingly shows that periodic activations are beneficial for periodic data, it does not explore whether or how such activations might be incorporated into modern architectures, or whether their advantages persist when combined with non-periodic components.

### Questions
### Q1  
Your analysis assumes a shared, position-independent embedding so the network only observes token counts.  
How sensitive do you think your results are to this “bag-of-tokens” assumption?  
If the model had positional or additive embeddings instead, would the same expressivity and generalization arguments still hold?


### Q2  
The proofs rely on the discrete, periodic nature of modular arithmetic.  
Could the same reasoning extend to continuous periodic functions, like $y=\sin(x_1+\cdots+x_m)$?  
And if the outputs were real-valued rather than categorical residues, would your margin-based analysis still apply?


### Q3  
Have you considered testing sine activations in larger or more realistic models—for instance, deeper MLPs or even Transformers?  
It would be interesting to know whether the same periodic bias improves performance on algorithmic or periodic reasoning tasks.


### Q4  
For the high-margin variant (width \(2p\)), what’s the trade-off between the minimal width construction and this larger model?  
Do you observe margin growth matching the theoretical prediction, or does it plateau empirically?


### Q5  
Your experiments use AdamW and Muon optimizers.  
Do you expect the main conclusions—particularly the margin and accuracy advantages—to hold if you switched to a simpler optimizer like vanilla SGD?


### Q6  
The paper https://arxiv.org/abs/2406.03445 shows that giving LLMs Fourier-style number embeddings helps them learn modular addition quickly, avoiding the grokking phase.  
Since your work explores periodic bias through sine activations, can you connect these two ideas?  
Could using sine activations inside a Transformer’s feed-forward blocks similarly speed up learning or reduce grokking?  
It might be worth a small experiment: train a 1-layer Transformer on modular addition, compare GELU vs. sine vs. a hybrid activation, and track how quickly each reaches high test accuracy and margin growth.

### Soundness
3

### Presentation
4

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
This paper analyzes the approximation power of two-layer neural networks with sine activation on modular addition data. It also presents an ERM bound derived through an analysis of the Natarajan dimension. Furthermore, it performs margin-based generalization in the overparameterized regime.

### Strengths
I believe that analyzing activation functions with periodic properties for periodic inputs is fair, natural, and meaningful research. The results presented in the paper appear to be novel in terms of originality, and the rigor of the proofs seems sufficient based on what I have seen.

### Weaknesses
1. There are several issues with the presentation of this paper. The meanings of the theorems and the logical flow of the proofs in both the main text and the appendix are not adequately explained. For example, Theorem 5.9 is obtained by appropriately bounding the presented Natarajan dimension and substituting it into an existing theorem. However, readers cannot find this logical flow in the main text before reading the proof. In addition, the appendix provides insufficient high-level explanations of the proofs. It is difficult to tell, before reading them in detail, whether the proofs are novel or simply follow existing proof techniques. A proper outline of the proofs should be added.

2. From an approximation perspective, the fact that the sine activation function approximates modular addition better than ReLU appears evident both in terms of the theorem’s content and its proof, and therefore does not seem to present significant novelty.

3. Even considering that this paper represents a basic attempt to study discrete periodic functions, the fact that its analysis focuses only on modular addition weakens its novelty. In fact, it is unclear why the paper limits its analysis to modular addition. The Natarajan-dimension-based analysis could seemingly be applied to many other types of periodic functions as well. 
The reason I think so is that the only factor that seems to prevent Theorem 5.9 from holding for other general discrete data is Lemma E.3, and it appears that a slight modification could make the result generalizable.
If not, the authors should clarify why it is difficult to do so.
In the same context, I think describing the inputs in Table 1 as discrete and bounded is an overstatement, since the theorem applies only to modular addition. If this is not the case, the authors should clarify the issue.
Also, it is a little bit strange that the theorems providing bounds on the Natarajan dimension appear only in the appendix, while Table 1 in the main text presents their results.

Minor 
When referring to the appendix, make sure to use a consistent format. For example, in line 195 (and many other places), it is referred to as Appendix B, whereas in line 202 it appears as Section F.1.1 or Section G.

### Questions
1. Could the authors clarify whether this result can be generalized to more general forms of discrete data?

### Soundness
3

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
4

### Summary
This paper investigates the role of activation functions in learning modular addition using two-layer neural networks, comparing sinusoidal (sine) and ReLU activations under a shared, position-independent input embedding setup. The authors demonstrate that sine activations achieve superior expressiveness, requiring only two neurons to exactly implement modular addition versus $\Omega(m/p)$ width for ReLU networks. They provide novel Natarajan-dimension generalization bounds yielding $\widetilde{\mathcal{O}}(\sqrt{dp})$ sample complexity for both activation families in the underparameterized regime, and establish that constant-width sine networks achieve near-optimal $\widetilde{\mathcal{O}}(p)$ sample complexity under ERM. For the overparameterized regime, they derive width-independent margin-based generalization bounds, showing that sine networks naturally attain large normalized margins ($\Omega(1)$) leading to $\widetilde{\mathcal{O}}(p/\sqrt{n})$ population error, while ReLU networks suffer exponentially decaying margins with the number of summands m. Experimental results validate these theoretical findings across both regimes, demonstrating consistent generalization advantages for sine over ReLU networks.

### Strengths
1. Strong theoretical framework with novel contributions. The paper makes rigorous theoretical contributions across multiple fronts. The Natarajan-dimension analysis (Theorem 5.9) is novel and broadly applicable, covering piecewise-polynomial, trigonometric-polynomial, and rational-exponential activations in a unified framework through elegant pairwise reduction techniques. The margin-based analysis for sine networks (Theorem 6.2) elegantly exploits the natural geometric properties of sinusoidal activations, establishing width-independent bounds under the $\lVert V\rVert_{1,\infty}$ norm constraint that align well with optimizer-induced geometries. The near-optimal sample complexity result (Theorem 5.11) for constant-width sine networks with matching lower bounds (Theorem D.6) demonstrates completeness of the theoretical picture.


2. Clear motivation and well-designed experimental validation. The paper motivates the study through connections to mechanistic interpretability results showing Fourier-like circuits in networks trained on modular addition, and broader applications of periodic representations across machine learning domains. The experimental design directly tests theoretical predictions with matched architectures, identical datasets, and controlled hyperparameters. The experiments cleanly separate underparameterized and overparameterized regimes, validating both uniform convergence predictions and margin-based generalization bounds. The use of $0.5%$-quantile margins rather than minimum margins to avoid outlier effects shows careful experimental design.

### Weaknesses
1. Incomplete comparison with ReLU network. The ReLU margin bound in Theorem 6.3 requires extraordinarily stringent conditions that may not be achievable in practice. Specifically, the theorem requires width $d \ge \frac{64,p^{m}}{m^{2}+2},4.67^{m}$ and normalized margin $\gamma_{\theta,\mathrm{ReLU}}=\Omega!\left(\frac{1}{\sqrt{p}}\cdot\frac{1}{m^{1.5m+2.5},6.34^{m}}\right)$, which involves exponential dependence on $m$ that becomes prohibitively large even for moderate values. For example, with $m=5$, the required margin scales as $\Omega(10^{-9})$, making the comparison potentially unfair. The paper does not discuss whether these conditions are artifacts of the proof technique or inherent limitations of ReLU networks.

2. Weak connection between theory and experiments. While the paper claims the experiments "validate" the theory, there are significant gaps between theoretical predictions and experimental observations that are not adequately addressed. The uniform convergence bound predicts $\widetilde{\mathcal{O}}(\sqrt{dp})$ sample complexity (Theorem 5.9), yet Figure 1 shows accuracy curves as functions of width $d$ for what appears to be fixed training set sizes, without explicitly demonstrating the $\sqrt{dp}$ scaling prediction. The margin-based theory predicts test error $\widetilde{\mathcal{O}}(p/\sqrt{n})$ when normalized margins are $\Omega(1)$ (Theorem 6.2), but Figures 2,3 plot test accuracy against weight decay rates rather than directly validating the predicted scaling with $n$ or $p$.

3. Missing ablations on key parameters. The experiments in Section 7 do not systematically vary the number of summands m or modulus p, which are the primary parameters in the theoretical analysis. The paper claims "we conduct experiments with two-layer sine and ReLU MLPs on modular addition under various settings" (Section 7, line 378), but the figures do not indicate what values of m and p were used.

### Questions
Theorem 6.3 requires ReLU networks to have width $d \ge \frac{64,p^{m}}{m^{2}+2},4.67^{m}$ and achieve normalized margins $\gamma_{\theta,\mathrm{ReLU}}=\Omega!\left(\frac{1}{\sqrt{p}}\cdot\frac{1}{m^{1.5m+2.5},6.34^{m}}\right)$. For $m=5$, this margin requirement becomes approximately $\Omega(10^{-9})$, which seems extraordinarily small. 
1. Can you clarify whether these exponential dependencies are fundamental limitations of ReLU networks for modular addition, or are they artifacts of your proof technique? 
2. Did any of your trained ReLU networks actually satisfy these width requirements and achieve these margin values?

Your theory suggests exponential separation between sine and ReLU (comparing Theorems 6.2 and 6.3), yet your experiments show sine "consistently outperforms" ReLU without quantifying the magnitude of improvement. 
3. Can you provide quantitative measurements of the performance gap, such as the percentage by which sine exceeds ReLU test accuracy at specific width d and sample size n?

### Soundness
2

### Presentation
2

### Contribution
3
