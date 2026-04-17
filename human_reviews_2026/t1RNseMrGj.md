# Grouped Dirichlet Diffusion for Structured Generative Modeling

- Decision: Reject
- Scores: 2, 6, 6, 2

## Abstract
We present Grouped Dirichlet Diffusion (GDD), a novel generative model that employs the Grouped Dirichlet distribution to facilitate hierarchical and structured diffusion processes for high-dimensional bounded probability vectors, such as multichannel images. Unlike conventional diffusion methods that rely on Gaussian noise, GDD partitions data into meaningful feature groups (e.g., color channels in images) to preserve intra-group dependencies while allowing adaptive inter-group interactions over diffusion timesteps. Our theoretical framework ensures that both the forward marginals and reverse conditionals remain within the Grouped Dirichlet family, enabling closed-form transitions through multiplicative noise scheduling. This approach not only simplifies training dynamics but also guarantees numerical stability during sampling. Additionally, we replace the traditional evidence lower bound (ELBO) with a loss function based on the Kullback-Leibler divergence. Experimental evaluations validate the feasibility of GDD, with quantitative metrics demonstrating superior image generation performance compared to traditional diffusion models and several contemporary image generation methods.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces Grouped Dirichlet Diffusion (GDD), a generative model that employs the Grouped Dirichlet distribution as the foundation for its diffusion process. Unlike conventional diffusion methods that rely on Gaussian noise, GDD partitions data into feature groups (such as color channels in images) to capture inherent group dependencies and hierarchical structures in high-dimensional bounded data. This structured formulation improves modeling flexibility and numerical stability. This is achieved by ensuring the diffusion process operates strictly within the simplex constraints of the Dirichlet distribution. The main contributions include the introduction of GDD for diffusion modeling, improved flexibility in capturing multi-channel group patterns, and the development of a novel loss function based on KL divergence upper bounds (KLUBs).

### Strengths
- Achieves a very good FID score on CIFAR-10.
- Runs faster than traditional DDPMs, offering clear efficiency improvements.

### Weaknesses
- Overall, the paper is hard to follow and lacks clarity (see questions below).
- **Citations:** Extremely sloppy.
    - Missing parentheses for indirect citations.
    - Missing years for several citations (e.g., lines 49, 123).
    - Redundant author mentions (e.g., lines 119, 124).
    - **Table 1:** Some models are cited, others are not. Even if cited earlier, include all model references in the table for consistency. Also, consider left-aligning the “Model” column.
- **Writing quality:** Inconsistent formatting and missing spaces between figure references and citations (e.g., lines 63, 74, 240, 275, 458, ...).
- **Structure and readability:**
    - The model figure introduces components (e.g., the mapping stage) that are never discussed in the main text, only in the appendix.
    - Some variables and hyperparameters appear without prior definition.
        - New indices are introduced without explanation (e.g., ($x_{g0}$, $x_{g0i}$) — what does the “0” represent?).
        - ($S_{\text{scale}}$) and ($S_{\text{shift}}$) appear in Eq. 6 but are never described — are these hyperparameters?
    - Suggestion: Introduce the concentration parameter later, when it first appears in Eq. 7.
- **Results:** Insufficient experimental evidence to fully support the authors’ claims.
    - Experiments only use color channels as groups; additional experiments with other grouping structures would strengthen the generality claim.
    - Missing quantitative results on datasets other than CIFAR-10, despite apparent training on others (see Fig. 8).
    - Line 349–350: Consider moving external links to a footnote.
- **Table 1 issues:**
    - **LSGM:** Best FID (2.10) not listed.
    - **Consistency Models:** Only FID for consistency training (CT) shown, not the better consistency distillation CD, which is 2.93.
    - **DDIM and GET:** Unclear where the reported results come from — not found in the original papers.

### Questions
## 

- **General:**
    
    What specifically makes the process **hierarchical** (beyond Markov/sequential as in normal diffusion)? Across the paper you frequently mention “hierarchical,” e.g., line 142: *“hierarchical structure of grouped probability vectors.”* Are “grouped probability vectors” just images? If so, what is hierarchical about dividing it into color channels?
    
- **Lines 55–57 (claim on prior methods):**
    
    *“traditional diffusion methods based on Gaussian Ho et al., 2020; Guo et al., 2023 or Beta Zhou et al., 2023 struggle to capture group dependencies and hierarchical structures …”*
    
    Can you provide evidence for this claim (e.g., an experiment or citation)?
    
- **Lines 104–105 (masking):**
    
    *“… simultaneously adds noise to **and masks the data** …”*
    
    Where does masking occur? I couldn’t find a description or equation for it.
    
- **Notation (x_g):**
    
    You define $g$ as the group, so $x_g$ should be the **group vector**, not the group itself. (e.g., lines 152, 184).
    
- **Lines 172–174 (constraints & dependencies):**
    
    *“Standard Gaussian Ho et al. (2020)* *or scalar-Beta diffusion Zhou et al. (2023)* *models* *violate simplex non-negativity, unit-sum constraints, and overlook group dependencies.”*
    
    Briefly explain why these violations occur and why the first two constraints matter here.
    
- **Line 191 (conditioning direction):**
    
    You write the conditional $q(z_s∣z_t,x_0)$ with $s < t$. How does this fit the **forward** process (since $s$ is closer to $0$ than $t$)? Are you using future data $z_t$ to noise $z_s$?

- **Lines 211–213:**
    
    This statement would benefit from a supporting citation or other evidence.
    
- **Line 225 (bidirectional transitions):**
    
    When introducing bidirectional transitions, clarify their purpose and how they are used.
    
- **Line 255 (replacement step):**
    
    *“… then replaces $x_{g0}$ with its approximation $\hat{\alpha}_g$.”*
    
    Since we aim to estimate $x$ anyway, is this an intermediate approximation with *$\hat{\alpha}_g$*? Why is this intermediate approximation beneficial to the final estimation of $x$?
    
- **Figure 4 (DDIM vs. DDPM):**
    
    DDIM results appear sharper than DDPM, yet DDIM typically trades sample quality for speed. How do you explain this (e.g., different model capacity or settings)?
    
- **Equation 2 (indices):**
    
    Indices look odd (e.g., $x_{g0i}$). Please define and justify the 0 index.
    
- **Difference from Dirichlet diffusion:**
    
    Can you elaborate on the difference betwee the Grouped Dirichlet diffusion and Dirichlet diffusion [1], especially in terms on novelty? 
    

[1] Avdeyev, Pavel, et al. "Dirichlet diffusion score model for biological sequence generation." *International Conference on Machine Learning*. PMLR, 2023.

### Soundness
1

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
2

### Summary
The paper introduces Grouped Dirichlet Diffusion (GDD), a novel diffusion-based generative model that replaces Gaussian noise with multiplicative Dirichlet noise applied to feature groups (e.g., RGB channels). Each group remains on a probability simplex, ensuring non-negativity and unit-sum constraints throughout the diffusion process. The model preserves intra-group dependencies while allowing adaptive inter-group interactions, offering both theoretical closure (forward and reverse distributions remain Dirichlet) and practical stability via KL Upper Bound (KLUB) loss instead of the ELBO.

### Strengths
1. GDD introduces a strong structural prior by explicitly modeling data as grouped probability vectors on a simplex. This design allows the model to naturally capture correlations within each feature group while maintaining valid probabilistic constraints throughout the diffusion process.

2. The method ensures theoretical consistency through closed-form forward and reverse Dirichlet transitions. Its KL Upper Bound (KLUB) loss provides smooth optimization and avoids unstable boundary behavior, resulting in stable and efficient training.

3. In experiments, GDD achieves superior FID/KID scores on benchmark image datasets and faster sampling speeds than comparable diffusion frameworks, demonstrating that the grouped Dirichlet design improves both generation quality and computational efficiency.

### Weaknesses
1. “Meaningful groups” are claimed but only RGB channel grouping is demonstrated. The paper frames GDD as partitioning data into meaningful feature groups—explicitly citing image color channels as the running example—so as to preserve intra‑group dependencies while allowing adaptive inter‑group interactions. In practice, however, all implementations and evaluations instantiate grouping via fixed RGB channels in image space; the method section even notes that image channels are grouped as a prior modeling assumption to greatly simplify the mathematics and ensure closed‑form marginals. No experiments demonstrate learned group discovery or alternative semantics beyond color channels across domains. Consequently, the central claim about “meaningful groups” remains under‑validated empirically.

2. No strategy for latent‑space grouping; evidence is limited to image‑space generation. GDD is formalized on grouped probability vectors derived from the observed data (after scaling/shifting), with the entire diffusion defined on the data/pixel space rather than a learned latent manifold. While the architecture includes an encoder–decoder U‑Net, the paper does not propose an algorithm to design or learn group partitions in latent space, nor does it report latent‑space generation results. As a result, portability to latent‑diffusion setups and non‑image domains is currently an open question rather than a demonstrated capability, narrowing the immediate applicability and generalization of the approach

### Questions
See weaknesses.

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
4

### Summary
This paper proposes Grouped Dirichlet Diffusion (GDD), a generative model for high-dimensional bounded probability vectors, such as images. The framework extends Beta Diffusion by using the Grouped Dirichlet distribution, which allows the model to partition data into feature groups (e.g., RGB channels) and preserve intra-group dependencies. Unlike traditional Gaussian-based diffusion, GDD features multiplicative noise. The method ensures closed-form transitions in theory, and replaces the standard ELBO with a KL divergence-based loss (KLUB) for optimization. Experiments on image datasets show that GDD achieves SOTA performance on FID and KID metrics compared to several baselines, including DDPM and Beta Diffusion.

### Strengths
1. The core idea is principled and elegant. Using the Grouped Dirichlet distribution to model dependencies between feature groups (like RGB channels) is a natural extension of Beta Diffusion and is more suitable for this data structure than independent Gaussian noise.
2. The experimental results are strong. Table 1 shows that GDD achieves state-of-the-art FID (2.76) and KID (1.22) on CIFAR-10, outperforming the most relevant baseline, Beta Diffusion (FID 3.06).
3. The framework appears computationally efficient. Table 3 shows GDD has a faster average processing time per batch and generates more images per second than both DDPM and Beta Diffusion, which is a significant practical advantage.

### Weaknesses
1. The loss function (Eq. 19) is a heuristic KL Upper Bound (KLUB). It relies on an arbitrary-looking weighting factor ($\omega=0.97$) to combine two different bounds. The paper provides no theoretical justification for this specific value, making the final training objective feel ad-hoc and not really that principled.
2. The paper's central claim is about modeling "hierarchical and structured" data using "meaningful feature groups." However, all experiments are on image datasets only, where the "group" is simply the three RGB channels. This is the simplest, most obvious grouping possible and does not sufficiently validate the model's ability to handle more complex group structures (*e.g.,* hierarchical features in tabular data or other modalities).
3. The ablation study in Table 2 is weak. It only shows that a 4-layer MLP mapping network is better than a 1- or 2-layer one, and that removing it entirely breaks the model. This is not insightful and does not provide any analysis on the grouping strategy itself, which is the paper's main contribution.

### Questions
1. The KLUB loss in Eq. 19 uses a weight $\omega=0.97$. How sensitive is the model's performance to this hyperparameter? Is there a more principled way for choosing this value, or was it found empirically?
2. The core contribution is "grouping." How does the model perform if the group structure is mis-specified (*e.g.,* grouping pixels spatially instead of by channel)? The paper needs to demonstrate that the model is truly leveraging the group structure, rather than just performing well on CIFAR-10.
3. Table 3 shows GDD is faster than Beta Diffusion, even though it is handling higher-dimensional (grouped) Dirichlet distributions instead of 1D Beta distributions. What is the source of this efficiency gain?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes Grouped Dirichlet Diffusion (GDD), a Dirichlet analogue of Beta Diffusion applied group-wise on the simplex, trained with a KL upper bound (KLUB). While the construction is mathematically clean, the empirical claims are not supported under matched settings, and key baselines are missing.

### Strengths
This paper presents an interesting generative modeling idea where leveraging constraints in certain data types should theoretically be advantageous. The paper could benefit from a more compelling use case (for example, applying it to data that is truly compositional and showing an advantage compared to other methods in that type of data)!

### Weaknesses
Major concerns

1) “Faster convergence / better performance” not supported under matched settings.
To claim faster/better, the paper must train competing methods under identical recipes (same U-Net capacity, augmentation, schedule, optimizer, data budget, NFE grid) and report compute-normalized metrics. As written, no such matched study is presented.


2) Missing strong baselines and modern solvers.
DDPM++ 2.78
DDPM++ cont. (VP) 2.55
DDPM++ cont. (sub-VP) 2.61
DDPM++ cont. (deep, VP) 2.41
DDPM++ cont. (deep, sub-VP) 2.41
NCSN++ 2.45
NCSN++ cont. (VE) 2.38
NCSN++ cont. (deep, VE) 2.20

already meet or beat GDD’s reported numbers in most configurations. More recent Karras-style v-parameterization with fast ODE solvers reports ≈1.8–2.0 FID at ≈35 NFEs, whereas GDD uses much higher NFE and attains worse FIDs. Thus the “faster/better” claim is not supported.

3) DDIM / Consistency numbers unclear.
The DDIM FID around ~15 appears inconsistent with commonly reported values (e.g., ~13.6 at ~10 NFEs). Please provide exact citations, configs (conditional vs. unconditional), sample counts, and NFE for every table entry. For Consistency Models, ECT reaches ≈2.15 FID on CIFAR-10 with 2 function evaluations—two orders of magnitude fewer than GDD—so comparisons should be cost-normalized.

4) Fairness and capacity parity.
The original DDPM U-Net (~35M params; limited attention) is dated. Modern backbones are ~50–60M and materially improve FID. Given your models are ~55M params, comparisons to older, smaller baselines without re-training are inconclusive.

5) Evaluation diagnostics aren’t actionable.
Peak memory utilization / similar stats are presented without a clear framing (what was expected, why differences arise, and how they tie to the method’s design). As is, it’s hard to draw conclusions.

6) Visuals / preprocessing (minor thing).
AFHQ samples appear muted/gray relative to modern baselines; this likely reflects range/logit/renormalization or plotting. Please clarify preprocessing and the exact visualization pipeline.

7) Serious concern: baseline omission undermines claims. The paper acknowledges Karras et al. in Related Work but excludes it from experiments. This selective reporting makes the efficiency/quality conclusions non-actionable and, as presented, misleading. At minimum, include Karras-style v-param + modern solvers at matched NFE and parameter count. Similarly, for Consistency Models, please justify omissions of modern pipelines (e.g., ECT) or include them.


Methodological positioning (incremental over Beta Diffusion)

GDD is a near-mechanical lift of Beta Diffusion from Beta to Dirichlet noise, with in-family marginals and time-separable conditionals per group. A compact Gamma→Dirichlet derivation makes this immediate.

Beta Case:

$$ A \sim \Gamma(a, 1) $$
$$ B \sim \Gamma(b, 1) $$
$$ C \sim \Gamma(c, 1) $$
$$ T \equiv A + B + C $$

Now, let 

$$z_{t} = \frac{A}{T} \sim \beta(a, b + c) $$ 
$$ z_s = \frac{A+B}{T} \sim \beta(a + b, c) $$

and define 

$$ \pi = \frac{A}{ A + B } \sim \beta(a, b) $$
$$ p = \frac{B}{B + C} \sim \beta(b, c) $$ 
Then, it is clear that 

$$ z_{t} = z_{s} \pi $$

$$ z_{s} = z_{t} + (1-z_{t}) p$$  

These equations are precisely those that we see in the Beta Diffusion paper. 

Generalization to the dirichlet distribution is almost immediate from this point of view 

$$ A_{i} \sim \Gamma(a_{i}, 1) $$ 
$$ B_{i} \sim \Gamma(b_{i}, 1) $$

$$ z_{g, t} := \text{Dir}( \frac{A}{\sum_{j} A_{j} } )$$

$$ z_{g, s} := \text{Dir}( \frac{A+ B}{\sum_{j} A_{j} + B_{j} } )$$

Because Dirichlet = normalized Gammas, the marginals are Dirichlet, which, with the appropriate choice of $A$ and $B$ recovers the equations from the paper exactly

$$ a \equiv \eta \alpha_{t} x_{g,0} $$ 
$$ b \equiv \eta (\alpha_{s} - \alpha_{t}) x_{g,0} $$

Now, if we set 

$$ R_{i} : = \frac{A_{i} }{B_{i} + A_{i} }$$ 
$$ U_{i} := A_{i} + B_{i} $$ 

and recall that $R_{i} \perp U_{i}$ then we see that 

$$ z_{g,s} = \frac{U}{\sum_{i} U_{i} }$$

$$ z_{g,t} = \frac{U \odot R}{\langle U , R \rangle } $$ 

Now, it is clear that the forward update (which falls naturally) is 

$$ R_{i} \sim \beta(a_{i}, b_{i}) $$ 
$$ z_{g,t} = \text{Normalize}(z_{g,s} \odot R) $$ 

This yields the forward multiply-then-renormalize update; the reverse follows by adding Gamma increments then renormalizing, which also justifies the post–Euler–Maruyama renormalization step. The construction is clean but incremental relative to Beta Diffusion.

In its current form, I recommend rejection. The construction is sound but incremental; empirical claims require matched baselines and cost-normalized evidence.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
