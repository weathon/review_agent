# On Uniformly Scaling Flows: A Density-Aligned Approach to Deep One-Class Classification

- Decision: Reject
- Scores: 4, 2, 6

## Abstract
Unsupervised anomaly detection is often framed around two widely studied paradigms. Deep one-class classification, exemplified by Deep SVDD, learns compact latent representations of normality, while density estimators realized by normalizing flows directly model the likelihood of nominal data. In this work, we show that uniformly scaling flows (USFs), normalizing flows with a constant Jacobian determinant, precisely connect these approaches. Specifically, we prove how training a USF via maximum-likelihood reduces to a Deep SVDD objective with a unique regularization that inherently prevents representational collapse. This theoretical bridge implies that USFs inherit both the density faithfulness of flows and the distance-based reasoning of one-class methods. We further demonstrate that USFs induce a tighter alignment between negative log-likelihood and latent norm than either Deep SVDD or non-USFs, and how recent hybrid approaches combining one-class objectives with VAEs can be naturally extended to USFs. Consequently, we advocate using USFs as a drop-in replacement for non-USFs in modern anomaly detection architectures. Empirically, this substitution yields consistent performance gains and substantially improved training stability across multiple benchmarks and model backbones for both tabular and image anomaly detection. These results unify two major anomaly detection paradigms, advancing both theoretical understanding and practical performance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper connects Deep SVDD and normalizing flows for anomaly detection by introducing a subclass called Uniformly Scaling Flows (USFs), where the Jacobian determinant is constant across inputs. Under this setup, the maximum-likelihood objective of a flow becomes equivalent to the Deep SVDD loss, offering a simple link between one-class classification and flow-based modeling. The authors argue that this keeps the stability and invertibility benefits of flows while avoiding the collapse issues of Deep SVDD. In experiments, they replace affine coupling layers with additive USF layers in existing flow-based detectors (FastFlow, CFlow, U-Flow) using frozen pretrained backbones. Results on MVTec AD and VisA show that the USF variants reach similar or slightly better accuracy, with improved training stability and less sensitivity to initialization. The paper positions USFs as a lightweight and theoretically grounded alternative to standard flow-based anomaly detectors.

### Strengths
- The paper formalizes a clear theoretical link between Deep SVDD and a subclass of normalizing flows with constant Jacobian determinants, unifying two separate paradigms of anomaly detection through an analytical derivation.
- The paper reinforces previous observations that the log-determinant term can dominate likelihood in flow-based detectors, and systematically validates that removing input-dependent volume terms leads to more stable training.
- The paper includes a range of controlled experiments and ablations that clearly isolate the effect of the proposed change, making the empirical findings easy to interpret.
- The paper is generally easy to follow, with clean derivations and consistent notation. The theory-to-experiments flow is coherent, and the ablation results are presented in a readable and well-organized manner.

### Weaknesses
- The paper builds its main theoretical motivation on Deep SVDD, a 2018 one-class method that has largely fallen out of use in both image and tabular anomaly detection. Modern benchmarks (e.g., ADBench, MVTec, VisA) consistently show that simple feature-based or transformer-based approaches, contrastive learning, and diffusion-based detectors far outperform Deep SVDD, which is now mostly cited for historical context. As a result, grounding the paper’s core theory around an outdated baseline places it in an awkward position. The conceptual bridge it draws may be elegant, but its practical impact is limited unless validated against current methods.
-  The paper frames Uniformly Scaling Flows (USFs) as a key innovation for removing input-dependent volume effects, but constant-Jacobian / volume-preserving flows have long existed in the literature (e.g., NICE; FlowSVDD; OneFlow), and have already been applied specifically to anomaly detection to avoid SVDD-style collapse. As such, the only substantial novelty lies in the formal SVDD–USF MLE equivalence and the controlled additive-vs-affine ablations, not in the idea of using constant-determinant flows itself. 
- The empirical section mostly replaces affine coupling with additive coupling in pre-existing architectures (FastFlow, CFlow, U-Flow) using pretrained frozen features. The practical contribution is thus modest. It is more of an architectural ablation than a fundamentally new detection framework. The theoretical bridge to Deep SVDD does not clearly lead to a new practical algorithm beyond this substitution.
- Each baseline uses a different pretrained backbone (ResNet-18, WRN-50-2, CaiT), so absolute numbers are not directly comparable across architectures; only within-architecture swaps are fair. A shared backbone control would strengthen the empirical claims.
- In the main text, in Proposition 1, $F_\alpha(x)=x/(\alpha||x||)$ gives $||F_\alpha(x)||\equiv 1/\alpha$ for all $x\neq0$, so the claimed monotonicity $||x||\uparrow\Rightarrow||F_\alpha(x)||\downarrow$ is false; yet the loss is then (incorrectly) written as $\alpha^{-2}\mathbb{E}||x||^{-2}$. In Appendix B.4, $F_\alpha$ is redefined as the scalar $1/(\alpha|x|)$ but treated as vector-valued, creating a type mismatch with the Deep-SVDD loss $\mathbb{E}||F(X)-c||^2$. A minimal fix is the vector-valued radial inversion $F_\alpha(x)=x/(\alpha||x||^2)$ with $F_\alpha(0)=0$, which yields the intended ordering and $L=\alpha^{-2}\mathbb{E}\big[1/||x||^2\big]=1/\big(\alpha^2(d-2)\big)$ for $d>2$.
- The citations should be in parentheses using \citep.

### Questions
- How sensitive are the reported gains to the choice of pretrained backbone? Would the same improvements hold if all methods shared an identical architecture or were trained from scratch?
- How does the proposed USF formulation differ from prior work such as FlowSVDD, which also uses constant-Jacobian (volume-preserving) flows for anomaly detection and explicitly argues that this avoids hypersphere collapse?
- How does the improved flow baselines compared to state-of-the-art non-flow baselines?
- The experiments rely on MVTec AD and VisA, both image datasets, even though all features come from pretrained ImageNet backbones. Why restrict the evaluation to images? If the method is architecture-agnostic, testing on tabular anomaly detection (e.g. ADBench) could better show its generality.

### Soundness
2

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
4

### Summary
This paper proposes Uniformly Scaling Flows (USFs), a restricted class of normalizing flows where the Jacobian determinant is constant across the input space. The authors show that maximum-likelihood training of such a flow can be reformulated as a Deep SVDD-style objective with an implicit regularization term preventing collapse. Experiments on replacing standard coupling layers in FastFlow, CFlow, and U-Flow with USF variants showed moderate AUROC improvements and notably reduced run-to-run variance on MVTec AD and VisA benchmarks.

### Strengths
1. The paper provides a clean mathematical connection between one-class classification and flow-based objectives.
2. This work empirically shows improved training stability across several flow architectures.
3. The idea of exploring Jacobian regularization may inspire follow-up work on stable density modeling.

### Weaknesses
1. The theoretical equivalence is mathematically valid but largely follows from simplifying the log-det term in standard flows. It does not lead to a new learning principle.
2. The main idea is an extension of the prior hybrid flow–SVDD and One-Flow works. The difference from these prior works is a mild variation rather than a substantive theoretical or algorithmic innovation.
3. Only modest empirical improvements are shown on many classes in the experiments.
4. The evaluation is limited to two industrial anomaly detection datasets, and the experimental comparisons are limited to flow-based methods. They didn't include the latest SOTA methods into the experimental comparisons.

### Questions
1. The derivations are internally consistent and reproduce known properties of flow-based AD models. However, the equivalence theorem is straightforward once the constant determinant assumption is applied, and the empirical validation does not seem to justify the claimed unification.
2. Section 5 on experiments reads like an afterthought and does not deeply analyze why stability is improved.

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
The theory is clear and careful: training a uniformly scaling flow (USF) with MLE reduces to a Deep-SVDD-style objective with an implicit weight regularizer, which explains why collapse is avoided. The constant-Jacobian view cleanly links density level sets to latent norms. Empirically, the drop-in USF swap improves stability strongly and often accuracy across MVTec AD and VisA on CFlow and U-Flow (e.g., ~72.5→90.8 and ~92.7→94.5), while FastFlow is roughly neutral. The scope is mainly image AD; dimensionality-reduction needs the proposed VAE hybrid. Overall, technically solid.

### Strengths
1. Clear theoretical link between USFs and Deep SVDD that explains behavior and avoids collapse.
2. Practical “drop-in” recipe that substantially reduces run-to-run variance; accuracy gains on CFlow/U-Flow are convincing.
3. Broad evaluation across datasets, metrics, and architectures; ablations isolate the US modification.
4. Writing is professional; related work coverage is adequate.

### Weaknesses
1. Architectural novelty is incremental (additive/volume-preserving flows are known); the novelty depends mostly on the connection and analysis.
2. Expressivity trade-off vs. affine coupling is under-analyzed; FastFlow gains are limited.
3. Evaluation scope is visual AD; no results on tabular/time-series or non-vision AD.
4, VAE-USF section is promising but under-evaluated (few settings, limited ablations).
5. Baselines emphasize flow variants; adding non-flow one-class/reconstruction baselines would strengthen the broader claim.

### Questions
1. Beyond isotropic Gaussian bases, do heavy-tailed or mixture bases preserve the same alignment benefits in USFs?
2. For FastFlow, why are gains small? Capacity, optimization, or interaction with 2D conv flows?
3. Did you actually apply the log-normal prior on det(J) during training, and how sensitive are results to its hyperparameters?
4. In VAE-USF, how did you set the reconstruction vs. likelihood weighting, and did varying latent dimension change the story?
5. Any path to dimension reduction without the VAE (e.g., relaxed invertibility or partial flows)?

### Soundness
4

### Presentation
3

### Contribution
3
