---
job_id: 3f872a5c-4371-4643-a2f3-d263057bd116
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: khBHJz2wcV.pdf
paper: Physics-Constrained Fine-Tuning of Flow-Matching Models for Generation and Inverse Problems
main_score_norm: 0.8
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper is squarely about generative modeling, flow matching, physics-informed / neurosymbolic ML, and inverse problems for PDEs, all of which fall within ICLR’s core scope.

## Minimum Quality
Pass ✅.  
All required sections (Abstract, Introduction, Related Work, Method, Experiments, Results, Conclusion) are present. The work is technically substantial, written in English, and provides nontrivial methodology plus extensive experiments; I do not see fatal methodological or theoretical flaws that would justify desk rejection.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I do not see any hidden instructions or prompt injections targeting automated reviewers; the content is standard scientific text.

---

# Expected Review Outcome:

## Summary

The paper presents a post-training fine-tuning framework for flow-matching generative models that enforces PDE-based physical constraints and simultaneously infers latent physical parameters. Starting from a state-only flow-matching model, the authors (i) define physics rewards via weak-form PDE residuals, (ii) introduce a joint evolution of states and parameters with surrogate base dynamics derived from an inverse predictor, and (iii) optimize a control policy via an adjoint-matching objective under a (scaled) memoryless noise schedule. Experiments on several PDE systems (Darcy, elasticity, Helmholtz, Stokes) and a natural-image recoloring experiment show improved physical residuals and meaningful state–parameter distributions, with controllable trade-offs between physics enforcement and distributional fidelity.

## Strengths

1. **Clear, technically grounded extension of Adjoint Matching to joint state–parameter flows.**  
   - Section 3.2 describes a principled way to augment a pre-trained state flow with a parameter flow $v_{t,\alpha}^{\text{ft}}$, using a surrogate base flow $v_{t,\alpha}^{\text{base}}$ derived from an inverse predictor $\varphi$. This is not just heuristic conditioning but an actual SDE over $(X_t,\alpha_t)$ controlled via $\hat{u}_t$ in Eq. (2) and trained under the adjoint-matching loss in Eq. (4).  
   - The architecture in Appendix D.2, with residual corrections $\mathcal{U}_x$ and $\mathcal{U}_\alpha$ on top of the U-FNO backbone, is well-motivated and respects the pre-trained model’s structure.

2. **Thoughtful use of weak-form residuals with local test functions.**  
   - Section 3.1 and Appendix D.3 carefully derive weak residuals for each PDE (Darcy, elasticity, Helmholtz, Stokes), moving from strong residuals $\|\mathcal{L}_\alpha x\|_{L^2}^2$ to local probes $\langle \mathcal{L}_\alpha x, \psi \rangle_{L^2(\Omega)}$.  
   - The Wendland–wavelet test family in Eq. (D.3) and the mollified compact support are a nice design to obtain numerically stable, local PDE diagnostics at moderate cost. This is a solid engineering contribution for physics-informed generative models.

3. **Theoretical clarification of memoryless noise scaling.**  
   - Section 3.3 and Appendix D.4 provide Lemma 1, showing that scaling the canonical memoryless schedule $\sigma^2(t) = 2\eta_t$ by a factor $(1-\kappa)$ with $0\le\kappa<1$ preserves the memoryless property. The derivation around the condition  
     \[
       \lim_{t'\to 0} \beta_{t'}\exp\Big(-\int_{t'}^t \frac{\chi(s)}{2\gamma_s^2}\,ds\Big)=0
     \]  
     is clearly spelled out and correct. This gives a technically sound tuning knob for noise magnitude, which is practically important given PDE sensitivity to high-variance sampling at small $t$.

4. **Comprehensive and varied experimental evaluation, with nontrivial physics setups.**  
   - The paper covers four quite different PDE classes (Darcy elliptic, linear elasticity, Helmholtz wave, Stokes flow), with purposeful train–fine-tune mismatches (e.g., damping/losslessness in Helmholtz, different BC amplitudes in elasticity, forcing mismatch in Stokes).  
   - Table 1 (elasticity BC misspecification) and the Elasticity-specific table on Page 9 show that the proposed method attains the lowest or near-lowest residuals ($R_{\text{weak}}, R_{\text{strong}}$) while keeping $\mathrm{MMD}_x$ and $\mathrm{MMD}_\alpha$ moderate, beating or matching PBFM and ECI.  
   - Figure 5 (Stokes) is particularly informative: for similar weak residual levels ($R_{\text{weak}} \approx 4-15$), only the joint model achieves low $\mathrm{MMD}_\alpha\approx 0.07-0.13$, whereas the ablations are stuck around $0.22-0.28$, illustrating the concrete benefit of explicitly modeling $\alpha$ flows.

5. **Good use of figures to illustrate qualitative behavior and trade-offs.**  
   - Figure 2 (Darcy) clearly visualizes the effect of regularization: with $\lambda_f=1$ the pressure $x^{\mathrm{ft}}$ is denoised while $\alpha^{\mathrm{ft}}$ remains close to $\alpha^{\mathrm{base}}$, whereas without regularization, you get much more coherent permeability but lose sample-specific details. This directly supports the text’s claim about trajectory-level fidelity vs aggressive denoising.  
   - Figure 3a–b quantitatively shows how $\lambda_x,\lambda_\alpha$ and $\lambda_f$ move the model along the residual–diversity and residual–MMD trade-off frontiers, which is exactly the kind of controllability one wants in practice.  
   - Figure 6 (macaw Pop Art) demonstrates that the natural-image extension with parametric recoloring is not a toy: vanilla AM gives slightly stylized macaws, but the joint model produces significantly more vibrant Pop-Art-like palettes, confirming that coupling $\alpha$ dynamics can systematically explore new appearance modes.

6. **Nontrivial comparison baselines and reasonably fair evaluation.**  
   - The authors compare to (i) the base FM model, (ii) a PBFM-style training-time physics-regularized baseline, and (iii) ECI-style inference-time constraint enforcement (for elasticity), plus two ablations of AM without joint flow.  
   - Tables 10–13 in Appendix F.1 show full grids of hyperparameters, not just cherry-picked results, and scatter plots (Figures 7, 9, 10, 11, 31) make it clear how configurations populate the residual–MMD plane for all methods.

7. **Practicality and efficiency for PDE scenarios.**  
   - The Darcy fine-tuning takes only 20 gradient steps (<15 minutes on a single L40S) and sampling afterward costs the same as the base model, unlike inference-time projection schemes that add cost per sample.  
   - The architecture modifications only add ~6M parameters to a 19M U-FNO backbone, so the overhead is modest.

## Weaknesses

1. **Reliance on a pre-trained inverse predictor $\varphi$ whose role is conceptually delicate and not fully analyzed.**  
   - The entire joint evolution construction (Section 3.2) hinges on $\varphi$, both for computing $\hat{\alpha}_1 = \varphi(\hat{x}_1)$ and defining $v_{t,\alpha}^{\text{base}}$ and $v_{t,\alpha}^{\text{reg}}$. However, there is no systematic study of what happens when $\varphi$ is biased or multimodal.  
   - In Darcy, Figure 2 and the text explicitly note that $\alpha^{\mathrm{base}}$ is “artifact-ridden” and fragmented due to noise, yet $\hat{\alpha}_1^{\text{base}}$ from exactly that predictor is used as the anchoring direction in $v_{t,\alpha}^{\text{reg}}$ and as a surrogate base flow. There is no guarantee that this direction is physically meaningful.  
   - A more rigorous discussion of identifiability and error propagation is missing: for instance, under what conditions can the control $u_{t,\alpha}$ “override” systematic bias in $v_{t,\alpha}^{\text{base}}$, and how does $\lambda_f$ trade off between fitting the PDE residual and tracking a potentially wrong base parameter estimate?

2. **Theoretical justification of the reward-tilted target distribution becomes murky with nonzero running cost $f(\alpha)$ and surrogate base drifts.**  
   - The adjoint-matching result from Domingo-Enrich et al. (2025) guarantees consistency with $p_r(x)\propto e^{\lambda r(x)}p(x)$ only when $f=0$ and the base drift is the original generative process. Here, the authors add a nonzero running cost  
     \[
       f(\alpha) = \lambda_f \|v_{t,\alpha}^{\text{ft}}(\alpha) - v_{t,\alpha}^{\text{reg}}(\alpha)\|^2,
     \]  
     and also change the state space to $(X_t,\alpha_t)$ with a surrogate $b_{t,\alpha}^{\text{base}}$ defined from $\varphi$.  
   - There is no formal analysis explaining what distribution this control problem converges to with $f\neq 0$ and with a base drift that itself depends on $\varphi$ and on $\hat{x}_1 = x_t + (1-t)v^{\text{base}}_{t}(x_t)$. In particular, the connection to a simple “PDE residual-tilted” distribution (with reward $r=-g$) is no longer obvious, yet the narrative seems to lean on this intuition.  
   - I do not consider this fatal, but the paper should be more explicit that once $f\neq0$ and a surrogate $\alpha$-flow is introduced, the warranty of exact tilt disappears and we are in a heuristic but empirically effective regime.

3. **Experimental baselines for inverse problems are somewhat shallow.**  
   - The paper positions itself as solving inverse problems (Section 1, 3.2) by jointly generating $(x,\alpha)$, yet the baselines are primarily physics-constrained generative samplers for $x$ (FM, PBFM, FM+ECI) plus AM variants that only partially include $\varphi$.  
   - There is no comparison to standard Bayesian or physics-informed inverse methods, e.g., PINN-based inverse solvers, conditional diffusion/flow models trained on $(x,\alpha)$ pairs, or variational surrogates for $p(\alpha|x)$ even in settings where paired data could be simulated (Darcy, Helmholtz, Stokes). This makes it hard to judge whether the method is actually competitive as an inverse solver, beyond internal ablations.  
   - For instance, in Darcy, we never see any quantitative measure like $\mathbb{E}\|\alpha_{\text{true}}-\hat{\alpha}\|^2$ against a classical baseline, even though the dataset generation process in Appendix B.1 makes ground truth $\alpha$ available.

4. **Metrics and trade-offs are not clearly tied to target distributions or downstream physics tasks.**  
   - The chosen evaluation metrics (relative $R_{\text{weak}}$, $R_{\text{strong}}$, $\mathrm{MMD}_x$, $\mathrm{MMD}_\alpha$) are sensible, but the interpretation is somewhat floating. For example, in Table 10 for Darcy, AM with $\lambda_x=20\text{k}, \lambda_f=0$ achieves very low weak residual (0.915) and moderate MMDs; but we never see whether this actually improves any physically relevant quantity (e.g., effective permeability, flux errors, or posterior predictive accuracy).  
   - In Figure 3a, the “SSIM diversity” metric is used to quantify diversity in $\alpha$, but diversity per se is not necessarily good or bad; what matters is matching the reference distribution. The paper notes that diversity decreases at high $\lambda_x,\lambda_\alpha$, but does not clarify how far that is from the true distribution of $\alpha$.  
   - For the inverse-problem framing, it would be helpful to explicitly consider posterior coverage: do samples from the joint model correctly surround the ground-truth parameters given sparse observations, or are they overly concentrated? Figure 4 is compelling qualitatively but remains anecdotal.

5. **Weak explanation and ablation of the memoryless noise scaling $\kappa$ and time-tilting in practice.**  
   - Theoretical Lemma 1 shows that any $\kappa<1$ yields a memoryless schedule, but in practice the authors always pick $\kappa=0.9$ for PDEs “to mitigate blow-ups near $t\to 0$” (Section 3.3, Appendix D.4, Tables 7–8) and introduce an additional time-tilting parameter $q=0.9$.  
   - There is no empirical ablation on $\kappa$ or $q$, so it is unclear whether they materially affect performance, stability, or convergence speed. Given that $\kappa$ is advertised as a “numerical stabilisation knob” and a central technical point, one would expect at least a small study, e.g., residual vs $\kappa$ for Darcy.  
   - Moreover, Eq. (1) uses $\eta_t=\gamma_t(\frac{\dot{\beta}_t}{\beta_t}\gamma_t - \dot{\gamma}_t)$ with $\beta_t=t,\gamma_t=1-t$, then later they redefine $\eta_t = \frac{1-t+h}{t+h}$ as a heuristic. This change disconnects the scaling $\kappa$ from the originally derived formula; a brief explanation of how much this deviates from the theoretical $\eta_t$ and any impact on the memoryless property would be helpful.

6. **Some mathematical details are opaque or slightly inconsistent.**  
   - In the joint Lean Adjoint dynamics (Eq. (3)), the Jacobian blocks $J_{ij}$ are defined as $J_{ij} = \nabla_j b_{t,i}^{\text{base}}(X_t,\alpha_t)$. Since $b_{t,\alpha}^{\text{base}}$ is built from $\hat{\alpha}_1 = \varphi(\hat{x}_1)$ and $\hat{x}_1 = x_t+(1-t)v_t^{\text{base}}(x_t)$, $b_{t,\alpha}^{\text{base}}$ depends nontrivially on $x_t$ and on the base vector field $v_t^{\text{base}}$. The paper does not specify how gradients w.r.t. $x$ are handled in practice: are $J_{x\alpha}$ and $J_{\alpha x}$ computed via automatic differentiation through $\varphi$ and $v_t^{\text{base}}$, or are some dependencies stopped?  
   - In Algorithm 1, Step 27–28, the definition of drifts $b^{\mathrm{ft}}$ and $b^{\mathrm{base}}$ uses  
     \[
       b_{x,t}^{\mathrm{ft}} = v_{x,t}^{\mathrm{ft}} + \frac{\sigma(t)^2}{2\eta(t)}\left(v_{x,t}^{\mathrm{ft}} - \frac{1}{t+\varepsilon}x_t^{\mathrm{ft}}\right)
     \]  
     which heuristically replaces $\frac{\dot{\beta}_t}{\beta_t}$ by $\frac{1}{t+\varepsilon}$ and $\eta_t$ by $\frac{1-t+h}{t+h}$. It works empirically, but the mathematical connection to Eq. (1) is loosened. A short derivation tying these approximations back to the original continuous-time formulas would clarify the soundness of this choice.  

7. **Limited discussion of failure modes and cases where PBFM / ECI might be preferable.**  
   - In Stokes, PBFM “fails to converge,” leading to very large strong residuals; the paper notes that training-time physics with misspecification is challenging. However, there is no deeper diagnostic (e.g., showing sample fields in the main text) or discussion of when PBFM might still be superior (e.g., no misspecification, small noise, or correctly specified PDE).  
   - Similarly, for tasks where exact hard constraints are mandatory, inference-time projection methods like ECI might be more interpretable than adjusting the generative distribution. The paper alludes to this in Related Work, but the empirical sections do not clearly spell out such trade-offs.

8. **Missing important recent related work on flow-matching inverse problems and fine-tuning.**  
   - Recent work on physics-constrained inverse problems with conditional flow matching (e.g., *Dasgupta et al., “Solving Physics-Constrained Inverse Problems with Conditional Flow Matching”, 2026*) appears highly relevant: it also targets inverse PDE problems via flow models, and should be contrasted with the proposed approach that avoids paired training data by using $\varphi$ and adjoint matching.  
   - Likewise, *Thorkelsdottir & Banerjee, “Gradual Fine-Tuning for Flow Matching Models”, 2026* propose a fine-tuning scheme for flow-matching models that seems methodologically aligned with the present post-training strategy. It should be discussed in Section 2, especially concerning stability and distribution shift during fine-tuning. Their absence weakens the positioning of this paper within the rapidly growing FM fine-tuning literature.

## Potentially Missing Related Work

1. **A. Dasgupta, A. Fardisi, M. Aminy, “Solving Physics-Constrained Inverse Problems with Conditional Flow Matching”, 2026.**  
   - Directly deals with inverse PDE problems using conditional flow matching. It is strongly related to the stated goal of generative modeling for inverse problems. Should be cited and contrasted in Section 2 (“Flow-Matching Models for Simulation and Inverse Problems”) and in Section 4 when discussing inverse capabilities (e.g., Darcy guidance, Helmholtz, Stokes): their method trains on paired $(x,\alpha)$, whereas this paper uses a post-hoc inverse predictor and adjoint-matching; a discussion of trade-offs (data requirements, accuracy) is needed.

2. **G. Thorkelsdottir, A. Banerjee, “Gradual Fine-Tuning for Flow Matching Models”, 2026.**  
   - Proposes a fine-tuning framework for FM models that is conceptually related to this paper’s adjoint-matching-based post-training. It should be referenced in the fine-tuning discussion in Section 2 and compared conceptually in Section 3.3, especially regarding stability, distributional drift, and computational overhead.

## Questions

1. **Robustness to inverse-predictor misspecification.**  
   - Can you provide an experiment where $\varphi$ is intentionally degraded (e.g., trained on fewer samples or with added noise) to quantify how errors in $\hat{\alpha}_1$ propagate into the joint flow and final $(x,\alpha)$ distribution? This would help assess how critical the quality of $\varphi$ is.

2. **Theoretical interpretation with nonzero running cost $f$.**  
   - With $f(\alpha)=\lambda_f\|v^{\mathrm{ft}}_{t,\alpha}-v^{\mathrm{reg}}_{t,\alpha}\|^2$, what distribution do you expect the process to converge to? Can you at least provide an informal derivation showing that the terminal distribution solves a KL-regularized control problem, or clarify that the guarantee of a simple exponential tilt no longer holds?

3. **Quantitative inverse-accuracy metrics.**  
   - For at least one PDE benchmark (e.g., Darcy or Helmholtz, where $\alpha$ is low-dimensional or piecewise constant), could you report a simple parameter error metric, e.g., $\mathbb{E}[\|\alpha_{\text{true}}-\hat{\alpha}\|_2]$ across samples, and compare to a baseline inverse method (e.g., a PINN inverse solver or a supervised conditional FM trained on $(x,\alpha)$)? This would greatly clarify the method’s practical utility as an inverse solver.

4. **Ablations on $\kappa$ and time-tilting $q$.**  
   - Could you include or describe experiments varying $\kappa$ and $q$ (at least on Darcy) to show how they affect stability and final residuals? This would justify the emphasis on Lemma 1 and the heuristic time-tilting.

5. **Adjoint Jacobian implementation details.**  
   - In Eq. (3), are $J_{xx},J_{\alpha x},J_{x\alpha},J_{\alpha\alpha}$ computed via automatic differentiation through $\varphi$ and $v^{\text{base}}$, or do you detach some dependencies for efficiency? If the latter, which blocks are approximated, and have you observed any impact on performance?

6. **When is PBFM preferable?**  
   - Can you comment on regimes where a training-time physics-constrained method like PBFM might outperform your post-training approach (e.g., no misspecification, more training data, or simpler PDEs)? Some guidance here would help practitioners choose between the methods.

## Flag For Ethics Review

- No ethics review needed.  

## Details Of Ethics Concerns

N/A.

## Soundness Rating

3: good.  
The core adjoint-matching and weak-residual machinery is technically solid, and the experimental evidence is generally convincing. Some theoretical guarantees are loosened by heuristics ($f\neq 0$, surrogate $\eta_t$, reliance on $\varphi$), but these are acknowledged and seem empirically benign.

## Presentation Rating

3: good.  
The paper is dense but mostly clear, with helpful figures (especially Figures 1, 2, 3, 5, 6) and detailed appendices. Some conceptual caveats (effect of $f$, role of $\varphi$) and the relationship between theory and heuristics could be explained more candidly.

## Contribution Rating

3: good.  
The work combines a careful extension of adjoint matching to joint state–parameter flows with a well-executed weak-form PDE residual framework and substantial experiments. It is not a fundamental theoretical breakthrough, but it is a meaningful and useful advance for physics-constrained generative modeling and inverse problems.

## Overall Rating

8: Accept, good paper (poster).  
The method is well-motivated, technically coherent, and empirically well supported across multiple challenging PDE scenarios, providing a practically useful framework for physics-constrained fine-tuning and joint state–parameter generation. Remaining issues mostly concern theoretical neatness under added heuristics and more thorough inverse-problem baselines, but they do not undermine the main contributions.

## Reviewer Confidence

4: confident.  
I am familiar with flow matching, physics-informed learning, and PDE-constrained generative models, and I carefully examined the math and experiments, though I did not attempt to re-derive every implementation heuristic.