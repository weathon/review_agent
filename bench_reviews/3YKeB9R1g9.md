## Summary
This paper investigates the phenomenon of training loss curve (TLC) collapse—where normalized loss trajectories align across model sizes—in the context of large-scale LLM training under practical scaling recipes. Building on Qiu et al. (2025), it identifies the AdamW timescale (τ), tokens-per-parameter (TPP) ratio, and learning rate schedule as the key controls for collapse. The authors demonstrate that collapse emerges when τ is set optimally for a given, fixed TPP, framing it as a signature of compute-efficient training. They apply this insight in two ways: using collapse residuals for early training diagnostics and enabling early stopping in hyperparameter tuning. These contributions are validated by training the Celerity model family (300M to 3.9B parameters), which exhibits collapse and achieves competitive compute efficiency.

## Strengths
- **Substantial empirical validation at relevant scale:** The paper provides extensive experiments across a meaningful range of model sizes (up to 3.9B parameters) and TPP settings (20, 80, 234). The successful training of the Celerity family serves as a concrete, large-scale proof-of-concept that collapse can be achieved under practical conditions, moving beyond small-scale theoretical demonstrations.
- **Clear identification of governing factors and practical utility:** The paper clearly isolates and explains the three key factors (τ, TPP, LR schedule) controlling TLC shape. It delivers actionable insights: the demonstration that fixing τ (not just λ) during hyperparameter sweeps preserves curve ordering (Fig. 7) is a valuable guideline for practitioners, and the use of collapse residuals to diagnose a numerical instability (Fig. 1, right) is a compelling, real-world case study.
- **Novel application for efficient hyperparameter tuning:** The proposed method for early stopping in hyperparameter tuning (Sec. 5) is a clever and well-motivated application of collapse. The results show that by aligning partial runs to a predicted normalized curve, the best hyperparameter setting can be identified after only 10-30% of training, offering potential for significant compute savings.

## Weaknesses
### Major
- **Insufficient validation of the early-stopping method's generality:** The proposed early-stopping procedure is validated only for tuning the weight decay λ (which directly controls the surrogate model's parameter τ). The paper does not test the method on hyperparameters that are *not* directly captured by the model's parameters (e.g., architectural choices like width-to-depth ratio, Adam β parameters, or LR schedule shape). This limited scope leaves the method's practical utility for broad hyperparameter search uncertain. (Sec. 5, Figs. 8 & 9)
- **Lack of quantitative rigor in defining and measuring "collapse":** The claim of collapse is supported primarily by visual alignment. The paper lacks a quantitative, statistical measure of alignment quality (e.g., mean squared error between curves with confidence intervals across seeds) and an analysis of variance. This makes it difficult to assess the strength and consistency of the collapse phenomenon, especially when deviations are noted (e.g., for 20 TPP and late in 234 TPP training). (Figs. 1, 6)
- **Under-explained breakdown of collapse at high TPP for large models:** The paper notes that at 234 TPP, "divergences appear late in training for larger models" and attributes this vaguely to disproportionate improvement on training data versus held-out data. This observation challenges the universality of the collapse claim under the stated conditions, yet no deeper analysis (e.g., of overfitting dynamics, sharpness, or gradient statistics) is provided to explain this breakdown. (Sec. 4, Fig. 1 middle)

### Minor
- **Theoretical foundation could be deeper:** While the connection to a noisy quadratic model (Appendix B.3) provides intuition, the derivation of the full collapse condition—specifically the interaction of τ, TPP, and an arbitrary LR schedule—remains largely empirical and is not formalized within the µP framework. The proposed parametric surrogate model (Eq. 4) is fitting but not rigorously derived from first principles.


## Nice-to-Haves
- Extending validation of the collapse phenomenon to a significantly different model architecture (e.g., Mamba, MoE) or optimizer to bolster claims of generality.
- A more comprehensive downstream evaluation of the Celerity models, including reasoning benchmarks (e.g., MMLU) and analysis of fine-tuning propensity, to better contextualize its competitiveness.

## Removed Points
*These points are flagged to be removed; treat them with caution.*

**Strengths Removed:**
- "The paper is well-written." (Generic, applies to any paper)
- "The topic is important." (Generic, applies to any paper)

**Weaknesses Removed:**
- **"Structural (Claims vs. Evidence for Collapse in 'Practical Scaling Recipes'):"** The harsh critic argues the "practical scaling recipe" is not clearly described. However, the paper explicitly states the recipe: fix TPP, set τ optimally for that TPP, and use a fixed LR schedule. The scaling of batch size and other parameters is handled via standard µP/CompleteP, which is referenced and for which details are provided in the appendix (e.g., Table 2, Appendix C.2). The core claim of demonstrating collapse under these conditions is supported by the Celerity experiments. This criticism misunderstands the paper's stated methodology.
- **"Evidential (Causal Link Between Optimal Hyperparameters and Collapse):"** The harsh critic demands a direct sweep showing optimal τ simultaneously minimizes final loss *and* maximizes collapse. The paper establishes this link indirectly but reasonably: Sec. 3 shows TLC shape is controlled by τ; prior work (Bergsma et al. 2025a, cited) shows optimal τ is a function of TPP; therefore, fixing τ at this optimal value (as done for Celerity) should yield both good final loss and collapse. Fig. 6 shows collapse occurs at these chosen τ values. The paper does not manufacture the optimality claim; it builds directly on cited, established results.
- **"Limited scale of empirical validation... only up to 3.9B":** While testing at larger scales would be interesting, the paper's scale (100M to 3.9B, a 40x range) is substantial and appropriate for the study. Demanding validation at "hundreds of billions" of parameters is outside the scope and resources of most academic research and is not a valid criticism of the work presented.
- **"Insufficient comparison with alternative methods (e.g., AdaSGD)":** The paper's scope is understanding collapse under standard, widely-used LLM training practices (AdamW, µP). Requiring comparison with a different optimizer family is scope creep and not necessary to evaluate the paper's core claims.
- **Vague criticisms about "Conflation of 'final loss' normalization methods" and "Sensitivity analysis of 'early-align'":** The paper clearly discusses two strategies ("Estimate" and "Early-align") and justifies its choice. A full sensitivity analysis is a detail that, while nice to have, is not a core flaw.
- Nitpicks about figure parsing artifacts. (Formatting issue from PDF extraction, not the paper's fault).

## Suggestions
- Conduct a validation experiment for the early-stopping method by applying it to tune a hyperparameter *not* directly tied to τ (e.g., the Adam β2 parameter or an architectural hyperparameter) to demonstrate broader applicability.
- Add a quantitative metric (e.g., average MSE of aligned curves) to the collapse figures and report the variance across model sizes/seeds to strengthen the empirical claim.
- Augment the discussion of the late-training divergence at 234 TPP with a brief, focused analysis (even if post-hoc) of the train/validation loss gap or related metrics to provide a more concrete explanation.