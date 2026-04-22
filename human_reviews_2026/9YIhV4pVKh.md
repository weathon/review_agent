# Controllable Representation Learning for Time-series Analysis

- Avg Score: 2.00
- Decision: Reject
- Scores: 2, 2, 2, 2

## Abstract
Representation learning for time series typically relies on reliable anchors: smooth input signals or dense supervision that constrain latent dynamics. When both are degraded due to noise, missing values, or irregular sampling, hidden states will drift and standard methods will collapse. To tackle this problem, we propose a conceptual shift: treating representation learning itself as a control problem. Our framework, Neural Feedback Control (NFC), actively regulates latent trajectories using confidence-weighted pseudo-observations and pseudo-labels, combining pseudo data-based controllers with continuous-time dynamics and residual-based feedback. This design transforms latent space evolution from passive inference into a controllable process. In contrast to Neural ODEs/CDEs, which model latent dynamics without stability guarantees, and predictive coding approaches that propagate errors without explicit contraction control, NFC provides a feedback-driven mechanism with provable stability under partial observability.  Theoretically, we prove that under mild conditions, NFC guarantees exponential decay of output error to a bounded region, providing a certified stability guarantee. Every module in NFC (pseudo-signal generation, confidence weighting, and feedback penalties) plays a role in a single closed-loop control system, 
transforming representation learning into active regulation rather than passive inference. Empirically, NFC achieves substantial robustness gains: over 50\% lower forecasting error on power load datasets and more than 10\% higher accuracy on human activity dataset with 30\% missing data. These results highlight task-aware latent control as an effective approach for stabilizing representation learning when conventional anchors fail.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
For complex time series analysis tasks, the authors propose a conceptual shift: treating representation learning itself as a control problem. The proposed framework, Neural Feedback Control (NFC), actively regulates latent trajectories using confidence-weighted pseudo-observations and pseudo-labels, integrating a pseudo-data-based controller with continuous-time dynamics and residual-based feedback. This design transforms the evolution of the latent space from passive inference into a controllable process. Experiments demonstrate the superiority of the proposed model.

### Strengths
S1. The author's perspective is insightful, introducing NFC into time series analysis tasks to enhance robustness.

S2. The author's experiments cover both forecasting and classification tasks.

S3. Under limited benchmarks, the author's model achieved advanced performance.

### Weaknesses
1.  Inconsistent citation formatting throughout the paper.

2.  Some sentences are difficult to interpret.  For example:
> “Instead of assuming that useful features will emerge from corrupted inputs and labels, we treat latent states as dynamical variables whose trajectories must be actively regulated.”
It is unclear why making the trajectories of these variables *traceable* or *controllable* would specifically address the challenges at hand.  The benefit of this perspective needs clearer justification.

3.  The paper is extremely hard to follow, largely due to an overuse of undefined technical terms borrowed from control theory without sufficient explanation.  While I appreciate the conceptual novelty of introducing control-theoretic ideas into time series analysis, this does not justify the uncritical blending of terminology from two distinct fields.  For instance:
- What does “terminal” refer to?  Does it relate to edge devices or something else?
- Does “multiple steps” mean time steps?
- The claim that “by explicitly modeling how errors compound over time, NFC learns to correct instabilities in real time” raises questions: Is there a specific optimization objective designed to enable such real-time error correction?  This mechanism is not adequately explained.

4.  The introduction focuses overwhelmingly on the NFC framework itself, rather than on the unique challenges of applying it to time series analysis.  A more balanced discussion of domain-specific difficulties (e.g., irregular sampling, missing data, noise) would better motivate the proposed approach.

5.  Subjectively, the introduction and problem formulation feel disconnected.  The introduction emphasizes complex scenarios such as missing values and noisy observations, yet the formal problem definition narrows the scope exclusively to *irregularly sampled* time series.  While I understand that random masking is a common proxy for irregular sampling in experiments, this creates an imbalance—making the paper feel “top-heavy” with broad claims but a narrow formalization.

6.  The term “composite signals” appears in the problem definition without clarification.  What is its precise meaning in this context, and why is it included in the formal setup?

7.  The related work section is placed too late in the paper.  Readers must sift through most of the manuscript before encountering a discussion of recent advances, which hinders understanding of the paper’s positioning within the field.

8.  I vehemently object to the authors’ decision to relegate critical experimental details—including baseline descriptions, hyperparameter configurations, and dataset specifications, entirely to the appendix. Under ICLR reviewing policies, reviewers are explicitly not required to consult supplementary materials, rendering the main paper fundamentally incomplete and scientifically inadequate. By offloading essential methodological and empirical information into the appendix, the authors effectively circumvent page limits while depriving readers and reviewers of the necessary context to properly evaluate the work’s validity, reproducibility, and contribution. This practice constitutes an unfair exploitation of supplementary space, disadvantaging authors who conscientiously adhere to submission guidelines.

9.  Appendix C is empty and appears to be an unnecessary placeholder.

10.  The authors should clarify the criteria used to select baseline models.  The experiments omit numerous state-of-the-art methods:
- For forecasting: PatchTST, TimeNet, CycleNet, DUET, TimeXer, etc.
- For missing-value-aware forecasting: BRITS, CSDI, BiTGraph, PriSTI, etc.
- For irregular time series: t-PatchGNN, ACSSM, etc.
The absence of these relevant baselines significantly weakens the credibility of the reported results.

11.  The experiments rely solely on numerical metrics and lack qualitative case studies that connect back to the core claims in the introduction.  For instance:
- What do the learned latent trajectories look like?
- How exactly does NFC achieve robustness to missing data?
Without such analysis, the empirical evaluation feels disconnected from the paper’s conceptual contributions.

### Questions
See the Weaknesses.

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces Neural Feedback Control (NFC), a framework that reformulates representation learning for time series as a control problem. The authors aim to stabilise latent dynamics under degraded supervision—such as noisy, missing, or irregularly sampled data—by introducing feedback-based regulation in latent space. The idea is conceptually interesting and potentially valuable, and the paper presents both theoretical stability guarantees and empirical results suggesting improved robustness compared to existing approaches like Neural ODEs and predictive coding.

However, the paper suffers from several major issues that significantly undermine its contribution. Many of the empirical claims are not supported by statistical analysis, and several results are overstated or even inaccurate upon inspection. The methodology and mathematical exposition are unclear, with inconsistent notation and missing key definitions (e.g., ELBO, context variables). Figures are poorly formatted and difficult to interpret, and the lack of quantitative evaluation in critical areas weakens the paper’s empirical credibility. Moreover, the writing lacks clarity and structure—particularly in motivating the problem, defining key concepts (e.g., “collapse,” “degraded supervision”), and explaining the proposed method.

Overall, while the paper proposes an interesting conceptual angle, the lack of rigour in evaluation, clarity in presentation, and precision in mathematical formulation make it not yet suitable for publication. Substantial revisions are needed to improve the clarity, correctness, and credibility of both the theoretical and experimental components.

### Strengths
The authors are working on an interesting problem, and with major modifications, I'm sure they can have a paper that is of value to the community.

### Weaknesses
This paper has several major weaknesses that currently prevent it from being publishable.

1. Lack of statistical rigour and unsupported claims
    - Many claims in the results section are not backed by statistical analysis or, at times, even quantitative evidence.
        - In Table 1, the authors bold their method across all rows, but the confidence intervals show that most results are not statistically different from Chronos. A paired t-test or similar analysis should be performed. It is also unclear how many random seeds were used to compute means and standard deviations.
        - On Line 333, the authors state:
            > “The results confirm that supervised latent control is especially effective in degraded supervision settings, yielding robust forecasting performance when input observations and labels are partially missing.”
            
             However, the presented results do not substantiate this claim.

    - On Line 436, the claim:
        > “Despite severe degradation, NFC faithfully aligns reconstructed trajectories with the original high-quality patterns, while mitigating the adverse effects of corrupted signals.”

        cannot be justified based solely on a figure. Moreover, the “Missing Data” reconstruction appears very noisy. Quantitative comparisons are needed before drawing such conclusions.

    - On Line 458, the authors claim:
        > “Our method consistently drives the loss downward with stable convergence, whereas ODE–RNN and ContiFormer exhibit slower and less reliable decay.”

        In fact, all methods show decreasing training loss, and if the x-axis were extended, they would likely converge to similar values. There is no clear evidence of instability in ODE–RNN or ContiFormer; the only valid observation is that the proposed method converges faster. Furthermore, the focus on training rather than validation/test metrics is concerning.

    - On Line 364, the claim:
        > “Our approach better preserves the trajectory shape and yields tighter uncertainty bounds…”

        is difficult to verify visually from Figure 3. This should be quantified over multiple seeds, and significance should be reported. Additionally, tighter uncertainty bounds are not inherently better—evaluation should include a likelihood-based metric (e.g., negative log-likelihood) to assess calibration.

    - On Line 355, the authors state:
        > “At a missing ratio of 0.7, RNN-decay and RNN-Δt exhibit test errors more than twice as large as ours.”

        This is inaccurate. Based on Table 1, RNN-Δt’s error is less than twice as large. Claims in the paper must be factually correct.

2. Clarity and presentation

- The paper lacks clarity in both motivation and exposition. The Introduction and Problem Formulation sections do not clearly explain the motivation or intuition behind the method. The absence of a Background section—particularly one covering Neural ODEs and related concepts—makes the paper difficult to follow.

- Terms such as “preventing collapse” and “degraded supervision” (e.g., Line 40) should be explicitly defined and supported by citations. The introduction currently focuses too heavily on implementation details rather than motivation.

- The Experiments section also begins abruptly; an introductory paragraph outlining the research questions or evaluation goals would greatly improve readability.

3. Mathematical clarity

Section 3’s mathematical presentation is imprecise and difficult to follow:

- Several equations (e.g., Lines 181, 205, 214) are written inline and should instead be displayed and numbered.
- On Line 230, a previous equation is referenced vaguely as “defined earlier” instead of “see Equation (X)”.
- The ELBO for VAE training should be explicitly defined at Line 169.
- The notation for $l_{\phi}^2(\cdot)$ is inconsistent: Line 195 describes it as a categorical distribution, while Line 203 defines it as producing Gaussian parameters. This needs correction.

4. Figures

The figures are difficult to interpret and poorly formatted:

- Figure 1: Hard to read; the caption lacks explanation. The top-left subfigure is too small. The figure should be placed at the top of the page.
- Figure 3: Text is too small; the caption should explain what is being shown.
- Figure 4: Missing legend; unclear what colours and lines represent. Placement is poor (floating under a subsection).
- Figure 5: Text too small. Clarify what “VAE 95%” means (is it a 95% CI?). Label the confidence levels numerically. Define “normalized value” on the y-axis.
- Figure 6: Caption is vague (“Results in ablation studies”). Specify dataset, number of seeds, and loss type (regression/classification). The results appear to be from a single seed—averaging over multiple runs is required.

5. Minor corrections and formatting

- The appendix is incomplete: Appendix C (notation) is empty, and the title of Appendix G should be moved to the correct page.
- Textual vs. parenthetical citations are frequently misused (e.g., Lines 37, 41, 53, 59, 61, 64, etc.).
- Line 120: “See” → “see”.
- Line 157: Define the context $\mathcal{C}$.
- Line 219: “equation” → “Equation”.
- The abstract is too long and should be shortened.

### Questions
See weaknesses

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes Neural Feedback Control (NFC), a novel framework for time-series representation learning.
NFC treats the latent dynamics as a controlled system.
Instead of relying completely on passive encoder-decoder inference, NFC introduces an active feedback controller that modulates latent dynamics using pseudo-observations, pseudo-labels, and confidence weighting.
For both classification and forecasting tasks, NFC outperforms various ODE-based models.

### Strengths
1. The idea of reframing latent representation learning for time series as a control problem is novel. By explicitly modeling a feedback control signal in the latent dynamics, NFC can stabilize predictions even under high levels of missing inputs, which is a major challenge for existing methods.

2. Authors provide a formal stability guarantee (I admit, I assume the theorem is correct) 

3. The experiments on both classification and forecasting tasks demonstrate that NFC consistently outperforms several state-of-the-art ODE-based models, such as ODE-RNN, NCDE, and Latent ODE.

### Weaknesses
1. Several key components of the paper are under-specified, making it hard to clearly understand the working of the model

(i) What exactly is the input to the VAE? The authors mention masked history. Does it include the historical timepoints or only historical observations and labels?

(ii) The paper mentions predicting controls for the next $M$ time steps, but does not specify how 
$M$ is chosen, whether it varies across datasets, or why a multi-step horizon is used instead of a single-step control.

(iii) Because the control horizon $M$ is not defined and the procedure for computing multi-step controls is unclear, it is ambiguous how the latent states $h_t$ are computed at each step. Is the latent ODE integrated step-by-step with updated controls or over a fixed horizon with precomputed controls?

(iv) The paper does not describe the VAE architecture (encoder/decoder) or how it summarizes historical observations into a latent context.

(v) Hyperparameters and training details missing:
Latent dimensions, VAE latent size, and ODE solver settings are not reported.
Training hyperparameters (learning rate, batch size, optimizer, number of epochs, loss weights) are missing.
Without these details, results are not reproducible.

2. Algorithm 1 does not help in clearly understanding the model. 

3. NFC outperforms ODE-based models, but comparisons with non-ODE state-of-the-art methods (eg, [1] for classification, [2,3] for forecasting) are needed to validate its practical utility. While good to see the comparison with Contiformer, it is useful to see more recent baselines.

[1] Luo, Yicheng, et al. "Knowledge-empowered dynamic graph network for irregularly sampled medical time series." Advances in Neural Information Processing Systems 37 (2024): 67172-67199.

[2] Zhang, Weijia, et al. "Irregular multivariate time series forecasting: A transformable patching graph neural networks approach." Forty-first International Conference on Machine Learning. 2024.

[3]  Yalavarthi, Vijaya Krishna, et al. "Grafiti: Graphs for forecasting irregularly sampled time series." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 38. No. 15. 2024.

### Questions
See weakness

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposed the Neural Feedback Control with differetial equation to deal with time series analysis with noise, missing values, or irregular sampling. The main contributions of the paper include framing latent representation learning as an active control problem, providing a theoretical guarantee that output error toward a bounded region and can deal with time series forecasting with missing values, sparsity, and irregular sampling.

### Strengths
- This paper provides Neural Feedback Control with differetial equation to deal with time series analysis with noise, missing values, or irregular sampling. 
- The experiments of the proposed method are presented clearly and are easy to follow.

### Weaknesses
My main concerns include:
- Could the authors clarify how the VAE loss and the task-specific loss are jointly optimized? Specifically, how are these two terms combined in the overall objective? The manuscript does not currently define the total loss combining above loss.
- The current mathematical presentation is confusing. Lines 165 and 181 of the manuscript both introduce the variable u_i, but the mathematical definitions differ. In addition, line 181 introduces a variable z without a prior definition. The authors need to clarify the meaning of z and resolve the notational inconsistency for u_i.
- The manuscript claims that NFC can handle forecasting analysis with missing data and data sparsity; however, several existing models (e.g., SSSD, CSDI) also address these challenges. What are the concrete advantages of NFC over these approaches? The paper currently lacks experimental comparisons against such baselines.
- Time-series foundation models can address data-sparse scenarios in a zero-shot manner, the current comparison against Chronos is insufficient. Please update the baselines by including the latest, top-performing zero-shot models from the GIFT-Eval leaderboard and add PatchTST to the comparisons.

### Questions
See Weaknesses

### Soundness
2

### Presentation
2

### Contribution
2
