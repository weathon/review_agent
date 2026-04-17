---
job_id: bc8026d8-854e-40de-8f55-2ce2a97dca8e
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: O75R5yyw4R.pdf
paper: Effective Test-Time Scaling of Discrete Diffusion through Iterative Refinement
main_score_norm: 0.6
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper proposes a new test-time scaling / reward-guided sampling method for discrete diffusion models, with both theoretical analysis and extensive experiments on language and image generation. This falls squarely within generative models, probabilistic methods, and representation learning, which are central ICLR topics.

## Minimum Quality
Pass ✅.  
The paper is written in English and has all required sections: Abstract, Introduction, Method (Sections 2–3), Experiments (Section 4), Results and Analysis (Sections 4.2–4.5, C), Related Work (Section 5), Conclusion (Section 6). The method is non-trivial (MTM-based refinement for discrete diffusion with a specialized kernel), experiments are fairly extensive, and there are no obvious fatal theoretical or empirical flaws that would justify a desk rejection.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I did not find any hidden prompts or instructions targeted at automated reviewers, nor any suspicious formatting that suggests prompt injection or steganographic manipulation.

---

# Expected Review Outcome:

## Summary

The paper proposes Iterative Reward-Guided Refinement (IterRef), a test-time scaling method for discrete diffusion models based on Multiple-Try Metropolis (MTM). The core idea is to iteratively refine each intermediate diffusion state via a reward-guided noising–denoising kernel, instead of only steering the forward denoising trajectory once. The authors design a specific MTM transition kernel and balancing function, prove convergence to a reward-aligned intermediate distribution under a reversibility assumption, and demonstrate strong empirical gains over prior inference-time guidance methods across diffusion language models (MDLM, LLaDA-8B) and a discrete image model (MaskGIT) under multiple reward functions.

## Strengths

1. **Clear conceptual advance within test-time scaling for discrete diffusion.**  
   The paper attacks a very relevant and currently underexplored problem: how to do test-time scaling / reward-guided generation for discrete diffusion, where tokens are “hard” and cannot be easily corrected later. Most prior work in this space uses SMC/importance sampling or trajectory search on a single denoising pass. IterRef’s idea of *iteratively re-noising and re-denoising a single intermediate state via MTM* (Section 3.1, Algorithm 2) is a nontrivial conceptual shift. It leverages the masked-token nature of discrete diffusion to enable “in situ” correction of already decoded tokens.

2. **Principled probabilistic framing with MTM and detailed balance.**  
   The method is not just a heuristic. Section 3.1 defines a specific transition kernel  
   \[
   K(x_t, x'_t) = \sum_{x_s} q(x_s\mid x_t)\,p_\theta(x'_t\mid x_s)
   \]
   and a balancing function  
   \[
   \lambda(x_t, x_t') = \frac{1}{p(x_t) K(x_t,x_t')\exp((r(x_t)+r(x_t'))/\alpha)}
   \tag{Eq. 2}
   \]
   which, together with Multiple-Try Metropolis, yields simple importance weights and acceptance probabilities in Eq. (3). Proposition 1 then states convergence of the resulting Markov chain to the reward-aligned target \(p^*(x_t)\), assuming reversibility of \(q\) and \(p_\theta\). The detailed derivations in Appendix D (especially D.2–D.4) show that the authors understand the MCMC machinery, and this helps elevate the work above ad-hoc “search over candidates.”

3. **Strong and consistent empirical performance across settings.**  
   The experimental section is broad for a paper on discrete diffusion and inference-time guidance.  
   - **Figure 2(a)** (MDLM) shows that IterRef often matches or exceeds baselines’ best reward scores with an order of magnitude fewer NFEs; for example, for Sentiment and CoLA, IterRef at \(2T\) NFEs is as good or better than all baselines at \(32T\).  
   - **Figure 2(b)** (LLaDA-8B) shows similar advantages for larger models; on Toxicity and Perplexity, the red IterRef curve dominates the baselines for almost all compute levels, and the gains widen at higher NFEs.  
   - **Table 1** (MaskGIT + CLIPScore) shows that IterRef wins across all compute budgets (1–16) and by nontrivial margins; e.g., at cost 16, IterRef improves CLIPScore from 34.8 (best baseline FK) to 35.8.  
   - The improvements are also reflected in human-aligned metrics: **Table 5** shows IterRef selected as best for goal alignment 42.1% of the time vs. ≤18.4% for baselines, and **Tables 4 & 8** give better ImageReward scores, suggesting the reward gains are not just gaming CLIPScore.

4. **Insightful analysis of where and how to refine.**  
   Section 4.4 offers some genuinely interesting insights into discrete diffusion dynamics.  
   - **Figure 4** and **Table 3** empirically study the tradeoff between the number of MTM iterations \(k\) and the number of candidates \(N\). Both in MDLM (Figure 4) and LLaDA (Table 3), increasing \(k\) is more beneficial than increasing \(N\) under the same cost, supporting the paper’s narrative that iterative refinement is more effective than just more particles.  
   - **Table 2** analyzes which timesteps to apply IterRef on MDLM. Applying IterRef toward the *later* stages (e.g., 0.1T) often outperforms early application (0.9T), and in some cases even beats an even distribution. This is in contrast to continuous diffusion where early steps dominate, and it gives a concrete, data-backed insight into discrete diffusion behavior.

5. **Safety alignment case study with LLaDA-8B.**  
   Section 4.5 presents a focused detoxification study on LLaDA-8B using RealToxicityPrompts. **Figure 5(a)** shows that IterRef achieves <10% toxicity already at 4× budget and maintains a ~10% gap over baselines at higher budgets. **Figure 5(b)** gives qualitative examples illustrating how IterRef steers generations toward benign contexts (e.g., framing toxic seeds as reported quotes). This is a nice, concrete application where inference-time scaling is practically useful.

6. **Figures and qualitative examples are informative.**  
   - **Figure 1(a)** effectively contrasts single-pass importance sampling and SMC with IterRef’s iterative refinement, clearly illustrating the “refine around a state” picture; the blue vs gray nodes, and the labeled “Iterative Refinement” bracket, make it visually obvious how the method differs conceptually.  
   - **Figure 3** (MaskGIT samples) shows that IterRef’s images have sharper structure and better semantic alignment compared to BoN/FK/SVDD, consistent with the quantitative gains.  
   - **Figures 6 and 7** in Appendix F offer concrete LLaDA and MDLM generations for CoLA and Sentiment rewards, giving the reader a sense of how the style of text changes.

7. **Reasonable complexity discussion and practical tricks.**  
   Section 3.3 discusses computational bottlenecks and introduces practical optimizations such as pool reuse and a choice of balancing function that avoids explicit backward sampling. This makes it plausible for practitioners to adopt IterRef and understand the NFE vs wall-clock tradeoffs; the wall-clock tables (Tables 12–13) also show that for heavier models IterRef can be competitive at higher costs.

## Weaknesses

1. **Mathematical inconsistencies and confusing use of the importance weights.**  
   The heart of the MTM formulation is Eq. (2) and the resulting weights and acceptance ratio in Eq. (3). There are several issues here that need to be clarified or corrected:
   - In Section 3.1 / Eq. (2), the balancing function is chosen so that, as derived in Appendix D.2, the importance weights become  
     \[
     w_n = \frac{\exp(r(x_t^{\prime(n)})/\alpha)}{\sum_j\exp(r(x_t^{\prime(j)})/\alpha)} = \frac{1}{N}.
     \]
     The last equality is incorrect: the numerator and denominator clearly depend on the individual rewards, so the weights only collapse to \(1/N\) under pathological conditions (all rewards identical). Yet Eq. (3) and the main text present \(w_n=N^{-1}\) as a general result.  
   - Algorithm 2, Line 7, states “Compute weights \(w_n\) and select \(x_t'\) by weighted sampling with \(w_n\) (Eq. 3)”, and the text above says “reward-weighted sampling using \(w_n\)”, but Eq. (3) shows uniform weights. There is a mismatch between the informal description (“reward-weighted”) and the actual formula. If the intention is to sample candidates proportional to their reward, then the lambda choice should not force uniform \(w_n\); if the intention is uniform sampling with reward in the acceptance test only, the text should say so and drop “reward-weighted” phrasing.  
   - The acceptance probability in Eq. (3) is written as  
     \[
     \beta = \min(1,\exp((r(x_t') - r(x_t))/\alpha)).
     \]
     However, Appendix D.2 clearly derives  
     \[
     \beta = \min\left(1, \exp\left(\frac{r(x_t) - r(x_t')}{\alpha}\right)\right),
     \]
     which has the opposite sign. The current Eq. (3) would *prefer lower-reward* proposals if \(\alpha>0\). This is potentially a serious sign error; either Eq. (3) or the derivation needs correction. At minimum, the authors must clarify which version is actually implemented in experiments.  
   These inconsistencies undermine confidence in the claimed “principled MTM” formulation, since the effective transition kernel in code may differ materially from what is written.

2. **Strong and somewhat unrealistic reversibility assumptions.**  
   Proposition 1 assumes that the forward corrupting kernel \(q\) and the reverse denoising kernel \(p_\theta\) “form a reversible Markov kernel”. For absorbing-state discrete diffusion with a learned reverse model, exact reversibility is a very strong condition and is practically never satisfied. The proofs in Appendix D.3–D.4 use reversibility to turn sums involving \(q(x_s\mid x_t)p_\theta(x_t'\mid x_s)\) into symmetric expressions. However:
   - The main text does not spell out the exact reversibility condition (e.g., a detailed balance relation of form \(p(x_t)q(x_s\mid x_t)=p(x_s)p_\theta(x_t\mid x_s)\)), nor does it discuss how far real trained MDLM/LLaDA/MaskGIT models are from satisfying it.  
   - The reader is left with a convergence theorem that holds under a rather idealized assumption with no quantitative robustness argument: what is the bias in the stationary distribution if reversibility is only approximate?  
   The result is that while the MTM chain is “theoretically neat” on paper, it is not entirely clear whether the guarantees are meaningful for the actual models used in the experiments.

3. **Ambiguity around the true target distribution and intermediate reward approximation.**  
   In Section 2 “Reward-Guided Generation”, the target marginal is derived as  
   \[
   p^*(x_0) \propto \exp(r(x_0)/\alpha)p_\theta(x_0),
   \]
   and Remark 1 then gives \(p^*(x_t) \propto p(x_t)\exp(r(x_t)/\alpha)\), with \(r(x_t)\) defined via the log-moment expression over \(x_0\). However, throughout the method and experiments:
   - The “intermediate reward” \(r(x_t)\) is *approximated* by applying the reward model to a prediction of \(x_0\) from the diffusion model, not via the proper log-expectation. This is acknowledged briefly (“Intermediate rewards … can [be] approximate by evaluating the reward function on the diffusion model’s prediction of \(x_0\)”), but the consequences are not discussed. This approximation effectively changes the target distribution, and the theory no longer matches the implemented reward function.  
   - The notation conflates the prior \(p(x_t)\) in Remark 1 with the learned denoising process \(p_\theta\) and the implicit stationary distribution used in the convergence proof. It would help to make explicit what base measure \(p(x_t)\) is assumed in Proposition 1 (the unaligned generative distribution? the forward noising marginal?).  
   As it stands, the theoretical story is clean only under an idealized intermediate reward, while the actual algorithm uses a quite different proxy, and this misalignment is not analyzed.

4. **Compute accounting and real-world efficiency are underexplained.**  
   The paper argues that IterRef is especially effective at low NFEs and higher reward budgets, but the picture is less clear when considering *wall-clock time* and parallelism, which matter for test-time scaling in practice.  
   - Section 3.3 and Appendix C.4 admit that IterRef is inherently sequential and can be slower than more parallelizable baselines when the model is small or hardware parallelism is high. **Table 13** (LLaDA) shows, for instance, that at NFE=32, IterRef takes 41.59s vs 30–32s for all baselines, despite comparable NFE accounting.  
   - The main figures (e.g., **Figure 2**) optimize for NFE, not time. Since IterRef interleaves refinement and denoising at selected steps, the practical NFE/step ratio is different from SMC-style baselines. The paper should be more explicit about how much of IterRef’s advantage survives if one normalizes by *time* or *GPU FLOPs* rather than NFE.  
   - Section 3.3 describes a “balancing function and pool reuse” trick that avoids backward resampling but does not rigorously justify that the modified algorithm still exactly preserves the target stationary distribution; the theoretical MTM (Algorithm 1) and the practical IterRef (Algorithm 2 plus modifications) are not clearly distinguished. This calls into question how “exact” the MCMC interpretation remains once efficiency tweaks are applied.

5. **Missing or weak ablations on robustness and reward mis-specification.**  
   All experiments use reasonably “well-behaved” rewards (CoLA, toxicity, sentiment, perplexity, CLIPScore, ImageReward). There is little analysis of how IterRef behaves under:
   - Very sparse or noisy rewards (e.g., thresholded classifiers).  
   - Non-monotonic reward landscapes where small token changes can dramatically flip reward.  
   - Strong KL regularization vs weak regularization (the role of \(\alpha\) is not thoroughly explored beyond setting it to 0.1).  
   Given that IterRef is, in spirit, a local MH-style refinement, one would expect sensitivity to these issues; from the current results it is unclear how robust the gains are beyond the set of tasks studied.

6. **Limited empirical diversity of discrete modalities.**  
   Despite a broad set of *tasks*, the set of *model families* is relatively narrow: MDLM and LLaDA-8B for text, and MaskGIT for images. There is no evaluation on other important discrete diffusion use cases, such as protein/molecule or biological sequence generation (which are major targets of reward-guided discrete diffusion in other work) or code-specific rewards. This is not fatal, but it weakens the claim of generality, especially given that one of the motivations is reward-aligned design over discrete spaces.

7. **Positioning vs very recent related work on discrete diffusion test-time scaling is incomplete.**  
   The Related Work section covers SVDD, PGDLM, DSearch, DTS, and re-masking for discrete diffusion, but omits several 2026 works that are closely aligned in goal and sometimes in technique (see “Potentially Missing Related Work” below). In particular, there are already reward-guided discrete diffusion samplers using Markov chains over clean samples, sophisticated importance weighting, and stitching / lookahead strategies. Without discussing how IterRef compares theoretically and empirically to these, it is harder to assess the true incremental contribution.

8. **Some clarity issues and minor writing problems.**  
   - There are several typos and grammatical mistakes (e.g., “astudies” in Appendix A, “calcuate” in Section 4.1, stray “$\sim$” characters in sample texts).  
   - The description of Algorithm 2 is slightly confusing: Line 9 refers to “Accept \(x_t^{\mathrm{can}}\)” but then sets \(x_t \gets x_t'\) upon acceptance, and the role of \(x_t^{\mathrm{can}}\) is not explicit.  
   - The function \(p(x_t)\) is used in several key equations (Remark 1, Eq. 2) but never clearly defined in the main text; whether it is the unaligned diffusion marginal, a prior, or something else must be stated.

Overall, the paper is clearly above the bar in terms of idea and empirical work, but the theoretical-implementation mismatch and some mathematical sloppiness reduce confidence in the “principled MTM” narrative.

## Potentially Missing Related Work

The following directly related works appear to be missing from the paper and should be discussed:

1. **Kim, Y., Shin, D., Na, B. (2026). “Lookahead Sample Reward Guidance for Test-Time Scaling of Diffusion Models.”**  
   - This work introduces a test-time scaling method that does reward-guided lookahead sampling to steer diffusion trajectories toward high-reward regions. It is conceptually close to IterRef in that it uses multiple candidate samples per step and a reward model, though with a different search strategy (lookahead vs in-place refinement).  
   - It should be compared and cited in Section 5 (“Reward-Guided Generation” and “Scaling”) and considered as a baseline or at least discussed qualitatively in Section 4, especially regarding efficiency vs reward gains.

2. **Phunyaphibarn, P., Sung, M. (2026). “Reward-Guided Discrete Diffusion via Clean-Sample Markov Chain for Molecule and Biological Sequence Design.”**  
   - This paper proposes a Markov chain sampler over *clean* samples for reward-guided discrete diffusion in structured biological domains. This is highly relevant since IterRef also uses an MCMC-style approach to correct discrete samples.  
   - It should be mentioned in Section 5 in the paragraph on reward-guided discrete diffusion, and the authors should clarify how their noising–denoising MTM kernel differs from a clean-sample Markov chain (e.g., mixing behavior, reversibility assumptions, and applicability to language/image).

3. **Ou, Z., Pani, C., Li, Y. (2026). “Inference-Time Scaling of Discrete Diffusion Models via Importance Weighting and Optimal Proposal Design.”**  
   - This work focuses specifically on inference-time control of discrete diffusion through sophisticated importance weighting and proposal distribution design, which is closely related to the optimal kernel \(p^*(x_{t-1}\mid x_t)\) and approximations discussed in Section 2.  
   - It should be discussed near SVDD and FK in Section 5, and the authors should explain how IterRef’s MTM kernel compares to their optimal proposals (e.g., is IterRef’s noising–denoising kernel a special case, or orthogonal?).

4. **Miles, R., Toker, A., Oncescu, A. (2026). “Test-Time Scaling with Diffusion Language Models via Reward-Guided Stitching.”**  
   - This paper proposes “reward-guided stitching” for diffusion language models, a self-consistency framework for test-time scaling. The goal and domain (diffusion LMs) are almost identical to the MDLM and LLaDA experiments in this paper.  
   - It should be cited in Section 5’s discrete diffusion / inference-time scaling discussion and ideally compared empirically (or at least conceptually) to IterRef, since both are targeted at reward-guided test-time improvement for DLMs.

Including and situating against these works would significantly strengthen the paper’s positioning and make the contribution clearer.

## Questions

1. **Clarification of Eq. (3) and implementation.**  
   - Did you implement the acceptance probability as in Eq. (3) in the main text (\(\beta = \min(1,\exp((r(x_t')-r(x_t))/\alpha))\)) or as in Appendix D.2 (\(\beta = \min(1,\exp((r(x_t)-r(x_t'))/\alpha))\))? Please explicitly state which form is used in code and correct the inconsistent equation.  
   - Similarly, when selecting a proposal among \(\{x_t^{\prime(n)}\}\), are you sampling uniformly (as Eq. (3) suggests) or proportional to \(\exp(r(x_t^{\prime(n)})/\alpha)\)? If the latter, please adjust the derivation and balancing function accordingly, or if the former, clarify why the term “reward-weighted sampling” is used.

2. **Reversibility assumption and practical validity.**  
   - Can you specify the exact reversibility relation you assume between \(q\) and \(p_\theta\)? For real MDLM or LLaDA-8B models, how far from reversible are they empirically (e.g., if you estimate \(p(x_t)q(x_s\mid x_t)\) vs \(p(x_s)p_\theta(x_t\mid x_s)\) on random samples)?  
   - Even a small experiment or argument about how approximate reversibility affects the stationary distribution of IterRef would improve confidence that the theoretical insights are not purely formal.

3. **Effect of intermediate reward approximation.**  
   - You approximate \(r(x_t)\) via the reward of the predicted \(x_0\). Have you compared this against evaluating the reward on partially denoised samples directly (e.g., decoding visible tokens only) or against a better approximation to the log-expectation?  
   - Is there any evidence that using this approximation biases IterRef to overfit to reward-shortcuts or degrades calibration compared to baselines?

4. **Time-normalized comparisons and parallelism.**  
   - Could you provide an additional figure analogous to **Figure 2**, but with wall-clock time on the x-axis instead of NFEs, at least for one backbone (e.g., LLaDA-8B)? This would help quantify the practical advantage when parallelism is fully exploited.  
   - Are there obvious parallelization strategies for IterRef (e.g., parallel over candidates \(N\), batch processing of refinement steps) that would narrow the time gap in Tables 12–13?

5. **Behavior under noisy or adversarial rewards.**  
   - Have you evaluated IterRef under intentionally noisy or sparse reward signals (e.g., thresholded toxicity classifiers, random label flips)? Does MTM-based refinement prove more or less robust than SMC/importance sampling baselines?  
   - Relatedly, can you discuss how the method behaves when \(\alpha\) is varied more aggressively (e.g., 0.01 vs 1.0)? Some sensitivity analysis of IterRef’s acceptance rates and reward vs. fluency tradeoff would be useful.

6. **Extension to structure-rich domains.**  
   - Do you foresee any obstacles in applying IterRef to molecule or protein sequence diffusion models, as in the missing related work listed above? In particular, does the noising–denoising kernel over sequences with strong structural constraints mix sufficiently, or would you need domain-specific modifications?

## Flag For Ethics Review

No ethics review needed.

## Details Of Ethics Concerns

N/A.

## Soundness Rating

3: good.  
The methodological core is sound in spirit and supported by extensive experiments, but there are nontrivial inconsistencies in the MTM derivations (especially Eq. (3)) and the reversibility assumption is idealized without empirical justification. These issues should be fixable but currently weaken the theoretical story.

## Presentation Rating

3: good.  
The paper is generally well written with clear figures (e.g., Figure 1, Figure 2, Figure 4, Figure 5) and informative tables (Tables 1–3), but there are several confusing notational choices and minor typos, and some key definitions (e.g., \(p(x_t)\)) are not fully specified.

## Contribution Rating

3: good.  
IterRef represents a meaningful step forward in test-time scaling for discrete diffusion, both conceptually (iterative refinement via MTM) and empirically (consistent improvements across backbones and tasks). However, incomplete positioning against closely related very recent work and some theoretical sloppiness keep it from the “excellent” tier.

## Overall Rating

6: Marginally above the acceptance threshold. But would not mind if paper is rejected.  

The paper proposes a genuinely interesting and practically effective method for reward-guided test-time scaling in discrete diffusion, with strong empirical evidence and a nontrivial MCMC-based formulation. However, several mathematical inconsistencies (especially around Eq. (3)), a strong and unexamined reversibility assumption, and incomplete discussion of very recent related work temper enthusiasm. With clarifications and corrections, this could be a solid ICLR contribution; as is, I lean to accept but can understand a negative decision.

## Reviewer Confidence

4: confident.  
I am familiar with diffusion models, MCMC/MTM, and test-time scaling, and I carefully checked the key equations and experimental setup. Some implementation details are necessarily inferred, but overall I am confident in the assessment.