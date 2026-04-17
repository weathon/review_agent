---
job_id: 2aba81fb-3923-4cd5-bdf5-52bcc4693d6a
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: ck0aZTAnwK.pdf
paper: Pre-training Under Infinite Compute
main_score_norm: 0.8
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper studies language model pre-training scaling laws, regularization, ensembling, and distillation under data constraints, which falls squarely within representation learning, optimization, and large-scale learning, all core ICLR topics.

## Minimum Quality
Pass ✅.  
The paper is in English and has all key sections: Abstract, Introduction, Methodology/Approach (Sections 2–6), Experiments/Results (Sections 2–7 and Appendix A–I), Related Work (Section 8, Appendix J), and Discussion (Section 9). The work is technically serious, with substantial empirical evidence and clear exposition. I do not see a fatal methodological or statistical flaw that would justify desk rejection.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I see no attempts to manipulate automated reviewing systems, no hidden prompts, and no suspicious meta-instructions inside the paper content.

---

# Expected Review Outcome:

## Summary

The paper investigates how to pre-train language models in a regime where compute is effectively unlimited but training data is fixed. Using a 200M-token DCLM subset as the main testbed, the authors show that standard recipes that increase epochs and parameter count overfit, then propose aggressively tuned weight decay to recover monotone power-law scaling in parameter count and to define a “best possible” performance via the asymptote of this scaling law. They further show that logit-averaged ensembles and a joint limit in model size and ensemble size attain a substantially lower loss asymptote (corresponding to a 5.17× effective data efficiency gain at 200M tokens), that these gains largely persist under distillation into smaller models and at higher token budgets, and that improved validation loss correlates with stronger downstream performance on multiple benchmarks and in a continued-pretraining (CPT) setting.

## Strengths

1. **Crisp formulation of an important emerging regime.**  
   The paper directly tackles data-constrained, compute-unconstrained pre-training, which is becoming realistic given projected growth rates of compute vs web text. Formalizing the objective as minimizing validation loss under fixed data, with compute effectively unbounded, is conceptually clean and distinct from Chinchilla-like compute-optimal regimes.

2. **Strong empirical case that “standard practice” fails under data constraints.**  
   Section 2.1 and **Figure 2** compellingly document that simply increasing epochs or parameter count (with moderate hyperparameter tuning) at 200M tokens leads to clear overfitting: validation loss decreases up to a point then rises, and even a 10× parameter increase (150M→1.4B) yields <0.1 improvement before degradation. **Figure 14** reinforces this by showing train loss dropping while validation loss increases with more epochs, which is consistent with classical overfitting rather than optimization failure.

3. **Aggressive regularization and careful hyperparameter tuning recover clean scaling.**  
   The coordinate-descent–style tuning procedure in Appendix C.1 is thorough and results in hyperparameters that vary sensibly with parameter count and token count (**Figure 11**). The key empirical punchline is **Figure 3**, where, after tuning weight decay, epoch count, and learning rate, loss scales roughly as  
   \[
   \hat{\mathcal{L}}_{D,N} \approx \frac{0.05}{N^{1.02}} + 3.43,
   \]
   monotone in \(N\) even at parameter-to-token ratios ~140× Chinchilla. The fact that the optimal weight decay grows to ~0.8–3.2 (30× the Brown et al. 0.1 default) is a useful and non-obvious practical insight for data-constrained regimes.

4. **Asymptote-based evaluation is conceptually clean and well-used.**  
   Rather than comparing recipes at fixed compute, the paper proposes using the asymptote \(E_D = \lim_{N\to\infty} \hat{\mathcal{L}}_{D,N}\) (and similarly for \(K\)) as the “best achievable” loss under a recipe at a given data budget. This is explicitly encoded in the power-law forms like  
   \[
   \hat{\mathcal{L}}_{D,N} = A_D / N^{\alpha_D} + E_D
   \]
   and the double limit definition in Section 4.3:
   \[
   \hat{\mathcal{L}}_D = \lim_{N\to\infty}\lim_{K\to\infty} \min_H \mathcal{L}\big(\mathcal{E}_\mathcal{A}(D,N,K,H)\big).
   \]
   While asymptote extrapolation is an approximation, the idea of comparing *recipes* by their asymptotic loss under fixed data is a clear and novel lens.

5. **Ensembling is shown to be systematically better than pure parameter scaling in this regime.**  
   The ensemble scaling experiments in Section 4 are carefully done. **Figure 4** demonstrates that excess loss decays roughly as \(1/K\) for 300M-member ensembles, with a fitted asymptote of 3.34 at 200M tokens, outpacing the 3.43 asymptote of pure parameter scaling. **Figure 5** then constructs the “joint scaling” recipe via two nested power-law fits (over \(K\) and then \(N\)), giving an estimated best-possible loss of 3.17. The empirical result that a \(K=3\) ensemble already beats the best infinitely large single model is practically significant: at large total parameter budgets \(NK\), multiple small models can be preferable to one huge model.

6. **Multi-level scaling-law methodology is consistently applied and reasonably validated.**  
   Section 5 and **Figures 6–7** extend the asymptotic-analysis machinery across token counts up to 1.6B, showing that:
   - Single-model regularized recipes are about 2.29× more data efficient than the standard recipe at 200M tokens.
   - The joint parameter+ensemble scaling asymptote is about 5.17× more data efficient at 200M, and **Figure 7 (right)** suggests a roughly constant data-efficiency gap across token counts, with both recipes sharing similar data-scaling exponents (~0.23–0.24).  
   The sensitivity analyses in **Figure 20** and the extrapolation test in **Figure 21** (predicting 1.5B and 3.2B model losses within <0.01) help substantiate that these simple 1D power laws are not completely spurious.

7. **Distillation results show that asymptotic gains can be made practical under parameter constraints.**  
   Section 6 demonstrates that the infinite-compute conceptual gains are not just theoretical:
   - **Figure 8** shows ensemble distillation: distilling an 8×300M ensemble (loss 3.32) into a 300M student achieves 3.36 loss, preserving ~83% of the ensemble improvement over the best regularized 300M model (3.57). This student even beats the regularized asymptote.
   - The self-distillation result in **Figure 8** and **Table 4** is particularly striking: a 300M teacher self-distilled with a 1:1 mix of real and synthetic data yields a 300M student with loss 3.44, *better* than the teacher (3.71) and matching or surpassing the regularized 300M model, while self-distillation on synthetic-only data collapses to 4.07 loss. This gives a nice, data-constrained counterpoint to recent “model collapse” narratives.

8. **Downstream validation and CPT transfer are convincing.**  
   The paper does not stop at validation loss:
   - **Figure 9** and **Table 5** show that validation loss reductions translate monotonically into lower average error on ARC-Easy, PIQA, and SciQ. Ensembles of 1.4B members with \(K=4\)–5 improve average accuracy by ~9% over the best unregularized model, and distilled 300M students gain ~7% over the unregularized 300M baseline.
   - The extended results in Appendix K, especially **Figure 22** and **Table 12**, show consistent ranking on a broader suite (ARC-Challenge, HellaSwag, LAMBADA, Winogrande).  
   - **Table 1** in Appendix A is a strong sanity check: in a continued-pretraining setting on Llama 3B with 4B MegaMath-Web-Pro tokens, the same interventions (smaller batch size, more epochs, ensembling) deliver average math-benchmark accuracy that matches or exceeds a 73B-token CPT baseline, implying a 17.5× effective data-efficiency gain.

9. **Exposition and experimental detail are high quality.**  
   The paper is well written and unusually transparent about hyperparameter search and assumptions. **Figure 10** and **Figure 11** do a good job illustrating why naive transfer of hyperparameters (e.g., fixed weight decay or epoch count) fails and why coordinate descent is needed. Appendices C–F document architectures, hyperparameters (e.g., **Table 2**, **Table 3**), and ablations with enough detail for serious reproduction. The math is mostly straightforward but clearly presented.

## Weaknesses

1. **Reliance on extrapolated asymptotes from a relatively small range of \(N\) and \(K\).**  
   The central narrative rests on quantities like the 3.43 (regularized \(N\to\infty\)) and 3.17 (joint \(N,K\to\infty\)) asymptotes, but these are fit from four model sizes (150M, 300M, 600M, 1.4B/1.5B/3.2B) and at most five ensemble sizes up to \(K=8\). For example:
   - In **Figure 3**, the fit \(\hat{\mathcal{L}}_{200\text{M},N} = 0.05 / N^{1.02} + 3.43\) is based on four points, and \(\alpha\approx 1\) is unusually large compared to Chinchilla’s ~0.34. With such a small set, the exponent–asymptote tradeoff is not well identified; small changes in curvature could move \(E_D\) materially.
   - **Figure 5** and **Figure 7** introduce *two* or *three* nested power-law fits (first over \(K\), then over \(N\), then over \(D\)), each step compounding uncertainty. The paper reports high \(R^2\) (Table 11), but with 3–4 degrees of freedom and only a handful of points, \(R^2\) is not very informative.  
   The authors acknowledge noise (Appendix I) but the main text leans on fairly precise-sounding numbers (e.g., “5.17× more data efficient”) that may overstate confidence. Providing confidence intervals on \(E_D\), or a simple robustness analysis to perturbed exponents, would make the asymptote comparisons feel less brittle.

2. **Heuristic, not principled, treatment of hyperparameters in the double limit.**  
   The double limit
   \[
   \hat{\mathcal{L}}_D = \lim_{N\to\infty}\lim_{K\to\infty} \min_H \mathcal{L}\big(\mathcal{E}_\mathcal{A}(D,N,K,H)\big)
   \]
   is mathematically clean, but in practice the paper departs from this definition. For ensembles, they show in **Figure 15** and **Figure 17** that the hyperparameters that minimize the asymptote at \(K\to\infty\) are different from those at \(K=1\), but then globally adopt a simple heuristic of “double epochs, half weight decay” relative to the single-model optimum (Appendix D.4).  
   This means the inner \(\min_H\) in the limit is actually replaced by a fixed heuristic hyperparameter schedule \(H'(N)\), not a proper minimization over \(H\) for each \(N, K\). This weakens the formal claim that the joint scaling asymptote is the *best possible* performance under infinite compute; it is, more accurately, the best performance under a particular family of hyperparameter scalings. The paper should make this distinction explicit in the main text and, ideally, quantify how far the heuristic is from the true minimum for at least a few additional \(N,K\) settings.

3. **Limited exploration of regularization space despite strong claims about “optimal” weight decay.**  
   The narrative centers almost entirely on \(L_2\) weight decay as the regularizer. This is reasonable for a first pass, but the conclusions occasionally verge on overgeneral (e.g., “it is critical to regularize pre-training with much higher weight decay than standard practice,” Section 3).  
   Missing or underexplored aspects:
   - Other standard regularizers known to matter in data-scarce regimes, such as dropout, label smoothing, or stochastic depth, are not tuned or even reported. Given that the paper already leans heavily on compute-unbounded coordinate search, it is somewhat unsatisfying that the regularization axis is effectively 1D.
   - Even within weight decay, the search space is restricted to powers of 2 times 0.1 up to 6.4. For very overparameterized settings like the 3.2B model at 200M tokens, it is not clear that 3.2 is near the optimum (Figure 11 suggests monotonic growth with \(N\)), and no ablation explores beyond the upper bound.  
   Concretely, **Figure 12 (right)** only shows two models (300M and 1.4B) at a few decays; a more systematic sweep would clarify whether the large gains are robust to alternative regularizers and higher/lower weight decay ranges.

4. **Empirical scope is modest relative to some claims.**  
   All main pretraining-from-scratch experiments are on a single dataset family (DCLM) with token budgets up to 1.6B and model sizes up to 3.2B parameters. While this is not trivial, the paper’s framing is about “a compute-rich future” and “infinite compute”, which invites the question of robustness at much larger scales.  
   Two specific concerns:
   - The DCLM subset may have idiosyncratic redundancy or domain composition that particularly favors heavy regularization and ensembling. For instance, **Figure 2 (left)** shows overfitting as early as 16–32 epochs for 300M models; this might not generalize to more diverse or noisier corpora.
   - The continued-pretraining experiment on Llama 3B (**Table 1**) is reassuring but still a single alternative setting, again with 4B tokens, not tens of billions. It would be useful to see at least one larger-scale setting (e.g., 10B–50B tokens from DCLM) where ensembling still beats single-model scaling at comparable total parameter counts.

5. **Theoretical discussion of ensembling advantage is thin and not directly connected to experiments.**  
   Section 4.2 briefly invokes Allen-Zhu & Li’s “multi-view” theory, where each ensemble member learns a different informative feature. However:
   - The paper does not attempt to empirically test whether members actually specialize on different “views” of the data (e.g., via representational similarity or per-example disagreement).  
   - There is no analysis of whether the observed near-\(1/K\) scaling in **Figure 4** and **Figure 5 (left)** has a principled link to the theory, or is just an empirical curve fit.  
   Given that some theoretical works (e.g., Vyas et al., Ruben et al.) are cited as suggesting ensembling does *not* beat scaling in certain regimes, it would be interesting to characterize when and why language-model pretraining is in the “good for ensembles” regime. Without this, one has to trust the empirical fits somewhat blindly.

6. **Some aspects of the experimental setup may interact with conclusions, but are under-explored.**  
   A few choices raise questions:
   - The largest dense model in the initial experiments, 1.4B, has a different aspect ratio (much wider, fewer layers) than the smaller models (**Table 2**), and the authors note that this could contribute to overfitting (Appendix C.5). The rebuttal models (1.5B, 3.2B) fix this, and **Figure 21** suggests the regularized power law extrapolates well, but the main text still uses the 1.4B–based asymptote extensively.
   - Data order shuffling is done once and reused across epochs (Appendix B). This choice minimizes instability but could exacerbate overfitting to specific sequences across many epochs, potentially exaggerating the gains of strong regularization and ensembling. An ablation with epoch-wise reshuffling could help disentangle this and verify robustness.
   - Batch size is fixed to 64 based on **Figure 12 (left)** at 200M tokens, but it is unclear if the same choice is optimal at 1.6B tokens or for larger models; some sensitivity analysis would be informative given known batch–generalization tradeoffs.

7. **Metrics of “data efficiency” deserve more nuance.**  
   The paper defines data efficiency as the factor \(\frac{D'}{D}\) such that the standard recipe at token count \(D'\) would match the loss of a new recipe at \(D\), where \(D'\) is found by inverting the fitted data-scaling law. This is reasonable, but:
   - Because both numerator and denominator are model-based extrapolations, small errors in exponent or asymptote propagate into large multiplicative factors. The headline “5.17× more data efficient” for the joint scaling recipe at 200M (**Figure 7, right**) reflects this compounded uncertainty.
   - Different stakeholders care about different operational tradeoffs. For example, a 5× data-efficiency improvement at the cost of, say, 20× more training FLOPs due to ensembling and distillation is not directly evaluated; the paper explicitly de-prioritizes compute, which is fair for its stated setting, but the empirical compute-cost curve would still be useful context.  
   Clarifying these caveats, and perhaps reporting a range (e.g., “between 3× and 6× given reasonable exponent variation”) would make the claims more measured.

8. **Minor mathematical and clarity issues.**  
   - In the ensembling definition (Appendix D.1), the LogitAvg model is defined as  
     \[
     \text{LogitAvg}(M_{i\in [K]})(x) \propto \exp\Big(\frac{1}{K}\sum_{i\in[K]} \log M(x)\Big)
     \]
     which appears to omit the index on \(M_i(x)\) in the log term; presumably it should be \(\log M_i(x)\). While this is an obvious typo, it reflects a slight casualness in the notation around critical constructs like ensembles.
   - In Appendix D.2, the equation illustrating that
     \[
     \arg\min_H \mathcal{L}(\mathcal{A}(\cdot,H)) \neq \arg\min_H \lim_{K\to\infty}\mathcal{L}(\mathcal{E}_\mathcal{A}(\cdot,K,H))
     \]
     is written with heavy letter spacing and inline subscripts that slightly obscure the structure. A cleaner math notation would help.
   - The assumption that \(f(N,K) = \min_H \mathcal{L}(\mathcal{E}_\mathcal{A}(\cdot))\) is monotone in both \(N\) and \(K\) (Appendix D.6) is empirically plausible in the tuned experiments, but it is nontrivial and not theoretically justified. The proof sketch that double limits commute relies entirely on this assumption; better to state clearly that this is an empirical regularity, not a proven property of the training dynamics.

Overall, though, these are issues of caution and completeness rather than fatal flaws.

## Potentially Missing Related Work

The paper’s related work is fairly comprehensive in scaling laws, ensembling, and distillation, but it omits some classic and widely cited pretraining baselines that would help contextualize its contribution.

1. **Raffel et al., “Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)”, 2020.**  
   T5 is a major milestone in large-scale language pretraining and explicitly discusses tradeoffs in data, model size, and training strategies. While the architecture details are less central here, T5’s discussion of pretraining efficiency and transfer could usefully inform Section 2 or Section 8, especially when motivating why the authors focus on next-token language modeling rather than alternative objectives.

2. **Devlin et al., “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding”, 2019.**  
   BERT is not cited and is the canonical example of language pretraining for downstream tasks. Citing it in Section 1 or Section 8 when discussing prior pretraining practices and benchmarks (ARC, SciQ, etc.) would help non-specialist readers orient this work within the broader LMs-for-NLU literature.

3. **Radford et al., “Language Models are Unsupervised Multitask Learners (GPT-2)”, 2019.**  
   While the paper cites Brown et al. (GPT-3), GPT-2 is the immediate predecessor that popularized the large-scale autoregressive LM pretraining paradigm. A brief mention in Section 1 or 2 would fill a small historical gap.

4. **Clark et al., “ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators”, 2020.**  
   ELECTRA proposes a radically more sample-efficient pretraining objective. Given this paper’s emphasis on data efficiency at fixed token budgets, ELECTRA is directly relevant as an alternative *objective* rather than an alternative recipe within the same objective. It should be discussed in Section 8 as a contrasting line of work that tackles data efficiency via changing the loss, whereas this paper keeps the objective fixed and changes training/regularization.

5. **Vaswani et al., “Attention is All You Need”, 2017.**  
   The base Transformer architecture is implicitly used, but not explicitly cited. A brief citation in Section 2 when describing the Llama-style architecture would be appropriate.

6. **He et al., “Deep Residual Learning for Image Recognition (ResNet)”, 2016; Zoph et al., “Learning Transferable Architectures for Scalable Image Recognition”, 2018.**  
   These are less critical but relevant for the broader message about scaling deep networks and architectural choices under large-compute regimes. A short acknowledgment in Section 8 could help connect to the vision literature on scaling.

These are mostly “background” rather than directly competitive works; still, citing ELECTRA and T5 in particular would make the framing around data efficiency and pretraining recipes more complete.

## Questions

1. **Robustness of asymptote estimates.**  
   Could the authors provide uncertainty estimates for the key asymptotes (e.g., 3.43, 3.34, 3.17) and derived data-efficiency factors (2.29×, 5.17×)? For example, bootstrapping power-law fits over seeds or slightly different subsets of \(N,K\) could give confidence intervals. Evidence that the joint scaling recipe is better than regularized single-model scaling *within those intervals* would significantly increase my confidence in the asymptote-based ranking.

2. **Effect of data shuffling and curriculum.**  
   You fix a single permutation of data windows and reuse it across epochs. Have you tried epoch-wise reshuffling, or alternative curriculums, and if so does the need for very large weight decay (≥0.8) and the scale of ensemble benefits persist? A small ablation here would clarify whether current conclusions are tied to this somewhat unusual choice.

3. **Alternative regularization mechanisms.**  
   Have you tried augmenting or replacing weight decay with dropout, label smoothing, or other regularizers that historically matter for data-limited language modeling? If you have preliminary runs, indicating whether such methods can further push the asymptote down (or reduce the required weight decay) would be helpful; if not, a short discussion of expected interactions would still be valuable.

4. **Compute–performance tradeoffs.**  
   While the paper explicitly adopts an “infinite compute” view, in practice, training multiple large ensembles plus distillation is nontrivial. Could you quantify, even approximately, the training FLOPs per token (or wallclock factor) for your best joint-scaling+distillation recipe compared to the standard recipe at 200M and 1.6B tokens? This would help practitioners decide whether, for example, a 3–5× data-efficiency improvement is worth a 10× increase in training compute.

5. **Characterization of ensemble diversity.**  
   Do you have any diagnostics (e.g., pairwise KL divergence between member predictions, disagreement rates on validation tokens, or representation similarity) that indicate whether ensemble members indeed learn complementary “views” of the data? Such analysis could either support or refine the Allen-Zhu & Li–inspired story in Section 4.2.

6. **Generalization across datasets and domains.**  
   Beyond the Meta-math CPT experiment, have you tried applying the same regularization + ensembling strategy on a qualitatively different pretraining corpus (e.g., a more code-heavy or dialogue-heavy subset of DCLM)? If so, does the basic picture (overfitting under standard recipes, monotone scaling with heavy weight decay, ensembles beating scaling) hold?

Author responses addressing these points, especially 1, 2, and 4, could materially increase my confidence in the strength and generality of the conclusions.

## Flag For Ethics Review

No ethics review needed.

## Details Of Ethics Concerns

N/A.

## Soundness Rating

3: good.  
The methodology is empirically solid and carefully executed, with extensive ablations on weight decay, epochs, and ensemble size. The main concerns lie in the reliance on multi-level power-law extrapolation for asymptotes and the heuristic treatment of hyperparameters in the double limit, which affect the sharpness of the quantitative claims but do not undermine the qualitative conclusions.

## Presentation Rating

4: excellent.  
The paper is very well written, figures and tables (e.g., **Figures 2–5, 8–9, 18–22**, **Tables 1, 3, 5, 10, 12**) are clear and informative, and the appendices provide unusually detailed experimental documentation. The structure makes a complex empirical story easy to follow.

## Contribution Rating

4: excellent.  
The work offers a clear, conceptually interesting reframing of pretraining in the data-constrained, compute-unconstrained regime, and backs it with substantial empirical evidence that standard practice is suboptimal, that aggressive regularization and ensembling can materially improve data efficiency, and that these gains carry over to distillation, CPT, and downstream tasks. This will likely influence how practitioners think about using extra compute when data is scarce.

## Overall Rating

8: Accept, good paper (poster).  
The paper is technically careful, well presented, and addresses an important emerging regime in language model pretraining. Despite some caveats about asymptote extrapolation and the heuristic handling of hyperparameters in double limits, the empirical evidence that (i) heavy regularization recovers monotone scaling far into the overparameterized regime and (ii) ensembles with distillation give substantial, practically realizable data-efficiency gains under fixed data is convincing and valuable. I recommend acceptance.

## Reviewer Confidence

4: confident.  
I am familiar with scaling-law literature, LM pretraining practice, and ensemble/distillation methods. I carefully checked the main equations, scaling-law fits, and key figures and tables. While there is always room for debate on extrapolation and generality, I am confident in my assessment of this paper’s technical soundness and contribution.