---
job_id: 0de7532a-a61a-4f37-b7e2-67cb5044c4e5
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: tswBfpkwHn.pdf
paper: Can Mamba Learn In Context With Outliers? A Theoretical Generalization Analysis
main_score_norm: 0.8
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.  

## Topic Compatibility
Pass ✅.  
The paper is a theoretical analysis of in-context learning for Mamba and linear Transformers, clearly within learning theory / representation learning scope for ICLR.

## Minimum Quality
Pass ✅.  
All required sections (Abstract, Introduction, Related Work, Method, Experiments, Results, Conclusion) are present. The work presents nontrivial new theory and supporting experiments; I do not see fatal methodological errors or missing core components that would mandate a desk rejection.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I did not detect any hidden prompts, instructions to reviewers, or other manipulative content in the provided text.

---

# Expected Review Outcome:

## Summary

The paper analyzes the training dynamics and in-context learning (ICL) generalization of a one-layer Mamba model trained with stochastic gradient descent on binary classification tasks, under a sparse pattern-based data model with additive label-corrupting outliers in the prompt.  

The authors derive an exact linear-attention-plus-gating form of one-layer Mamba (Equation (3)), prove convergence and sample complexity guarantees (Theorem 1), show robust ICL generalization under distribution-shifted outliers (Theorem 2), and compare these guarantees to those of an analogous one-layer single-head linear Transformer (Theorems 3–4).  

They further characterize the mechanism of Mamba’s linear attention and nonlinear gating (Corollaries 1–2) and validate the theoretical predictions with synthetic experiments and a small real-data study on SST-2, including comparisons of robustness to outliers and positional sensitivity (e.g., Figure 2 and Table 1).

---

## Strengths

1. **Nontrivial theoretical analysis of Mamba’s training dynamics**  
   - The main technical contribution is a rigorous convergence and generalization analysis for a one-layer Mamba trained by SGD on prompts, including the gating parameter updates.  
   - The decomposition of one-layer Mamba into a linear attention layer plus a multiplicative gating term, leading to Equation (3) and the explicit form of \(G_{i,l+1}(\mathbf{w})\), is carefully derived (Appendix E.1) and then used consistently throughout the proofs.  
   - Lemmas 4–6 dissect the gradient flow of the gating parameter \(\mathbf{w}\), split into two phases, which is genuinely novel compared to prior Transformer ICL analyses that do not track gating dynamics.

2. **Clear theoretical comparison between Mamba and linear Transformer under the same data model**  
   - The paper does not just prove properties of Mamba in isolation; it directly compares to an analogous one-layer single-head linear Transformer obtained by setting \(G_{i,l+1}(\mathbf{w})=1\) in Equation (3).  
   - Theorems 1 vs 3 and Theorems 2 vs 4 make the tradeoff explicit: linear Transformers converge faster and with milder conditions, but Mamba can tolerate a much higher fraction of outlier-containing context examples (up to \(\alpha \to 1\) in Theorem 2, vs \(\alpha<1/2\) in Theorem 4) and uses fewer clean context points to achieve the same ICL accuracy.  
   - This comparison is thoughtfully interpreted in Remarks 4–5 and is directly reflected in **Figure 2**, where Mamba’s classification error remains below \(10^{-2}\) up to \(\alpha \approx 0.8\) across three outlier-labeling schemes, while the linear Transformer’s error blows up once \(\alpha > 0.5\).

3. **Mechanistic insight into how Mamba achieves robust ICL**  
   - Corollary 1 shows that the learned linear attention score \(\hat{\mathbf{p}}_i^\top W_B^{(T)\top} W_C^{(T)} \hat{\mathbf{p}}_{\text{query}}\) is large when the relevant pattern matches the query and small otherwise (Equation (16)), aligning with “induction head” style behavior.  
   - Corollary 2 then quantifies how the gating term \(G_{i,l_{ts}+1}(\mathbf{w}^{(T)})\) almost nullifies outlier-containing examples (Equation (17)) and imposes an exponentially decaying weight with respect to index distance (Equation (18)).  
   - These predictions are empirically backed by **Figure 3** (attention concentrating on same-pattern examples as training progresses) and **Figure 4** (gating values: green bars for clean examples decaying roughly exponentially with distance from the query, red bars for outliers near zero), which strongly supports the claimed mechanism.

4. **Robustness to distribution-shifted outliers is handled explicitly**  
   - Theorem 2 does not just assume the same outliers at test time. It allows test-time outliers \(\mathbf{v}_s^{*'}\) to be arbitrary positive linear combinations of training outliers plus an orthogonal component (Condition (a), Equation (11)).  
   - It also allows their magnitudes \(\kappa_a'\) to differ from training magnitudes \(\kappa_a\), within a quantitative range (Condition (b)). This is significantly more realistic than assuming identical training and test corruptions.  
   - The condition \(\alpha < \min(1,p_a l_{tr}/l_{ts})\) in Theorem 2 is clearly interpreted in Remark 3, explaining how one can choose prompt lengths \(l_{tr},l_{ts}\) to ensure robustness even when a large majority of test context examples have corrupted labels.

5. **Proofs are technically detailed and mostly consistent**  
   - The math is heavy, but key steps are spelled out: gradient expressions for \(W_B,W_C,\mathbf{w}\) (e.g., Equations (93), (118), (135)), use of Chernoff and Hoeffding bounds (Lemmas 1–2), and careful control of gradient noise via batch-size conditions like Equation (41).  
   - The two-phase analysis of \(\mathbf{w}\) (Lemmas 4 and 5), with explicit conditions on \(t\) and how \(\mathbf{w}^T\mathbf{p}_i\) separates outlier vs clean examples, is particularly informative and goes beyond standard “black-box gating” arguments.

6. **Experiments are aligned to the theory and reveal nontrivial qualitative phenomena**  
   - The synthetic experiments in Section 4.1 directly target the regime of Theorems 2 and 4: they vary \(\alpha\) and outlier-labeling schemes (flipping, targeted, random) and show the qualitative phase transition around \(\alpha=0.5\) for linear Transformer vs much smoother degradation for Mamba (**Figure 2**).  
   - The three-layer experiments in Section 4.2, especially **Table 1**, probe the positional sensitivity predicted by Corollary 2: Mamba is very strong when outliers are far from the query (FQ) or randomly placed (R), but its accuracy collapses in the CQ setting where all outliers are closest to the query, consistent with the exponential decay in Equation (18). The linear Transformer is much less sensitive to position, which nicely illustrates the cost of the gating-induced locality bias.  
   - The SST-2 real-data experiment (Table 7) effectively maps the abstract pattern + outlier model to “James Bond” trigger phrases and shows qualitatively similar trends (Mamba generally more robust, but still worst in CQ placement), giving some evidence that the toy model captures aspects of real prompt poisoning.

7. **Extensions and discussion are thoughtful and technically grounded**  
   - Appendix E.6 shows how the same linear-attention-plus-gating decomposition extends to Mamba-2, RetNet, Gated Retention, and Gated Linear Attention by writing their recurrences in the form \(h_t=\sum_i G_{i,t} v_i k_i^\top\). This positions the results as indicative for a broader class of SSM-like architectures.  
   - Sections E.7–E.8 sketch clear, technically plausible paths to multi-class and regression ICL analysis, including how gradient expressions would change under squared loss (Equations (202)–(204)), even though full proofs are omitted.

---

## Weaknesses

1. **Extremely restrictive data model limits practical interpretability of the guarantees**  
   - The core assumptions in Section 3.2 are very strong: a finite set of orthogonal relevant patterns \(\{\mu_j\}\) and orthogonal irrelevant patterns \(\{\nu_k\}\) with equal norms, each input containing exactly one relevant + one irrelevant pattern, and outliers \(\{v_s^*\}\) also orthogonal to both sets.  
   - Additionally, tasks are defined purely by “which of two relevant patterns corresponds to +1 vs -1”, inputs use uniformly sampled coefficients \(\kappa,\kappa'\), and labels of outlier-containing examples are uniformly random (training) or arbitrary (test).  
   - While this is standard in some prior ICL theory, these very artificial geometric and label assumptions make it hard to extrapolate the quantitative robustness statements, such as Mamba tolerating \(\alpha \to 1\), to real-world language or vision ICL beyond the qualitative level. The authors acknowledge this partially, but the discussion (mainly in Remarks 1–3 and Section 5) is brief and does not deeply explore how violations (non-orthogonality, multiple relevant features per token, structured label noise) would affect the proofs.

2. **Strong and sometimes opaque technical conditions weaken the takeaways of the main theorems**  
   - The sufficient conditions in Theorem 1 and Theorem 2 are complex and somewhat opaque:  
     - The batch-size condition \(B\gtrsim B_M = \max\{B_T,\beta^{-4}V^2\kappa_a^{-2}(1-p_a)^{-2}\log\epsilon^{-1}\}\) with \(B_T=\max\{\epsilon^{-2},M_1(1-p_a)^{-1}\}\log\epsilon^{-1}\) is reasonable, but coupled with a prompt length requirement \(p_a^{-1}\text{poly}(M_1^{\kappa_a})\gtrsim l_{tr}\gtrsim (1-p_a)^{-1}\log M_1\) (Equation (8)), which hides a huge dependence in \(\text{poly}(M_1^{\kappa_a})\).  
     - The outlier magnitude range \(V\beta^{-4}\lesssim\kappa_a\lesssim V\beta(1-p_a)p_a^{-1}\epsilon^{-1}\) in Theorem 1, and the test-time magnitude condition \(\kappa_a'\in [\kappa_a, \Theta(V\beta p_a^{-1}\kappa_a^{-1}L^{-1}(1-p_a)\epsilon^{-1})]\) in Theorem 2, essentially require a “Goldilocks” regime: outliers must be large enough to be detectable yet not so large that they swamp relevant patterns.  
   - Many of these dependences are only discussed qualitatively (Remarks 1 and 3), and it is often unclear whether they are artifacts of the proof technique (e.g., the two-phase bound in Lemma 4) or fundamental. Some more explicit scaling plots or simplifications (e.g., plugging \(\beta=V=1\)) in the main text would help readers understand “how bad” the polynomial factors really are.

3. **Fairness of the comparison with Transformers is not fully convincing, especially given softmax results**  
   - The theoretical comparison in Section 3.4 is restricted to linear attention, single-head Transformers vs Mamba. From Theorems 3–4, Transformers converge faster but are less robust to \(\alpha > 1/2\).  
   - However, Appendix B.1 shows that multi-head linear attention does not help much in robustness (Table 2) *but* softmax Transformers are nearly as robust as Mamba and avoid the severe CQ drop (Table 3; Tables 4–5). This undercuts the narrative that Mamba is inherently more robust to outliers than “Transformers” as such; the main advantage appears to be relative to linear attention, not to realistic LLM-like softmax attention with multi-heads.  
   - Remark 6 briefly acknowledges this, but the main text still emphasizes Mamba’s “superior robustness to a high density of outliers in ICL” (Contribution 2 in Section 1.1 and Remark 5) without clearly delimiting that the theoretical comparison is only to a particular linear-attention baseline. ACs and readers could easily overgeneralize these theoretical claims to full Transformers.

4. **Mechanistic results rely heavily on strong concentration and do not fully parse through corner cases**  
   - Corollary 1’s separation between same-pattern and different-pattern attention scores uses inequalities (43)–(44) that depend on small terms like \((1-p_a)^{-1}\epsilon/M_2\) and \((1-p_a)^{-1}p_a\kappa_a V^{-1}\beta^{-1}\epsilon\), plus the gradient noise bounds from Lemma 3. It would be useful to see explicit conditions under which “clean patterns dominate outlier patterns” truly holds, rather than hiding them behind big-Oh notation.  
   - Similarly, Lemma 5’s results \(\mathbf{w}^{(t)\top}p_i\lesssim -\log M_1\) (Equation (37)) vs \(\gtrsim -\Theta(1)\) for clean examples (Equation (38)) are key to Corollary 2’s gating behavior, but the proof relies on nontrivial induction with \(\gamma_1,\gamma_2\) and several inequalities like (167)–(181). The current exposition makes it hard to verify there are no hidden sign flips or parameter regimes where the conclusions could fail (for instance, large \(M_1\) but modest \(\kappa_a\), or vice versa).  
   - It would help if the authors could isolate and summarize, in the main text, a simplified “phase diagram” of parameter ranges under which the Corollary 2 behavior is guaranteed, rather than only in scattered inequalities in Appendix E.

5. **Experimental section is relatively narrow given the strength of the theoretical claims**  
   - All synthetic experiments are conducted in one particular setting (\(d=30,M_1=6,M_2=10,V=3,p_a=0.4\) or 0.6, \(l_{tr}=l_{ts}=20\)) and primarily vary \(\alpha\) and outlier label rules. There is no exploration of how performance scales with number of patterns (\(M_1\)), the magnitude parameters (\(\kappa_a,\kappa_a'\)), or prompt length \(l_{tr},l_{ts}\), despite these being central quantities in Theorems 1–4.  
   - **Table 1** is very informative about positional sensitivity, but again evaluated only at \(\alpha=0.5\) in a small synthetic setup. We never see larger-scale or more realistic prompts or tasks that would test how well the theory carries over to more complex distributions (e.g., multiple relevant patterns per token, partial non-orthogonality).  
   - The SST-2 experiment in Appendix B.2 is appreciated, but somewhat minimal: a single dataset, one poisoning keyword (“James Bond”), and fixed architecture (3 layers, 2 heads). There is no exploration of varied outlier phrases, different poisoning rates, or evaluation on more modern language models. As such, the empirical evidence is supportive but not strong enough to claim robust practical superiority.

6. **Some notational and editorial issues hamper readability**  
   - There are several small but distracting typos and formatting glitches, e.g., mis-rendered equations around (6)–(7), some “lts” vs “lts”, “v_s*” vs “v_s^{*'}”, and a poorly formatted inline table under “Input / label” on Page 5. This is minor, but in such a technical paper, consistency is important.  
   - The reuse of notation \(l_{tr}\) and \(l_{ts}\) across training and test, together with different ranges for \(\alpha,p_a\), is sometimes confusing. For example, in Theorem 2, condition (d) uses \(\text{poly}(M_1^{\kappa_a})\) and \((1-\alpha)^{-1}\log M_1\) for \(l_{ts}\), while Theorem 1 uses similar quantities for \(l_{tr}\). A short summary table of notations (Table 8 is helpful but buried in the appendix) and a recap of the key scaling relationships in the main text would greatly aid comprehension.  
   - Equation numbers sometimes appear inline and hard to track (e.g., they refer to (200) as the prompt definition, but the actual equation is typed earlier in Section 2 with a different formatting). Cleaning this up would make it much easier to map proofs back to definitions.

7. **Missing discussion of some closely related recent Mamba-ICL theory**  
   - While the paper cites several works on Mamba-like models and ICL (Li et al. 2024b, 2025b; Bondaschi et al. 2025; Joseph et al. 2024), there are other theoretical studies on Mamba’s in-context learning behavior (see “Potentially Missing Related Work” below) that are not discussed.  
   - Given the paper’s focus on being “the first theoretical analysis of the training dynamics of Mamba” (Abstract, Section 1.1), it is important to position precisely how its setting (binary classification with orthogonal patterns and outliers) differs from other ICL regimes (e.g., low-dimensional nonlinear targets or regression tasks) and what is truly novel in terms of mechanisms vs. what is consistent with or extends existing analyses.

---

## Potentially Missing Related Work

1. **Oh, J., Huang, W., Suzuki, T. (2025): “Mamba Can Learn Low-Dimensional Targets In-Context via Test-Time Feature Learning”**  
   - This paper provides a theoretical analysis of Mamba’s ICL capabilities for tasks defined by low-dimensional nonlinear target functions, analyzing feature extraction dynamics and showing that Mamba can adapt via test-time feature evolution.  
   - It is directly related because it also studies Mamba’s in-context learning, albeit for different task families and mechanisms. The authors should (a) cite and discuss this paper in Section 1.2 (Related Works) and (b) clarify in Section 3.1 / 5 how their analysis of binary classification with outliers complements or differs from low-dimensional nonlinear targets, especially regarding robustness and gating behavior.

2. **Jiang, J., Huang, W., Zhang, M. (2025): “Trained Mamba Emulates Online Gradient Descent in In-Context Linear Regression”**  
   - This work analyzes Mamba trained on linear regression ICL tasks and shows that the trained model effectively performs a variant of online gradient descent in context.  
   - It is highly relevant since this paper also interprets Mamba as implementing an algorithm (here, a robust pattern-matching + gating mechanism) via training dynamics. It should be cited in Section 1.2 and contrasted in Section 3.5: this submission focuses on classification under outliers and gating-induced locality, whereas Jiang et al. focus on regression and online optimization. A short comparison after discussing Li et al. (2024b, 2025b) would make the landscape clearer.

(If the authors are already aware of closely related unpublished or concurrent work, they should at least acknowledge them explicitly and sharpen their claim about being the “first” to analyze Mamba’s training dynamics under outliers.)

---

## Questions

1. **How tight are the conditions on \(\kappa_a\) and \(\kappa_a'\)?**  
   - Could the authors clarify which parts of Theorems 1–2 critically depend on the lower bound \(\kappa_a \gtrsim V\beta^{-4}\) and the upper bounds scaling as \(\epsilon^{-1}\)? For example, are there counterexamples if \(\kappa_a\) is small (outliers almost indistinguishable) or huge (dominating the norm), or are these bounds mainly proof artifacts to control certain terms in Lemmas 4–5?  
   - Providing a simple 1D or 2D example (even in the appendix) where too-small or too-large \(\kappa_a\) breaks the gating behavior would significantly clarify necessity vs sufficiency.

2. **Can the authors give a more interpretable bound on prompt length \(l_{tr},l_{ts}\)?**  
   - Conditions like \(p_a^{-1}\text{poly}(M_1^{\kappa_a})\gtrsim l_{tr}\) and \(\alpha^{-1}\text{poly}(M_1^{\kappa_a})\gtrsim l_{ts}\) are quite abstract. For a concrete simplified setting (say \(M_1=O(1)\), \(V=O(1)\), \(\beta=1\)), what do these translate to? E.g., can you show that \(l_{tr},l_{ts}=O((1-p_a)^{-1}\log(1/\epsilon))\) up to moderate constants, or are there hidden exponential dependencies?  
   - A table or simple theorem corollary where constants are instantiated for a canonical regime would help practitioners assess feasibility.

3. **How would the analysis change for multiple relevant patterns per token?**  
   - Real ICL prompts often have inputs that depend on several “features” (e.g., words or patches) jointly. Is there a straightforward extension of the orthogonal pattern framework where each \(\mathbf{x}\) is a sum of \(K>1\) relevant patterns and multiple irrelevant patterns?  
   - If so, are there obvious bottlenecks in your proofs (e.g., Lemma 3’s gradient decomposition) that become intractable, or could one simply track a larger set of pattern coefficients?

4. **Softmax Transformers vs Mamba in this setting**  
   - The softmax experiments in Appendix B.1 (Tables 3–5) suggest that multi-layer softmax Transformers are both robust to outliers and less CQ-sensitive than Mamba. Could the authors clarify whether they expect the same gating-style analysis to be extendable to softmax attention (perhaps via logit clipping or temperature scaling)?  
   - If not, what exactly is the conceptual advantage of Mamba’s gating from a robustness perspective, beyond linear attention, given these empirical results?

5. **Interpretation of Figure 2’s low error for Mamba at high \(\alpha\)**  
   - In **Figure 2**, Mamba achieves classification error below 0.01 at \(\alpha \approx 0.8\) even when all outlier-containing examples have labels flipped or targeted. This is very strong performance. How sensitive is this to hyperparameters like learning rate \(\eta\), depth, and initialization (\(\delta\))?  
   - Have the authors observed regimes where, despite satisfying Theorem 2’s conditions qualitatively, optimization fails (e.g., getting stuck in a bad local basin) and robustness deteriorates? Some discussion would be helpful for practitioners.

6. **Real-world poisoning setups**  
   - For SST-2 (Table 7), the poisoning is implemented using a single trigger phrase. Have the authors tried varying the trigger phrase or using multiple triggers per dataset? Do the qualitative conclusions about Mamba vs linear attention and CQ sensitivity still hold?  
   - If not yet done, could you comment on whether you’d expect similar behavior or whether the orthogonality assumption is too strong once multiple triggers with correlated embeddings are used?

Answers and clarifications to these questions could increase my confidence in the practical implications of the theory and might shift my evaluation slightly upward.

---

## Flag For Ethics Review

No ethics review needed.  

---

## Details Of Ethics Concerns

N/A.

---

## Soundness Rating

3: good.  
The technical development is detailed and largely consistent with stated assumptions, and the proofs of the main theorems appear correct at the level of conference review. Some conditions are quite strong or opaque, and the empirical validation is modest in scope, but there is no obvious fatal flaw in the mathematics or experimental protocol.

---

## Presentation Rating

3: good.  
The paper is generally well organized, with explicit theorems, remarks, and an illustrative experimental section, and key formulas like Equations (3)–(5) are clearly highlighted. However, notational clutter, occasional formatting errors, and the heavy reliance on appendices for core intuitions make it harder to read than necessary.

---

## Contribution Rating

3: good.  
The work advances the theoretical understanding of Mamba’s in-context learning, especially under adversarially corrupted prompts, and offers a meaningful comparison with linear Transformers. While the setting is stylized and the experimental coverage limited, the analysis is nontrivial, mechanistically insightful, and likely to be of interest to the representation learning and sequence modeling communities.

---

## Overall Rating

8: Accept, good paper (poster).  
The paper provides a substantial and technically careful theoretical analysis of Mamba’s training dynamics and robust in-context learning under outliers, including a mechanistic account of how linear attention and gating interact, and a clear comparison with linear Transformers. The main weaknesses lie in the restrictive data model, strong technical conditions, and relatively modest experimental scope, but these do not undermine the core theoretical contributions. I recommend acceptance as a solid theory-oriented poster.

---

## Reviewer Confidence

4: confident.  
I am familiar with the recent ICL and attention-theory literature, read the core proofs (and key lemmas) carefully, and checked the experiments against the stated claims. There might still be technical subtleties or related-work nuances I have not fully captured, but it is unlikely that I have missed a fundamental flaw.