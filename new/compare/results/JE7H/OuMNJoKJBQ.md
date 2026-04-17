---
job_id: 1ee11355-893e-4387-9fb4-55ca4253dcf1
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: OuMNJoKJBQ.pdf
paper: Alignment-Weighted DPO: A Principled Reasoning Approach to Improve Safety Alignment
main_score_norm: 0.8
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper focuses on safety alignment of LLMs using representation probing, CoT fine-tuning, and a modified DPO objective, which fits squarely within ICLR’s scope on representation learning, optimization, RLHF/ preference learning, and societal considerations (safety).

## Minimum Quality
Pass ✅.  
All required sections (Abstract, Introduction, Related Work, Method, Experiments, Results, Conclusion) are present and reasonably detailed. The work is technically non‑trivial, provides extensive experiments and comparisons, and the methodology is clearly enough described to evaluate, despite some issues discussed below.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I did not find hidden prompts, attempts to manipulate LLM reviewers, or other integrity violations in the provided content.

---

# Expected Review Outcome:

## Summary

The paper investigates whether current safety alignment in LLMs is grounded in genuine reasoning or superficial refusal heuristics. Using linear probes and a causal intervention that prunes “reasoning-critical” attention heads, the authors show that reasoning accuracy collapses while safety probing and benchmark safety remain almost unchanged, suggesting decoupling between reasoning and safety behavior.  

To improve reasoning-based alignment, the authors build a long Chain-of-Thought (CoT) dataset covering both safety-critical and utility tasks, then introduce Alignment-Weighted DPO (AW-DPO), which decomposes outputs into reasoning and final answer segments and applies segment-specific preference weights derived from an LLM safety judge. Experiments on multiple base and instruct models, mainly evaluated on SorryBench safety attacks and MMLU utility, show that CoT fine-tuning improves safety with modest utility cost, and AW-DPO further reduces jailbreak success compared to standard DPO and several strong alignment baselines.

## Strengths

1. **Interesting causal probe of “superficial alignment”**  
   - The linear probing plus pruning study is thoughtful and well executed.  
   - **Figure 1** (Page 4) clearly shows that alignment heads achieve near‑100% probe accuracy across layers, while reasoning accuracy is near chance in early layers and only rises later. After pruning the top 10% “reasoning” heads in the first 11 layers, reasoning accuracy collapses but alignment accuracy stays near 100%.  
   - **Table 6** (Page 15) complements this with end‑to‑end metrics: reasoning accuracy drops by 31–42% while safety rates change by ≤1.24%. This provides relatively compelling empirical evidence that current refusal behavior is not tightly coupled to the model’s reasoning pathways.

2. **Reasoning-aware safety dataset with explicit CoT**  
   - The construction of a combined safety + utility CoT dataset (Appendix E) with <think>…</think> tags and detailed rationales for both harmful and benign prompts is timely and practically useful.  
   - The dataset is released, which increases the paper’s value to the community beyond the specific method. The prompts for GPT‑4o rationale generation are clearly documented.

3. **Simple but meaningful extension of DPO to segment‑weighted preferences**  
   - AW-DPO keeps the DPO machinery but introduces a token‑level reward decomposition over “reasoning” and “response” segments, with scalar weights \(w_{\text{reasoning}}, w_{\text{respond}}\) derived from a judge model’s harmfulness scores.  
   - **Figure 2** (Page 5) is a helpful schematic that walks through candidate generation, triple scoring (full / reasoning / response), construction of chosen/rejected pairs, and formation of weighted losses. This figure makes the pipeline easy to follow and highlights the distinction from vanilla DPO which treats the whole answer uniformly.  
   - The formulation in Equations (2)–(4) is largely consistent with standard DPO and gives a clean way to modulate the KL‑like token log‑ratios by segment-specific weights.

4. **Empirical results show consistent safety gains with reasonable utility**  
   - **Table 1** (Page 7) is quite comprehensive: across four model families (Llama‑2‑7B, Llama‑3.2‑3B, Llama‑3.1‑8B, Mistral‑7B) the proposed “+CoT Safety SFT” already substantially reduces attack success rate compared to Safety SFT, and AW‑DPO on top of CoT typically has the best or near‑best safety numbers. For example, on Llama‑3.2‑3B, Average ASR drops from 11.29% (Safety SFT) to 7.60% (CoT Safety SFT) to 0.58% (AW‑DPO), while MMLU remains around 48–52%.  
   - **Table 2** shows that on Llama‑3.1‑8B, AW‑DPO is competitive with or slightly behind STAIR‑DPO‑3 in safety but uses only a single SFT + DPO stage instead of three iterative cycles, which is a meaningful efficiency trade‑off.  
   - **Table 3** demonstrates that an AW‑DPO dataset constructed on Llama‑2‑7B transfers reasonably well to other models, keeping ASR around 1.7–3.1% while maintaining decent MMLU.

5. **Comparison to reasoning models and existing aligned LLMs**  
   - **Figure 3(c)** and **Table 9** compare AW‑DPO models to specialized reasoning LMs (Phi‑4‑Reasoning, Qwen‑Thinking), which do well on utility but are significantly less safe than AW‑DPO. This supports the core claim that generic reasoning alone does not guarantee better safety alignment, and alignment‑specific reasoning is needed.  
   - **Figure 3(b)** and **Table 7** convincingly show that on SorryBench, AW‑DPO can surpass existing aligned chat models (e.g., Mistral-7B-Instruct) in safety metrics, while sometimes trading a small amount of MMLU.

6. **AW-DPO vs. standard DPO ablations**  
   - **Figure 4(b)** (radar chart) and **Table 12** compare standard DPO to AW‑DPO on Llama‑3.1‑8B across different SorryBench categories. AW‑DPO reduces Average ASR from 1.83% to 0.81% while slightly improving utility. This directly supports the claim that fine‑grained weighting over reasoning/response yields extra gains over ordinary full‑sequence DPO.  
   - Ablations on scaling factor \(\alpha\) (Table 4) and learning rate (Table 5) show that performance is relatively stable in \(\alpha\) and behaves as expected with lr, suggesting the method is not hyperparameter‑fragile beyond standard DPO issues.

7. **Clear qualitative intuition and error taxonomy**  
   - The identification of two key failure modes after CoT SFT, “correct reasoning but unsafe final answer” vs. “incorrect reasoning but safe final answer,” is plausible and resonates with practical alignment failures. Figure 3(a) visualizes the distribution of unsafe full responses across these categories, grounding the decision to weight reasoning and answer separately in observed data rather than purely speculative motivation.

## Weaknesses

1. **Causal claims about “superficial alignment” lean stronger than the evidence supports**  
   - The pruning experiment is informative, but the causal conclusion that “current alignment is superficial and does not depend on deep reasoning” (Page 4, last paragraph) is somewhat overstated.  
   - The method deactivates attention heads deemed important for *CommonsenseQA-style* reasoning based on last‑token linear probes, then observes safety probes and a coarse “safety rate” (Table 6) remain high. This does not rule out that other, unpruned pathways support both safety and some forms of reasoning, or that different notions of reasoning (e.g., planning, abstraction) might be implicated in safety.  
   - Additionally, the safety probe itself is a *binary classification* of harmful vs safe prompts, which is arguably closer to pattern‑matching than principled moral reasoning, so showing its independence from a *different* reasoning task may be unsurprising. The paper should temper the causal language and clarify that these are empirical correlates, not definitive mechanistic isolation of a safety “module.”

2. **Mathematical formulation of AW-DPO is slightly inconsistent / underspecified**  
   - In Section 4, the per‑segment weights are defined as continuous fractions
     \[
     w_{\text{reasoning}} = \frac{d_{\text{reasoning}}}{d_{\text{reasoning}}+d_{\text{respond}}},\quad
     w_{\text{respond}} = \frac{d_{\text{respond}}}{d_{\text{reasoning}}+d_{\text{respond}}}
     \]
     with \(d_{\text{reasoning}}, d_{\text{respond}}\) derived from harmfulness score differences.  
   - However in **Equation (3)** the token‑level mask is written as \(w_{s_t}\in\{0,1\}\), which contradicts the earlier fractional definition and suggests a hard mask instead of soft weighting. Then **Equation (4)** uses scalar \(w_{\text{reasoning}}\mathcal{L}^{rs}_{\text{DPO}} + w_{\text{respond}}\mathcal{L}^{rp}_{\text{DPO}}\), but it is not mathematically clear whether Equation (3) uses the *same* scalars or additional binary indicators.  
   - The appendix introduces an extra scaling factor \(\alpha\) multiplying both weights for stability (Appendix H), but \(\alpha\) does not appear in the main equations. This inconsistency makes it harder to precisely reproduce the training objective and to reason about how gradients are redistributed between segments. A corrected notation such as \(m_{s_t}\in\{0,1\}\) for token masks and continuous \(w_{\text{reasoning}}, w_{\text{respond}}\) for scalar coefficients would clarify the architecture.

3. **Heavy reliance on a single LLM judge, with limited calibration of judge biases**  
   - The harmfulness scores \(h_{rs}, h_{rp}, h_f\) and thus AW‑DPO weights are entirely based on GPT‑4o judgments (Appendix J). While the prompt is detailed, the paper does not quantify inter‑rater agreement between GPT‑4o and human annotators for these specific scoring tasks or across the distribution induced by CoT responses.  
   - Section J.3 reports Pearson correlations between two paraphrased prompts (0.66–0.91), which is about robustness to *prompt wording*, not to fundamental biases like over‑penalizing elaborate refusals, being inconsistent across topics, or being systematically lenient in borderline safety cases. These biases could directly affect which segments are upweighted, and hence the direction of optimization, but are not deeply analyzed.  
   - Because all major safety metrics, preference construction, and DPO weighting are judged by the *same family of models* (GPT‑4o or similar), there is a risk of “judge overfitting”: AW‑DPO might optimize for that judge’s idiosyncrasies rather than objective safety.

4. **Evaluation is still concentrated on a single main safety benchmark and an LLM‑based metric**  
   - Almost all safety results rely on SorryBench attack success rate judged by GPT‑4o. This is a strong benchmark, but the paper’s claims about robustness to “diverse jailbreak strategies” would be more convincing with evaluations on additional suites (e.g., JailbreakBench, AdvBench variants, or logic/code-based attack sets), ideally with at least some human adjudication on a subset.  
   - **Table 13** mentions GPTFuzz and PAP attacks, but the table is small (only Llama‑3‑3B) and the metric “LLM Score (Adaptive)” is still LLM‑based. The paper does not detail the attack generation process or report any qualitative patterns that AW‑DPO addresses better than standard DPO there.  
   - Given that the central selling point is “reasoning-aware robustness to jailbreaks,” the empirical scope feels somewhat narrow and could benefit from at least one additional family of distributionally different attacks.

5. **Limited analysis of potential trade‑offs and failure modes of reasoning-based alignment**  
   - While Section 5.7 shows robustness to a simple prefix attack that suppresses explicit `<think>` traces, there is relatively little exploration of *new* failure modes introduced by teaching the model to generate long rationales (e.g., rationalizing harmful behavior, providing arguments for disallowed actions while still refusing at the surface).  
   - The error taxonomy in Figure 3(a) is used mainly to motivate AW‑DPO, but there is no deeper qualitative analysis of *post‑AW-DPO* failure modes. For example, how often do we still see cases with “safe answer, but harmful or logically flawed reasoning,” which might be problematic in high‑stakes domains even if ASR is low?  
   - This is especially important since the judge scores reasoning harmfulness independently; a mis‑alignment between human semantics of “unsafe reasoning” and GPT‑4o’s could lead to subtle risks.

6. **Positioning relative to other DPO‑style safety methods is incomplete**  
   - The related work section on RLHF and DPO (Section 2.2) focuses on classical DPO and online variants (Guo et al. 2024) but does not discuss recent DPO‑based safety methods that also adjust the DPO objective to better handle safety alignment (see “Potentially Missing Related Work” below).  
   - Several of these works, such as SafeDPO or balanced DPO, explicitly weight preferences or introduce safety constraints. A more systematic conceptual comparison would clarify what is new in AW‑DPO: is the main novelty the *token‑level* decomposition into reasoning/response segments, as opposed to instance‑level reweighting? Clarifying this distinction would strengthen the contribution.

7. **Some experimental choices and reporting could be clearer**  
   - **Table 1** mixes base vs instruct models and several training variants. For example, “@Llama‑2‑7B Base” is clearly a base model, but “open-source chat models” are also in the baselines list yet appear only later in Appendix I. It would help to explicitly mark in Table 1 which rows start from base vs instruct models, and which correspond to commercially or community aligned baselines.  
   - It is not fully clear how many training samples are used for AW‑DPO across different models, nor how the balance between harmful and benign prompts is enforced in the preference dataset. Although Appendix G/H provide some numbers (e.g., WildJailbreak adversarial harmful subset, \(k=5\), \(\gamma=0.5\)), a concise summary in the main text would support interpretability of Table 3’s “dataset transfer” experiment.  
   - Statistical uncertainty for safety metrics is only partially reported (mean ± std for subcategories), but Table 1 and 2 do not give confidence intervals across random seeds or different WildJailbreak subsets. This is particularly relevant in the very low‑ASR regime (e.g., 0.5–2%), where a small number of absolute failures can change percentages noticeably.

## Potentially Missing Related Work

The following appear directly relevant yet are not cited:

1. **Kim et al., “SafeDPO: A Simple Approach to Direct Preference Optimization with Enhanced Safety”, 2025**  
   - Proposes a DPO variant specifically tuned for safety alignment. It should be discussed in Section 2.2 (RLHF and DPO) and compared conceptually to AW‑DPO, especially regarding how each method modifies the DPO loss to prioritize safety without overly sacrificing utility.

2. **Zhao et al., “Improving Safety Alignment via Balanced Direct Preference Optimization”, 2026**  
   - Introduces Balanced DPO to mitigate overfitting and collapse in safety alignment. This is directly relevant to the paper’s claim of better robustness to diverse jailbreaks. It should be added to the related work and, if feasible, included as an additional DPO-style baseline in Tables 1–2.

3. **Du et al., “Primal-Dual Direct Preference Optimization for Constrained LLM Alignment”, 2025**  
   - Uses a primal‑dual scheme to enforce constraints within DPO. Since AW‑DPO also introduces structure into the DPO objective (via weighted segments), a discussion in Section 2.2 or 4 comparing constrained vs weighted approaches and their pros/cons for safety would improve positioning.

4. **Erdogan, “Tangent Space Fine-Tuning for Directional Preference Alignment in Large Language Models”, 2026**  
   - Explores preference alignment in low‑dimensional tangent spaces with directional control. This is relevant to the idea of controlling *where* in parameter space DPO updates apply, complementary to AW‑DPO’s control over *which tokens* receive higher preference weight. A short mention in the related work or discussion could clarify how AW‑DPO sits among methods that constrain the update geometry.

5. **Liu et al., “Direct Large Language Model Alignment Through Self-Rewarding Contrastive Prompt Distillation”, 2024**  
   - Proposes a DPO‑like alignment method without human preference labels. Since AW‑DPO also relies on an automated judge (GPT‑4o) for preference construction, this work is methodologically close and should be mentioned in Section 2.2 and/or Section 5.5 when discussing data efficiency and automation.

6. **Xiao et al., “On the Algorithmic Bias of Aligning Large Language Models with RLHF: Preference Collapse and Matching Regularization”, 2025**  
   - Analyzes biases and collapse behaviors in RLHF and DPO. Given this paper’s focus on “superficial alignment” and brittleness, connecting to this analytical perspective would enrich the discussion in Sections 1–3 about why naive DPO may not suffice.

7. **Lu et al., “Alignment and Safety in Large Language Models: Safety Mechanisms, Training Paradigms, and Emerging Challenges”, 2025**  
   - A survey that contextualizes many safety alignment approaches. Citing it in Section 2.1 as a high‑level overview would help readers see where CoT‑based and DPO‑based methods fit within the broader landscape.

## Questions

1. **Clarification of Equation (3) and weighting scheme**  
   - Can the authors clarify the exact mathematical form used in implementation? In particular:
     - Are token‑level masks binary indicators (reasoning vs response) while \(w_{\text{reasoning}}\) and \(w_{\text{respond}}\) are scalar multipliers outside the token sum, or are the fractional weights applied directly at each token?  
     - How is the scaling factor \(\alpha\) from Appendix H integrated into Equations (3)–(4)? A precise rewritten objective would help reproducibility and understanding of gradient flow.

2. **Distribution of harmfulness scores and weight magnitudes**  
   - What is the empirical distribution of \(h_{rs}, h_{rp}, h_f\) from GPT‑4o for chosen vs rejected samples? Are the resulting \(d_{\text{reasoning}}\) and \(d_{\text{respond}}\) typically large and positive, or often small/negative?  
   - It would be useful to see a histogram or summary statistics to understand how often reasoning is judged more harmful than response, and how concentrated the weights \(w_{\text{reasoning}}\) are near 0 or 1. This could also validate the 15% “reasoning‑related misalignment” estimate in Figure 3(a).

3. **Robustness to judge choice**  
   - Have the authors attempted to re‑score a subset of samples with a different judge (e.g., Llama‑Guard, a different GPT series model, or a human panel) to check whether the harmfulness rankings and thus preference pairs are stable?  
   - Even a small‑scale study would increase confidence that AW‑DPO is not overfitting to GPT‑4o’s particular preferences.

4. **Failure mode analysis after AW-DPO**  
   - Can the authors provide qualitative examples of remaining jailbreaks after AW‑DPO, categorized by whether the reasoning, the final answer, or both are problematic?  
   - It would be especially helpful to know whether AW‑DPO fixes the “correct reasoning, unsafe answer” and “incorrect reasoning, safe answer” modes, or mostly shifts failures to more subtle patterns.

5. **Generalization to non‑CoT models and shorter reasoning styles**  
   - The method is designed around explicit `<think>` tags and long CoT traces. What happens if we apply AW‑DPO to models that generate only brief or implicit reasoning, or to tasks where reasoning is primarily internal (no CoT in outputs)?  
   - Could the authors comment on how AW‑DPO might be adapted in those settings (e.g., by segmenting outputs with heuristics, using hidden‑state decomposition, or rewarding implicit reasoning markers)?

Author responses with empirical evidence or clearer mathematical description on these points would increase my confidence in the technical soundness and generality of the approach.

## Flag For Ethics Review

No ethics review needed.

## Details Of Ethics Concerns

N/A.

## Soundness Rating

3: good.  
The main methodology builds reasonably on known tools (probing, DPO) and is implemented consistently; empirical evidence is thorough for SorryBench and several LLM families. Some causal claims are somewhat overstated, and the AW‑DPO objective is under‑specified mathematically, but not fatally flawed.

## Presentation Rating

3: good.  
The paper is generally well written, with helpful figures (especially Figures 1–3 and 4(b–c)) and detailed appendices. Some notational inconsistencies (weights vs masks, missing \(\alpha\) in main text) and slightly cluttered tables detract from clarity, but overall exposition is solid.

## Contribution Rating

3: good.  
The combination of causal probing of safety vs reasoning, a released CoT safety dataset, and a segment‑weighted DPO variant focused on reasoning constitutes a meaningful contribution to the safety‑alignment literature. While AW‑DPO is an incremental modification of DPO, the empirical benefits and practical usability justify a good contribution score.

## Overall Rating

8: Accept, good paper (poster).  
Despite some over‑strong causal rhetoric and missing related work on safety‑aware DPO, the paper offers a well‑motivated and practically relevant technique, backed by substantial experiments and a useful dataset release. The alignment‑weighted DPO idea is simple but effective, and the initial mechanistic probing of safety vs reasoning provides valuable insight for the community. With some clarification of the objective and positioning, this work merits presentation at ICLR.

## Reviewer Confidence

4: confident.  
I am familiar with DPO, RLHF, and LLM safety work, and I carefully examined the equations, figures, and tables. Some implementation details (judge behavior, exact loss scaling) rely on trust, but overall I feel confident in my assessment.