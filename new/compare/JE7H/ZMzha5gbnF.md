---
job_id: f97adc9e-1db9-4101-a5ca-bee77ab28711
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: ZMzha5gbnF.pdf
paper: Toward Safer Diffusion Language Models: Discovery and Mitigation of Priming Vulnerability
main_score_norm: 0.6
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The work is squarely about safety and robustness of diffusion language models, including new training objectives and extensive empirical evaluation, which fits ICLR topics on generative models, representation learning for language, safety, and robustness.

## Minimum Quality
Pass ✅.  
All required sections (Abstract, Introduction, Related Work, Methodology, Experiments, Results, Conclusion) are present, in English, and the work presents a coherent method with reasonably strong experiments and nontrivial analysis. I see no fatal methodological flaw or clearly incorrect theorem.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I did not find any hidden prompts, steganographic text, or instructions targeting automated reviewers beyond normal paper content.

---

# Expected Review Outcome:

## Summary

The paper studies a safety weakness specific to masked diffusion language models (MDLMs), termed *priming vulnerability*: if an affirmative token toward a harmful query appears at an intermediate denoising step, later steps are steered toward a harmful response even for ostensibly aligned models.  

The authors (i) quantify this effect via an “anchoring attack” that directly injects harmful responses at chosen denoising steps and (ii) derive a lower bound (Theorem 4.1) that enables a more efficient GCG-style jailbreak (First-Step GCG) without intervening in the denoising process.  

To mitigate the vulnerability they introduce *Recovery Alignment (RA)*, an RLHF-style training scheme that initializes denoising from intentionally contaminated intermediate states and trains the model to “recover” to safe outputs, showing improved robustness to both priming-type and standard jailbreak attacks with minimal degradation on 11 general capability benchmarks.

---

## Strengths

1. **Clear identification and characterization of an MDLM-specific safety failure mode.**  
   The priming vulnerability is precisely defined in terms of intermediate denoising states and is not just a rebranding of standard jailbreaks on ARMs. The anchoring attack in Section 4.1 provides a clean, controllable knob through the intervention step \(t_{\text{inter}}\), which allows the authors to isolate and measure the effect.  
   - **Figure 2** is quite compelling: even at \(t_{\text{inter}}=1\) (one token injected, with \(L=T=128\)), ASR jumps from roughly 2% to ~20% on LLaDA Instruct, and quickly rises above 80% as \(t_{\text{inter}}\) increases. This visually substantiates that a *single* early affirmative token can severely undermine safety.

2. **Theoretical link between first-step logits and end-to-end harmful generation.**  
   Theorem 4.1, together with Equations (2)–(4), gives a neat lower bound:  
   \[
   \log p_{\pi,m_t}(r_T=r\mid q, r_0) \ge \frac{1}{T}\log \pi_\theta(\tilde r_1=r\mid q, r_0)
   \]  
   under a monotonicity assumption on \(\log\pi_\theta(\tilde r_{t+1}=r\mid q,r_t)\). This yields the First-Step GCG objective (Eq. (4)), which is tractable and differentiable. While the bound is loose, it is conceptually interesting to connect a single-step mask-predictor objective to the whole denoising chain and ties the theory nicely to the priming vulnerability.

3. **Substantial empirical evidence that the vulnerability is practically exploitable.**  
   First-Step GCG is not just a toy construction:  
   - **Table 1** shows that on JBB-Behaviors, ASR for LLaDA Instruct jumps from 2% (no attack) to 58% under First-Step GCG, compared to only 20% under Monte Carlo GCG, while being ~20× faster (0.2h vs 4.3h per prompt). This strongly supports the claim that priming can be exploited by realistic, query-only attackers.  
   - Section C.2 and **Figure 6** empirically validate the monotonicity assumption underlying Theorem 4.1 across three MDLMs, by showing the monotonicity gap \(\Delta_t\) is positive and grows with the fraction of injected tokens.

4. **Recovery Alignment is well-motivated and experimentally solid.**  
   The paper convincingly argues why standard MDLM training (Eq. (5)) and prior MDLM alignment MOSA fail: they only train from a fully masked start \(r_0\), not from contaminated intermediate states \(r_t\) that already contain harmful anchors (cf. inequality (6)).  
   - RA’s objective (Eq. (7)) explicitly conditions on \(r_{t_{\text{inter}}}\sim m_{t_\text{inter}}(\cdot \mid r)\), i.e., re-masked versions of known harmful responses, and then trains the model via GRPO to maximize a safety/usefulness reward over \(r_T\). This is conceptually clean and clearly tailored to the MDLM denoising process.

5. **Strong robustness gains against both priming-style and conventional jailbreaks.**  
   The experimental section is extensive and multi-faceted:  
   - **Table 2** (key table) shows that RA consistently and dramatically reduces ASR for anchoring attacks and PAD/DiJA across LLaDA, LLaDA1.5, and MMaDA. E.g., for LLaDA Instruct, Anchoring at \(t_{\text{inter}}=4\) drops from 44.0% (original) to 1.3% (RA), while MOSA only reaches 24.0%. For First-Step GCG, ASR goes from 58.0% (original) to 11.3% with RA, much lower than DPO (46.3%) or MOSA (28.0%).  
   - **Table 3** shows that RA also improves resilience to PAIR, ReNeLLM, and Crescendo. For example on LLaDA, PAIR ASR falls from 44.3% to 10.0%, and Crescendo from 81.3% to 45.0%. This is an important point: the method is not overfitted to the toy anchoring attack, but helps with attack families that never intervene in the denoising process.

6. **Evidence that RA preserves or slightly improves general capability.**  
   - **Table 4** reports accuracy on 11 benchmarks. For LLaDA and LLaDA1.5, overall averages remain almost unchanged (52.2 → 52.6, and 52.7 → 52.8). Some safety-relevant tasks like TruthfulQA and MBPP modestly improve, which is plausible given the reward-model based alignment. The MMaDA case is even stronger, with average accuracy rising from 33.2 to 35.0.  
   This alleviates a common concern that stronger safety alignment necessarily cripples performance.

7. **Thoughtful ablations and mechanistic analysis.**  
   - **Figure 3(a)** shows that larger \(t_{\max}\) in RA training yields lower ASR across \(t_{\text{inter}}\), but extremely large values lead to reward hacking, which is a nuanced and realistic observation.  
   - **Figure 3(b)** compares linear, uniform, and constant schedules for \(t_{\text{inter}}\); linear clearly dominates, especially at \(t_{\text{inter}}=16\). This supports the curriculum design choice instead of just being another hyperparameter.  
   - The refusal-probability analysis in **Figure 4** and **Figure 5** is insightful: affirmative tokens like “Finally” or even semantically neutral tokens like “mountain” significantly reduce the model’s probability mass on refusal phrases, while some “harmful” tokens do not, clarifying what actually drives priming in practice.

8. **Clarity and presentation quality.**  
   The paper is quite readable given the technical content. **Figure 1** nicely situates the contributions: (a) standard denoising, (b) priming vulnerability via affirmative-token injection, and (c) RA’s recovery from contaminated states. The notation in Section 3 is mostly consistent, Algorithm 1 provides a succinct overview, and Algorithm 2 plus appendix details make implementation reasonably reproducible.

---

## Weaknesses

1. **Theoretical assumption and its scope are under-discussed relative to how heavily it is used.**  
   The central bound in Theorem 4.1 rests on the monotonicity assumption  
   \[
   \log\pi_{\theta}(\tilde r_{t+1}=r\mid q,r_t) \ge \log\pi_{\theta}(\tilde r_1=r\mid q,r_0),\quad \forall t.
   \]  
   This is intuitively plausible but fairly strong: it must hold for *all* \(t\) and the specific harmful response \(r\). The authors provide empirical support in **Figure 6**, but only for three particular MDLMs and the JBB-Behaviors dataset.  
   - There is no analysis of when the assumption might fail (e.g., for odd-length or highly entropic responses, alternative masking schedules, or different \(T,L\)), nor any sensitivity study relating violations of monotonicity to the effectiveness of First-Step GCG.  
   - Moreover, the ELBO derivation in Appendix A relies on Equation (10), but in the main text they jump directly from Eq. (10) to (11)–(13). It would be helpful to explicate that the inequality in (12) comes from plugging in the minimum over \(t\), and to note explicitly that the bound can be extremely loose when the monotonicity gap is large.  
   This is not a fatal flaw, but the paper presents Theorem 4.1 as a relatively general guarantee; in reality, it is an empirical property of a specific family of schedules and models.

2. **Some mathematical / algorithmic details of Recovery Alignment are underspecified or slightly inconsistent.**  
   - In Eq. (7) RA optimizes \(\mathcal{R}(q,r_T)\) where \(r_T\sim p_{\pi,m_t}(\cdot\mid q,r_{t_{\text{inter}}})\), but Algorithm 2 later notes that in practice they replace the full generation probability \(p_{\pi,m_t}\) with \(\pi_\theta(r_T\mid q,r_{t_{\text{inter}}})\) in the GRPO loss (line 16) “because computing gradients of \(p_{\pi,m_t}\) is expensive.” This is reasonable, but then the objective effectively becomes  
     \[
     \max_\theta \mathbb{E}_{(q,r)\in\mathcal{D}_h} \mathcal{R}(q,r_T) \qquad\text{s.t. } r_T\sim \pi_\theta(\cdot\mid q, r_{t_{\text{inter}}}),
     \]  
     which is *not* the same as Eq. (7) with a fixed masking strategy. The paper should be explicit that RA is implemented as RL on the first-step predictor conditioned on intermediate masks, not on the full diffusion chain, and discuss whether this mismatch affects learned behavior.  
   - In Algorithm 1 and Algorithm 2, there is a small inconsistency: Algorithm 1 uses `t_inter ← [t_min + s/S (t_max - t_min)]` while Algorithm 2’s line 6 uses division by \(B\) (\(s/B\)), which seems like a typo. This is minor but confusing for reproduction.  
   - Equation (6) is central to the motivation (“when \(r_t\) includes such affirmative tokens, \(p(r_T=r\mid q,r_t) > p(r_T=r\mid q,r_0)\)”), yet it is stated informally and never empirically quantified. Given the rest of the paper’s care, a small controlled experiment measuring both sides of (6) (for a set of harmful responses and intervention steps) would make the argument more concrete.

3. **Limited model and architecture diversity.**  
   All main experiments are on three closely-related MDLMs from the same family (LLaDA Instruct, LLaDA 1.5, and MMaDA MixCoT), with identical diffusion schedule \(T=L=128\) and standard random masking. It is unclear how RA and the measured vulnerability generalize to:  
   - Different masking schedules (e.g., more sophisticated non-random schedules, fewer steps, dynamic re-masking).  
   - Other discrete DLMs such as DiffusionBERT-like models or those trained with different loss formulations.  
   - Larger-scale models where safety training has already been extensively tuned.  
   This matters because the paper’s framing is quite general (“Toward safer diffusion language models”), yet all evidence rests on one architectural line. Even an ablation on a smaller continuous DLM or a different discrete DLM would help demonstrate broader applicability.

4. **Evaluation slice is focused but somewhat narrow in tasks, judges, and settings.**  
   The safety evaluation mainly uses JBB-Behaviors and a 50-prompt subset of AdvBench (Section C.3). Those are solid datasets but do not cover, for example, multilingual harms, more subtle misinformation or political persuasion prompts, or long multi-turn dialogues.  
   - While the authors *do* use three evaluators (GPT-4o, LlamaGuard 3, and refusal phrases), all are automated. Given RA’s nontrivial training and potential for reward hacking (acknowledged in Section 6.4), including at least a small-scale human evaluation on ambiguous cases would considerably increase confidence that lowered ASR corresponds to genuinely safer behavior rather than style changes that trick judges.  
   - The paper notes in C.3 that ASR from the guardrail model is often lower than keyword-based ASR and attributes this to semantic paraphrasing. This raises the question of whether RA pushes the model toward evasive wording that still semantically responds to the harmful intent but avoids trigger phrases. A few concrete qualitative examples would be useful.

5. **Comparison to closely related, concurrent safety work on DLMs is missing.**  
   There is an explicit discussion of MOSA (Xie et al., 2025) as the only MDLM-specific safety training baseline, but other very relevant concurrent work appears absent. In particular, more systematic DLM safety studies such as *DiffuGuard* (see below) are not cited or contrasted. Given that the paper’s main conceptual claim is “MDLMs exhibit a specific safety failure mode rooted in their denoising process,” it is important to clarify how this complements or differs from other recent analyses of safety in diffusion LMs, beyond saying that PAD/DiJA “implicitly exploit the priming vulnerability.”

6. **Some experimental design choices deserve more justification.**  
   - The training in RA uses BeaverTails, which contains both harmful and harmless prompts, but Eq. (7) and Section 5 assume a set \(\mathcal{D}_h\) of *harmful* query-response pairs. Section D.4 acknowledges that including harmless pairs avoids over-refusal, but the exact sampling scheme (harmful vs harmless mix, whether \(r\) in Eq. (7) is always harmful, etc.) is not fully spelled out. Since the core idea is to start from “contaminated” states derived from harmful responses, it is important to describe what happens for harmless samples: do they also get “harmful initialization,” or is RA only applied to harmful subsets in practice?  
   - First-Step GCG experiments use a suffix length of 20 tokens and 500 iterations (Section 4.2). It is not clear how sensitive ASR is to these choices. For example, if a weaker attacker (smaller suffix or fewer iterations) is still able to reach 40–50% ASR, that would underscore the severity of the vulnerability; conversely, if performance drops sharply, then the practicality of First-Step GCG for real adversaries is less obvious.

7. **Generality with respect to generation length and masking is under-explored.**  
   Section C.5 and **Figure 9** study generation length \(L\in \{32,64,128,256,512\}\). For the original model, ASR is nearly saturated across lengths, and for RA, ASR increases when the intervention step is a large fraction of \(L\), especially when \(L\) is small. The text speculates that long generation length can push the model to answer rather than refuse. However, this section feels somewhat superficial:  
   - There is no discussion of *how* RA could be adapted to better handle longer \(L\) (e.g., scaling \(t_{\max}\) with \(L\), or adjusting reward model emphasis on early vs late tokens).  
   - The masking schedule is always random; in real systems, more complex schedules might be used to trade speed and quality, and it is not demonstrated whether RA remains effective there.

8. **Computational cost and practicality concerns.**  
   RA is RL-on-top-of-DLM with GRPO and multiple candidate generations per step. Section C.4 gives training time ~16 hours on 4×H100 for 2,500 steps for each 8B model. This is acceptable for research but non-trivial for deployment at scale, especially if RA needs to be rerun for each model variant or updated regularly.  
   - The paper does not quantify the number of GRPO rollouts per batch or give a concrete comparison vs more lightweight methods like SFT/DPO or MOSA in terms of FLOPs.  
   - Given that many safety efforts for ARMs aim to be data- and compute-efficient, some discussion of whether RA can be approximated by a simpler supervised variant (they briefly mention this in Limitations) or combined with DPO-style objectives would be valuable.

9. **Minor clarity / notation issues.**  
   - In Eq. (1) the notation \(p_{\pi, m_t}(r_T \mid q, r_0)\) uses multiple nested integrals; strictly, the inner integral is a sum over discrete token sequences for \(\hat r_t\), not a Lebesgue integral. This is conceptually fine but slightly sloppy; explicitly writing \(\sum_{\hat r_t} m_t(r_t\mid \hat r_t)\pi_\theta(\hat r_t\mid q,r_{t-1})\) would avoid confusion.  
   - In Eq. (9) and surrounding text, the indicator is written as \(\mathbbm{1}[r^i=M]\), but by definition only *masked* positions contribute to the loss; explaining that \(\tilde r_{t+1}\) only predicts masked tokens (as in discrete diffusion) would clarify the semantics.  
   - Some symbols are overloaded: \(r_t\) sometimes is “partially masked response,” other times in Algorithm 2 line 7 they write \(r_{t_{\text{inter}}}\gets m_{t_{\text{inter}}}(\cdot\mid r^{(i)})\) where the input is the *full harmful response*. This is correct but would benefit from an explicit remark that in RA’s training the model’s predicted \(\hat r_{t_{\text{inter}}}\) is replaced by a ground-truth harmful response rather than its current prediction.

Overall these issues are mostly about scope, generality, and clarity rather than correctness, but they do limit how broadly one can take the conclusions “as is.”

---

## Potentially Missing Related Work

1. **Li, Z., Nie, Z., Zhou, Z. (2025): “DiffuGuard: How Intrinsic Safety is Lost and Found in Diffusion Large Language Models.”**  
   This work (not cited in the submission) analyzes safety in diffusion language models and how intrinsic safety can be compromised and restored. It seems directly relevant to the paper’s core theme of safety vulnerabilities specific to diffusion LMs. The authors should:
   - Discuss DiffuGuard in Section 2.2–2.3, contrasting their notion of “priming vulnerability” with any safety failure modes cataloged there.  
   - Clarify whether DiffuGuard considers intermediate denoising states and how its proposed defenses relate or differ from Recovery Alignment.  
   - Ideally, if DiffuGuard includes defenses or evaluation protocols, they should be compared empirically or at least conceptually in Section 6.

If DiffuGuard already observes a similar phenomenon (e.g., sensitivity to intermediate tokens or trajectories), proper positioning is crucial to avoid the impression that priming vulnerability and RA are rediscovering or repackaging prior insights.

---

## Questions

1. **Scope and robustness of the monotonicity assumption.**  
   Can the authors provide more detailed statistics on how often the monotonicity condition in Theorem 4.1 is violated across prompts and models? For example, what fraction of JBB-Behaviors samples have \(\Delta_t<0\) for some \(t\), and how does that correlate with First-Step GCG’s success or failure on those samples?

2. **Exact training mix of harmful vs harmless examples in RA.**  
   Section D.4 states that the entire BeaverTails dataset (harmful + harmless) is used because training only on harmful pairs leads to over-refusal, but Eq. (7) and Algorithm 2 are written in terms of a harmful set \(\mathcal{D}_h\) and contaminated initialization from a harmful response \(r\).  
   - For harmless examples, do you still construct contaminated states via anchoring with some harmful response, or do you skip the intervention step (i.e., use \(t_{\text{inter}}=0\))?  
   - How does the presence of harmless examples interact with the linear schedule on \(t_{\text{inter}}\)?

3. **Qualitative behavior and possible evasiveness.**  
   Can the authors show a small number of before/after examples for RA under First-Step GCG and conversational attacks, where GPT-4o and LlamaGuard disagree about harmfulness? This would clarify whether RA is truly steering the content toward safe refusals or mostly rephrasing harmful information in less detectable forms.

4. **Generality to other masking schedules or continuous DLMs.**  
   Have you tried RA on MDLM variants with fewer denoising steps, deterministic (non-random) masking, or on continuous DLMs (e.g., Diffusion-LM-style embedding noise)? Even a small pilot experiment or argument would help support the claim that priming vulnerability is a general phenomenon of diffusion LMs and that RA is not narrowly tied to the specific LLaDA architecture and schedule.

5. **Data and compute efficiency.**  
   RA training currently requires GRPO with many rollouts. Do you have any preliminary results or intuition on a supervised analogue (as hinted in the Limitations) where one constructs safe “recovery” responses for contaminated states and trains with a DPO-style or SFT objective? Would you expect similar robustness with significantly less compute?

6. **Sensitivity to First-Step GCG hyperparameters.**  
   How does ASR for First-Step GCG change if you halve the number of optimization iterations or reduce the suffix length? This would clarify whether the attack is practical for less resourceful adversaries and whether RA’s robustness is robust to such variations.

Clarifications along these lines would strengthen the paper’s claims and might shift my rating upward.

---

## Flag For Ethics Review

No ethics review needed.

---

## Details Of Ethics Concerns

N/A. The paper uses public datasets of harmful prompts and responses, explicitly focuses on increasing safety, and does not release any new harmful data beyond what is needed for reproducibility. The methodologies are standard in safety research.

---

## Soundness Rating

3: good.  
The central claims are largely supported by experiments and the math is internally consistent given its assumptions, but the monotonicity assumption’s scope and the gap between the ideal RA objective and the implemented algorithm deserve more detailed discussion.

---

## Presentation Rating

3: good.  
The paper is generally clear, well-organized, and uses figures/tables effectively, though a few notational inconsistencies and missing clarifications (e.g., Algorithm 2’s schedule formula) detract from full excellence.

---

## Contribution Rating

3: good.  
Identifying and quantifying an MDLM-specific priming vulnerability, proposing a tailored alignment method, and demonstrating both a stronger attack and a mitigation are solid contributions, though the evaluation is somewhat limited to a single architectural family and misses comparison to at least one closely related concurrent work.

---

## Overall Rating

6: Marginally above the acceptance threshold. But would not mind if paper is rejected.  

The work is technically solid, timely, and offers a clear conceptual contribution (priming vulnerability + recovery alignment) with convincing empirical evidence on multiple MDLMs. At the same time, theoretical assumptions are somewhat narrowly validated, model/architecture diversity is limited, and related work coverage has a notable gap. With these caveats, I lean toward acceptance, particularly given the growing importance of safety for diffusion LMs, but I would welcome stronger positioning and broader experiments in a camera-ready version.

---

## Reviewer Confidence

4: confident.  
I am familiar with diffusion models, alignment methods, and jailbreak literature, and I carefully examined the equations and experimental tables. Some aspects (e.g., concurrent DLM safety work, alternative architectures) could change my view slightly if new evidence is presented, but I am reasonably confident in this assessment.