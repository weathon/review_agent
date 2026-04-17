---
job_id: 08061bd9-25ed-4e82-8ef4-af2feeb2da66
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: eu3PwSle8J.pdf
paper: Enforcing Instruction Hierarchy via Augmented Intermediate Representations
main_score_norm: 0.6
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper studies defenses against prompt injection in LLMs via architectural changes to how instruction-hierarchy signals are represented, which squarely fits ICLR’s scope on representation learning, safety, and optimization for large language models.

## Minimum Quality
Pass ✅.  
All required sections (Abstract, Introduction, Related Work, Method/Approach, Experiments, Results, Conclusion) are present and reasonably clear. The method is technically simple but coherent, experiments are substantial across multiple models and attacks, and there are no obvious fatal methodological or theoretical errors.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I see no signs of hidden prompts, steganographic instructions, or attempts to manipulate review systems; the text is standard scientific prose.

---

# Expected Review Outcome:

## Summary

The paper addresses defenses against indirect prompt injection attacks on LLMs by strengthening the representation of an instruction hierarchy (IH) signal inside the model. Existing IH defenses encode privilege levels (system/user vs data) only at the input layer via delimiters or segment embeddings, which the authors empirically show to “wash out” across layers.  

They propose Augmented Intermediate Representations (AIR), which add a small, layer-specific embedding table and inject the IH embedding into every decoder block (and before the final logits). Experiments on Llama‑3.2‑3B, Qwen‑2.5‑7B, and Llama‑3.1‑8B with both SFT and DPO show substantially lower attack success rates for strong gradient-based attacks (GCG, Astra) and improved separation on the SEP benchmark, with minimal utility degradation.

---

## Strengths

1. **Clear and targeted problem formulation with a focused design change.**  
   The paper isolates a specific limitation in IH-based defenses: they only inject privilege-level information at the input. Figure 3 (Page 5) is quite compelling: average cosine similarity between hidden states for different privilege levels with Delimiters and ISE climbs toward 1.0 in deeper layers, whereas AIR maintains noticeably lower similarity, supporting the claim that input-only IH signals degrade and that AIR keeps them more distinguishable.

2. **Simple, well-motivated mechanism with minimal overhead.**  
   The AIR change is architecturally small: for each decoder layer \(j\), a \(K\)-entry embedding table \(S_j\) is indexed by privilege level \(k_i\), and the corresponding vector \(\vec{s}_j^k\) is added to the intermediate token representation \(\vec{x}_{ij}\) as in Equation (1) (Page 5). This is easy to implement in existing decoder-only transformers, and the parameter overhead calculation for Llama‑3.1‑8B (0.4M parameters, 0.005%) makes the cost credible.

3. **Strong empirical robustness across models, training methods, and attacks.**  
   Table 1 (Page 8) is a key result: across three models and both SFT and DPO, AIR consistently yields the best or near-best attack success rates for gradient-based attacks. For example, for Qwen‑2.5‑7B with DPO, GCG ASR drops to 1.6% with AIR versus 7.7% (ISE) and 32% (Delim); similar large margins appear for Astra. The loss curves in Figure 7 (Page 9) show that under momentum-boosted GCG, the attacker’s loss decays much more slowly against AIR, with higher mean and variance, which convincingly supports the robustness claim.

4. **Robustness–utility tradeoff is measured and generally favorable.**  
   Utility is not hand-waved; Figure 6 (Page 8) shows AlpacaEval win rates for all combinations of models / training schemes / IH mechanisms, and AIR is almost always within ~2 percentage points of the non-adversarial “None” baseline, often matching or slightly exceeding Delim and ISE. On SEP (Figure 8, Page 9), AIR+DPO sits at the top-right in the separation–utility plane for all three models, suggesting it is not merely overfitting “ignore data” but preserving instruction usefulness.

5. **Additional interpretability evidence that AIR maintains IH information internally.**  
   The linear probing experiment in Appendix E (Figure 10, Page 14) is a useful sanity check: probes trained to predict privilege level from hidden states are near chance for Delim, degrade over depth for ISE (from perfect to ~91%), and stay near perfect across all layers for AIR. Along with Figure 3, this strongly reinforces the central narrative that AIR keeps IH information accessible throughout the network.

6. **Good breadth of evaluation datasets and scenarios (including BIPIA).**  
   In addition to AlpacaFarm and SEP, Appendix F reports results on the BIPIA benchmark (Table 2, Page 15), with three task types (email, code, table). Again AIR is consistently strongest or tied for best across most tasks and models, especially under DPO. This reduces the risk that the gains are specific to one synthetic setup.

7. **Positioning relative to concurrent work and existing paradigms.**  
   The analogy with positional embeddings and RoPE (Page 5–6) is apt and helps conceptualize why repeatedly injecting a signal across layers can matter. The discussion of ASIDE (Appendix D) also shows awareness that orthogonal approaches (e.g., enforcing orthogonality of IH vs content embeddings) exist, and frames AIR as a complementary architectural route.

---

## Weaknesses

1. **Limited ablation and design-space exploration of AIR itself.**  
   The method’s core decision is “inject per-layer IH embeddings via addition at every decoder block,” but the paper does not seriously test alternatives. Examples of missing ablations that matter for understanding the mechanism:
   - Injecting IH embeddings at a subset of layers (early only vs late only vs every 4th layer) to see whether robustness is mostly driven by a few critical depths.
   - Sharing the embedding table \(S_j\) across layers vs layer-specific tables; this could reduce parameters and test whether layer specialization is important.
   - Scaling factors for \(\vec{s}_j^k\): Equation (1) uses direct addition, but given the sensitivity to magnitude (as seen in Appendix B.2 for Qwen, where \(\sigma\) had to be 5× larger), it is important to know whether a learned scalar or normalization changes robustness/utility.
   Right now, AIR is essentially a single design point, and without ablations we cannot tell if the gains come from per-layer injection, from magnitude differences, or from some incidental regularization effect.

2. **Hyperparameter dependence and model-specific tuning are under-explored.**  
   Appendix B.2 notes that AIR underperforms on Qwen unless the IH embedding initialization standard deviation is hand-tuned from 0.02 to 0.1, and then applies the same tweak to ISE “for fairness”. This raises two concerns:
   - It suggests robustness is quite sensitive to the relative scale of \(\vec{x}_{ij}\) and \(\vec{s}_j^k\), but the main paper does not quantify how sensitive, nor does it explore whether a single heuristic (e.g., match norms of intermediate states) works across architectures.
   - The tuning was done for Qwen but not systematically for others; it is unclear whether the reported AIR vs ISE comparisons are on equally well-tuned settings across models.
   Without a more systematic treatment, the practical recipe for extending AIR to new architectures is under-specified.

3. **Threat model and attack diversity are narrower than advertised.**  
   All robustness results, including the flagship claims in Table 1 and Figure 7, are with a fixed target string (“hacked!”) and two white-box attack families (momentum GCG, Astra). While these are strong baselines, some relevant dimensions are missing:
   - No evaluation with untargeted or multi-target attacks (e.g., different malicious payloads, more complex adversarial instructions) to test whether AIR generalizes beyond this single-token pattern.
   - No explicit evaluation on indirect attacks exploiting semantic/topic shifts (e.g., “TopicAttack”-style strategies) where the adversary gradually changes context rather than inserting a hard override.
   The paper hints that AIR improves “instruction separation” in general, but the concrete attack space explored is still limited and somewhat specialized.

4. **Missing direct empirical comparison to concurrent architectural work (ASIDE).**  
   Appendix D briefly mentions ASIDE as a concurrent method that also recognizes IH signal degradation and proposes orthogonality constraints at the input layer. However, the paper does not empirically compare AIR to ASIDE or even reproduce a low-cost approximation (e.g., orthogonalizing IH vs token embeddings at training start). Since both methods attack the same core problem (preserving IH signals across depth) via different mechanisms, a head-to-head comparison would meaningfully contextualize AIR’s contribution and clarify whether per-layer embeddings provide a clear advantage or just another point in the design space.

5. **Evaluation scope is restricted to single-turn chat; agentic and multi-turn scenarios are only discussed abstractly.**  
   The motivation is framed around agentic settings and untrusted data sources (Introduction, Page 1–2), yet all experiments are single-turn, non-tool-using chat completions (AlpacaFarm, SEP, BIPIA). Important questions remain untested:
   - Does AIR interfere with tool-using behavior, where data segments are not always “lower privilege” but sometimes contain instructions to tools?
   - In multi-turn histories where older user turns might become “data” relative to new instructions, how is the privilege assignment handled and does AIR still enforce sensible hierarchies?
   These limitations are acknowledged in Appendix A but are significant for assessing real-world impact.

6. **Mathematical formulation is minimal and leaves several important details implicit.**  
   The central equation (1) defines \(\vec{x}'_{ij} = \vec{x}_{ij} + \vec{s}_j^k\) with \(\vec{s}_j^k = S_j[k_i]\), but several aspects are under-specified:
   - How are privilege levels assigned for response tokens (\(P_2\)) during generation, especially at decoding time when future tokens are unknown? The text says \(P_2\) is for response tokens (Page 6), but it is not clear whether the IH embedding is applied to the entire generated sequence, and if so, whether this exacerbates exposure bias or affects sampling.
   - During DPO fine-tuning with LoRA, are the IH embedding tables \(S_j\) updated, frozen, or jointly trained with LoRA adapters? The paper states that DPO uses parameter-efficient tuning on q_proj and v_proj, but does not specify how AIR parameters fit into that optimization.
   - For SEP, the evaluation defines separation \(S\) in terms of responses \(y_i^f\) and \(y_i^D\), but all IH-aware models share the same positional/template formatting (Figure 5). It would be helpful to explicitly connect the mathematical treatment of IH levels \(k_i\) with the way probes/instructions are inserted.
   These omissions do not invalidate the experiments but make it harder to reason formally about the behavior of AIR.

7. **Related work coverage on indirect prompt injection and detection is incomplete.**  
   The main related work section and Appendix D discuss several detection and IH-based defenses, but key recent works on indirect prompt injection, especially around BIPIA and detection, are not cited (see next section). Given that the paper even uses BIPIA in Appendix F, omitting its primary reference is a noticeable gap. Similarly, several recent detection-based defenses specific to indirect injection are not discussed, which weakens the contextualization of AIR relative to the broader security space.

8. **Some inconsistencies or ambiguous details in experimental protocol.**  
   A few places where clarification is needed to fully trust the numbers:
   - In Table 1, SEP is listed under “Attack” with ASR values (e.g., 2.7%), but SEP in Section 5.4 is defined via utility and separation metrics. It appears the authors report the fraction of examples where the model incorrectly follows data-embedded probes, but this mapping from SEP to “ASR” is never explicitly defined in the main text.
   - For AlpacaFarm robustness, ASR under GCG and Astra uses likelihood-based criteria (Page 7), not actual generation. The thresholding details (e.g., what probability mass or log-likelihood cutoff constitutes success) are not described, which affects the interpretation of the absolute percentages in Table 1.
   - Figure 6 shows win rates with small deltas labeled above bars, but it is never stated whether differences are statistically significant or within expected judge noise of AlpacaEval.

Overall these are issues of thoroughness and clarity more than fatal flaws, but they limit how definitive one can be about the generality and reproducibility of the reported gains.

---

## Potentially Missing Related Work

1. **Yi et al., “Benchmarking and Defending Against Indirect Prompt Injection Attacks on Large Language Models,” 2023.**  
   This work introduces the BIPIA benchmark for indirect prompt injection, which the authors actually use in Appendix F (Table 2) but do not cite. It should be explicitly referenced when BIPIA is first mentioned and discussed in Section 3.2 or Appendix D as a foundational benchmark for this threat model.

2. **Chen et al., “TopicAttack: An Indirect Prompt Injection Attack via Topic Transition,” 2025.**  
   TopicAttack proposes a more subtle, topic-transition based indirect attack, which directly relates to the paper’s goal of robust IH enforcement. It should be discussed in Section 3.1 as a representative of non-trivial indirect attacks, and ideally AIR should be evaluated against it or at least argued about qualitatively.

3. **Kang et al., “Mitigating Indirect Prompt Injection via Instruction-Following Intent Analysis,” 2025.**  
   This introduces IntentGuard, a defense focusing on analyzing and filtering instruction-following intent. It is complementary to architectural IH defenses like AIR and should be cited in Appendix D’s “Detection-Based Defenses” or Section 3.2, clarifying the distinction between internal architectural robustness (AIR) and external intent analysis.

4. **Wen et al., “Defending against Indirect Prompt Injection by Instruction Detection,” 2025.**  
   InstructDetector is another detection-based mechanism tailored to indirect injection. It belongs alongside the detection work already sketched in Appendix D and should be cited and briefly contrasted (e.g., detector vs. integrated IH).

5. **Chen et al., “Can Indirect Prompt Injection Attacks Be Detected and Removed?”, 2025.**  
   This paper studies the feasibility of detecting and neutralizing indirect prompt injection. It is relevant both to Appendix D’s survey of detection methods and to positioning AIR as a complementary approach; including it would help emphasize that AIR targets robustness rather than detection.

---

## Questions

1. **Scope of IH injection across layers and tokens.**  
   Have you tried variants where IH embeddings are injected only into a subset of layers (e.g., first \(L/2\), last \(L/2\), or every 4th layer)? If so, how does robustness (GCG/Astra ASR in Table 1) and separation (Figure 8) change? This would help confirm that per-layer injection is actually necessary.

2. **Training behavior of AIR parameters under DPO.**  
   In the DPO setting, are the AIR embedding tables \(S_j\) (and the final-layer IH embedding) trainable together with LoRA, or are they frozen after SFT? If they are trainable, can you report approximate parameter counts updated in each stage and whether freezing vs training them materially affects the robustness curves in Figure 7?

3. **Clarification on SEP-as-ASR in Table 1.**  
   How exactly is SEP converted into “ASR (%)” in Table 1? Is it the fraction of cases where the model follows data-segment probes (i.e., includes the witness in \(y_i^D\))? Please formalize this mapping and, if possible, add a sentence in the main text clarifying why lower SEP-ASR corresponds to better separation.

4. **Generality beyond a single target string.**  
   Have you tested AIR using different targeted payloads (e.g., longer or more complex adversarial instructions, or non-ASCII content) for GCG/Astra on AlpacaFarm? If yes, how stable are the relative gains vs ISE/Delim? If not, can you comment on whether the defense might be over-adapted to the specific “hacked!” target?

5. **Interaction with multi-turn or tool-using agents.**  
   How would you propose assigning privilege levels \(k_i\) in realistic agentic systems where some data (e.g., tool outputs) might legitimately contain instructions? Do you anticipate AIR to interfere with such flows, and are there heuristics (e.g., time-varying IH levels) that would mitigate this?

6. **Comparison with ASIDE.**  
   Given that ASIDE tackles the same “IH signal degradation” issue via orthogonality constraints, are there specific reasons you did not include a comparison (e.g., complexity of reimplementation)? Any qualitative arguments or preliminary experiments on whether AIR and ASIDE could be combined (orthogonalized input IH + per-layer embeddings) would be useful.

---

## Flag For Ethics Review

- No ethics review needed.

## Details Of Ethics Concerns

N/A.

---

## Soundness Rating

3: good.  
The core mechanism is technically simple and correctly specified, and the empirical evaluation is broad across models, datasets, and attack types, though there are missing ablations and some under-specified training details.

---

## Presentation Rating

3: good.  
The paper is generally clear and well-structured, with informative figures (especially Figures 2–4, 6–8, and 10) and tables (Tables 1 and 2). Some methodological details, related work coverage, and notation around SEP/ASR could be sharpened.

---

## Contribution Rating

3: good.  
The conceptual step of repeatedly injecting IH information across layers is incremental but meaningful, and the experimental evidence suggests a real robustness gain in an important security problem. Lack of ablations and incomplete comparison to concurrent architectural work limit how high this score can be.

---

## Overall Rating

6: Marginally above the acceptance threshold. But would not mind if paper is rejected.  

The paper addresses a practically important problem with a focused architectural modification, supports its claims with diverse and convincing experiments, and provides interpretability-style evidence that the mechanism works as intended. At the same time, the design space is only lightly explored, several implementation details and attack variations are missing or under-specified, and related work coverage has notable gaps. I lean toward acceptance because the core idea is simple, plausible, and seems to offer real robustness improvements with tiny overhead, but I would expect a camera-ready version to significantly strengthen ablations, clarify protocols, and better position against recent and concurrent work.

---

## Reviewer Confidence

4: confident.  
I am familiar with IH-based defenses, prompt injection attacks, and transformer architectures, and I carefully checked the modeling and experimental methodology, though I did not attempt to re-implement the system.