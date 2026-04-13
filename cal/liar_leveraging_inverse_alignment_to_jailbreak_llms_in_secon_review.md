=== CALIBRATION EXAMPLE 25 ===

# Final Consolidated Review
## Summary
This paper proposes LIAR, a jailbreak method that treats adversarial prompt generation through an alignment-style lens and implements a training-free best-of-\(N\) attack: sample many suffixes from a small base LM (primarily GPT-2), score them via target-model behavior, and keep the best. Empirically, the paper shows that this simple black-box sampling approach can produce very low-perplexity jailbreak prompts quickly, but the main comparative claims are weakened by mismatched evaluation protocols and by theory that is only loosely connected to the implemented method.

## Strengths
- **The paper surfaces a practically important attack regime: cheap multi-attempt black-box jailbreaking with readable prompts.** Concretely, LIAR uses a very small adversarial generator (GPT-2, 124M) and achieves very low suffix perplexity in Table 1 (around 2.1, versus 12.09 for AdvPrompter and far higher for GCG/AutoDAN). This is a specific and meaningful observation: strong attacks need not rely on unnatural strings or expensive optimization.
- **The method is genuinely lightweight and black-box with respect to the target model.** The paper does not require gradients/logits from the target model, and the attack setup in Figure 1 / Section 3 is operationally simple: generate candidate suffixes externally and test them by querying the target. That makes the work relevant for API-only threat models.
- **The ablations are useful and concrete.** Tables 2–5 show how adversarial generator choice, temperature, suffix length, and target response length affect success rate, perplexity, and latency. In particular, the temperature ablation provides a useful empirical insight that moderate temperature reduction improves ASR while preserving diversity enough for best-of-\(N\).
- **The paper’s conceptual framing is interesting even if overstated.** Rewriting prompt search as a KL-regularized reward optimization problem (Eq. 4) does provide a clean bridge between alignment-style objectives and jailbreak generation. As a lens for thinking about the problem, this is a worthwhile perspective.

## Weaknesses

### Major:
- **The headline empirical comparison is not like-for-like, so the central “comparable ASR in seconds rather than hours” claim is not adequately established.**  
  This is explicit in the paper itself. Table 1 states: **“TTA1 for our method is computed for ASR@100, whereas TTA1 for all other methods are computed for ASR@1.”** That asymmetry makes the speed/effectiveness claim hard to interpret as a fair comparison. The paper’s framing repeatedly leans on “comparable to SoTA” while using different query budgets and asymmetric TTA accounting. Since this is a central empirical claim in the abstract, Figure 1, and Section 5, it is a substantive issue rather than a presentation nit.
- **The ASR evaluation protocol differs from standard settings in a way the paper acknowledges can affect ASR, weakening cross-method comparisons.**  
  Section 5.3 states that the paper reports ASR based on the **first 32 generated target tokens instead of the more standard 150**, and further notes that this can change ASR: *“Generating fewer \(y\) tokens does result in a slightly lower chance of an unsuccessful attack keyword being present resulting in a higher ASR.”* The paper claims the difference is “relatively small,” but does not provide matched comparisons against baselines under the same 32-vs-150 protocol. Because both ASR and TTA depend on this choice, the reported superiority is not fully supported by apples-to-apples evidence.
- **The implemented method is much simpler than the paper’s central “jailbreaking via alignment” framing suggests.**  
  The actual attack used in experiments is Eq. (6): sample \(N\) suffixes from a base model and select the highest-reward one. No policy optimization or alignment update is performed. The RLHF-style formulation in Eqs. (4)–(5) is therefore more of a conceptual reinterpretation than an algorithm the paper actually instantiates. That does not make the paper invalid, but it does mean the novelty is narrower than the title/abstract imply: the concrete contribution is a best-of-\(N\) black-box attack with a small unsafeguarded generator, not an alignment procedure in the usual sense.
- **The theoretical section is only loosely aligned with the implemented attack, and some notation/object mismatches reduce confidence in the claimed insights.**  
  There are several places where the formal objects shift in a confusing way: Section 3 defines optimization over suffix distributions \(\rho(q|x)\), while Section 4 introduces quantities over output distributions \(\pi(y|x)\); Eq. (9) defines the suboptimality gap in terms of \(y \sim \rho(\cdot|x)\), conflating prompter distributions and response distributions. This is not just cosmetic, because the paper uses these results to motivate claims about why LIAR works and how close it is to optimality. As written, the theory reads as a stylized analogy rather than a tight analysis of the actual attack pipeline.
- **The practical success metric appears to rely on a brittle notion of attack success, and the paper itself acknowledges susceptibility to prompt-drift / keyword-matching artifacts.**  
  Section 5.3 explicitly notes a limitation: longer suffixes can cause **“prompt-drift”** where \([x,q]\) may ask for content far from \(x\), and this is *“a limitation of the keyword matching aspect of the ASR metric being used.”* Given that the method is a high-volume sampler, this matters: some apparent successes may be artifacts of the evaluation criterion rather than genuine harmful compliance with the original request. The paper would need a stronger evaluator or case-based validation to support its stronger effectiveness claims.

### Minor
- **Performance is uneven across targets, which weakens the broad claims about alignment vulnerability.**  
  Table 1 shows strong results on some models (e.g., Vicuna-7b, Mistral-7b, Falcon-7b, Pythia-7b by ASR@100), but much weaker results on LLaMA-2-7b (3.85% ASR@100). Given the paper’s framing around inherent vulnerabilities of current alignment strategies, this heterogeneity should be discussed more carefully rather than folded into a broad general claim.
- **The “unsafe reward” used in practice is under-specified.**  
  The formalism introduces \(R_u\), but the paper does not clearly spell out how this reward is computed operationally during selection beyond the downstream success metric discussion. Since LIAR depends on ranking sampled suffixes by reward, this is an important missing methodological clarification.
- **Some formal definitions/equations are imprecise.**  
  For example, Eq. (1) omits the target token inside the log-probability term, and the notation around \(\Delta_{\text{sub-gap}}\) vs. \(\tilde{\Delta}_{\text{sub-gap}}\) is inconsistent. These are not fatal on their own, but they contribute to the broader impression that the theory section was not executed with sufficient care.

### Trivial
- **Some claims are overstated relative to the evidence.**  
  In particular, claims that low perplexity “challenges the effectiveness of perplexity-based defenses” or that the work reveals broad “inherent vulnerabilities in current alignment strategies” go beyond what the presented experiments rigorously show.

## Nice-to-Haves
- Report comparisons under a matched protocol: same response length, same ASR@\(\!k\), and preferably equal total time budget.
- Add a stronger success evaluator (e.g., judge-based or manual validation) to separate genuine harmful compliance from prompt-drift and keyword-matching false positives.
- Show concrete adversarial suffix examples and failure cases, especially on weak-performing targets like LLaMA-2-7b.
- Clarify exactly how \(R_u\) is computed during best-of-\(N\) selection and what computational cost it incurs.
- Discuss the query-budget tradeoff more directly from both attacker and defender perspectives, since LIAR’s strength is speed but its mechanism is also query-heavy.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“The paper should evaluate more diverse application domains / high-stakes domains.”**  
  Removed as scope creep. The paper is a jailbreak method paper evaluated on a standard harmful-instruction benchmark; lack of healthcare/finance-specific tests is not a core flaw.
- **“The paper should include broader attack-generator diversity beyond the chosen pipeline.”**  
  Weakened/removed as a core criticism. While broader comparisons would help, the current method’s main issue is not lack of extra datasets or generators but the fairness and validity of the existing evaluation.
- **“The paper should evaluate long-term durability / extended fine-tuning behavior.”**  
  Removed as outside the paper’s scope; this is not a method about post-training robustness over epochs.

## Novel Insights
The most important synthesis is that this paper’s real contribution is not a new optimization-based “alignment attack,” but the demonstration that **fast, repeated, black-box sampling from a small unsafeguarded LM can be operationally competitive while producing highly natural-looking prompts**. That is a useful practical security insight. However, the paper currently wraps that insight in a stronger conceptual and theoretical narrative than the implementation and evaluation can support. In other words, the spark is real, but it is narrower and more empirical than the paper claims.

## Suggestions
- Reframe the contribution more honestly around **training-free best-of-\(N\) black-box jailbreak generation**, with the alignment formulation presented as motivation rather than the core realized algorithm.
- Redo the main comparison under a strictly matched protocol: same target response length, same ASR@\(\!k\), and ideally same total wall-clock budget.
- Strengthen attack validation with a more reliable success criterion than keyword-based matching, especially given the paper’s own acknowledgment of prompt-drift.
- Tighten the theory so that the random variables and distributions consistently refer either to suffixes or outputs, and explicitly connect the theory to the actual best-of-\(N\) implementation.
- Add qualitative examples of generated suffixes and failure cases to substantiate the claims of readability and to show whether successes genuinely answer the original harmful query.

# Actual Human Scores
Individual reviewer scores: [5.0, 3.0, 6.0, 5.0, 6.0]
Average score: 5.0
Binary outcome: Reject
