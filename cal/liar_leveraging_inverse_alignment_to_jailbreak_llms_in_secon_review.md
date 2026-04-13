=== CALIBRATION EXAMPLE 10 ===

# Final Consolidated Review
## Summary
The paper proposes LIAR, a training-free jailbreak method that reframes adversarial suffix generation as an alignment problem using best-of-N sampling from a small adversarial LLM (GPT-2). The authors introduce a theoretical framework characterizing why safety-aligned models remain vulnerable and provide bounds on the "safety net" concept, while empirically demonstrating faster attack generation and lower perplexity prompts compared to training-based baselines.

## Strengths
- **Significant efficiency gains**: Eliminates training overhead entirely, enabling 100 adversarial queries in ~14 minutes vs. ~22 hours for AdvPrompter. The lack of training phase allows immediate attack generation, addressing a real practical limitation of existing methods.
- **Extremely low perplexity prompts**: Achieves perplexity ~2.14 compared to >90,000 for GCG and >80 for AutoDAN on Vicuna-7b (Table 1). These human-readable adversarial suffixes present a genuine challenge to perplexity-based defenses that filter unnatural text.
- **Novel theoretical framing**: Conceptualizing jailbreaking as "inverse alignment" provides a principled lens for understanding safety vulnerabilities, with formal bounds on the "safety net" (Theorem 1) and best-of-N suboptimality (Theorem 2).

## Weaknesses
- **Black-box claim contradicted by theoretical formulation**: The paper claims "fully black-box" operation (Abstract, Figure 1) yet defines the reward as $R_u(\mathbf{x},\mathbf{q}) = -J(\mathbf{x},\mathbf{q},\mathbf{y})$ requiring target model log-probabilities (Eq. 3). The practical implementation appears to use keyword matching on responses ("an attack is considered successful if at least one of the k attempts bypasses the TargetLLM's censorship mechanisms"), but this is never explicitly stated. This gap between theory and implementation undermines the theoretical narrative—best-of-N requires knowing rewards *before* selecting samples, while the empirical evaluation counts successes *after* evaluating all samples.

- **Poor performance on safety-aligned models**: ASR@100 on Llama-2-7b is only 3.85% (Table 1)—dramatically lower than Vicuna-7b's 97.12% or GCG's 23.7% at ASR@1. This failure on the most practically relevant safety target receives minimal discussion despite undermining claims about "inherent vulnerabilities in current alignment strategies."

- **Asymmetric time comparisons**: Table 1 explicitly states "TTA1 for our method is computed for ASR@100, whereas TTA1 for all other methods are computed for ASR@1." This makes time comparisons misleading—LIAR's favorable speed requires 100 queries while baselines are measured at single-query success.

- **Low single-query success rate**: ASR@1 of 12.55% on Vicuna-7b is substantially below AdvPrompter (26.92%), AutoDAN (92.70%), and GCG (99.10%). For rate-limited or cost-sensitive black-box scenarios, requiring 100 queries per attack attempt is a significant practical limitation.

- **Missing relevant baselines and ablations**: No comparison to other black-box methods like PAIR or TAP, which would be natural baselines. No random suffix ablation to establish whether GPT-2's language modeling provides attack signal beyond random perturbation—particularly important given that Falcon-7b and Pythia-12b show 100% ASR@1 for all methods (Table 1), suggesting trivial jailbreakability.

- **Theoretical contributions lack depth**: Theorem 1's bound ($\Delta_{\text{safety-net}} \leq \text{range}(R_u - R_s)$) is a trivial range bound that doesn't use specific properties of RLHF alignment or provide predictive insight. Theorem 2's suboptimality bound is a direct application of Amini et al. (2024) without sufficient novel adaptation to the jailbreaking context.

- **Model selection unexplained**: Table 2 shows GPT2-PMC achieves 19.68% ASR@1 vs. GPT-2's 12.55%, yet the paper uses GPT-2 for main results, justifying this choice only as "foundational." The performance gap warrants explanation or use of the better-performing model.

## Nice-to-Haves
- Evaluation against GPT-4 or Claude-3 to demonstrate effectiveness against state-of-the-art safety alignment
- Defense evasion experiments against perplexity filters and LLM-based detectors
- Compute-normalized comparison (total GPU-hours) rather than query-normalized metrics
- Failure mode analysis: why does ASR drop from 97% (Vicuna) to 4% (Llama-2)?

## Removed Points
*These points are flagged to be removed, treat them with caution*

- The claim that the title's "seconds" is misleading because competitive ASR requires 100 queries: while technically true that high ASR requires multiple queries, generating 100 queries in 45 seconds IS meaningfully fast compared to hours of training overhead. The efficiency claim is legitimate, though the asymmetric comparison in Table 1 should be addressed as a weakness.

- The claim that contribution (4) bundles "incompatible favorable numbers": the paper correctly states that 99% ASR requires k=100, consistent with the method description. This is not misrepresentation.

- The demand for standard deviations across multiple runs: while good practice, this is not a critical flaw for a method that achieves near-100% ASR with deterministic sampling parameters.

## Novel Insights
The inverse alignment framing provides a clean theoretical lens: if alignment machinery can steer models toward safety, the same machinery can steer them away. The perplexity results reveal an important practical tension—gradient-based methods find adversarial suffixes that are unnatural (perplexity >90,000), while sampling from pretrained models produces natural text (perplexity ~2) but weaker attacks. This suggests that the adversarial signal from pretrained language models is fundamentally limited, explaining why the method fails on strongly-aligned models like Llama-2. The "safety net" concept, while boundedly weak, points toward a measurable quantity: models with larger reward divergence between safety and harmlessness objectives should be more robust.

## Suggestions
- Resolve the reward computation ambiguity: explicitly state whether the practical implementation uses keyword matching, a classifier model, or target model log-probabilities, and revise the "black-box" claim if log-probabilities are required.
- Report ASR@1 with standard query budget (e.g., 1 query for all methods, or 100 queries for all methods) to enable fair efficiency comparisons.
- Add a random suffix baseline to establish whether GPT-2's language modeling provides meaningful adversarial signal.
- Discuss Llama-2 failures prominently—this limitation is critical for understanding method applicability.
- Clarify whether Theorem 2's best-of-N selection (picking highest-reward sample before evaluation) matches the empirical ASR@k protocol (counting successes across independent trials).

# Actual Human Scores
Individual reviewer scores: [5.0, 3.0, 6.0, 5.0, 6.0]
Average score: 5.0
Binary outcome: Reject
