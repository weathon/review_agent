Now I have sufficient calibration context. Let me write the final review.

## Summary
This paper systematically demonstrates that Retrieval-In-Context (RIC) RAG systems built on instruction-tuned language models are vulnerable to simple prompt injection attacks that can extract datastore content verbatim. The paper evaluates 9+ models across 7B-72B parameters, shows instruction tuning increases vulnerability by ~65 ROUGE points, demonstrates a 100% success rate attack on 25 production GPTs extracting 41% of a book in 100 queries, and proposes a position-bias-based mitigation that reduces reconstruction rate from 88.9% to 52.3%.

## Strengths
- **Comprehensive empirical evaluation across model families and scales**: Table 1 evaluates 9 instruction-tuned models (Llama2, Mistral/Mixtral, Vicuna, SOLAR, WizardLM, Qwen1.5, Platypus2) spanning 7B-72B parameters with consistent methodology. The results show instruction-tuned models universally vulnerable, with Llama2-Chat achieving ROUGE-L of 80.4 (7b) → 83.6 (13b) → 89.6 (70b), and Qwen1.5-72b reaching 99.15, providing a valuable reference for the community.
- **Production system attack with near-perfect success rate**: Section 4 demonstrates 100% attack success on 25 randomly selected GPTs from the GPT Store across security, law, finance, and medical domains, with 17/25 compromised on the first query. This extends beyond open-source experiments to show real-world impact, differentiating from prior work (Zeng et al., 2024) that did not test production systems.
- **Mechanistic ablation studies revealing root causes**: Figure 2 shows instruction tuning increases ROUGE-L by ~65.76 on average compared to base models, directly linking vulnerability to instruction-following capability. Figure 5 reveals a U-shaped position bias curve showing injections at context boundaries are most effective, and Table 2 shows previously seen data (Harry Potter) is more extractable (+5.7 to +12.4 ROUGE-L), disentangling memorization from context-copying.
- **Actionable RAG configuration insights**: Figures 3-4 demonstrate that semantic-aware chunking increases vulnerability and that fewer longer chunks are more extractable than many short chunks at fixed context size, providing concrete guidance for practitioners designing safer RAG pipelines.
- **Careful experimental design controlling for contamination**: Using Wikipedia articles created after November 2023 as the datastore reasonably controls for pre-training data contamination in open-source model experiments.

## Weaknesses

### Fatal
None

### Major
- **Scaling claim overgeneralized across model families**: The abstract claims "exploitability exacerbates as the model size scales up" as a general finding, but Table 1 shows this holds monotonically only within the Llama2 family. SOLAR-10.7b (≈13B) achieves ROUGE-L of only 46.1, substantially below Llama2-Chat-13b (83.6) and even below some 7B models. The scaling trend appears confounded by instruction-tuning intensity and architectural differences rather than pure parameter count. The claim should be qualified to specify it holds within model families or for maximum values per size tier, not as a universal law.
- **Mitigation effectiveness overstated relative to residual vulnerability**: Table 3 shows the best defense (Safety-Aware Prompt + PINE) reduces Reconstruction Rate from 88.88% to 52.34%, yet Section 3.2.3 states the combined strategy "effectively addresses" the vulnerability. An adversary reconstructing 52% of a private datastore has achieved meaningful data exfiltration. The mitigation is a partial improvement, not a solution, and the paper does not test it against adversarially-adapted attacks (e.g., injecting malicious instructions within document groups rather than at boundaries). This framing misleads readers about the defense's practical sufficiency.

### Minor
- **No benign RAG baseline for overlap metrics**: ROUGE-L, BLEU, and F1 are computed between attacked model output and retrieved context, but RAG systems naturally incorporate retrieved content into outputs. Without reporting these metrics for normal (non-attacked) RAG interactions, the incremental damage attributable specifically to the attack versus ordinary grounded generation is unclear. If benign RAG achieves ROUGE-L of 50-60 for accurate answers, the attack's delta is smaller than absolute values suggest.
- **Position bias mechanism untested beyond hypothesis**: Figure 5's U-shaped curve is an empirical observation, but the causal explanation attributing it to "lost in the middle" effects and RoPE recency bias remains speculative. The paper states "We hypothesize..." appropriately, but alternative explanations (e.g., syntactic proximity to instruction structure) are not ruled out, and no controlled tests validate the proposed mechanism.

### Trivial
- **GPT experiment sample details under-specified**: Section 4 states 25 GPTs were "randomly selected" spanning various domains, but no stratification criteria or selection methodology is provided. Reporting what differentiated the 17 GPTs compromised in 1 query versus 8 requiring 2 queries would strengthen reproducibility.

## Nice-to-Haves
- Analyze which portions of the 1.5M-word Wikipedia corpus were extracted in the 3.22% reconstruction to understand whether extraction correlates with topic frequency or retrieval patterns.
- Report confidence intervals or variance metrics for the GPT attack success rate given n=25 binary outcomes.
- Include a case study examining which injected prompts still succeed under the PINE defense to reveal patterns exploitable by more sophisticated adversaries.

## Removed Points
These points are flagged to be removed; treat them with caution:

- **Weakness**: "The 3% Wikipedia extraction rate presented without commentary as implicit success; extracting 3% of 1.5M words in 100 queries is low yield, calling this 'scalable' is aspirational." **Removal justification**: The paper reports both 41% (Harry Potter) and 3% (Wikipedia) transparently in the abstract and Figure 6, with the lower rate naturally demonstrating scalability limits on large heterogeneous corpora. This is honest reporting, not overclaiming. The title's "Scalable" refers to the attack methodology scaling with query count, not claiming full extraction of arbitrary corpora.

- **Weakness**: "Harry Potter experiment interpretation is circular—using Harry Potter to prove Llama2 memorized it when prior work already established this." **Removal justification**: The paper explicitly frames this as a "confound check" and states the results "lead to a hypothesis" about seen knowledge being more extractable. The circularity concern misreads the paper's intent: it uses a known-memorized corpus to validate the metric's sensitivity, then hypothesizes the broader principle. This is appropriate experimental design, not a flaw.

- **Weakness**: "No criteria or stratification given for 25 GPTs; unclear if representative or cherry-picked." **Removal justification**: Section 4 explicitly states the 25 GPTs span "various data-sensitive domains including cyber security, law, finance, and medical," which is reasonable domain stratification. The critic overlooked this detail.

- **Weakness**: "Adversarial Prompt 4 exploits myfiles_browser.search() specific to March 2024 GPT API; should note dependency explicitly." **Removal justification**: The paper states "as of March 2024" in the abstract and Section 4 introduction, appropriately scoping the temporal validity. This is not an omission.

## Novel Insights
The paper's most novel contribution beyond prior RAG security work is the mechanistic connection between position bias in context processing and extraction vulnerability, leading to the PINE defense. While Zeng et al. (2024) demonstrated prompt injection for privacy leakage, this paper uniquely identifies that (1) instruction tuning—not just model scale—drives vulnerability, (2) position bias creates a U-shaped attack surface exploitable at context boundaries, and (3) semantic coherence in chunking decisions inadvertently increases extractability. The production GPT attack exploiting function call interfaces (`myfiles_browser.search()`) is also a novel attack vector distinct from prior prompt extraction work.

## Suggestions
- Revise the scaling claim in the abstract and Section 3 to specify that exploitability increases with scale *within model families* or that *maximum observed vulnerability* increases with scale, acknowledging cross-architecture variation (e.g., SOLAR-10.7b anomaly).
- Reframe mitigation claims to emphasize "partial mitigation" or "significant reduction" rather than "effectively addresses," and discuss the residual 52% vulnerability as an open challenge requiring future work.
- Add a benign RAG baseline condition in Section 3 measuring ROUGE-L/BLEU/F1 for normal query-answer interactions to calibrate the attack's incremental effect.
- Discuss limitations of the GPT attack's temporal specificity more explicitly in Section 6, noting that API changes could affect reproducibility while emphasizing the principled vulnerability (RIC architecture + instruction-following) remains.

## Score and Decision
**Calibration reasoning**: Compared to retrieved anchors:
- vjel3nWP2a (scores 6,6,8,6,8,6; avg ~6.7, accepted poster): Similar empirical extraction attack on production models but narrower scope. This paper has broader model coverage (9+ vs fewer), more ablation studies, and clearer mechanistic analysis, warranting a higher score.
- fsW7wJGLBd (scores 8,5,8; avg ~7, accepted spotlight): Large-scale prompt injection dataset paper. This paper has comparable empirical rigor and stronger practical impact via production GPT attack, suggesting similar or slightly higher score.
- 6Mxhg9PtDE (scores 10,8,10,10; avg 9.5, accepted oral): Safety alignment paper with partial defenses. Shows partial mitigation can still be strong contribution if well-motivated—supports not penalizing this paper heavily for 52% residual vulnerability.
- H6i47PKXSN (scores 5,6,5,5; avg 5.25, rejected): Overclaimed scaling contributions led to rejection. This paper's overclaims are less severe (qualified within sections, not fundamentally flawed methodology).

This paper's core empirical contributions (production attack success, systematic model evaluation, mechanistic ablation) are solid and practically significant. The major weaknesses (overgeneralized scaling claim, overstated mitigation) are fixable framing issues rather than fundamental flaws. Positioning between fsW7wJGLBd (7) and recognizing stronger empirical scope: **7.5**.

MY FINAL SCORE: <pineapple>7.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>