Now I have enough information to write my final review. Let me synthesize everything.

## Summary

The paper introduces a dual-system framework that pairs general-purpose LLMs (handling System-1 language tasks) with an external "Thinker" module (handling System-2 strategic reasoning), demonstrated in the 9-player Werewolf game. The architecture decomposes into Listener (NLU), Thinker (RL+BC for action and speech instruction generation), and Presenter (NLG). The authors contribute the FanLang-9 dataset (~18,800 human game sessions), the largest for social deduction games. Experiments show the Thinker module improves GPT-3.5 and GPT-4 on deductive reasoning accuracy, human-preferred speech quality, and online win rates; a fine-tuned WereLLM-T (6B) with Thinker matches GPT-4.

## Strengths

- **Large-scale dataset contribution**: The FanLang-9 dataset (18,800 sessions, ~7,000 hours of gameplay, ~6,000 hours of audio, 1.4M-character domain corpus) is, as claimed, the largest social deduction game dataset available and represents a genuine community resource (Section 3.1).

- **Pragmatic and interpretable architecture**: The Listener/Thinker/Presenter decomposition is a well-motivated engineering design. The language feature matrix **F** ∈ Z^{N×M} (Eq. 1) provides an interpretable communication protocol, and the Thinker's ability to be "optimized independently of the Listener and Presenter" (Section 3.3) is a genuine architectural advantage that avoids RL-through-LLM costs. This distinguishes it from Cicero's approach that requires fine-tuning the LLM itself.

- **Consistent Thinker improvement across online evaluation**: Table 1 shows win rate improvements for both GPT-3.5 (36.7% → 47.4%) and GPT-4 (37.9% → 41.1%) when the Thinker is added, across three different model combinations totaling ~1,800 games. The improvements for GPT-3.5 are substantial and consistent.

- **Honest reporting of werewolf identity exposure**: The paper candidly reports (Section 4.3, Table 2 discussion) that AI werewolves struggle to conceal identity from human players, and that illegal speech ratios are high for werewolf roles (Figure 4). This negative finding adds credibility.

- **Multi-dimensional evaluation**: Rather than only reporting AI-vs-AI win rates, the paper evaluates deductive reasoning (Figure 3), human-rated speech quality (Figure 4), and online gameplay (Tables 1-2), providing a more complete picture than typical game AI papers.

## Weaknesses

### Fatal
None.

### Major

- **The deductive reasoning evaluation does not isolate the Thinker architecture's contribution from training data effects (Section 4.1, Figure 3)**: The Thinker includes an identity model trained with supervised labels from the game environment (ℒ_id, Eq. 4) and behavioral cloning on human data (ℒ_BC, Eq. 2). Comparing this domain-specific trained model against GPT-3.5/4 in zero-shot or LtM prompting shows that supervised training on domain data outperforms prompting general-purpose models on a domain-specific classification task—a result that is expected regardless of architecture. The paper claims the Thinker "significantly improve[s] deductive reasoning" (Abstract), but the key missing comparison is: what happens when GPT-4 is fine-tuned on the same labeled data for these same tasks (role prediction, voting)? Without this, the claimed reasoning enhancement cannot be distinguished from the value of the training data alone. The online evaluation partially addresses this (since GPT-4-T adds the Thinker to GPT-4 without retraining GPT-4), but the reasoning evaluation specifically should be qualified.

- **The "surpasses GPT-4" and "on par with GPT-4" claims are overclaimed relative to the evidence (Abstract, Conclusion)**: WereLLM-T is a 6B model fine-tuned on 260k domain speech instances plus the Thinker module trained on domain data, compared against unmodified GPT-4 with prompting. The paper states WereLLM-T "surpasses GPT-4" (Conclusion) and "achieves performance on par with GPT-4" (Abstract). In Table 1, the comparison is between WereLLM-T and GPT-4-T (not GPT-4 alone), and the win rates are within noise of each other (50.3% vs. 41.1% in C1, 43.1% vs. 41.1% in C2, 45.9% vs. 46.3% in C3). Moreover, WereLLM also replaces the Listener and Presenter, so the improvement conflates LLM fine-tuning with Thinker augmentation. These claims should be more carefully qualified to reflect that domain-specific fine-tuning is a major factor.

### Minor

- **The sim-to-real communication gap is unanalyzed (Section 3.3)**: The Thinker is trained assuming generated language features perfectly match speech instructions (Section 3.3: "we assume that the Presenter generates speech accurately based on the speech instructions, and the Listener processes this speech and generate a language feature that precisely matches the original speech instruction"). The Presenter filtering mechanism (Section 3.4) confirms this assumption is imperfect in practice. The paper does not provide statistics on Presenter instruction-following rates, Listener feature extraction accuracy, or Thinker robustness to cascading errors. This is a valid methodological concern, though the online evaluation demonstrates the system functions end-to-end, suggesting the gap is manageable.

- **Online evaluation team composition effects (Table 1)**: The three model "combinations" mix different models in 9-player games, where a model's win rate depends on teammates and opponents. Combination 3 yields notably higher Werewolf win rates across all models (57-79%), suggesting composition effects. The paper acknowledges this implicitly ("the combination involving GPT-4 and WereLLM-T models yields the highest win rate for Werewolves") but does not control for composition. The 600 rounds per combination partially mitigates variance but don't isolate Thinker effect from team effects.

- **Lack of component ablations for the Thinker**: The Thinker uses BC + RL + identity model + population training + fictitious self-play with a 5:1 werewolf:goods training ratio. Which components matter? The appendices reportedly contain some ablations (Appendix B.1/B.2 are referenced), but visible content does not include a systematic ablation of Thinker components (e.g., BC alone vs. BC+RL, with vs. without identity model).

## Trivial

- None worth noting.

## Nice-to-Haves

- Comparison against a GPT-4 model fine-tuned on the same domain data for role prediction/voting, to isolate the Thinker architecture contribution from training data access.
- Analysis of Presenter instruction-following rates, Listener extraction accuracy, and Thinker robustness under perturbed inputs.
- End-to-end trace examples (Listener→features→Thinker→instructions→Presenter→speech) to illustrate the system's reasoning process.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Missing appendix ablations**: The harsh critic flagged that Appendices B.1, B.2, C, D, F containing ablations against Cicero-style and LLM-prompting approaches were "not available for review." Per the rules, appendices are stripped by the parser but exist in the original submission; the paper explicitly references ablation comparisons (Appendix B.1/B.2) in Section 2 and Section 4.3. This is not a valid criticism.

- **Inter-annotator agreement for speech evaluation**: The critic requested Kendall's W for the 10 evaluators ranking 2000 speeches. This is a reasonable methodological suggestion but is standard in its absence; the evaluation still provides meaningful signal given 10 evaluators and random ordering.

- **Training ratio ablation (5:1)**: The critic flagged the lack of ablation for the n_werewolf:n_goods = 5:1 ratio. While an ablation would be informative, this is a hyperparameter choice justified by the asymmetric difficulty of the werewolf role, not a core architectural claim.

- **Formatting/rendering artifacts**: Any formatting issues from the PDF extraction are parser artifacts, not paper problems.

## Novel Insights

The paper's most interesting empirical finding is the pattern in Figure 3 where GPT's reasoning accuracy declines over successive game days while the Thinker's improves—suggesting that LLMs become confused by accumulating speech context while the Thinker's structured feature representation scales better with game length. This is a genuine insight into when structured reasoning modules outperform raw LLM inference, and it points toward a more nuanced understanding than simply "Thinker is better."

## Suggestions

- Add a fine-tuned LLM baseline (e.g., GPT-4 fine-tuned on the same labeled data) for the reasoning evaluation to properly isolate architectural contribution from training data access.
- Report Presenter failure rates (how often filtering triggers, how many retries on average) and Listener accuracy metrics to quantify the sim-to-real gap.
- Tone down "surpasses GPT-4" / "on par with GPT-4" to acknowledge the role of domain-specific fine-tuning—e.g., "a domain-specific 6B model with Thinker achieves comparable performance to GPT-4 with prompting."
- Include systematic ablations of Thinker components (BC vs. BC+RL, with vs. without identity model, different population sizes) to clarify which design choices drive performance.

## Score and Decision

**Assessment dimensions:**
- **Originality**: Moderate. The dual-system architecture is motivated by cognitive theory but engineering-heavy rather than scientifically novel. The language feature/speech instruction protocol is a useful contribution.
- **Importance of research question**: High. Enhancing LLM reasoning for complex social deduction games with both language and strategy is an important challenge.
- **Claims well supported**: Partially. The online evaluation supports Thinker's practical benefit, but the reasoning evaluation does not cleanly isolate the architecture's contribution from training data.
- **Soundness of experiments**: The experiments are extensive (three evaluation dimensions, multiple model combinations, human evaluation) but the key reasoning comparison is confounded.
- **Clarity**: Good. The paper is well-structured and the architecture is clearly described.
- **Value to community**: High. The dataset and the practical framework design add value.

**Calibration comparison:**

| Anchor | Path | Avg Score | Comparison |
|--------|------|-----------|------------|
| Knowledge Card | WbWtOYIzIK | 8.0 | More principled method, stronger baselines, cleaner experiments. This paper is below it. |
| CivRealm | UBVNwD3hPN | 7.33 | Env/benchmark paper with clean evaluation. This paper has more complex claims that are harder to verify. Below. |
| Hypothetical Minds | otW0TJOUYF | 6.75 | LLM+RL multi-agent game paper, similar scope. This paper has a real dataset advantage but weaker experimental isolation. Similar tier. |
| GameArena | SeQ8l8xo1r | 6.5 | LLM reasoning evaluation with human gameplay. Comparable scope, slightly weaker methodology here. |
| Avalon/ReCon | fvhJu0FODp | 3.75 | LLM social deduction game, prompting-only approach. This paper has significantly more engineering substance and real data. Above. |
| FreeLM | qgLyKwXVDs | 2.0 | Unfair fine-tuned vs. zero-shot comparison. This paper has a similar confound but far more substance. Far above. |

The paper sits between the medium-scoring game agent papers (5-7) and the Avalon/ReCon paper (3.75). It has real engineering substance and a valuable dataset, but the overclaimed reasoning improvement and confounded evaluation pull it down. I place it at 5.5 — a borderline paper with genuine contributions but with a meaningful gap between its claims and its evidence.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>