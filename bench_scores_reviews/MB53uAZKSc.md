## Summary
TiC-LM introduces a large-scale benchmark for continual pretraining of language models, centered on TiC-CommonCrawl (TiC-CC): 2.9T tokens from 114 monthly Common Crawl dumps spanning May 2013–July 2024 — more than 100× larger than prior benchmarks for this setting. Complementary evaluation suites (TiC-WIKI, TiC-STACKE, TiC-CODEDOCS) enable multi-domain temporal assessment. The authors evaluate optimization-, replay-, and regularization-based continual learning strategies, finding that data replay most effectively reduces forgetting on previously seen domains but creates domain-dependent trade-offs; and that scaled-up replay with auto-regressive learning rate annealing can match a series of oracle models while using 62% less compute.

---

## Strengths

- **Genuine benchmark scale leap.** TiC-CC's 114 monthly timesteps and 2.9T tokens dwarf all prior time-continual LLM benchmarks (Table 1), enabling for the first time the study of forgetting dynamics across a full decade of web evolution rather than 4–10 snapshots. This is not an incremental improvement but a qualitative change in the kind of questions that can be studied.

- **Domain-specific insight on replay directionality.** The paper demonstrates concretely that replay of early data (2013–2016) benefits slowly-evolving domains (TiC-STACK-MATH, TiC-CODEDOCS-NUMPY) while harming rapidly-evolving ones (TiC-STACKOVERFLOW, TiC-CODEDOCS-PYTORCH). The NumPy vs. PyTorch case — explained by PyTorch's 2016 release date meaning 3 years of replayed early CC data predates its existence — is a sharp, mechanistically grounded finding most benchmark papers never reach.

- **Efficiency result with concrete compute accounting.** Section 6.3 goes beyond accuracy tables and constructs a carefully token-matched oracle series (seven oracles, 1.16T tokens total), then shows that Replay+AR at 440B tokens surpasses this oracle on TiC-CC and TiC-WIKI subsets while requiring 62% less compute and delivering monthly updates instead of biennial ones. This is an actionable, practitioner-relevant result.

- **Regret-relative-to-oracle evaluation design.** Using $R_{i,j} = E_{i,j} - E_j^*$ rather than absolute perplexity controls for the inherent difficulty gradient across evaluation months. Combined with the full $T_t \times T_e$ evaluation matrix (visualized via heatmaps), this yields a nuanced decomposition of backward transfer, ID performance, and forward transfer that reveals otherwise hidden trade-offs.

- **Surprising EWC finding.** EWC is the best single method on TiC-WIKI-Diff, diverging from prior work (TiC-CLIP, Jin et al. 2022) where EWC had negligible impact. The paper traces this to the diff-set's emphasis on changed Wikipedia content where forward plasticity outweighs replay's backward advantages — a domain-structure insight that points toward principled method selection.

---

## Weaknesses

### Fatal
None.

### Major

- **All method comparisons conducted at 3B parameters only.** The paper's own introduction motivates the work by citing degradation of 7B–9B models (Gemma-7B, DCLM-7B-2x, Figure 2), yet every continual training experiment uses 3B models. Forgetting dynamics, optimal replay ratios, and the relative merit of regularization vs. replay may change substantially at larger scales due to differences in memorization capacity, effective learning rates, and Fisher information geometry. Without at least one 7B experiment, the method comparison findings cannot be confidently attributed to "LLMs" as the paper implies. This is the most significant empirical limitation.

- **50% of the total token budget front-loaded to May-2013, with no ablation.** This is one of the most consequential experimental design choices in the paper. The paper provides qualitative justification (Section 6: "practitioners are likely to have access to more than enough data"), but no sensitivity analysis is provided. The heavy initialization bias toward 2013-era data makes the continual learning task harder (the model must forget a strong prior), influences the oracle gap, and potentially inflates the measured benefit of replay. Whether conclusions change with, say, a 10% or 25% initialization allocation is unknown.

- **Statistical rigor limited to a single method.** Only Cyclic Cosine is run with three seeds; all others are single runs. The reported standard deviations for Cyclic Cosine are uniformly 0.000 or 0.001 in most columns (Table 2–3). The "bold = within one std of best" criterion is then vacuous for distinguishing other methods, since a std of ~0.000 makes any difference statistically significant. Given the small absolute differences between many method pairs (e.g., LwF and Cyclic Cosine are numerically identical in Table 2, Backward TiC-CC: both 0.072), lack of multiple seeds for other methods is a meaningful gap for a benchmark paper claiming to rank strategies.

### Minor

- **EWC implementation details insufficient.** EWC requires computing an approximate Fisher information matrix for a 3B-parameter model. The choice of diagonal vs. block-diagonal approximation, number of samples, and implementation can vary substantially and significantly alter results. Since EWC is presented as the best method on TiC-WIKI-Diff — a highlighted finding — the lack of implementation details weakens reproducibility of this specific claim.

- **Unexplained anomaly in efficiency analysis.** Table 4 shows TiC-StackExchange-Cat7 for Replay at 440B tokens degrades to 0.150 (from 0.028 at 220B), a 5× worsening. Similarly TiC-CD-PyTorch at 440B Replay+AR worsens from 0.082 to 0.013... wait, it shows 0.013 which is actually better. But the Cat7 regression at 440B is dramatic and unexplained. If the "average" summary metric in Section 6.3 is hiding such regressions, readers need to understand why scaling replay hurts certain domains — this speaks directly to the reliability of the efficiency claim.

- **Oracle baseline definition shifts between sections.** Sections 6.1/6.2 use a single Oracle (trained on all 114 months from scratch), while Section 6.3 shifts to a "series of Oracles" with different cutoff dates. The shift is well-justified logically, but the paper does not explicitly flag this change when Table 4 appears, making direct comparison of regret values between Table 2/3 and Table 4 non-obvious to readers.

- **Confound between monthly data volume and temporal recency.** Monthly token counts vary by ~5× (100B–500B, Figure 3), but the training budget allocates equal tokens per month during the continual phase. Months with smaller crawls are therefore over-sampled per unique token. This means "forgetting" metrics partly conflate temporal recency with training data density effects, an uncontrolled variable the paper does not discuss.

### Tiny

- The abstract says "up to 45%" without noting the range across models is 28–45%; including the range would give readers a more accurate picture of the magnitude.

- The naming of TiC-CC-WIKI (Wikipedia pages in CC) and TiC-WIKI (full Wikipedia dumps) is easily confused when both appear side-by-side in Tables 2–4; a brief reminder in the table headers or captions would help.

---

## Nice-to-Haves

- **Quantify storage overhead for replay.** Section 5 acknowledges retaining old data is a downside but provides no numbers. A simple table showing TB of data required per replay configuration (e.g., $\alpha_t = 1/t$ vs $\alpha_t = 1/2$) would complete the efficiency narrative in Section 6.3.

- **Ablate the cross-month deduplication choice.** The paper explicitly avoids global fuzzy deduplication and explains why (Section 3), but this design choice could affect replay benefit estimates. Even a qualitative analysis of inter-month document overlap rates would help bound the potential confound.

- **Per-domain performance degradation curves.** Plotting how quickly each domain (news, math, code) degrades without updates, as a function of months since training, would substantiate the "domains evolve at different rates" claim more quantitatively than aggregate metrics alone.

- **Forward transfer discussion.** All methods show large forward transfer regret (0.161–0.181 for TiC-CC, 0.155–0.162 for TiC-StackOverflow), meaning continual checkpoints provide near-zero benefit on future unseen months relative to the Oracle. This is an important negative finding for practitioners hoping continual training builds durable representations. A brief discussion of implications would add value.

- **Wall-clock time and GPU-hour accounting per method.** The paper notes EWC/LwF have larger memory footprints but does not quantify actual runtime differences. Since the efficiency analysis is a key contribution, concrete wall-clock numbers would strengthen it.

---

## Removed Points

*These points were flagged for removal; treat with caution.*

- **No instruction-tuned evaluation (Harsh Critic).** The paper explicitly scopes itself to base model pretraining (Section 4: "we focus on evaluations without instruction-tuning"). Criticizing the absence of RLHF/SFT interaction analysis is scope creep against a benchmark paper that clearly delimits this direction.

- **Replay buffer not token-count-proportional to monthly volume (Harsh Critic).** Equal redistribution of replay tokens across prior months is a deliberate design choice enabling controlled comparisons. Demanding proportional allocation is a methodological alternative, not a flaw, and arguing the paper should have done it instead without evidence it would change conclusions is speculative.

- **Societal impact of temporal CC data (Harsh Critic).** While a real concern in general, this is outside the paper's scope and not a reviewable weakness for a benchmark methodology contribution.

- **220B vs. 2.9T training scale mismatch framed as a misleading claim (Harsh Critic).** The paper is transparent about using a 220B subset: "We use smaller subset of 220B tokens from a single global shard with 2.9T for our training while future work can expand to the full 2.9T/2.9T tokens" (Section 3). The "100×" claim refers to the dataset scale relative to prior benchmarks, which is accurate and not misleading.

- **Missing recent LLM-specific CL baselines like LoRA or gradient projection (Spark Finder).** At 3B-parameter pretraining scale on generic web data, parameter-efficient methods like LoRA are not standard practice; gradient projection is computationally infeasible. Demanding these is importing expectations from a different experimental regime.

- **Request for theoretical proofs / mechanistic explanation of why replay outperforms regularization (Spark Finder).** This is an empirical systems paper and a benchmark contribution. Providing theoretical guarantees or mechanistic analysis of EWC failure modes is a future research direction, not a prerequisite for acceptance.

---

## Novel Insights

The most genuinely novel observation — one that emerges from the multi-year scale and domain diversity that prior benchmarks could not have surfaced — is the **directional dependence of replay on domain age structure**: replay of early data helps domains whose primary content predates 2016 (NumPy, StackMath) but actively harms domains whose content postdates 2016 (PyTorch, StackOverflow). This implies that the standard recommendation to "replay old data to prevent forgetting" requires a domain-level audit of when relevant content actually appeared in the training corpus. A related insight is the **temporal lag in TiC-WIKI performance**: models achieve peak accuracy on a given Wikipedia month only years later, suggesting that Wikipedia knowledge percolates through subsequent CC crawls gradually rather than being captured in the corresponding monthly dump — a finding with direct implications for how practitioners should think about knowledge cutoffs and temporal alignment in pretraining data.

---

## Suggestions

1. **Run at least one 7B continual training experiment**, even if only for the best-performing method (Replay+AR) versus Cyclic Cosine. A single data point at a larger scale would substantially strengthen the claim that findings generalize to modern LLMs.

2. **Add an ablation on the initialization budget split** (e.g., 10%, 25%, 50% of total budget on Month-0) to demonstrate robustness of conclusions to this critical design choice.

3. **Run multiple seeds for at least the top 3 methods** (Cyclic Cosine, Replay $\alpha_t=1/2$+AR, EWC) to make the bolding criterion in Tables 2–3 statistically meaningful.

4. **Explain the 440B Replay regression on TiC-StackExchange-Cat7** (0.028 → 0.150). If this reflects a genuine failure mode of scaling replay for fast-evolving domains, it should be highlighted as a warning for practitioners, not buried in a table.

5. **Explicitly flag the Oracle definition change** at the start of Section 6.3, and explain whether regret values in Table 4 are comparable in magnitude to those in Tables 2–3.

6. **Provide EWC implementation details** (Fisher approximation type, number of samples for Fisher estimation) in the main text or appendix, given that EWC is the highlighted winner on TiC-WIKI-Diff.

---

**Evaluation summary:**
- *Novelty*: High — the dataset and benchmark protocol represent a genuine step change in scale and temporal breadth for this problem.
- *Technical soundness*: Moderate — benchmark design is careful and evaluation methodology is well-reasoned, but the 50%-initialization design choice and absence of ablations introduce unresolved uncertainty about result generalizability.
- *Empirical support*: Moderate — comprehensive within the 3B scale, across many domains and methods, but the single-scale constraint is a real gap given the paper's stated motivation around modern LLMs.
- *Significance*: High — this is a resource and evaluation framework the community will use; the efficiency finding and domain-directionality of replay are actionable insights.
- *Clarity*: Good — the paper is well-organized and the heatmap visualizations are effective; minor oracle-definition confusion between sections.