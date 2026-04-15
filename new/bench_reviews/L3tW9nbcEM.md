## Summary
This paper introduces "Schrödinger's Memory" as a conceptual framing for memory in Large Language Models, arguing that LLM memory is only observable when queried. The theoretical grounding is drawn from the Universal Approximation Theorem (UAT): the authors claim that Transformer architectures implement a "dynamic UAT" whose input-conditioned parameter adjustment constitutes the mechanism of memory. Empirically, they fine-tune seven small LLMs (0.5B–1.5B) on 2,000 Chinese and English poems for 100 epochs and measure exact-match recitation accuracy. They additionally propose poem memorization accuracy as a new objective benchmark for general LLM capability, and conclude with speculative parallels between LLM and human brain memory.

---

## Claims and Support

**Claim 1: UAT / "dynamic fitting" explains the mechanism of LLM memory.**
*Unsupported.* The paper shows that Transformers can be written in a UAT-like form with input-dependent parameters (Eq. 3, citing UAT2LLMs), then asserts: *"This dynamic fitting capability is the ultimate source of the powerful memory observed in LLMs"* (Sec. 2.2). No mechanistic consequence is derived from this assertion; no experiment distinguishes it from simpler explanations (parametric overfitting, standard sequence modeling). The mathematical formulation is a representation of how Transformers work during inference — it says nothing uniquely predictive about memory.

**Claim 2: LLMs possess memory (demonstrated by poem recitation from cues).**
*Narrowly supported, broadly overstated.* The experiments show fine-tuned models can reproduce training poems from metadata prompts. But the paper does not mention a train/test split; the models are trained and evaluated on the same 2,000 poems. This establishes training-set overfitting, not a general theory of LLM memory.

**Claim 3: Memory ability assessment is a new, objective measure of overall LLM capability.**
*Unsupported.* The evidence is that better-pretrained or larger models score higher on poem memorization after 100-epoch fine-tuning — a well-known fact about model capacity. No external validation against established benchmarks is provided; no correlation analysis is reported; the metric is confounded by tokenizer, language domain, and training recipe differences.

**Claim 4: Output length negatively affects memory accuracy.**
*Partially supported.* Table 2 is consistent with this direction, but the experimental design does not isolate length as a causal factor — longer poems may differ in vocabulary, genre, rarity, and complexity. The analysis is limited to Chinese.

**Claim 5: Human brain memory and LLM memory share the same fundamental mechanism.**
*Unsupported.* Sec. 4 acknowledges its speculative nature (*"we make logical assumptions and generalizations about the brain's mechanisms"*), yet the conclusion states they are "fundamentally the same." No neuroscientific evidence, behavioral data, or computational analysis is presented to support this.

---

## Strengths

- **Operationally clean memorization task:** Prompting with author/title/dynasty metadata to reconstruct full poems provides a well-defined and replicable test of cue-triggered recall from parametric storage, which is a legitimate contribution at an operational level.
- **Cross-lingual and cross-model comparison:** Testing Qwen and BLOOM families across Chinese and English at the same capacity range provides a useful descriptive snapshot of how model pretraining quality affects fine-tuning-based memorization, including the interesting observation of a gap between Qwen1.5 and Qwen2 at identical parameter counts.

---

## Weaknesses

### Fatal

**The paper's central theoretical claim — that UAT explains the memory mechanism of LLMs — is an unsubstantiated assertion.** The paper simply states (Sec. 2.2) that because Transformer weights are input-conditioned, *"This dynamic fitting capability is the ultimate source of the powerful memory observed in LLMs."* This is not a mechanistic explanation but a relabeling: calling attention-mediated computation "dynamic UAT" does not predict any specific memory phenomenon. The experiments that follow do not test any consequence of the UAT framing, do not compare against non-Transformer or static architectures, and do not distinguish the proposed mechanism from ordinary parametric overfitting. The gap between the theory introduced in Sec. 2 and the experiments in Sec. 3 is complete — the theory generates no hypothesis that is tested.

**The experimental setup tests training-set overfitting, not memory.** The paper trains and evaluates on the same 2,000 poems with no held-out split. Under 100-epoch fine-tuning, achieving high exact-match on training items is the expected behavior of any sufficiently parameterized model. The paper presents this as evidence of a memory mechanism, but this is indistinguishable from memorization/overfitting. There are no controls with shuffled metadata (to rule out surface-level association), no evaluation on held-out poems, and no comparison to a simple lookup baseline. The core empirical claim is therefore uninterpretable.

### Major

**The "memory ability assessment" proposal is unsupported.** The claim that poem recitation accuracy after 100-epoch fine-tuning is *"an objective measure of LLMs' language abilities"* (Sec. 3.3) is not validated. The only evidence is that larger or better-pretrained models score higher — a trivially known fact. No correlation with recognized benchmarks is shown, no cross-domain robustness is established, and the metric is heavily confounded by tokenizer choice, domain familiarity, and fine-tuning setup. This is not a measurement instrument; it measures ease of overfitting a narrow mapping.

**The "Schrödinger's memory" concept provides no explanatory power.** The paper's framing is: *"we can only determine whether the LLMs have a particular memory when we ask a question and receive a response"* (Sec. 3.3). This is trivially true of any function — you can only determine f(x) by evaluating it at x. It applies equally to databases, hash maps, and any retrieval system. The paper does not specify what would *falsify* the Schrödinger framing or how it makes predictions different from simply saying "LLMs approximate functions seen during training."

**The human brain comparison (Sec. 4) is speculation presented as a finding.** The paper claims *"their fundamental mechanisms are the same: both rely on dynamically fitting outputs based on inputs"* and that *"the mathematical model of the human brain may resemble that of a Transformer-based dynamic approximation UAT model."* The Henry Molaison case is used to claim the hippocampus acts as *"a switch controlling whether the weights responsible for long-term memory in the brain can be updated"* — this is a drastic oversimplification with no neuroscientific citation to support the specific computational interpretation. The section's own preamble admits these are *"logical assumptions and generalizations"*, yet the conclusion presents them as findings.

### Minor

**The definition of memory in Sec. 3.1 conflates the question with the conclusion.** The paper rejects the "storage+retrieval" definition on the grounds that brains lack database-like storage, then defines memory as input-output behavior. This definitional move ensures that any cue-conditioned model behavior automatically qualifies as "memory," making the subsequent experiments nearly tautological rather than informative.

**The length-effect analysis (Sec. 3.4) lacks experimental controls.** Longer poems may differ systematically in vocabulary, genre, and rarity. Without matched subsets or within-dataset length-binned analysis, the observed accuracy drop cannot be attributed to length alone.

### Trivial

**Human recitation comparison is asserted without evidence.** The statement that *"an average person, without specific memory training, would struggle to remember 1,000 poems under similar conditions"* (Sec. 3.3) is presented as empirical context without citation or study.

---

## Nice-to-Haves

- Testing recall of pre-trained knowledge without fine-tuning (e.g., factual QA) would better address the paper's motivating question (the "Who is the President?" example).
- Reporting character-level or token-level partial accuracy alongside exact match would reveal whether failures are total or near-misses.
- Including a frozen-parameter baseline (frozen LLM + learned retrieval head) would help isolate what role the "dynamic UAT" mechanism actually plays.
- Plotting accuracy across training epochs would clarify whether memorization emerges gradually or abruptly.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"Missing related work on training-data memorization"** (Human Finder reviewer): Per the hard rules, missing related work is not cited since external sources cannot be confirmed. Removed.
- **Reproducibility concerns about hyperparameters and decoding settings** (Harsh Critic, Sec. 3.2): These are implementation-level nitpicks. Removed per hard rules on reproducibility.
- **Criticism that the Qwen/BLOOM models cited may not be available or independently verifiable**: Not raised explicitly but would be removed per hard rules if raised.
- **(Neutral reviewer, Strength) "Intuitive Metaphor" as a general strength**: The Schrödinger framing is noted as being useful for conceptualization — however, as analyzed above, it is trivially true of any queryable system. Removed as a genuine strength.
- **(Neutral reviewer, Strength) "Interdisciplinary Ambition"**: Bridging CS and neuroscience via analogy, without any neuroscientific evidence, does not constitute a scientific strength. Removed.

---

## Novel Insights

None beyond the paper's own contributions. The observation that fine-tuned LLMs can reproduce training examples from metadata cues is well-established. The UAT reframing does not generate new predictions. The cross-lingual and cross-model memorization comparison is useful descriptively but does not constitute a novel scientific insight.

---

## Suggestions

1. **Redesign the experiment around pre-trained knowledge retrieval** rather than fine-tuning: test whether models already "remember" factual content without supervised memorization epochs. This would directly address the paper's motivating examples and test the UAT-based theory.
2. **Add a train/test split and control conditions**: evaluate on held-out poems, test with shuffled metadata, and compare to a non-Transformer baseline of similar capacity to support any claims about mechanism.
3. **Sharpen or remove the UAT theory section**: derive at least one testable prediction from the UAT framing (e.g., a scaling law, a capacity threshold, an attention-ablation result) or reframe Sec. 2 as background motivation rather than a theoretical contribution.
4. **Recast Sec. 4 explicitly as a speculative discussion**, not a finding. Remove the claim that brain and LLM mechanisms are "fundamentally the same" unless empirical or formal evidence is provided.
5. **Validate the "memory ability assessment" claim externally**: report Spearman/Pearson correlation of poem memorization scores against established benchmarks (e.g., MMLU, ARC, HellaSwag) across a wider set of models before claiming this is an objective general capability measure.

---

## Score and Decision

**Originality:** Very low. The concept of LLM memorization is well-studied; the UAT connection to Transformers is borrowed from prior work; the Schrödinger framing is a metaphor without formal content.

**Importance of research question:** Moderate. Understanding parametric memory in LLMs is a genuinely important question.

**Support for claims:** Very weak. The central mechanistic claim is undemonstrated; the evaluation proposal is unvalidated; the brain comparison is speculative.

**Soundness of experiments:** Poor. No train/test split, no controls, single metric, narrow model zoo, no ablations.

**Clarity of writing:** Adequate but overclaiming throughout, especially in the abstract and conclusion.

**Value to the research community:** Low in current form. The paper repackages known phenomena (LLMs can overfit training data) with an unjustified theoretical framing.

The FUNDAMENTAL ISSUES rule is triggered: the paper's core claim — that UAT explains the memory mechanism of LLMs — is not demonstrated by the experiments or the theory. The experimental setup conflates overfitting with memory and provides no controls. The human brain comparison is speculation presented as a conclusion. These issues are structural, not fixable by minor revisions.

**Score: 2.0**

MY FINAL SCORE: <pineapple>2.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>