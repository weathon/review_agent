## Summary

The paper proposes a “First Generate, Then Evaluate” framework to assess what the authors call the “self-knowledge” of LLMs and LMMs: models first generate content under a prompt and are later asked questions about that same content. They instantiate this across a broad suite of tasks (word counting, designated word counts, facts, math, code, basic linguistic manipulations, and simple multimodal perception) and report that current models often fail to give answers consistent with the original prompts, with some exploratory analysis using attention patterns and small fine-tuning experiments on self-generated math data.

## Strengths

- **Simple, broadly applicable evaluation recipe.**  
  The two-phase structure—(1) generate, (2) later query about the generated artifact—is cleanly formalized in Eqs. (1)–(3) and easy to apply to many tasks. The paper demonstrates this for counting, facts, math, inequalities, code, basic grammar, and LMM perception. This could inspire better-designed self-consistency and self-evaluation benchmarks, even if the current instantiation is limited.

- **Wide empirical coverage and some creative probes.**  
  The authors evaluate 7 LLMs and 2 LMMs across multiple, heterogeneous tasks (Tables 1–4). The “SQL-like” text operations (add/delete/change word index) and the preposition-counting with sentence reordering (§4.4.1–4.4.2, Fig. 2, Table 3) are particularly interesting as probes of fine-grained text inspection and robustness to simple structural edits. These components could be useful in more principled future benchmarks.

- **Ambitious inclusion of multimodal self-evaluation.**  
  Extending the generate-then-query paradigm to LMMs (generate images then ask about counts, colors, positions; §5, Table 4) is a natural and underexplored direction. Even with only two LMMs and simple tasks, this helps connect similar issues across language and vision-language models.

- **Exploratory link to training utility.**  
  Section 6.3 shows that self-generated math question–answer pairs can, in some cases, yield small GSM-8K performance gains after fine-tuning (Fig. 3, Table 7). While the effects are modest and under-analyzed, the experiments at least suggest that the evaluation pipeline can double as a data-generation procedure worth investigating further.

## Weaknesses

### Fatal

None of the issues rise to “this is not even a paper,” but there is one structural problem that severely undercuts the central claims.

### Major

- **1. The operational definition of “self-knowledge” collapses into generic self-consistency and conflates multiple abilities.**  

  Throughout §§3–4, the self-knowledge score is defined as `I(a = \hat{a})`, where `a` is the answer implied or specified during generation and `\hat{a}` is the answer produced in a later verification call. Across tasks, this unavoidably mixes several distinct components:

  - **Constraint-following / controllable generation** (did the model actually generate what the prompt specified—e.g., exactly 56 words or a program whose true output is 10?),
  - **Content understanding / discrimination** (can the model answer questions about a given artifact, e.g., count words or reason about code?),
  - **Stochastic variation and prompt sensitivity** across calls.

  Yet the paper systematically interprets any mismatch between `a` and `\hat{a}` as failure of “self-knowledge,” e.g.:

  > “The inconsistency of the answers \(a = 56\) and \(\hat{a} = 63\) gives rise to a case of not comprehending the self-knowledge.” (§3)

  and more broadly:

  > “We find that modern LLMs and LMMs have unsatisfactory behaviors on self-knowledge evaluations, which is far from perfect.” (§1)

  However, the framework never checks whether the generated artifacts actually satisfy the constraints that define `a`. For example:

  - §4.2.1: models are asked to generate a paragraph with exactly N words, then later asked “How many words are there?” Only the prompt-specified N is used as “ground truth”; there is **no actual count** of the generated text reported.
  - §4.2.2, §4.2.7: same issue for designated word frequencies and program outputs; the text describes checking consistency with the specified target (“we ask the model to generate a program that has its execution result given, e.g. 10… then ask the model the executed result”), but there is no external verification of what the program actually computes.
  - In the fact and arXiv tasks (§4.2.3–4.2.4), the “correctness” notion is again just consistency between two model outputs; no external factual ground truth is checked.

  Consequently:

  - A model can generate **non-compliant** text (e.g., not exactly 56 words) and then **correctly** report the true count. This is good self-assessment of its own output but scores as “self-knowledge failure” because `a` (the requested count) ≠ `\hat{a}` (the actual count).
  - A model can be **consistently wrong** in both generation and verification (e.g., hallucinating a celebrity’s birth date or arXiv ID twice) and be scored as having perfect “self-knowledge.”

  Because the metric is defined on (prompt-intended answer, second-call answer), not on (true property of generated artifact, second-call answer), the quantitative results in Tables 1–3 and 6 are not interpretable as measuring “understanding of own creations.” They measure a hybrid of constraint-following and self-consistency under changed prompts. This is a fundamental mismatch between what the framework claims to capture and what it actually measures.

- **2. The reuse and dual-generating protocols further entrench consistency-as-self-knowledge without correctness.**  

  The consistency-based reuse (§3, Eq. (3); §4.4) and dual-generating strategy (§4.3, Eqs. (4)–(5)) are introduced as ways to assess self-knowledge without access to the original answer `a`. But they define self-knowledge purely as **invariance of the model’s answer over correlated prompts**:

  - Eq. (3): `I(LLM(prompt,x) = LLM(prompt, τ(x)))`, where τ is a transformation that is *assumed* to leave the answer unchanged (e.g., moving the first sentence to the end while counting prepositions).
  - Dual-generating: generate `x'` intended to preserve some property of `x`, then ask the same verifying question and compare answers.

  In both cases:

  - There is no check that τ actually preserves the property for the specific instance (e.g., whether prepositions were preserved in practice).
  - A model that is **perfectly but consistently wrong** before and after τ scores 1.0 “self-knowledge” by definition.

  Moreover, nothing in these protocols is specific to **self-generated** content. The same procedures would work identically on human-written paragraphs or arbitrary benchmark text, so the “self-” aspect is mostly nominal. Conceptually, these are generic robustness/self-consistency tests under simple input transformations, not evaluations of a model’s introspective understanding of its own outputs.

- **3. Interpretive claims about “alignment with human attention mechanisms” are speculative and overstated.**  

  Section 6.1 attempts to explain poor performance on the designated keyword counting task by comparing a custom “attention-based score” to the initial self-knowledge score (Table 5), and then posits:

  > “This may imply that models that perform better at the initial self-knowledge task may behave more similarly to humans… There may be an *additive effect*… misalignment with this attention-based mechanism and the less-concentrates of LLM attention than humans.”

  This interpretation is not adequately supported:

  - The “attention-based score” is an **ad hoc heuristic**: last-layer attention weights to the keyword are averaged across heads, the top 15% tokens are selected, and a ratio `min{k,s}/max{k,s}` is computed. The choice of last layer and the 15% threshold is not motivated or varied.
  - There is **no human data**: no measurements of human attention or performance under this task, so any claim of similarity to “human-inspired attention-based mechanisms” is conjecture.
  - Differences in Table 5 are small (e.g., differences of 0.04–0.21 between the two scores) and no variance or significance analysis is provided. With only 100 samples mentioned for task generation and only 5 models in the table, any pattern could be noise.
  - The “additive effect” story (phase 1: misalignment with mechanism; phase 2: lack of “concentrates”) is narrative; no experiment varies such factors independently.

  As written, §6.1 overinterprets a very rough attention proxy and presents it as a substantive mechanistic explanation involving “human attention mechanisms.” That is not justified by the evidence in the paper.

- **4. Fine-tuning claims are based on very small, uncharacterized gains and lack proper controls.**  

  Section 6.3 and the abstract claim:

  > “fine-tuning on self-generated math task may enhance the model’s math performance”

  The reported GSM-8K changes (Fig. 3, Table 7) are modest:

  - Llama3: +3.08 or +1.86 points,
  - Gemma: +0.11 or +0.19,
  - Llama2: +0.80 or +1.21,
  - GPT-3.5: +0.04 or −0.06.

  However:

  - There are **no error bars, standard deviations, or repeated runs**. With single runs on a large benchmark, it is entirely plausible that +0.1–+1.2 points fall within ordinary run-to-run variation from random initialization of adapters, sampling noise in the synthetic data, or prompt/evaluation variance.
  - There are **no baselines** such as:
    - tuning on comparable-size math data from another synthetic source,
    - tuning on the same questions with shuffled or corrupted answers,
    - or simply more general instruction-tuning of similar size.
  - The paper infers qualitative patterns (e.g., “models with higher initial accuracy will have higher accuracy when tuned on the correct answer and vice versa when the accuracy is low” and “self-improving is a promising direction”) from these small differences without statistical support.

  Given this, the fine-tuning results are best described as preliminary and suggestive, not as substantive evidence that the proposed self-knowledge framework yields especially beneficial training data. The current wording overstates what can be concluded.

- **5. Several task interpretations assume external correctness that is not actually tested.**  

  For tasks like “Facts” and “ArXiv” (§4.2.3–4.2.4):

  - The text frames them as testing factual precision and retrieval of true dates and IDs:
    > “Testing models on their ability to accurately recall important dates related to historical figures is crucial because it assesses their precision in handling factual information.” (§4.2.3)  
    > “Testing large models on their ability to accurately retrieve arXiv IDs is important because it assesses their precision and efficiency in handling specific, detailed queries within academic and scientific contexts.” (§4.2.4)

  - But the actual metric is again **internal consistency**: generate an entity for a date/month, then later ask if that entity was born then or what its ID is, and check if the later answer matches the earlier one. There is no mention of checking these against real calendars or the arXiv.

  Thus, a model that confidently hallucinates impossible dates or random identifiers but repeats them consistently will get a high “self-knowledge” score, yet the text describes this as “good consistency” on factual tasks. The interpretation (“assesses factual precision”) does not match what is actually measured (self-consistent hallucination vs self-inconsistent hallucination).

### Minor

- **6. Scope and naming overreach.**  

  The paper repeatedly invokes Feynman’s “What I cannot create, I do not understand” and presents the framework as evaluating whether models “really understand the questions and solutions created by itself” (§1). However:

  - Many of the tasks are **purely structural** (word counts, keyword counts, preposition counts, word indexing) and do not involve semantic “understanding” in the usual sense.
  - The same generate-then-query paradigm could be applied to externally provided content; nothing in the method specifically hinges on being the original creator. The current experiments do not compare self-generated vs externally generated content, so it remains untested whether anything uniquely “self-” is being measured.

  The framing would be more accurate if toned down to “self-consistency under generate-then-query protocols,” without strong claims about “understanding” or Feynman’s principle.

- **7. Limited statistical depth and analysis across tasks.**  

  Each task uses 100 samples; tables report point estimates only. While this is not inherently disqualifying for an exploratory benchmark paper, it does limit the ability to draw fine-grained conclusions about differences between models (e.g., 0.24 vs 0.30) or across protocols (e.g., changes of 0.05–0.1 in Table 6). No error analysis is provided beyond anecdotal examples (Fig. 1, Fig. 2), especially for the more complex tasks (math, theorem proving, code). This weakens the evidential weight of many claims and leaves open whether some differences are just noise.

- **8. LMM experiments are too small-scale to support general claims.**  

  Section 5 evaluates only two LMMs on three very simple perception tasks (counting, color, position) with no human-verified labeling of the generated images. The paper notes, e.g., that SEED-LLaMa “demonstrates satisfactory performance in color generation and perception with a high score of 0.81,” but:

  - We do not know whether the generated images actually match the textual prompts in all cases.
  - There is no breakdown of where failures arise: misgeneration vs misperception vs language understanding.

  These results are fine as illustrative anecdotes but do not yet constitute a robust multimodal benchmark or strong evidence for specific claims about LMM self-knowledge.

- **9. Some interpretive leaps in §6.2 (noise and “stochastic resonance”).**  

  In Table 6, some models improve when noise is added between generation and evaluation (e.g., GPT-3.5: 0.90→0.96; Qwen: 0.70→0.89). The paper attributes this to “stochastic resonance” (§6.2) without experimental support or ablations varying noise content/length. More mundane explanations (prompt reshaping, attention shifts, context window truncation behavior) are not explored. This does not invalidate the results, but the proposed cognitive analogy is speculative and should be framed as such.

### Trivial

- Some task descriptions are verbose and narrative (e.g., long didactic paragraphs in §4.2.3–4.2.7) relative to the simple underlying procedures; this slightly obscures the exact evaluation rules and could be tightened.

## Nice-to-Haves

- **Add explicit ground-truth checks for key tasks.**  
  For a subset of tasks (e.g., total word counting, designated word counts, code outputs), automatically compute the true property of the generated artifact and separately report:
  - generation compliance rate (did the artifact satisfy the prompt’s constraint?), and
  - verification accuracy (did the model’s answer match the true property?).
  This would disentangle generation vs verification failures and support a more precise notion of “self-knowledge.”

- **Compare self-generated vs externally generated inputs.**  
  To justify the “self” in self-knowledge, run the same verification prompts on (a) self-generated paragraphs and (b) human-written or benchmark paragraphs of the same form, and see whether models exhibit systematically different behavior. That would clarify whether there is anything uniquely self-referential here.

- **Provide more targeted qualitative error analysis.**  
  For at least one structural task (e.g., total word count) and one semantic task (e.g., math proofs), show a small but diverse set of failure cases, indicating whether the generation, the verification, or both are wrong, and how.

## Removed Points

These points are flagged to be removed, treat them with caution. I include them for completeness but they should *not* be treated as valid criticisms of the paper as written.

- **Questioning availability or existence of cited models/benchmarks.**  
  Any concern of the form “this model/dataset is not yet released or cannot be independently verified” would be removed under the instructions, but none of the provided reviews raised such points explicitly.

- **Claims that the paper never evaluates correctness against any ground truth whatsoever.**  
  While the main criticism—that many metrics use internal consistency rather than external truth—is valid, the paper *does* acknowledge where some tasks are chosen because they are easily computer-verifiable (e.g., inequalities in §4.2.6) and that some math answers are manually corrected (§6.3: “another is the correct one that has its answer human-corrected”). Any blanket statement that “the paper never checks any outputs against ground truth” would overstate the issue and is therefore set aside.

- **Overly broad dismissals that the results are ‘entirely unsurprising and thus useless’.**  
  The human calibration reviews note that similar work can be judged unsurprising; here, however, the paper does bring some creative task designs and a broad empirical sweep. Therefore, a claim that there is “no originality at all” would be excessive and is not retained as-is.

## Novel Insights

The genuinely novel aspect here is less the conceptual framing of “self-knowledge”—which, as instantiated, largely reduces to self-consistency under generate-then-query pipelines—and more the collection of concrete, low-level probe tasks that expose surprisingly brittle behavior in modern LLMs, especially around basic structural properties of their own outputs. The SQL-like text operations, preposition-count invariance under sentence reordering, and dual-generating protocols illustrate that even when models produce coherent paragraphs or code, their ability to answer simple, mechanically checkable questions about those artifacts is far from robust. However, because the framework does not carefully separate generation from verification or correctness from consistency, these observations are more indicative of general limitations in controllable text generation and local reasoning than of a well-isolated “self-knowledge” construct. None beyond the paper’s own contributions.

## Suggestions

- **Clarify and narrow the central construct.**  
  Reframe from “self-knowledge” plus Feynman’s quote toward “self-consistency under generate-then-query evaluation.” Explicitly distinguish:
  - generation-constraint adherence,
  - inspection/verification of a given artifact, and
  - consistency across runs/prompts.

- **Redesign key metrics to incorporate ground truth.**  
  For the core, easily automatable tasks (total word count, designated word frequencies, code outputs, inequalities), compute and report:
  - true property of the generated artifact,
  - whether it matches the requested constraint,
  - whether the model’s later answer matches the true property.

  Then define separate metrics for:
  - **Instruction-following** (constraint satisfaction),
  - **Self-knowledge / introspection** (accuracy of answers about its own outputs, conditional on the artifact),
  - **Self-consistency** (agreement between prompt-defined target and later answer, conditional on correctness).

- **Tone down speculative cognitive analogies.**  
  In §6.1–6.2, recast claims about human attention, “additive effect,” and “stochastic resonance” as hypotheses, not explanations. Either support them with additional experiments (e.g., varying attention heads, human baselines, noise types/lengths) or significantly soften the language.

- **Strengthen the fine-tuning study.**  
  If data and compute permit:
  - run multiple fine-tuning seeds and report mean ± std on GSM-8K,
  - compare to baselines such as tuning on existing math data of similar size,
  - analyze overlaps between generated QA pairs and GSM-8K to rule out trivial memorization.

- **Improve task taxonomy and exposition.**  
  Group tasks under clearer categories (e.g., structural counting, factual recall, arithmetic reasoning, code semantics, linguistic robustness, multimodal perception) and state, for each, *exactly* what is measured (consistency vs correctness vs both). Trim redundant narrative text and move more prompt templates to the appendix.

- **For LMMs, add minimal sanity checks.**  
  For a subset of image prompts, manually or automatically confirm that the generated images satisfy the textual constraints before querying the model. Alternatively, fix a set of pre-made images with known properties and use them for the verification phase to isolate perception from generation.

On standard axes:

- **Originality:** Moderate. The high-level idea overlaps with prior work on generator–validator consistency and self-consistency, but some task instantiations and multimodal extensions are novel.
- **Importance of question:** Moderate to high; evaluating models on their own outputs is timely and relevant, though the current framing overstates the conceptual leap.
- **Support for claims:** Weak for the strongest interpretive claims (human attention, special value of self-generated fine-tuning data); moderate for the basic empirical observation that models behave inconsistently on these tasks.
- **Soundness of experiments:** Mixed. Implementation seems competent but core metrics are conceptually misaligned with intended meaning; lack of ground-truth checks is a major limitation.
- **Clarity:** Generally readable; formalism in §3 is clear, but conceptual motivations and distinctions are underdeveloped and some discussions are over-interpretive.
- **Value to community:** As-is, more as a source of ideas and task designs than as a ready-to-use benchmark; substantial conceptual and methodological tightening would be needed for publication in a strong venue.

## Score and Decision

For calibration, I compared against:

- **Generator-Validator Consistency paper (phBS6YpTzC.md, scores 6/8/6, Accept poster).**  
  That work clearly defines a specific consistency notion (generation vs validation on the *same content*), measures correctness against task labels, and shows substantial, well-analyzed gains from consistency fine-tuning. The present paper is weaker methodologically: it does not separate correctness from consistency or generation from verification, and its interpretive claims are significantly more speculative. It is therefore **below** this paper in soundness and impact.

- **Self-supervised evaluation framework (zH6zBoktYO.md, scores 5/5/5/3, Withdrawn/Reject).**  
  That submission proposed self-supervised metrics based on invariances/sensitivities, had some originality but was criticized for weak evidence and over-reliance on correlations with a small set of benchmarks. The current paper is similar in spirit (self-based evaluation, transformation-based consistency) and has comparable issues with overinterpretation and shaky grounding. I would place this work in a similar quality band, perhaps slightly weaker due to the central misalignment between metric and claimed construct.

- **Self-Known / Self-Unknown factuality paper (qLxkXgmWwx.md, scores 3/3/5/5, Reject).**  
  That paper also introduced self-assessment metrics but was criticized for obvious findings and lack of deep analysis. This submission is comparable in that it tackles an interesting question but falls short in conceptual clarity and evidential strength. Given the fundamental metric conflation discussed above, I would place this paper in the **lower half** of that range.

Balancing these, I judge that while the topic is interesting and there are some creative task designs, the core methodological issues are substantial and would require a non-trivial redesign of the evaluation framework. This is not a matter of minor fixes; the current self-knowledge scores do not measure what the paper claims. I therefore lean to rejection, with an intermediate-low score rather than the very bottom because the idea and some probes are potentially useful starting points.

**Final score:** 4.5  
**Final decision:** Reject

MY FINAL SCORE: <pineapple>4.5</pineapple>  
MY FINAL DECISION: <orange>Reject</orange>