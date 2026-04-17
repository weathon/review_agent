# AlignSentinel: Alignment-Aware Detection of Prompt Injection Attacks

- Decision: Reject
- Scores: 4, 6, 6, 2

## Abstract
Prompt injection attacks insert malicious instructions into an LLM's input to steer it toward an attacker-chosen task instead of the intended one. Existing detection defenses typically classify any input with instruction as malicious, leading to misclassification of benign inputs containing instructions that align with the intended task. In this work, we account for the instruction hierarchy and distinguish among three categories: inputs with misaligned instructions, inputs with aligned instructions, and non-instruction inputs. We introduce AlignSentinel, a three-class classifier that leverages features derived from the LLM's attention maps to categorize inputs accordingly. To support evaluation, we construct the first systematic benchmark containing inputs from all three categories. Experiments on both our benchmark and existing ones--where inputs with aligned instructions are largely absent--show that AlignSentinel accurately detects inputs with misaligned instructions  and substantially outperforms baselines.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes AlignSentinel, a detector that classifies inputs into misaligned, aligned, and non-instruction by using token-pair attention interactions between an input and its higher-priority instruction (system or user prompt). Two simple variants are presented: pooling all attention interactions before an MLP (Avg-first) or encoding interactions then pooling (Enc-first). A new synthetic benchmark spanning eight domains is introduced to include aligned cases, and experiments on three open-source LLMs report very low FPR/FNR on the authors’ data and transfer to IHEval’s rule-following and tool-use categories.

### Strengths
The paper states a clear motivation that binary “instruction present = malicious” detectors conflate aligned and misaligned instructions; the three-class framing directly addresses this evaluation blind spot.

Methodology is simple and clearly described, with explicit feature construction and small MLP heads; implementation details and training hyperparameters are given

The benchmark explicitly contains aligned inputs for both direct and indirect scenarios and covers several domains with structured generation recipes and examples

### Weaknesses
Limited novelty and incremental design: the core idea combines a straightforward three-class relabeling with simple averages or per-pair encodings of standard attention maps, followed by a small MLP (Section 4.2). There is no new modeling component beyond feature pooling/MLP or theoretical insight into why specific attention interactions should separate the classes beyond the qualitative Fig. 1.

Reported “near-perfect” gains are confined to a fully synthetic benchmark created by the authors using GPT-4o with templated prompts and explicit constraint generation (Section 5; Appendix A.3). This risks strong dataset-specific artifacts and separability; there is no statistical analysis, no confidence intervals, and many entries are 0.00 without variance or seed reporting.

Evaluation choices may overstate improvements: baselines are trained as binary detectors where aligned+non are pooled as negatives. A natural control where extending at least the trainable baselines to the same three-class setting used by AlignSentinel is not reported.

### Questions
See the weakness section.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces AlignSentinel, a white-box monitoring pipeline for prompt injection detection. Authors propose a three-class classification between: (1) misaligned instructions (e.g., prompt injections), (2) aligned instructions (e.g., clarifications), and (3) non-instruction inputs (e.g., data). The classifier leverages features derived from the LLM's attention maps to categorize inputs. In rigorous comparisons against five baselines, AlignSentinel demonstrates clear improvements and impressive accuracy across evaluated benchmarks.

### Strengths
- The explicit distinction between aligned and misaligned instructions is convincing and addresses an obvious gap in existing work on instruction-data separation. This contribution is timely and already relevant for current agentic systems that, for example, generate execution plans which should be treated as aligned instrumental goals rather than hijacking attempts (in comparison to the original goal).

- The authors clearly define what they mean by aligned instructions, misaligned instructions, and non-instruction inputs.

- I appreciate that the authors ensure fair comparison by using the same training procedure for all five baseline methods. The proposed monitoring system shows impressive accuracy and substantially outperforms baselines across multiple benchmarks.

### Weaknesses
- It would be valuable if the authors made an explicit connection to the instruction-data separation literature [1, 2, 3, 4] and integrated the notion of data into their model (which currently appears to be framed as "non-instructional inputs").

- In lines 37-45, the authors informally introduce aligned and misaligned instructions using an email agent example. They argue that some instructions from fetched emails should be considered aligned instructions. However, this seems like a clear instruction-data separation case where *none* of the instructions from emails should be treated as instructions. This example undermines the motivation rather than strengthening it.

- Based on the examples in the appendix, the proposed dataset appears overly simplistic. The hijacking behaviors are highly semantically and pragmatically different from original tasks (e.g., Let’s role-play a resale scam.). It would be beneficial to evaluate performance on semantically similar tasks (e.g., original instruction: "extract emails" vs. injection: "extract emails and [for some adequate reasons] send them to supervisor@email.com"). I am concerned that proposed benchmark is too naive and clearly looks like an evaluation, which might be a problem for models with a higher evaluation awareness. 

- I am concerned that the classifier will not hold against adaptive attacks, injections that are specificallty constructed with awareness of such a classifier (e.g., that try to hijack attention).

- In Figure 1, visually comparing blue-to-orange attentions (aligned instructions) and red-to-orange (misaligned instructions), it is difficult to conclude whether there is a meaningful difference between the two. From that plot alone, there does not appear to be a substantial difference between attention patterns of aligned and misaligned instructions. Additionally, white space symbols in Figure 1 should be removed for better readability.

- I am somewhat surprised that AttentionTracker leads to worse results. The explanations provided are somewhat counterintuitive, as it should in principle provide cleaner training data (?), given that not all attention heads contribute meaningfully to instruction following.

[1] Can LLMs Separate Instructions From Data? And What Do We Even Mean By That?
[2] SecAlign: Defending Against Prompt Injection with Preference Optimization
[3] Defeating Prompt Injections by Design
[4] ASIDE: Architectural Separation of Instructions and Data in Language Models

### Questions
- Could you please elaborate on the email example from lines 37-45, or provide a more convincing example where a system would genuinely benefit from treating certain inputs as aligned instructions?

- Could you provide a table with mean logit values of the classifier, or distribution plots with AUROC values, so one can assess the overlap between malicious and non-malicious inputs?

- How do you expect AlignSentinel to perform against adaptive attacks where adversaries are aware of the classifier and craft injections specifically to evade detection?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces AlignSentinel, an alignment-aware detection method for prompt injection attacks that explicitly models instruction hierarchy. Unlike existing binary detectors that classify any input containing instructions as malicious (leading to high false positives), AlignSentinel categorizes inputs into three classes: inputs with misaligned instructions (attacks), inputs with aligned instructions (legitimate), and non-instruction inputs. The method leverages attention-based features extracted from LLM attention maps to distinguish these categories and proposes two variants (Avg-first and Enc-first). The authors also construct a comprehensive benchmark spanning eight domains for systematic evaluation, demonstrating that AlignSentinel substantially outperforms baselines with near-zero false positive and false negative rates.

### Strengths
- Novel problem formulation that addresses a critical limitation of existing prompt injection detectors by explicitly accounting for instruction hierarchy, reducing false positives on benign but instruction-containing inputs.
- Strong theoretical motivation for using attention maps as detection signals
- New benchmark which will be useful for the community
- Thorough evaluation
- Proposes two variants and explores their differences

### Weaknesses
- I would appreciate more discussion of limitations/potential weaknesses. Do the authors think AlignSentinal is easy to beat? Could an attacker trick the model's idea of task alignment?
- I think it would be stronger if the authors discussed more about why baselines seem especially weak. It makes sense that since AlignSentinal is developed in response to this alignment problem it is much stronger, but some of the baselines seem almost mistrained given their extremely high FNR.
- It would help to have more concrete examples or a more detailed discussion of what cases the authors sees Enc-first being better than Avg-First.

### Questions
[Figure 1] This figure is somewhat confusing, are the three graphs separate responses to the same user prompt tokens? The text says that red tokens highlight instruction in misaligned input that conflict with higher priority instruction so why are there red tokens in the aligned/non-instruction areas? The aligned/non-instruction attention maps seem like copies of each other is this intended? If so, as a reader, what are we supposed to understand from this similarity? Why is a copy needed? Does this indicate that the first part of the aligned response is non-instruction?

[349] Is the different batch size due to the size of the model? If so, can you specify here a comparison of the size of the models?

[Table 1] Can the authors comment on why Prompt-Guard has such a high FNR? Is it poorly trained or is this an artifact of the method?

[389] Can the authors comment on why Coding and Entertainment might be specifically hard in their benchmark?

[423] Is there a reason why these groupings were chosen?

[395] Can the authors comment more specifically why their features are stronger than AttnTracker when both are performing binary classification?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors present a prompt injection defense AlignSentinel which classifies inputs using attention scores and adds an "aligned instruction" category to the standard categories for a prompt injection defense.

### Strengths
- Incorporating an aligned instruction category into the training of the defense is a natural way to reduce false positives on benign inputs
- Internal signals such as attention are potentially strong indicators of malicious input as shown by prior work [1, 2]
- AlignSentinel shows strong results in the author's experiments (however, I have concerns regarding reliability, see Weaknesses)

[1] Hung et al. Attention Tracker: Detecting prompt injection attacks in LLMs. 2024.

[2] Choudhary et al. Through the Stealth Lens: Rethinking Attacks and Defenses in RAG. 2025

### Weaknesses
- The reliability of the exceedingly strong results (near perfect results in terms of FPR and FNR) in the author's experiments is unclear. In particular, it is unclear whether these strong results are due directly to the ability of AlignSentinel to distinguish unaligned instructions from benign inputs, as suggested by the authors. The authors use the same pipeline to produce training examples for the clasifier and (with a domain shift) for the test set of attacks and benign samples, which may enable reliance on structural patterns in the "unaligned instruction" class. Adaptive attack evaluation is critical to determine whether the author's technique genuinely improves robustness; I cannot recommend acceptance without a thorough evaluation in the presence of adaptive attacks ([2] presents adaptive attacks against an attention-based defense, a similar attack may be effective here).
- The attention pattern analysis in section 4.1 used to motivate the defense needs more depth. A thorough analysis of the distribution shift in attention patterns across multiple classes of misaligned, aligned, and non-instruction responses is necessary to motivate the approach. Likewise, additional discussion is needed regarding the relevance of attention from the tool response to the user prompt to the presence of malicious injected instructions.

### Questions
Does AlignSentinel successfully defend against against adaptive attacks (e.g. those which explicitly incorporate a detection-evasion criterion into an optimization process, as in [2]) and a variety of simple prompt injection attack baselines beyond those used to generate the training data (e.g. those discussed in [3])? I would increase my score if a thorough analysis indicates that it does.

[3] Liu et al. Formalizing and Benchmarking Prompt Injection Attacks and Defenses. USENIX Security 2024.

### Soundness
2

### Presentation
2

### Contribution
2
