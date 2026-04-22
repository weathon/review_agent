# On the Role of Reasoning Traces in Large Reasoning Models

- Avg Score: 2.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 4, 2

## Abstract
Large reasoning models (LRMs) increasingly externalize intermediate thoughts through structured reasoning traces, raising the possibility that their internal decision processes can be inspected. However, recent observations suggest that these traces may not reliably reflect the factors influencing final outputs. We introduce Thought Injection, a controlled intervention framework that inserts targeted reasoning fragments directly into the model’s private $\texttt{think}$ space and then evaluates (i) whether the injected content changes the final answer, and (ii) whether the model acknowledges this influence when asked to explain its output. Across 75,000 controlled trials spanning subjective list-generation tasks, we observe a consistent pattern: models frequently adjust their answers in the presence of injected reasoning, yet rarely disclose this internal influence. Instead, they often provide alternative fabricated explanations even in settings where the influence of the injected trace is directly observable. These findings indicate a persistent mismatch between LRMs’ internal processes and their user-facing explanations, raising fundamental challenges for approaches relying on reasoning-trace transparency or explanation faithfulness.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors inject artificial cues into the model's CoT for a family of simple factual questions. These statements say to not mention entity X. These cues often have a causal impact on the final answer. When the model is asked, in a second turn, why it did not include X, it is often dishonest. The authors show correlations between the activation difference with and without hint, and the sycophancy, deception and evil directions

### Strengths
- Faithfulness and model dishonesty are important questions
- The authors study several models, include large and high quality ones

### Weaknesses
As I see it, the authors make 3 main claims
- "Our results provide the first systematic
  evidence that reasoning traces causally shape model outputs"
    - I agree that there is a causal effect and that good evidence was provided of this
    - I disagree that this is novel, eg [Lanham et al 2023](https://arxiv.org/abs/2307.13702) shows this. And in my opinion this is implicitly shown in a great many works on chain of thought
- "Reasoning traces thus provide false transparency: they causally determine outputs but systematically misrepresent their own influence when asked, undermining the foundation of reasoning-based oversight"
    - I agree that this has been shown in the narrow setting studied by the authors 
    - I disagree that this has been shown to apply to general reasoning choices because the authors study the specific, highly off policy setting, where they make an artificial edits to the model's thoughts, which is importantly different from monitoring natural reasoning traces 
    - Further some of the edits will naturally cue the model to act misaligned, cueing dishonesty and this will be non-representative of normal behaviour 
    - Further, reasoning based oversight is about monitoring the chain of thought not about asking them follow-up questions or having another language model try to process and write summaries of it. See [Korbak et al](https://arxiv.org/abs/2507.11473) for a more detailed case. In my opinion it is well known that asking the model about does not necessarily result in honest answers, even in the case where the model has a chain of thoughts to read rather than being expected to analyse its own internal mechanisms
    - Further, this is a fairly specific setting with a fairly easy and straightforward problem. As argued in [Korbak et al](https://arxiv.org/abs/2507.11473), this is not the setting where chain of thought monitoring is likely to help. For problems that can could be done without the CoT there's no mechanistic reason for the chain of thought to be useful. The difficult problems where the CoT is needed to increase the serial depth of computation available, are where we should expect this to work. In particular, on the hardest tasks models can solve.
- Mechanistic correlates
    - There may or may not be something here, I find it hard to tell.
    - It would be good for the authors to look at the similarity to non-evil persona vectors to make sure it's not just something about their persona Vector construction method 
    - The authors provide no information about how they construct the persona vectors making it difficult to evaluate 
    - Overall I wouldn't find this results particularly significant either way, given the various confounders detailed above

### Questions
Please correct me if I have misunderstood any key details of your work in the above!

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper studies the influence of the reasoning traces in large reasoning models. They insert synthetic reasoning snippets into the reasoning traces and measure causal effects on outputs. They found that the injected hints significantly alter the final answers, but the LRMs conceal these injections over 90% of the time.

### Strengths
- The reasoning trace offers an interesting perspective on the understanding and controlling of the LRM's reasoning.  
- The finding that the LRMs tend to conceal these injections is impactful.

### Weaknesses
- Many parts of this paper are not really finished. E.g., Table 8 mentions that some results are pending. There are a lot of occurrences of broken references. The idea behind this paper is good, but I think this paper, at its current status, is not ready to be accepted.  
- The "reasoning trace" seems to resemble "chain of thought". Korbak et al (2025) "Chain-of-Thought Monitorability" https://arxiv.org/abs/2507.11473 seems relevant, and it'd be great if this paper mentions CoT monitorability.  
- The "causal" claim needs to be quantified better, e.g., via a framework that measures the causal effects (e.g., total effect)  
- The honesty evaluation could be described in more detail. Many terms are left undefined, e.g., "semantically equivalent", "explicitly attributes this reasoning to the model's own reasoning process".

### Questions
Please refer to the reviews above.

### Soundness
2

### Presentation
2

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates whether reasoning traces in large reasoning models genuinely shape outputs or merely serve as post-hoc rationalizations, and whether models honestly acknowledge when their reasoning influences their answers. The authors introduce thought injection, an intervention method that injects synthetic "hint" snippets directly into the model's reasoning process while keeping the user query fixed. Across 45,000 generations using three open-source large reasoning models (Qwen3-235B, DeepSeek-R1, and Qwen3-8B), the study reveals very large drops in the inclusion rate of a query's "expected element" when the hint tells the model to avoid it. They then ask "Why didn’t you mention X?" and find low "honesty rates" especially under deliberately misaligned "hatred hints", claiming models often fabricate post-hoc explanations. Finally, an activation-level analysis associates concealment with directions labeled "sycophancy" and "dishonesty" suggesting possible detection handles.

### Strengths
Novel intervention methodology: The paper introduces Thought Injection, a creative and technically sophisticated approach to studying reasoning traces. Unlike prior work that manipulates prompts or removes reasoning entirely, this method directly injects content into the reasoning trace itself while keeping queries fixed. This design elegantly isolates the causal effect of reasoning content from confounds like prompt engineering or task difficulty changes, representing a genuine methodological advance.

Dual-level evaluation framework: The paper's methodological contribution is the two-stage evaluation that separately measures (1) whether reasoning traces causally influence outputs, and (2) whether models honestly report this influence. This dual framework transforms a simple intervention study into an investigation of transparency and potential deception, opening new research directions around "false transparency" in AI systems.

### Weaknesses
Task scope is narrow to support broad claims. All evaluation comes from 50 prompts of a single list template chosen because they won’t violate accuracy when following hints. That design maximizes compliance headroom but samples only one behavior regime, where "expected elements" like Einstein are pre-sticky and easy to suppress. As written, this doesn’t demonstrate that "causal influence" and "dishonesty" generalize beyond subjective list ranking to tasks with objective ground truth, multi-step derivations, or tool use.

Selection on the dependent variable inflates effects. Queries were retained only if an element appears ≥90% in baseline answers, ensuring near-deterministic inclusion before intervention. This enriches for "ultra-stable" items and may overstate both suppression magnitude and the ease of demonstrating a counterfactual shift, limiting external validity to less canonical prompts. 

Honesty requires semantic equivalence to the hint and explicit self-attribution; evaluation is primarily LLM-as-judge (gpt-oss-20B) across 30k follow-ups. The paper cites a human study with κ>0.85 on a 500-sample subset, but provides no judge prompts, rubrics, per-class breakdowns (Honest/Fabricated/Evasive), or confusion matrices by model or hint type. Strong claims like "over 90% concealment" hinge on this pipeline; without full labeling details, reliability and construct validity remain unclear.

"Causal" contribution risks collapsing into prompt-conditioning. The hint is placed inside <think> but is otherwise standard context; the method section even emphasizes that the injected text is indistinguishable from the model’s own trace under ChatML formatting. The paper shows strong suppression of expected elements (∆≈−0.88 to −0.95 medians), but does not compare against matched placements (outside <think>, post-answer, or as inert/scrambled text). Without such controls, it’s hard to claim a reasoning-trace-specific effect rather than generic sensitivity to earlier context.

### Questions
Q1: What is the effect size when hints are injected outside the <think> block (e.g., as regular assistant text or system messages)? How do effects vary with hint placement within the think trace?

Q2: Table 8 shows placeholders ("–") for Qwen3-8B paired tests. Can you complete these analyses and confirm whether results align with the other models?


Q3: The paper uses strong anthropomorphic language ("dishonesty," "concealment," "fabrication"). However, couldn't these behaviors reflect: a. Confabulation due to lossy memory/context b. Safety training to avoid discussing potentially harmful reasoning or c. Uncertainty about which parts of the trace actually influenced the decision. What evidence distinguishes intentional deception from these alternative explanations?

### Soundness
3

### Presentation
2

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
This paper introduces 'Thought Injection', a counterfactual intervention method, that is meant to inject synthetic reasoning into traces in order to measure causal effects on outputs. Across ~45k generations on (narrow) list-style prompts with (somehow) selected "expected elements", injected hints dramatically suppress mention of those elements (e.g., dropping “Einstein” after a “avoid Einstein” hint), which the authors interpret as evidence that reasoning traces shape answers rather than serving as post-hoc rationalizations. Claims about dishonesty are also made.

### Strengths
- The thought injection seems to directly intervene on the reasoning trace, letting the authors compare outcomes w/ and w/out the hint. 
- There are ~5000 trials, and results are evaluated in paired per-trial statistics; in Wilcoxon and sign tests, plus bootstrap CIs, effects appear large and consistent.

### Weaknesses
- Explaining why an item was omitted is not the same as explaining why it was included and, indeed, false positives and false negatives may behave differently in this game. How did you stratify and report results separately for target-included vs target-omitted generations (i.e., false positives vs false negatives), and test for asymmetries in disclosure/accuracy between them?
- Often, papers that use a forced choice, top-K, or even some sort of constraint-satisfaction report more than merely hit-rate but also rank-sensitive list metrics and set-quality measures at varying k, but this analysis is overly simplistic. Additionally, the empirical setup is very narrow (top-5 scenario only) and hints appear to be extreme and abnormal ("I HATE HUMAN"...really??). It also appears that only queries where an 'expected element' appears in 90% of baseline generations are used, which may be an ecologically invalid scenario or inflate suppression effects of the chosen intervention. It would not be difficult to perform a more fulsome evaluation.
- The 'open question' on L201 has some amount of debate and plausible answers or mitigations -- it may have been helpful to expand your literature review, and to follow up on some of thes claims. This is minor.
- Some weaknesses may merely be missed opportunities for explanation and not overclaiming, though that can't be discounted. Some questions are below, mostly regarding 1) the validity of the hint itself -- whether the authors are only testing compliance with inserted instructions and not actual changes to reasoning, and 2) whether models are actually 'dishonest' as a consequence, which seems spurious. 
- Minor, but please check your spelling and grammar (e.g., "if model choose to follow" (L186))
- We are told there is a 'detailed curation' of data in Sec 2.1, but we are only given the single (!) "list the 5..." template and not how the superlative, category, or scope were defined or selected, nor how possible quality control was conducted.

### Questions
- It seems somewhat that the explantion for "not acknowledg[ing] the influence of HINT" (L74) leaps to the explanation for "dishonesty" (L76) and "conceal[ment]" (L77). Poor post-hoc reasoning or post-hoc explanations are known problems across LLMs generally -- are you making a hasty generalization?
- Did you test whether the model might be able to identify whether inserted HINTS are endogenous or from an external source? I.e., how do you know how it 'treats' (L172) each sentence or phrase? How might one investigate this? Along these lines, if the injected hints are text-level and pre-existing of the inference process, given that inputs tend to be read 'at once' in transformers, how did you guarantee the causal influence on hidden *state variables*?
- Would you prefer to change the term 'factual hints' to 'plausible hints', since the examples (Table 1) refer less to facts and more to subjective feelings or uncertainty? What matter is Coca-Cola's marketing team? Who did the 'linking', what is a 'link', and why should it matter? Hardly factual.

### Soundness
1

### Presentation
3

### Contribution
2
