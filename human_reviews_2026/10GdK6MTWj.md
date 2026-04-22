# The Invisible Mind: Auditing Privacy Invocation in Latent Chain-of-Thought Reasoning

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Latent Chain-of-Thought (Latent CoT) enables reasoning in the continuous internal states of large language models (LLMs), allowing non-linguistic paths beyond token-level explicit CoT. While this creates an implicit privacy risk, models can invoke and reason over private knowledge inside the latent chain, bypass content guardrails, and produce answers that causally depend on that knowledge without reproducing it. We formalize this risk as Private Implicit Knowledge Invocation (PIKI), defined as non-verbatim causal dependence on private knowledge within an implicit chain. We introduce \textit{PIKI-Test}, a dataset with single- and multi-hop privacy questions for auditing Latent CoT LLMs. Using \textit{PIKI-Test}, we audit Latent CoT LLMs and evaluate content guardrails to study how privacy propagates under Latent CoT. We also present \textit{PIKI-Attack} to backtrace latent exposure, and \textit{PIKI-Solve}, a top-down hop decomposition with conservative decoding that reduces exposure and improves auditability. Across multiple models and guardrails, Latent CoT LLMs show about 56\% privacy exposure under multi-hop evaluation, and content guardrails see a 37\% drop in recall on multi-hop privacy QA. These results clarify the privacy risk of latent reasoning in Latent CoT and establish a new audit target for safety-critical LLM deployments. Our code and dataset are available at [this link](https://anonymous.4open.science/r/PIKI-076D).

Privacy note: All privacy-sensitive data are synthetic; no real personally identifiable information (PII) is present.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper highlights safety-concerns from opaque latent CoT’s. It is focused specifically on the risk of latent CoT enabling private knowledge leakage. The paper argue's that latent CoT models can reason over private knowledge without detection, and attackers could exploit this via the use of multi-hop questions (where a middle hop involves use of private knowledge) to extract the private knowledge while bypassing guardrails.

To explore the above concerns, the paper contributes the following:
- They introduce a novel multi-hop privacy-related reasoning dataset which can be used to explore these concerns empirically.
- They empirically demonstrate that existing latent CoT LLMs can (latently) reason over private knowledge.
- They empirically demonstrate that guardrails (for preventing private knowledge leakage) decline in efficacy as hop size increases. This further confirms the potential risk and vulnerabilities.
- They use a protocol, Piki-Attack, to demonstrate that an attacker can backtrace throuhg multi-hop queries to expose private knowledge.
- They introduce Piki-Solve, a defence that decomposes multi-hop queries into more granular natural language requests, thus allowing content guardrails to better detect potential privacy leakages.

### Strengths
- Risks from opaque latent CoT is an important area, and is certainly worth highlighting
- The focus on privacy leakage via opaque latent CoT seems novel. It is plausible this focus is both: (i) interesting in its own right, and (ii) a concrete setting that enables empirical investigations that can bear on opaque latent CoT concerns more generally.
- The Piki-Test dataset is a valuable contribution to the community. Reasonable effort seems to have been put into the datasets curation and quality-assurance.
- Evidence of the stated risks are provided via experiment. The paper demonstrates models can latently reason over private knowledge, that guardrails are vulnerable in multi-hop settings, and that attackers could exploit this. In particular, the vulnerability of guardrails in this setting is a noteworthy result that highlights the risks.
- A solution to the problem is proposed and validated (Piki-Solve)

### Weaknesses
**Importance and relevance of the multi-hop privacy-leak setting.**

I have have some doubts about the relevance of the particular privacy-leakage setting focused on in the paper. Perhaps the authors could better motivate the importance of this setting. This could make the results and overall paper seem more significant. For example: (1) (if intended) the paper could better highlight that this setting is a sub-setting (privacy-leakage via multi-hop queries) that could be generally relevant as a test-bed that can also bear on broader opaque CoT concerns. (2) The paper could better detail whether this particular multi-hop privacy leakage scenario could lead to real-world risks, for example by providing a concrete example. (3) In future work, it is said “we will extend PIKI to the broader class of unauthorized knowledge” - the paper could better highlight the similarity between their setting and the broader class to help better motivate their work.

**Missing experimental details.**

In general, many of the low-level details of the experiments in Section 5 are unclear to me. The appendix also does not seem to clarify my question related to these experiments (I will elaborate on these questions below). This lack of clarity makes it hard to contextualize the results and understand the extent of their contributions.

Section 5.3 contributions seem significant, but I have a number of questions below I would like clarified to help better understand the results.

I am unconvinced about the value of the Section 5.4 experiments (the Piki-Attack contributions) due to very unclear experimental details.

I think the Piki-Solve experiments and method are worthwhile. However, I am unconvinced by the significance of this contribution. The Piki-solve solution seems highly specific to the Piki dataset, and it is unclear how it would generalize to a broader set of real-world concerns. The method generally seems quite computationally expensive (requiring multiple LLM calls to construct the probe cluster)? The method could perhaps be better motivated by highlighting a concrete deployment scenario where using this method could be useful and practical.

**Other.**

The discussion of limitations is too brief and lacking details and serious thought.

The dataset seems to be well constructed. However, some empirical validation of its quality could be reassuring.

### Questions
Could you provide prompts and example generations for all model evaluations in Section 5?

**Section 5.1**

When you use the Partial Exposure metric - is the private hop always the first hop in a multi-hop task?

**Section 5.2**

Could you use a frontier LLM answering instantly (i.e., with no CoT) as an analogy for latent CoT, and include these results additionally?

Related, In Table 1, I would like to see “baselines” of (i) a standard frontier LLM answering directly/instantly without the use of CoT, (ii) a standard/reasoning LLM answering with the use of explicit CoT. This would help contextualize where the multi-hop capability levels of the Latent CoT LLMs lie.

If the data used is synthetic, I am unclear how the models know the synthetic private knowledge. Do they have access to the privacy corpus mentioned later? Have you finetuned them to learn the synthetic private data?

**Section 5.3**

Many details are unclear here.

The different types of guardrails used are not described in any details. It is unclear what the inputs and outputs are for e.g., the LLM-judges. I assume the prompt to the judge is something like “Could the following query leak private knowledge?” ?

What do you mean by an "oracle-view"? Does this mean an “oracle” is answering the hop questions, and the guardrails are assessing the oracle solution?

The per-hop detection and risk grading metrics are not well described?

A limitation seems that the private knowledge is synthetic, and the guardrails must refer to a privacy corpus? This may be disanalogous to real-world scenarios where private knowledge is from the training corpus.

I’m generally unclear about the details and the significance of the PCA experiments. What model do these embeddings come from? Is it from a guardrail model? You show the embedding semantics change, then claim that this “makes privacy clues harder to track” and that this “dilutes the privacy clues”. I am unclear how exactly the demonstrated changes in embedding distribution relate back to the performance of the guardrails - you seem to have made an unjustified logical leap here?

**Section 5.4**

The details of the Piki-Attack in Section 5.4 are unclear. It seems you have constructed a dataset where exposing the private knowledge requires multiple questions, and each question is a 2-hop question - is this correct? What exactly are the experiments in Table 3: What are the input prompts and final outputs of the models in Table 3? How exactly are they performing backtracing? Are they performing both the “latent CoT role” and the “attacker” role illustrated in Figure 15? In the introduction, Piki-attack is described as a “protocol” but the exact protocol never seems to be described?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a new class of privacy risks in Latent CoT reasoning, termed Private
Implicit Knowledge Invocation (PIKI). In this setting, Latent CoT models may invoke private
information during hidden reasoning, which, despite not being explicitly reproduced in the
output, can still be indirectly retrieved. The authors build a synthetic dataset, PIKI-Test, to
encode such latent privacy dependencies and demonstrate that the **“single privacy hop”**
within multi-hop reasoning chains can be backtraced from model outputs. They further
propose PIKI-Solve, a defense approach that decomposes multi-hop queries into smaller,
auditable single-hop probes, improving the visibility of latent privacy cues and enhancing
existing content guardrails.

### Strengths
1. The paper identifies a new privacy risk that latent CoT reasoning may create a false
sense of privacy preservation. Even when private content is not explicitly generated,
it can still be invoked implicitly, revealing the (potential) limitations of current guardrail
systems and motivating the need for latent-level auditing.

2. The constructed synthetic dataset can serve as a useful benchmark for privacy
auditing in this domain, given the lack of such datasets.

3. Informative visualization: In Section 5.3, the PCA visualization of multi-hop privacy-
QA embeddings illustrates how the semantic distribution progressively shifts away
from the single-hop reference and aligns with general knowledge. This provides an
intuitive explanation for why guardrail detection weakens with reasoning depth.

### Weaknesses
1. **On the dataset construction:**
● The dataset setup assumes a deterministic tree-structured reasoning process
where each hop has a unique, fixed answer and only one hop contains
private information. This setting is highly idealized and differs from natural
multi-hop reasoning, where questions may have multiple valid answers or
involve several unknown or private hops. The benchmark reflects a worst-
case risk where an attacker knows (N-1) hops and tries to infer the remaining
private one, and it may overstate real-world exposure. The authors should
explicitly discuss and mention this in the paper.
● Given that the dataset is a core contribution, more practical variants (e.g.,
multiple private hops) should be included, and the paper should discuss how
privacy leakage would change on existing models as the number of private
hops increases.

2. **Paper is hard to follow and requires substantial improvements.**
● It is almost impossible to clearly understand the dataset setup in Section 4 without
going through all contents from page 14 to page 31 in the appendix. A simple and
concrete example or case study in the main paper would help a lot. Some useful
details (e.g., Tables 6, 8, 11, and many other example tables in the appendix) are
never referenced in the main text.
● Both the backtracking attack and the decomposing defense are conceptually simple
and straightforward. However, the current presentation makes them unnecessarily
hard to understand due to missing information:
● Sections 5.4 and 5.5, which should describe the core attack and defense
methods, lack methodological details. The referenced appendix (where such
details are expected) only includes two abstract figures with three-word
captions.
● Figures 1, 15, and 16 are visually appealing but not informative: Figure 1
appears too early, before readers understand the multi-hop and hierarchical
dependency structure. Figures 15–16 lack meaningful captions or textual
linkage. For the content around Figures 15–16, it would be important to
include (a) details of the attack/defense methods, (b) illustrative examples,
and (c) algorithmic descriptions if necessary. Proper figure referencing, with
captions explaining what each figure conveys, is necessary. In fact, Figure 1
is a better example than Figure 15 in illustrating the backtracing attack.
Overall, the paper would benefit greatly from **clearer examples, detailed
methodological explanations, and better figure integration**.

### Questions
Does latent privacy exposure show any trend with model size or architecture? It would be
helpful to understand whether PIKI is revealing an inherent property of latent-reasoning
LLMs, or whether the observed leakage largely stems from the synthetic control over the
reasoning structure.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper investigates privacy risks in latent reasoning models, where reasoning happens in hidden representations rather than explicit text. It introduces Private Implicit Knowledge Invocation (PIKI) to describe cases where private information influences model reasoning without appearing in the output. Using a synthetic benchmark (PIKI-Test) and related tools (PIKI-Attack and PIKI-Solve), the authors show that even latent chain-of-thought models can implicitly use private data. The work highlights a new privacy concern for future reasoning architectures, though its current real-world impact may still be limited.

### Strengths
The paper looks at an interesting and underexplored problem — privacy risks in latent reasoning models. Even though these models don’t reason explicitly in text, the experiments show that private information can still influence their internal reasoning. This a valuable insight and helps broaden how we think about privacy beyond just what the model outputs.

### Weaknesses
- It’s a bit hard to see how **critical or realistic the implicit invocation setup** is in practice. Latent reasoning models are still pretty experimental, and not many real-world systems use them yet. So while the idea is thought-provoking, the risk might feel a bit too hypothetical right now. The fact that open-source models already show fairly low exposure rates (around 10%) makes the threat seem limited.

- The **evaluation environment feels quite synthetic**. If you scale it up to more complex or longer inputs — like real-world text — the private information would probably get diluted even more, which might make the effect much weaker.

- The **dataset construction part is tough to read**. There’s a lot of math notation that doesn’t add much clarity, given everything is text-based anyway. Some simple concrete examples — like `(Alice_Kim, lives_at, 27_Garden_St)` — would make the process much easier to follow.

### Questions
I’m still a bit confused about the synthetic identity setup. Isn’t it very unlikely that the model could answer questions about these synthetic identities correctly? I assume there could even be multiple possible answers?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Private Implicit Knowledge Invocation (PIKI), a framework for auditing non-verbatim privacy risks in Latent Chain-of-Thought models. The authors formalize the risk of models invoking and reasoning over private knowledge internally without reproducing it in outputs. They develop PIKI-Test, a multi-hop privacy reasoning dataset with synthetic data across 12 dimensions, and evaluate how Latent CoT models expose private information through implicit reasoning chains. The paper also proposes PIKI-Attack for backtracing latent exposure and PIKI-Solve for mitigation through hop decomposition.

### Strengths
1. Interesting threat model angle: I like the idea that models can invoke and reason over private knowledge internally, bypassing content guardrails without verbatim reproduction. The non-verbatim causal dependence perspective is worth exploring.

2. Structured dataset: The PIKI-Test dataset has reasonable coverage with single- and multi-hop questions across 12 dimensions. The tree-structured design for incremental evaluation makes sense.

3. Practical defense: PIKI-Solve integrates with existing guardrails and shows improvements

### Weaknesses
Critical: Threat Model Clarity and Scope
1. Parametric vs. Inference Privacy Must Be Distinguished Early
The paper needs to clearly define upfront whether it is addressing:

Parametric memory: Information memorized from training data
Inference privacy: Information from data stores, context, RAG systems, etc.

This distinction is essential in privacy research [1]. The current framing conflates these two fundamentally different privacy surfaces. I recommend adding a clear statement in Section 1 or early Section 3 that explicitly scopes which privacy model you are addressing.


2. This Threat Model Applies More to Inference Privacy Than Training Privacy

I think the threat model actually maps more naturally to inference privacy scenarios rather than training-data privacy:

Pretraining/training data is largely assumed to be public (even if that assumption is imperfect in practice)
As shown in [2] and [3], training data extraction is extremely difficult even in the verbatim case, particularly for non-repeated information
Your synthetic "private" data (celebrity hotels, medical info) would rarely appear in pretraining in the first place
The multi-hop reasoning scenarios (e.g., "Where does J.K. Rowling stay?") feel more like RAG/context-based queries than training-data-memorization scenarios


3. Is This a Privacy Problem or a Multi-Hop Reasoning Problem?
I'm concerned that what you're observing may be more about compositionality gaps [4] than privacy specifically."Implicit chain reasoning is the main limiting factor for multi-hop privacy exposure"—this sounds like models failing at multi-hop reasoning, not a privacy-specific phenomenon.

Missing critical reference: Press et al., "Measuring and Narrowing the Compositionality Gap in Language Models" (EMNLP 2023) [4] directly addresses multi-hop knowledge composition failures and should be discussed prominently.

[1] Mireshghallah et al., "Can LLMs Keep a Secret? Testing Privacy Implications of Language Models via Contextual Integrity Theory" (ICLR)
[2] Huang et al., "Demystifying Verbatim Memorization in Large Language Models"
[3] Duan et al., "Do Membership Inference Attacks Work on Large Language Models?" (CLM)
[4] Press et al., "Measuring and Narrowing the Compositionality Gap in Language Models" (EMNLP 2023)

### Questions
1. Threat model scope: Can you add an explicit statement in Section 1-3 clarifying whether you're addressing parametric memory privacy, inference privacy, or both? Which scenarios are in vs. out of scope?

2. Inference vs. training: For your threat model to apply to training data, the private knowledge must be memorized during training. Can you provide evidence (e.g., from membership inference or data extraction) that Latent CoT models actually memorize the types of facts in PIKI-Test from pretraining?

3. Compositionality gap: How do you disentangle privacy-specific failures from general multi-hop reasoning failures? Can you show that models perform differently on multi-hop chains with privacy hops vs. equivalent-difficulty public hops?

### Soundness
3

### Presentation
2

### Contribution
3
