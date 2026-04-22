# Making, Not Taking, the Best of N

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 6, 8, 2

## Abstract
Obtaining high-quality generations in modern LLMs has largely been framed as a selection problem: identifying a single winning generation from a diverse pool of $N$ samples, the Best-of-$N$ (BoN).
Yet, this approach is inherently zero-sum, discarding diverse and potentially useful information from the pool. Instead, we explore a collaborative setup, where all candidates can potentially contribute to the final winning generation. To this end, we propose Fusion-of-$N$ (FusioN): a method that uses a general LLM judge to synthesize the most informative elements of each sample into a single final answer. 
We compare FusioN to BoN in two settings, (i) test-time scaling, where we sample and aggregate from a single model at test-time (ii) synthetic data generation, where we fuse samples from a pool of diverse teachers to improve a student model. We extensively benchmark both setups across 11 languages, 3 diverse benchmarks and varying model scales. Across the bench, FusionN consistently outperforms BoN showing versatility and robustness both in test-time scaling and in downstream gains from synthetic data generation. We also perform extensive analysis on FusioN, where it shows surprising strengths and robustness under challenging settings.
These results show that we should shift how we think about evaluating and utilizing LLM generations from a monolithic measure of quality, to embracing their polylithic nature. This shift allows us to integrate diverse strengths, unlock latent potential, and achieve improvements that were previously inaccessible through selection alone.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper argues that the dominant “choose one winner” aggregation paradigm—Best-of-N or its variants like self-consistency—is wasteful and brittle because it discards potentially useful information present across candidate generations. The authors propose Fusion-of-N (FUSION): instead of selecting, a strong LLM synthesizes a final answer by integrating the best parts of N candidates. They evaluate FUSION in two common aggregation settings: (i) test-time scaling (multiple samples, one model) and (ii) synthetic data generation (multiple teachers, one student). Across 11 languages and three benchmark families (Arena-style open-ended dialogue, WMT24++ translation, and reasoning datasets like MGSM/GeoFactX), FUSION is reported to (a) improve mArena-v2 win-rates vs GEMINI2.5-PRO, (b) raise XCOMET-XL translation scores (even beating the oracle best sample on some languages), and (c) produce better synthetic datasets that yield downstream gains when used to fine-tune 7B/111B students.

### Strengths
### S1. Clear, compelling reformulation (selection → synthesis)
The paper articulates a neat conceptual shift: treat aggregation as synthesis rather than selection. The formalization and narrative around a polylithic view of quality (good and bad fragments within each sample) nicely motivate why a generative judge could outperform scalar scorers. The method is also simple to adopt: keep the same candidate pool, swap BON for a fusor prompt.

### S2. Broad empirical coverage with multilingual focus
The evaluation spans 11 languages, two scales for test-time scaling (8B and 111B) and multiple downstream tasks. The work puts emphasis on multilingual open-ended dialogue (mArena-Hard v2.0), WMT24++ translation (scored by XCOMET-XL), and GeoFactX/MGSM reasoning—giving the reader a more realistic view of behavior beyond English.

### S3. Strong results; synthesis can beat oracle in MT
In machine translation, FUSION beats BON by large margins and even exceeds the oracle (best single candidate given ground-truth references) in several languages (e.g., German, Russian, Chinese), establishing that generative fusion is not upper-bounded by selection. That is a crisp and important empirical takeaway.

### S4. Test-time scaling wins and consistent synthetic-data gains
On Arena-style evaluation, FUSION improves win-rates against a frontier model, Gemini 2.5-Pro, using the same N as BON; e.g., +9.5% in German for COMMAND A with N=5. Fine-tuning on FUSION-selected data yields +2.5 Arena win-rate avg and +0.8 XCOMET-XL over BON-selected data showcasing that fusion not only helps at inference, but also transfers to training.

### S5. Useful analysis of how synthesis works and when it fails
The paper inspects contribution at the character/segment level (difflib SequenceMatcher), showing the fusor often copies >80% from the presented candidates, rather than hallucinating wholecloth. Further, FUSION tends to reach BON’s quality using fewer samples. It also shows a certain size threshold—bigger fusors unlock better generative fusion—clarifying practical requirements. Moreover, the paper is candid about math being a weaker regime for out-of-the-box FUSION.

### Weaknesses
While I like the paper and it has strong contributions, I have a few minor suggestions. If my main requests are fulfilled (W1, W2), I would be happy to raise my score.

### W1. Dependence on a single proprietary judge.

Headline results often rely on GPT-4o for LLM-as-judge preferences (for Arena-style head-to-heads). This raises a concern: Are fusor outputs advantaged by stylistic preferences of GPT-4o (e.g. verbosity, hedging tone), rather than objective quality?
Please report, at least for a subset of languages, human eval or a second independent judge to ensure robustness (and estimate judge bias if any).

### W2. Reproducibility / openness.

A key component of the evaluation, the internal RM for scoring BoN, is proprietary or unreleased. The paper promises to release fused vs BON-selected datasets “where licenses allow,” which is good, but full reproduction remains difficult. I would strongly suggest the authors to also use an open-source reward model to make the findings reproducible for the open-source community. Please also release (i) judge prompts, (ii) fuse prompts (Table 4 already helps), (iii) a BON baseline using only public RMs, so independent researchers can validate.

### W3. Math / verifiability tasks expose a limitation.

On MGSM, BON outperforms FUSION both at test-time aggregation and after SFT. FUSION can actually degrade logical consistency by “cleaning up” reasoning steps in ways that introduce subtle arithmetic or chain-of-thought errors. This undercuts the “drop-in replacement” message: FUSION does not dominate BON in all domains, especially where correctness is binary and checkable.

I would appreciate if the authors can be more direct in the main text. For example, something like: “FUSION helps open-ended generation and translation, but BON (plus verification) is still stronger for objectively verifiable tasks such as math.” 

### W4. Statistical reporting could be tighter.

Arena-style win-rates are given mostly as point estimates (+10.8, +9.5, etc.). The paper notes position biases and tie-handling, but doesn’t present full confidence intervals or absolute counts of judged pairs per language. Please add CIs or bootstrap intervals for all win-rates.




## Presentation Nitpicks:

### Typos / phrasing

1. In the fusion prompt section (Table 4), I noticed minor spelling issues such as “pormpt” → “prompt”, and awkward phrases like “scaler rater,” which should probably read “scalar rater.” 
2. GeoFactX evaluation text includes a phrase like “agains the list,” which should be “against the list.” 
3. MGSM discussion: “math task compared BON” should read “compared to BON.” 
4. Figure captions / axis labels occasionally misspell “mArenaHard-v2” (e.g. “m-AreanHard-v2”). Please normalize. 
5. “XCOMETXL” vs “XCOMET-XL” appears inconsistent; use a single canonical form in text and figures. 


### Figures / presentation

1. In Figure 1 / main win-rate barplots, please annotate $N$ (e.g. N=5 samples fused), identify the judge model, and the number of judged pairwise comparisons so readers can gauge statistical strength at a glance. 

2. Add error bars / confidence intervals for win-rates, or at least show per-language sample counts. Right now improvements are large, but we can’t tell which differences are stable and which are noisy.

### Questions
Q1 - Cost–quality trade-off: Can you plot final quality metric (win-rate vs GEMINI-2.5-PRO, XCOMET-XL, MGSM accuracy) against actual inference cost? For BON and FUSION at N = 1, 2, 5, 10? Include both FLOPs/token estimates and observed latency on your hardware. This would make the “FUSION is more sample-efficient” claim more actionable. 

Q2 - Robustness to adversarial / low-quality candidates: You test robustness to weaker teacher pools, but what if a candidate is malicious: toxic, off-policy, prompt-injected, etc.? Does the fusor tend to copy problematic spans, or does it filter them out? Any ablations where you intentionally inject garbage/noisy generations into the pool?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper argues that the dominant Best‑of‑N (BON) framing of generate N candidates and select one throws away useful signal. The authors propose Fusion‑of‑N (FUSION): a training‑free, prompt‑driven synthesis step in which a fusor LLM reads the prompt and the N candidates and writes a new answer that integrates the best parts and discards the worst. 

They evaluate FUSION in two settings that currently rely on BON: (i) test‑time scaling (sampling multiple candidates from the same model and aggregating), and (ii) synthetic data generation (sampling across a pool of teacher models and aggregating for SFT). Experiments span 11 languages, three task families (open‑ended generation on mArenaHard‑v2, WMT24++ translation, and factual/math reasoning), and two model scales (8B and 111B).

FUSION consistently beats BON at the same sampling budget and, on WMT, sometimes exceeds the “oracle” single candidate scored against references, establishing that selection is not an upper bound. The paper further analyzes when FUSION works best (larger fusors, small N, diverse teacher pools) and where it struggles (close‑ended math)

### Strengths
- With a single prompt, FUSION is drop‑in for selection and consistently improves win‑rates on open‑ended prompts and XCOMET‑XL scores on translation across 11 languages. The fact that FUSION sometimes surpasses the reference‑scored oracle in WMT (German/Russian/Chinese; Fig. 2, p. 5) is a sharp, convincing result showing synthesis > selection.
- The paper evaluates both test‑time and data‑generation use cases, the two places BON is used most heavily today, and does so across 11 languages

### Weaknesses
- The attribution method (difflib) cannot detect paraphrastic reuse or semantic fusion. Also, the fusion prompt is English‑only, and Table 3 (p. 8) notes slightly lower “language correctness,” suggesting the prompt could disadvantage lower‑resource languages. A multilingual fusion prompt and a semantic contribution analysis (e.g., span‑level alignment or entailment) would strengthen claims about “polylithic” quality.
- The core sales point is sample efficiency at small N, yet the paper does not report wall‑clock or token‑level costs for: sampling N candidates, concatenating them into a long‑context window, and running fusion. Since BON can parallelize N samples and fusion adds a sequential long‑context pass, end‑to‑end time/compute curves vs. quality are necessary to substantiate the “efficient” claim (authors briefly note this qualitatively, and Fig. 5 shows win‑rates only).
- The authors observe a mild position bias (Fig. 11b, p. 25) and randomize candidate order, but do not explore robust mitigations (e.g., segmented fusion with random reshuffles, majority‑vote over multiple shuffled fusions)

### Questions
- Beyond the position‑bias probe, did you (a) repeat Arena comparisons with two heterogeneous judges and report agreement, or (b) conduct a small‑scale human‑blind study to validate GPT‑4o outcomes, as recommended by recent LLM‑as‑Judge surveys? Please add such a calibration or discuss why it’s not feasible.
- Can you provide token‑level accounting for (i) sampling N candidates, (ii) concatenation length, and (iii) fusion generation—and compare end‑to‑end latency vs. BON at equal quality? This matters because FUSION’s long‑context pass is not trivially parallelizable, and likely adds significant user compute time
- Table 3 notes slightly lower “language correctness” for FUSION, and the fusion prompt is English‑only (Table 4). Have you tried localized fusion prompts (per language) or a multilingual safety/style rubric? If so, do GeoFactX non‑English results improve?
- What happens if 1–2 of 5 teachers inject subtly wrong facts or unsafe content? Does FUSION filter them or amplify them, and can adding a pre‑filter materially help?
- Could you complement string‑matching with semantic span alignment (e.g., NLI‑based attribution or claim‑level voting) to better characterize “fusion” vs. “copy”? This will clarify whether FUSION is truly synthesizing or mostly selecting + rephrasing

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes Fusion-of-N (FUSION), a paradigm that replaces the traditional Best-of-N (BON) selection strategy for LLM responses. Instead of picking a single “best” candidate among N sampled outputs, a fusor model synthesizes a new response by combining the strengths of all N candidates. The authors claim this approach better utilizes complementary information among responses.
They evaluate FUSION in two scenarios: (1) test-time scaling—sampling multiple responses from a single model and fusing them; and (2) data synthesis—fusing outputs from multiple teacher models to generate training data for a student model. Experiments span several multilingual and multi-domain tasks (open-ended generation, translation, factual QA, and math reasoning). The paper provides a universal prompt template for fusion and reports that FUSION outperforms BON on most tasks, though BON remains stronger on MGSM (math reasoning).

### Strengths
1. The paper identifies a real limitation of BON, the inability to exploit complementary information between candidates—and argues for synthesis rather than selection as a reasonable alternative.

2. The method is lightweight: it requires no additional training or fine-tuning, only a general “fusion prompt,” making it reproducible and accessible for practitioners.

3. The authors openly report cases where FUSION fails (e.g., MGSM), which adds transparency.

### Weaknesses
1. The proposed method is largely a prompt-level heuristic. The core idea of asking a strong LLM to compare and fuse multiple outputs, is conceptually simple and closely resembles prior response aggregation or mixture-of-agents paradigms. The paper mostly offers a new name and systematic evaluation rather than a substantive algorithmic innovation.

2. The BON baseline uses an unspecified reward model and selection protocol. Details such as training data, tuning strategy, and model scale are unclear, making it difficult to judge fairness. The authors only compare against small-N BON (typically N=5), omitting modern BON improvements (hierarchical sampling, de-duplication, or diverse beam reranking). All comparisons rely on a single LLM judge (GPT-4o), introducing possible evaluation bias.

3. The paper claims FUSION sometimes surpasses the oracle best candidate (highest XCOMET score among N). However, this “oracle” is defined by an automatic metric; it is not clear that human annotators would agree. This undermines one of the paper’s central claims.

4. The fusion step concatenates N long answers into a single context, increasing token length and making inference sequential and costly. The authors briefly mention sample efficiency but ignore wall-clock latency, memory, or dollar cost. In contrast, BON’s selection is trivially parallelizable.

5. FUSION underperforms BON on verifiable reasoning tasks like MGSM but outperforms on subjective or open-ended tasks. The paper fails to explain why or to propose an adaptive choice mechanism.

6. Input formatting (order of candidates, truncation, token limits, multilingual mixing) is under-specified. The authors mention minor order bias but do not quantify its effect.

### Questions
1. What reward model and settings were used for BON? Would a stronger public RM or combined judge close the reported gap?

2. Can you provide human or multi-reference evaluations showing that FUSION genuinely produces higher-quality translations than the best candidate?

3. Why does FUSION fail on MGSM? Is it due to logical reasoning disruption, verbosity, or hallucination?

4. Please report the average input length, latency, and cost per query for N={2,4,8}.

### Soundness
2

### Presentation
2

### Contribution
1
