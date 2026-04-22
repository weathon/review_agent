# Reproducing and Dissecting Denoising Language Models for Speech Recognition

- Avg Score: 4.80
- Decision: Reject
- Scores: 6, 6, 4, 4, 4

## Abstract
Denoising language models (DLMs) have been proposed
as a powerful alternative to traditional language models (LMs)
for automatic speech recognition (ASR),
motivated by their ability to use bidirectional context
and adapt to a specific ASR model's error patterns.
However, the complexity of the DLM training pipeline has hindered wider investigation.
This paper presents the *first independent, large-scale empirical study* of DLMs.
We build and release a *complete, reproducible pipeline* to systematically investigate the impact of key design choices.
We evaluate dozens of configurations across multiple axes, including various data augmentation techniques
(e.g., SpecAugment, dropout, mixup),
different text-to-speech systems,
and multiple decoding strategies.
Our comparative analysis in a common subword vocabulary setting
demonstrates that *DLMs outperform traditional LMs*,
but only after a distinct compute tipping point.
While LMs are more efficient at lower budgets, DLMs scale better with longer training,
mirroring behaviors observed in diffusion language models.
However, we observe smaller improvements than those reported in prior character-based work,
which indicates that the DLM's performance is conditional on factors such as the vocabulary.
Our analysis reveals that a key factor for improving performance
is to condition the DLM on *richer information from the ASR's hypothesis space*,
rather than just a single best guess.
To this end, we introduce *DLM-sum, a novel method for decoding from multiple ASR hypotheses*,
which consistently outperforms the previously proposed DSR decoding method.
We believe our findings and public pipeline provide a crucial foundation for the community
to better understand, improve, and build upon this promising class of models.
The code is publicly available at https://anonymous.4open.science/r/2025-dlm/.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a **large-scale reproduction and analysis** of **Denoising Language Models (DLMs)** for ASR error correction. Prior work showed strong results with character-based DLMs, but their complex pipeline limited independent study. This paper:

* Releases a **fully reproducible pipeline** including TTS → ASR → DLM data generation 
* Conducts **systematic ablations** on augmentation, vocabularies, and decoding approaches
* Introduces **DLM-sum**, which incorporates multiple ASR hypotheses and improves WER over prior DSR decoding 
* Demonstrates **state-of-the-art LibriSpeech WER** using only LibriSpeech data
* Offers insights suggesting that **richer ASR hypothesis inputs**, including dense probability distributions, lead to further gains 

Overall, the work builds a strong empirical foundation for future DLM research under standard subword vocabulary settings.

### Strengths
* **Strong reproducibility focus** — complete open-source pipeline provided 
* **Broad coverage of design variables** (augmentation, checkpoint selection, vocabularies, decoding)
* **Novel decoding method (DLM-sum)** demonstrating consistent improvements 
* **Thorough experimental analysis**, including error categorization, model calibration, and search behavior
* **Clear SOTA claim under strict data constraints** (LibriSpeech-only)

### Weaknesses
1. **Limited novelty in model architecture**

   * The DLM itself is largely replicated from prior work; main contribution lies in empirical re-evaluation.

2. **Absolute performance still behind prior character-based DLMs**

   * Despite strong engineering, the study shows subword-based DLMs yield **smaller improvements** than character models .

3. **Some findings are inconclusive**

   * E.g., augmentation impacts vary and sometimes conflict with prior reports (YourTTS combination not helpful) .

4. **High computational cost vs. performance gain**

   * Training a full TTS+ASR+DLM stack for modest improvements over strong LM rescoring may limit practical adoption.

5. **Underdeveloped theoretical interpretation**

   * Paper recognizes hypothesis-space richness matters but does not establish principled understanding or predictive metrics.

### Questions
1. **Why do character-based DLMs outperform subword ones?**

   * The paper suggests granularity/hypothesis-space richness, but controlled experiments with equal ASR baselines would help isolate the cause.

2. **Can DLM-sum scale to large ASR n-best lists or streaming ASR?**

   * Complexity and latency considerations for real-world deployment are not discussed.

Optional further question:

Have you evaluated model robustness on out-of-domain speech or conversational datasets?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a dataset and method to post-edit ASR output with a denoising non-autoregressive language model.

### Strengths
The paper contains a lot of detail about data generation methods and details of the proposed method to post-edit output from an ASR model. The method shows small but significant gains on a dataset where the WER error rate is already low (~2 on "clean" subset).

The dataset will be useful for follow-up work.

Overall, the paper is very well written and filled with interesting details and observation.

### Weaknesses
The title implies that this replaces autoregressive language models, however, it does only operate on the output of a ASR model that typically (and in the case of the experiments) operates auto-regressively. The dataset does not link back to the acoustic features, so this is only post-editing.

There is a lot of content off-loaded into the appendix, including proper descriptions of the individual steps in the method.

A more detailed error analysis would have been nice - how many changes from good -> bad, bad -> good, bad->still bad.

### Questions
n/a

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper is a large-scale empirical study of denoising language models (DLMs) for automatic speech recognition (ASR), an approach recently introduced by Gu et al. 2024 that uses a conditional language model to correct ASR outputs.  DLMs aim to convert errorful ASR outputs to clean outputs, and are learned using training data consisting of paired (noisy, clean) transcripts.  The paper contributes a DLM codebase, evaluates DLM performance while varying a number of design choices, and introduces a new technique (DLM-sum) for decoding from multiple ASR hypotheses.  The experiments are conducted on the LibriSpeech dataset, training on the standard 960-hour training set and evaluating on the clean and noisy dev and test sets.  The main findings are that DLMs outperform decoding with a traditional LM, but less so than claimed in Gu et al. 2024, and that the proposed DLM-sum decoding approach outperforms the DSR technique of Gu et al. 2024.  A large number of additional experiments are provided via appendices.

### Strengths
- The paper provides an impressively large range of experiments with DLMs under controlled conditions, making the findings reliable.

- The provided open-source pipeline should make it possible for others to reproduce the work (I haven't tried the code myself).

- The new DLM-sum technique is a sensible way of taking better advantage of ASR n-best lists.

### Weaknesses
- Experiments on LibriSpeech alone limit the impact of the contribution.  LibriSpeech is extremely clean, dictated speech with very low ASR errors.  In order to make a convincing case, the experiments should include additional, more challenging datasets.  Other recent studies of LM-based ASR error correction, e.g. Ma et al. 2025 (cited in the paper), include several datasets.

- The length and organization of the paper makes it a tough read.  Most of the results, and even some of the basic notation, are given in the 60 pages of appendices.  The results in the appendices are just described qualitatively in the main text, leaving the reader to either trust the conclusions or else read through the dozens of pages of appendices.  ASR-specific concepts (e.g., CTC) are used without definition, which I believe would make this an even tougher read for the general ICLR community.  The large number of experiments do make it difficult to present the work in an ICLR-sized paper, though I think it could have been done at least somewhat better.  Overall, this makes the paper a better fit for a speech-oriented journal than ICLR.

- State-of-the-art results are claimed, but no results from prior work are included for comparison in the tables in the main text (I have not checked all of the appendices).

### Questions
- I wonder how a simple ROVER n-best rescoring baseline would do (Fiscus, "A post-processing system to yield reduced word error rates: Recognizer Output Voting Error Reduction (ROVER)", ASRU 1997).  This seems like a relevant baseline if you are using n-best lists as input to the LM.  Have you tried this?  Or has it been shown in prior work to be dominated by your other baselines?  If the latter is the case, then it would be good to state this in the intro/related work and give a citation.

- When greedy decoding is introduced (Eq. (1) and (2)), it is not clear to me why both \hat{a} and \tilde{a} are needed.  I was expecting Eq. (1) with the conditioning variable being \tilde{a}, which in turn corresponds to the arg max of p_ASR.  Could you please explain if I am missing something?

- The text after Eq. (2), "Equation (2) is approximated with framewise argmax followed by ..." appears to assume that the ASR model is CTC-based, but this is not stated until much later.  It would be good to state it here and cite and define CTC briefly.

- Was DSR introduced by Gu et al. 2024?  This is implied but is not clearly stated.

- It would be good to define (and ideally give a citation for) label-synchronous search and time-synchronous search.

- There are too many results reported on test sets.  Ideally all of the ablations and hyperparamer tuning experiments would be done on dev sets.

- Can you say how the vocabulary sizes were chosen?

- Can you define \hat{=}?  Why is it used (and not a regular "=") in 3.1?

### Soundness
3

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
3

### Summary
- The paper presents a comprehensive analysis of denoising language models (DLMs) as text-based error correction model for ASR from multiple perspectives, such as decoding strategies, data augmentation methods and comparison with traditional decoder-only LM.

- The paper also presents a new decoding strategy referred to as DLM-sum, which leverages the combination of several ASR hypotheses as condition. The effectiveness of DLM-sum is demonstrated through superior WER results on the standard LibriSpeech corpus for ASR.

### Strengths
- The paper provides empirical evidence through detailed and systematic experiments on DLMs for ASR, offering useful insights for improving current approaches.
- The proposed DLM-sum decoding appears to be a simple yet effective decoding solution.
- The paper is clearly written and well-organized.

### Weaknesses
- The paper lacks substantial novelty. The proposed DLM-sum decoding method appears to be an incremental extension to current methods.
- All analysis is conducted on a single ASR system, which may limit understanding of how the findings about the DLMs generalize across different ASR architectures.

### Questions
- The limited novelty of the proposed decoding method, combined with the use of only a single ASR system for the DLM analysis, reduces the strength of the paper's overall contribution. Based on the current version, I would lean toward rejection. However, please correct me if I am mistaken, as I am open to revisiting my assessment.
- Could the authors clarify the rationale for selecting Glow-TTS and YourTTS as the TTS systems, given that they are relatively older models as compared to newer models such as CosyVoice 2?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper investigates denoising language models (DLMs) for ASR error correction on LibriSpeech, aiming to reproduce and extend recent DLM work. The authors build a full LibriSpeech-only pipeline (ASR, TTS, LM/DLM, decoding) with subword vocabularies and systematically compare a large set of DLM training data generation schemes (various TTS settings, ASR sampling/checkpoints, token-level augmentations). They propose a new decoding method, DLM-sum, which approximates marginalization over an ASR n-best list, and also explore a dense k-probability input representation that feeds top-k ASR posteriors into the DLM. Experiments compare DLMs against standard LMs and different decoding strategies, reporting modest WER improvements and several empirical observations about when DLMs help.

### Strengths
The paper’s main strengths lie in its systematic empirical study and carefully engineered pipeline, with some degree of originality in how existing ideas are combined.

Originality
- Combines denoising LMs with an n-best–style decoding scheme (DLM-sum) and a dense k-probability input, leading to a coherent configuration that has not been extensively studied in prior work.

Quality
- Substantial experimental effort: an end-to-end LibriSpeech-only pipeline (ASR, TTS, LM/DLM, multiple decoding schemes) with many controlled ablations on augmentation and decoding strategies.
- Comparisons are internally consistent because all components are trained and evaluated under the same data constraints.

Clarity
- Core probabilistic formulations (DLM, DSR, DLM-sum) are clearly specified, and implementation details in the appendices are sufficient to understand and reproduce the experimental setup.

Significance
- Provides practical takeaways for ASR error correction (e.g., favoring moderate denoising and using richer ASR information via n-best lists or dense posteriors).
- The planned release of code and configurations makes the work a potentially useful reference implementation for this class of DLM-based ASR error-correction systems.

### Weaknesses
The main weaknesses concern novelty, positioning, and how sharp the empirical story is.

Conceptual novelty may be limited. 
- The main ideas (n-best marginalization in DLM-sum, log-linear score combination, using top-k posteriors as input) appear closely related to standard ASR practices such as n-best/lattice rescoring and confusion-network–style combination. At present, the paper does not fully clarify what is conceptually new beyond instantiating these ideas with a DLM, or how this perspective leads to qualitatively different behavior.

Reproduction framing could be more precise. 
- The work is framed as “reproducing and dissecting” prior DLM results, but the experimental setup differs in several important aspects (vocabulary, model architecture, TTS data, training regime). This makes the comparison to earlier work less direct, and some of the explanations for the performance gap feel somewhat speculative rather than supported by tightly controlled experiments.

Narrow evaluation and modest gains. 
- All experiments are on LibriSpeech, with no tests on more challenging or different domains. The improvements over strong LM baselines are small, yet require a significantly more complex pipeline (TTS + synthetic data + DLM), and there is no cost/efficiency discussion or comparison to strong modern error-correction baselines (e.g., LLM-style correctors).

Tech-report feel / weakly distilled insights. 
- The paper contains many ablations and appendices, but the key lessons are not distilled into a small set of clear takeaways. Some central hypotheses (e.g., about vocabulary effects) are not tested with tightly controlled experiments, which makes the overall message less convincing than the amount of work would suggest.

### Questions
1. Overall goal / positioning.
How do you position this work: primarily as a reproduction and diagnosis of Gu et al. (2024), or as a new subword-based DLM/decoding design? It would help if you could clearly separate which parts are intended as faithful reproductions vs deliberate deviations, and adjust the claims accordingly.

2. Novelty of DLM-sum vs standard n-best / lattice methods.
Conceptually, how is DLM-sum different from classical n-best/lattice rescoring and confusion-network–style log-linear combination? Can you provide (or add) a direct comparison to a simpler “ASR n-best + DLM rescoring” baseline under matched conditions, to better isolate what DLM-sum is adding?

3. Missing modern error-correction baselines.
Could you elaborate on the reasons (e.g., compute, policy, or data constraints) for not including any LLM-style or seq2seq error-correction baselines? If feasible within your constraints, would it be possible to add at least one reasonably strong corrector trained under the same LibriSpeech-only setting, so that readers can better judge practical competitiveness?

4. Significance vs complexity / cost.
The WER gains of DLM-sum over the best LM setup appear small relative to the added complexity (TTS + synthetic data + DLM training + more complex decoding). Can you quantify training and inference cost, and clarify whether you view the main contribution primarily as a practical recipe or mainly as a scientific/diagnostic study?

### Soundness
3

### Presentation
3

### Contribution
2
