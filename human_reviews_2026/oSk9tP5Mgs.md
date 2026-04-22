# FAFO: Lossy KV Cache Compression for Lossless Inference Acceleration via Draftless Fumble Decoding

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 4, 6, 6

## Abstract
Lossy KV cache compression is a well-explored subfield of machine learning efficiency, with improved latency being one of its major gains. However, lossy compression techniques can fumble from time to time, exhibiting various — and often catastrophic — failure patterns that are not only difficult to resolve but sometimes even hard to identify in the first place, making the direct deployment of models with compressed KV cache a risky endeavor. In this work, we explore a way to preserve lossless generation quality while still benefiting from the acceleration provided by attending only to a compressed KV cache. Specifically, we draw inspiration from the n-gram candidate pool decoding paradigm pioneered by Lookahead Decoding — a largely overlooked and underdeveloped way to achieve efficient yet lossless decoding — where we purposely allow the model to Fumble Around with compressed KV cache to generate multiple lossy ''n-gram guesses'' with just one forward pass, while Find Out via lossless verification in the same forward pass in truly parallel fashion. From a conceptual standpoint, our proposed framework is compatible with all typical static or dynamic KV cache compression methods from the token dropping realm, thus opening up a new avenue for the stagnant n-gram decoding paradigm. Practically, we show that — with careful system support — this framework presents many useful traits that similar draftless baselines (e.g., Self-Speculative Decoding) simply cannot achieve, such as requiring only one set of KV cache and being far less sensitive to model, task, and input-length scenarios. Our comprehensive empirical results show FAFO provides 1.20-2.71x latency speedup over the original model, while consistently outperforming other lossless + draftless solutions by a large margin.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes FAFO (Fumble Around and Find Out), a lossless LLM inference acceleration framework based on combining KV-cache compression with lookahead decoding. The KV-cache compression method used in FAFO is StreamingLLM or Quest, while draftless fumble decoding is presented to realize lookahead decoding. Empirical results show a latency speedup of 1.20-2.71x on a set of Llama and Qwen LLMs.

### Strengths
1. The idea of combining KV-cache compression with lookahead decoding (draftless fumble decoding) looks novel and interesting.
2. A latency speedup of 1.20-2.71x over the original model is significant.

### Weaknesses
1. This paper is poorly structured and written, very verbose, and difficult to follow. The writing quality hinders fair evaluation of the technical content. I suggest the authors completely rewrite and resubmit the paper -- the technical content and contribution could not be fairly evaluated in the current shape.
2. A latency speedup of 1.20-2.71x over the original model is significant. However, if the "Lookahead" case is considered as the baseline, the speedup will be much less. In addition, it is unclear how much of the speedup comes from FlexAttention -- this should not be regarded as FAFO's contribution.

### Questions
Please see "Weaknesses" for my major comments and suggestions.
1. From Weakness 1: The very first issue of this paper is the writing. Please fix it.
2. From Weakness 2: Should the baseline be the original model, or the "Lookahead" case?
3. From Weakness 2: How much of the speedup comes from FlexAttention?
4. Lack of ablation studies on lookback window size
5. Missing related work: LongSpec

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
Authors present FAFO, an n-gram candidate-based decoding method for efficient token generation while also preserving full model quality. The Fumble Around step generates multiple n-gram candidate guesses using a compressed KV cache while the Find Out step verifies the candidates conditioned on a set of tokens. Both steps run in parallel, achieving draftless decoding. The work also includes customized cache managers built on FlexAttention kernels.

### Strengths
- The problem and proposed approach are very timely and needed in the current scenario. The ideas and implementations allow us to realize the benefits of n-gram candidate-based decoding methods in a practical setting.
- Authors present a complete engineering solution, implementing a custom KV cache manager with a smart memory layout (well explained in appendix) designed to work with a sparse attention kernel (FlexAttention). The system-level contributions, especially the fixed-size KV block design and swapping mechanism (Appendix G), address real implementation challenges that have prevented prior work from achieving practical speedups.
- Comprehensive empirical evaluation across multiple models (Llama-2, Llama-3, Llama-3.1, Qwen2.5), tasks, and settings. The experiments demonstrate consistent improvements and robustness across scenarios where baselines struggle (e.g., MT-Bench, long-context tasks).

### Weaknesses
**The Motivation and Introduction of the work feels all over the place**

- The title says, "Lossy KV Cache compression" then "Lossless Inference". Line 46, “Lossless KV cache compression framework". The abstract emphasizes “..lossy compression techniques can fumble..”.
  - Throughout the paper, the term *loss* conflates two concepts: (1) using lossy compression methods as a component within the system, and (2) the end-to-end generation quality being lossless. The introduction should clearly establish how FAFO uses lossy KV cache compression for candidate generation and maintains lossless output quality through verification similar to speculative decoding. 
- Section 1.1 needlessly spends significant space (lines 72-101) articulating the lossy nature of lossy compression methods. While motivation is important, this is tangential to FAFO since it does not solve the compression problem. The key idea, “lossy methods can generate useful candidates even if they're not reliable for end-to-end generation” can be explained much more concisely.
- Section 1.2 first motivates how KV cache compression helps the already established SD paradigm by giving rise to SSD methods followed by their drawbacks. Authors then abruptly transition to “lossless efficient decoding channels” with n-gram candidate pool decoding established by Lookahead decoding. This is a jarring transition that leaves several questions unanswered:
  - Do SSD methods exist that use token-dropping KV cache compression? If yes, please discuss how (if any) they solve general drawbacks of SSD.
  - Does there already exist an efficient implementation that realizes the gains from above? Does Table 1 use it? 
- Overall the limitations of SSD+compression approaches should be established **before** introducing n-gram methods as an alternative.
- Furthermore, the distinction between n-gram methods and SD as different paradigms (Line 199: "..given the parallel draft-and-verify vs the sequential draft-then-verify difference..") is crucial but introduced too late. Table 1 underscores the effectiveness of the n-gram candidates over SSD, but by this point we are already beyond introduction. Kindly make it clear at the start.
- From my understanding, A possible reframing: 
  - Lossy compression enables efficient candidate generation but cannot be used end-to-end
  - Self-SD addresses this end-to-end by using compression for drafting + full cache for verification but faces memory and efficiency issues.
  - N-gram candidate pool decoding offers an alternative paradigm that can overcome these limitations through parallel verification.
  - However, integrating compression with n-gram methods is non-trivial and hasn’t been done before.



**Unclear Positioning relative to existing solutions and possibly overstated claims**

- Now, as the n-gram based methods are in a different paradigm than SSD, all the benefits provided by FAFO should **primarily** be compared with Lookahead decoding rather than with SSD methods. The fundamental properties, lossless output, single KV cache, parallel verification, etc follow directly from the “draft-and-verify vs draft-then-verify” paradigm rather than FAFO’s specific contributions. **Authors should clarify this** and avoid overstating claims. 
  - Line 157, “..FAFO is the only approach capable of delivering such a trifecta[lossless quality, latency improvements, memory footprint] other than Lookahead Decoding..”  This is a bit misleading since these properties are inherent to the n-gram paradigm(and hence Lookahead decoding) and not unique contributions of FAFO.
  - The extensive comparison with TriForce is helpful for showing practical superiority but the paper should be clearer that these advantages come from the paradigm rather than algorithmic innovations.
- A clearer framing would be expanding the contribution point 4, where authors correctly identify that this work unlocks the capabilities from a stagnant paradigm by making n-gram based decoding practical and effective rather than saying “we solve SSD’s problems by using n-gram methods”


**Limited Conceptual Novelty beyond existing Paradigms**

- As noted in the previous points, conceptually the additional improvements to Lookahead decoding include:
  - Fumble Decoding (Using compressed KV-cache to generate candidate n-grams):  While this is extremely non-trivial from an implementation standpoint, conceptually it does not stand out. The key idea that compressed caches could generate useful candidates has been explored in previous SSD contexts. 
  - Find out Verification (Retrieving candidates based on a longer suffix of tokens): This key innovation (Section 4.3) alone does not represent substantial conceptual advancement.

From a research perspective these ideas do not contribute substantially to existing work. 
The real achievements (as noted in Strengths) are in engineering: Making FlexAttention work for dynamic decoding through fixed size block allocation, The swapping mechanism, Custom KV Cache manager. While these are certainly non-trivial accomplishments, the work would greatly benefit from either, (a) identifying additional algorithmic or theoretical insights beyond, “use compression in the n-gram paradigm” or (b) reframing as primarily systems contribution and submitting to an appropriate venue. As currently positioned, the conceptual contribution feels thin despite the solid engineering work.

### Questions
See Weaknesses

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes FAFO (Fumble Around, Find Out), a speculative decoding method that integrates lossy KV-cache compression and verification within a single forward pass.
Instead of relying on a separate draft model, FAFO constructs an n-gram (typically 2-gram) candidate pool that predicts likely next tokens from the compressed KV cache.
Verification with the full KV cache occurs in parallel, enabled by a masked-attention design that allows both “fumble” (draft) and “find-out” (verify) computations in one pass.

### Strengths
+ **Interesting idea:** FAFO’s capability to perform n-gram drafting using a compressed KV cache and verify the n-gram drafts within the same forward pass is novel and technically elegant.

+ **Clear presentation:** The paper clearly contrasts FAFO with prior speculative decoding methods (TriForce, Lookahead, Self-SD). Figure 2 effectively illustrates the mechanism, making the concept easy to grasp.

### Weaknesses
+ **Limited workload consideration:** Experiments are limited to batch = 1. It remains unclear whether FAFO’s advantages persist under larger-batch or multi-sequence settings, where system-level bottlenecks may differ.

+ **Scalability concern:** FAFO introduces extra computation per decoding step. This additional work could shift the decoding regime from memory-bound to compute-bound, potentially diminishing speedups on larger batches or lower-end GPUs.

### Questions
+ **Scalability under larger batches:**
Have you evaluated FAFO with larger batch sizes or multi-sequence decoding? Since FAFO adds per-step computation and coordination, it would be helpful to understand whether the speedup persists as batch size increases or on GPUs with limited compute resources.

+ **Computation vs. memory trade-off:**
Can you quantify how much additional FLOPs FAFO introduces per decoding step compared to a standard n-gram (lookahead) or speculative decoding baseline? This would clarify when the approach transitions from memory-bound to compute-bound.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper makes a solid technical contribution to the speculative decoding domain. By unifying KV-cache compression with n-gram parallel verification under a single-KV, draftless design, FAFO advances the state of the art in both efficiency and practicality, especially for memory-constrained deployments such as local or on-device LLM inference. The work is conceptually well-motivated—aiming to reduce memory usage, preserve losslessness, and improve latency—and is empirically validated with consistent 1.2–2.7× speedups across multiple models and tasks.

However, the presentation and structure significantly weaken the paper’s impact. The writing often lacks focus, devotes excessive space to background discussion, and scatters key insights (particularly in Sections 1–3). Substantial portions are spent re-explaining the failure modes of lossy KV-cache compression and speculative decoding, even though the paper’s main contribution is not about memory savings from compression. Meanwhile, the core innovations—the parallel fumble–verify mechanism, KV-block management, and FlexAttention integration—are buried deep in the text and described with unnecessary verbosity. Overall, the paper’s technical clarity and organization fall short of top-tier standards. In my view, the framework would benefit from a more focused exposition that devotes space to the design rationale and engineering challenges behind its proposed solutions.

### Strengths
* Practical improvement to speculative decoding: single KV cache, no auxiliary draft model.

* Strong empirical results with robust gains across tasks and models.

* Careful engineering insight on attention sparsity and block-mask reuse.

### Weaknesses
* **Poor organization and writing quality:** The first 5–6 pages mix motivation, background, and criticism without a clear focus. The manuscript contains substantial redundancy; large portions of Section 1 repeat well-known discussions on the failure modes of lossy KV-cache compression. Due to limited high-level clarity, readers must piece together the core contributions from scattered descriptions, figures, and appendices.

* **Limited novelty in algorithmic ideas:** The core algorithmic concept is not particularly new — it essentially combines token sampling or KV-cache compression techniques with n-gram (Lookahead) decoding. The contribution lies more in integration than in conceptual innovation.

* **Insufficient explanation of system-level contributions:** While much of the technical advancement appears to stem from the system integration — such as the custom KV-cache manager and FlexAttention-based implementation — these components are described only at a high level, without sufficient technical depth or analysis to fully support the claimed efficiency improvements.

### Questions
* By default, FAFO offloads speculative subsequences to CPU memory and later reloads them to GPU memory. How is this offloading process managed in practice? Does it introduce noticeable latency or transfer overhead? Additionally, can FAFO operate in a fully GPU-resident (non-offloading) mode, and if so, how would that affect its performance and memory footprint?

* The evaluation only reports results with a batch size of 1. How does FAFO perform under larger batch sizes or even with a multi-GPU system? In particular, since the TriForce baseline shows improved throughput with batched inference, how does FAFO scale in comparison?

### Soundness
3

### Presentation
1

### Contribution
3
