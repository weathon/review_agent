# SpecRA: Monitor Degenerative Repetition in LLM Agents using Randomized FFT

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
LLM-based agents also suffer from "degenerative repetition" like chatbots, which leads to task failure and results in significant waste of computational resources and API costs until token limit is reached. Existing methods require modification of training process or customization of model deployment, and detection algorithms are brittle to approximate or structural recurrence. We therefore introduce SpecRA, a simple yet effective algorithm for detection of self-repetitions in text. Via a randomized projection from the large LLM vocabulary onto a unit-norm complex sequence, our method leverages the power of the Fast Fourier Transform (FFT) to compute the sequence's autocorrelation. Peaks in the autocorrelation function robustly reveal the underlying periodicity of the content, with tolerance to minor variations. Through an analysis of 813 repetitive samples identified from 1.13M records of anonymized agent outputs, we build a taxonomy of repetition modes in agents and show that SpecRA offers a lightweight, non-intrusive mechanism for constructing more reliable and cost-efficient LLM agents accross both standard open-source model deployments and proprietary models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper addresses the issue of degenerative repetition in large language model (LLM)-based agents, where models get stuck in repetitive cycles, leading to wasted computational resources and increased API costs. The authors propose SpecRA, a fast, spectral method for detecting self-repetitions in LLM outputs. By leveraging the Fast Fourier Transform (FFT) and a randomized token projection, the method analyzes the periodicity of token sequences and identifies when repetition occurs. The paper presents a detailed taxonomy of repetition modes and demonstrates that SpecRA can reliably detect and classify repetitions across a range of LLM outputs. The authors argue that their method provides an efficient, non-intrusive mechanism to reduce computational waste and improve agent reliability.

### Strengths
The idea of applying spectral analysis to detect degenerative repetition in LLM outputs is novel. SpecRA offers a unique approach compared to existing penalty-based methods by focusing on post-generation detection through signal processing, providing a lightweight and non-intrusive solution.

### Weaknesses
1. Problem Importance: The problem of degenerative repetition is important, but the paper does not convincingly argue why it is a significant issue that warrants a completely new approach. Simple methods like repetition penalty or temperature adjustments can mitigate repetition without much loss in performance, and these are commonly used in practice. Also for long repetitive responses, techniques like early truncation could resolve these issues in a more cost-efficient manner, and the paper doesn’t explore these simpler alternatives in detail. While SpecRA is interesting, it’s unclear if the problem is as pressing as suggested.
2. Threshold Management: SpecRA requires setting a threshold for repetition detection, which can be cumbersome. The paper mentions that this threshold must be carefully adjusted for different vocabularies, which might require reconfiguration for each model. This adds a layer of complexity that could make the approach harder to deploy in production.

### Questions
Why is degenerative repetition such a critical issue that requires a new solution like SpecRA? Can existing solutions like repetition penalty or temperature adjustments not achieve similar results without introducing the complexity of SpecRA?

### Soundness
3

### Presentation
2

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
This paper introduces SpecRA, a new algorithm to detect degenerative repetition in LLM agents. It works by projecting the token stream onto a random complex-valued sequence and then using the FFT to efficiently compute its autocorrelation.

### Strengths
- The method's primary strengths are its efficiency, and its robustness to the approximate repetitions that break traditional exact-match detectors.
- The paper is very well motivated and is easy to read. The paper also openly mentions the limitations of the method, which is highly appreciated.

### Weaknesses
- In L195, the authors claim that the randomized projection into the complex plane makes the detector "robust to lexical variation" while preserving the overall periodic structure. Can the authors explain the connection between this projection and improved robustness to lexical variation?
- The decision to utilize only the real component of $R_l$ should be explicitly justified. While I could infer the motivation behind it, the exclusion of the imaginary component is not theoretically grounded in the text. A brief explanation would improve the conceptual clarity of the proposed method.
- The motivation behind the randomized projection step remains unclear. Mapping a vocabulary space of approximately 200K tokens into a 360$\degree$ complex plane appears arbitrary and potentially lossy. The authors mention this limitation briefly, but it would be more compelling to compare this approach with projection strategies, such as those based on embedding spaces or a higher-dimensional latent space. 
- The experimental results are interesting but lack comparative context. Without evaluation against baseline models, for instance, n-gram based detectors or other lexical similarity measures, it is difficult to assess the effectiveness of SpecRA.

### Questions
All the questions and suggestions have been listed in the weaknesses.

While I do appreciate the authors addressing several limitations that I thought of while reading the paper (high FPR, failure in insertions / deletions & more), SpecRA, by itself, is a tool that would not be very useful realistically. Regardless, I would push this paper towards acceptance - based on the author's responses & for the theoretical insights and formulations provided by the authors.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces SpecRA, a spectral-based algorithm for detecting degenerative repetition in LLM-based agents, where models become trapped in recursive loops generating near-identical sequences that cause task failure and computational waste. The method works by projecting each token to a unit-magnitude complex number via randomized phase assignment, then computing autocorrelation using FFT via the Wiener-Khinchin theorem to efficiently identify periodic patterns with O(W log W) complexity. The paper also provides theoretical guarantees including false-positive bounds and detection efficacy under $\epsilon$-mismatch conditions (Theorem 1), demonstrates robustness to substitutions, insertions, and deletions through synthetic experiments, and analyzes repetitive samples from >1M agent traces to build a taxonomy of four failure modes such as structural repetition and syntactic degradation. The authors recommend practical detection thresholds for different use cases.

### Strengths
+ Novel and theoretically grounded approach: The paper presents a creative solution by recasting text repetition detection as a signal processing problem. The use of randomized phase projection with FFT-based autocorrelation is elegant and well-motivated.

+ Strong theoretical foundation: Provides rigorous probabilistic guarantees through Lemma 1 (false-positive bounds using Hoeffding's inequality) and Theorem 1 (detection power under approximate periodicity), making the approach principled rather than heuristic.

+ Practical guidance: Provides actionable threshold recommendations based on empirical percentiles from diverse corpora, facilitating real-world adoption.

+ Clear presentation: The paper is well-written with good motivation, clear problem formulation, and effective visualizations (especially Figure 1).

+ Comprehensive empirical validation: The paper includes synthetic experiments (Section 6.1), real-world corpus analysis across multiple domains (Wikipedia, GitHub, agent traces), and builds a valuable taxonomy of repetition failure modes.

### Weaknesses
+ Missing baseline comparisons: The empirical analysis (Section 6) lacks comparisons with existing detection methods. Even if exact string matching and edit-distance methods have limitations, quantitative comparisons on the same test sets would strengthen claims about SpecRA's advantages in accuracy, speed, and robustness.


+ Taxonomy validation: The classification of 549 repetitive samples into four categories appears to be manual. The paper lacks details on: (a) inter-annotator agreement if multiple annotators were used, (b) whether the 264 excluded samples introduce selection bias, (c) validation that categories are mutually exclusive and comprehensive.

+ Real-world deployment details: The paper mentions "1.13M records of anonymized agent outputs" but provides minimal context about: the agents' tasks, which models were used, what triggered the repetitions, and how representative this dataset is.

### Questions
+ Baseline performance: Can you provide quantitative comparisons against n-gram overlap, suffix trees, or approximate string matching algorithms (e.g., using sliding windows with edit distance) on your test sets? Even if they're slower, understanding the accuracy trade-offs would be valuable.

+ False negative analysis: What is the false negative rate of SpecRA on your agent trace dataset? Are there patterns that SpecRA consistently misses, and what characterizes them?

+ Interaction with decoding parameters: How does SpecRA's detection rate vary with temperature, top-p, or other sampling parameters? Do higher temperatures reduce repetition occurrence or just change repetition patterns?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
In this paper, the authors claim to introduce a new approach for detecting repetition degeneration in texts generated by LLMs, based on the discrete Fast Fourier Transform (FFT). The authors claim that the FFT is more scalable and robust for natural language repetitions than existing n-gram (k-mer)- and edit distance-based approaches. Authors provide proof that their method has a controllable false positive rate and exhibits exponentially improved performance with increasing attention frame size.In this papers the authors claim to introduce a new approach for the detection if repetition degeneration of texts generated by LLMs based on the discrete Fast Fourier Transform (FFT). The authors claim that FFT is more scalable and robust for natural language repetitions than the existing n-gram (k-mer) and edit distance- based approaches. Authors provide a proof that their method has a controllable false positive rate and exponentially improved performance with the increase in the attention frame size.

### Strengths
The detection of repetition degeneration in LLM output is a critical step in both its day-to-day usage, as highlighted by the authors, as well as in model training to suppress unwanted repetition at the Reinforcement Learning stage. The overall approach undertaken by the authors is sound and likely applicable in practice. Additionally, authors introduce an additional criterion for performance evaluation- "timely detection", which is relevant in the context of repetition detection but not a statistical criterion commonly used in other settings.

### Weaknesses
- While the overall citations are well-selected and consistent with the text, authors repeatedly cite (Holtzman et al., 2019) as the paper introducing or using the repetition penalty (eg, L040-047). However, this paper focused only on nucleus sampling, addressing the repetition text through a temperature-based sampling. It does not use repetition penalties.

- The authors do not sufficiently justify the need for their method. For instance, the k-mer sliding approaches that authors discard as insufficiently scalable (L091-097) can be applied to UTF-8 encodings of the character n-grams, effectively reducing the vocabulary back to 16 characters, making direct bioinformatics k-mer approaches computationally feasible again. Similarly, it is unclear how frequently the partial repetition problem is used to justify their approach. 

- The authors do not seem to evaluate the computational performance of the proposed method, one of its advantages compared to existing methods, as claimed by the authors.

- Finally, the contribution of the paper is hard to identify and seems minor at first glance. Teglanceepetition analysis has been commonly cited as an application of the FFT to texts (e.g., https://math.stackexchange.com/questions/422948/fourier-transform-of-text; https://cp-algorithms.com/algebra/fft.html). The author's addition appears to be the random embedding of text in the complex plane before applying the FTT, but it is unclear why this step is important, except that it makes the proofs of Lemma 1 and Theorem 1 simpler. I am not sure, however, that such a contribution could justify acceptance to a conference of the notoriety of ICLR. 


Minor comment: In Lemmas 1 and Theorem 1, tau is commonly used to denote time or position index in frequency analysis and integration; using it as a threshold may be confusing to readers from that background.

### Questions
- How frequent are the real-world scenarios in which the repetition degeneration occurs with minor variations, as suggested on L044-047? Most of the literature and reports on the topic suggest exact repetitions as the dominant degeneration mode.

- Is it possible to switch from a custom embedding of the tokens to the embedding provided by the model tokenizer? This is likely to lead to better performance by removing the need for a separate embedding step.

- Could you please clarify what you mean by "constant energy" on L049?

### Soundness
2

### Presentation
3

### Contribution
2
