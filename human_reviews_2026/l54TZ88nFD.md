# WaveletGPT: Wavelet Inspired LLMs

- Avg Score: 2.50
- Decision: Reject
- Scores: 2, 0, 4, 4

## Abstract
Large Language Models (LLMs) have ushered in a new wave of artificial intelligence advancements impacting every scientific field and discipline. We live in a world where most of the data around us, e.g., text, audio, and music, has a multi-scale structure. This paper infuses LLMs with a traditional signal processing idea, namely wavelets, during pre-training to take advantage of the structure. Without adding any extra parameters}to a GPT-style LLM architecture in an academic setup, we achieve the same pre-training performance almost twice as fast in text, audio, and images, by imposing a structure on intermediate embeddings. When trained for the same number of training steps, we achieve significant gains, comparable to pre-training a larger neural architecture. Further, we show this extends to the Long Range Arena benchmark and several input representations such as characters, BPE tokens, bytes, waveform, math expression, image pixels. Our architecture allows every next token prediction access to intermediate embeddings at different temporal resolutions in every decoder block. We hope this will pave the way for incorporating multi-rate signal processing instead of going after scale.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
In this paper, the author incorporate wavelets into GPT-style LLM pretraining by imposing a multiscale structure on intermediate embeddings without adding parameters. They match pretraining quality almost twice as fast on text, audio, and images and, at equal training steps, obtain gains comparable to training a larger model. They also show effectiveness on Long Range Arena and on inputs including characters, BPE, bytes, waveform, math expressions, and pixels, suggesting multi-rate signal processing as a practical alternative to scaling.

### Strengths
The idea of incorporating wavelets into GPT may be useful.

### Weaknesses
**(1) Outdated and selective baselines.** The abstract claims that “we achieve significant gains, comparable to pre-training a larger neural architecture,” yet most comparisons are against models from three to four years ago. To substantiate the claim, the paper should include strong, up-to-date baselines and parameter-matched ablations with recent open models (e.g., Llama, Qwen, Phi families). As indicated by the title, this is an LLM paper, so comparison with these models—which are becoming the baseline for LLM—is essential. Without such comparisons, the central claim is not convincingly supported.

They describe it as follows in their paper.
>The goal in all four modalities is not to chase state-of-the-art pre-training performance, as this
paper was written in an academic setting with very few computational resources. (L103,104)

Given that the work explicitly targets efficiency at small scale with limited academic compute, the method does not qualify as **Large**LM pre-training in the contemporary sense. The current title/abstract/intro oversell the scope. The paper should adopt small-scale GPT language throughout and revise claims to reflect the actual experimental setting.
Although the paper frames the method as “LLM pre-training,” the actual setup is a shrunk-down GPT: 10 decoder layers with d_model = 128, FFN = 512, 8 heads, context length 512, trained from scratch on small/modest datasets (e.g., text8, MAESTRO, LibriSpeech tokens, YouTube-Mix-8) with 1 M training points ; the authors explicitly state they do not compare against larger architectures and that the work was conducted with very limited compute in academia. Accordingly, the experiments are small-scale and do not constitute LLM pre-training in the contemporary sense.

**(2) Insufficient discussion of prior work and limited novelty positioning.** There is a substantial body of research applying wavelets within Transformer-based models (e.g., for tokenization, attention, positional encodings, or hybrid frequency–time representations). The submission does not adequately survey this literature or clearly articulate its differences. Moreover, the method does not appear to fundamentally modify the GPT architecture to the extent implied by the name “WaveletGPT.” The paper should cite the relevant wavelet-Transformer works and provide a precise comparison that clarifies what is genuinely new.

> To the best of our knowledge, the paper’s contributions are: 1) We propose the first instance of incorporating wavelets into LLM pre-training. (L83,84)

The statement  is clearly erroneous. It is simply a result of the authors' insufficient survey. Even a quick search of ICLR papers already surfaces multiple works that integrate wavelets into Transformer components. For example, Adaptive Wavelet Transformer Network (ICLR 2022), which fuses lifting-scheme wavelet subbands with Transformer blocks for 3D shape representation, and Wavelet-based Positional Representation for Long Context (ICLR 2025), which incorporate for Transformers. 
These examples alone show that “incorporating wavelets into LLM” is not unprecedented, so the novelty claim should be carefully repositioned.

**(3) Clarity and presentation issues.** Sections 3 and 4 are largely presented as single paragraphs, which makes the paper difficult to follow. I recommend restructuring with subsections, shorter paragraphs, and (where appropriate) bullet points to highlight key design choices, algorithms, and experimental settings. In addition, several citations are missing or are written as plain text rather than using proper citation commands (e.g., \cite{}), which does not meet typical ICLR formatting standards.

**(4) Incorrect and unreliable references/URLs.** Some references appear to be erroneous. For example, at L538 the entry “Fernando Flores-Mangas. Discrete waveelet transform” contains a spelling error (“waveelet”) and appears to misattribute or misidentify the source; I could not verify this citation. Please carefully audit all references and URLs for correctness and completeness. I checked the URL provided but was unable to access it.

>Fernando Flores-Mangas. Discrete waveelet transform. The Washington Post, Spring 2014. URL https://www.cs.toronto.edu/˜mangas/teaching/320/slides/CSC320L11.pdf. (L538)

It appears Fernando Flores-Mangas did not actually write the document titled "Discrete wavelet transform".  And he doesn't seem to be a researcher in wavelet transform.

https://scholar.google.com/citations?user=jJDpzNQAAAAJ&hl=en

While I can overlook minor typos in the main text, spelling mistakes in citations or references to non-existent papers are plainly unacceptable; such errors should not occur. In the main text, Flores-Mangas (2014) is cited in the Figure 2 caption. 
I knew about wavelet transforms, so I immediately noticed this citation was wrong. The author's name is clearly incorrect.
Did the author not check their own paper and fail to notice such a mistake?
**Given that the Discrete Wavelet Transform constitutes the paper’s core contribution, it is deeply concerning that its citation is wrong.**

**Furthermore, the authors mention Haar wavelet in their paper but fail to cite it in the references.** If they had properly studied wavelet transforms, such an error would be unthinkable. (To draw an analogy in LLM research, it would be like writing an LLM paper but either misquoting “Attention is All You Need” or failing to cite it altogether.)

**I remain concerned whether the author truly understands the mechanism and background of the wavelet transform.**

----
The idea may have merit, but the current submission has serious issues in experimental validation, scholarship, and exposition. Including modern baselines, thoroughly situating the work within prior research, correcting the bibliography, and improving the organization would be necessary before the paper can be considered for acceptance. Moreover, improper or inaccurate citations cast doubt on the authors’ reliability as researchers. As it stands, I cannot recommend acceptance.

### Questions
- Have you reviewed the ICLR Author Guidelines and used the official template for this submission? 
https://iclr.cc/Conferences/2026/AuthorGuide

- Have you read Ten Lectures on Wavelets by Baroness Ingrid Daubechies? It is a foundational text on wavelet theory, and I strongly recommend studying it first. 

- Have you also rread ICLR-accepted papers that include “wavelet” in the title? Using their paragraph structure, citation practices, and exposition as guides would improve the manuscript.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
2

### Summary
This paper proposes a novel idea to integrate traditional wavelet transformations in signal processing into the LLMs, which speeds up the LLM pre-training without adding extra parameters to the model.

### Strengths
In terms of originality, the authors designed a novel module called WaveletGPT that can be inserted between the decoder layers. The module modifies the intermediate embeddings using the wavelet transformations.

In terms of significance, the evaluations show that this new approach improves the convergence rate and the negative log likelihood (NLL) scores across multiple tasks.

### Weaknesses
I am not familiar with wavelets, but I believe the presentation of this paper is far below the ICLR standard. The presentation is so poor that I cannot understand the details of the paper.

1. Many symbols are not sufficiently defined or are hard to find their definition. For example, I do not know if the symbol $x[n]$ on Line 141 is a $n$-dimensional vector or a vector depending on the time $n$, as $n$ is defined (much later on Line 146) as time or the context length. The letters $g$ and $h$ on Line 150 are not defined.

2. Algorithm 1 is not clear. For example, I do not know why $i < E/2$ and $i \ge \frac{E}{2}$ are put on the right side and what the algorithm is trying to do. This is not the usual way to write pseudocode.

3. Figures are repeated and inconsistent. For example, Figures 1 and 2 have a lot of overlap. And in Figure 2, the levels in the left subplot are in Roman numerals, whereas the levels in the right subplot are in Arabic numerals. This numbering difference is also presented in the main text of the paper.

4. The evaluation results in Figure 5 (left) lack explanation. I do not know how the Base, Ours, SPE, Gain, and Time are actually computed, what they represent, and whether larger is better or smaller is better. And treating the table as a figure is also inappropriate.

5. There are misused math symbols. For example, on Line 140, $[0 - 128)$ should be $[0, 128)$. On Line 262, $h(.)$ should be $h (\cdot)$

6. The citations violate the ICLR formatting instructions. For example, on Line 39, "(Nix, 2024)" should be put without the parenthesis, and `\citet{}` should be used instead of `\citep{}`. On Line 300, "Yu et al. (2023)" should be put into the parenthesis with `\citep{}`. Also, the hyperlinks do not work at all. Please check Section 4.1 of the "Formatting Instructions for ICLR 2026 Conference Submissions" document in the provided LaTeX template.

7. The grammar mistakes and typos are so many that they severely affect the reading experience, e.g., Line 134 "theycan". I cannot list all of these typos, and I suggest the authors carefully proofread their paper and fix their language. 

8. The placeholder appendix is not removed from the template.

9. The sentences generally lack smooth and logical connections.

### Questions
What is an academic setup? What and how many GPUs are used? Do the conclusions generalize to non-academic setups?

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes WaveletGPT, a novel approach that combines wavelets into LLM architecture. They modify intermediate embeddings of the decoder blocks with wavelet coefficients, incorporating multi-scale structure. In order not to disrupt the embeddings, they keep the level of wavelet coefficients fixed for an embedding coordinate. The Language models incorporated with these modifications converge faster than the baseline unstructured LLM across multiple modalities.

### Strengths
1. They propose a novel way of using wavelets for pretraining language models
2. Empirical evaluations show that their model can achieve the performance of a GPT-like architecture around 40-85% faster
3. They have an option to train the wavelets and they show that this improves the model's performance.

### Weaknesses
1. The benefits of using Wavelets in Language models is studied from an empirical standpoint without theoretical formulation and analysis.
2. The experiments conducted were broad and on smaller-scale models without evidence of how the architecture scales to larger sizes
3. The downstream performance of the language models and the reasoning capabilities are not studied in the paper.
4. An ablation study on the wavelet kernel size function along the embedding dimension will help understand its impact better.
5. Does the idea work for image/video generation?

### Questions
Please see the weaknesses

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
The paper inserts a "causal multi-scale convolution/averaging" module grouped by embedding dimension between standard decoder layers: half of the dimensions retain the original resolution, while the other half are smoothed using a window that increases with dimension, allowing the same layer to see both fine-grained and coarse-grained context simultaneously. The authors pre-trained or modeled it on text, raw/symbolic music, acoustic tokens (LibriSpeech), and LRA, concluding that it converges to the same loss faster (approximately 40–65% epoch saving) with almost no increase in parameters.

### Strengths
1.The idea is simple and straightforward, easy to integrate into existing decoders, and requires no changes to the attention mechanism.

2.Multimodal experiments demonstrate that this inductive bias does not only affect one tokenization method.

3.The learnable convolutional kernel version provides a variant that is "not purely handmade Haar," indicating that the structure is learnable.

### Weaknesses
1. The model is clearly outdated, and the comparison scope is too narrow. The main experiments are still based on GPT-2, short contexts, early LM datasets, and small-scale audio sets, which cannot prove effectiveness for modern autoregressive models in 2025. There is a lack of comparison with modern LLM configurations. I understand the difficulties of experimentation under resource constraints, but if the goal is to shift the focus away from scaling, as claimed in the authors' abstract, it should at least adapt to some newer strategies and architectures, and demonstrate the potential for this method to achieve superior results when combined with scaling.
2.The current manuscript reads as a sequence of loosely connected experiments and clarifications. Sections are long and under-segmented; method, extensions, and dataset-specific details are interleaved. This makes the paper harder to follow than it needs to be.
3.Your current appendix section is a template. You need to delete it if you don't use it.
4.These are narrow downstream targets compared to what is typically used to assess LLMs (QA, world knowledge, etc.) in text domain. In current LLM practice, pre-training and post-/instruction-tuning are tightly coupled: the value of a pre-training modification is determined by whether it produces a better starting point for those downstream stages. A faster reduction of language-modeling loss on small or homogeneous corpora, even together with modest gains on FSD-50K or LRA, does not by itself establish that the resulting checkpoint will transfer better to mainstream LLM downstream workloads.

### Questions
See above.

### Soundness
2

### Presentation
2

### Contribution
2
