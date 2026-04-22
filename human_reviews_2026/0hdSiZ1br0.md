# dMARK: Decoding-Guided Watermarking for Discrete Diffusion Language Models

- Avg Score: 3.33
- Decision: Reject
- Scores: 2, 4, 4

## Abstract
We introduce dMARK, the first decoding-guided watermarking method for discrete diffusion language models (dLLMs).  Unlike prior approaches that modify token probabilities, dMARK embeds watermark signals by steering the decoding order according to a binary hashing rule that prioritizes tokens whose indices match a target parity, leaving the underlying probability distribution intact.  dMARK is broadly compatible with common decoding strategies (e.g., confidence, entropy, and margin-based) and can be further enhanced with beam search.  Experiments on multiple dLLMs and benchmark datasets show that dMARK achieves strong detectability 
with minimal quality degradation.  The watermark also remains robust under post-editing operations, including insertion, deletion, substitution, and paraphrasing, establishing decoding-guided watermarking as a practical solution for dLLMs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a watermarking method for discrete diffusion language models, where the decoding order is controlled based on binary hash rules to embed watermark signals. This method is compatible with common decoding strategies and can be further enhanced by beam search. Experimental results across multiple models and datasets demonstrate its effectiveness.

### Strengths
1. The paper explores watermarking schemes for the decoding process of dLLMs.

2. The paper provides experimental analysis from the perspectives of detectability, text quality, and robustness.

### Weaknesses
1. The writing is quite rough; for example, references to appendices in the main text lack chapter numbers, and the wording is imprecise. The abbreviation for discrete diffusion language models should be "ddlms," not "dllms," which refers to "diffusion large language models."
2. The logic in the Introduction section is somewhat disorganized, alternating between discussions of LLMs and dLLMs (e.g., the first paragraph mentions dLLMs, the second paragraph shifts to LLMs, the third paragraph discusses related work on LLMs, and only towards the end does it address dLLMs). The motivation for the paper is not clearly presented.
3. The description of the proposed method is rather simplistic, with much of it being based on prior work, such as dLLM decoding, beam search sampling, and window-based detection, making it hard to discern the core innovation.
4. There is a lack of an Ethics Statement and Reproducibility Statement sections.

### Questions
1. What are the fundamental differences between this approach and KGW? It seems to be merely a substitution of the hash scheme.
2. The input to the hash function only consists of the watermark key and tokens from the vocabulary. Does this mean the segmentation across all positions is identical?
3. Does the "beam search" mentioned in the paper refer to sampling tokens from multiple positions at the same timestep?

### Soundness
3

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
4

### Summary
The authors introduced a watermarking method for discrete diffusion language models (dLLMS) called dMARK. The authors claimed that the proposed method is the first decoding-guided watermarking method for such models. dMARK embeds watermark signals without altering token probabilities, ensuring strong detectability and minimal quality degradation. Experiments were done demonstrating dMARK method’s performance and robustness.

### Strengths
1. The proposed method is designed to be effectively combined with any decoding strategy, the ones listed in the paper include confidence, entropy, margin, and greedy-based rules.

2. The authors claim that this is the first watermarking method done on dLLMs, signaling research significance and a current need for such a method.

3. The overall motivation of this paper is solid - identifying a gap in research on a fast-rising new variant of LLMs (dLLMs) and exploiting new vulnerabilities introduced by the unique design of such models: the order-agnostic property forces decoding strategy to be an essential design choice.

### Weaknesses
1. The paragraph on robustness to post-editing is very short, it would be valuable to see more discussions on robustness evaluations against more sophisticated post-editing transformations instead of just insert/delete/substitution. 

2. Although it is claimed to be the first watermarking method done on dLLMs, comparing against other watermarking methods on other language models can help demonstrate the effectiveness of the method. 

3. The proposed method can increase detectability but also increase inference cost and PPL. More analysis on the computational overhead is needed.

4. It would be helpful to have a method figure, describing a pipeline framework of the proposed method from input to output. The only figure the authors demonstrated before the experiments was an output sample watermarked paragraph.

### Questions
1. According to Table 3, the text generation quality between the dMARK method and the dMARK+ 3-beam does not seem to vary much on MMLU and GSM-8K. Is there any extended discussion on why that is?

2. What is the performance on short answers (small number of output tokens)?

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
3

### Summary
This paper proposes a watermarking method for dLLMs. For dLLMs, the generation process is no longer auto-regressive but in a specific order decided by a reward function per token position. The authors modify the generation order by prioritizing those positions of which the to-be-generated token's hash value (determined by a watermarking key) is 1 (mod 2). Detection is performed with a parity‑matching ratio and one‑sided $z$‑test. The authors also propose a beam search variant. Experiments are conducted on LLaDA‑8B, LLaDA‑1.5‑8B, and Dream‑7B, showing low FPR with competitive text quality. Robustness under random token edits and paraphrasing is also tested and discussed.

### Strengths
1. The paper proposes the first method that deals with watermarking dLLMs. The order of generation is modified, which makes sense in the scenario of dLLMs where the order matters very much.

2. The self-hashing algorithm is easy to implement. Also, it is robust to edit.

### Weaknesses
1. The idea of self-hashing has already been conducted long ago (for example, *A Watermark for Large Language Models* Algorithm 3). This self-hashing algorithm in the work is very similar to theirs. The algorithm works like rejecting all other $n - 1$ tokens and accepting only $1$ token per generation, and the rejection probability is higher if the sampled token is in $\mathcal{R}$ (which is what *A Watermark for Large Language Models* Algorithm 3 is doing). Although the conduction may be different, the final outcome is very similar. Also, the detection method is exactly the same. In that sense, the novelty of this algorithm is undermined.

2. Theory: watermarking algorithm's "unbiasedness". There are two concerns. One major concern is that the algorithm's "unbiasedness" is not proved. I know the proof might seem trivial to the authors, but that claim must be proved in a self-contained manner for completeness. One minor concern is that the idea of "unbiasedness" is only a marginal unbiasedness with respect to key $\xi$ (for example, see *Towards Better Statistical Understanding of Watermarking LLMs*). More specifically, only $\mathbb{E}_{\xi}[p_{\text{watermarked}}(y|x, \xi)] = p_{\text{unwatermarked}}(y|x)$ holds rather than $p_{\text{watermarked}}(y|x, \xi) = p_{\text{unwatermarked}}(y|x)$. I think the authors should be more precise. Adjusting the priority of generation is actually equivalent to filtering out those tokens in $\mathcal{R}$, and that should be clearly addressed.

3. The text quality seems to degrade by a large degree. I think that might be due to modifying the order of generation, which is a core ingredient in developing high-quality dLLMs.

### Questions
1. Please add more dicsussions on the "unbiasedness" of this algorithm. A proof is necessary.

2. Please address the difference between your algorithm and the self-hashing algorithm of Kirchenbauer et al. 2023.

### Soundness
2

### Presentation
2

### Contribution
2
