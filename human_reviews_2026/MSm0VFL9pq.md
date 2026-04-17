# Hyperdimensional Probe: Decoding LLM Representations via Vector Symbolic Architectures

- Decision: Reject
- Scores: 0, 6, 4, 4

## Abstract
Despite their capabilities, Large Language Models (LLMs) remain opaque with limited understanding of their internal representations.
Current interpretability methods either focus on input-oriented feature extraction, such as supervised probes and Sparse Autoencoders (SAEs), or on output distribution inspection, such as logit-oriented approaches. 
A full understanding of LLM vector spaces, however, requires integrating both perspectives, something existing approaches struggle with due to constraints on latent feature definitions.
We introduce the *Hyperdimensional Probe*, a hybrid supervised probe that combines symbolic representations with neural probing. 
Leveraging Vector Symbolic Architectures (VSAs) and hypervector algebra, it unifies prior methods: the top-down interpretability of supervised probes, SAE’s sparsity-driven proxy space, and output-oriented logit investigation. 
This allows deeper input-focused feature extraction while supporting output-oriented investigation.
Our experiments demonstrate that our method consistently extracts meaningful concepts across different LLMs, embedding sizes, and setups; uncovering concept-driven patterns in analogy-oriented inference and QA-focused text generation.
By supporting joint input–output analysis, this work advances semantic understanding of neural representations while unifying the complementary perspectives of prior methods.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
The work proposes a novel probing procedure, in which a new tokenization is first constructed in order to orthogonally represent the concepts in the training corpus. Empirical results are shown on a synthetic dataset and a real one to show that this method can successfully extract interpretable concepts from the LLM's hidden space. A novel method for reducing the dimensionality of the full LLM hidden space is also proposed as part of the procedure.

### Strengths
The paper is laid out in a logical way, making it easy to follow. The method is novel, and some interesting decisions are made as to its implementation.

### Weaknesses
1.Experimental design is highly questionable

1(a). The method is only compared against naively prompted LLMs, despite the description of a full body of related work above. Also, since your method is trained, fine-tuning the LLM on your dataset should be considered as a baseline. The free-form setting which was chosen both conveniently eliminates most probes, and obscures LLM evaluation.

1(b) In 5.1 (l. 345), the LLM is considered half-correct by a 0.5 accuracy value when its next-token prediction is correct but the full answer to the question is multi-token. This seems extremely questionable. 

1(c) LLM comparison throughout the results section seems to be conducted with naive-prompting. Few-shot examples are not provided. Certainly, the very low performance on what should be a very simple task (simple analogical reasoning) reported for the LLMs is indicative of problematic treatment. Without considerate prompt design, or few-shot examples demonstrating output format, LLMs are liable to provide answers in a variety of forms, such as "the answer is [answer]", rather than "[answer]", making next-token measurement unsuitable. Your later evaluation on the real dataset, based on inclusion of the answer in the text, shows much more reasonable performance (supporting this suspicion). 

1(d) l. 443: the reported F1 and EM statistics for the real-world dataset show exceptionally large variances ( 0.69 ± 0.38, 0.52 ± 0.5 respectively). How were these variances calculated? Statistical significance of your empirically supported claims should be provided.

1(e) a single non-synthetic dataset is tested on. The dataset used does not capture notions of reasoning, despite the synthetic data being reasoning-focused. 

2. Impact: the impact of this paper seems weak. Correctly, no claims of performance gains are made. The main claim to impact in this paper is the demonstration that VSAs can work, and some promisory demonstrations of potential downstream use cases.

### Questions
1. In practice, this method effectively mirrors the tuned lens framework in which a mapping is learned from the hidden state through to the decoding matrix, in order to address layer-decoder un-alignment that may be present due to the intermediate layers' lack of explicit training for this application. The main distinction is that in this approach, a new decoder is designed before training. Your claim is that the nature of this new vectorization addresses some of the weaknesses of supervised probes. Can you demonstrate this empirically in any way?
2. Presumably, if a latent representation learned for interpretability contains more information on the important concepts pertaining to a problem than a different representation, then it has a larger potential performance as a model-input. How do you support your claims that VSA leads to a richer and more conceptually faithful space relative to the various related work discussed, when performance compared to even vanilla LLM prompting is similar.

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors present an approach to probing that is based on Vector Symbolic Architecture (VSA) and avoids the limitations that direct logit attribution (DLA)/logit lens-based methods and sparse autoencoders (SAEs) have. Their approach requires three steps: (1) computing the input representations of transformers by feeding inputs into the model, (2) training a supervised model to map token embeddings from a transformer into VSA encodings, and (3) examining the VSA encodings and extracting the concepts embedded into them. Performance with the hyperdimensional probe shows that probing reveals concepts more accurately than LLM next token prediction on the analogy task.

### Strengths
The paper is well motivated with the goal of improving over existing interpretability methods. The paper applies a unique approach to interpretability by techniques in vector symbolic architectures to analyze a transformer's residual stream. Furthermore, the approach is also applied to two settings which enables understanding of the generalizability of the approach. References to the appendix are comprehensive and the paper is generally thorough.

### Weaknesses
1. A comparison against either probe or SAE should be included in the main paper, if the claim is that the hyperdimensional probe outperforms (in some way) these existing methods. Although it is noted in the appendix, the baseline is important enough that it should be included in the main text.
2. It is unclear in Table 1 what each of the extracted concepts mean, from first glance. For instance, Key Values is only briefly mentioned in the caption, and as is Out-of-context. It would be better to include a section that also elaborates on the evaluation (and each "extracted concept") since this is a core finding of the paper.

### Questions
1. How are the features decided for 5.3 for question answering? It seems like the approach does not seem to be easy to apply to different tasks since it is not obvious what features to enumerate when the format is not analogy-style.
2. How compute intensive is constructing the hyperdimensional probe? Meaning, how many GPU hours are used per probe + dataset?
3. How does this approach generalize out-of-domain on datasets that the hyperdimensional probe is not trained on?

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
This paper proposes Hyperdimensional Probe: a method based on Vector Symbolic Architectures to extract concepts from a latent. With controlled settings the authors validate the proposed approach and show that it can successfully extract meaningful concepts.

### Strengths
* This paper tackles a very important problem of Eliciting Latent Knowledge (ELK), although in a controlled setting where the concept space is known.
* I think the proposed method, building on ideas from Vector Symbolic Architectures, is both interesting and novel. This approach may unlock new ways for concept extraction or interpretability in general.

### Weaknesses
* Poor readability: I had a very hard time reading parts of the paper. Even after multiple passes I couldn't fully grasp certain sections. See my clarifying questions below.
* Missing obvious baselines and comparisons to prior work on ELK. See Question 1 below.

### Questions
1. Baselines: Eliciting Latent Knowledge (ELK) being a very important problem in interpretability, has been widely studied in the literature. [nostalgebraist, 2020](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens), [Belrose et al, 2023](https://arxiv.org/abs/2303.08112), [Ghandeharioun et al, 2024](https://arxiv.org/abs/2401.06102), [Hernandez et al, 2024](https://arxiv.org/pdf/2308.09124) (among others) have all proposed different methods for ELK. The authors should compare their method against at least some of these baselines to better position their contributions.
    
    1.1. Correct me if I am wrong, but *probing@1* corresponds to accuracy in detecting the correct concept, right? If so, it should be easy to compare against the mentioned baselines.


2. Line 20-21: *"This probe combines the strengths of SAEs and conventional probes while overcoming their key limitations."*: This is a strong claim and I am not sure I understand the authors' justification for this. 

    2.1. For SAEs, Hyperdimensional Probes overcome the feature labeling part of the limitation, while introducing the need for supervision to learn mapping to the proxy space, exchanging one limitation for another.

    2.2. For probing: Line 37-38 says "... though their decoding capabilities have been debated". I basically didn't get this statement. Which part of [Hewitt and Liang, 2019](https://arxiv.org/abs/1909.03368) are you referring to? And how does your approach overcome this?

    2.3. [Belinkov, 2021](https://arxiv.org/abs/2102.12452) lists some shortcomings of probing methods, such as the need for causal validations. I believe that Hyperdimensional Probes also may suffer this shortcoming. However, I am not sure what could be done to test this as I don't think the mapping to proxy space, $\mathcal{M}$, is invertible. But I would like to hear the authors' thoughts on this.


3. Line 244-245: *"We apply sum pooling, which consists of summing all centroid embeddings."*: In modern transformer LMs, latents in later layers tend of have higher magnitudes compared to earlier layers. So, summing all latents may bias the resulting vector towards later layers. Was there any normalization step involved to address this?

4. Assigning 0.5 precision score for multi-token targets (lines 345-346) definitely underestimates the actual performance of the LM. Why not decide correctness based on the first token only? Or, even better, just generate multiple tokens and check if the full target is generated?

    4.1. I also think the probe performance should be reported in terms of the samples where the LM generated the correct target. After all, the goal is to extract concepts that the LM actually uses in its computation. So, reporting probe performance on samples where the LM failed to generate the correct target seems misleading.

5. I didn't understand Table 1 and. Adding illustrative examples for each of the extracted concepts (what was used for unbinding, what was the target, ...) would help a lot. 

    5.1. I also didn't understand the subsequent discussion around Table 1. For example, in Lines 397-399 the authors discuss how GPT-2 struggles with the task. This is probably because I didn't fully understand Table 1.

6. In Section 5, how were the VSA encodings compared? Just cosine similarity? Was there a threshold for concept detection?

7. In Figure 3 and Lines 456-460, you seem to be expecting the concepts to persist even after text generation. Why is that? Doesn't it make sense that once the concept is used in generation, it is not needed anymore? Also, how many tokens were generated? For the given example did the authors mean that the LM deletes the input concepts (*hydrogen*) immediately after generating (*water*) leading to errors?

### Soundness
2

### Presentation
1

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
This paper proposes a new method to extract and interpret information contained in LLM hidden states called hyperdimensional probing. The method is built on vector-symbolic architectures (VSAs). They apply their method to analogy tasks, a question answering task, and other various tasks in the appendix to show the flexibility of their method in extracting knowledge from llm hidden states.

### Strengths
- The paper applies the VSA probe methodology to a wide variety of tasks of varying complexity, as well as across various model sizes and families which is helpful to show the broadness of applicability.
- The experimental details in the appendix are extensive, and several times I was able to answer questions I had while reading through some of the extra content there. (Though this sometimes felt like a weakness as well - see below).
- While I may be misunderstanding some pieces (due to lack of comparison/explanation), the paper does feel overall well-put-together. The use of VSA for probing seems novel to me.

### Weaknesses
- The paper was somewhat hard to follow at times and I found myself having to read things back several times regarding the VSA training details, architecture, dataset, and evaluation procedure. One challenge with reading is that because there is so much packed into this paper, constant referrals to the appendix from the main text sometimes felt overwhelming. Sometimes the question I had was quickly answered, and other times I had to read a few pages in the appendix to understand a single result. It seems like there is something in the big picture here that I'm missing because the results/paper feels well-put together, but I just didn't quite get why this VSA approach is better/how it compares to other existing methods. The main reasons I may misunderstand your method I'll describe below, but for now they seem like limitations of the current presentation in the paper.

- One downside of probing classifiers in general is that, while they are often used to “detect” whether an attribute is present in a hidden state, they are often correlational and not causal (see [Belinkov et al](https://aclanthology.org/2022.cl-1.7/) for an overview, but also related work). To address this, more recent work has begun addressing whether probes can be used to causally steer generation, which is something I didn’t see attempted in this work. I worry that the hyperdimensional probing suffers from similar problems as traditional probes, and it’s unclear to me whether the VSA probes could “steer” LLM generation, or are they only able to “extract” information from them.

- Similar analogy tasks tested here have been examined in prior work, but there is little comparison with previous methods. There is an attempt at comparison with logit lens (DLA), but there are more sophisticated methods that could be helpful points of comparison for the proposed VSA method. For example, the LRE from [Hernandez et al](https://openreview.net/forum?id=w7LU2s14kE) seems to extract entities for various relations. Similarly, the AxBench paper [Wu et al](https://openreview.net/forum?id=K2CckZjNy0) provides a benchmark for both concept detection and model steering for various concepts that could be formatted as analogies. Evaluating on this benchmark might be helpful to see how VSA probing compares to other standard probing techniques, SAEs, DAS, prompting, etc.

### Questions
- Line 53: what is meant by “hybrid supervised probe”? This is mentioned a few times, but not really explained.
- Several times you mention that VSA probing is an attempt to overcome some of the limitations of other information extraction methods (DLA -> vocabulary/token space, SAEs -> feature naming, etc.), but it’s not clear that these VSA features are any more interpretable/better than alternatives, particularly because it seems like you also tie your features to predefined tokens/concepts. How is this any different?
- If you were to train the VSA codebook for concept/task-steering, how would it differ from [Shao et al](https://openreview.net/forum?id=6axIMJA7ME3)’s compositional task representation codebook approach? Their method seems very related (though obviously not VSA based), as they also train a codebook of features for various LLM tasks.
- What do you expect to get out of doing unbinding with other candidates that aren't the final key/target? I didn't quite understand the breakdown, for example, of the rows in Table 1 and what one might expect to retrieve in each row. Do you similarly evaluate probes/DLA at all token positions for that same information?
___
- Other Notes:
  - Line 160 presents section 5.3 in between 4.1 and 4.2 which was a little odd. I would consider changing the order of these sentences.
  - Line 292-293: “this supports our hypothesis that LLM embeddings can be represented using fully distributed encodings such as MAP-B in VSAs“ - this feels like somewhat of an over-claim because this statement seems like it's only based on the findings from analogy dataset. I would suggest tapering this claim a bit (e.g. If LLMs hidden states can be represented w/ VSAs why are we using transformers and not VSAs).

- Minor Typos
  - Line 80: “token mbedding” -> token embedding
  - Line 275: “valide” -> validate?
  - Line 1148: “attak” -> attack?
  - Figure 2: Label says "GTP-2"

### Soundness
3

### Presentation
2

### Contribution
2
