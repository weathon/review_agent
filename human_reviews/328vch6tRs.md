# From Tokens to Words: On the Inner Lexicon of LLMs

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 8

## Abstract
Natural language is composed of words, but modern large language models (LLMs) process sub-words as input. A natural question raised by this discrepancy is whether LLMs encode words internally, and if so how. We present evidence that LLMs engage in an intrinsic detokenization process, where subword sequences are combined into coherent whole-word representations at their last token. Our experiments show that this process primarily takes place within the early and middle layers of the model. We further demonstrate its robustness to arbitrary splits (e.g., “cats” to “ca” and “ts”), typos, and importantly—to out-of-vocabulary words: when feeding the last token internal representations of such words to the model as input, it can “understand” them as the complete word despite never seeing such representations as input during training. Our findings suggest that LLMs maintain a latent vocabulary beyond the tokenizer’s scope. These insights provide a practical, finetuning-free application for expanding the vocabulary of pre-trained models. By enabling the addition of new vocabulary words, we reduce input length and inference iterations, which reduces both space and model latency, with little to no loss in model accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies how subwords are detokenized through transformers into the original word for further processing and understanding (where LLM is able to distinguish between words and non-words are shown are shown in a preliminary study in this work). In this research direction, the paper makes the following contributions:
- The paper shows that detoenization process happens in the beginning-middle layers using techniques using techniques such as logic lens (single token) and patchscope (multi-token)
- The paper then carry on experiments suggesting that the detokenization happens within FFN layers
- Leveraging the above results, the paper shows that that transformer efficiency can be enhanced by introducing "decodable" token embeddings; the paper examines both input embeddings and output embeddings.

### Strengths
The paper presents a significant amount of content and materials while being easy to follow. The paper incorporates appropriately the related works so that it is relatively straightforward to situate the paper in the literature. Concretely, I think the paper has made the following contributions:
- Through techniques such as logic lens and patchscope, the paper demonstrates convincingly where the model performs the detokenization process by presenting clearly such studies.
- The paper shows that FFN serves to combine subword information in section 5.
- In the final section, the paper shows how the understanding can help transformer decoding in practice. The paper adds word embeddings both in input matrix and output matrix and show that the model can accelerate the inference while maintaining a good performance.

### Weaknesses
While the paper presents a complete study (with no missing component) in the detokenization study, it feels that the paper can still be further enhanced with some more in-depth studies, some of the questions I have put in the questions section but in general:
- The cumulative curve shows that FFN indeed contributes to detokenization. What about other components? Are there any hints/patterns that the authors observe in the detokenization process (e.g. what words are first detokenized)?
- Cumulative rate saturates at around 0.7 shown in the figure. What about the rest 30%? Are these limitations for the measured model? Do better models (e.g. llama3) perform better at these?
- More details will help section 6 and I list some of them in the questions section. I think these are just some of the questions that a common reader would have after reading the paper. I think the results in this section may be of practical importance and deserve to be enhanced with more empirical results.

### Questions
For section 5, the feedforward mechanism, why only FFN output are measured please? Does it make sense to also measure the residual part please?
For section 6:
- For all original vocabulary tokens, all models perform less well than the model with original vocabulary? Are there examples to illustrate these and some hints where models fall short?
- What would be the full newly token accuracy for the model with original vocabulary?
- With such techniques, what would be an estimated inference speed gain? For input embedding as well as for output embedding?

### Soundness
2

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
In this paper, the authors investigate how LMs internally reconstruct word-level representations from sub-word tokens, a process they term "detokenization". They provide evidence that Lms can inherently combine sub-words into hidden representations which can be mapped into coherent words, even for out-of-vocabulary items, across the early to middle model layers. By probing LMs on both known words and artificial nonwords, they show that the model forms distinct representations for these categories, suggesting an "inner lexicon" that extends beyond tokenized inputs. The findings reveal that this detokenization mechanism leverages feedforward layers and attention patterns to generate whole word representations, which could, in theory, improve vocabulary flexibility without finetuning (though this is not shown in practice).

### Strengths
This paper addresses a crucial question: how can language models construct symbolic representations of entire words when their input comes from tokenizers that often fragment words in ways that disregard their morphological structure? Specifically, the authors investigate whether LMs internally form representations of morphological units that help bridge the gap between the tokenized input and the naturally holistic nature of words in language. Through experiments, the paper presents some evidence that whole-word representations emerge within the model’s hidden states, even when it processes fragmented word tokens. Additionally, the writing is clear, and the experiments are easy to replicate.

### Weaknesses
I believe there is a disparity between the paper’s claims and the experimental evidence provided to support them. Specifically, some of the experiments lend themselves to alternative interpretations, which could be clarified with additional baselines or experiments. The paper claims is that model come up with an "internal lexicon" that create hidden representations of "virtual" words, even when fed, e.g., word pieces as input. This is a claim on the computation carried out by the model, i.e., it is implied that there are some modules whose explicit computation is forming this internal lexicon. I am not sure that the experiments provide sufficient evidence for this claim:

* First, the "motivating experiment" in Section 3 lacks sufficient controls. The authors demonstrate that there is a linear separation in the hidden state between the representations of actual multi-token words and fictional ones created by randomly mixing word pieces. However, this separation could simply reflect the model's ability to distinguish between linguistically valid English morphology and nonconforming sequences, rather than providing evidence of "internal detokenization." For instance, an alternative hypothesis is that the model has learned distributional cues—such as suffixes like "ing" rarely appearing at the beginning of a word—which causes out-of-distribution effects in the hidden states when encountering atypical token sequences.

* In Section 4.1, the authors hypothesize that "if the model performs detokenization, it will represent the last token of a word similarly to the original word token." However, even if such similarity is observed, it could be attributed to the distributional properties of language rather than any explicit "detokenization" process. For instance, in the example provided in the paper where "cats" is split into "ca" and "ts," it is plausible that the pretraining corpus contains instances where this split occurs unnaturally, such as in URLs like "catsanddogs.com" (an actual website) or in cases with typos. Such occurrences might push the representation of "ca ts" closer to that of "cats" without requiring an explicit detokenization step. Furthermore, it is known that such similarities exists also in word2vec methods like Glove, and it is difficult to argue that any explicit detokenization happens there. 

* In Section 4.2, the authors feed the hidden state of the last token of a multi-token word into the model and prompt it to repeat the word. Instances where the model accurately reproduces the entire word are taken as evidence that it has stored the multi-token word in an "internal lexicon." However, a key baseline is missing: including phrases that are not single words, such as "repeat this word: rainy day." The observed results could simply reflect the model's tendency to form contextualized representations that transfer information across tokens, rather than indicating an internalized whole-word representation.

* Finally, the paper’s closing sections aim to illuminate the model's internal computations and the supposed formation of an internal lexicon. While the results provide some evidence of contextualization in the feedforward layers, it's unclear to me whether they genuinely support the existence of an internal detokenization process. Intervention-based experiments could strengthen this claim. For example, could we identify a subset of parameters where ablation specifically impairs performance on multi-token words without affecting single-token words? Or could linear concept erasure techniques reveal a subspace whose neutralization removes all distinctions between multi-token and single-token representations?

### Questions
* The experiments in 4.1 focus on logit lens. What about cosine similarity or more direct measrues of similarity?

### Soundness
3

### Presentation
2

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
This paper analyzes the process of latent detokenization inside the transformer-based LM forward pass, as it occurs across network layers. It shows that models are able to recognize words from pretraining even when they are noised with slight token variations or artificially split across multiple tokens. These experiments are different from earlier works, but ultimately show very similar findings about hierarchical processing in transformers. Using these findings, a novel method is briefly introduced to leverage the internal states of merged tokens to automatically expand the token vocabulary, which can hypothetically improve inference costs with fewer lookups. This method appears initially effective, but could be explored more.

### Strengths
1. This paper analyzes the process of detokenization across transformer network layers via a series of targeted experiments. It builds an intuitive understanding that agrees with many prior works in layer-based analysis.

2. The paper proposes an interesting method for training-free expansion of the model vocabulary by leveraging the insights into internal word representations. This method is shown to be effective in limited experiments. See below in "weaknesses" for further thoughts on this.

3. The writing is clear, but sometimes too abstract (see weakness 5).

This paper shows very solid work and I greatly appreciate the thorough breadth of exploration, though it could possibly be more effective to focus on fewer areas. I want to emphasize that I enjoyed reading the paper and believe it will be strong after some revision, including reworking the claims and focusing more on the novel contributions which are begun later in the paper. I believe it would be more impactful to explore sec 6 in more depth; see weakness 4 below.

### Weaknesses
1. The concept of an inner lexicon is interesting, but not novel as is claimed in this work. The idea follows implicitly from prior work in the memorization of training data, and explicitly in works about tokenization, such as the introduction of BPE (which is discussed greatly in this paper). It is the stated goal of subword tokenizers to enable learning a vocabulary of words and concepts which is larger than the vocabulary of concrete tokens through the process of token combination. It is nice to see these findings reproduced and analyzed, but they are not new.

2. The experiment in section 3, which motivates the idea of an inner lexicon, is not very strongly designed. Why are nonwords created by randomizing tokens, and not by some other method on the morphological level or otherwise something more linguistically motivated? Resulting nonwords do not seem to follow English conventional morphology (eg. the nonword "chha") and this could make it trivial to distinguish words from nonwords. Prior work has shown LLM sensitivity to word frequency in training corpora, and this experiment seems to reproduce those findings. This experiment seems to me to show that LLMs can distinguish easy cases such as "chha" which are very dissimilar to real words, and predictably struggles with more difficult cases that more closely resemble real words (see appendix) but there doesn't seem to be strong evidence that the LLM representation is doing more than locating words on a gradient based on their prior likelihood of appearing in the pretraining corpus. This fact is fairly well established at this point.

3. The experiments in the paper seem mostly sound and reasonable, but their novelty is overstated. Several of the earlier experiments in particular build on each other to show that early and intermediate layers in the network are responsible for aggregating and disambiguating word representations (sec 4 and 5). However, these findings may be seen to be subsumed by many prior works in the analysis of syntactic and semantic composition of tokens across transformer layers (see section 4 in [1] for many citations).

4. The paper may have been too ambitious in scope. The first several experiments were good reproductions of findings. The last experiment was novel to me, and it would have been interesting to expand on it more deeply. However, it did not require many of the earlier experiments in order to understand it, which took up most of the room in the paper. Other reviewers may have different opinions, but mine is that the paper would be more valuable if it explored the final research question more deeply, and provided more concrete findings for it. For example, can we estimate a size/contents of the inner lexicon? Does this lexicon scale with model capacity and/or training size? Can we provide some guarantees or estimates about the boundaries of the method of finetuning-free vocabulary expansion? For what kinds of words is this method effective and when is it ineffective?

5. There were many smaller experiments given in the paper, and this resulted in important implementation details being often omitted. For example, experiments often hinge on model memory of tokens from training, and the natural distributions of those tokens in the corpora, but details about how words/tokens were sampled in the tests (such as construction of nonwords) were not often given in enough detail to reproduce experiments. I would expect there to be significant influence of such distributions on test outcomes, so these details are important.


[1] Anna Rogers, Olga Kovaleva, Anna Rumshisky; A Primer in BERTology: What We Know About How BERT Works. Transactions of the Association for Computational Linguistics, 2020.

### Questions
1. How were tokens assembled into nonwords in sec 3? I am missing detail here which could be useful in understanding the method. I also do not understand what it means to "fit" a KNN classifier (which is non-parametric) -- were there representations used which were different from those taken from the model hidden states?
2. There was a claim made that the proposed method in section 6 can improve inference-time costs, though I cannot find any experiments or numbers for this in the paper. Can the authors point me to or provide any information about this? Thank you.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper explores the process in which models transform tokens, which often split long words into subwords (e.g., "un" "h" "appiness"), into higher level representations for the full word through "detokenization". Detokenization has been observed in LMs before, but has not been directly studied extensively. This work shows that LMs can recognize when a word is part of a larger word, and show that early attention fuses subwords together (in the last token of the word), and uses early MLP layers to then recall the full word from multiple subwords in an "internal dictionary" (e.g., representing "unhappiness" as a single vector internally even though it is not in the tokenizer). The authors then show that this can be used to expand a model's tokenizer by including the hidden 'internal dictionary' representation as an input token. This works to some extent.

Overall, this paper enhances our understanding of early layer processing in language models, and provides a path towards enhancing models to reduce inference time.

### Strengths
This paper answers particular unanswered questions surrounding "detokenization", which has been repeatedly observed and discussed without being properly studied. These are important for observations around, for example, stages of inference in language models.

Interpretability results on early layers of LMs are often lacking, as vocab projections are much easier to perform at later layers. This work provides interesting and convincing results for one role early layers take on in these models, which is indeed different from the roles of later layers.

The vocab expansion experiments are a nice proof of concept, and could be expanded on in the future to decrease inference times.

The results on typos are interesting and to my knowledge, novel

### Weaknesses
The evidence for a third stage of processing in Figure 2b is a little sparse. These results are only for one model, and the degree to which accuracy drops is not substantial enough to obviously be due to a difference in processing altogether. These results could be made stronger by including results for more models. As a motivating example, it is fine, but perhaps isn't the best use of that space if this point can't be made more strongly.

Typos:

L370: "form"

### Questions
If the model wants to generate some multitoken word that it represents in its 'internal dictionary' is it "planning" multiple tokens ahead? Why or why not?

### Soundness
4

### Presentation
4

### Contribution
3
