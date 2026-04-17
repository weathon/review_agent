# Alice: An Interpretable Neural Architecture for Generalization in Substitution Ciphers

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 2, 2

## Abstract
We present cryptogram solving as an ideal testbed for studying neural network reasoning and generalization; models must decrypt text encoded with substitution ciphers, choosing from 26! possible mappings without explicit access to the cipher. We develop ALICE (an Architecture for Learning Interpretable Cryptogram dEcipherment), a simple encoder-only Transformer that sets a new state-of-the-art for both accuracy and speed on this decryption problem. Surprisingly, ALICE generalizes to unseen ciphers after training on only ${\sim}1500$ unique ciphers, a minute fraction ($3.7 \times 10^{-24}$) of the possible cipher space. To enhance interpretability, we introduce a novel bijective decoding head that explicitly models permutations via the Gumbel-Sinkhorn method, enabling direct extraction of learned cipher mappings. Through early exit and probing experiments, we reveal how ALICE progressively refines its predictions in a way that appears to mirror common human strategies---early layers place greater emphasis on letter frequencies, while later layers form word-level structures. Our architectural innovations and analysis methods are applicable beyond cryptograms and offer new insights into neural network generalization and interpretability.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates how Transformer models can learn to decode monoalphabetic substitution ciphers. The authors propose a pooling-based decoder, and a bijective decoding head designed to enforce one-to-one token mappings. The proposed method achieves state-of-the-art performance on short cipher sequences and contributes an intuitive and efficient architectural component for modeling 1:1 substitutions.

### Strengths
1. The pooling method is intuitive and effective, and the paper proposes a bijective decoding head to enhance interpretability.
2. The approach achieves state-of-the-art performance on short 1:1 ciphertext sequences.

### Weaknesses
The task studied is relatively simple, and the interpretability of how the model solves it remains incomplete.
1. The 1:1 monoalphabetic substitution cipher is a simple problem, and it is simpler than the homophonic ciphers already studied in Kambhatla et al., 2023, which allow multiple ciphertext symbols per plaintext character and flatten frequency distributions.
2. The main innovation of this paper is the bijective decoding head, but it does not improve model performance. The claimed interpretability benefit also seems limited: even without a bijective head, the cipher map can still be easily visualized. The only difference is that the resulting mapping is no longer strictly monoalphabetic.

### Questions
I am curious what the average Symbol Error Rate (SER) curve looks like as a function of ciphertext length, like figure 1 in Kambhatla et al., 2018. It would help clarify how quickly the model begins to recover the cipher mapping, especially on very short sequences.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper employs encoder-only Transformer models to solve substitution ciphers and demonstrates that their performance on short-length ciphers surpasses that of existing algorithms. In parallel, the authors perform an interpretability analysis, revealing that the model has successfully learned the underlying substitution rule. Finally, they apply a linear probe to analyze how the model encodes the learned functionalities.

### Strengths
1. The paper is written clearly with a proper description of the experiments.
2. Source code was provided.

### Weaknesses
1. The baseline comparison is misleading. For instance, the cited work by Kambhatla et al. (2023) uses a significantly larger dictionary size than the authors’ setup, which invalidates the claimed SOTA performance.
2. The authors repeatedly emphasize that the search space in their setting is $26!$, but this is not how simple substitution ciphers work. I would encourage them to check standard references such as https://cacr.uwaterloo.ca/hac/about/chap7.pdf and the concept of unicity distance. From theoretical considerations, about 28 characters are needed to brute-force such a cipher for English. Even from a data perspective, the dataset used in this work already contains more than enough samples to fully recover the character frequency distribution of the plaintext.
3. From an AI perspective, the authors do not demonstrate the scalability of their algorithm. Particularly, for larger dictionary sizes, which are both more challenging and more practically relevant. This issue is closely related to my first point.
4. The interpretability study is limited. For example, line 368 states: “because we explicitly model the latent permutation of the alphabet, we directly recover the key, as shown in Figure 4.” This statement is essentially tautological to me: “because we explicitly model the latent permutation, we can extract the permutation.” Moreover, while the linear probe and n-gram analyses show that the model progressively solves the task, the authors do not provide any mechanistic explanation for how this happens, relying instead on vague verbal reasoning.

### Questions
On top of the weaknesses mentioned above, I have a few more detailed questions for the authors:

1. Why do the authors average features from the same character at different positions? Wouldn’t simply removing positional embeddings achieve the same goal, as this naturally enforces positional invariance? An encoder-only architecture does not introduce imbalance in-context like decoder-only models.
2. Figure 9 might not be as difficult to interpret as suggested. Layer 1, head 1 already appears to capture similarities between identical characters and some substitution patterns. I also recommend computing cosine similarities or correlations of internal features across different characters, which would provide a straightforward observable for how the model learns the deciphering algorithm layer by layer.
3. Do we really need 12 layers? A smaller model should be able to solve this task and might be more interpretable and faster to run.
4. It would be very interesting if the authors could increase the dictionary size or vary the language distribution and evaluate whether their conclusions continue to hold under these more challenging and realistic conditions.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper is concerned with the cryptogram task: given a bijection between alphabets, applying the bijection transforms a text to a cipher text. One needs to recover the original text from the cipher text.
The paper introduces an encoder-only transformer architecture that, given a training dataset consisting of a pair of (text, ciphertext), it solves the cryptogram task in a data efficient way that is claimed to be the state-of-the-art. In addition, it further explored the interpretability angle and presented some findings in interpreting the networks' prediction mechanism.

### Strengths
Clear presentation and thoughtful ablations. The paper itself is a good one, and is fairly complete in terms of a proper research paper which deserves to be published somewhere. It is just debatable how good the problem and solutions are, and how much new insights does it tell us so that it can be presented in this venue.

### Weaknesses
Lack of novelty. There isn't much insight that is new. Also I'm not sure if I miss anything, but the difficulty and significance of the work appears to be significantly exaggerated in the abstract and some of the main text. 
Also, the task could have been generalized and pitched a lot more toward interpretability research. But to be fair, the paper did make some interesting effort using linear probing.

### Questions
Is Figure 2 the only justification for the state-of-the-art status? Are you trying to compare with "any" other methods without constraints, or has there been certain limit on the amount of compute spent at either training time and/or serving time? As to the SoTA status, I do not mean that every paper needs to chase SOTA, but you did make the this claim in the abstract: "... that sets a new state-of-the-art for both accuracy and speed on this decryption problem". And typically in this scenario, constraint is one of the most important factors in defining where you are at (e.g., equi-training-compute and equi-serving-compute already have huge ramifications). For tasks this common, it is also fairly reasonable for the authors themselves to present a sweep of best-effort experiments on various parameters, especially since many literatures are not really aiming for claiming SoTA (e.g. dictionary learning in interpretability research).

In abstract, you wrote "Surprisingly, ALICE generalizes to unseen ciphers after training on only ∼1500 unique ciphers, a minute fraction (3.7 × 10−24) of the possible cipher space.". Why is it surprising? Looking at it alphabet-by-alphabet, do the ciphers contain all the possible vocabulary mapping already, or you meant that the training set only has partial information and it generalizes to full vocabulary mapping?

"This task is combinatorially complex and requires reasoning over a space with V! possible ciphers ...". Why is it combinatorially complex? The way that presents it here seems extremely misleading. In the space of bijective maps between vocabularies, yes, we are talking about V! functions. For any language modeling tasks (toy tasks such as arithmetics, or real applications such as language translations), if we talk about the entire function space, we in most cases have orders of magnitude larger amount of functions than this.

Appendix I mentioned that LLM fails at this task. Yes, this is a very difficult task in a prompting-only regime. But have you tried supervised fine-tuning so that the methodology becomes apple-to-apple comparing with this paper?

### Soundness
2

### Presentation
4

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Summary: 
- Introduces cryptogram solving as a controlled testbed for studying reasoning and generalization in neural networks.
– Models must decrypt substitution ciphers without seeing the cipher key, facing a combinatorial search space of 26!.
– Proposes ALICE, a simple encoder-only Transformer that achieves state-of-the-art accuracy and speed on decryption tasks.
– Offers general architectural and analytic insights into model interpretability and generalization beyond cryptography.

### Strengths
I am a fan of using ciphers (and broadly encryptions) to better understand the nature of compression and intellihgent behavior in AI systems. So overall, broadly a fan of the direction.

### Weaknesses
The work motivates the study of cryptogram solving as a testbed for "understanding reasoning and generalization in neural networks". If I had seen this work 5 years ago, I would have been very excited. But at this point it is not clear to me what new knowledge it is adding to our current understanding.  

I do not find the contributions as compelling. For example,  "novel decoding head that explicitly enforces bijectivity" sounds rather incremental: unclear if your work is adding on top of what we have known from Mena et al. 



==A few related work==

in the context of modern LMs, there are works that use ciphers that should be discussed:
 - [1] https://arxiv.org/pdf/2504.19395
 - [2] https://arxiv.org/abs/2308.06463

There is also a classical work in NLP on ciphers. Few examples, among others:  

 - [3] Kevin Knight, Anish Nair, Nishit Rathod, and Kenji Yamada. 2006. Unsupervised analysis for decipherment problems.
 - [4] Qing Dou and Kevin Knight. 2012. Large scale decipherment for out-of-domain machine translation.
 - [5] Nima Pourdamghani and Kevin Knight. 2017. Deciphering related languages.
 - [6] Sujith Ravi and Kevin Knight. 2008. Attacking decipherment problems optimally with low-order n-gram
models.
 - [7] Sujith Ravi and Kevin Knight. 2011. Bayesian inference for zodiac and other homophonic ciphers
 - [8] Taylor Berg-Kirkpatrick, Greg Durrett, and Dan Klein.
2013. Unsupervised transcription of historical documents.

### Questions
No questions at this point. 
Looking forward to the authors' response for further exchange.

### Soundness
4

### Presentation
2

### Contribution
1
