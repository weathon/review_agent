# Memorization for Good: Encryption with Autoregressive Language Models

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 3, 6

## Abstract
Over-parameterized neural language models (LMs) can memorize and recite long sequences of training data. While such memorization is normally associated with undesired properties such as overfitting and information leaking, our work casts memorization as an unexplored capability of LMs. We propose the first symmetric encryption algorithm with autoregressive language models (SELM). We show that autoregressive LMs can encode arbitrary data into a compact real-valued vector (i.e., encryption) and then losslessly decode the vector to the original message (i.e., decryption) via random subspace optimization and greedy decoding. While SELM is not amenable to conventional cryptanalysis, we investigate its security through a novel empirical variant of the classic IND-CPA (indistinguishability under chosen-plaintext attack) game.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposes a new secret-key encryption scheme based on the idea that LLMs memorize. At a high level, in this approach Alice and Bob both have access to a public LLM but Alice, who wants to send message m to Bob, will find a fine-tuning to the LLM that is based on the secret key k and will allow Bob to extract the planted message m from the fine-tuned model. The idea is that the changes to the model are hidden behind a random projection that is based on the secret key and hence the eavesdropper, at least through an obvious attempt, can not find the information that Bob is able to find and extract the message m.

The paper also does experimental attacks, trying to break its own scheme, and their best efforts *does* indeed break the proposed scheme. The author(s) plan ideas for future improvements that might resist the attacks.

### Strengths
Aiming to find a new encryption scheme.

### Weaknesses
I do not think the paper is looking for the right application for LLM's memorization. Encryption is a very subtle task and has its own key parameters, such as the size of the ciphertext and the ability to resists attacks. 

This paper's proposal leads to quite large ciphertexts, and even a "black-box" attack based on ML (itself) is breaking the proposed scheme. So I don't see why this proposal might actually lead to a useful scheme.

I also believe a new proposal for encryption should be submitted to a cryptography venue to get proper checks, not a learning venue. And in doing so, the presentation should be fully accessible to crypto audience, not written the way it is, with fully clear exposition of the assumptions, tools, notions, etc, so that the crypto community can fully understand and review the new proposal for encryption.

I would be more lenient if the proposal was a public key scheme, for which the crypto community is much more interested in new proposals. But to me, the bar for proposing a new secret key scheme is way higher.

### Questions
You say in page 6: "Other symmetric ciphers like the Data Encryption Standard (National Bureau of Standards, 1977, DES) and the Advanced Encryption Standard (Pub, 1999, AES) do not have this flexibility; the ciphertext is always the same size as the message."
How come this is a weakness? Cannot you always artificially *increase* their ciphertext, if that is needed as a "feature"?

Your security game in page 6 is not clear. What does "encrypt them" in step 2 of your re-framed game is done? which one of m_0 or m_1 is encrypted?

in page 7 and 8, you say that "operating on bit sequences" is a factor that *limits* previous crypto-analysis approaches. But is not it that without loss of generality? As far as I understand, everything can be re-written in zeros and ones.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
In this paper, the authors proposed an LM-based symmetric key scheme, and they claimed that autoregressive LM can encrypt arbitrary data as a real vector and recover it.

### Strengths
I agreed that it is possible to perform encryption and decryption based on LM, and this paper attempted to construct a corresponding method.

### Weaknesses
There must be more exact security proof or argument for the proposed "symmetric key encryption" method. Although the proposed method is based on a random project, Pk is just linear transformations such as a combination of orthogonal matrices and diagonal matrices. The activation function may act as a non-linear transformation, but this is unclear from the algorithm description. Since this structure is similar to the primary design principles of existing block ciphers, conventional cryptanalysis methods may also be applied by adapting to the domain. However, this has not yet been attempted in this paper. Instead, the IND-CPA game was modified to fit the classification problem. Nevertheless, looking at the presented simulation results, such as Fig. 4, the distinguishability between two distributions is far from negligible in all three cases. Therefore, it seems to be evidence that it is not secure.

### Questions
It seems that the secret key k is used for the generation of projection P_k, but what is the specific generation method of projection from the given k? What is the size of different subspaces generated from k? 
Are projections always orthogonal to each other? If not, is there any possibility of information leaking from subspaces close to each other?

### Soundness
2 fair

### Presentation
1 poor

### Contribution
3 good

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes an 'encryption' scheme that uses language models. The basic idea is as follows:

1. The two parties share the public parameters of the model.
2. Alice finetunes the changes in the model's parameters and sends them across to Bob.
3. Bob uses the changes to recover the original message.

This doesn't work as is, but the others propose using a hidden map that maps from a low-dimensional space to the full space of the network. The hidden map needs to be known to both Alice and Bob.

### Strengths
Interesting model but several drawbacks.

### Weaknesses
There is not much in the way of security, which is the most essential part of any encryption/decryption mechanism.

### Questions
Can you provide evidence for why the scheme could be secure? Traditionally, the standard schemes have concrete quantitative assumptions under which one can prove security.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper introduced a novel approach that leverages the (unintended) memorization in LLMs to create a cryptographic system named SELM. They demonstrate for the first time that LLMs can be harnessed to implement a system capable of lossless encryption and decryption. I believe this work can potentially catapult a new line of research.

### Strengths
1. A novel approach to harness unintended memorization in LLM for the first time. 

2. The overhead of the proposed cryptographic system does not depend on the absolute number of trainable parameters in LLMs but rather on a smaller subset. This reduces the communication bandwidth requirements, a crucial bottleneck for interactive crypto-systems. 


3. The authors have presented a rigorous analysis of  (intuitive) security provided by the proposed crypto-system. 

4. Authors have also shown how the security of SELM can be improved using various regularization techniques.

### Weaknesses
1. The proposed protocol does not provide a provable security guarantee.  


2. Only evaluates short message lengths (<1000 tokens), so scalability is not yet proven.  

3. Lack of discussion on how the proposed method stacks up against traditional crypto methods like MPC (such as extended oblivious transfer or VOLE) regarding communication and computation requirements. In particular, a discussion on the tradeoffs between message length, ciphertext size, and compute and communication requirements.


**In summary, while the contributions are novel, the scalability of the proposed crypto-system is questionable, especially as an alternative to the existing crypto-primitives such as homomorphic encryption and MPC**.  Moreover, the challenge of transmitting messages securely is not new and is largely considered resolved, especially when contrasted with the complexities of performing computations on encrypted data (e.g, using homomorphic encryption).  I'm open to increasing the score, provided critical concerns are addressed.

### Questions
1. Can we further improve memorization speed using Prompt engineering strategies?

   
 2. What attack vectors or cryptanalyses are the biggest threats to SELM security?


3. Can we employ off-the-shelf LLM optimization techniques to reduce the computation burden of the SELM without compromising its security guarantee? 

4. What are the key (potential) backdoors in LLM to undermine the security of SELM?

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
3 good
