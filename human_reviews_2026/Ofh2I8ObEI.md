# Breaking the Reversal Curse: How Masked Diffusion Models Achieve Reverse Inference

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
The reversal curse, failing to answer "$B$ is $A$" after learning "$A$ is $B$", 
is a persistent pathology of autoregressive language models (ARMs).
Masked diffusion based language models (MDMs), however, appear to escape this curse.
A seemingly plausible explanation attributes this ability to their any-order training objective,
but we show this intuition is incomplete.
In particular, training to replace the mask in "$\textbf{[M]}$ is $B$" with $A$ learns the probability $p(x=A | y=B)$,
which has nothing to do with the probability required to answer the reverse query, $p(y=A | x=B)$.
Thus, the objective formulation alone cannot explain reversal ability.
We demonstrate that the true reason lies in the architecture: in a one-layer Transformer encoder,
attention scores for forward and reverse contexts are positively correlated,
implicitly coupling probabilities that would otherwise be treated as unrelated.
This structural bias gives MDMs a principled advantage for reverse inference.
Our theory is supported by both synthetic and real-world experiments,
where MDMs consistently succeed on reverse queries that cause even strong ARMs to fail.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors examine the reversal curse: the inability of autoregressive models to reverse the logic of data given at training time. The authors note that traditional wisdom for why MDMs can overcome the reversal curse is faulty; the exact reversal logic is not covered in train time even with arbitrary masking. However, intuitively the masked training example is connected to the exact reversal via global attention scores, and the authors examine some theory for why the connection emerges. These are accompanied by empirical evaluations of large scale MDMs and ARMs on reversal tasks as well as on synthetic toy data.

### Strengths
- The authors do point out the misconception that the reversal problem is covered during MDM training due to masking, and correct it by noting that the masking that appears in training is subtly different.

### Weaknesses
- The theory result doesn't really provide much insight into what's going on empirically. As the authors remark, the "simplifying assumptions of independent, isotropic queries and keys [do not hold] in practice". While the intuition that [M] is B and B is [M] are linked by correlative attention scores makes sense, it is not clear why, a) the scores correlate without the assumptions and, b) passing through general transformer layers will maintain high probability on A for B is [M].

- A strong standard for "reversal curse" theory is Theorem 3 in [Zhu et. al Towards a Theoretical Understanding of the ‘Reversal
Curse’ via Training Dynamics], which provides a guarantee that after some point in training the reversal probability is essentially uniform. They also avoid making strong assumptions about q/k vectors, so a similar result showing a lower bound on the probability of A for "B is [M]" would be a stronger theory result to have.

- On the experimental side, the toy experiments are insightful, but corresponding analysis for general text MDMs like LLaDa are needed to confirm that these attention correlations are sustained when training on general text datasets.

### Questions
- Do the attention score correlations in Figure 6 also appear in the LLaDa model?
- Are there any empirical properties you can observe about the q/k vectors that loosen the strong assumptions of Theorem 4.1 but still make the correlation result hold?
- In the toy example, why do both attention weights increase and then decrease to a plateau during training?

### Soundness
2

### Presentation
3

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
This paper studies a phenomenon appearing in Masked Diffusion Models (MDMs) - previous work has observed that such models break the reverse curse faced by autoregressive language models. However, no formal justification were given for this phenomenon and a common hand-wavy argument of "any-order" modeling was often referred to. The authors first show that the "any-order" argument is misleading by pointing out that the two probability distributions used for reverse unmasking and reverse inference are actually different. Then they provide a new theoretical justification based on the attention score correlation, leveraging a property of the ROPE positional encoding. Finally, empirical evidence was provided by plotting the attention score correlation in real problems.

### Strengths
* The paper is well-written, has a clear scientific question to ask (why MDMs have better performance than AR models in reverse inference problems), and then provide a theoretical answer together with empirical justification.
* The theory explaining why MDMs break the reverse curse appears sound. I checked the proofs and they look correct under the given assumptions.
* The empirical evidence seems to align well with the theory. 
* I would like to also highlight one aspect - this paper is rare in the sense that it combines both statistical and architectural analysis to investigate a real question - most ML papers focus on one side of them. Papers on generative models often spend most main text playing with probability distributions and leave only few words on architecture specifics - while deep learning architecture papers seldom look at the statistical structure of data distributions. And we need more research like this in the community.

### Weaknesses
* There are some claims that need a bit more back-ups in the main text. For example, "During training, the [M] token must attend strongly to B, which increases the forward score S_fwd", although I agree this is likely to happen, it would be better to have evidence about making this an assumption.

* The attention score correlation in Figure 6 only shows the MDMs. What about the AR models?

### Questions
Please see weaknesses (I will consider increase the score if these results are provided).

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies the so-called “reversal curse” in LLMs, where GPTs are observed to fail on reverse queries while the alternative masked diffusion models (MDMs) perform significantly better. The authors provide a larger-scale empirical comparisons between AR models (LLaMA-3.1-8B, Qwen-2.5-7B) and a diffusion-based model (LLaDA-8B) and uncover that the advantages of MDMs for this task lie in their encoder architecture. Theoretical analysis is done on a simple one-layer transformer encoder architecture and toy synthetic experiments are conducted to validate their theoretical predictions. The key claim is that positive correlation between forward and reverse attention scores—induced by the encoder’s full-attention structure—allows MDMs to implicitly link p(x=A|y=B) and p(y=A|x=B), enabling reversal even without explicit training on reversed data.

### Strengths
The paper is clearly written and the storyline is easy to follow. 
The perspective of linking the correlation induced by the encoder architecture to the reversal task is novel to me. 
The theoretical analysis and the toy experimental setting are well-motivated and intuitively connected to the main claim. 
Moreover, the inclusion of controlled synthetic experiments provides concrete validation of the theory, making the overall argument coherent and convincing.

### Weaknesses
### Vague problem formulation
I don't think the reversal problem studied in this work is well-defined. Depending on the specific scenarios, "A is B" may not the same as "B is A". For examples in the paper such as "Nikki Holland’s child is Tom Holland", the reversal problem requires the model to understand the concept of child and parent, which is not discussed or modeled in this work. 
Therefore, the designed toy setting and the synthetic toy task seems less relevant to the so called “reversal curse”.
It may be more clear to just studying how LLMs learn the concept of equality, i.e., "A=B" is strictly equivalent to "B=A". 


### Limited significance and practical impact

The “reversal curse” itself is a fairly narrow behavioral artifact. Demonstrating that MDMs perform better on this particular test does not necessarily imply a more general cognitive or practical advantage. I think the work is solid within its niche, but the overall importance of the problem is low. 
It is unclear how understanding or “solving” the reversal problem translates into improved real-world functionality of LLMs. This paper would be stronger if it connected the analysis to some practical algorithms or architecture modifications for important tasks such as reasoning, compositional generalization, or real-world applications.

### Questions
* If we were to stud how LLMs learn the concept of equality, i.e., "A=B" is the same as "B=A", with the extra symbol "=", would the theoretical analysis or empirical results change?

* I am not fully convinced that results shown in Table 1 should sound any alarm to GPTs. I wonder how will stronger GPTs such as ChatGPT handle the tasks evaluated in Table 1 by pure in-context learning with chain of thoughts.

* How would the claimed architectural correlation translate to deeper, multi-layer Transformers with residual connections and multi-head attention? Does the correlation persist or diminish?

### Soundness
2

### Presentation
3

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
The paper studies the 'reversal curse' in left-to-right autoregressive language models (ARMs) where given training data of the form "A is B", the model struggles to learn the reverse direction ("B is A"). Others e.g. (Nei et al. 2025) have shown that masked diffusion models (MDMs) are effective at tackling the reversal curse.  

The authors study this here at larger scale showing that MDMs at larger scale (7-8B) outperform ARMs at the 'reversal curse'. The authors also provide a theoretical explanation by showing that under certain assumptions the correlation between the forward and reverse attention scores is lower bounded by a constant > 0.4 and provide further experiments.

### Strengths
This is an interesting problem that ARMs struggle with and the authors seek to provide both theoretical and empirical analysis.

### Weaknesses
-Its not super clear to me how insightful the theory is. Is it unique to the transformer attention or would similar correlations be present in any neural network and is this just a consequence of shared parameters? 

-The paper focuses on MDMs but is the diffusion aspect necessary? It seems that the loss at the bottom of page 2 is similar to the masked language model loss (e.g. BERT) which also uses a transformer architecture.

-Other authors have already empirically shown that MDMs are effective at the reversal curse which reduces the novelty of this paper.

### Questions
See questions above.

### Soundness
3

### Presentation
3

### Contribution
2
