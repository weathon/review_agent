# Uncovering hidden geometry in Transformers via disentangling position and context

- Decision: Reject
- Scores: 5, 6, 5

## Abstract
Transformers are widely used to extract complex semantic meanings from input tokens, yet they usually operate as black-box models. In this paper, we present a simple yet informative decomposition of hidden states (or embeddings) of trained transformers into interpretable components. For any layer, embedding vectors of input sequence samples are a tensor $h \in R^{C \times T \times d}$. Given embedding vector $h_{c,t} \in R^d$ at sequence position $t \le T$ in a sequence (or context) $c \le C$, extracting the mean effects yields the decomposition 
$$
h_{c,t} = \mu + pos_t + ctx_c + resid_{c,t}
$$
where $\mu$ is the global mean vector, $pos_t$ and $ctx_c$ are the mean vectors across contexts and across positions respectively, and $resid_{c,t}$ is the residual vector. For popular transformer architectures and diverse text datasets, empirically we find pervasive mathematical structure: (1) $(pos_t)_t$ forms a low-dimensional, continuous, and often spiral shape across layers, (2) $(ctx_c)_c$ shows  
clear cluster structure that falls into context topics, and (3) $(pos_t)_t$ and $(ctx_c)_c$ are mutually incoherent---namely $pos_t$ is almost orthogonal to $ctx_c$---which is canonical in compressed sensing and dictionary learning. This decomposition offers structural insights about input formats in in-context learning (especially for induction heads) and in length generalization (especially for arithmetic tasks).

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper investigates the intermediate representation of Transformer by viewing that each token embedding is decomposed into (i) position-wise information and (ii) sequence-wise information. There are 3 main findings --- (a) (i) forms spiral curves in a low-dimensional space, (b) (ii) contains a cluster structure, (c) (i) and (ii) are almost orthogonal --- are observed on pre-trained language models.

### Strengths
The analysis based on the decomposition is original. I haven't seen this type of decomposition of token embeddings.

The third finding (c) is an interesting property, which might open a new research direction.

### Weaknesses
The first two findings (a and b) sound relatively trivial. I think the behavior of (a) mainly comes from the sinusoidal positional embedding. The effect of positional embedding propagates to upcoming layers via skip connections, which would explain why the spiral patterns are consistently observed across layers. For (b), since ctx vector is computed by averaging token embeddings over each sequence, it's natural to contain topic-like information. More precisely, each token embedding should contain context (or topic) related information to predict the next word. Taking the average will emphasize the context information, which should be distinguished from other context information obtained from a different document.

The paper analyzes the Transformer models in many aspects. However, each analysis is not tightly connected, and it's hard to capture concrete outcomes.

### Questions
Why does Equation (7) not include the residual term?

Section 4.2 starts with the following question:
"positional information is passed from earlier layers to later layers … How does transformer layer TFLayer enable this information flow?"
Isn't this simply because of the skip connections? Also, why do you consider the low-rank plus diagonal form in Equation (8)? Don't you observe the alignment with the positional basis by svd(W) instead of svd(W - diagg(W))?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper attempts an investigation of geometric structures of embeddings leaned by transformers. It first proposes a decomposition of the embeddings to a positional component (mean vectors across context) and a contextual component (mean vectors across position). Then, it studies the geometry of each of these components. For the positional component, they find that that it is low-dimensional and smoothly varying. Concretely the Gram matrix of positions is low-rank in the Fourier domain. For the contextual component, they identify clustering structures. Finally, they find that contextual component is incoherent (almost orthogonal) to the positional component.

### Strengths
-- The investigation of geometry of token embeddings is in my opinion interesting and could shed light on the operation of LLMs

-- I like the proposed decomposition of embeddings into their global mean, positional, contextual and residual part. It is simple, but interesting

-- The authors have conducted rather thorough investigation with multiple experiments 

-- The discussion on low-rankness of positional embeddings and its connection to smoothness via fourier analysis is interesting

### Weaknesses
I am torn on my decision about the paper. I like the investigation and there are ideas in the paper which I find nice. At the same time though, I  believe the paper could benefit from an attempt to better discern and articulate the messages of the findings. Moreover, my opinion is that by discussing too many (and many of them incomplete) topics, main (and potentially interesting) messages are "lost". 

-- Several topics discussed feel incomplete, such as (1) clustering of contexts in Sec. 3; (2) content of Sec. 4.2; (3) Last paragraph on Section 5.2 (there doesn't seem to be anything informative being said here including App E other than reporting of figures)

-- The discussion on induction heads is distracting and I don't see the relevance to the rest of the paper

-- I find the presentation of the paper particularly after Sec 2 confusing. There is no clear coherence between sections/subsections. Eg., not made clear how Sec. 4.2 and 4.3 fit within the story. Overall the paper would benefit from a careful read.

*************************************************************************
AFTER REBUTTAL
************************************************************************
I continue thinking that the paper can greatly benefit from an attempt to better discern and articulate the key messages of the findings. The responses did not shed particular light on that. That said, I am raising my score as I believe the approach taken by the authors is interesting and (although not immediately clear now) could lead to new ideas towards better understanding the mechanisms of transformers.

### Questions
-- Do you have an intuition/interpretation for the spiral shape? I believe I understand the point you are making on smoothness, but what does the particular shape tells us (if anything)? If nothing, then why is it emphasized?

-- Any explanations on the non-spiral trends in Figs 9-11 in the appendix?

-- last paragraph "Why smoothness" of Sec 2: Can you please elaborate why smoothness allows attention to neighboring tokens easily? Also, you discuss there about QK scores, but those involve the WQ,WK matrices which from what I understand are not considered in Sec 2 (only gram matrix of positional embeddings)

-- How the clustering property of the contextual part of embeddings on per document basis is informative? Also, how would the results change based on the four sampled documents?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper aims to demystify the internal workings of transformer models by presenting a novel decomposition of the hidden states (or embeddings) into interpretable components. For a layer's embedding vectors, a decomposition is achieved which distinguishes the mean effects of global, positional, and contextual vectors, as well as residual vectors, providing insights into the input formats and learning mechanisms of transformer

### Strengths
**S1**. The paper introduces a novel decomposition method that separates the mean effects of global, positional, and contextual vectors within transformer embeddings. This approach offers a fresh perspective on understanding the internal mechanisms of transformers, revealing insights into how they encode and process information.

**S2**. The paper provides extensive numerical experiments on a variety of models, including Llama2-7B and BLOOM, to validate the proposed decomposition approach. These experiments include token randomization and arithmetic tasks, which demonstrate the ability of the decomposition to capture different aspects of transformer embeddings.

### Weaknesses
Please see the Questions below.

### Questions
**Section 1:**

The significance of Transformers in research is well-known, but the paper's introduction does not clarify the purpose of the proposed method. Could the authors detail how this new decomposition relates to ANOVA and previous work on positional embeddings and induction heads? Moreover, what practical benefits does this decomposition provide? The main outcomes of the experiments in both the paper and appendix also need clarification.

**Section 2:**

- Please define 'smoothness' in the context of your paper. It is essential to relate this to DFT and IDFT for better understanding.

-  The term $|| ||_{op}$ is used but not defined. 

- In Equation (6) (LHS) of Theorem 1, there is a dimension mismatch; the first term is \(T \times T\) and the second is \(k \times k\). 

**Sections 3 and 4:**

- The writing in these sections needs improvement. Starting with a summary of the key findings before referring to figures would enhance clarity. What are the main points of these sections?

- What does \(O(1)\)-sparse representation by bases mean in Theorem 2?

**Section 6:**

The claim of providing a "complete picture" is too broad. How does this research stand apart from earlier studies on positional embeddings and induction heads?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good
