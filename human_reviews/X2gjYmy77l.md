# Taming AI Bots: Controllability of Neural States in Large Language Models

- Avg Score: 4.00
- Decision: Reject
- Scores: 5, 1, 5, 5

## Abstract
We tackle the question of whether an agent can, by suitable choice of prompts, control an AI bot to any state. We view large language models (LLMs) and their corresponding conversational interfaces (AI bots) as discrete-time dynamical systems evolving in the embedding space of (sub-)word tokens, where they are trivially controllable. However, we are not interested in controlling AI Bots to produce individual words but rather sequences, or sentences, that convey certain ''meanings''. To tackle the question of controllability in the space of meanings, we first describe how meanings are represented in an LLM: after pre-training, the LLM is a deterministic map from incomplete sequences of discrete tokens to an inner product space of discriminant vectors (''embeddings'') of the next token; after fine-tuning and reinforcement, the same LLM maps complete sequences to a vector space. Since no token follows the special end-of-sequence token during pre-training, that vector space can be co-opted to represent meanings and align them with human supervision during fine-tuning.  Accordingly, ''meanings'' in trained LLMs can be viewed simply as equivalence classes of complete trajectories of tokens. Although rudimentary, this characterization of meanings is compatible with so-called deflationary theories in epistemology. More importantly, defining meanings as equivalence classes of sentences allows us to frame the key question as determining the controllability of a dynamical system evolving in the quotient space of discrete trajectories induced by the model itself, a problem that to the best of our knowledge has never been tackled before. To do so, we characterize a ``well trained LLM'' through conditions that are largely met by today's LLMs and show that, when restricted to the space of meanings, a well-trained AI bot is controllable under verifiable conditions. More precisely, we introduce a functional characterization of AI bots, and derive necessary and sufficient conditions for controllability. The fact that AI bots are controllable means that they can be designed to counteract adverse actions and avoid reaching undesirable states before their boundary is crossed.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors consider the problem of controlling LLMs. In general, the authors seek to find whether an arbitrary well-formed output can be generated from a well-trained LLM.

The authors characterize LLMs as discrete-time dynamical systems, moving in the token or embedding space. On top of this low-level representation space, the authors consider "meaning" spaces based on discriminators.

Overall, given some assumptions about the surjective nature of LLMs within the set of meaningful sentences, the authors conclude that well trained LLMs are controllable.

### Strengths
## Originality
This work appears relatively novel, considering the important problem of LLM controllability and bringing insight from dynamical systems literature to characterize controllability.

## Quality
For the most part, this is a very carefully written theory paper that lays out important assumptions and characteristics for notions of controllability.

## Clarity
The paper was largely clear. I tend to favor figures or diagrams, but the first figure (which I found helpful) only appears in the appendix. I confess that I am typically more of an empiricist than a theorist, but I am familiar with LLMs, dynamical systems, and the math presented in the paper, which I was able to follow.

## Significance
I am undecided about the significance of this work for one particular reason: Postulate 1. The authors are considering a very important problem that many people care about - if a paper can establish that real-world LLMs that we use are controllable, that would be an extremely important paper. The problem, in my mind, is that to reach this conclusion, the authors make a very big assumption about well-trained LLMs in Postulate 1. To their credit, the authors note this limitation, both in the main text and Appendix 1. Overall, I fear that the strength of the assumption in Postulate 1 could significantly limit the significance of this work.

### Weaknesses
## Postulate 1: 
As noted earlier, the biggest weakness of this work is the strong assumption in Postulate 1 about the bijective/invertible nature of the map $F_w$. To me, this assumption almost gives away the controllability conclusion: if there is an invertible map, then it at least intuitively follows very clearly how such a system would be controllable.

## Presentation of results:
I recognize that the primary contribution of this work is theoretical, rather than empirical, in nature. However, the authors do conduct experiments and present results in Appendix A. Even while keeping results in appendices, I would encourage the authors to at least provide a sentence or two in the main paper highlighting high-level trends of the experiment results.

I wish to emphasize again that I traditional study more empirical AI methods and analysis. Thus, if other reviewers find that this work would contribute to the more theory-focused community, I am happy to change my mind.

### Questions
1. In Section 3, the authors describe how, "For a sufficiently high temperature parameter $T$..." random sequences can be generated from any start token. This is true for any $T > 0$, right (barring numerical precision errors from very small numbers)? I do not see why one would need a larger value of $T$ for this theoretical claim (although I recognize that in practice, higher $T$ would "spread out" faster to cover more random sequences).

2. As stated earlier, my main concern about this paper is Postulate 1. What evidence or arguments do the authors have supporting this assumption?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
1: strong reject

### Rating Number
1

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper aims to define "meaning" in the context of LLMs, use this definition to characterize well-trained models, and establish the conditions for controllability of LLMs.

### Strengths
The paper sets out to formally characterize controllability of LLMs which is an important issue in preventing adversarial attacks on language models and preventing LLMs from producing undesirable content.

### Weaknesses
It's unclear what the contribution of the paper is. The related work section does not connect this paper to specific prior work (only citing two survey papers). Then, most of the paper is spent discussing preliminaries and introducing notation and definitions. The first use of the proposed definitions to make a non-trivial claim is in Section 5 with Postulate 1, but this claim is not justified. Theorem 1 in Section 5 seems to follow directly from the proposed definition of controllability. Theorem 2 is presented with no intuition and the proof in the appendix is only for a special case. It's also not explained how Theorem 2 justifies the main conclusion: that "a prompt engineer aided by enough time and memory can force an LLM to output an arbitrary sequence of ℓ tokens."

### Questions
See weaknesses section.

### Soundness
1 poor

### Presentation
1 poor

### Contribution
1 poor

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper hypothesize that the state space of a language model is sentences understandable to human (rather than sheer tokens or sequence of tokens). It then tries to provide a theory for controllability of language systems in the meaning space.

### Strengths
I appreciate the effort to formalize controllability of LLMs as a generic question. This is an interesting topic that can spur further research and help predictability and understand the LLM's behaviours in general. However, the presented theory suffers from various core issues.

### Weaknesses
- Definition of equivalence seems to be insufficient and lacks important components. 

- Definition of meaning seems to defeat the whole purpose of this paper, as it allows for any gibberish/random sequence of tokens to still induce a meaning and possibly a set of many other gibberish sequences to be in their equivalent class. 

- How to find a discriminant for meaning is left out as the authors explicitly assume that “the mechanism is provided by human annotators and other providers of training data.” While the authors emphasize in the introduction that such information can be used in the LLM training without external reward model: “This observation shows that sentence-level annotations can be incorporated directly into the trained model without the need for any external reward model nor external policy model, simply by sentence-level feedback,” I do not see the advantage of this approach over using the very same data to train a reward model and use that either during the training (as in RLHF) or as an augmentation (as in Rectification method), the latter indeed provides quite strong theoretical guarantees. If this secondary goal is valid, the paper requires sufficient reasoning for why the presented approach is superior. If not, the presentation requires to change and reflect only the controllability analysis.

### Questions
- Last paragraph of introduction: “we hope to encourage the design of actual new control mechanism” --> It looks like the authors are not aware of the [*Rectification* method](https://arxiv.org/abs/2302.14003), which is indeed a formal control paradigm for avoidance any definable “meaning,” as in the terminology of this paper. See the point bellow.

- Let me first emphasize that I am aware of the differences between these two works (no need to tell me they say X while we say Y). My main point is to help you broaden your research and make the current paper be more comprehensive and useful for the readers, also this is aligned with your last paragraph of the introduction: The concept of probabilistic avoidance, which is akin to *stochastic* controllability but in preventing a meaning from being reached is [studied before](https://arxiv.org/abs/2302.14003). Even though the authors of that paper did not call it controllability (they call it security condition which also comes [from an older paper](http://proceedings.mlr.press/v97/fatemi19a/fatemi19a.pdf)), but it is indeed stochastic controllability in the context of avoidance. To use the terminology of this submission, what that work shows mathematically is that the LLM’s output probabilities can be minimally corrected and that will guarantee the avoidance of any prescribed meaning of interest (the one under their study was toxicity, but essentially the same analysis is applicable to any other meaning). Importantly, such results show that a certain optimal value function corresponding to reaching the prescribed meaning (with reward of -1 and no discount) is key at least in principle for the avoidance, which also highlights the sufficiency of RL as the core learning paradigm for closed-loop controlling/steering an LLM. 

- Section Preliminaries, first part --> An LLM in its standard form provides a simplex over token space, not an embedding + a metric. How can an LLM be a discriminant?

- Section Preliminaries, second part --> The definition of equivalence is quite confusing. It looks like that equivalence requires both discriminant and a set of classes, no matter how they are defined (does it?). However, the presented definition only uses discrimanent, which seems incomplete. In that case, $x^1 \stackrel{\phi}{\sim} x^2$ is ambiguous.

- Definition 1 --> [related to the point above] without a set of classes (meanings), the definition is incomplete. Indeed, $\phi$ only maps a given sentence to a metric space, that is a vector $\in \mathbb{R}^K$ (plus a given metric), nothing more. One needs to have (i) a set of classes in addition to (ii) a criteria (like argmax in your example) to define equivalence. 

- By extension, $[x]$ does not provide a comprehensible set by just using $\phi$. Intuitively, one can ask, given another vector $y$, then $<\phi(x), \phi(y)>$ must be what in order for $y$ to be part of $[x]$?

- Even if the above issue is resolved, the provided definition of meaning lacks any connection to linguistic meanings, as any random sequence of tokens will induce its own $[x]$. This conflicts with the authors initial claim that they would like to distinguish between a meaningful sentence and a random sequence (first line of page 2). Importantly, this is totally separate from the question of what a good $\phi$ is, and it hold from *any* given $\phi$.

- As for selecting $\phi$, the authors say “Here, we assume that the mechanism is provided by human annotators and other providers of training data.” How is this different from an external reward model, as the authors named as a disadvantage of RL-based methods? (Note: the reward is defined for a complete text not a given point at the middle, so these seem to be exactly the same.)

- Eq 1 --> This is confusing. The range of $\phi$ is $\mathbb{R}^K$, where $K$ was the dimension of matric space, yet the output of $\phi$ is a probability distribution (which is a simplex).

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper studies whether an agent can use prompts to steer LLMs to generate any sentences as wanted. A dynamical system perspective is adopted, and LLMs are viewed as discrete-time systems evolving in the embedding space of tokens. The authors first describe that  ''meanings'' in trained LLMs can be viewed as equivalence classes of complete trajectories of tokens. Based on this viewpoint, they end up with a question of determining the controllability of a dynamical system evolving in the quotient space of discrete trajectories induced by the model itself. Several conditions for controllability are then provided.

### Strengths
The paper is very original, and presents some unique idea on how to determine the controllability of "well-trained" LLMs. Connecting LLMs with control is definitely interesting. A new theoretical perspective is developed. This paper may also inspire more researchers to think about the fundamental theory of LLMs.

### Weaknesses
1.  The characterization of meanings seems quite subjective. The authors mentioned that their characterization of meanings is compatible with the deflationary theories in epistemology. It seems that this explanation itself does not justify why such a characterization is meaningful for studying the controllability of LLMs in general.

2. The authors claim that their conditions are largely met by today’s LLMs. More justifications are needed. This also seems a hand-waving statement. "Largely met" means "not always met"?

3. Many of the equations are quite difficult to understand. I have tried very hard to follow the theoretical arguments in this paper. However, I still feel very confused in the end. I will ask some questions in the "Question" section.

### Questions
1. Does Definition 1 assume some sort of underlying classifier such that the equivalent classes can be defined? What is that specific classifier? The authors mentioned "the mechanism is provided by human annotators and other providers of training data." This is very confusing. I don't understand what exactly this phi is.

2. Equation (6) is very confusing. I mean, in Equation (4), y is sampled from the softmax operation. Then all of a sudden, it becomes an additive noise? It seems that the noise n_t depends on x_{1:t}?

3. Regarding Definition 3, an LLM is well-trained if theta is any small positive number?

4. What does Claim 1 mean? Things that happen before C do not matter? Then how large does this C need to be for practical LLMs? I mean, suppose for a task of writing a book, C has to be really large?

5. What does Postulate 1 really mean? How to justify this?

6. How can we justify the assumptions in Theorem 2?

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair
