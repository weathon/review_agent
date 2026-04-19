# LatticeGen: A Cooperative Framework Which Hides Generated Text in A Lattice For Privacy-Aware Generation on Cloud

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 6, 3

## Abstract
In the current user-server interaction paradigm of prompted generation with large language model (LLM) on cloud, the server fully controls the generation process, which leaves zero option for users who want to keep the generated text to themselves. We propose LatticeGen, a cooperative framework in which the server still handles most of computation while the user controls the sampling operation. The key idea is that the true generated sequence is mixed with noise tokens by the user and hidden in a noised lattice. Considering potential attack from a hypothetically malicious server and how the user can defend against it, we propose the repeated beam-search attack and the mixing noise scheme. In our experiments we apply LatticeGen to protect both prompt and generation. It is shown that while the noised lattice degrades generation quality, LatticeGen successfully protects the true generation to a remarkable degree under strong attacks (more than 50\% of the semantic remains hidden as measured by BERTScore).

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper introduces a framework named LatticeGen, which protects the privacy of user text and generated text through a method of generating token lattices via interaction between the user end and the server. Additionally, the authors propose a potential attack method, the beam-search attack, and introduce an analytical metric. They also analyze how using LatticeGen can protect privacy.

### Strengths
1.Compared to traditional NLP encryption methods, this paper considers the approach of simultaneously obfuscating both the user-uploaded prompts and the generated text, which aligns well with the inherent privacy requirements of Large Language Models (LLMs).

2.The paper also addresses potential vulnerabilities by proposing a potential attack method, the beam-search attack, and subsequently introduces a metric for analyzing privacy protection under this attack.

Besides, the clarity of the paper is commendable, with logical flow, well-defined terms, and illustrative examples ensuring accessibility to a broad audience. The use of figures enhances understanding, making complex concepts more digestible.

### Weaknesses
1.From a motivational perspective, the necessity of a collaborative and interactive process between the client-side and the server for text generation is questionable. Often, users utilizing large language models are primarily interested in obtaining results, rather than performing operations on the data. Moreover, they may not necessarily possess the computational power required for such operations.

2.The authors utilize the metric max-true-ratio to demonstrate how their privacy-preserving mechanism can withstand attacks. However, it is crucial to acknowledge that a sentence inherently possesses its own structure and semantics, and often, obfuscating just a part of it may not suffice to protect the privacy of the entire sentence. The authors do mention that there is often a trade-off between privacy and model effectiveness. In scenarios where not enough words in a sentence are obscured, it raises a concern whether the semantics of the sentence could still be exposed. This aspect deserves further attention to ensure comprehensive privacy protection.

### Questions
1.Under the attack scenario presented in this paper, could you elucidate why the collaborative and interactive text generation process is a necessary operation to protect privacy?

2.Does a max-true-ratio of 50% ensure that the semantics of the original text are protected from being disclosed?

3.Can the framework presented in this paper withstand other common attacks targeted at Large Language Models (LLMs), apart from the attack method proposed in the article?

### Soundness
3 good

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
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper considers the setting of a server computing the result of running a language model on user input and suggests the following approach to hiding the generated text (and the user prompt) from the server. instead of being given the prompt tokens by the user the user instead gives a collection on N tokens in a lattice which the server can then (at little extra cost) run the model on all of at once (up to some approximation) the user is then given N possibilities for the next token and knows which one is correct because it set the lattice up. The paper analyses a couple of rounds of attack/defence against this protocol and presents an experimental analysis of the semantic leakage and the accuracy.

### Strengths
The paper does a reasonably thorough job of considering attacks, granted they are not aiming to be exhaustive as would be very hard for something like this.
The use of lattice techniques for this purpose seems like an interesting idea to me (though I am not an expert by any means) and thus something that was worth exploring.

### Weaknesses
The intersection of the degradation in quality and the fact that by their own measure the model leaks about 40% of the semantic content anyway makes this not a clearly useful idea.
Whilst I don't know if it would be feasible in the space available tht fact that I can read the paper and not have much idea what a lattice is afterwards does seem unsatisfying as it seems to be the main technique being brought to bear.

### Questions
Could the idea of what is going on with this lattice be made clearer here?
Can you motivate the importance of hiding generated output?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper studies how to let a server generate text using a large language model such that the generated text received by the client is private.

### Strengths
The problem of keeping generated text private is an interesting one.  The paper contains ideas of how certain elements of an algorithm for this task could be performed.

### Weaknesses
The paper (and the preliminaries section) don't introduce all terminology and concepts.  While a reader may potentially read all cited work and try to understand in this way the text, the text is insufficiently self-contained to be easily digestible by a non-expert.  For example,
* The preliminaries section says that a lattice structure is used and cites two papers (which explain different things) but doesn't provide a clean definition of "lattice".  As probably "lattice" in not meant in the purely mathematical sense (a partial order with least upper bound and greatest lower bound operators), more details are needed.  
* The paper doesn't introduce "transformer".
* In "As the name suggests, we conduct a simple linearization operation before feeding it to the LM" it is unclear to what "it" refers, and it is unclear on what object the "linearization" operation is performed.
* In Sec 3.1, there is a "lattice-finetuned LLM", but the text doesn't explain what this means.


Also at other points, the presentation is hard to follow.  For example, at the bottom of page 3 the text starts to describe an algorithm informally, but there is no pseudo-code or other algorithm formalization which may help the reader to get a more precise view of what is intended.


The text makes no precise formal claims, but mainly describes an approach.
The paper presents a number of experiments, essentially investigating how robust the proposed approach is to a set of attacks the authors have selected (without much motivation on why defending against these attacks is sufficient).


There are quite a few language issues, e.g., missing articles before "cloud" (the cloud, a cloud, ...)

### Questions
--

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair
