## Human Reviewer 1

### Summary
The authors study the concept of memory in LLMs. They draw inspiration from recent work demonstrating how UAT can be applied to Transformer architectures, propose a new definition of memory that can be applied to both humans and transformer LLMs, and introduce a concept of "Schrodinger's Memory" - a type of memory that can only be detected upon probing. Lastly, the authors run a few experiments, demonstrating remarkable memory capacity of transformer architectures and proving conclusively that transformers are capable of a certain form memory.

### Strengths
The paper addresses an important topic. It is crucial that we study in detail the similarities and differences in memory processes between humans and LLMs, as well as devote plenty of effort do understand how memory manifests in transformer architectures.

The paper is relatively clearly written.

The work has both a theoretical and a practical component. It is always commendable when practical experiments are informed by theory.

### Weaknesses
** Novelty **
Unfortunately, the paper is not sufficiently novel to pass the high standards of the ICLR conference. The main empirical result in the paper is a rediscovery of a well-known phenomenon of overfitting. As is well-known, LLMs do indeed possess a remarkable ability for verbatim memorization of items in the training set.

**Quality**
Unfortunately, the quality of the literature review or the theoretical justification of the proposed work is not sufficient to pass the high standards of the ICLR conference. The authors pose a number of rhetorical questions, such as "So, is this sentence stored within a single neuron?" entirely ignoring decades of memory research in Psychology, Cognitive Science, and Neuroscience. It is also staggering to see them claim 'in summary, the term ”memory” was traditionally used to refer specifically to human memory before the emergence of LLMs'. This entirely ignores the research on memory in animals and even plants, not to mention the related concepts of the memory of materials, cultural memory, and a plethora of other related concepts.

I find it commendable that the authors ask ambitious, fundamental questions, such as "moreover, if this memory is stored in a fixed set of neurons, then every time the question is raised, the response should be identical, since the retrieval would be from the same static content." Unfortunately, however, such questions have been extensively studied not even for centuries, but for millennia. If these questions are truly of interest, I suggest starting with the works of Pavlov, Skinner, and Tolman to gain a historical perspective on how this and related questions have been approached by the scientific community.

### Questions
I struggle to understand why the authors introduced the concept of Schrodinger's Memory. It seems extremely broad. Isn't human memory also an example of Schrodinger's Memory? We have no other way of finding out whether a human has a specific memory other than presenting certain stimuli and recording reactions.

Moreover, some aspects of the Schrodinger's experiment analogy don't work: for example, the memory is not in an indeterminate state before querying. Moreover, the act of querying (observing) does not affect the memory in the same way as the wave function is affected in quantum experiments.

Suggestions:
I deeply hope that my negative review does not prevent the authors from further pursuing research in this direction. I highly suggest taking a step back and focusing on a more narrow aspect of LLM memory. It is also absolutely crucial to perform a thorough, deep literature analysis before performing any experiments. For example, the authors currently ask a lot of question about human memory organization in the brain, but they never mention an enormous body of literature that has been written on the topic. Similarly, the main result they obtained is a well-known phenomenon called "training set overfitting". It might be difficult to navigate the many unspoken rules of academic research, hence I highly suggest that the authors seek input from members of scientific community - mentorship arrangements, internships, and other forms of peer guidance help authors create a better and refined version of their research.

### Soundness
1

### Presentation
2

### Contribution
1

### Rating
1

### Confidence
4

---

## Human Reviewer 2

### Summary
The authors investigate the concept of memory for LLMs via “memory ability assessment”. To this end, they create a novel definition of memory based on an input-output relation and connect the Transformer architecture to the Universal Approximation Theorem. They claim that the weights and biases of Transformer-based LLMs can “dynamically change according to the input” as the basis of memory. Empirically, they show that fine-tuning LLMs on English and Chinese corpora of poems enables them to recite the latter based on the information about the author and title (and dynasty for Chinese poems). They conclude by comparing human brains and LLMs, discussing the model dependence on model size, data quality, and quantity.

### Strengths
The general idea of connecting deep theory about functions to concrete Transformer-based LLMs (for example, via the Universal Approximation Theorem) is interesting and promising.

### Weaknesses
Unfortunately, the article suffers from several shortcomings that I will point out section-wise:


Introduction: 

Overall, the main topic of the article should be clarified. There needs to be a proper section on related work to outline and delineate current research on this topic so that readers know what the state-of-the-art is and why this paper's contributions are contributions in the first place. This would also help readers unfamiliar with the topic to navigate the article more efficiently. Also, a large portion of the article is concerned with the Transformer architecture, but the original paper is not cited.



UAT and LLMs:

Overall, the formulation of the theoretical background severely lacks precision and formalism. A significant portion of the notation has not been defined (for example, $C(I_n)$) before using it. Furthermore, the authors write (starting in line 095/096):
 
“[...] then a finite sum of the following form: [...] is dense in $C(I_n)$”

However, "a finite sum" can not be dense in the set of continuous functions on the hypercube (which is meant by $C(I_n)$). The authors then continue to reference the article by Wang & Li (2024b), stating (line 131):

"[...] parameters in the multi-head attention mechanism are modified dynamically in response to the input."

It is unclear whether this refers to the forward or backward process. Overall, the entire section is very confusing, with statements like (starting in line 137):

"[...] the UAT's parameters are fixed once training is completed, [...]"

"UAT" stands for Universal Approximation Theorem - what is meant by parameters?

A large portion of section 2.1 is very similar to the beginning of Section 2 of the paper by Wang & Li (2024b) (who also do not cite the original transformer paper). As an example, the error in line 102 is the same ($\theta \in \mathbb{R}$ needs to be $\theta_j \in \mathbb{R}$).  

Finally, the reference "(Cybenko, 2007)" refers to the article published in 1989 (the same year Hornik et al. published their paper) - is this reference incorrect?



The Memory of LLMs:

This section discusses memory for humans and LLMs and introduces the authors' definition of memory. They criticise other works for the lack of a fundamental theoretical framework and vague definitions of memory (see line 059 and following), but the definition presented in this work seems no different in these regards. The authors should tie their definition to the introduced theory in the previous section and make the formulation more rigorous.

Overall, there is no cited work (apart from the Wikipedia definition) in Section 3.1, which seems more like a blog post than an academic article. This leaves statements like (line 168)

"The brain does not have a structure analogous to a database for storing information."

unfounded. 

The datasets are attributed to "Unknown" - although the Huggingface user accounts are available via the provided links.

The authors calculate the mean average accuracy to evaluate the models' ability to recall poems. However, it is unclear when a poem counts as predicted correctly. Did the authors employ plain string matching? If so, how are newline characters and translations by spaces handled? 

Furthermore, most training details are missing. For example, what were the learning rate and batch size? Based on the provided information, no experiments are reproducible (no code is available).

Regarding Table 1: What do the hyphens stand for?

The authors also claim in line 253/254 that—based on the results in Table 1—" LLMs possess memory capabilities, which align precisely with the definition of memory we established."

In my view, the authors seem to reduce memory to a capability every overfitted LLM can develop: reproducing tokens in order. Here, I use the term "overfitted" as training for 100 epochs on such a comparatively small corpus of text (2000 poems) seems excessive.

Finally, given the same input, if the LLM could reproduce the poems in reversed order, starting with the last token (or character) and ending with the first token (or character) - would this count as memory according to the provided definition and would the metric in Equation (4) reflect this?



A Comparison Between Human Brains and LLMs:

Similar to Section 3.1, the discussion about similarities and differences between human memory and memory in LLMs at the beginning of the fourth section lacks academic references and empirical evidence to support claims such as (line 373/374)

“These poems are not stored in specific areas within the model; they are dynamically generated based on input.”

and (line 395)

“Although the predictions in Figure 3 are incorrect, they still align with linguistic conventions and somewhat correspond to the titles of the poems. This can be seen as creativity.”

Some related articles discuss the case of Henry Molaison, which led to the authors hypothesising about similar dynamics for LLMs, but these are likewise unfounded. In particular, I do not see sufficient evidence for the third contribution mentioned in the introduction.  

The authors likewise state in line 082 that they

“[...] conduct a comprehensive analysis of human and LLM abilities, with a focus on memory ability.”, 

which I also do not see provided.



Overall, the article severely lacks formalism, experimental details and references/experiments to support the author's claims. The introduced definition of memory needs to be improved, that is, sharpened, theoretically motivated and empirically justified. 







Minor: 

Grammar and spelling mistakes need to be corrected, for example:

Line 137: “This ability enables the Transformer to adaptively fit based on the input [...]”

Line 219: “We select the poems from datasets and the requirement is the combined length of the input and output to a maximum of 256 characters.”

Line 241/242: “[...] are the prediction and ground true of the i-th exmple.”

Line 351/352: “After fine-tuning the model for 100 epochs on CN Poems, the results are shown in Table 2.”

Line 406/407: “The larger and higher the quality of dataset, the stronger [...]”

### Questions
See Weaknesses.

### Soundness
1

### Presentation
1

### Contribution
1

### Rating
1

### Confidence
4

---

## Human Reviewer 3

### Summary
This work first relates the LM architecture to the UAT, then proposes a new definition of memory in LLMs, and tests the memorization capabilities of Qwen and Bloom models on Chinese and English poems, showing that LMs can memorize poems after 100 epochs of finetuning. It conclude with a comparison of memorization between brains and language models.

### Strengths
The paper investigates memory in LLMs, which is timely given the current explosion of research on LMs, and is thematically aligned with ICLR.

The decision to test memorization on poetry is a step in the right direction.

### Weaknesses
While the experimental work is a step in the right direction, I do not believe the paper in its current form would be a good fit for ICLR.


1. The authors aim to provide a precise definition of memory, yet the definition they provide is informal, and the only other definition they compare to is one from Wikipedia (ignoring the vast literature on, e.g., dense associate memory, episodic vs working memory etc).
	1. While the goal of Section 3.1 is to define memory, the definition provided is imprecise. For instance, is memory a function from inputs to outputs? Is it a function from (inputs, outputs) to a truth value? The authors go on to use next-token prediction accuracy as their metric for memory in Section 4. I suggest the authors to formally define memory in the next version of the manuscript. 

2. The work presented is quite speculative, and the tone of writing imo would be better aligned to a non-ML conference.
	1. Section 3, especially 3.1, reads to be long-winded and imprecise-- while it tries to define memory, it instead provides many examples without a formal definition of memory.
	2. Section 4, which compares brains to LMs, does not cite the relevant literature on neuroscience (except for one case study in l427). 
	3. l424 "Each update may be right or wrong, but with a vast number of humans exploring the world, we gradually inch closer to the truth, ultimately leading to innovation." This line (and similar ones) should ideally be toned down or omitted for a machine learning conference.

3. The work does not engage with the vast literature on memory in neuroscience, cognitive science and machine learning. Instead, it overinterprets and over-draws parallels between neural networks and human cognition. I'm listing several examples here:
	1. l269 The connection to human cognition requires a citation.
	2. l421: It could be viewed as the weight parameters in our brains are randomly initialized... 
	3. Table l207-212: Answer 2,3,4 -> Answer 1, 2, 3. I disagree that one would conclude "minor distortions, severe distortions, and memory loss" from these examples. If the answer is "I do not know", then it could be that the person never learned Newton's first law.

4. Methodology
	1. The selected poems are common ones likely found in training data, especially for the Chinese language models. Therefore, it is not guaranteed that each poem is trained on the same number of times (t=100 epochs). How do you disentangle the effects from pre-training on these poems?
	2. The memorization metric based on accuracy is not sufficiently different from Carlini et al 2022, which defines memorization as $k$-extractability, $k$ being the number of tokens in the input prompt. 
	3. The connection between experiments and UAT is never made.

5. Several unsubstantiated claims about linguistics
	1. l271 "Chinese is a more complex language" is vague-- I would remove this sentence altogether or specify exactly which components of Chinese are more complex than English with citations from the linguistic typology literature.
	2. l214-215 is rather abrupt and would go better in the introduction of the section. The second sentence "Now, we believe that LLMs also exhibit memory"-- needs citation

6. Lack of engagement with the UAT / LLMs literature [1-3], which show that Transformers, in the general case, are not universal approximators-- only under certain conditions.

[1] Alberti et al. 2023. Sumformer: Universal Approximation for Efficient Transformers
https://proceedings.mlr.press/v221/alberti23a.html

[2] Kratsios et al. 2022. Universal Approximation Under Constraints is Possible with Transformers 
https://openreview.net/forum?id=JGO8CvG5S9

[3] Luo et al. 2022. Your Transformer May Not be as Powerful as You Expect
(RPE based transformers aren't Universal Approximators)

### Questions
See weaknesses

### Soundness
1

### Presentation
1

### Contribution
1

### Rating
1

### Confidence
5

---

## Human Reviewer 4

### Summary
This work explores the memory capabilities of LLMs using the Universal Approximation Theorem (UAT). The authors introduce the concept of "Schrödinger's memory" - suggesting that LLMs' memory only becomes observable when queried and remains indeterminate otherwise. The paper presents experimental results comparing different models' ability to memorize Chinese and English poems and draws comparisons between LLMs and human memory mechanisms.

### Strengths
The paper tries to address an important and timely question about the memory mechanisms in LLMs;  
The choice of structured datasets (particularly poetry) is suited for testing memory recall in LLMs and evaluating the relationship between model architecture and memory capacity;  
The attempt to connect UAT with LLMs memory is interesting and provides a mathematical basis for discussing dynamic response mechanisms in LLMs;  
The comparison between human and LLM memory provides insights that could bridge cognitive science and AI.

### Weaknesses
1.The paper claims to use UAT to explain LLMs' memory abilities, but the connection between Eq. (3) and the memory mechanism is not rigorously established. The authors need to prove how the dynamic fitting capability directly relates to memory retention. This connection might be strengthened by providing a step-by-step derivation linking Eq. (3) to specific memory processes, or designing an experiment that directly tests the relationship between dynamic fitting and memory retention.  
2.While the paper develops an extensive theoretical framework using UAT in Section 2, the experiments in Section 3 do not directly validate or connect to this theory. This gap might be addressed by quantitatively relate the parameters in Eq. (3) to the observed memory performance in the experiments.  
3.Section 3.3 proposes using memory ability as an objective measure of LLM capabilities but does not relate this metric to standard benchmarks. The authors should demonstrate how their memory assessment correlates with or complements existing evaluation methods.  
4.The output length effect analysis in Section 3.4 only tests up to 512 characters. The authors should investigate how memory performance degrades with sequence length and compare this with theoretical predictions from their UAT framework.   
5.The paper fails to examine an essential aspect of memory systems - their ability to recognize unknown information. Specifically, there are no experiments testing whether models can reliably indicate when they encounter previously unseen poems, nor is there a comparison between "memorized" vs. "hallucinated" outputs' characteristics. The paper could benefit from adding specific experimental designs to test the models' ability to distinguish between known and unknown information. This might be achieved by including a set of previously unseen poems in the test set, or evaluating how the models distinguish between known and unknown information.  
6.The experiments only measure binary success (correct/incorrect recitation) without examining the deviations as metioned in Section 3.1.  
7.The comparison between human and LLM memory is largely speculative and lacks scientific rigor. Many claims about human memory mechanisms are made without proper citations or empirical support. This could be improved by designing experiments to more rigorously compare human and LLM memory, perhaps drawing on established methods from cognitive psychology.

### Questions
1.How do you justify that the UAT format in Equation 3 specifically relates to memory rather than a general dynamic fitting argument? For instance, are there specific properties of UAT that correspond to the characteristics of memory, such as retention, retrieval, or even forgetting? How would you define these aspects mathematically within the UAT framework?  
2.Which specific aspects of memory can the theoretical framework of UAT explain better than standard Transformer-based fitting capabilities?  
3.Can you provide theoretical bounds or guarantees for the memory capacity of LLMs based on your UAT analysis?  
4.How does the proposed memory assessment method account for different types of memory (e.g., factual vs. procedural)?  
5.Memory typically involves not only retention but also changes in recall accuracy over time or with repeated exposure to new information. Did you investigate how memory in LLMs degrades or strengthens with time, additional training, or varying contexts?

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
3

### Confidence
4