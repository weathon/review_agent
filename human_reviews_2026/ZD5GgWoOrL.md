# Machine-Generated Text Detection Requires Fewer Machine-Human Mixed Texts

- Decision: Reject
- Scores: 8, 4, 2, 4, 4

## Abstract
Machine-generated texts (MGTs) of large language models (LLMs) show significant potential in many fields but also pose challenges like fake news propagation and phishing, highlighting the need for MGT detection. Most paragraph-level detection methods implicitly assume that MGTs are entirely machine-generated and ignore the scenarios where only part of the MGT is machine-generated or inconsistent with human-generated text. To this end, this paper first reveals the prevalence of implicit human-machine mixed texts, which contain subtexts that are common to human texts, and then theoretically analyzes their impact on detection. Based on our theoretical findings, we develop a stacked detection enhancement framework decoupled from the detection model, which involves revisiting the detection optimization objective and the balance between feasibility and efficiency during optimization. Extensive experiments demonstrate its superior improvements over existing detectors. Notably, our boosting strategy can also work in a training-free manner, offering flexibility and scalability. The source code is available at \url{https://anonymous.4open.science/r/MGTD}.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper addresses the challenge of machine-generated text (MGT) detection by first revealing the prevalence of implicit human-machine mixed texts. The authors provide a theoretical analysis of how such ``mixed texts" impact detection and, based on this, propose a novel stacked detection framework. Experimental results demonstrate that the approach outperforms existing detectors, and its boosting strategy can function in a training-free manner.

### Strengths
The investigation of human-machine mixed texts is a significant and compelling aspect of machine-generated text (MGT) research. 

The authors' analysis of this phenomenon is well-founded, and the method they propose is reasonable. 

The experimental validation is thorough, and the fact that their boosting strategy can operate in a training-free manner is a particularly useful and practical feature.

### Weaknesses
To better frame the contribution, the introduction should reference existing research on human-machine mixed texts (Sec. 2.2) earlier, even if the focus of this work is different. The current framing inadvertently suggests no prior work exists on this topic. 

Additionally, the experiments would be more complete if it could include comparisons and discussion with these relevant methods.

### Questions
See the above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses Machine-Generated Text Detection. By analyzing Jaccard similarity, the authors find that even fully LLM-generated texts contain substantial overlaps with human-written content, termed implicit hybrid text. They show that a higher mixing ratio (α) increases detection difficulty and propose a Stacked Detection Enhancement Framework that filters suspected human segments (thereby reducing α) to improve detection, optimized via a Hard-EM approximation.

### Strengths
The paper discusses the phenomenon of implicit hybrid text in MGT, and I highly appreciate the authors’ insight into this important issue.

### Weaknesses
- The paper lacks an in-depth analysis of implicit hybrid text. What characteristics do such texts exhibit? Are they in machine-generated text truly misclassified as human-written by the model ? What statistical patterns do they show?
- The validation of implicit hybrid text is rather coarse:
    - The Jaccard similarity metric (based on word sets) only reflects word-level overlap and cannot accurately capture fixed expressions or compositional templates shared between machine-generated and human-written texts.
    For instance, if the word order within a sentence is rearranged, the Jaccard score remains 1, this means the *vocabulary* is identical, not necessarily the *writing pattern*.
    In addition, the paired dataset used may induce topic bias, potentially exaggerating the implicit mixing phenomenon. The authors should explore this on more diverse datasets.
    - The analysis and detection methods primarily rely on word-level or sentence-level statistics.
    However, the core behavior of generative models lies in their conditional probability distributions (logits level): even identical surface sentences can exhibit completely different logits distributions under different contexts.
    Furthermore, most existing statistical metrics for detection also rely on logits-based computations; therefore, relying solely on lexical similarity is insufficient to uncover the true properties of implicit hybrid text.

### Questions
- Is the proposed enhancement method effective only for document-level detection? Would it fail or become less applicable for shorter texts?
- Is the proposed enhancement method effective for statistical-based methods？

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The article under review proposes a two-stage method for the detection of machine generated and mixed machine-human generated text. A two step algorithm is proposed, whereby text is first run through an E-step where sentences judged very similar to human sentences are removed, and then run through an M-step where an off the shelf detector is run on the remaining sentences. 

The idea is an intriguing one which is certainly worthy of detailed explanation. A good deal of theoretical justification is given, I have my criticisms of this theoretical justification given below. The authors also keep separate a theoretical discussion of their two step process from the practical implementation, I understand why they did this but am concerned that this makes it extremely hard to understand what the authors actually do. Nevertheless, I am intrigued by the basic idea, which I think could make a very strong article if better explained.

### Strengths
1. The idea is intriguing and I think very novel. It makes a lot of sense for detecting mixed text.
2. As far as I can tell, the results are strong, with a significant uptick in performance for all the detectors tested. This is good news! I would like to see the weaker parts of this paper removed in order that the strengths can be better explained.

### Weaknesses
1. I have spent a lot of time reading the article, I believe I have the ideal background to understand this article, yet I do not understand the final algorithm used. To be precise, I think Figure 2 gives a very clear overview of what is happening, subject to understanding the E-Step and the M-Step. And I *think* (but am not certain) that the E-step is to filter out all sentences with sufficiently high Jacard score compared to a large reference database of human text and the M-step is to use an off the shelf model, but I am not completely certain. This should be explained on page 2. 
2. Throughout, the authors refer to sufficiently 'human like' machine generated text as human text. For example, they describe 'LLMs, with their powerful generation capabilities, can generate texts consistent with human writing in simple sentence structures, fixed phrases, and so on.' My understanding is that the authors refer to this as 'human text' in what follows. I think this terminology is confusing, and I don't see what it adds. Part of my confusion comes because, in Figure 2, we have to think of two different definitions of human text, one is 'written by humans' and the other is 'considered very likely written by humans by whatever filter is used at the E-step'. If the authors wish to keep this terminology, then I would like to see an extremely precise definition (e.g. a sentence is considered human if it has 100% Jaccard similarity with a sentence in the human written collection of reference texts). 
3. I think RQ1 only really makes sense with the authors odd framing off 'human text'. I would reframe it 'given a collection of reference texts, what proportion of sentences in machine generated text are identical to sentences present in the reference texts?' But on the whole, I think this framing and this question distract from the much more interesting proposition that the authors algorithm can improve algorithms for detecting machine generated text.
4. I find section 2.1, and the theoretical sections that rely on it, to be rather weak. Specifically, text is considered in two settings, firstly where each sentence is picked iid from a collection of reasonable sentences, and secondly where some limited dependence on previous sentences is introduced. This resembles neither human language nor machine generated language. In particular, the non-IID setting is still a very constricted mathematical model of language, in which (for example) the probability of a sentence appearing at position k of a text is independent of the order in which the previous k-1 sentences appear. This is a profound restriction on models of language. 
5. In the problem definiton (line 115) a characterization of the problem of detecting machine generated text is given. But this holds only if one accepts that both human and machine text are generated according to the very unnatural model of language model described in my previous comment. 

6. Throughout pages 3-6 there is a series of descriptions where the authors first propose a method, and then describe how that method is unfeasible so a modification is made. The mathematics here is difficult, and I lose track of which of the mathematics is dependent on the (to my mind unrealistic) assumptions of section 2.1, and which is not. Moreover, at no point is a final description given of precisely what the algorithm used is. I think this is a shame, again, I like your basic idea! But I think you confuse the reader by putting lots of hard maths on a very unstable set of basic assumptions about language, rather than giving a clear explanation of exactly how your algorithm works.

### Questions
1. Figure 2 looks like it should work with detectors such as fast-detect GPT for the M-step. This would be a very natural experiment to run. Did you try, and if not is there a natural reason why not? In particular, if you tried it and it failed then I think this would be useful information for the reader.

2. I do not know whether the formula on line 110 is a mathematical formula or a vague heuristic. It looks like a precise formula, and I am reading it as a precise characterization of probability distributions. But, as I understand it, expectation only makes sense when random variables take values in a vector space. Should this formula be read with a semantic mapping from sentences to some vector space? Does s_i mean the ith sentence or some mapping of the ith sentence?

3. Usually the algorithm boxes help me understand what an author has done. But in your case, I am completely unable to find the information I need, where formulae are referenced the formulae do not say which model is being used for e.g. computing the values used in equation (3). 

4. Table 1 does not include the information from Squad, fine for saving space, but given that the information from Squad looks less good for your argument I would like to see at least some discussion of it.

5. In section H.2, I am slightly concerned about different hyperparameter settings being used to generate different table entries. Can you confirm that the set which you used to choose your hyperparameters was not the test set? You write No-
tably, the default settings are obtained through grid search, aiming to show the maximum potential
of detection enhancement., again, this is all performed on a separate training set right? Could you explain the way in which different algorithms are performed on different language models, in particular does your black box setting still have different parameter settings according to which language model is generating the text?

6. Please make your article easy to skim read. I can read the DetectGPT article and in five minutes I understand both the key idea and the implementation. I think I understand your key idea after five minutes from the helpful Figure 2, but I have spent several hours with your article and still cannot say with certainty what your final implementation is.

Again, I really like your basic idea! I have tried to be direct with my comments, this is meant to be helpful, my hope is that you can make significant changes and I can be in a position to recommend acceptance.

### Soundness
3

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
4

### Summary
The paper introduced a stacked detection enhancement that uses any detector to filter high-confidence human sequences from a passage and then runs the same detector on the residual text.

### Strengths
Test in Essay, Reuters, SQuAD1 datasets and ChatGPT, GPT-4, ChatGLM, Dolly, Claude/StableLM models. And the results show consistent gains in strict low-FPR regimes.

### Weaknesses
Please check the questions.

### Questions
Removing human-like parts to help has been discussed in prior mixed-authorship and localization work, like RoFT, TriBERT, and CoAuthor. 

Using Jaccard similarity of word sets per sentence to define that many AI sentences are “human” is not correct. For example,  "Thank you for XXX" will become 100% overlap regardless of authorship.

The sample-complexity bounds rely on the assumption that each sentence is independent and identically distributed. In the real world, it's impossible as sentences are interdependent, attribution is latent and ambiguous, and α varies by domain and author. 

TPR@FPR's improvement can be caused by easier post-filter distributions rather than better AI and human separation. 

Add full ROC/PR curves and expected cost under plausible base rates.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper challenges the common assumption in AI-generated text detection that a text is entirely machine- or human-generated. It reveals that human–machine mixed texts—where parts of a document resemble human writing even if produced by an LLM—are both common and harmful to detection accuracy.

### Strengths
(1) The work is theoretically sound, presenting a clear derivation that links mixed-text proportion to detection sample complexity and empirical accuracy.

### Weaknesses
(1) The analysis assumes independence among text segments, which may oversimplify linguistic dependencies in realistic mixed texts.
(2) The STK framework improves performance across detectors, but the paper does not deeply analyze what kinds of sentences are being filtered or how the stack changes the detector’s decision boundary.]
(3) The evaluation focuses mainly on English datasets and standard text genres (essay, news, QA). It remains unclear how well the mixed-text phenomenon and the proposed framework generalize to multilingual, multimodal, or code-switched content—settings increasingly relevant to global AI text use.
(4) while cross-LLM and paraphrase robustness are tested, the paper does not analyze computational efficiency (runtime, memory) or sensitivity to the number of stacked iterations.

### Questions
(1) Could you design an experiment where the degree of mixing is systematically varied (e.g., by concatenating known proportions of human and LLM sentences) to empirically validate the theoretical curve?
(2) How would your framework handle non-English or code-switched text, where sentence segmentation and tokenization are more ambiguous?
(3) Many real-world MGT scenarios involve human-edited or paraphrased AI outputs. How resilient is the stacked detection approach to such cases, where human post-editing blurs the segment boundary between human and machine?

### Soundness
2

### Presentation
2

### Contribution
2
