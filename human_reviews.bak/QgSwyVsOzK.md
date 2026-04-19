# Modeling Knowledge as Functionals for Knowledge Reasoning

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 5, 8, 1, 6

## Abstract
A bottleneck for developing general artificial intelligence is empowering machines with knowledge-reasoning capabilities to facilitate NLP tasks such as semantic search, reading comprehension, and question-answering.
Prior arts focus on integrating distributed knowledge embeddings and representations of pre-trained neural language models to produce outputs; however, there are still large areas for improvement in performance, explainability, and sustainability.
In this paper, we propose to represent ${\bf K}$nowledge ${\bf as}$ the ${\bf F}$unctional representation (${\it KasF}$) with a dynamics-based mechanism that simulates the semantic flow amongst tokens to facilitate knowledge reasoning.
The method utilizes a superposition of semantic fields to represent knowledge by building a dynamical mechanism to compute the similarity between semantic units.
This mechanism comprehensively captures the semantic features and eliminates ambiguities in representing entities and relations.
We first evaluate our ${\it KasF}$ on the WikiQA dataset to demonstrate its superiority in capturing semantic patterns.
Next, we evaluate our ${\it KasF}$ modules on the SQuAD2.0 dataset by replacing the last layer of pre-trained language models fine-tuned on this dataset.
We observe consistent improvements in accuracy with fewer parameters.
Then we evaluate ${\it KasF}$ on the CommonsenseQA benchmark.
On the official blind test set, we achieve state-of-the-art with a single model, 
outperforming the prior best ensemble and single models by $0.4\%$ and $3.1\%$, respectively.
It is worth noting that the prior best single model is $47\times$ larger than ours.
Further experiments also demonstrate that ${\it KasF}$ exhibits superiority in dealing with sophisticated sentences.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper considers that explicit mechanism for incorporating / exploiting knowledge is needed for LLMs to conduct knowledge-intensive tasks. Although existing methods are computationally efficient, they work differently compared to the way human brain works. Borrowing ideas from neuroscience, this paper proposes to represent knowledge as functionals to construct semantic fields to hold relationships among tokens within a text. From the empirical study, the paper shows the superiority of their proposed approach on several classical QA tasks (i.e., WikiQA, SQuAD 2.0, CommonsenseQA).

### Strengths
1. The paper addresses a critical issue within the domain, which is to promote the knowledge reasoning within LMs / LLMs. The idea of paper's proposed approach, which is inspired from neuroscience as stated, to treat knowledge as functionals is novel.
2. From the empirical study the proposed approach is promising.

### Weaknesses
1. This paper is somewhat difficult to follow. From the introduction part, the motivation to treat knowledge as functional representation is somewhat unclear for me. I am not sure about the reason to introduce functional representation in Sec. 2.1 because it seems that Eq. (4) - (9) themselves do not contain too much about functionals. 

2. Although the paper claims that they aim at incorporating KGs into LLMs for knowledge-intensive task, the effectiveness of the proposed approach with LLMs is lacked in the empirical study, as well as the baselines.

### Questions
1. Could you explain in more details why the concept of functional is important in KasF?
2. Could KasF be easily transferred to the use of LLMs?
3. What is the additional computational cost brought by KasF compared to the backbone language model?

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
In this paper, the authors address the existing gaps in knowledge-reasoning capabilities, highlighting the need for enhanced performance and sustainability. To tackle this issue, they introduce a novel approach named Knowledge as Functional representation (KasF), which leverages a dynamics-based mechanism. This mechanism is designed to simulate the semantic flow amongst tokens, thereby facilitating the process of knowledge reasoning. The authors present empirical evidence to demonstrate the superiority of KasF in capturing intricate semantic patterns, showcasing consistent improvements in accuracy while utilizing fewer parameters compared to traditional methods.

### Strengths
1.The introduction of KasF stands out as a new method, utilizing a superposition of semantic fields to represent knowledge. This is achieved through the development of a dynamic mechanism, which calculates the similarity between semantic units, ensuring a more precise representation.

2.KasF exhibits an impressive ability to comprehensively capture semantic features, effectively eliminating ambiguities in the representation of entities and relations. This leads to a more robust and accurate knowledge-reasoning process.

3.The paper is well-written, particularly in the methods section, where the authors provide clear and concise explanations of their approach, making it accessible to readers.

### Weaknesses
1.Despite the complexity and innovation of the proposed method, there is a sense of lack of novelty, especially when viewed in the context of semantic compression or representation enhancements applied to knowledge reasoning tasks.

2.The results presented in the experimental section are not entirely convincing, with the authors relying on outdated baseline models for comparison, which undermines the validity of their findings.

### Questions
1.The example provided in Section 2.1 of the methods part seems inadequate. Why not illustrate the concept with a more direct example, such as a Question-Answering (QA) scenario? The current example on semantic compression is not as intuitive and might not effectively aid in understanding the method defined in the paper.

2.In Table 2, only the time cost for KasF is listed. For a comprehensive comparison, it would be beneficial to include the time costs associated with other methods as well.

3.Regarding Table 4, which specific model from the GPT-3.5 series was selected for the comparison? A clarification on this would enhance the transparency of the experimental setup.

4.The authors emphasize KasF's advantage over FC Linear in terms of having fewer parameters. However, it would be more logical to compare KasF with other semantic compression methods to provide a fair and accurate assessment.

5.To strengthen the credibility and significance of KasF, applying it to newer knowledge reasoning datasets or integrating it with open-source large language models like LLaMa would be a valuable extension of this work.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a novel representation method, "Knowledge as the Functional (KasF)", for knowledge reasoning tasks, inspired by the semantic field concepts in biological and linguistic study. Specifically, based on the dynamics inspiration, a sentence is treated as a semantic flow among semantic units (tokens/words). The representation is formulated as a functional, from the initial to the end token of a sentence, with a task specific objective. The authors implement the formulation with LLMs like RoBERTa, ALBERT as base models. They evaluate the proposed KasF on multiple QA benchmarks, i.e., WikiQA, SQuAD2, and CSQA. Specially, on CSQA, KasF achieves the state-of-the-art single-model performance on the blind test set.

### Strengths
* This paper is well-written and easy to understand.
* The proposed method has a clear intuition and gets nice performance on QA benchmarks.

### Weaknesses
I don't see clear weaknesses in this paper.

### Questions
* Would it be possible to adapt the approach beyond classification-style QA tasks? e.g., generative QA, reasoning on knowledge graphs, etc.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
1: strong reject

### Rating Number
1

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
I confess I don't really understand more than 10% of this paper but I will try to provide a summary of parts that I understood.

Overall, the paper tries to provide a new mechanism to improve performance on reasoning tasks by encoding various inputs (queries, documents, etc) using "functionals". These functionals are somehow supposed to capture multiple meanings associated with any input word and group provide a grouped representation of words with shared semantics in context. In Section 2, the authors provide a description of their method but I don't understand any part of it since most of the definitions of mathematical symbols are either assumed or just not provided. The first task that authors initialize their method for is semantic compression, which in standard NLP, is simply the task of mapping input text (queries, etc) to simpler representations (for example, remapping the tokens with hypernyms, using distributed representation, etc). In Section 2.1, the authors are trying to do this compression for a query vector defined as $y \in R^{D_v \times 1}$, yet  it is unclear what D_v is. They also use a symbol $V \in R^{n \times D_v}$ but again it is unclear what n or V is supposed to be here. Other symbols, $z, \gamma, P$, etc are introduced in this section but not defined. As such. by end of it, I am not sure what really is the output we are aiming for and how is it derived.

Similarly, Section 2.2, which I conclude is the main method description is mostly unreadable to me. The most I can conclude is that the authors are trying to use their method to provide a sequence to sequence mapping mechanism (from a input sequence X to output sequence Y). 

In the experiment section, the authors are evaluating their method on 3 datasets - wikiQA, SQuAD and CommensenseQA. But given my lack of understanding of method section, I can't really evaluate this intelligently. The best I conclude is that the authors method outperform their baselines for both semantic compression and Reading Compression.

### Strengths
It is possible with rewriting the readability of the paper can be improved bringing into focus its core contributions. Specifically I do think a method that can represent multiple distinct semantics associated with individual tokens can be useful but in its current form, I have a hard time understanding how the authors are able to achieve it.

### Weaknesses
The paper is unreadable. I would suggest AC either discard my review from consideration or put much less weight on my review if other reviewers are able to better understand the paper.

### Questions
In Section 2.1, I would like to understand what exactly is the task being performanced -- 1) what is the input (please provide an example), 2) what is the output, 3) what does symbols D_v, V, N, z means?

### Soundness
1 poor

### Presentation
1 poor

### Contribution
1 poor

---

## Human Reviewer 5

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper introduces Knowledge as Functional representation (KasF), a new method for knowledge reasoning in NLP that models knowledge as dynamic semantic fields. This approach outperforms traditional models on several NLP tasks by requiring fewer parameters and providing more precise knowledge encoding. The main contribution is the innovative functional representation that achieves state-of-the-art results on benchmarks such as CommonsenseQA, with the potential for more efficient and sustainable NLP models.

### Strengths
Innovative Knowledge Representation: The KasF model introduces a new functional representation of knowledge that enhances semantic reasoning in NLP.

State-of-the-Art Results: It achieves superior performance on benchmarks such as CommonsenseQA, indicating its potential for accurately handling complex reasoning tasks.

Computational Efficiency: The model's efficiency in terms of parameters used suggests it is less resource-intensive, contributing to more sustainable AI development.

### Weaknesses
- Complexity of the method: The KasF model's novel approach might be complex for the broader research community to understand and replicate. 

- While the paper shows strong empirical results, it may not thoroughly address how the model generalizes to tasks beyond those evaluated. It's unclear if the approach can be applied effectively to more reasoning tasks that are unseen in training.  

- The paper might lack a deeper theoretical discussion on the limitations of the functional representation of knowledge, which would be important for future research to build upon or address its shortcomings.

### Questions
- How well does the KasF generalize to unseen datasets that are similar to CommonsenseQA, e.g., SocialIQA, PIQA, RiddleSense? 
- How does the KasF model integrate with large external knowledge bases, and what is the impact on its performance when external knowledge is incorporated? Is it possible to do that?
- Is it feasible to connect KasF to the decoder-only models such as Llama?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
