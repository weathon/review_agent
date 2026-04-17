# Echoes of BERT: Do Modern Language Models Rediscover the Classical NLP Pipeline?

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 2, 4

## Abstract
Large transformer-based language models dominate modern NLP, yet our understanding of how they encode linguistic information relies primarily on studies of early models like BERT and GPT-2. Building on prior BERTology work, we analyze 25 models spanning classical architectures (BERT, DeBERTa, GPT-2) to modern large language models (Pythia, OLMo-2, Gemma-2, Qwen2.5, Llama-3.1), probing layer-by-layer representations across eight linguistic tasks in English. Consistent with earlier findings, we find that hierarchical organization persists in modern models: early layers capture syntax, middle layers handle semantics and entity-level information, and later layers encode discourse phenomena. However, larger models compress this entire hierarchy toward earlier layer positions, suggesting they build richer representations more quickly. We dive deeper, conducting an in-depth multilingual analysis of two linguistic properties - lemma identity and inflectional features - that help disentangle form from meaning. We find that lemma information concentrates linearly in early layers but becomes increasingly nonlinear deeper in the network, while inflectional information remains linearly accessible throughout all layers. Additional analyses of attention mechanisms, steering vectors, and pretraining checkpoints reveal where this information resides within layers, how it can be functionally manipulated, and how representations evolve during pretraining. Taken together, our findings suggest that, even with substantial advances in LLM technologies, transformer models learn to organize linguistic information in similar ways, regardless of model architecture, size, or training regime, indicating that these properties are important for next token prediction.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The study employs both linear and non-linear probing techniques to investigate linguistic properties encoded into the intermediate representations across a wide range of mono- and multilingual models. The analysis reveals that modern decoder-only architectures reproduce several processing patterns observed in less-recent encoder-only models. Specifically, earlier layers predominantly capture syntactic features, middle layers encode semantic information, and later layers relate more closely to discourse-level phenomena. Furthermore, the study demonstrates that lexical information is more effectively extracted using non-linear probes, whereas inflectional morphology can be linearly identified across all layers. Overall, the findings suggest that the token prediction objective inherently drives transformer-based architectures to learn shared structural and linguistic properties of language.

### Strengths
1. The investigation takes into account large set of models 
2. The set of metrics employed is large and meaningful

### Weaknesses
1. Short and outdated "Related work" section. There are more recent studies concerning the analysis of linguistic capabilities at different levels in modern LLMs (e.g., Cheng et al.: Emergence of a High-Dimensional Abstraction Phase in Language Transformers, 2025; Skean et al.: Layer by Layer: Uncovering Hidden Representations in Language Models, 2025)
2. lack of strong evidence for some of the highlighted results (see questions)
3. Many analyses dubbed as relevant and reported even in the abstract are discussed too quickly and superficially. For instance, Section 5.2 many potentially interesting properties and observations are just mentioned and then relegated to the appendix. I think the multilingual results are interesting, but they would deserve a discussion on their own. I'd suggest to prioritize some results and dwell on them in a proper manner.

### Questions
1. I think the paper would benefit from an explanation, even if very brief and in Appendix, of the tasks that are used throughout the paper (NER, SRC, SPR, ...), together with a paragraph telling which one of them relate to syntax/semantic/discourse properties.
2. conclusions drawn in section 4.1 are not so clear to me by looking at Figures 2, 6-9. For instance, the authors state "Mid-level semantic tasks such as SRL and SPR peak in middle layers" but it often look pretty flat (for linear probes, Fig. 8,9) if not peaking at early/late layers (see LLama, olmo, qwen in Fig 6,7)
3. In Fig 3 the grey columns (Accuracy(?)/Selectivity for Regression/MLP) show values averaged across the layers? This is not clear from the caption or the text. Furthermore, a large selectivity means the extraction of a linguistic information, while a low one implies memorization. How do the authors comment on the fact that only "Entities" has a positive large values, while all the others are close to zero if not significantly negative?

### Soundness
2

### Presentation
1

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
This work aims to answer this question *"Do current decoder LLMs keep the classical NLP pipeline?"* To this end, the authors analyze 25 language models, including both encoder and decoder language models through the probing tests of eight linguistic tasks. The findings indicate that hierarchical organization persists in modern models: early layers capture syntax, middle layers handle semantics and entity-level information, and later layers encode discourse phenomena. After, they study two properties, lexical identity and inflectional morphology, and find that  lexical information concentrates linearly in early layers but becomes increasingly nonlinear deeper in the network, while inflectional information remains linearly accessible throughout all layers.

### Strengths
* This work conducts extensive studies across 25 LLMs and 8 tasks  
* It is interesting to see how LLMs handle lexical identity and inflectional morphology, which is worth studying
* The steering experiments provide a potential application about how to leverage their findings

### Weaknesses
* Some main claims are not well supported by their experimental results. For example, the authors claim that LLMs exhibit a hierarchical NLP pattern in Section 4. However, it is not clear to see the performance change of different tasks across layers in Figure 2. Likewise, Figure 3 is hard to understand and it is tricky to see the layer-level comparisons
* I appreciate that the authors conduct extensive analyses, but this work would be more novel if they can provide more in-depth studies. I am not surprised that all LLMs have classical NLP pipelines since they use the same architecture and training objectives. The section 4 should be more succinct and concise. I find the lexical identity and inflectional morphology are under-explored by prior work. I would suggest focusing on this point and exploring if LLMs are capable of handling the two tasks, where do LLMs store knolwedge for them, and finally how to perform interventions (steering) 
* Some figures lack readability, which strongly undermines the contribution of this work (see questions). This work would be improved if they can make revisions about the presentation and readability

### Questions
* Figure 2 is not easy to understand and cannot support the authors' claims. In the left figure, *"The top row shows MLP probe accuracy, the bottom row shows linear
probe accuracy"*. However, I cannot find the corresponding top and bottom rows for different probes. In the right figure, *"Pearson correlations between models are computed by vectorizing each model’s per-layer, per-task accuracy grid"*. I can only see the pearson correlations between models instead of layers. More importantly, the authors' observations are not well supported by the figure. For example, *"Mid-level semantic tasks such as SRL and SPR peak in middle layers, while later layers capture discourse-level phenomena like Coreference and Relations. "*. I find that these tasks perform similarly across different layers. Additionally, why do BERT families have low correlations in the right figure? Can the authors clarify this?
* Figure 3 has the same issue. For example, *"discourse-level features are encoded most strongly in later layers"*. However, the expected layer of BERT-base for the task of Relations is 4.43, and the BERT-base model has 12 layers. It is confusing that the authors derive this finding from this figure. Since there are no baselines in this figure, it is tricky to judge how significant the selectivity is here
* Figure 4 and 5 have a low readability. These lines are entangled with each other and hard to distinguish between different language models 
* In section 5.2. the authors conducted extensive analyses,  but there are no focal points
* I did not fully understand why the authors use linear and nonlinear probes, which should be clarified in section 2.1

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
4

### Summary
This paper examines 25 transformer-based language models (both MLM and generative), from early architectures like BERT and GPT-2 to modern ones such as Llama-3.1 and Gemma-2, to investigate how linguistic information is represented in the multiple layers of these models. The authors claim that these models exhibit a hierarchical organization, with syntax supposedly concentrated in early layers, semantics and entity-level information in middle layers, and discourse phenomena in later ones. They further suggest that larger models show these patterns emerging earlier in the network. Finally, they report that lexical identity becomes increasingly nonlinear with depth, while inflectional morphology remains linearly accessible, arguing that such regularities persist across architectures and training regimes.

### Strengths
- the work is relevant and timely: revisiting and updating BERTology to observe how information is represented in more recent, generative LMs is an interesting challenge
- the authors carried out a very large number of experiments on many different LMs of multiple types (MLM, generative [base], generative [instruct] ) and sizes (from BERT base to LLAMA 3.1 8B)
- the authors commit to publishing a GitHub repository with the code necessary to reproduce their experiments, and display a strong

### Weaknesses
- the authors should make sure they use the correct terms. For instance, "lemma" and "lexeme" are not interchangeable. "Inflectional morphology" is a subfield of linguistics, not a task (the authors have "morphological analysis" in mind).
- the eight tasks are not defined/described, and the labels used in Figure 1 to name them are not defined in the text. The reader should not have to read Jindal et al. 2022 to understand what "Universal Proposition English-EWT (SRL)" is, and with which other tasks it should intuitively be grouped (is it a syntactic, semantic or discourse task, as per the authors' classification?)
- more importantly, I fail to understand, based only on Figures 2 and 3, how the authors reach the strong conclusion that starts Section 4.3: "our results demonstrate that modern language models consistently rediscover the classical NLP pipeline […]. […] we find that syntactic information is typically represented in earlier layers, and discourse-level features are encoded most strongly in later layers." The authors do not indicate which tasks they consider syntactic (is morphological analysis a "syntactic" task?), semantic and discourse tasks. The content of the two above-mentioned figures does not seem, to my naked eyes at least, to corroborate their conclusions, nor display consistent behaviours with one another. And results for only 3 models are shown anyway. The fact that other results are given in the appendices is not enough: the main part of a paper must be self-contained, and the authors should have given overall quantitative results demonstrating that their conclusion holds for all modern language models, since this is what Section 4.3 implies.
- steering experiments and their results are difficult to understand when they only take 6 lines of the main paper. It might be the case that the authors have tried to include too much content in a single paper, when a separate paper focused on the content of Section 5 could have make sense.

### Questions
- why use Random Forests in section 5, after having explained at the beginning of Section 2 that it is important to only use *simple* probabilistic classifiers as probes?

### Soundness
2

### Presentation
2

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
The paper evaluates a bunch of older LMs as well as modern LLMs on probing tasks used to measure layer-by-layer learning. The findings correlate strongly with "the classical NLP pipeline" (as in Tenney et al): early layers are syntax heavy, middle layers focus more on semantics.. and so on.

The authors follow the probe design methodology from Tenney et al. The setup is pretty similar to Tenney et al, however, they also use the methodology on more SoTA LLMs. In doing so, they compare the learnings b/w more dated LMs (like BERT) v.s modern open source models (like Qwen). They notice some interesting findings, such as, as model sizes increase, learning is peaked at earlier layers indicating that modern models learn rich representations more quickly and need fewer layers to consolidate linguistic knowledge.

In addition, the paper does a thorough analysis on the lexical and morphological inflection tasks, in various settings across layers, multiple languages and probing setups (to check non linear nature)

### Strengths
* The experiments are well designed and thorough. The results are explained well and there is strong rationale for most of the results. Overall, seems like a very sound setup. I must also applaud the thoroughness, there is good level of depth to the experiments (so many variables like layers, model sizes, task, probing design)
* The paper is fairly well written. Its easy to understand and keeps you interested.

### Weaknesses
* My biggest issue with the paper is the strong similarities to Tenney et al. Its not clear from the paper what is being introduced  in a novel manner v/s ideas derived from Tenney et al. I'm worried that this isn't a significant scientific contribution beyond what previous papers have introduced.
* While this style of probing (for latent knowledge) is interesting, it would be useful to relate this to more "downstream" tasks like math/reasoning/coding/reading comprehension etc. And perhaps we could then compare linguistic probing (like do earlier layers learn hard math problems) v/s downstream task probing? This would be very useful for the community.

### Questions
Please answer questions about differences wrt Tenney et al and how this work is contributing in a novel manner.

### Soundness
3

### Presentation
3

### Contribution
2
