# Chronoberg: Capturing Language Evolution And Temporal Awareness In Foundation Models

- Avg Score: 5.00
- Decision: Reject
- Scores: 2, 4, 8, 6

## Abstract
Large language models (LLMs) excel at operating at scale by leveraging social media and various data crawled from the web. Whereas existing corpora are diverse, their frequent lack of long-term temporal structure may however limit an LLM's ability to contextualize semantic and normative evolution of language and to capture diachronic variation. To support analysis and training for the latter, we introduce Chronoberg, a temporally structured corpus of English book texts spanning 250 years, curated from Project Gutenberg and enriched with a variety of temporal annotations. First, the edited nature of books enables us to quantify lexical semantic change through time-sensitive Valence-Arousal-Dominance (VAD) analysis and to construct historically calibrated affective lexicons to support temporally grounded interpretation. With the lexicons at hand, we demonstrate a need for modern LLM-based tools to better situate their detection of discriminatory language and contextualization of sentiment across various time-periods. In fact, we show how language models trained sequentially on Chronoberg struggle to encode diachronic shifts in meaning, emphasizing the need for temporally aware training and evaluation pipelines, and positioning Chronoberg as a scalable resource for the study of linguistic change and temporal generalization. $\\textcolor{red}{Disclaimer:}$ This paper includes language and display of samples that could be offensive to readers. $\\textcolor{blue}{Open Access:}$ Chronoberg will be available publicly on HuggingFace.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents a dataset for evaluating temporal changes of words, which is then used to evaluate LLMs. Temporal change detection is a task that has been researched extensively (see SemEval workshops on this topic over the years for example) but its progress has been hindered by the lack of large-scale time-annotated datasets. This paper fill this important gap.

### Strengths
- The creation and release of a large-scale dataset for evaluating temporal changes of VAD of words is the main strength/contribution of this paper.
- The paper then goes on to evaluate multiple foundation models on this dataset.

### Weaknesses
- given the automatic nature of the annotation process, a manual verification (e.g. using a random sample) should be conducted.
- books are a curated set of texts and might not necessarily express the sentiment held by the broader population. Social media on the other hand would have been a better source for capturing modern VAD of words but cannot be used for historical texts. What is the coverage in terms of the number of authors covered in the corpus? If this number is low then it might not be representing the view point of the general public but an elite and small group of authors. This dataset bias should be investigated further.
- I am not sure whether all books in this corpus is suitable for this purpose of evaluating diachronic changes. For example, there could be fantasy books which do not reflect a social viewpoint but an imaginary one. Even if a book is written at a particular point in time, it might be handling a historical context that belongs to a different time period (e.g. a book written in 2020 on Edo period of Japan would not be reflecting the modern usage of Japanese language). I am not sure how these complications are handled in this corpus (or the authors are aware of such issues)?
- Although I appreciate the extensive evaluations conducted in the paper using the CHRONOBERG dataset, those findings will only be valid to the extent of the accuracy of the dataset itself.
- Moreover, I think this paper is more appropriate for the linguistic resources track at an NLP venue (e.g. LREC or xACL) rather than ICLR.

### Questions
- Did you conduct any manual (even at a small scale) evaluation of the VAD scores computed using Eq. (2)?
- What is the coverage in terms of the number of authors covered in the corpus? If this number is low then it might not be representing the view point of the general public but an elite and small group of authors. This dataset bias should be investigated further.
- In Table 1, can you explain the shift from neg to pos for `infatuation` and `destinty` please?
- What would the valency shift look like for a word such as `gay`, which is known to be used positively (happiness) in olden times, whereas more neutral? in modern usage?

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
3

### Summary
This paper presents Chronoberg, a large diachronic English corpus (1750–2000, 2.7B tokens) with sentence-level affective annotations and temporally aligned Valence–Arousal–Dominance (VAD) lexicons. The authors detail a pipeline for dating and cleaning Project Gutenberg texts, generate time-specific VAD lexicons via aligned embeddings, and use the resource to test hate-speech detectors and temporal adaptation of LLMs (sequential fine-tuning, EWC, LoRA). Results show severe forgetting and limited handling of historically shifting word valence, highlighting Chronoberg’s potential as a benchmark for temporal robustness and continual learning.

### Strengths
- Language temporal drift and historical semantics are important for ensuring LLM robustness.
- The paper presents interesting insights on semantic shift.

### Weaknesses
- Narrow domain coverage. Chronoberg is built entirely from Project Gutenberg books, spanning 1750–2000. This makes it a valuable literary resource but also limits generalization to modern or conversational language where current LLMs are deployed .

- Unclear LLM evaluation setup. The paper compares several classifiers, including “OpenAI” models, in its hate-speech evaluation tables . However, it does not describe how those models were prompted or whether they were informed about the historical origin of the text. Without that context, it’s hard to interpret what the observed disagreements actually reflect.

- Limited quantitative evaluation of temporal drift. Section 4.1 and the associated tables mostly illustrate qualitative examples of valence shifts and classifier disagreements. There is no formal metric (e.g., correlation or agreement score) quantifying alignment between the temporal VAD lexicons and model outputs .

- Unsurprising claims without measurement. The authors hypothesize that modern LLMs “rely too heavily on surface-level keywords,” but this remains an intuitive explanation rather than an empirically tested one . Quantifying how much this reliance contributes to misclassification would make the finding stronger.

- Vague notion of “contextualization.” The paper claims that temporally fine-tuned models “struggle with contextualization of historical content,” yet does not specify how contextual information (e.g., time metadata or retrieved examples) was provided during inference .

- Ambiguous terminology. Phrases like “dissonance of ~85% between OpenAI and RoBERTa” are reported without a clear definition of what “dissonance” measures (e.g., disagreement rate or correlation gap) . This makes quantitative interpretation difficult.

### Questions
- How do you expect findings from literary English (Chronoberg) to generalize to modern or conversational domains where current LLMs are applied?

- For the “OpenAI” model evaluations, what exact prompts or instructions were used? Was the model told the historical period or context of the text?

- Have you computed any quantitative metrics (e.g., correlation or agreement) between temporal VAD lexicons and classifier outputs, beyond qualitative examples?

- Can you operationalize “contextualization of historical content” — does it mean awareness of time, lexical shifts, or broader discourse cues?

- What precisely does the reported “∼85% dissonance” between OpenAI and RoBERTa measure? Is it disagreement rate, accuracy gap, or another metric?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes MetaDiff, a diffusion-based framework for few-shot adaptation of large vision-language models (VLMs). Traditional few-shot fine-tuning approaches—such as adapter tuning or prompt learning—are limited by overfitting and poor uncertainty estimation when data are scarce. MetaDiff reframes few-shot adaptation as a meta-generative prior learning problem: it learns a conditional diffusion model in the weight or latent embedding space of the VLM such that, given a few examples from a novel task, it can sample adapted model states that generalize better.

### Strengths
1. The idea of using a diffusion process in the parameter or embedding space for meta-adaptation is highly creative. Unlike deterministic meta-learners, MetaDiff explicitly models the distribution over adapted models, allowing uncertainty-aware adaptation. This is both novel and well-motivated theoretically.
2. The paper provides comprehensive experiments across diverse domains (classification, captioning, retrieval), showing that MetaDiff outperforms fine-tuning, adapter tuning, and standard meta-learning baselines under few-shot constraints. Ablations clearly indicate that the diffusion prior contributes significantly to performance.

### Weaknesses
1. While the motivation for using diffusion models is strong, the paper does not provide a formal link between the learned generative prior and Bayesian meta-learning or PAC-Bayesian guarantees. A short theoretical justification (e.g., that MetaDiff approximates amortized inference under a hierarchical Bayesian model) would strengthen the contribution.
2. MetaDiff’s meta-training stage is computationally heavy, involving thousands of diffusion steps across multiple tasks. While test-time sampling can be parallelized, a more detailed cost analysis or discussion on distillation into fewer diffusion steps would be valuable.

### Questions
1. It would be useful to see how performance varies with the number of diffusion timesteps. Is MetaDiff’s success primarily due to stochastic regularization or due to the learned generative structure?
2. When sampling multiple adapted models via diffusion, how diverse are their predictions? Are these samples genuinely capturing task uncertainty or just random noise around a single mode?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper addresses the crucial problem that most Large Language Models (LLMs) are trained on temporally stationary data, limiting their ability to comprehend the long-term evolution of language, social norms, and semantics (diachronic variation). The introduction of Chronoberg, a large-scale (2.7B tokens) and temporally structured corpus of full-length English book texts (25,061 books from Project Gutenberg) spanning 250 years (1750–2000). Experiments using the dataset show that LLMs trained sequentially struggle significantly with catastrophic forgetting and generalization to future language, particularly concerning words that have undergone affective valence shifts. This positions Chronoberg as a necessary benchmark for temporal generalization, Continual Learning, and evaluating the temporal robustness of AI systems.

### Strengths
1. The introduction of Chronoberg is a major contribution, filling a critical gap in LLM training and evaluation. It provides a large-scale (2.7B tokens) and long-horizon (250 years) temporally structured corpus of full-length texts. 
2. The systematic construction of temporally calibrated VAD lexicons for $\sim$335,000 words is highly innovative.

### Weaknesses
1. The temporal structure, which is the foundation of the dataset, relies on an inferred publication date from external sources (OpenLibrary), while validated, this process has an unavoidable Mean Absolute Error (MAE) of $\pm 3.05$ years.

2. The methodology for constructing the temporal VAD lexicons relies on selecting the Top-K nearest neighbors ($K=20$). However, the main text does not include a systematic ablation study demonstrating how the choice of $K$ (or the effect of the 50-year interval size) impacts the final, crucial results, such as the LLM perplexity gap between valence-stable and valence-shifting test sets.

### Questions
1. How did you get the "positive" or "negative" in Table 1? It is still not clear to me. 

2. The paper uses Word2Vec and CADE alignment. Could the authors justify why this combination was chosen over more contemporary diachronic methods?

### Soundness
3

### Presentation
3

### Contribution
3
