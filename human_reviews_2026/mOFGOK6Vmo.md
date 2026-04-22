# LexSign: Learning Sign Language from Lexical Descriptions

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Sign languages are well-defined natural languages that convey meaning through both manual postures and non-manual expressions. While recent methods effectively transcribe sign language videos into compact textual tokens, they often overlook the intrinsic subunit-level structures of sign language. In this work, we explore leveraging the hierarchical structure within lexical descriptions to enhance fine-grained sign language understanding. Specifically, we first construct LexSign, a large-scale dataset comprising both manually curated and automatically generated lexical descriptions of signs. To guarantee the quality of generated descriptions, we build LexSign-Bench, a benchmark to comprehensively evaluate the sign language understanding capability of Multi-modal Large Language Models (MLLMs), and further propose a perceive-then-summarize pipeline that leverages large foundation models to generate high-quality lexical descriptions. Based on the constructed LexSign, we propose Hierarchical Action-Language Interaction (HALI) that conducts hierarchical alignment between lexical descriptions and sign language videos to obtain more distinguishable and generalizable visual representations. Experimental results on public datasets demonstrate that incorporating the collected lexical descriptions with the proposed HALI significantly improves performance across different sign language understanding tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents LexSign, a large-scale dataset of lexical descriptions for sign language signs, collected both manually from authoritative sign language dictionaries and automatically via a “perceive-then-summarize” pipeline using multimodal large language models (MLLMs) and LLMs. To ensure the quality of generated descriptions, the authors introduce LexSign-Bench, a comprehensive benchmark evaluating MLLMs’ ability to understand sign language at subunit, gloss, and sentence levels. Based on LexSign, they propose HALI (Hierarchical Action-Language Interaction), a novel framework that aligns sign language videos with lexical descriptions across multiple granularities—video, snippet, subunit, and textual fragments—through multi-granularity contrastive learning and hierarchical consistency constraints. Experiments on public datasets (e.g., ASL-Text, WLASL) demonstrate that HALI significantly improves performance in both zero-shot and supervised isolated sign language recognition, highlighting the value of fine-grained linguistic structure in advancing sign language understanding.

### Strengths
1. The problem proposed in this paper carries significant real-world relevance, and the LexSign dataset can greatly facilitate unified understanding and generation of sign language.
2. The approach that combines contrastive learning with hierarchical feature alignment is relatively novel.
3. The paper presents fairly comprehensive experiments that effectively support its claims.

### Weaknesses
1. The paper suffers from substantial writing issues, particularly concerning focus and the missing of details.
2. Although the proposed hierarchical consistency loss yields strong empirical results, the paper lacks a clear motivation or justification for its design.

### Questions
1. Writing issues:
a) The paper devotes excessive space to the method section. I recommend rebalancing the narrative in a revision by strengthening the description of dataset construction, for instance, by moving Figure 6 and its accompanying explanation from the supplementary material into the main paper.
b) The experimental section omits critical implementation details: What model serves as the baseline? What training protocol is used? How do the experimental setups differ across datasets?

2.  Hierarchical consistency concerns:
a) The definitions of snippet-level and subunit-level representations are ambiguous. Are they meaningfully distinct? If the granularity is merely defined by manual segmentation without linguistic grounding, does it carry genuine semantic significance?
b) Do Equations 5 and 8 include weighting hyperparameters to balance the contributions of different loss terms? Moreover, the paper lacks sensitivity analyses for key hyperparameters such as M , batch size B , and temperature τ .

3. Additional suggestions:
a) The set of baseline methods appears outdated; the authors should include comparisons with more recent state-of-the-art approaches.
b) Consider integrating the MLLM evaluation results from Table 10 into the main paper, and, if feasible, add a direct comparison between MLLM-based and traditional methods.
c) Could HALI be extended to MLLM SFT? Such an extension would significantly strengthen the paper’s impact and practical relevance.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper focuses on sign language recognition from the perspective of fine-grained recognition with vision-language models. For this, paper has three main contributions: (i) a new dataset (and a new benchmark), (ii) a training loss structure that aims to enforce learning vision-language alignment for sign language recognition in a hierarchical and fine-grained manner, and (iii) a comprehensive evaluation that involves existing open+closed source multimodal LLMs.


The dataset is collected in a semi-manual manner. Human annotators matched videos with descriptions at lexical description level. In addition, a combination of LLM+MLLM based heuristic is used to add lexical descriptions to other videos.

The proposed method relies on an enhanced CLIP-like training mechanism. The core method relies on contrastive aligment loss across the sign videos and descriptions. This is enhanced by introducing additional loss terms based on (i) video-snippet & description-fragments (eq 4) and (ii) a term enforcing consistency across the similarity distributions over fragments and video snippets versus fragments and subunits.

### Strengths
The makes contributions on three valuable ends: (i) sign language datasets & benchmarks, (ii) hierarchical vision-language formulations for sign video understanding and (iii) evaluation of MLLM abilities.

The paper is overall well written (though I do have several confusions, as listed below in the questions field). 

The proposed method is sensible.

The experimental results contain several interesting bits.

Commitment to make the data & techniques publicly available that ensures full reproducibility.

### Weaknesses
Some important details are hard to find in the paper, and some of the experimental result discussions are confusing (see below).

There is no clear model selection protocol defined in the paper for the proposed benchmark, which may incorrectly encourage model selection directly over the test results -- a huge problem for any (future) researcher who wants to do it right and avoid any test-set based hyper-parameter tuning. For instance, one can imagine an extremely complex prompt that simply maximizes the MLLM performance on the proposed test set, w/o actually contributing to the real-world generalization.

No discussion on model selection & hyper-parameter tuning for the proposed method.

### Questions
Can please summarize the techniques to split video & text data into subunits, video snippets and description fragments with points to the detailed descriptions in the paper? I understand the concepts but also curious about how to practically obtain them, currently I am bit lost here.


Are the comparisons to prior work in Table 1 & Table 2 fair? Do you use the same training data with those models in comparison? 


I understand you measure the quality of auto-generated descriptions by using them for training and evaluating on the manually collected dataset. However, I think here there is a missing baseline that involves training with manually collected data. This evaluation could have been more complete if there was a version that used a different data split where some of the manually annotated data is used at training time (and comparing against use those images with auto-generated descriptions).

What is the conclusion on the performance of general purpose MLLMs versus training ad-hoc sign models?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes to model sign language representations by introducing lexical descriptions. To this end, the authors extend the WLASL and DEVISIGN datasets with annotations that include lexical descriptions. They then learn sign language representations by aligning sign language videos with these descriptions. Additionally, the authors introduce LexSign-Bench, a benchmark designed to evaluate the sign language understanding capabilities of multimodal large language models (MLLMs).

### Strengths
**Strengths:**

- Although the authors do not clearly define what they mean by “lexical descriptions,” the attempt to exploit a new data modality to enrich sign-language features is a welcome effort.
- The proposed approach yields modest improvements on isolated sign-language recognition (isolated SLR).

### Weaknesses
**Weaknesses:**
- The paper never provides a rigorous definition of “lexical descriptions.” From the given examples they appear to be either an augmented gloss or a short action caption rather than a faithful translation of the sign video. The authors offer no justification for aligning videos to this coarse-grained surrogate instead of to the actual translation text, nor do they explain what benefit such a proxy signal could bring to end-to-end sign-language translation (SLT).
- The technical contribution is thin. The work focuses almost exclusively on data curation, while the algorithmic side adds little novelty. Video–text contrastive learning has become commonplace in SLT, and the proposed method scarcely departs from existing pipelines.
- I question the adequacy of off-the-shelf video-understanding models for sign language. Without targeted finetuning on sign-language data, these models are likely to produce only superficial or even misleading representations, undermining both the alignment phase and the subsequent evaluation on LexSign-Bench.

### Questions
- Could the authors provide a formal definition of “lexical descriptions” and explain why they choose this specific form of annotation—rather than the actual translation text—as the alignment target? How does this coarse-grained proxy benefit downstream tasks, particularly end-to-end sign language translation?
- While the introduction of lexical descriptions is interesting, the core algorithm appears to follow standard video–text contrastive learning. Could the authors clarify the novel technical contributions that distinguish this work from existing sign-language representation methods?
- The paper uses off-the-shelf video understanding models without sign-language-specific fine-tuning. Have the authors evaluated whether these models capture sign-language-specific visual-linguistic patterns? If not, how reliable are the learned representations and the subsequent benchmark results?

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
4

### Summary
The paper provides a dataset and benchmark for isolated sign classification using subunit information, including natural language descriptions of them (partially ground and partially pseudolabelled). It provides some experiments training sign language recognition models (which perform better using subunit information) and zero-shot with MLLMs, which show a little bit of sign language understanding.

### Strengths
The idea of natural language descriptions for signs is interesting, and the experiments with MLLMs are interesting, especially that they do better at the gloss questions than the subunit questions. Nice variety of baselines/experiments that look sound and reproducible, including prompts in the appendix etc. It's hard to imagine the ultimate solution for sign language understanding looking like HALI (all the hand-engineering) but it does improve the scores.

### Weaknesses
Overall I think the contribution is sound but low-moderate novelty and low-moderate significance. Related work misses some relevant papers from Kezar et al. https://arxiv.org/abs/2310.00196 https://arxiv.org/abs/2411.03568v1. I think I've seen at least one other work recently that used natural language descriptions of signs for MLLM sign language pretraining, but it was maybe within a reasonable timeframe before the ICLR deadline.

The paper could mention the possibility that the benchmark is contaminated wrt LLMs, because it's built on top of previously publicly available data.

I found myself having to refer back to earlier parts of the paper frequently, like the presentation could have been clearer, but maybe I'm just tired.

### Questions
I'm open to increasing my rating to 6 for the following:
* I'd like to see more about how this work differs from the Kezar papers which I linked above
* It would be interesting to see some qualitative analysis of the MLLM model outputs. I don't see them in Appendix A.2. I'm a bit suspicious of how high the scores in Table 10 are, though I guess it's multiple choice. Maybe I missed it, but how many multiple choices are there? 4?


There are some weird/incorrect/insensitive phrasings about sign language towards the beginning of the paper, but the rest of the paper looks fine. My rating is conditioned on these being fixed. Specifically:

> "Sign language is a well-defined visual language"
This isn't really what "visual language" means (https://en.wikipedia.org/wiki/Visual_language). Sign language also isn't "a language", it's a category of language (cf. spoken language) that includes many languages.

> "sign language... is largely unintelligible to hearing individuals"
"Unintelligible" has an unfortunate connotation that the thing can't be understood, rather than it isn't understood. I would just say "but it is not largely known by hearing individuals"

> " To bridge this communication gap, vision-based sign language understanding (SLU)"
It's a bit weird to phrase it this way when it's only half the communication gap.

> "enabling automatic recognition and translation of sign language from video input into textual or symbolic representations in a non-intrusive manner"
I wouldn't say this has been "enabled" yet; sign language models are still terrible. It aims to enable this

> "As a well-defined natural language"
Same as above, it's not "a" language.

Another misc comment: The acronym "ZSSLR" is introduced without explicitly introducing that it stands for zero-shot sign language recognition, ZSL, GZSL etc.

### Soundness
3

### Presentation
3

### Contribution
2
