# SCREEN-SBERT: EMBEDDING FUNCTIONAL SEMANTICS OF GUI SCREENS TO SUPPORT GUI AGENTS

- Decision: Reject
- Scores: 8, 4, 6, 4

## Abstract
Recent GUI agent studies show that augmenting LLM prompts with app-related
knowledge constructed during a pre-exploration phase can effectively improve
task success rates. However, retrieving relevant knowledge from the knowledge
base remains a key challenge. Existing approaches often rely on structured
metadata such as view hierarchies, which are frequently unavailable or outdated,
thereby limiting their generalizability. Purely vision-based methods have emerged
to address this issue, yet they typically compare GUI elements only by visual appearance,
leading to mismatches between functionally different elements. We consider
a two-stage retrieval framework, where the first stage retrieves screenshots
sharing the same functional semantics, followed by fine-grained element-level retrieval.
This paper focuses on the first stage by proposing Screen-SBERT, a purely
vision-based method for embedding the functional semantics of GUI screenshots
and retrieving functionally equivalent ones within the same mobile app. Experimental
results on real-world mobile apps show that Screen-SBERT is more effective
than several baselines for retrieving functionally equivalent screenshots. As
a result, (1) we formally define the concepts of functional equivalence and functional
page class; (2) design a contrastive learning-based embedding framework;
and (3) conduct ablation studies that provide insights for future model design.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper addresses the limitation of existing GUI retrieval methods that rely solely on visual similarity, which often leads to mismatches between elements that look alike but serve different functions. To overcome this, the authors use a two-stage retrieval framework:  
(1) a screenshot-level retrieval stage that identifies GUI screens sharing similar functional semantics, and  
(2) a fine-grained element-level retrieval stage for detailed matching.  

The innovation is introduced in the first stage. They introduce Screen-SBERT, a model designed to embed the functional meaning of GUI screenshots using purely visual cues. The framework’s key innovations include:  
(1) capturing functional semantics of GUI screens in a vision-only setting, and  
(2) employing a contrastive learning approach that supports effective few-shot learning and achieves robust performance even with small datasets.

### Strengths
1. Unlike previous works that treat GUI retrieval mainly as a design-assistance task emphasizing visual similarity, this study redefines the task as retrieving functionally equivalent knowledge for augmenting LLM prompt. The introduction of the concept of Functional Equivalence provides a valuable and meaningful new perspective.  

2. The authors reimplemented several closed-source models, and their released code closely follows the method reported in the original papers, which is a notable contribution to reproducibility.  

3. The paper includes a detailed discussion of design alternatives and decision rationales, making the final framework convincing and well-grounded.

### Weaknesses
1. While the paper claims that the proposed method can facilitate knowledge retrieval for augmenting LLM prompts, it lacks experiments or case studies demonstrating this motivation in practice.

2. The work overlooks a potentially important baseline — using a Multimodal LLM (MLLM) to generate functional descriptions for each screenshot and performing retrieval based on these descriptions.

### Questions
See weakness.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses a core challenge in knowledge retrieval for GUI agents: existing methods either over-rely on unreliable structured metadata or merely compare visual appearance , failing to accurately retrieve functionally equivalent GUI screens.
Compared to baselines that rely on unreliable metadata, only match visual appearance, or perform poorly on small datasets due to the use of MLM , Screen-SBERT is a purely vision-based bi-encoder model. It leverages contrastive learning to effectively learn functional semantics from a small dataset and achieves retrieval efficiency far exceeding that of cross-encoder models .
The main limitations are that the annotation of "functional page classes" relies on subjective judgment , all experiments are confined to a small-scale dataset (i.e., few-shot learning) , and computational constraints (a single GPU) prevented comparison against larger VLM baselines.
The paper uses a manually constructed dataset of 1,814 screenshots from six real-world applications . The code and preprocessed modality data are open-sourced on GitHub, but the original screenshots are not released due to privacy concerns .

### Strengths
1. The core contribution of this paper is the introduction of the "functional equivalence" concept. This enables the model to retrieve screens that share the same function despite dynamic content (e.g., products, posts) , addressing a fundamental problem where existing purely vision-based methods (like CLIP) are confounded by superficial appearance .

### Weaknesses
1. The page class annotation lacks clear, objective criteria. The dataset used for training and evaluation was manually labeled based on the authors' "intuitive judgment" , which introduces "a degree of subjectivity" to the class boundaries . This makes the annotation process difficult to scale and verify.
2. Due to the "lack of a large-scale public dataset" for this task , all experiments in this study fall under "few-shot learning". Therefore, it remains unknown whether the paper's conclusions (e.g., the superiority of contrastive learning over MLM ) would "generalize to large-scale scenarios".
3. Due to "computational constraints (training on a single GPU)" , the authors were unable to conduct fine-tuning experiments against "larger VLM (Vision-Language Models)" that require multiple GPUs. This leaves the SOTA comparison incomplete.

### Questions
1. To what extent can the functional semantics learned by Screen-SBERT (particularly its heavy reliance on layout structure ) generalize to application domains with entirely different layout paradigms and interaction logic, such as productivity tools, banking/finance apps, or complex game interfaces?
2. How did you ensure the consistency and reliability of the annotations? Was any form of cross-validation performed to validate the subjective labeling?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Screen-SBERT, a vision-based framework for learning functional embeddings of GUI screenshots in mobile apps. The core motivation is to support GUI agents by enabling retrieval of functionally equivalent screens (e.g., “Home” or “Product Details”) even when their visual content differs. Screen-SBERT uses a bi-encoder architecture inspired by Sentence-BERT and trains with contrastive learning to embed screenshots based on functional semantics rather than pixel similarity. It does not rely on metadata like view hierarchies, which are often unavailable or outdated.

The framework consists of:
1. A GUI Parsing Module that extracts multimodal features (vision, text, coordinates, functional type) for each GUI element.
2. A GUI Embedding Module that converts these features into unified token embeddings.
3. A Transformer encoder with 2D relative positional bias to model spatial layout.
4. A contrastive learning objective (InfoNCE) to learn screenshot-level embeddings.

Experiments are conducted on a manually curated dataset of 1,814 screenshots from 6 apps (Instagram, Facebook, X, Amazon, Coupang, Temu), with out-of-domain evaluation. Screen-SBERT outperforms baselines (CLIP, Screen Correspondence, PW2SS, Screen Similarity Transformer) in retrieval accuracy and efficiency.

### Strengths
1. It is the first work to formally define functional equivalence and functional page class for GUI screenshots.
2. The proposed method outperforms baselines in Macro F1 and Top-1 accuracy across multiple OOD settings.
3. This paper thoroughly analyzes the impact of modalities, positional encodings, and training objectives.

### Weaknesses
1. The dataset is somewhat small. Only 1,814 screenshots from 6 apps; limited generalizability to larger or more diverse apps.
2. Only CLIP-Base is used; no comparison with larger VLMs like CLIP-Large or SigLIP due to GPU constraints.
3. Functional equivalence is labeled manually by authors; no inter-annotator agreement reported.

### Questions
1. Have you considered evaluating cross-app retrieval (e.g., retrieve “Home” screen from Temu using Instagram’s “Home” as query)? This would test whether the model learns general functional semantics rather than app-specific layout patterns.
2. How consistent are your functional equivalence annotations? Were multiple annotators involved? If not, how do you ensure that the model is not simply memorizing your subjective grouping criteria?
3. Your rule-based classifier for GUI element types is brittle. Have you tried fine-tuning a small classifier on a few hundred labeled examples instead of relying on captioning? This could be a lightweight improvement.

### Soundness
2

### Presentation
3

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
This paper proposed Screen-SBERT, a vision-based framework for embedding and retrieving the functional semantics of GUI screenshots to support GUI agents in mobile apps. Unlike prior work that relies on structured metadata, it uses only visual information to identify functionally equivalent screens. The method also employs contrastive learning for learning GUI embedding. Evaluation results show that Screen-SBERT outperforms baselines in retrieving functionally similar screens.

### Strengths
+) This paper proposed a pure-vision solution to GUI embedding which not requires view hierarchy

+) Interesting ablation study to reveal the importance of each modality

### Weaknesses
-) This paper lacks comparison to methods that use view hierarchy when they are available, to better understand the gap

-) The evaluation is limited to a relatively small, manually constructed dataset

### Questions
a) How does the approach compare to methods that use structured metadata, both in terms of accuracy and computational efficiency?

b) How does Screen-SBERT handle screens with dynamic or context-dependent elements that may not be visually distinguishable but differ functionally?

### Soundness
2

### Presentation
3

### Contribution
2
