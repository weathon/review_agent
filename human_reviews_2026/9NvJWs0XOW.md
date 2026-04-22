# Chinese Character Decomposition with Compositional Latent Components

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 2, 6

## Abstract
Humans can decompose Chinese characters into compositional components and recombine them to recognize unseen characters. This reflects two cognitive principles: $\textit{Compositionality}$, the idea that complex concepts are built on simpler parts; and $\textit{Learning-to-learn}$, the ability to learn strategies for decomposing and recombining components to form new concepts. These principles provide inductive biases that support efficient generalization. They are critical to Chinese character recognition (CCR) in solving the zero-shot problem, which results from the common long-tail distribution of Chinese character datasets. Existing methods have made substantial progress in modeling compositionality via predefined radical or stroke decomposition. However, they often ignore the learning-to-learn capability, limiting their ability to generalize beyond human-defined schemes. Inspired by these principles, we propose a deep latent variable model that learns $\textbf{Co}$mpositional $\textbf{La}$tent components of Chinese characters (CoLa) without relying on human-defined decomposition schemes. Recognition and matching can be performed by comparing compositional latent components in the latent space, enabling zero-shot character recognition. The experiments illustrate that CoLa outperforms previous methods in both character the radical zero-shot CCR. Visualization indicates that the learned components can reflect the structure of characters in an interpretable way. Moreover, despite being trained on historical documents, CoLa can analyze components of oracle bone characters, highlighting its cross-dataset generalization ability.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes CoLa, a compositional latent–component model for Chinese character recognition that eliminates manual part annotations. An encoder decomposes an input glyph into K permutation-invariant “slots,” a frozen teacher guides the slots via feature reconstruction, and classification is performed by measuring distances between the input’s slot representation and class “prototypes” computed from a small set of font-rendered templates per character. This design aims to support zero-shot recognition of unseen characters and rare components. Experiments report notable gains on zero-shot splits (including a constructed “component zero-shot” setting) and on harder domains such as historical documents; ablations indicate both the teacher-guided reconstruction and the prototype classifier contribute materially. Visualizations hint at cross-script transferability.

### Strengths
Pros:
1.	Clear, coherent pipeline (slots + teacher alignment + prototype matching).
2.	Removes reliance on handcrafted part labels; offers intuitive visualizations.
3.	Broad evaluations with ablations; strong reported gains on defined splits.

### Weaknesses
Cons:	
1.	Motivation not substantiated with real-world evidence; zero-shot largely constructed.
2.	Heavy dependence on font-rendered templates and a closed label set; no open-set handling.
3.	Limited analysis of style/domain shift and cross-script transfer (mostly qualitative).
4.	Insufficient transparency: exact fonts, template policy, hyperparameter/cost sensitivity, and teacher dependence not systematically studied.

### Questions
Zero-shot claims rest critically on the availability of font templates and a closed candidate set, leaving open questions about domain/style gaps (fonts vs handwriting/archives), open-set rejection, and robustness. Important engineering details (font sources, template sampling, cost/speed, sensitivity to K and template count) are insufficiently reported, limiting reproducibility and practical credibility.

### Soundness
3

### Presentation
2

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
The paper proposed a method to decompose Chinese characters into components and then recognize them. The core contribution is a latent variable model that learns components of Chinese characters without relying on human-defined decomposition annotations. Overall, the problem addressed in the paper is interesting and the method is relatively novel, but the experimental results indicate that the method is not yet satisfactory.

### Strengths
1. Decomposing Chinese characters into meaningful components using unsupervised learning is an interesting topic of good academic value.
2. To my best knowledge, this is the first work that explores unsupervised component decomposition for Chinese characters. The overall framework that combines decomposition and recognition is also novel.
3. As far as I can see, the proposed method is technically sound.
4. The paper is well-organized and clearly written.

### Weaknesses
1. My main concern about the paper is its performance. From the visualization results (especially Fig. 6), the model frequently outputs fragmented and discontinuous components, which are inconsistent with human understanding of character components. Given that most of the demonstrated characters have simple structures (such as left-right or top-bottom), this indicates that the model has not learned to decompose characters into meaningful components.
2. The core part of the proposed method (slot attention and feature reconstruction) is directly based on an existing method ('Object-Centric Learning with Slot Attention'), which limits the novelty of the method.
3. Due to the high variability in Chinese character structures, it is crucial to either output a variable number of components based on the input character or set a relatively large K. However, the paper fixes the number of components at K=3, without any explanation or relevant experimental analysis.

### Questions
In the visualization experiments, it is recommended to first categorize Chinese characters by their structural types and then present the results by category, enabling readers to gain a more comprehensive understanding of the model's decomposition performance on characters with different structures.

### Soundness
3

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
This paper addresses the zero-shot Chinese Character Recognition (CCR) problem, a challenge arising from the long-tail distribution of Chinese character datasets. Inspired by human cognitive principles of compositionality and learning-to-learn, the authors propose CoLa, a deep latent variable model that learns compositional latent components of Chinese characters without relying on human-defined schemes (e.g., radicals or strokes). CoLa decomposes characters into latent components via input/template encoding, reconstructs high-level features using a frozen DINOv2 teacher encoder, and predicts classes by comparing components in the latent space. Experiments show CoLa outperforms baselines (e.g., CCR-CLIP) in character/radical zero-shot settings on HWDB, Printed, and Historical Document datasets, and generalizes to oracle bone, Japanese, and Korean characters.

### Strengths
Originality： Self-supervised component learning for CCR: CoLa treats components as latent variables that emerge via reconstruction and classification objectives, avoiding bias from manual decomposition rules.
Cognitive inspiration: Grounding the model in compositionality and learning-to-learn (rather than heuristics) provides a principled inductive bias, which is a thoughtful design choice for character recognition.

Rigorous experimental design: Multi-dataset evaluation (handwritten, printed, historical) and ablation studies (teacher encoder, prediction loss) confirm model components’ necessity. Quantitative (mIoU for component masks) and qualitative (visualization) analyses of interpretability add rigor.

### Weaknesses
Primary Weakness: Insufficient Topic Significance and Practical Relevance
The most critical limitation of this work is that the topic of "zero-shot Chinese character recognition" lacks broad practical demand and generalizable research value, making the contribution narrow and marginal in the broader CV landscape.

The zero-shot scenarios CoLa targets (e.g., training on 500 characters to recognize 1,000 rare characters, radical zero-shot with n<10) are artificially extreme and rarely encountered in practice:

Chinese characters are highly structured, low-diversity visual objects (fixed writing norms, finite structural patterns) that differ fundamentally from natural images (e.g., scene understanding, object detection) or unstructured text (e.g., handwritten Latin script). CoLa’s core innovation—self-learning compositional latent components—relies on the unique structure of Chinese characters and cannot be transferred to other CV tasks:

Lack of Comparison with Practical Alternatives： The paper ignores simpler, more practical solutions for rare character recognition that outperform CoLa in real-world scenarios:

Overemphasis on Niche Cross-Dataset Generalization

CoLa’s cross-dataset results (OBCs, Japanese, Korean) are technically interesting but further highlight the topic’s marginality:
Japanese (with hiragana/katakana) and Korean (with hangul) have distinct writing systems—CoLa’s ability to decompose their characters is irrelevant to their core recognition challenges (e.g., hangul’s syllabic structure).

No quantitative metrics (e.g., retrieval accuracy for OBCs/Japanese) are provided, making it impossible to assess whether this generalization has practical value for researchers in these fields.

Few-shot learning (e.g., fine-tuning CCR-CLIP on 10 samples per rare character) would likely achieve comparable or higher accuracy than CoLa’s zero-shot approach, with lower computational cost.

Human-in-the-loop annotation (common in 古籍整理) is more efficient for rare characters, as experts can label 100 samples in hours—rendering zero-shot models unnecessary.

### Questions
Topic Significance Justification:

The authors focus on zero-shot Chinese character recognition, but this problem is niche with limited real-world demand. Could you:
Provide concrete examples of industrial or academic users who require zero-shot recognition of rare characters (e.g., training 500 characters to recognize 1,000 rare ones)?

Explain why this problem is more important than addressing core CCR pain points (e.g., low-resolution character recognition, scene text with complex backgrounds) or broader CV challenges (e.g., general zero-shot object detection)?

Generalizability and Transfer Value:

CoLa’s component learning is tailored to Chinese characters. Can you demonstrate that your method (or its core ideas) can be transferred to other CV tasks (e.g., decomposing natural objects like "car" or "cat," or recognizing unstructured handwritten Latin text)? 

Practical Alternative Comparison:

Few-shot learning or human annotation is more cost-effective for rare character recognition. Have you compared CoLa with few-shot baselines (e.g., CCR-CLIP fine-tuned on 5–10 samples per rare character)? If CoLa is outperformed by these simpler methods in real-world rare character scenarios, what is its competitive advantage?

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces CoLa (Compositional Latent components), a deep latent variable model designed to recognize Chinese characters, especially in zero-shot scenarios where characters seen during testing were absent from the training data.

Unlike previous methods that rely on rigid, human-defined decomposition rules (like predefined strokes or radicals) , CoLa learns to automatically discover its own compositional components as latent variables. This self-learning approach allows it to generalize far better to unseen characters and character parts.

### Strengths
The paper's main strength is moving beyond rigid, human-defined decomposition schemes (like strokes or radicals). The CoLa model instead learns to decompose characters on its own, which is inspired by the cognitive principles of "compositionality" and "learning-to-learn"

The visualizations show that the model is not just a black box. The learned latent components (Cmp#1, #2, #3) clearly focus on distinct, independent regions of the character.

Instead of trying to reconstruct noisy, raw pixels, the model is trained to reconstruct the high-level semantic features from a frozen, pre-trained DINOv2 encoder. An ablation study confirms this "teacher" is "crucial" for the model's success

### Weaknesses
The model architecture hard-codes the number of components to K=3

The model's success relies on a DINOv2 "teacher" that was pre-trained on natural images (like animals, objects, and landscapes). While this works, its features are not optimized for the specific domain of orthography.

### Questions
Since DINOv2 was trained on general-domain natural images (animals, landscapes, etc.), is the model learning to decompose characters into "natural image parts" (e.g., generic lines and curves) rather than "orthographic parts"?

How sensitive is the model to this template set?

### Soundness
3

### Presentation
3

### Contribution
2
