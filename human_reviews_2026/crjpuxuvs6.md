# What is the Color of RED? Vision–Language Models Prefer to Read Rather Than See

- Decision: Reject
- Scores: 4, 2, 4, 2

## Abstract
A Visual Language Model (VLM) learns a joint understanding of image and text and generates text based on this understanding. Yet when multiple visual cues within an image conflict—such as a written word and its ink color—we do not fully understand how the model decides which signal to prioritize. A classical psychological paradigm to study how conflicting cues affect decision is the Stroop test, where participants are shown words in incongruent ink colors (e.g., the word "red" written in blue) and are instructed to report the ink color rather than read the word. We adapt the Stroop paradigm to VLMs and study how conflicting cues in the written word or ink color influence model behavior. Applying the Stroop test on a range of contrastive and generative VLMs suggests the models favor textual cues over color when text and color conflict. Analyzing the representation of the two cue types suggests that text cues in images are more salient than the color cues. This difference in saliency also translates to different intervention success in steering the VLMs: we found that it is easier to steer the embedding to make the model favor text cues than color cues. Overall, using the Stroop test, our findings suggest VLMs, similar to humans, are biased to "read" an image rather than to "see," and the saliency of the two cue types is reflected in their embedding space. We will release our dataset and code to support future research upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper examines how visual-language models (VLMs) respond to stimuli inspired by the classic Stroop test, where word semantics and ink color are in conflict. The authors evaluate both contrastive models (e.g., CLIP, SigLIP) and generative VLMs (e.g., Qwen2-VL) and find that these models systematically favor textual content over visual color when making predictions. The most novel parts are the representation-space analyses (Sec. 6–7), which reveal that word features are more salient and steerable than color ones. However, these sections alone are not strong enough to carry the full paper, and the rest of the work closely follows prior studies without clear added value.

### Strengths
1. The topic is conceptually engaging, combining cognitive psychology with VLM analysis in a way that is intuitively appealing and easy to motivate.

2. Section 6–7 provide meaningful representation-level insights. The differential RDM analysis quantifies the incremental contribution of “word” vs “ink” cues in the latent space, and the steering results support the claim that VLMs encode textual content more robustly.

### Weaknesses
1. Limited novelty. The core finding: VLMs prefer textual/semantic cues over color, has already been established, most notably in [1] and related works. Much of the behavioral setup, scoring approach, and analysis overlaps with existing literature.

2. Fundamental asymmetry in the Stroop setup. The core comparison between color and semantics is conceptually unfair: they belong to different representational levels in a vision–language model. Color is a low‑level perceptual feature mainly encoded in early layers, while semantic meaning is a high‑level abstraction usually emerging in later layers. Since all analyses rely on final‑layer embeddings that already emphasize high‑level semantics, the reported “semantic dominance” may simply reflect this architectural hierarchy rather than a bias.

3. Underdeveloped narrative and structure.
The paper lacks cohesion. Section 3 is too short to introduce the protocol properly, while Section 4 mixes in methodology that should appear earlier. Key setup components (e.g., scoring rules, dataset variants) are scattered or deferred to the appendix, making it hard to follow.

[1] Color in Visual-Language Models: CLIP Deficiencies https://arxiv.org/pdf/2502.04470

### Questions
1. Can you replicate the latent space analysis (Sec. 6–7) on non-CLIP models, e.g., SigLIP or MLLMs like Qwen or LLaVA?

2. Can you replicate the experiments in Sec. 7 using early and mid‑layer embeddings?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
In this paper, the authors generate a Stroop-style dataset that contains visually consistent & visually conflicting color-text pairs (for example, BLUE written in BLUE vs BLUE written in RED).  They test contrastive vision-text dual encoders like CLIP and generative vision-language models on this dataset and find that models have a strong preference for reading the color name in these text examples instead of detecting the actual color.  These results are robust to controlled variations of the Stroop task (varied font size, weight, contrast) and disappear when nonsense words are introduced.  They complement these results with an investigation into the latent space of CLIP and argue that this phenomenon arises from the relative entanglement of / relative separation of text in CLIP’s latent space through experiments that investigation representation similarity, representation projections and the steerability of representations through modification of subspaces.

### Strengths
This paper is fairly well-written and adapts an interesting result from the cognitive sciences to VLM evaluation.  Additionally, the authors present a fairly thorough investigation of their automatic Stroop test – in particular, I liked the stress testing they perform with textual overflow, contrast and pseudowords.  I also appreciated the efforts to understand what drives the phenomena they observe in their experimental results.  Finally, color is a rich area of exploration in VLMs and a work like this whose focus is understanding how VLMs make sense of color is a welcome addition to the literature.

### Weaknesses
### Novelty

My primary concern with this paper is the novelty of the results presented.  The community has been aware of textual bias in vision-language models as early as 2021 (https://distill.pub/2021/multimodal-neurons/#typographic-attacks) and later work has convincingly argued that this bias is an artifact of task ambiguity during pre-training (see “Task bias in vision-language models” by Menon et al, IJCV 2022).    

Moreover, Pezeshkpour et al (whom the authors cite) and “Words or Vision: Do Vision-Language Models Have Blind Faith in Text?” by Deng et al. CVPR 2025 (whom the authors should include in their related work, along with the references above) both present a thorough characterization of VLM textual bias across a wide range of tasks beyond color recognition alone.  

### Soundness of Novel Elements

The authors argue that their latent space analysis differentiates their work from Pezeshkpour and while a paper whose focus is how models represent and recognize text & color in their latent space would certainly be interesting, I would have liked to have seen more here beyond the relatively naïve RDM / UMAP projections / subspace steering of CLIP embeddings presented in the paper (for example, circuit identification / characterization as performed in mechanistic interpretability literature).  Additionally, I found Figure 4B quite difficult to make sense of – it could benefit from a different style of visualization.

### Smaller Issues

- It would be better for scale experiments to compare models of the same family (eg, Qwen2-VL-7B to 13B); while they can and do often differ in some hyperparameters, they differ less than models in completely different families and so are a better way to isolate the effects of scale.
- Framing: In some areas, the paper’s contributions are framed as being broader than seems appropriate for the results presented.  The experiments are focused on visual cases where color and text conflict but in several areas, they are presented as evidence that models “prefer to read rather than see” which is much broader.  I would encourage the authors to narrow these claims: line 26, lines 82-83

### Proofreading

- line 22: “suggest that text cues ARE more salient”
- line 26: “suggest VLMs ARE biased”
- Figure 1, top left; there’s superimposed text that looks like a mistake?  I think it’s meant to be under the bar on the far right
- line 156: missing a space
- line 181: “this” should be “these”
- line 204: “to word always perfectly”
- line 306: unclear what was intended here 
- line 308: missing space

### Questions
How do you evaluate the responses of generative VLMs (the mapping described on line 255)?  If it’s done via templating, what % of model responses fail to produce a templated answer?  If it’s done via an LLM-as-a-Judge, is judgment quality evaluated somehow?  

Given the small number of congruent examples, it seems likely that VLMs have seen / memorized them during training.  What does this mean for making sense of your results?  Are they really a fair baseline for how models handle congruence?   

In the nonsense experiments, for it to be comparable, I would think you need to include the nonsense word as a candidate color in addition to the 10 you describe on line 81. Do you? This result would have some interesting implications in terms of the task bias being OCR bound (models respond to ANY letters) or actual word bound.

Are the three RDMs described in section 6 the same size?  

Can you justify your interpretation of differential RDMs a bit more?  In particular, are these summary statistic matrices actually comparable through subtraction?  Your analysis seems to assume that if A and B are dissimilar but A’ and B’ are similar, A and B must drive the dissimilarity (or the we observe in A’’ and B’’ but I don’t think that necessarily follows. 

Do your steering results extend to VLMs?  Many build on CLIP-style encoders – when you apply your steering, does it produce similar results in LLaVA etc?

Suggestions:

Given the timing result for the Stroop test (lines 46-47) – that is, that humans take longer to handle incongruent examples -- it would be interesting to measure changes in similarity / VLM token probability in addition to changes in absolute accuracy.  Results have shown that token probabilities (or perplexity more broadly) are predictive of reading times in humans.  Perhaps there’s something analogous with color/text incongruency.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates how Vision-Language Models (VLMs) resolve multimodal conflicts by adapting the classic psychological Stroop test. The authors create a synthetic dataset where color words (e.g., "RED") are rendered in incongruent ink colors (e.g., blue ink). The study evaluates a range of models, including contrastive encoders (CLIP, SigLIP-2) and generative VLMs (LLaVA, BLIP-2, Kosmos-2, etc.). The primary finding is a consistent behavioral bias that models prefer to "read" the textual word rather than "see" the ink color, even when explicitly prompted to report the color. This text dominance persists across variations in font size, weight, and contrast, only disappearing when the text becomes illegible or is replaced by non-semantic pseudowords. 

To further investigate the phenomenon, the authors quantify how dissimilar different stimuli are in embedding space of CLIP's internal representations using Representational Dissimilarity Matrices (RDMs). The result shows that adding a word cue ($\Delta Word$) causes a much larger shift in the embedding space than adding a color cue ($\Delta Ink$). Additionally, latent interventions reveal that steering an image embedding toward a new word concept (e.g., "RED" $\rightarrow$ "BLUE") is 100% successful, while steering toward a new ink-color concept (e.g., red $\rightarrow$ blue) succeeds only 36.67% of the time. The authors conclude that this word-over-color bias is rooted in the embedding space, where text concepts are encoded with stronger, more distinct, and more "steerable" vectors than ink-color concepts, which are weak and highly collinear.

### Strengths
1. The finding is clear and well conveyed. 
2. The experiments are comprehensive across a group of VLM models trained under different paradigms. This makes the finding 
3. The steering representation experiment provides a deeper insight into the internal mechanism for the cause of VLM behavioral bias, which is well appreciated.

### Weaknesses
1. The idea of using Stroop test for VLM evaluation is already been explored in the literature [1][2], thus questioning the novelty of the paper.  
1. The central claim is based on a highly synthetic and constrained dataset. The dataset is limited to 10 basic color words and their corresponding ink colors. It is unclear how these findings about "reading vs. seeing" would generalize to more complex and naturalistic conflicts involving wider behavioral gaps between models and humans, thus limiting the significance of this work. 
2. The representational and interventional analyses (Sections 6 and 7) are performed only on CLIP. While this is a good starting point, these claims cannot be generalized to the modern generative VLMs that the paper also tests behaviorally. It is left unanswered whether models like LLaVA or Qwen2-VL suffer from the same underlying representational flaw (i.e., weak and collinear color vectors). 
3. The paper identifies the problem but does not substantially explore or test any mitigation strategies beyond the failed steering attempt.
4. The paper's discussion of prior work on multimodal conflicts and visual cognition is incomplete. It misses several recent and highly relevant papers that have established benchmarks for VLM cognition and conflict resolution. Specifically:
- [1] Sheta, Hala, et al. "From Behavioral Performance to Internal Competence: Interpreting Vision-Language Models with VLM-Lens." EMNLP 2025.
- [2] Zhang, Yichi, et al. "Grounding Visual Illusions in Language: Do Vision-Language Models Perceive Illusions Like Humans?." EMNLP'23. 
- [3] Schulze Buschoff, Luca M., et al. "Visual cognition in multimodal large language models." Nature Machine Intelligence 7.1 (2025): 96-106.
- [4] Jia, Yifan, et al. "Benchmarking Multimodal Knowledge Conflict for Large Multimodal Models." arXiv preprint arXiv:2505.19509 (2025).

### Questions
The results for Qwen2-VL-7B showed that a simple change in prompt phrasing nearly doubled the ink accuracy (from 31.1% to 60.0%). This suggests the model can access the ink color information. Does this finding perhaps undermine the core claim that the bias is a fundamental representational flaw? Could this bias be more of an instruction-following or attentional failure that can be largely resolved with better prompting, rather than a fixed property of the embedding space?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper adapts the Stroop paradigm to VLMs and study how conflicting cues in the written word or ink color influence the model’s behavior. The authors observed a consistent behavioral bias that when word and ink color disagree, models overwhelmingly align their preferences with the word. They also conduct experiments on the representation space. They find that word over color dominance is represented in the embedding space and models prioritize word over color cues when they conflict.

### Strengths
The topic is interesting.

The control experiment on text properties is good.

### Weaknesses
The importance and the implications of the findings in this paper are not convincing.

The scope of the experimental study is narrow.

Business models are not studied.

Some experimental designs need to be justified.

### Questions
1 Why is analyzing the Stroop phenomenon of Multi-modality models important ? What are the impacts and implications behind your findings? Do you consider why these problems are caused and how to solve them?

2 The scope of the ink color analysis is too narrow. The core research question behind this paper is to analyze whether VLMs understand the semantics (content) first or perceive the low-level features (color) first. Text content and ink color are two special cases of this research question. I hope the authors can extend the scope of this paper; otherwise, this work is only a workshop paper.

3 The analyzed models are all open-sourced. No business models are analyzed, so we do not know if the SOTA models have the same problem. Therefore, the importance of this finding cannot be validated.

4 To test contrastive models, you design a yes/no question, but you design a QA question to test VLMs. The discrepancy between the two types of models may influence your performance analysis. Additionally, you can design multiple-choice questions to test the models. Can you give explanations on why you design in this way?

5 The Stroop examples are not general enough. You just draw the text on white paper. What is the influence of the background? Additionally, the benchmark with only 100 testing examples is not enough. You should construct at least 1000 examples.

### Soundness
2

### Presentation
2

### Contribution
2
