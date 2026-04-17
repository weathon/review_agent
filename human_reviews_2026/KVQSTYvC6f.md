# Factor-Wise Homogeneity of Slot-Attention for Continual Object-Centric Learning

- Decision: Reject
- Scores: 2, 6, 2, 4

## Abstract
Can current AI models continually learn object-centric representations?
Object-Centric Learning and Continual Learning are both critical areas of AI research, yet their intersection remains underexplored. In this work, we observe that Slot Attention, a popular OCL method, exhibits a distinctive behavior: 
It organizes latent representations into small and separated regions, each of which 
preserves the same factor states, referred to as \textit{factor-wise homogeneity}.
This phenomenon emerges not only in previously trained data but also in upcoming data with unseen factor states, offering significant advantages for continual learning that incrementally expands factor states, such as novel shapes. To harness this property, we propose a simple and effective method, \textit{Decoder only Post Replay}, that freezes the encoder and the Slot Attention as a generator of factor-wise homogeneous representations and employs a decoder-only fine-tuning strategy after the novel task training is done.
Although Slot Attention has been widely studied, its representational behavior has been largely overlooked. This paper highlights its unique strengths in continual object-centric learning. We also introduce a novel validation and analysis environment for Continual-Object Centric Learning, establishing a strong baseline for future research.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper studies continual object-centric learning and finds that Slot Attention naturally organizes slot features into small, factor-consistent, well-separated neighborhoods across tasks, even for unseen objects and tasks. Following this, they propose Decoder-Only Post Replay (DPR): after each task, freeze encoder + slots and fine-tune only the decoder on a mix of buffered past reconstructions and some current data to combat forgetting. Across Tetrominoes/CLEVR/COCO and multiple task schedules, SA+DPR improves FG-ARI and MSE vs. vanilla SA.

### Strengths
This paper studies the problem continual learning in object-centric learning. There is not lot of work on this topic therefore it seems like an interesting topic to study.

### Weaknesses
My main concern is that the the authors say that the encoder + slot attention can transfer to new tasks and only the decoder needs to be trained for adaptation. I have two critiques regarding this - 

1. I think the main goal of object centric representations is to learn good representations for downstream tasks. This paper only evaluates segmentation via ARI. SAM and other segmentation methods can already do that well and still generalize, so in my view evaluating these methods on ARI is not well motivated.
2. If the authors observe that only the decoder needs to be updated for transfer, why not just use the masks from slot attention for computing ari? why do you need the decoder at all? Many works do this - https://arxiv.org/abs/2110.11405, https://arxiv.org/abs/2209.14860

Secondly, the observation that the encoder + slot representations transfer well across domains was already shown in https://openreview.net/forum?id=bSq0XGS3kW, so I believe that the observation is nothing new. In fact the mentioned paper shows that object-centric models can transfer to new domains and objects very well without seeing much samples during training so I don't see why we need a continual learning setup at all in object-centric learning?

### Questions
See weaknesses

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates the factor-wise homogeneity property of Slot Attention and its implications for Continual Object-Centric Learning (C-OCL). The authors introduce new benchmarks—Continual-Tetrominoes and Continual-CLEVR—to evaluate continual unsupervised object discovery, and propose a simple method termed Decoder-only Post Replay (DPR). DPR freezes the encoder and slot attention modules while fine-tuning only the decoder, thereby leveraging the observed factor-wise homogeneity to mitigate catastrophic forgetting. Extensive experiments, both quantitative and qualitative, demonstrate the stability of this property and the effectiveness of DPR across synthetic and real-world datasets.

### Strengths
- The identification and empirical characterisation of factor-wise homogeneity in Slot Attention is an original and conceptually interesting contribution. It provides a new perspective on how object-centric representations are organised in latent space.
- The introduction of the Continual-Tetrominoes and Continual-CLEVR benchmarks represents a meaningful step toward systematic evaluation of continual object-centric learning. These resources could be of lasting value to the community.
- The proposed Decoder-only Post Replay (DPR) is simple, elegant, and easy to reproduce. Its minimalistic design strengthens the argument that the observed behaviour of Slot Attention itself underpins the improvements.
- The experiments are extensive and cover synthetic, complex, and real-world datasets. The authors further conduct ablations, comparisons with diverse baselines, and analysis of compatibility with regularisation-based methods.
- The manuscript is well structured, with clear motivation, detailed explanations of methodology, and coherent visualisations that support the claims.

### Weaknesses
- While the empirical findings are compelling, the paper lacks a rigorous theoretical analysis explaining why factor-wise homogeneity emerges in Slot Attention. The argument remains largely observational.
-Although effective, the DPR method could be viewed as a minor algorithmic variation (a post-hoc replay scheme). The conceptual novelty mainly lies in the analysis, not in the proposed learning strategy.
- The continual learning scenarios considered (mainly new shape classes) are somewhat narrow. Broader evaluations involving more complex semantic shifts (e.g., textures, dynamics, or object relationships) would better support general claims.
-The improvements, while consistent, are moderate in some settings. It would strengthen the paper to include statistical significance tests or further qualitative explanations of the observed gains.
-The paper could engage more deeply with recent object-centric continual learning frameworks or compositional representation studies (e.g., those integrating diffusion-based or transformer-based architectures).
- Although the authors mention code availability in the supplement, hyperparameter and architectural details are partly deferred to the appendix, which may limit immediate reproducibility from the main text.
-Some sections (especially Section 4) are lengthy and could benefit from tighter exposition. The key insights are occasionally obscured by repetition and detail overload.

### Questions
plz see my detailed comments above

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper aims to tackle an important and underexplored problem: how to enable object-centric learning (OCL) models to perform continual learning (CL), i.e., to learn new object categories without catastrophically forgetting old ones. In summary, this work would make for a good workshop paper. It successfully defines a problem, establishes a benchmark, and provides a simple yet effective baseline. However, it lacks the "eye-opening" insight and technical depth, and is unlikely to have a profound methodological impact on the field. Therefore, I recommend rejecting this paper.

### Strengths
1. The authors observed a phenomenon they name "Factor-wise Homogeneity." Specifically, after training, the latent representations (slots) of the classic OCL model, Slot Attention, spontaneously organize into compact and separated clusters, where each cluster corresponds to the same semantic factor (e.g., shape). More importantly, this separation property also holds for unseen object categories.

2. Based on this discovery, the authors propose an extremely simple method, "Decoder-only Post Replay" (DPR). After learning a new task, this method freezes the encoder and the Slot Attention module (treating them as a stable generator of factor-wise homogeneous representations) and then fine-tunes only the decoder using a replay buffer containing both old and new samples.

3. The authors propose the first benchmark for Continual Object-Centric Learning (C-OCL), including Continual-Tetrominoes and Continual-CLEVR, providing an evaluation platform for future research.

### Weaknesses
1. The novelty of the core discovery (Factor-wise Homogeneity) is limited. This is the cornerstone of the paper, but its novelty is questionable. A well-trained representation learning model's fundamental goal is to map inputs of different semantics to separable regions in the latent space. The "factor-wise homogeneity" observed by the authors can largely be seen as an expected property that any successful representation learning model should possess, rather than a surprising, entirely new discovery. The authors' work feels more like naming and empirically verifying that Slot Attention has this desirable property, rather than unveiling a previously unknown mechanism. Therefore, packaging it as a core "discovery" seems like an overstatement.

2. The technical solution (DPR) severely lacks novelty. This is the paper's most critical weakness. The DPR method can be seen as a simple combination of existing techniques: freezing the feature extractor, experience replay, and two-stage training are all standard procedures. Essentially, DPR just assembles these simple building blocks. While effective, it feels more like a clever engineering shortcut or a "trick" than a new algorithm with profound insights. It introduces no new theory, model architecture, or optimization objective.

3. Although the authors' logic is that "the simple DPR works precisely because of factor-wise homogeneity," this feels more like a post-hoc explanation for the effectiveness of a simple method, rather than the discovery itself inspiring a novel and ingenious solution.

4. It is questionable whether the core idea proposed in this paper can lead the field. A truly inspiring work should excite other researchers and make them willing to explore new models and theories based on its core ideas. For instance, if the authors had proposed a new regularization term or model architecture to actively enhance this "homogeneity" instead of merely exploiting it, the paper's inspirational value would be much greater. The current DPR method feels more like an endpoint than a starting point.

### Questions
1. You present "factor-wise homogeneity" as a core discovery. A critical question is: is this property unique to Slot Attention, or is it a general characteristic found in other mainstream object-centric learning (OCL) models (e.g., MONet, IODINE, SAVi)? If it is a common property, the novelty of this discovery is diminished. If it is unique to Slot Attention, can you provide a mechanistic explanation as to why its iterative attention and GRU updates specifically give rise to this phenomenon? The current ablation study shows the importance of these components but falls short of offering a fundamental explanation.

2. The central claim of the paper is that DPR is effective precisely because of the existence of "factor-wise homogeneity." This is a causal assertion that requires more direct evidence. Could you design an experiment to demonstrate this link more explicitly? For instance, what happens if you apply the DPR method to a model that does not exhibit this property (like the SlotMLP baseline in your experiments)? Does its performance collapse catastrophically? Conversely, if you were to disrupt the homogeneity in Slot Attention through some means, would DPR's effectiveness also fail? This would provide strong support for your central argument.

### Soundness
2

### Presentation
3

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
The paper studies an interesting and less explored topic about combining object-centric learning and continual learning. The idea of using Slot Attention’s factor-wise homogeneity for continual learning is quite novel. The proposed method is simple and seems effective. The experiments are a bit limited, but with more comprehensive studies, the work could be much stronger.

### Strengths
- originality: 4/5
- quality: 4/5
- clarity: 3/5
- significance: 3/5

### Weaknesses
W1
---
Section 2 Related Work: Should review OCL literatures. 
Especially various Slot Attention variants, e.g., BO-QSA, ISA and MetaSlot. 

It is necessary to experiment or at least discuss the potential effects of different OCL decoders, as the spatial broadcast CNN decoder was proposed 5 years ago, while there are many advanced OCL decoders have been proposed:
- auto-regressive-based: SLATE, using conditional Transformer as the decoder;
- spatial broadcast-based: DINOSAUR, using MLP as the decoder (similar to CNN as the decoder, used in this paper, according to Line 138-139);
- de-noising-based: SlotDiffusion, using conditional Diffusion model as the decoder.
The authors can either conduct the suggested experiments on these three OCL methods separately, or on a unified OCl method, VVO at once.

Including such experiments or discussions could make this work more complete and more impactful to broader audiences.


W2
---
Typo: 
In Line 144, `where and M is the` should remove "and".


W3
---
Line 191, the authors included SlotMLP, which is an earlier work of Slot Attention (both fairly old). However, there are many later works with great improvements, e.g., BO-QSA, ISA and MetaSlot, which should be included in the analysis.


W4
---
Line 202, 
> Since the model has not yet encountered images from (E1) during training on (T0), this separation suggests that the behavior is primarily determined by the encoder and slot attention modules rather than influenced by the decoder.

In Section 4.3, DPR is described, i.e., freezing the encoder and slot attention. So DPR is not applied here, right?

Besides, based on the well separation of E1/T0, how was that only the encoder and slot attention matters concluded? The logic is unclear.


W5
---
Line 212.
> inter-task representations For each slot,

There should be a "." before "For".


W6
---
Line 215,
> one highlights inter-task similarity, the other emphasizes within-task consistency

According to your contexts, "within-task" should be "intra-task".


W7
---
Line 287
> preserve factor-wise homogeneous
Should be "homogeneity".


W8
---
Line 296-301,
> DPR is based on two core components: (1) we freeze the encoder and slot attention module and only fine-tune the decoder, in order to maintain factor wise separated representations observed in Slot Attention space S; and (2) we introduce a Post Replay (PR) strategy, wherein the model is fine-tuned after the task training phase, i.e., training on the current task Tt without any continual learning methods, thus completely excluding sources of interference during initial slot representation learning.

Unclear and repetitive writing. Please reoganize and polish it.


That said, I am willing to change my rating if my conerns are addressed.


Reference
---
- BO-QSA: Improving Unsupervised Object-centric Learning with Query Optimization
- ISA: Invariant Slot Attention: Object Discovery with Slot-Centric Reference Frames
- MetaSlot: Break Through the Fixed Number of Slots in Object-Centric Learning
- SLATE: Illiterate DALL-E Learns to Compose
- DINOSAUR: Bridging the Gap to Real-World Object-Centric Learning
- SlotDiffusion: Object-Centric Generative Modeling with Diffusion Models
- VVO: Vector-Quantized Vision Foundation Models for Object-Centric Learning

### Questions
Please refer to the former section.

### Soundness
3

### Presentation
2

### Contribution
3
