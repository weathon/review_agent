# Phantom-Data:  Towards a General Subject-Consistent Video Generation Dataset

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 10, 6, 4, 4

## Abstract
Subject-to-video generation has witnessed substantial progress in recent years. However, existing models still face significant challenges in faithfully following textual instructions. This limitation, commonly known as the copy-paste problem, arises from the widely used in-pair training paradigm. This approach inherently entangles subject identity with background and contextual attributes by sampling reference images from the same scene as the target video. To address this issue, we introduce \textbf{Phantom-Data, the first general-purpose cross-pair subject-to-video consistency dataset}, containing approximately one million identity-consistent pairs across diverse categories. Our dataset is constructed via a three-stage pipeline: (1) a general and input-aligned subject detection module, (2) large-scale cross-context subject retrieval from more than 53 million videos and 3 billion images, and (3) prior-guided identity verification to ensure visual consistency under contextual variation. Comprehensive experiments show that training with Phantom-Data significantly improves prompt alignment and visual quality while preserving identity consistency on par with in-pair baselines.

## Human Reviews

## Human Reviewer 1

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
The paper tackles a central pain point in subject-consistent video generation—the “copy–paste” failure mode that emerges from in-pair training, where a reference frame and target video come from the same clip and the model inadvertently binds identity to background/pose. The authors introduce **Phantom-Data**, a large-scale **cross-pair** dataset (1M identity-consistent pairs across humans, animals, products, IP characters), and a three-stage pipeline to build it: (1) S2V detection via open-vocabulary grounding with quality checks (completeness/specificity/text match), (2) large-scale retrieval from **~53M videos + 3B images** with category-specialized encoders, and (3) prior-guided identity verification (e.g., logo presence for products; same long video for living beings) plus a VLM check to ensure both identity match and context diversity (Fig. 4, p.5). 

Using Phantom-Data to train an open-source subject-consistent model (Phantom-wan), the paper reports clear gains in **prompt following** and **video quality** while keeping **identity consistency** comparable to in-pair baselines (Table 2, p.7). Qualitative examples (Fig. 5, p.8), ablations on subject diversity/scale (Tables 3–4, p.9), and a user study (Table 5, p.15) corroborate the improvements and the importance of cross-pair diversity.

### Strengths
* **Addresses a key failure mode.** Clearly identifies and targets the copy–paste issue caused by in-pair supervision, and proposes a principled data solution rather than ad-hoc augmentations (Fig. 2, p.3; Sec. 3). 
* **Well-designed, scalable pipeline.** The three-stage pipeline (detection → cross-context retrieval → prior-guided verification) is systematic, with concrete checks (completeness/specificity; upper/lower similarity thresholds; VLM verification) that materially reduce false positives (Fig. 4–6, pp.5–9). 
* **General beyond faces.** Moves cross-pair training from face-only domains to **general subjects** (humans, animals, products, multi-subject scenes), aligning with real user inputs (Fig. 3(d–e), p.4; Table 1, p.3). 
* **Fair, controlled comparisons.** Same base model, objective, resolution, and inference settings across training regimes; metrics cover text alignment (Reward-TA), identity (DINO/GPT-4o scores), and video quality (VBench), with both quant and qual results (Sec. 4, Table 2, Fig. 5). 
* **Clear, strong empirical gains.** Cross-pair training with Phantom-Data improves prompt alignment and overall video quality, with identity consistency on par with in-pair training (Table 2); ablations show benefits from subject diversity and scaling from 100k → 1M pairs (Tables 3–4).

### Weaknesses
* **Ambiguity around the “Face Cross-pair” baseline (Table 2).** The paper states this baseline “utilizes face-level identity matching across videos,” but the exact construction is unclear relative to **Ours**. Did “Face Cross-pair” (i) rely solely on ArcFace-based retrieval without the **prior-guided verification** step, (ii) omit **clothing/body** features for people, and/or (iii) exclude **non-face** subjects entirely? Clarifying the settings would explain the large Reward-TA gap (3.022 vs. 3.827) and the differences in DINO/GPT-4o scores (Table 2, p.7).

### Questions
None

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Phantom-Data is a large-scale, general-purpose dataset designed to address the "copy-paste" problem in subject-to-video (S2V) generation. Existing models often fail to follow textual prompts because they are trained on in-pair data (reference and target frames from the same video), leading them to copy irrelevant background details from the reference image. Phantom-Data mitigates this by providing "cross-pair" data: identity-consistent reference and target frames from different videos or contexts.
Experiments using the Phantom-wan model show that training with Phantom-Data significantly improves text-prompt alignment and video quality compared to in-pair training baselines, while maintaining high identity consistency.

### Strengths
1. The "copy-paste" problem is a major, well-known limitation in current open-source video generation models. Tackling this via better data is a highly practical and valuable approach.

2 Unlike previous cross-pair datasets that mostly focused on faces, this dataset covers general objects (products, animals, etc.) and is very large (1 million pairs)

3. The data construction pipeline is well-designed. Specifically, combining large-scale retrieval with strict VLM-based verification  is a smart way to automate generating high-quality pairs without human annotation.

### Weaknesses
1. While the dataset is a great contribution, the pipeline requires an enormous pool of data (53M videos, 3B images) to work effectively. This scale is likely out of reach for most academic labs to replicate or extend on their own.

2. The experimental results primarily compare different custom data setups (in-pair, in-pair + aug, face cross-pair) against their proposed method. There is no comparison against models trained on currently existing public datasets to provide a true external baseline.

3. While quantitative metrics are provided for the different data ablation setups (Table 3), visual examples comparing the outputs of these specific ablations (e.g., "face only" vs. "+ product" vs. "+ multi-subject") would strengthen the argument for the necessity of each data component.

### Questions
1. Table 1 states the dataset is "Publicly Available". Does this mean the full 1 million pairs (images and corresponding video clips) will be directly downloadable, or will it be a list of URLs/IDs that users need to scrape themselves?

2. Regarding the "same long-form video" (L339) constraint for living beings: Did you experiment with trying to find the same human/animal across completely different videos?

3. What is the estimated error rate of the final VLM verification step? Are there many valid cross-pairs that get thrown out because the VLM is too conservative?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Phantom-Data, a cross-pair subject-to-video consistency dataset for video personalization, comprising one billion identity-consistent pairs across diverse categories. Experiments indicate the high quality and potential impact of the dataset.

### Strengths
- The paper presents a well-designed pipeline for curating subject-video paired datasets, leveraging Visual Language Models (VLMs) to achieve robust subject-attribute pairing.
- The dataset is large-scale, providing valuable resources for advancing research in video personalization.
- The curated data effectively addresses the prevalent copy-paste issue encountered in video personalization tasks.

### Weaknesses
- It remains unclear how much the VLM-based pipeline improves over simpler approaches, such as using GPT-4o for generating image variations. As shown in Figure 7, though having drawbacks, generative models might sometimes even offer advantages in achieving controllable variations. Additionally, potential artifacts from generative approaches could be mitigated through VLM-driven verification and filtering, which the paper does not explore.
- The evaluation is somewhat limited, as the dataset is tested exclusively with Wan2.1. Given that many established methods in video personalization predate the Wan series, broader evaluation across multiple models would strengthen the paper’s claims and demonstrate the wider utility of the dataset.

### Questions
n/a

### Soundness
3

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
This paper introduces Phantom-Data, a large-scale cross-pair dataset for subject-consistent video generation, aiming to mitigate the copy-paste problem caused by in-pair training. The dataset is built via a three-stage pipeline combining open-vocabulary detection, cross-context retrieval, and prior-based identity verification. Experiments show improved prompt alignment and video quality while maintaining subject consistency.

### Strengths
1. The motivation is clear and addresses a real issue (“copy-paste”) in subject-consistent video generation.
2. The dataset construction pipeline is well-designed, integrating VLM-based detection and verification modules.
3. The ablation and user study provide some evidence of effectiveness.

### Weaknesses
1. The prompts used in the dataset are overly simple and sparse, failing to accurately describe video semantics or capture complex spatiotemporal relations. This limits the dataset’s potential to improve text-video alignment in realistic scenarios.
2. The paper evaluates only on the Phantom-Wan model. Without experiments on other open-source frameworks (e.g., CogVideoX, VACE, HunyuanVideo), it’s hard to judge whether the dataset quality generalizes.
3. The paper compares mainly with older datasets, while recent large-scale subject-consistent datasets are not included. It is unclear how Phantom-Data performs relative to the latest baselines, such as OpenS2V.

### Questions
As seen in weakness

### Soundness
3

### Presentation
3

### Contribution
2
