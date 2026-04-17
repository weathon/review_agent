# OmniPortrait: Fine-Grained Personalized Portrait Synthesis via Pivotal Optimization

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Image identity customization aims to synthesize realistic and diverse portraits of a specified identity, given a reference image and a text prompt. This task presents two key challenges: (1) generating realistic portraits that preserve fine-grained facial details of the reference identity, and (2) maintaining identity consistency while achieving strong alignment with the text prompt. Our findings suggest that existing single-stream methods fail to  capture and guide fine-grained identity details.
To address these challenges, we introduce \textit{OmniPortrait}, a novel diffusion-based framework for fine-grained identity fidelity and high editability in portrait synthesis. Our core idea is pivotal optimization, which leverages dual-stream identity guidance in a coarse-to-fine manner. First, a Pivot ID Encoder is proposed and trained with a face localization loss while avoiding the degradation of editability typically caused by fine-tuning the denoiser. Although this encoder primarily guides coarse-level identity synthesis, it provides a good initialization that serves as the identity pivot for optimization during inference.
Second, we propose Reference-Based Guidance, which performs on-the-fly feature matching and optimization over diffusion intermediate features conditioned on the identity pivot. In addition, our approach is able to generalize naturally to multi-identity customized image generation scenarios. Extensive experiments demonstrate significant improvements in both identity preservation and text alignment, establishing a new benchmark for image identity customization.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper explores the field of personalized image generation. It begins by introducing the Pivot ID Encoder to achieve coarse-grained identity consistency, thus avoiding the degradation of prompt alignment caused by directly fine-tuning the base model. The authors then propose RB-Guidance to enhance fidelity during inference. This approach leverages the correlation between the denoised latent features of the generated image and the reference image, improving the fidelity of facial features. Additionally, the paper presents the construction and curation of the large-scale dataset OmniPortrait-1M, which will contribute to further advancements in the community.

### Strengths
- The proposed method can be extended to multi-ID personalization without requiring multi-ID training data.
- The paper is well-written and easy to follow.
- The ablation study is thorough and effectively demonstrates the validity of each module.
- The introduction of the OmniPortrait-1M dataset is a valuable contribution.

### Weaknesses
- The paper claims that current state-of-the-art methods suffer from two main issues: low facial fidelity and failure to maintain prompt alignment. However, the methods demonstrated (InstantID, PhotoMaker, FastComposer) are outdated. In fact, more advanced approaches such as ConsistentID, Pulid, and InfiniteYou have already addressed these issues, making the motivation less academically compelling.
- 1）The operation of associating visual embeddings with text embeddings in the Pivot ID Encoder has already been widely adopted, for example, in PhotoMaker. Similarly, the idea of face localization loss is also present in FastComposer.   2）The insight behind RB-Guidance—utilizing diffusion feature matching to improve facial fidelity—has been extensively explored by other methods, such as Visual Persona and Consistory.
- The base model used in the paper, based on the U-Net architecture, is outdated. Currently, competitive base models are using the DiT architecture, and the effectiveness of the proposed method on DiT remains untested.
- The methods compared in the paper are outdated, while results comparing to more advanced methods are relegated to the appendix. Furthermore, only qualitative results are provided, without any quantitative experiments.
-  When using InstantID, the reference image should not be cropped. Instead, the original reference image should be provided.

### Questions
- This method does not propose a solution to the multi-person identity confusion problem but instead relies on the semantics of the prompt for distinction, e.g., “a man and a woman.” How would the method perform if the prompt were “two men” or “a man and a man”? Would it still function properly?
- On line 377, the authors state that previous methods often produce results resembling a “copy-and-paste” of the reference image, which is expected, since during training, the reference face and the ground truth face are always aligned. However, this method does not appear to make any substantial improvements to this issue during training. Why does this method avoid this problem in the qualitative experiments?
- The authors claim to have constructed and released OmniPortrait-1M, a large-scale, high-quality multimodal face dataset. However, I don't find access provided in the manuscript or appendix. Will the dataset be released upon acceptance?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work proposes method called OmniPortrait to generate photo-realistic image while preserve identity consistency and achieve high alignment with text prompt by introducing pivotal optimization and reference-based guidance (RB-Guidance). It also supports multi-identity customization.

### Strengths
1. Only Pivot ID Encoder and linear protection layer are needed for training which makes OmniPortrait more efficient than finetuning UNet model.
2. The proposed RB-Guidance enable OmniPortrait to have better ID-preserving generation.
3. Authors have conducted various experiments to verify the effectiveness of the method.

### Weaknesses
1. The size of evaluation dataset is too small, only 50 reference images are used. Moreover, the content of 30 text prompts used are not clear. Based some samples in Figure 4, it seems the prompts mostly descript general object rather than descript facial attribute itself.

2. Authors claim that OmniPortrait have better 
fine-grained identity details. However, it seems no relevant experiments support this. For example, authors should measure how facial attributes changes in generated face images according to text prompts that descript the wanted facial attributes. 

3. Author use RB-Guidance to optimize the energy function. However, how energy function associated with ID-preservation generation is under explained.

### Questions
1. Author mentioned they utilized SD and SDXL in Sec. 4.2. However, which base model are used in each experiment?
2. For measuring ID-preservation, why authors didn’t measure cosine distance between reference and generated faces images? It will be more convinced if measuring recognition accuracy among different face recognition models.
Other questions see in weakness.

### Soundness
3

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
4

### Summary
This paper introduces a personalized portrait generation framework based on pivotal optimization, i.e., combining the facial representation with a specific text token and optimizing it to preserve the identity. Specifically, the proposed method consists of two parts. First, an efficient pivot ID encoder is designed to generate the combined representation, which is learned by a face localization loss calculated between cross-attention maps and facial segmentation masks. Second, during the inference, a pivotal optimization objective is proposed to capture the feature correspondence between the reference image and the generated image. Moreover, a large-scale dataset is collected to train the proposed ID encoder. The experimental results on 50 reference images sampled from the CelebA-HQ and FFHQ datasets show that the proposed method outperforms a few baselines.

### Strengths
+ The proposed pivot ID encoder is simple (feature concatenation followed by a linear layer) and efficient. And from the quantitative results, it improves the base model significantly.

+ The designed facial localization loss and the reference-based guidance loss are reasonable and can be combined with other models seamlessly. 

+ Compared with the baselines, the proposed method balances text following and identity preservation better.

+ This paper is written well and conveys its idea clearly.

### Weaknesses
1. There are many hyperparameters (Eq. (3), (5), (6), (10), and (11)) in the proposed method, of which the settings seem to overfit the dataset (e.g., $t_0 = 671$) and require in-depth analysis. 

2. According to the ablation study (Table 2), the proposed face localization loss contributes most to the performance gains of the proposed method. However, it also lacks a comprehensive analysis, e.g., the qualitative examples of learned cross-attention maps, comparisons with other localization losses, and the selection of denoising layers for learning.

3. The base models of the proposed method and the baselines are out-of-date (e.g., SDXL). As a few qualitative results of some FLUX-based methods are reported in the Appendix, why not implement the proposed method based on FLUX for comparisons directly in the main paper?

### Questions
Q1: Among the baselines, are they trained on the proposed OmniPortrait-1M dataset as well (at least the key components of one baseline)? If not, the comparisons may be unfair. 

Q2: Will the proposed dataset be released?

Q3: How does the proposed method perform in the wild (e.g., non-celebrity)? 

Miscs.:
+ "PulID-FLUX" in Figure 10 should be "PuLID-FLUX".

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces fine-grained personalized portrait synthesis system upon diffusion models. To mitigate the shortcoming of existing models, it propose pivot-based optimization strategies, which ensures the identity of reference image and editability of input prompt.

### Strengths
- High fidelity with editability: The proposed system demonstrate pretty plausible outcomes while preserving input fidelity with prompt-based editability with reduced trade-off problem between them.
- Compatibility and extendability: This framework is basically plug-and-play approach with high compatibility. Moreover, some additional results including multi-identity injection showed the significant extendability of the proposed method.

### Weaknesses
- Advantage of the proposed pivotal optimization: For reference-based editing, there are several well-known technics in diffusion models such as cross-attention, attention map swapping, style injection via AdaIN and so on. Within the proposed RB-Guidance, there is no comparison about aforementioned approaches before proposing the pivotal-optimization. It is imperative to provide and discuss some limitations of existing guidance manner on reference path and how the proposed one is useful.
- Effects of Pivot ID Encoder: The authors propose a new trainable encoder to extract facial identity using CLIP-based model. However, it is common to adopt identity embedding models such as ArcFace, InsightFace to provide identity vectors within denoising process using cross attention module, etc. It is wondered the proposed Pivot ID Encoder offers more useful information in entire process, compared to off-the-shelf identity extractor.

### Questions
- Without pivotal optimization, is there any degradation when adopting existing manner like cross-attention or attention map swapping for reference path?
- Before validating the Pivot ID Encoder, how do off-the-shelf identity extractors work compared to the proposed one?
- It is recommended to discuss the diversity on prompt-based portrait synthesis in terms of facial expression and appearance. The results presented in the manuscript appear to have a tendency to just copy-and-paste the overall facial structure of the reference image, including the open mouth and smile, while PhotoMaker changed those attributes naturally.

### Soundness
4

### Presentation
4

### Contribution
4
