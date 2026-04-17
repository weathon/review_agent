# Textual Supervision Enhances Geospatial Representations in Vision-Language Models

- Decision: Reject
- Scores: 6, 6, 2, 4

## Abstract
Geospatial understanding is a critical yet underexplored dimension in the development of machine learning systems for tasks such as image geolocation and spatial reasoning. In this work, we analyze the geospatial representations acquired by three model families: vision-only architectures (e.g., ViT), vision-language models (e.g., CLIP), and large-scale multimodal foundation models (e.g., LLaVA, Qwen, and Gemma). By evaluating across image clusters, including people, landmarks, and everyday objects, grouped based on the degree of localizability, we reveal systematic gaps in spatial accuracy and show that textual supervision enhances the learning of geospatial representations. Our findings suggest the role of language as an effective complementary modality for encoding spatial context and multimodal learning as a key direction for advancing geospatial AI.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper explores how geospatial representations emerge in vision-only and vision–language interaction models without explicit geographic supervision. It analyzes how textual supervision influences the implicit spatial understanding of visual models by probing layer-wise embeddings and examining the role of prompts and feature manipulation. The results show that incorporating language signals enhances the encoding and controllability of geospatial semantics, suggesting that multimodal learning provides a stronger foundation for spatial reasoning than purely visual training.

### Strengths
The paper shows that CLIP and VLMs outperform larger vision-only models, confirming that language-based supervision induces finer spatial awareness.

The paper carried out several interesting experiments to prove the argument. Such as swapping geospatial features between images predictably alters generated locations while preserving semantics.

The paper is well-structured and easy to follow.

### Weaknesses
Without statistical testing, it is unclear whether observed differences are robust across runs.

The mechanism by which textual prompts re-activate spatial signals remains speculative. It is unclear whether this effect is specific to geospatially oriented queries or would also occur with unrelated textual inputs, making the causal relationship between language and spatial activation uncertain.

Section 4.4 defines “top p dimensions ranked by coefficients”, but it remains unclear how the obtained regression coefficients are ensured to be optimal or stable. Without verifying consistency across different initializations or regularization strengths, the ranking of dimensions may reflect run-specific artifacts rather than intrinsic geospatial factors.

Section 4.5 mostly shows qualitative examples, but there’s no quantitative analysis. Personally, I think it should include some measurable results, like how often the location edit actually works or stays consistent, to show the effect isn’t just anecdotal.

### Questions
In Section 4.3, does prompt-based improvement depend on the linguistic specificity of the query (e.g., “Where is this photo?” vs. “Guess lat/long”)?

What’s the meaning of “fine-grained”? It is not explicitly defined. Does it refer to coordinate-level precision (e.g., latitude/longitude accuracy) or to semantic granularity (e.g., distinguishing similar landmarks within a region)?

If two images have similar content (e.g., buildings sharing architectural styles) but are located in different countries or even continents, can the probe still recover their coordinates? And in such cases, which would perform better, vision-only or vision-language models? Another interesting observation is that clusters such as Food or People Closeups achieve positive R², even though their visual content seems sort of unrelated to geography from the examples. What cues do models rely on to localize these images?

### Soundness
2

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
This paper investigates how different model families — vision-only architectures (e.g., ViT), vision-language models (e.g., CLIP), and large-scale multimodal foundation models (e.g., LLaVA, Qwen, Gemma) — learn and encode geospatial representations.

Through clustering-based analyses of images (people, landmarks, objects) grouped by localizability, the authors demonstrate that textual supervision significantly enhances fine-grained geospatial understanding. The findings suggest that language provides complementary cues for encoding spatial context and highlight multimodal learning as a promising direction for advancing geospatial AI.

### Strengths
* Comprehensive evaluation across multiple model families, from vision-only encoders to advanced multimodal foundation models.
* Large-scale dataset construction and systematic analysis covering diverse image categories, enabling meaningful cross-model comparison.
* Empirical evidence supporting the hypothesis that textual supervision improves geospatial representation quality.
* Strong visualization and interpretability results, revealing how current models implicitly capture spatial cues.
* The topic lies at the intersection of interpretability and multimodal representation learning, which is timely and relevant to the ICLR community.

### Weaknesses
1. Insufficient discussion on recent visual self-supervised models (e.g., Web-SSL, language-free visual encoders [3]) that may already achieve strong spatial representations without textual supervision.

2. Data imbalance and representational bias — the paper acknowledges that landmark data are unevenly distributed. However, it does not examine whether this imbalance propagates into model performance, particularly for underrepresented regions or low-resource geographies.

3. Lack of temporal analysis — the study focuses on static, long-standing landmarks. It would be valuable to assess model understanding of newer or dynamic landmarks to evaluate the temporal robustness of geospatial representations.

4. Prompt-based enhancement is somewhat obvious — while textual supervision improves performance, the paper does not explore whether finetuning (e.g., SFT) introduces catastrophic forgetting or enhances model specialization.

### Questions
1. How does this work conceptually align with or differ from the Platonic Representation Hypothesis [1]? Does geospatial representation follow similar convergence trends across modalities?

2. Could you provide more discussion on how large-scale, language-free visual encoders (e.g., [3]) compare to multimodal ones in terms of spatial representation quality?

3. To what extent does data imbalance affect model geospatial reliability for low-visibility or low-resource regions?

4. Have you analyzed temporal sensitivity — i.e., whether newer landmarks absent from training data are recognized differently by various model families?

5. What would happen if the models were finetuned with explicit geospatial supervision? Would this improve or degrade their general multimodal reasoning ability?

**References**

[1] Huh, M., Cheung, B., Wang, T., & Isola, P. The Platonic Representation Hypothesis. ICML 2025.

[2] He, J., Nie, T., & Ma, W. Geolocation Representation from Large Language Models as Generic Enhancers for Spatio-Temporal Learning. AAAI 2025.

[3] Fan, D., Tong, S., Zhu, J., et al. Scaling Language-Free Visual Representation Learning. ICCV 2025.

[4] Menon, S., & Vondrick, C. Visual Classification via Description from Large Language Models. ICLR 2023.

### Soundness
3

### Presentation
3

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
The paper studies whether vision-only encoders, contrastive VLMs, and instruction-tuned MLLMs implicitly encode geospatial information without explicit geo-supervision. Using layer-wise ridge probes to predict latitude/longitude and reporting R^2, the authors find that models trained with textual supervision show stronger geolocation structure that vision-only models. They also report prompt-conditioning that preserves/boosts geospatial signal in later LLM layers, etc.. Data come from clustered YFCC100M and a sampled Google Landmarks subset with geocell balancing.

### Strengths
1. The claim is clear, that is textual supervision systematically strengthens geospatial representations across image types.
2. The authors attempt at geographic balancing via geocells; clusters in YFCC are described and visualized.

### Weaknesses
1. The conclusions of this paper are of no value. Its most important takeaway is that multimodal models outperform vision-only models in geospatial representation. This is easy to understand: multimodal models are trained on human knowledge—specifically, text—whereas vision-only models are trained merely to extract visual features without drawing on human knowledge. Under these circumstances, it is only natural that multimodal models would better capture geospatial representation.
2. The authors refer to the ability to identify where a photo was taken as ”geospatial representation,“ which is quite odd. Generally, geospatial representation should refer to the geospatial nature of the image’s own features.
3. Representation-swapping is a single-model, qualitative case study; interesting but not systematically evaluated (stability issues are acknowledged).
4. The prompt that asks for coordinates (“Guess the latitude and longitude…”) may teach the model to surface memorized textual priors rather than reflect purely visual geospatial encoding.

### Questions
If you size-match models and hold the image corpus constant while selectively corrupting or removing text supervision during pretraining, do the observed R² and downstream gains persist?

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
In this paper, the authors examine the geospatial representation capabilities of pre-trained vision-only foundation models, vision-language models and multimodal LLMs. The authors perform linear probing regression on various layers of the transformers in those pretrained models and analyze their performance in prediction of the longitude and latitude. Using this method, they are able to identify the top features associated with geospatial information and conduct an experiment where they swap that part of the features in text prompts to obtain manipulated model outputs from VLMs. They conclude that textual information is beneficial for geospatial representations in foundation models.

### Strengths
1. This paper presents a very interesting and creative research question, i.e. what kind of foundation models are good “geo-guessers”. This is a relatively new and previously unexplored area in understanding the emergent properties of large foundation models.
2. The authors are able to systematically explore and identify layers and dimensions where the models produce latent geospatial information representations. This method makes good contributions to the interpretability of these large models.
3. The experiments shown in Section 4.5 are genuinely surprising and interesting.

### Weaknesses
Although I do really like this paper, unfortunately there are a few aspects of this paper that make it not ready for publication right away.
1. The overall story of the paper is very messy and confusing:

    (i) In the introduction section, the authors summarize their research question as: “To what extent do these models internalize global location knowledge as an emergent property of their training and fine-tuning pipelines?” However, the remainder of the paper does not discuss or contain experiments that study this phenomenon as an emergent property. According to [1], “an ability to be emergent if it is not present in smaller models but is present in larger models”. However, this paper does not study model scaling as a factor of this ability.

    (ii) The experiments in Section 4.5, though the most interesting, seem very disconnected to the rest of the paper, since the rest of the paper is more about comparing vision-only encoders to textual-visual models, and Section 4.5 is mainly about steering VLMs. However, Section 4.5 is used as the featured example in Figure 1. This makes the overall flow of the paper extremely confusing.
2. The experimental results are not sufficient to support the main claim: as the title suggests, the authors claim that textual supervision can enhance geospatial representation in those models. However, the comparison conducted in this paper is among various pre-trained models that are trained in completely different settings. The setting differences include model size, training data, and training duration. Given that there are so many confounding factors, I don't think it is reasonable to directly draw the conclusion that textual-visual models are better because they incorporate text data – for example, it is possible that they are better simply because they have a better training data mixture, their model has better architecture, or their model is simply larger. It would be better if the authors compare vision-only models with textual-visual models in a more rigorous setting, i.e. keeping the data source, model size and model architecture the same and only excluding textual information in the vision-only setting.
3. Figure 3 is very confusing – what do these pictures represent in this plot?

Reference:

[1] Wei et al. Emergent Abilities of Large Language Models. 2022.

### Questions
It would be great if the authors can answer my questions in the Weakness section.

### Soundness
1

### Presentation
1

### Contribution
3
