# Extracted Weaknesses and Limitations for Multimodal Attribution Methods
## Analysis of Papers Related to Vision-Language Models

This document extracts specific weaknesses, limitations, and failure modes from three papers on multimodal vision-language learning that are directly applicable to a paper on "Adaptive Information Bottleneck for Multimodal Attribution in Vision-Language Models" (AdaIB).

---

## PAPER 1: SynthCLIP - Are We Ready for a Fully Synthetic CLIP Training?
**File:** 7DY2Nk9snh.txt

### Main Topic
Training CLIP (vision-language) models on entirely synthetic text-image pairs generated using LLMs and text-to-image models, analyzing the advantages and disadvantages of synthetic data.

### Key Weaknesses and Limitations Applicable to Multimodal Attribution

#### 1. **Distribution Shift Between Synthetic and Real Data**
**Direct Quote:** "This is most likely due to the distribution shift between real and synthetic data. As diffusion models advance towards generating more realistic images, we believe this gap will continue to narrow, resulting in greater benefits from synthetic data across various tasks."

**Application to AdaIB:**
- Multimodal attribution methods trained on synthetic or controlled data may suffer from poor generalization to real-world vision-language data
- Attribution explanations learned from synthetic data may not transfer to natural images and authentic text descriptions
- Distribution shifts can distort attribution patterns learned during training

#### 2. **Poor Alignment Between Images and Captions**
**Direct Quote:** "While it is easy to scale the number of unique samples, this comes at an increased difficulty in controlling their quality, which may result in a poor alignment between images and corresponding captions."

**Application to AdaIB:**
- Multimodal attribution requires precise correspondence between visual and textual modalities
- When text-image alignment is poor, the model cannot learn accurate attribution relationships
- This becomes critical when trying to explain which parts of an image correspond to specific text tokens

#### 3. **Alignment Issues in Image Generation**
**Direct Quote:** "TTI models could miss details in text prompts... recaptioning corrects alignment issues from the image generation process (e.g., the missing bench in the generated image)."

**Application to AdaIB:**
- Generated images may contain spurious features or miss important semantic elements
- Attribution methods may incorrectly attribute importance to missing or hallucinated visual regions
- Noisy text-to-image generation creates unreliable training signal for learning attribution

#### 4. **Content Generation Mismatches**
**Direct Quote:** "CLIP + Text-to-Image shows less marked improvements and no gains in few-shot performance... possibly due to domain shifts and content generation mismatches in synthetic images."

**Application to AdaIB:**
- When synthetic images don't accurately reflect the semantic content of their captions, attribution learning fails
- Multimodal attribution must map text to image regions, but mismatches break this mapping
- The model may learn to attribute text to incorrect visual regions when content generation is imperfect

#### 5. **Noisy Captions and Quality Control Issues**
**Direct Quote:** "Due to the noisy nature of LAION captions, this also leads to a big drop in image retrieval (IR) and text retrieval (TR) results where LAION-3M lags by around 10% to both CC3M and SynthCI-3M in both IR and TR."

**Application to AdaIB:**
- Noisy captions break the association between text and image regions
- Attribution methods relying on text-image pairs will propagate caption noise
- The model may learn spurious associations between text tokens and visual regions

#### 6. **Computational Efficiency Concerns**
**Direct Quote:** "The most evident disadvantage of synthetic CLIP models is the computational effort required to generate the training dataset, which may lead to a high carbon impact. Our generation process currently takes approximately 6.5 days using a 48-A100-80GB GPU cluster, equivalent to 313 GPU days."

**Application to AdaIB:**
- Training multimodal attribution models with synthetic data generation is computationally expensive
- Scaling up attribution methods may require prohibitive computational resources
- Efficiency becomes a practical limitation for deployment and iteration

#### 7. **Copyright and Memorization Issues**
**Direct Quote:** "Moreover, there may be concerns related to copyright issues and memorization in text-to-image diffusion models."

**Application to AdaIB:**
- If training data comes from potentially copyrighted sources, attribution methods may inherit these issues
- Models may memorize specific image-text pairs rather than learning generalizable attribution patterns
- This creates reproducibility and legal concerns for deployed attribution systems

---

## PAPER 2: Text-Based Person Search in Full Images (ProtoDis-TBPS)
**File:** iINUF4n33F.txt

### Main Topic
Cross-modal text-based person search using semantic context decoupling and prototype embedding learning for matching text descriptions to visual regions in full images.

### Key Weaknesses and Limitations Applicable to Multimodal Attribution

#### 1. **Difficulty in Complex Scenes with Multiple Entities**
**Direct Quote:** "In complex scenes, especially those with multiple pedestrians in the image, it is often challenging to distinguish the target pedestrian from the background or other individuals. This leads to limited generalization capabilities."

**Application to AdaIB:**
- Multimodal attribution becomes significantly harder when multiple semantically similar objects are present
- The model struggles to localize which visual features correspond to specific text tokens when distractors exist
- Attribution explanations may incorrectly assign importance to nearby objects or backgrounds

#### 2. **Background and Context Interference**
**Direct Quote:** "Existing methods often rely on generating a large number of candidate regions for matching, but their robustness and cross-modal matching capabilities are limited, especially in complex scenes."

**Application to AdaIB:**
- Background context can interfere with accurate multimodal attribution
- When attributing text to image regions, the model must suppress irrelevant background information
- Poor context decoupling leads to noisy attribution maps

#### 3. **Challenges with Occluded or Partially Visible Objects**
**Direct Quote:** "For future work, we aim to further optimize the cross-modal person re-identification module to better adapt to larger-scale and more complex scenarios. Additionally, we will explore incorporating external knowledge, such as scene context and camera viewpoints, to address real-world challenges like pedestrian occlusion."

**Application to AdaIB:**
- Occluded visual regions cannot be directly attributed to text
- Multimodal models must infer missing visual information from partial observations
- Attribution methods may struggle when text refers to occluded or invisible image regions

#### 4. **Modality Gap in Cross-Modal Matching**
**Direct Quote:** "The model can more accurately retrieve the target pedestrian based on textual descriptions" through techniques that reduce the "modality gap" between the two modalities.

**Application to AdaIB:**
- The fundamental gap between visual and linguistic representations hinders precise attribution
- Text tokens and visual regions inhabit different semantic spaces, making attribution ambiguous
- Bridging the modality gap is necessary but difficult for reliable multimodal attribution

---

## PAPER 3: ScImage - How Good Are Multimodal LLMs at Scientific Text-to-Image Generation?
**File:** ugyqNEOjoU.txt

### Main Topic
Benchmark evaluation of multimodal LLMs' ability to generate scientific images from text descriptions, assessing spatial, numeric, and attribute understanding.

### Key Weaknesses and Limitations Applicable to Multimodal Attribution

#### 1. **Failure on Combined Understanding Dimensions**
**Direct Quote:** "All models face challenges in this task, especially for more complex prompts... While GPT-4o produces outputs of decent quality for simpler prompts involving individual dimensions such as spatial, numeric, or attribute understanding in isolation, all models face challenges in this task, especially for more complex prompts."

**Application to AdaIB:**
- Multimodal attribution fails when multiple semantic dimensions must be jointly explained
- Models may separately attribute spatial relationships, numeric information, and object attributes but fail to jointly explain their combined effect
- Complex multimodal phenomena require integrated attribution across multiple modalities

#### 2. **Weakness in Spatial Understanding**
**Direct Quote:** "Code-based models have difficulties especially with spatial understanding, while image-based models struggle the most with numeric understanding."

**Application to AdaIB:**
- Different model architectures have complementary weaknesses in different attribution dimensions
- Spatial attribution (where in the image) is particularly difficult for some architectures
- Single attribution method may not work across diverse model types

#### 3. **Difficulty with Numeric/Quantitative Information**
**Direct Quote:** "For the image generation models STABLE DIFFUSION and DALL·E, numerical comprehension poses the greatest challenge. Both models score between below 1.8 for numerical understanding, substantially lower than their scores for attribute understanding (2.7) and spatial understanding (above 2.0)."

**Application to AdaIB:**
- Multimodal attribution struggles with quantitative aspects (counting objects, numeric values)
- Text tokens expressing numbers are poorly attributed to visual regions
- Vision-language models have inherent difficulty with precise numeric grounding

#### 4. **Combined Understanding Degradation**
**Direct Quote:** "Due to their weakness in spatial understanding, tasks that involve combined understanding types—including numerical & spatial understanding, as well as numerical & spatial & attribute understanding—also tend to receive lower scores. Both GPT-4O_python and GPT-4O_tikz record their lowest scores when addressing prompts that require all three understanding types, in comparison to prompts focused on individual understanding types."

**Application to AdaIB:**
- Attribution accuracy degrades combinatorially as the number of jointly-explained dimensions increases
- Joint attribution of multiple semantic aspects is harder than component attribution
- Information bottleneck approaches must be carefully designed to preserve crucial information in multiple modalities

#### 5. **Compilation Failures and Generation Errors**
**Direct Quote:** "A significant drawback of models that generate code is the potential for compilation failures. For instance, GPT-4O experiences 35 TikZ code and 27 Python code compilation errors. LLAMA 3.1 8B has even much higher failure cases: 116 for TikZ mode and 113 for Python code, representing approximately 28% of all prompts."

**Application to AdaIB:**
- When generating explanations or attribution outputs, models may produce invalid outputs
- Attribution methods must be robust to partial or malformed explanations
- Error rates increase with task complexity, affecting reliability of multimodal attribution

#### 6. **Object-Specific Challenges**
**Direct Quote:** "Graph theory representation (e.g., nodes and edges in a binary tree or graph) poses great challenges for models, with an average score below 1.7 across all models, compared to above 2.0 for the remaining object categories."

**Application to AdaIB:**
- Certain object types are inherently harder to attribute to text descriptions
- Abstract objects (graphs, trees) are harder than concrete objects (shapes, real-world objects)
- Attribution methods may need object-specific handling

#### 7. **Lack of Physics Knowledge**
**Direct Quote:** "The generated images from some models reveal a lack of physics knowledge. In cases requiring an image of liquid in a container, the liquid is often placed incorrectly, not at the bottom of the container... A challenging scenario for most models is generating an object moving along a parabolic path... Another common issue across models is their difficulty in generating images that depict 'boxes placed on a slope at a specific angle'."

**Application to AdaIB:**
- Models fail to respect physical constraints and real-world knowledge
- Attribution of text describing physical phenomena is unreliable without grounded physics knowledge
- Without understanding causality and physical laws, attribution methods cannot explain visual scenes accurately

#### 8. **Evaluation Metric Unreliability**
**Direct Quote:** "Automated evaluations can be unreliable, particularly when recognizing precise directions in text and images, such as 'up' and 'down'. The precision required for evaluating scientific graphs presents significant challenges for current automated metrics... Standard multimodal metrics employed in the community have low correlations to our human annotators in our scientific domain."

**Application to AdaIB:**
- Standard multimodal metrics (CLIPScore, BLIP-ITC, etc.) fail to capture fine-grained attribution quality
- Automated evaluation of attribution explanations is unreliable
- Human evaluation is necessary but expensive for validating attribution methods
- Existing metrics may not capture spatial/directional aspects crucial for attribution

#### 9. **Compilation Errors Mask Model Capability**
**Direct Quote:** "The low scores observed for LLAMA 3.1 8B in Table 2 can largely be attributed to penalties for these compilation errors. If these were ignored (or could be fixed), LLAMA 3.1 8B would outperform AUTOMATIKZ."

**Application to AdaIB:**
- When evaluating attribution methods, output format failures can confound capability assessment
- A perfectly capable attribution method may produce low scores due to formatting issues
- Evaluation must account for technical failures separately from semantic failures

---

## Summary of Cross-Cutting Weaknesses for Multimodal Attribution

### Critical Issues

1. **Multimodal Alignment Problems**
   - Poor text-image alignment degradates attribution learning
   - Mismatches between modalities create unreliable training signals
   - Requires precise correspondence that is difficult to achieve at scale

2. **Distribution Shifts and Generalization**
   - Synthetic training data creates domain gaps
   - Methods trained on controlled data fail on natural images
   - Attribution patterns are sensitive to data distribution

3. **Complex Scene Understanding**
   - Multiple similar objects cause attribution confusion
   - Background interference obscures target attributions
   - Occlusion and partial visibility create ambiguity

4. **Multi-Dimensional Understanding**
   - Single attribution methods fail on combined understanding tasks
   - Spatial, numeric, and semantic attribution must be jointly addressed
   - Information bottleneck constraints make joint explanation harder

5. **Quantitative Information Grounding**
   - Numeric and counting information is poorly attributed
   - Vision-language models have fundamental weakness in quantitative reasoning
   - Attribution of precise values is unreliable

6. **Lack of World Knowledge**
   - Physics understanding is missing from multimodal models
   - Causal and physical relationships are not properly grounded
   - Attribution cannot explain physically impossible scenarios correctly

7. **Evaluation Limitations**
   - Automated metrics are insufficient for fine-grained attribution quality
   - Human evaluation is expensive and necessary
   - Spatial precision (directions, relative positions) is hard to evaluate automatically

### Architectural Implications for AdaIB

- Information bottleneck must preserve multi-dimensional information simultaneously
- Adaptation mechanisms needed for different object types and scene complexities
- Must handle noisy or misaligned training data gracefully
- Evaluation strategy must combine human judgment with automated metrics
- Robustness to compilation/format errors in output explanations

---

## References

1. **SynthCLIP Paper:** Training CLIP on entirely synthetic data; highlights distribution shifts, alignment issues, and computational costs of synthetic multimodal learning.

2. **ProtoDis-TBPS Paper:** Cross-modal text-based person search; demonstrates challenges with semantic context decoupling, background interference, and modality gap bridging in complex scenes.

3. **ScImage Paper:** Multimodal LLMs for scientific image generation; reveals failures in combined understanding, numeric comprehension, physics knowledge, and evaluation metric limitations.
