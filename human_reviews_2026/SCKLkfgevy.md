# BioCAP: Exploiting Synthetic Captions Beyond Labels in Biological Foundation Models

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 4, 8, 6, 6

## Abstract
This work investigates descriptive captions as an additional source of supervision for biological multimodal foundation models. Images and captions can be viewed as complementary samples from the latent morphospace of a species, each capturing certain biological traits. Incorporating captions during training encourages alignment with this shared latent structure, emphasizing potentially diagnostic characters while suppressing spurious correlations. The main challenge, however, lies in obtaining faithful, instance-specific captions at scale. This requirement has limited the utilization of natural language supervision in organismal biology compared with many other scientific domains. We complement this gap by generating synthetic captions with multimodal large language models (MLLMs), guided by Wikipedia-derived visual information and taxon-tailored format examples. These domain-specific contexts help reduce hallucination and yield accurate, instance-based descriptive captions. Using these captions, we train BioCAP (i.e., BioCLIP with Captions), a biological foundation model that captures rich semantics and achieves strong performance in species classification and text-image retrieval. These results demonstrate the value of descriptive captions beyond labels in bridging biological images with multimodal foundation models

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes BioCAP, a biological multimodal model that leverages descriptive captions as an additional supervision signal alongside species labels. The authors argue that images and captions provide complementary views of a species’ latent morphospace, capturing biologically meaningful traits while suppressing spurious correlations. To overcome the scarcity of instance-level captions, they generate synthetic captions using a large multimodal language model (InternVL3 38B) guided by Wikipedia-derived visual information and taxon-tailored format examples. BioCAP is trained with a shared visual and text encoder, but uses dual visual projectors for taxonomic labels and captions. Experiments show BioCAP improves species classification and text–image retrieval with several baselines, while careful ablation studies demonstrate the benefits of their approach.

### Strengths
- **Novel approach**: The use of synthetic descriptive captions for biological multimodal models is innovative and addresses a clear bottleneck in organismal biology.
- **Strong empirical performance**: BioCAP outperforms multiple baselines (CLIP, SigLIP, BioTrove-CLIP, BioCLIP, FG-CLIP) on ten species classification benchmarks, retrieval tasks (PlantID, Cornell Bird), and natural language understanding (INQUIRE-Rerank).
- **Comprehensive evaluation**: The authors evaluate classification, retrieval, and language understanding, covering multiple organismal domains.
- **Human evaluation and careful ablation studies**: Caption quality is assessed using four metrics (groundedness, specificity, completeness, and clarity) showing the effectiveness of synthetic captions and addressing hallucination concerns. Ablations confirm that dual projectors outperform a single projector, and that adding synthetic captions improves generalization, even for species without Wikipedia coverage.

### Weaknesses
- **Choice of base model and baselines**: BioCAP is initialized from BIOCLIP, but the paper does not justify why **BIOCLIP 2**, the current state-of-the-art, was not used as the base model or included as a baseline, despite being cited in the experimentation section. Furthermore, there is no discussion of alternative multimodal models (e.g., LLaVA) or rationale for selecting BIOCLIP.
- **Evaluation scope and claims**: Evaluation is limited to zero-shot species classification and a subset of retrieval benchmarks; prior work includes few-shot tasks and additional biological visual datasets (FishNet, NeWT, AwA2, Herb. 19, PlantDoc). 
- **Coverage limitations**: While ablation studies indicate improvements even for non-covered species, the paper does not explicitly analyze performance specifically on underrepresented species, leaving it unclear whether the approach benefits the full long-tail of species. 
- **Trait diversity in curated examples**: Only up to three examples per taxonomic class may bias the model toward common traits, potentially underrepresenting rare or atypical characteristics. An ablation study on different number of curated examples would be a nice addition.
- **Propagation of synthetic caption errors**: While human evaluation supports caption quality, errors or omissions in synthetic captions could still affect model performance for species with limited or no reliable descriptions.

### Questions
**Questions for the Authors**

1. Why was BIOCLIP 2, the current state-of-the-art biological multimodal model, not used as the base model or included as a baseline?
2. Have the authors evaluated performance specifically on underrepresented or rare species to assess generalization across the long tail of biodiversity?
3. How sensitive is the model to the number and diversity of curated examples per taxonomic class? Would increasing this number improve generalization or trait coverage?
4. Are there analyses on how synthetic caption errors propagate through the training process, especially for species with limited textual coverage?

**Actionable Feedback**

1. Compare BioCAP against BIOCLIP 2, the current state-of-the-art, and provide justification for choosing BIOCLIP as the base model. Discussion of alternative multimodal foundation models (e.g., LLaVA) would also strengthen the paper.
2. Include few-shot classification, additional biological visual benchmarks (FishNet, Newt-AWA2, Herb-19, PlantDoc), and more diverse retrieval tasks. Explicit analysis of underrepresented or rare species would help assess generalization.  
3. Assess whether limiting to three examples per taxonomic class biases the model, and consider ablations with more examples.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors create synthetic instance-level captions of images from the TreeOfLife-10M dataset and use them to train a version of BioCLIP with a caption encoder. The captions are morphological trait-based descriptions of the animals that correspond to subregions in the images. The authors articulate their synthetic caption generation strategy, report benchmarks on classification and retrieval, and perform a detailed ablation study. The results indicate that the trait captions improve classification and help constrain model attention to animal parts that help with classification.

### Strengths
- Originality: the authors use an interesting approach to caption generation, prompting MLLMs with format specification on a per-class basis. The idea was to constrain output to focus on salient morphological description that can be challenging to pick out of raw text without guidance. The approach to leveraging the captions in BioCLIP is appealingly simple. 
- Quality: The overall qualtiy is quite good, lots of experiments executed with sufficient data. The ablation studies do a nice job of illustrating the benefits of adding these captions to the model. 
- Clarity: The paper is well-written and relatively easy to follow. There are a few sections (noted below) that could benefit from some rewriting. 
- Significance: This work provides more evidence that species classification is easier when you can direct the model to pay more attention to informative parts of the target. The use of captions, and the way in which they were generated, is a useful addition to the toolkit.

### Weaknesses
The format example design discussion needs expansion. This seems to be a critical element of the work and it is treated very narrowly. It isn't clear what the 'classes' are that were used to query Gemini Deep Research, who did the winnowing of the results, or how consistent the examples were across the classes. This element may itself benefit from an exploration of how variable those exemplars are between model runs and how consistent the human overseers where in selecting appropriate descriptions.

### Questions
- What is the taxonomic breakdown of the resulting descriptions? At line 224 you indicate the descriptions cover ~32% of the species in TreeOfLife-10M. Are there any biases in how the generated captions or is it uniformly distributed across organism groups?
- What are the limitations of the caption generation strategy? Is the 32% coverage a fundamental limit of what is available on Wikipedia? 
- Are there any potential issues when mapping genus level descriptions onto species? Does that result in multiple species having the same descriptions?
- Can you elaborate on the human evaluation task? Per the supplement, there was only one domain expert in the group of annotators and 15 computer science students. How well prepared are your human validators to assess the quality of the captions? What is the ecologist's area of expertise? 
- What is the overlap in the species coverage in the zero-shot datasets? How much of the performance improvement is related to prior knowledge of the organism groups being tested?
- At line 230, it is noted that there are 347 taxonomic classes in TreeOfLife-10M. Is that a typo? Or are classes in this case the level of the taxonomic tree?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This work present a biological multimodal foundation model that integrates synthetic, descriptive captions as additional supervision beyond taxonomic labels to improve species understanding. The authors propose to generate synthetic instance-level captions using multimodal LLMs guided by Wikipedia-derived visual information and taxon-specific format exam, then train a model with dual text views(species name + caption, )and dual visual projectors to decouple supervision. The results shown more than 20% improvements across 10 classification tasks.

### Strengths
- Multimodal alignment in biology is an under-explored and important task.
- The paper propose an interesting idea that images and captions are treated as complementary projections of a species’ latent morphospace, so aligning them helps capture diagnostic traits while suppressing noise.
- The paper introduced a dual-projector architecture elegantly separates taxonomy vs. caption supervision, and use context-guided caption generation (Wikipedia + format examples) effectively mitigates LLM hallucination.
- The evaluation is comprehensive, including multiple benchmarks, both quantitative and qualitative analysis.

### Weaknesses
- The caption reliance on Wikipedia-derived descriptors could reinforce taxonomic bias and exclude rare or poorly documented species. How does the framework handle species without any Wikipedia entry or minimal trait descriptions?
- The repeated LLM re-generation could produce inconsistent style or attribute focus across species, causing the semantic drift.
- Potential scalable issue since the caption generation and derail-view training are computational costly.

### Questions
- Is there any caption noise? how did you filter those?
- What effort is needed to extend the framework to video data?
- The method is specifically designed for biology domain. How is the generalization capability to other scientific domains?
- Is there any quantitative result for Grad-CAM?
- Who are the human annotator? are they biology expert?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes a BIOCLIP-style model that, during training, aligns images with taxonomic labels and instance-specific synthetic captions generated by an MLLM. The captions are constrained by Wikipedia-derived visual descriptions and taxon-tailored format examples to reduce hallucinations. The evaluation show strong performance for species classification and text-image retrieval by training on TreeOfLife-10: +8.8% average top-1 over BIOCLIP across 10 zero-shot species classification sets, and +21.3% average on natural-language tasks/retrieval. Ablations show that human eval favors the proposed model's captions over others. 
The qualitative analysis with Grad-CAM images and t-SNE embeddings further highlight the improved embeddings with BioCAP.

### Strengths
The paper proposes interesting tweaks like using images and captions as complementary views of a species’ latent trait vector and training with contrastive learning to emphasize diagnostic features. The dual projector cleanly separates heterogeneous supervision as compared to a single projector. 

The paper will be of reasonable interest for the broader community interested in training VLMs for life sciences.

### Weaknesses
Captions are biased toward the chosen InternVL3-38B; and there is no cross-MLLM comparison. 
Behavior labels for analysis are auto-assigned by GPT-4o that may cause label drift. 
Large deltas are shown in experimental results, but statistical significance/confidence intervals aren’t reported for benchmarks.

### Questions
can caption-aligned features also help zero-shot retrieval/classification?
How do results change with different captioners?
What is the dedup protocol used? How do you ensure held-out test set?
Are there any cases of observed hallucination in the final captions?

### Soundness
2

### Presentation
3

### Contribution
3
