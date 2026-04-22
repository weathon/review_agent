# VIRTUE: Visual-Interactive Text-Image Universal Embedder

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
Multimodal representation learning models have demonstrated successful operation across complex tasks, and the integration of vision-language models (VLMs) has further enabled embedding models with instruction-following capabilities. However, existing embedding models lack visual-interactive capabilities to specify regions of interests from users (e.g., point, bounding box, mask), which have been explored in generative models to broaden their human-interactive applicability. Equipping embedding models with visual interactions not only would unlock new applications with localized grounding of user intent, which remains unexplored, but also enable the models to learn entity-level information within images to complement their global representations for conventional embedding tasks. In this paper, we propose a novel **V**isual-**I**nte**R**active **T**ext-Image **U**niversal **E**mbedder (**VIRTUE**) that extends the capabilities of the segmentation model and the vision-language model to the realm of representation learning. In VIRTUE, the segmentation model can process visual prompts that pinpoint specific regions within an image, thereby enabling the embedder to handle complex and ambiguous scenarios more precisely. To evaluate the visual-interaction ability of VIRTUE, we introduce a large-scale **S**egmentation-and-Scene **Ca**ption **R**etrieval (**SCaR**) benchmark comprising 1M samples that aims to retrieve the text caption by jointly considering the entity with a specific object and image scene. VIRTUE consistently achieves a state-of-the-art performance with significant improvements across 36 universal MMEB (**3.1\%–8.5\%**) and five visual-interactive SCaR (**15.2\%–20.3\%**) tasks. The code, models, and benchmarks are available at https://github.com/sony/virtue.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a new visual-interactive text-image universal embedding model designed to address the limitations of existing embedding models in handling visual cues such as bounding boxes, masks, and clicks. It combines the segmentation model (SAM-2) with a pre-trained vision-language model (VLM), enabling users to visually specify regions of interest within images, thereby enhancing the model's understanding of entity-level information in images.To evaluate the visual interaction capabilities of VIRTUE, the authors also constructed a large-scale benchmark named SCaR.

### Strengths
1. New Task: This paper introduces a novel visual-text embedding paradigm that, for the first time, incorporates visual cues such as bounding boxes, points, and masks into a universal embedding model. This approach overcomes the limitations of traditional models that exclusively support textual prompts and expands the possibilities for human-computer interaction.
2. Valuable Benchmarks: SCaR is the first large-scale benchmark designed for visual-interactive image-text retrieval, filling a gap in this field with significant diversity and challenges.
3. Clear writing and comprehensive experiments.

### Weaknesses
The primary contribution of this paper lies in the introduction of a new dataset. The proposed method does not present any technical innovations; rather, it merely combines SAM and VLM. Consequently, the technical contributions of this paper are quite limited.

### Questions
1. VIRTUE has brought significant performance improvements. To clarify the respective contributions of the dataset and the methodology, I would like to konw performance variations of other methods in the two scenarios of using SCaR and not using SCaR, as well as the outcomes of the proposed method when SCaR is not employed.
2. The generation of negative samples in the SCaR dataset relies on GPT-4V. Are there any biases or issues related to semantic repetition?

### Soundness
3

### Presentation
3

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
This paper proposes a visual-prompt MLLM-based embedding model and a corresponding new benchmark for measuring visual-prompt retrieval. The authors observe that existing MLLM-based embedding models can only accept textual instructions. Therefore, they proposed a new model VIRTUE that leverages similar visual prompts with SAM. Experiments are conducted on both the proposed benchmark and an existing benchmark.

### Strengths
1. The new problem setting is interesting and novel. I think visual prompts are complimentary to textual prompts to MLLM-based embedding models and can be practical in real-world applications.

2. The paper contributes a new method and a new benchmark which may be beneficial for both the methodology and data perspectives. Meanwhile, the method itself seems reasonable.

3. Experiments on both the proposed benchmark and existing benchmark seem to validate the effectiveness of the proposed method.

### Weaknesses
1. The scope of the new benchmark seems narrow. The main task of the new benchmark is caption retrieval. However, most existing benchmarks (e.g., MMEB and MBEIR[1]) for these models contain more tasks such as VQA retrieval, composed retrieval, or visual-document retrieval. The proposed benchmark seems to be more narrow than existing benchmarks that can undermine its universality.

2. The technical novelty seems to be limited. The model seems like a combination of existing MLLM-based embedding models and SAM. The textual prompts and visual prompts are from existing works and the forwarding and training process are similar to existing MLLM-based embedding models.

3. Experiments are only conducted on Qwen2-VL models. It is also recommended to validate the method on different herds of MLLMs. Meanwhile, other important baselines GME-QWen[2] which also uses QWen2-VL  are missing.

4. While handling non-visual-interactive scenarios, the uniform visual prompts seem to be redundant. Since all inputs are concatenated into one sequence, I think directly discarding the visual prompts can be more efficient in this case.

1. Wei C, Chen Y, Chen H, et al. Uniir: Training and benchmarking universal multimodal information retrievers[C]//European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2024: 387-404.

2. Zhang X, Zhang Y, Xie W, et al. GME: Improving Universal Multimodal Retrieval by Multimodal LLMs[J]. arXiv preprint arXiv:2412.16855, 2024.

### Questions
See above weakness

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces VIRTUE, a multimodal embedding model that extends image-text embedding models with visual prompts/instructions. Unlike existing embedding models that only accept text instructions, VIRTUE integrates a segmentation model to process visual prompts (bounding boxes, points, masks) alongside text and images. The authors also contribute SCaR, a new synthetic dataset for evaluating visual-interactive image-to-text retrieval.

### Strengths
- They address an existing gap that current models do not support visual prompts. I would, however, love to see this point argued more. E.g., by including examples of real-world uses.
- Contribute a new dataset (SCAR) with challenging negatives
- Their method proposed an elegant adaptation of VLMs for producing embeddings, with image instructions using bounding boxes, which I believe has real-world applications

### Weaknesses
- The proposed method, while increasing performance and addressing an existing limitation, does so by increasing model complexity, which might make the model harder to run, optimize, and adapt
- The dataset (SCAR) only explores visual prompts, while I do think that most real-world use cases would use both visual and text prompts. E.g., in the "find similar product" feature, you could mark an item, which then retrieves using the bounding box and a prompt "find products similar to the one highlighted".
- The heavy reliance on GPT-4V and lack of metrics, such as inter-annotator agreement, makes it hard to evaluate the overall quality of the dataset. It is also uncertain how much of the performance gain is learning specific quirks of GPT-4V and how much is generalizable
- I would update figure 2 to include all elements in the filtering (e.g. wordnet)
- Appendix E: Don't name your section "extensive experiments", just call it experiments
- I would argue that the statement: “they only accept text as the human-machine interaction modality” (Anonymous, 2025, p. 1) is borderline incorrect. However, I agree that the model was at least not designed with that intention in mind.
- see question on choice of fine-tuned models

### Questions
- why did you choose MMEB over alternative comparable benchmarks
- I am unsure if this model actually supports interactive selection that does not align with the SAM2. Did you experiment with arbitrary interactive human selection?
- Why are only some of the models fine-tuned on MMEB? I am unsure what the selection process was here? Especially interesting it why seemingly competitive models (GME-7B), weren’t fine-tuned.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes VIRTUE, an embedding framework that enhances a frozen vision-language model with a frozen segmentation model (SAM‑2) and a lightweight connector to inject region-level tokens for fine-grained visual interaction. To evaluate this, the authors introduce SCaR, a large benchmark for region-in-context caption retrieval with challenging negatives generated by GPT‑4V. VIRTUE achieves notable gains over baselines on both MMEB and SCaR, with ablations covering alternative encodings and model variants.

### Strengths
1. The studied problem is sensible, articulating the gap between text‑only interactions in current embedders and region‑aware visual interactions.
2. A new segmentation-and-scene caption retrieval benchmark is proposed. It also includes case studies for image-to-image retrieval and on-the-fly correction via visual hints, showing great practical value.
3. The experimental results show the proposed framework achieves better results than baseline methods.

### Weaknesses
1) The core contribution is essentially feature concatenation from a frozen segmenter into a frozen VLM with a small connector and contrastive fine‑tuning. While effective, there is little architectural insight into how to fuse region tokens with global tokens beyond prepending and projecting; there is no principled modeling of cross‑token interactions (e.g., region‑aware attention, cross‑modal slotting, or routing). This makes the method feel incremental despite the strong results.
2) Baselines that “do not accept boxes” are evaluated primarily via textualizing the bbox or naive cropping; yet there exist stronger region encoders and grounding‑aware alternatives (e.g., leveraging GroundingDINO features, ROIAlign/feature pooling over detected regions, or using grounding‑style LLMs like Ferret) that might close the gap. It is not surprising to me that the proposed method can achieve better performance.
3) After SCaR‑finetuning, VIRTUE‑7B still drops on MMEB (68.6→66.8), while 2B is stable; baselines often degrade more. A deeper analysis is missing.
4) The study varies SSS, SAM‑2 size, and a few hyperparameters, but does not (i) compare frozen vs lightly finetuned SAM‑2, (ii) probe attention patterns to confirm that the LLM actually leverages segmentation tokens for compositional cues, or (iii) test robustness to noisy or misaligned visual prompts (jittered boxes, partial masks, off‑by‑k clicks). 
5) Please report end‑to‑end latency and memory for typical query modes (I2T with and without prompts; I2I) versus cropping and text‑only baselines, at matched precision.

### Questions
Refer to weakness part.

### Soundness
3

### Presentation
3

### Contribution
2
