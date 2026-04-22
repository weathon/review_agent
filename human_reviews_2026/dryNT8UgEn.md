# How far can we go with ImageNet for Text-to-Image generation?

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4

## Abstract
Recent text-to-image (T2I) generation models have achieved remarkable results by training on billion-scale datasets, following a `bigger is better' paradigm that prioritizes data quantity over availability (closed vs open source) and reproducibility (data decay vs established collections). We challenge this established paradigm by demonstrating that one can match or outperform models trained on massive web-scraped collections, using only ImageNet enhanced with well-designed text and image augmentations. With this much simpler setup, we achieve a overall score over SD-XL on GenEval and on DPGBench while using just 1/10th the parameters and 1/1000th the training images. This opens the way for more reproducible research as ImageNet is a widely available dataset and our standardized training setup does not require massive compute resources.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper argues you can train competitive text-to-image (T2I) diffusion models using only ImageNet by pairing it with (i) long, synthetic captions and (ii) simple image augmentations, instead of billion-scale web data. A 300–400M-param model trained this way reports +6% GenEval and +5% DPGBench over SD-XL, while using ~1/10 the parameters and ~1/1000 the training images, with a stated budget of ~500 H100 hours. The method first turns ImageNet into a T2I corpus via detailed captions (improving FID and compositionality), then combats overfitting and boosts multi-object reasoning with CutMix/Crop augmentations. The authors also show the ImageNet-only model can be fine-tuned for aesthetics (e.g., on LAION-POP), improving PickScore/Aesthetics/HPS/ImageReward and remaining competitive at higher resolutions.

### Strengths
1. Writing is clear and easy to follow. The intro crisply challenges “bigger is better,” motivates ImageNet as a reproducible alternative, and states the central question (“how far can we go with ImageNet for T2I?”). The paper lists concrete contributions (analysis of shortcomings, a standardized ImageNet-only training setup, models in the 300–400M range, and transfer to high-res aesthetics), which makes the narrative easy to follow.

2. Solid SOTA comparisons. Benchmarks on GenEval and DPGBench show competitive performance versus much larger web-scale models.

### Weaknesses
1. “general capabilities” claim is too broad. ImageNet is object-centric; the paper itself notes missing classes like person, and that AIO captions lack scene/interaction coverage—so action understanding, human/physics realism, etc., are under-tested here.

2. Proposed solutions are "just augmentations", limited novelty. Authors can make the contribution explicit as a standardized, reproducible ImageNet-only T2I recipe. Authors can add stronger ablations to show the recipe is robust, not just one augmentation choice.

3. Table 3 gains look trivial, some FID metrics get worse. DiT-I FID(Inception) worsens from 6.29 → 7.30 with CutMix, even though some compositional metrics improve; CAD-I FID goes 6.16 → 6.62 with CutMix. Crop sometimes preserves or slightly improves FID.

### Questions
Please see the weakness part.

### Soundness
2

### Presentation
3

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
This paper explores the data augmentation on ImageNet to crute a small but high-quality text-to-image dataset from a classification dataset without captions. For the text (caption) augmentation, it utilizes a open-souce MLLM as a synthetic captioner to generate comprehensive caption with diverse attrubutions. To avoid overfitting in training and improve compositionality abilities, the authors propose the image augmentation approach by CutMix on the original ImageNet samples. They trained the DiT and CAD architectures on the proposed augmented ImageNet with 1.2M samples. It can achieve outperformed or comparable performance with larger-scale models trained on larger dataset, such as SDXL, Pixart on GenEval and DPG-Bench. Further experiments on aesthetics and efficiency supports their method.

### Strengths
1. The propose data augmentation method on text and image of ImageNet is clear and well-motivated by original flaws: single categorical text, easy to overfit early, few samples for compositionality.

2.  The performance gain over AIO baselines is obvious and the model is comparable with larger models trained on larger dataset on classical metrics and human alignment metrics, such as GenEval, DPGBench.

3. The training and data efficiency make the method is easy to reproduce and practical.

### Weaknesses
1. Limited Novelty:  Although the data agmentation method on text and image is simple and insightful and the motivation for designs is complete and reasonable, there is limited novelty for the entire pipelines: the MLLM-based captioner and CutMix image augmentation are not new approaches, and are already widely used, such as in T2I evaluation (MLLM-based captioner) and general visual augmentation (CutMix).

2. The setting of "Task specific finetuning: aesthetics" is flawed. The data size of LAION-POP is on the similar scale level of ImageNet: 0.6M vs 1.2M. The performance of model further trained on the LAION-POP can not serve as the support "suggesting that the model has hidden aesthetics capabilities". It is not a light extra training to unlock the hidden capabilities, but a substantial additional supervised training. The performance gain is limited. Also, the test-time scaling results cannot serve as a evidence for authors' claim, as TTS is sensitive with the reward model and the comparaison between TTS of original model is missing. 

3. Limited baselines: the comparad SDXL, SD v1.5, SD v2.1 are not between current T2I SOTA.  More advanced small baselines should be included.

4. The proposed data augmentation method is only conducted on ImageNet. Extra implementation on similar classification/caption-missing/object-centered dataset can further support the design of the augmentation pipeline.

### Questions
Questions are listed in Weaknesses section (1-4).

### Soundness
2

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
3

### Summary
The paper proposes a reproducible text-to-image generation framework trained solely on ImageNet, challenging the belief that massive web-scraped datasets are essential. Using synthetic text augmentation (via LLaVA-generated captions) and image augmentations (CutMix, cropping), the authors train compact DiT-I and CAD-I diffusion models with only 1.2M images. Despite the small dataset, their 400M-parameter models outperform larger systems like SDXL and PixArt-Σ on GenEval and DPGBench benchmarks. The models also serve as strong pretraining checkpoints for fine-tuning, achieving high aesthetic quality after adaptation to the LAION-POP dataset, all within a modest compute budget.

### Strengths
1. Data efficiency: Demonstrates that high-quality text-to-image generation can be achieved with 1/1000th the data and 1/10th the parameters of leading models such as SDXL, challenging the “bigger is better” paradigm.

2. Low compute requirement: Trains full models within ~500 H100 GPU hours, making high-quality text-to-image generation accessible to small research groups.

3. Strong quantitative results: Outperforms large-scale models (e.g., SDXL, PixArt-Σ) on GenEval (+6%) and DPGBench (+5%), showing surprising competitiveness given its compact size.

### Weaknesses
1. Restricted data diversity: Even with synthetic captions, ImageNet remains object-centric and lacks diverse scenes, human activities, and fine-grained contextual relationships. This limits the model’s ability to generalize compositionally and handle abstract or artistic prompts.

2. Synthetic caption bias: The text augmentation relies on automatically generated captions using LLaVA, which can introduce hallucinations, repetitive phrasing, and dataset-specific biases that weaken genuine text–image understanding.

3. Overfitting risk: Due to the relatively small size of ImageNet (around 1.2 million images), the model begins to overfit after approximately 200k steps. Although augmentations like CutMix and cropping help, they do not fully eliminate the issue.

4. Limited benchmark coverage: The evaluation focuses mainly on GenEval and DPGBench, which assess compositional accuracy but not creativity, long-text reasoning, or stylistic variation, leaving gaps in holistic model assessment.

### Questions
1. Could the authors extend their evaluation to include creative or stylistic benchmarks to assess broader generation capabilities?

2.  How might the model perform if trained with additional open datasets that include humans, scenes, or artistic elements alongside ImageNet?

3. During fine-tuning, is there a measurable drop in compositional or semantic accuracy that accompanies the aesthetic gains?

### Soundness
2

### Presentation
3

### Contribution
2
