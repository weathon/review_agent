# PiQPerfect: Diffusion-based Same-identity Facial Replacement

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4, 2

## Abstract
With the growing success of diffusion models in computer vision, we explore their potential for same-identity facial replacement in photographs. Specifically, we propose a diffusion-based method, built on top of a pre-trained text-to-image model, that takes as input a portrait image of a person and a second reference image of the same individual, potentially captured under different conditions. The goal is to seamlessly replace the input face with the reference face, while keeping the background intact. Surprisingly, despite the clear real-world utility of this task, no recently published work has directly addressed face replacement in this specific setting. To support this goal, we construct a large dataset of image pairs depicting the same person under varying facial expressions and poses. Experimental results demonstrate that our approach produces more realistic and identity-consistent results than existing face reenactment models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes PiQPerfect, a diffusion-based method for same-identity facial replacement. Built on a pre-trained text-to-image model (SDXL), it replaces a face in a source image with the expression and pose from a reference image of the same person, preserving identity and background. Unlike prior cross-identity face reenactment methods that struggle with identity preservation, PiQPerfect leverages a distorted version of the driving face as input to avoid trivial copying and enhance realism. The authors also introduce the PiQPerfect dataset, a large-scale collection of same-identity image pairs with diverse expressions and poses. Experiments show superior performance over existing methods in both in-distribution (ID) and out-of-distribution (OOD) settings.
This paper also introduce a PiQPerfect dataset (~500k images, 6,724 identities) synthesized by animating ~25k neutral faces with LivePortrait to produce 20 expressive variants per identity, yielding >10M source–target pairs.

### Strengths
- The paper effectively identifies a practical yet underexplored gap—same-identity facial replacement—and convincingly argues why existing cross-identity reenactment methods are suboptimal for this specific task.
- The use of a distorted, masked driving face as input is an elegant solution that avoids trivial copying while providing strong supervision for pose and expression, directly addressing the core challenge of identity preservation.
- Multiple metrics (CSIM, FID, APD/AED/AGD, LPIPS/PSNR), ID vs OOD settings, and user study; appendix acknowledges baseline failures affecting FID reliability.  
- High-Quality, Purpose-Built Dataset: The introduction of the PiQPerfect dataset, with controlled generation of diverse expressions and poses from neutral faces, fills a critical data void and enables robust training for the target application.

### Weaknesses
- Missing Direct Comparison to Closest Contemporary Diffusion-Based Work. Despite categorizing the field and comparing to existing face-reenactment baselines, the paper omits discussion and direct comparison with several concurrent diffusion-based face swapping and same-identity replacement works, specifically Baliah et al. (2025)[1], Wang (2024)[2], and Galanakis et al. (2025)[3]. For example, DiffFace and REFace attempt diffusion-based face swapping leveraging facial guidance and are highly pertinent; not benchmarking against them weakens the empirical argument, especially as these methods may target similar strengths (identity, realism).

- The method described in this article was tested on the outdated SDXL and its migration performance was not tested on the modern DiT architecture. Currently, DiT has become mainstream in the field of image editing and generation.

- Artifacts and Failure Cases. The method, while overall strong, is not free from notable artifacts. Figure 11 and supplemental figures clearly display issues around eyes, hair, and illumination blending, especially in hard edge regions and with extreme poses. These failure modes would benefit from a more rigorous analysis—such as error maps or a systematic breakdown—rather than merely attributing them to limitations of the backbone.

Reference:
1.  Baliah, S., Lin, Q., Liao, S. (2025): "REFace: Realistic and Efficient Face Swapping: A Unified Approach with Diffusion Models" WACV2025
2.  Wang, F. (2024): "Face Swap via Diffusion Model" 
3.  Galanakis, S., Lattas, A., Moschoglou, S. (2025): "FitDiff: Robust Monocular 3D Facial Shape and Reflectance Estimation using Diffusion Models"

### Questions
See  Weakness.

### Soundness
2

### Presentation
3

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
The paper presents a diffusion based approach towards same-identity facial replacement in realistic photographs. The method revolves around finetuning a pretrained SDXL for this task on a unique dataset of source, replacement and target images. The primary insight appears to be using augmented copies of the replacement image to avoid trivial copying. The authors also contribute a dataset of triplets created using facial animation models. The paper shows empirical results for VoxCeleb, FFHQ and their new dataset.

### Strengths
1. The paper addresses a specific niche task of same face replacement that has not been addressed significantly before. The approach has fairly interesting use cases that may be relevant to modern imaging systems.

2. The authors release a new, well-curated dataset for face replacement that would be a useful training and evaluation benchmark. 

3. The presented empirical benchmarks show the models outperform existing face-replacement approaches on a variety of metrics.

### Weaknesses
1. The baseline comparisons do not consider models/pipelines that could be used for this task (for example, SimSwap-family, Roop variants, and even opensource/commercial general image generation/editing model ). The paper compares mainly to academic reenactment systems + one IP-Adapter setup (and some editing methods in the appendix) but should at least show the results of modern foundational generative models on this task as a baseline.

2. The generated training dataset relies on synthetic data generated using LivePortrait. However the OOD evaluation also uses a subset of the generated dataset, which overestimates the generalization performance. 

3. The draft mentions an internal dataset as being the seed for the generated data. Since this involves human images, I believe a diversity and demographic analysis of the data, and the synthetic data should be included with the paper. Also, the paepr does not mention if the human subjects were asked for consent. 

4. The presented approach also does not add any significant novelty beyond the dataset. The stacked reference image approach has been previously used by several state of the art image-gen models (Flux, Seedream).

Overall, I suggest the authors compare with frontier image gen models to see how they stack up on the same task to further bolster their empirical results and novelty of the work.

### Questions
See weaknesses above

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes a dataset and a model for replacing the face in the source image with that in the target image. The key is using a driving image capturing just the desired face with everything else blacked out so that the model can be trained without just providing the target image (which usually leads to trivial solutions). The target image can be of a different background or lighting condition, and only the face part should swapping.

The model called PiQPerfect segments out the face from the target image, data augments it, encodes the source, target, driving images, concats them with noise added to the target image (so that the model can’t cheat), and finally diffuses the stack into the target image.

The said dataset contains 500K images of almost 7K identities. Data come in (source, driving, target) tuples. The secret to this big size is that the authors started with 25K neutral faces and then autolabeled many new expressions using LivePortrait.

### Strengths
This is indeed a common use case, and I’m also surprised there was no prior work directly tackling this.

Good presentation, esp Figure 2, helped demonstrate the ideas clearly.

I like the autolabeling pipeline that enlarges the dataset by a lot. Also, the paper showed it’s not enough to just use LivePortrait since the background may change.

### Weaknesses
The method looks quite similar to many prior face works, one of which is DiffusionRig. DiffusionRig also targets identity preserving editing, but I am not seeing comparisons against DiffusionRig.

The wording on the dataset release sounds vague, so I’m not sure if the authors will eventually release the dataset or not. If the dataset is not released, the only useful part would be the method, but like I said above, it’s not much different than prior works. I’d say the whole package – the model, dataset, and the autolabeling pipeline – is much more valuable than just the model.

### Questions
Have you tried this on video? Can be per-frame. How will the temporal consistency look if we make a video of the same person with the face swapped?

Will the dataset be released if this paper gets accepted?

What was done to pseudo GT with artifacts from LivePortrait? Do you have filtering in the data pipeline?

### Soundness
2

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
3

### Summary
This paper introduces PiQPerfect, a diffusion-based method for same-identity facial replacement—replacing a person’s face in one image with another photo of the same person, while maintaining the background and overall realism.
The authors fine-tune SDXL using a distorted inner-face conditioning strategy that avoids trivial copying and introduce a large-scale synthetic dataset of ~500K identity-consistent image pairs generated with LivePortrait.
They report superior performance over state-of-the-art face reenactment and identity-preserving diffusion methods on both in-distribution and out-of-distribution benchmarks.

### Strengths
The focus on same-identity face replacement—as opposed to generic reenactment or swapping—is well motivated and practically relevant for photography and media applications.

### Weaknesses
1)  The proposed method primarily fine-tunes SDXL using spatial concatenation and a distorted-face conditioning input. These ideas are reminiscent of FaceAdapter, Instant-ID and IP-Adapter . While the task setup differs, the technical core offers only an incremental variation on prior conditioning strategies.

2) The PiQPerfect dataset is generated entirely via LivePortrait, raising concerns about domain shift and realism. There is no evaluation demonstrating generalization to real photographs.


3) Several baselines (FADM, FaceAdapter, HyperReenact) are designed for cross-identity reenactment and are not retrained for the same-identity setup, making comparisons less meaningful. Moreover, reported FID failures for some baselines compromise statistical validity.


4) The main metrics (FID, LPIPS, PSNR) do not reliably correlate with perceptual realism, and the user study (10 participants × 25 samples) lacks statistical power.

5) The ablation study focuses on architecture variants but does not isolate the effect of the distorted-face conditioning or quantify how each augmentation (relighting, masking) contributes.


6) The work directly intersects with face manipulation and identity-based generation, yet ethical safeguards are minimal. The “Potential Social Impact” section is cursory and limited to watermarking.


7) The main text spends disproportionate space on dataset construction and training details. The core ideas could be expressed more succinctly.


8) Claims of applicability to “group photo editing” are not convincingly demonstrated on real-world data.

### Questions
Generalization to Real Photos: The dataset and most evaluations are synthetic, generated using LivePortrait. Can the authors provide quantitative or at least visual results on real, non-synthetic photographs (e.g., celebrity photo datasets, mobile camera bursts, or natural photo pairs)?

Effect of the “Distorted Face” Conditioning: The main novelty is the use of a distorted version of the target face as a driving signal. Could the author comment on  (a) direct use of the unmodified target face, (b) random noise or masking instead of distortion, and (c) varying levels of distortion intensity?

Baseline Re-training: Were any baselines retrained or fine-tuned on your same-identity dataset (e.g., FaceAdapter, FADM)? If not, how do the author justify direct comparison to models trained for cross-identity reenactment?


Identity Preservation Metrics
The papers rely primarily on ArcFace cosine similarity (CSIM) and FID. Have you evaluated identity consistency with multiple face embedding models (e.g., AdaFace, ElasticFace, or MagFace) to verify robustness to embedding bias?

User Study Design
Could the authors provide more details about your user study (participant demographics, evaluation interface, instructions, and randomization)?


Ethical Data Collection and Consent
How did the authors ensure informed consent and proper licensing for the videos used to generate the PiQPerfect dataset? Were participants compensated, and will consent documentation be released with the dataset?

Watermarking and Misuse Prevention
The authors mention embedding digital watermarks as a mitigation step. Have you implemented or tested this? If so, how does the watermark survive downstream transformations (cropping, compression)?

Failure Modes Analysis
The appendix briefly mentions failures around eyes and hair. Can you categorize these errors (e.g., illumination mismatch, misalignment, missing details) and quantify their frequency?

Computational Efficiency and Latency
The paper states inference takes ~3 seconds at 576×576 on an A100 GPU. How does this scale to higher resolutions (e.g., 1024×1024) and to consumer hardware?

Public Dataset Release Plan
The paper stated the dataset “will be released publicly.” Could the authors clarify under what license, with what restrictions, and whether raw video data or only generated images will be released?

### Soundness
2

### Presentation
2

### Contribution
1
