# OpenFake: An Open Dataset and Platform Toward Real-World Deepfake Detection

- Decision: Reject
- Scores: 4, 2, 2, 6

## Abstract
Deepfakes, synthetic media created using advanced AI techniques, pose a growing threat to information integrity, particularly in politically sensitive contexts. This challenge is amplified by the increasing realism of modern generative models, which our human perception study confirms are often indistinguishable from real images. Yet, existing deepfake detection benchmarks rely on outdated generators or narrowly scoped datasets (e.g., single-face imagery), limiting their utility for real-world detection. To address these gaps, we present OpenFake, a large politically grounded dataset specifically crafted for benchmarking against modern generative models with high realism, and designed to remain extensible through an innovative crowdsourced adversarial platform that continually integrates new hard examples. OpenFake comprises nearly four million total images: three million real images paired with descriptive captions and almost one million synthetic counterparts from state-of-the-art proprietary and open-source models. Detectors trained on OpenFake achieve near-perfect in-distribution performance, strong generalization to unseen generators, and high accuracy on a curated in-the-wild social media test set, significantly outperforming models trained on existing datasets. Overall, our results offer encouraging evidence that detectors trained with high-quality data can generalize to real-world social-media distributions.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces a new dataset named OpenFake, which collects real images from the LAION-400M dataset using predefined filter prompts and generates fake images with 18 modern generative models, including Stable Diffusion, Midjourney, and Imagen, based on a curated prompt bank. Compared with existing datasets that are often constrained to a limited domain, such as single-face imagery, OpenFake covers a much wider range of scenarios with richer contextual diversity. The dataset also incorporates the latest generative models released in 2025.

In addition, the paper presents a web-based interactive game designed to encourage community participation. Users can create prompts using different generators to produce fake images capable of evading current deepfake detection systems, thereby continuously expanding the dataset’s scale and difficulty. Several experiments are conducted on existing benchmarks to demonstrate the effectiveness of the proposed dataset.

From my perspective, developing new, diverse, and large-scale datasets remains crucial for advancing this field. Hence, OpenFake has the potential to make a valuable contribution and inspire future research.

However, from a research standpoint, this work still faces several fundamental issues that should be carefully addressed.

### Strengths
As mentioned in the summary, creating new deepfake datasets that better align with real-world scenarios is always valuable and important, since data itself remains one of the most critical factors driving progress in this field. Moreover, the paper is well written, making it easy to read and follow.

### Weaknesses
1. Subjective statements.
   Some claims in the paper appear subjective. For example, in Table 1, the authors define three levels of realism: Low, Medium, Good, and High. However, this categorization lacks objective validation. How are these levels defined, and on what basis is the newly proposed dataset considered “High”? A clear quantitative or perceptual justification is necessary.

2. Unfair experimental setup.
   The experimental design raises fairness concerns. In Table 3, the OpenFake dataset includes all the generators listed in the first column, while other datasets do not. As a result, OpenFake’s in-domain evaluation naturally outperforms others, which is unsurprising. Moreover, the compared datasets listed in the first row are not representative; important benchmarks such as Celeb-DF, DFDC, OpenForensics, and Celeb-DF++ are missing. Additionally, different deepfake detectors are used across datasets (e.g., SwinV2-small is applied to OpenFake, GenImage, and S.-Truth, while less powerful models such as EfficientNet-B4 and ConvNeXt are used for others), which makes the comparison inconsistent and potentially biased.

3. Lack of efficacy demonstration.
   To convincingly demonstrate the effectiveness, it is essential to evaluate state-of-the-art deepfake detectors on OpenFake and examine whether their performance degrades. However, the paper only reports results using SwinV2-Small, without comparisons to other detectors. From my understanding, SwinV2-Small is not a specialized deepfake detector, yet it performs well on OpenFake, which may instead suggest that the dataset is not sufficiently challenging, contradicting the stated motivation of this paper.

4. Questionable definition of “expandable.”
   The concept of building an interactive platform for dataset expansion is indeed interesting and valuable. However, the set of generators is fixed, with no indication of future modification or updates. Changing only the prompts may diversify image content but does not alter the intrinsic generator fingerprint. Consequently, the underlying data distribution remains unchanged, limiting true diversity and questioning the appropriateness of calling the dataset “expandable.”

### Questions
NA

### Soundness
3

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
5

### Summary
The paper introduces OPENFAKE, a large-scale dataset designed to address the shortcomings of existing benchmarks for DeepFake detection. It provides nearly four million images, pairing three million real, politically relevant photos with almost one million synthetic counterparts created by state-of-the-art generative models. The paper also proposes a crowdsourced platform called OPENFAKE ARENA, intended to keep the benchmark current by continually integrating new, challenging examples submitted by the community.

### Strengths
- The paper attempts to develop a more updated dataset through crowdsourcing routes to continually collect more and more difficult examples as they come up.
- The proposed dataset is focused on politically relevant content, which is the main target of misinformation in this day and age.
- The paper emphasizes ease of access by hosting the dataset on the Hugging Face Hub in a streaming-friendly format which supports open-science.
- The inclusion of a human perception study provides a valuable baseline that clearly demonstrates the imperceptibility of fakes from modern proprietary generators.

### Weaknesses
- The dataset primarily contains images that are fully AI-generated (from T2V/I2V models), and does not encompass more traditional DeepFakes like face-manipulations where a large spatial portion is real, and only the face region is manipulated.
- The real images are sourced from the LAION-400M dataset, which contains web-crawled content from 2014–2021. Further, this dataset has received significant criticism related to copyright issues, and harmful content.
- The real images are further pre-2021 era, and the synthetic images are from more recent generators. This bias can be easily picked up by DeepFake detectors to classify real vs. fake, when in reality the models have learned a "shortcut" rather than actually grounding it's decision in visual cues. There is no evidence provided in the paper to prove actual visual grounding.
- The test set of real-world social media images is small, with only 163 fake images identified from about 2,000 samples. The authors state that difficult or uncertain cases were discarded during curation, which may have inadvertently removed the most sophisticated and challenging fakes, thereby making the detector's high performance appear more robust than it might be in a truly uncontrolled environment.
- The OPENFAKE ARENA is proposed as a novel solution to keep the benchmark updated. However, its effectiveness is entirely dependent on future, voluntary participation from the community. The paper does not provide evidence of its current usage or a concrete strategy to ensure a sustained flow of high-quality adversarial examples.
- The primary detector used for benchmarking is based on the SwinV2 architecture. While effective, this focus makes it difficult to know if the strong results are due to the quality of the dataset or specific advantages of the SwinV2 model. The paper lacks a broader comparison of different modern detector backbones trained on the same benchmark.

### Questions
- The paper is framed around political deepfakes and acknowledges a Western-centric data bias. What steps were taken to ensure the "politically relevant" filtering prompt does not favor specific political ideologies, and how might the dataset's geographic bias affect the detector's fairness and performance on content from underrepresented regions?
- The detector performs well on unseen models, but how can it be ensured it is learning a general concept of "fakeness" rather than overfitting to subtle, shared artifacts among the families of generators included in the training set? What is the risk that a completely novel generative architecture or family of generators would evade the detector?
- Is there any evidence of visual grounding achieved by the detector, or is it simply able to figure out the difference between the pre-2021 real images and more recent synthetic images.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This work introduces OPENFAKE, a large-scale dataset (3M real and ~1M synthetic images) targeting real-world deepfake detection, with an emphasis on politically relevant and high-realism images. The dataset is complemented by OPENFAKE ARENA, a crowdsourced adversarial platform designed to continually expand the benchmark with challenging, user-generated fakes.

The authors also conduct (1) a human perception study showing that humans perform near chance on modern proprietary models (e.g., Imagen 3, GPT Image 1), and (2) an extensive benchmarking study, demonstrating that a SwinV2-based detector trained on OPENFAKE achieves near-perfect in-distribution performance (F1 ≈ 0.99) and strong generalization to unseen generators and in-the-wild data (F1 = 0.86 vs. 0.08–0.26 for baselines). The paper concludes that with continually updated, realistic benchmarks, automatic detection of deepfakes remains feasible.

### Strengths
- The paper addresses the pressing issue of misinformation from AI-generated political content, filling an important gap left by outdated GAN-based datasets. 
- OPENFAKE combines millions of images from both open-source (e.g., Stable Diffusion, Flux) and proprietary (e.g., Imagen 4, GPT Image 1) generators—substantially broader in scope than existing benchmarks.

### Weaknesses
- The main technical contributions lie in dataset design rather than new detection or generation algorithms. This could weaken its appeal unless the ARENA framework is formalized further.
- The perception experiment is informative but small (100 participants × 24 images) and lacks demographic or statistical analysis beyond raw accuracy. A more systematic design (e.g., signal-detection metrics, inter-rater reliability) would strengthen credibility.
- The real image base (filtered LAION 400M) is mostly pre-2022, which may bias the dataset.

### Questions
See weaknesses above

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses key limitations of current deepfake detection datasets, which often rely on outdated generative models, narrow image content, and lack real-world context. The authors introduce OpenFake, a large-scale open dataset and platform designed to overcome these issues. Its central contribution is a politically-grounded dataset of approximately 3 million real images paired with nearly 1 million high-quality synthetic images from state-of-the-art models like Imagen 3 and GPT Image 1. To ensure the benchmark remains current, the authors also developed OpenFake Arena, a gamified crowdsourcing platform that continuously integrates challenging, user-submitted synthetic images. Empirically, detectors trained on OpenFake achieve near-perfect in-distribution accuracy and demonstrate strong generalization to both unseen generators and real-world social media data, significantly outperforming prior benchmarks. These results robustly support the continued viability of automated detection against increasingly realistic synthetic media.

### Strengths
- High Relevance and Timeliness: This paper directly confronts a real and pressing problem. In recent years, generative models have iterated at a breakneck pace—evolving from GANs to Diffusion and Flow Matching, with architectures shifting from UNet to DiT, and the landscape moving from open-source to powerful closed-source models. As the capabilities of these models grow, the deepfake detection community urgently needs a benchmark that reflects the current state-of-the-art (SOTA), particularly from these closed-source models. The introduction of OPENFAKE precisely fills this critical gap.

- nnovative Extensibility: Unlike nearly all existing deepfake datasets, which are static and fixed, OPENFAKE is designed for extensibility. Through the OPENFAKE ARENA platform, the dataset can be continuously augmented. This design is crucial for keeping pace with the rapid evolution of generative technology, fostering an ongoing adversarial dynamic between generation and detection.

- High Quality, Large Scale, and Realism: OPENFAKE surpasses existing datasets on several key dimensions. It incorporates 18 state-of-the-art open-source and closed-source models, ensuring technical currency. Furthermore, its real data is sourced from LAION-400M and intelligently filtered using a VLM (Qwen2.5-VL), specifically retaining "politically grounded" or "face-containing" content. This approach moves beyond the limitations of past benchmarks (e.g., ImageNet-based ones) and imbues the dataset with far greater real-world significance (e.g., for detecting political disinformation).

- Sufficient Experimental Comparisons: The paper provides robust experimental validation. It not only benchmarks various SOTA detection models horizontally but also vertically tests the generalization (OOD) of its trained model against unseen generators, demonstrating performance that significantly surpasses models trained on datasets like GenImage. Critically, this work strongly supports the thesis that detection should not rely only on digital watermarking; a high-quality, large-scale training dataset is itself a critical pathway to improving detection capabilities.

### Weaknesses
- Insufficient data types: The deepfake types in OPENFAKE are all generated by text-to-image. Although the quantity is large, many of today's mainstream models, such as gpt-image-1, seedream 4.0, gemin-2.5-flash, are multimodal image generation models that support both text-to-image generation and text-guided image editing. Many of today's deepfake images are not just generated from scratch (text-to-image) but are produced through local editing or inpainting. Therefore, it is suggested that the Arena platform could be expanded to include more image editing data in the future.

- Experimental validation is incomplete: The paper tests the model's generalization to unseen generators but does not consider adversarial attacks. For example, real-world scenarios may often contain both deepfakes and images with adversarial attacks targeted at deepfake detectors. It is recommended to add supplementary experiments to demonstrate performance in these "real-world scenarios". Furthermore, deepfake detection has recently shifted from single-modal (image-only) detection to multimodal methods combining text and images, for example, by fine-tuning VLMs such as Qwen-2.5-vl or Intern-VL for detection. The paper mentions using Intern-VL for zero-shot detection but does not report the detection performance of fine-tuned multimodal large models.

### Questions
1. The in-the-wild test set has an extremely small sample size (only 163 fakes), and there is a clear curation bias (i.e., difficult samples were discarded). How can the authors confidently draw the strong conclusion that 'automated detection in the real world is feasible and effective', rather than a more conservative conclusion—that 'the model performs well on your small, curated test set'?

2. Regarding adversarial robustness: Have you tested the performance of the SwinV2 model trained on OPENFAKE (F1 0.992) when faced with real-world images containing adversarial attacks (e.g., PGD, FGSM)? After all, in real-world scenarios, various types of attack images exist.

3. Regarding cross-modal detection: Considering the significant advantage of VLMs in understanding context, why did the paper not directly fine-tune a VLM (such as InternVL or Qwen-2.5-VL) as a SOTA detection baseline in the experiments? Given that real-world fakes (like fake news) always combine text and images, wouldn't fine-tuning a VLM be an evaluation method that is closer to 'real-world' detection needs?

### Soundness
4

### Presentation
3

### Contribution
3
