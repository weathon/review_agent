# Optimal Generative Cyclic Transport between Image and Text

- Decision: Reject
- Scores: 6, 5, 6, 5

## Abstract
Deep generative models, such as vision-language models (VLMs) and diffusion models (DMs), have achieved remarkable success in cross-modality generation tasks. However, the cyclic transformation of text $\rightarrow$ image $\rightarrow$ text often fails to secure an exact match between the original and the reconstructed content. In this work, we attempt to address this challenge by utilizing a deterministic function to guide the reconstruction of precise information via generative models. Using a color histogram as guidance, we first identify a soft prompt to generate the desired text using a language model and map the soft prompt to a target histogram. We then utilize the target color histogram as a constraint for the diffusion model and formulate the intervention as an optimal transport problem. As a result, the generated image has the exact color histogram as the target, which can be converted to a soft prompt deterministically for reconstructing the text. This allows the generated images to entail arbitrary forms of text (e.g., natural text, code, URLs, etc.) while ensuring the visual content is as natural as possible. Our method offers significant potential for applications on histogram-constrained generation, such as steganography and conditional generation in latent space with semantic meanings.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work proposes a novel text-image-text reconstruction method in cross-modality generation by guiding generative models with a deterministic function. The authors use color histograms as constraints to perform precise reconstruction, and this method can be applied for data protection.

### Strengths
1. The authors propose a novel and effective method for the cyclic transformation of text → image → text.

2. The manuscript is well-written and easy to understand.

3. Both the visualization and quantitative results seem promising.

4. The use of color histogram guidance is interesting.

### Weaknesses
1. Additional robustness analysis is needed to evaluate the effects of noise, blur, JPEG compression, and other transmission channel distortions.

2. The first section should ideally begin with a discussion on common applications of text → image → text translation.

3. There are already numerous generative methods for data protection, so the contribution stated in line 442 seems somewhat overstated. The authors should discuss their differences from existing methods in more detail, such as:

    [r1] StegaStyleGAN: Towards Generic and Practical Generative Image Steganography (AAAI 24)

    [r2] Cross: Diffusion Model Makes Controllable, Robust, and Secure Image Steganography (NeurIPS 24)

    [r3] StegaDDPM: Generative Image Steganography based on Denoising Diffusion Probabilistic Model (MM 23)

    [r4] Generative Steganography via Auto-Generation of Semantic Object Contours (TIFS 24)

    [r5] Secret-to-Image Reversible Transformation for Generative Steganography (TDSC 23)

4. There are a few typo errors throughout the manuscript. For example, in line 391, “a100” should be corrected to “A100.”

### Questions
Please refer to Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper propose a method to generate images more aligned with the text prompt using optimal generative cycle consistency between image and text. They ensure the image consistency, using color histogram matching using optimal transport and show the generated images are natural looking for various tasks.

### Strengths
1. The idea of using optimal transport for color histogram matching is interesting.

2. The generated results looks good, to some extent.

### Weaknesses
1. Why matching color histogram is enough to generate consistent images? E.g., it might happen that two images have exact same color histogram, but totally differnet content. Color domain discripency makes sense if the diffusion model is conditioned on images, but for text-to-image diffusion model, it is highly unlikely that the prompt can capture the exact details. Please clarify this.

2. There are works which address the issue of (image->text->image) consitency using pretrained caption and diffusion models, comparing with those methods need to be done. 

3. Exact prompt and seed might vary across machines, implementations, how the authors justify this? E.g., they attempt to retreive the specific diffusion model and seeds from the images, which I am not convinced to be extracted using these details. Please clarify.

4. Instead of just visualization, the authors might quantify their approach for some recognition tasks, e.g., captioning, classififcation etc.

5. The paper lacks coherence sometimes, e.g., in the abstract they mentioned about steganography, but it is not explored anywhere. 

6. Overall writing and flow need to be improved.

[1] Roy et al. "Cap2aug: Caption guided image to image data augmentation". WACV 2025 

[2] Kondapaneni et al. "Text-image Alignment for Diffusion-based Perception", CVPR 2024

### Questions
Please justify the weakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents a novel approach, Optimal Generative Cyclic Transport (OGCT), aimed at achieving exact, cyclic transformations between text and image modalities. The core of this work involves encoding text into images using a soft embedding that aligns with a color histogram, which can be recovered deterministically. The technique leverages diffusion models and optimal transport to generate images that encapsulate the encoded text as a recoverable soft prompt. The paper proposes and evaluates multiple binning strategies, with the random binning having perceptually indistinguishable outputs from the unencoded images

### Strengths
- The paper addresses an important challenge in multimodal generative models, where information is typically lost during cyclic transformations. This application could be instrumental for secure communication and data integrity.

- The introduction of a reversible soft embedding that maps text to color histograms, recoverable through a deterministic optimal transport algorithm, represents a novel approach in the field.

- The method is described in a structured manner, with an explicit algorithm provided for the entire OGCT process. This detailed exposition facilitates understanding and reproducibility.

- The random binning algorithm effectively enables the storage of a large quantity of text information within images while keeping the visual impact minimal, showing potential in practical scenarios.

### Weaknesses
- The method's reliance on color histograms raises concerns about robustness. While the authors have demonstrated resilience to rescaling, the algorithm may be sensitive to other common augmentations like color jitter, cropping, rotation, blur, and Gaussian noise. Testing against these transformations is necessary to confirm its applicability in realistic, potentially hostile scenarios.

- The experiments presented do not cover a comprehensive analysis of the algorithm. Including experiments such as variable perturbation time steps and perturbation strengths would provide deeper insights into the algorithm’s robustness and limitations.

### Questions
- In Section 2.4, the authors introduce the $\varphi$ function for histogram matching in the diffusion model but do not provide clarity on its calculation through the optimal transport approach. It would be beneficial if the authors included pseudo-code or an explanation of how the "OT-histogram-matching" function is implemented.

- Could the authors clarify the selection of perturbation steps, $\tau$, used in the experiments? Additionally, has there been any investigation into how using fewer or different sets of perturbation steps impacts the performance, especially if perturbations are applied only at the last generation step ($t=1$)?

- If the method is inherently unsuited to handle augmentations such as color jitter, what alternative solutions do the authors suggest? Could other functions besides color histograms be incorporated into the algorithm? The paper would gain greater value by incorporating a more generalized set of transformations alongside color histograms to enhance the robustness of the method.

- The suggested algorithm performs VAE-decoding on $z_0^t$ values calculated during the generation process. Can the authors provide a study or an experiment they performed to show the reliability of using VAE-decoding on these approximate values of $z_0$?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
The paper presents the Optimal Generative Cyclic Transport (OGCT) framework, aimed at achieving precise, lossless cyclic transformations across image and text modalities. Leveraging color histograms as a guidance vector, the proposed framework optimizes "soft prompts" within a language model and constrains the diffusion process in a generative model to encode text-based information directly into image color histograms. The encoded images are then decodable to the original text sequence using the same histogram vector, making this framework robust and flexible for applications in secure communication and generative encryption.

### Strengths
The paper is well-organized and detailed.

### Weaknesses
1. The paper lacks an evaluation of OGCT’s robustness against common image compression and information degradation techniques, as demonstrated in Table 2 of the paper [1]. In real-world scenarios, images often undergo transformations that introduce minor information loss, such as JPEG compression, added noise, and other alterations, which could potentially affect the integrity of OGCT’s color histogram-based decoding. Without experiments that test OGCT’s performance under these conditions, it is unclear how well the framework would perform in practical applications where images are subject to compression or other modifications. 

2. The robustness of the OGCT under different types of input text or image complexity is not thoroughly tested. The framework may encounter difficulties when encoding highly complex images or diverse language styles, particularly with non-standard characters, which could affect accuracy in real-world scenarios.

[1] Yu, Jiwen, et al. "Cross: Diffusion model makes controllable, robust and secure image steganography." Advances in Neural Information Processing Systems 36 (2024).

### Questions
See weakness.

### Soundness
4

### Presentation
4

### Contribution
3
