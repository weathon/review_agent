# Learning Modal-mixed Chain-of-thought Reasoning with Latent Embedding

- Decision: Reject
- Scores: 4, 6, 4, 2

## Abstract
We study how to extend chain-of-thought (CoT) beyond language to better handle multimodal reasoning. While CoT helps LLMs and VLMs articulate intermediate steps, its text-only form often fails on vision-intensive problems where key intermediate states are inherently visual. We introduce modal-mixed CoT, which interleaves textual tokens with compact visual ``sketches'' represented as latent embeddings. To bridge the modality gap without eroding the original knowledge and capability of the VLM, we use the VLM itself as an encoder and train the language backbone to reconstruct its own intermediate vision embeddings, to guarantee the semantic alignment of the visual latent space.
We further attach a diffusion-based latent decoder, invoked by a special control token and conditioned on hidden states from the VLM. In this way, the diffusion head carries fine-grained perceptual details while the VLM specifies high-level intent, which cleanly disentangles roles and reduces the optimization pressure of the VLM.
Training proceeds in two stages: supervised fine-tuning on traces that interleave text and latents with a joint next-token and latent-reconstruction objective, followed by reinforcement learning that teaches when to switch modalities and how to compose long reasoning chains. 
Extensive experiments across 11 diverse multimodal reasoning tasks, demonstrate that our method yields better performance than language-only and other CoT methods. Our code and data will be publicly released.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a method to enable VLMs to interleave images (in latent space) into their CoT along with text. The main idea they used is to train a VLM to learn when to produce vision tokens, which are then converted to latent visual embeddings using a diffusion decoder. During training, they re-use the image encoder from the VLM to generate a fixed set of image embeddings which are trained using reconstruction loss after passing through the diffusion decoder along with the standard AR loss for text tokens. They also use GRPO to improve reasoning capabilities of the model. Their results show gains on visual reasoning and perception benchmarks over baselines and latency increases by approximately 70% due to the extra generation of visual embeddings.

### Strengths
1. The paper introduces a novel technique to allow VLMs to "think" in both text and visual spaces without the need for additional tools, allowing the model to visualize and render visual concepts freely. 
2. The architecture and loss functions are intuitive and simple, and can be applied on top of any existing VLM with an image encoder by training it with the reconstruction loss. 
3. The diffusion decoder and joint training ideas are ablated well and their methods show clear gains on a diverse set of multi-modal reasoning and perception benchmarks.

### Weaknesses
1. Lack of visualizations/qualitative depiction of visual latent embeddings: While their method shows improvements on several benchmarks, it is hard to understand what latent concepts the model is trying to visualize. It would be useful if the authors can either: 
 a) run end-to-end diffusion from latents to pixels to see what images are actually rendered
 b) on some benchmark like V*, compare the cosine similarity (or any distance metric) between image outputs from tool-use (using the Qwen2.5VL baseline) and the image representations generated from their diffusion decoder. 
 c) provide heat maps of similarity between patches on the image and the image embeddings to show which regions on the image are being focused on during the visual "thinking" process
2. Trade-off between latency and performance: There is a significant increase in the latency due to the image latent space generation process. It will be great if the authors can show a trade-off between performance and as it varies with 
a) number of visual tokens rendered (K/M) 
b) number of diffusion steps (T) (note: the value of T is also not mentioned in the paper
3. RL hurts performance: It is a little surprising that RL seems to decrease performance on benchmarks, especially on LogicVista, which should ideally improve with GRPO given that its a reasoning benchmark and longer CoT traces should generally help with better understanding of the problem, can the authors provide some additional RL training curves (reward, KL, solution length) to see why this might be happening 
4. Missing clean comparison with tool-use: One of the claims of the paper is that latent space reasoning allows the model to "sketch" its own latents, but there is no clean quantitative comparison which clearly shows tool-use vs non tool-use baselines. The authors mention that in the V* benchmark the baselines have access to tools and their method doesn't but it will be good to have exact rows in the different benchmarks where the performance of existing baselines is measured with and without tool use.

### Questions
please check the weaknesses section above, in addition: 

1. what is the value of T used during training and inference? 
2. Are K (during inference) and M (during training kept the same)?
3. can you please include some sample completions (text tokens without image embeddings) from your method vs baselines on some queries to understand how the CoTs look like.
4. On average, how many times does the model invoke the <START> token to visualize something?

### Soundness
2

### Presentation
2

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
The paper presents a method to embed interleaved CoT to the latent space of a VLM. During training, the model is taught to reconstruct the generate the intermediate image that is relevant to the reasoning traces. During inference, the model generates visual embedding that is beneficial to the reasoning process. Experiment results on vision-centric reasoning benchmarks show the method improve SOTA VLMs like Qwen2.5-vl and InternVL-2.5.

### Strengths
* For text only input, methods like COCONUT use CoT data to iteratively embed the CoT into the latent space. This paper is conceptually interesting as it offers a promising approach to embed visual CoT to latent embeddings more explicitly and intuitively with a generative approach, as opposed to the previous cosine similarity approach.
* The paper is well written and clearly presents the motivation to extend the prior works on latent CoT by working with intermediate visual embedding instead of text CoT latents.

### Weaknesses
* Empirically, the RL stage yield only marginal gains, making it unclear what the purpose of this stage is.
* Based on the ablations, the latent CoT introduce noticeable amount of latency to inference. At the same time, the latent CoT also generates more tokens than the baseline. It would be important to conduct the experiment in a controlled compute budge to measure the latency vs performance relation of visual latent CoT and text CoT.
* Reasoning VLMs like R1-Onevision and MM-Eureka should be included as the baselines.

### Questions
* Could you clarify whether the text-only ablation contains interleave images from the zebra-cot data, only text, or both text and the original image? This is important because the fair comparison would be with models trained on the original zebra-cot data with interleaved image to demonstrate the effectiveness of the diffusion objective.
* It seems that the method is bottlenecked by the need of carefully curated interleaved CoT data. What are some potential solutions for such limitation?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces modal-mixed chain-of-thought to enhance multimodal reasoning in VLMs. Unlike language-only COT, the approach interleaves textual tokens with latent visual embeddings. The architecture integrates a VLM with a diffusion-based decoder, aligning latent embeddings with the VLM's feature space to avoid modality gap issues. Experiments demonstrate the effectiveness of the approach.

### Strengths
1. Clear Motivation: The use of latent embeddings to form a vision-language interleaved chain of thought is a well-justified strategy. It effectively complements the strengths of pure-text chain-of-thought methods, presenting a reasonable approach for enhancing model performance in visual perception and reasoning tasks while maintaining inference efficiency.

2. Comprehensive Analysis and Ablations: The paper includes a thorough set of ablation studies. The analysis convincingly addresses key aspects such as catastrophic forgetting, computational efficiency, and hyperparameter tuning, providing strong empirical support for the proposed method.

3. High-Quality Writing and Presentation: The paper is well-written. The main figure is particularly intuitive and effectively illustrates the proposed framework.

### Weaknesses
1. Limited Novelty: For multimodal tasks, the chain-of-thought reasoning approach that involves either generating images or selecting image regions before answering questions has already been extensively explored in prior works [1][2][3]. In the embodied domain, methods that first generate latent or image representations before producing the final output have also been investigated  [4][5]. This limits the novelty of paper.

2. Insufficient Experiments:
  - Unfair Ablation Study: The paper indicates that the Zebra-COT dataset is used for training. However, examples from this dataset suggest that simply removing images results in linguistically incoherent reasoning, as subsequent reasoning steps may refer to previously referenced images. Proper post-processing of the text-only version of this dataset may be required to form a coherent reasoning chain. This affects the reliability of the ablation studies in Section 5.3.
  - Lack of Experiments on More Base Models: All experiments are conducted solely on the Qwen2.5-VL-Instruct model. This raises concerns about whether the method can generalize to other model families (e.g., InternVL, miniCPM) and models of varying parameter sizes.
  - Lack of Comparative Baselines: Apart from the JanusPro baseline, the other baseline models in the paper are evaluated directly without enhancements, making the baselines appear relatively weak. The authors should consider incorporating stronger or more advanced baselines (e.g., applying chain-of-thought prompting directly to these models, training models to autoregressively generate image tokens for reasoning, leveraging large models' attention mechanisms to insert image tokens into reasoning chains, or using soft tokens to train models in selecting visual regions before reasoning—all of which represent training-free vision-language chain-of-thought methods). At a minimum, for tasks solvable by directly selecting image tokens as part of the reasoning chain, the authors should strengthen comparisons and discussions with such approaches.

3. Lack of In-depth Discussion on Main Experimental Results: Table 1 shows that the proposed method does not lead to improvements across all benchmarks. Specifically, on the LogicVista and MM-IQ datasets, performance after RL training shows a noticeable decline compared to supervised fine-tuning (SFT). The paper lacks an analysis of the reasons behind these performance drops.

4. The authors may consider including more case studies to further strengthen the claims of the paper.

[1] Gao et al. (2025). "Interleaved-Modal Chain-of-Thought." arXiv preprint arXiv:2411.19488.

[2] Li et al. (2025). "Imagine while Reasoning in Space: Multimodal Visualization-of-Thought." arXiv preprint arXiv:2501.07542.

[3] Chen et al. (2025). "MINT-CoT: Enabling Interleaved Visual Tokens in Mathematical Chain-of-Thought Reasoning." arXiv preprint arXiv:2506.05331.

[4] Zhao et al. (2025). "CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models." arXiv preprint arXiv:2503.22020.

[5] Chen et al. (2025). "Moto: Latent Motion Token as the Bridging Language for Learning Robot Manipulation from Videos." arXiv preprint arXiv:2412.04445.

### Questions
Please see weaknesses.

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
4

### Summary
This paper introduces a modal-mixed chain-of-thought reasoning approach that enables VLMs to interleave textual tokens with compact visual "sketches" represented as latent embeddings. The method addresses the limitation that language-only CoT struggles on vision-intensive problems where intermediate states are inherently visual. The approach consists of two key architectural components: (1) using the VLM's own vision encoder to produce latent embeddings for intermediate visual states, ensuring semantic alignment, and (2) a diffusion-based decoder that generates these latent embeddings conditioned on the VLM's hidden states. Training proceeds in two stages: supervised fine-tuning on modal-interleaved traces with joint next-token and latent-reconstruction objectives, followed by reinforcement learning to learn when to switch modalities. Evaluated on 11 vision-intensive tasks spanning perception and reasoning, the method demonstrates improvements over language-only CoT and competitive baselines.

### Strengths
1. The motivation is compelling and well-articulated. The observation that humans use mental imagery and visuospatial reasoning for vision-intensive tasks provides strong intuition for why purely language-based CoT may be insufficient. The analogy to a "visuospatial sketchpad" is effective and the problem is clearly motivated.

2. The architectural design is clear and well-motivated. Using the VLM's own vision encoder to produce target latent embeddings is a clever way to learn visual representations for the task.

3. The two-stage training strategy is well-designed. The SFT stage teaches the basic modal-interleaved format while the RL stage allows the model to discover effective switching points between modalities. The joint objective combining next-token prediction with latent reconstruction during SFT is natural and principled.

4. The paper is generally well-written and clearly structured, making the method easy to follow.

### Weaknesses
1. The most critical weakness is the complete absence of comparisons with existing visual tool-use and visual generation-based reasoning methods, which is a major gap given that this is the paper's core contribution area. There is extensive recent work on using visual information as intermediate reasoning steps, including: Visual Sketchpad (Hu et al., 2024), CoT-VLA (Zhao et al., 2025), Thinking with Generated Images (Chern et. al.), and Imagination and Contemplation (Christoduolopoulos et. al.) The broader landscape of visual tool-use methods documented in the recent survey "Thinking with Images for Multimodal Reasoning" (arxiv 2506.23918), which categorizes tool-augmented visual reasoning, visual programming (ViperGPT, VisProg), and other paradigms. Without comparisons to these similar approaches, readers cannot assess whether latent embeddings provide advantages over: (i) explicit visual tool use with actual image manipulation, (ii) code-based visual generation, or (iii) other forms of visual intermediate reasoning where an image generation model is used as an accessible LLM agent. The paper claims latent embeddings are better because they're "lightweight" but provides no empirical evidence for these claims against actual tool-use baselines.

2. Related to the above, the paper's positioning relative to existing work is incomplete. The related work section (Section 2) mentions "Recent works have proposed to use interleaved CoT for solving vision-intensive tasks (Gao et al., 2025)" and cites Wang et al. 2025a for tool use, but then dismisses these with "such a way cannot handle open questions that might require special operations on the images (e.g., drawing irregular masks)." This dismissal is unfounded—Visual Sketchpad and similar methods can handle diverse visual operations including masks, bounding boxes, and arbitrary drawings. The paper also mentions Mirage (Yang et al., 2025) only briefly in the latent reasoning paragraph, despite it being the most similar prior work. The relationship needs thorough clarification and at least some empirical comparisons to a subset of related works.

3. The improvements over baselines are modest and inconsistent across tasks. Table 1 shows the method achieves 26.7 (SFT) and 25.7 (RL) average scores compared to 24.3-24.7 for baselines: gains of around 2 points. Moreover, RL actually hurts performance on several tasks (Inductive reasoning drops from 28.0 to 21.5, Spatial reasoning from 31.6 to 25.3). The paper attributes this to "lengthy output patterns not well captured by RL data" but this suggests brittleness. Given these modest and inconsistent gains, stronger baselines using visual reasoning methods are essential to validate the approach.

4. The computational costs are substantial but inadequately analyzed. Table 5 shows inference latency increases from 18.31s to 31.79s (73% increase) and tokens per sample from 219 to 370. The paper dismisses this as "acceptable" without cost-benefit analysis. How does this compare to tool-use methods? To other multimodal CoT methods?

5. On this note, one dimension of comparison left out are multimodal CoT methods (such as Compositional CoT by Mitra et. al. and many others that can be found https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models) and latent CoT approaches like COCONUT (Hao et. al.). Though less comparable, it would be great to see a comparison to these CoT methods.

### Questions
My feedback is constructive and includes both critique and approaches for resolving that critique. Please refer to the Weaknesses section.

### Soundness
2

### Presentation
3

### Contribution
2
