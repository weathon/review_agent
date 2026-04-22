# VLM-Guided Adaptive Negative Prompting for Creative Generation

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 8, 8, 2, 4

## Abstract
Creative generation is the synthesis of new, surprising, and valuable samples that reflect user intent yet cannot be envisioned in advance. This task aims to extend human imagination, enabling the discovery of visual concepts that exist in the unexplored spaces between familiar domains.
While text-to-image diffusion models excel at rendering photorealistic scenes that faithfully match user prompts, they still struggle to generate genuinely novel content. 
Existing approaches to enhance generative creativity either rely on interpolation of image features, which restricts exploration to predefined categories, or require time-intensive procedures such as embedding optimization or model fine-tuning.
We propose VLM-Guided Adaptive Negative-Prompting, a training-free, inference-time method that promotes creative image generation while preserving the validity of the generated object.
Our approach utilizes a vision-language model (VLM) that analyzes intermediate outputs of the generation process and adaptively steers it away from conventional visual concepts, encouraging the emergence of novel and surprising outputs.
We evaluate creativity through both novelty and validity, using statistical metrics in the CLIP embedding space. Through extensive experiments, we show consistent gains in creative novelty with negligible computational overhead. 
Moreover, unlike existing methods that primarily generate single objects, our approach extends to complex scenarios, such as generating coherent sets of creative objects and preserving creativity within elaborate compositional prompts. Our method integrates seamlessly into existing diffusion pipelines, offering a practical route to producing creative outputs that venture beyond the constraints of textual descriptions.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
In this work, the authors presents a new way to make AI image generation more creative without extra training or heavy computation. The method, called VLM-Guided Adaptive Negative-Prompting, uses a vision-language model to gently push the model away from familiar ideas and toward more surprising and novel results. Unlike older methods that need fine-tuning or stay stuck in known categories, here, the model works on the fly during image creation. The authors demonstrate the effectiveness of the method, showing that their method makes images that are both novel and valid, unlike the other methods keeping realism while adding originality.

### Strengths
The paper has the following strengths:

1) The images shown on the paper are visually stunning, easily the best I have seen in this topic.

2) The paper is extremely well-written, I really enjoyed reading it. In particular, it has one of the best intros I have ever read with perfect merging of the intro with the figures, a really case-study of how pictures complement the writing. All parts of the papers are really well-written, and all the pictures are nice and helpful.

3) The results, be them qualitative, quantitative, or user study are really good, outperforming the other methods they compare with. The ablations studies are also very nice, further improving the confidence in the paper.

4) Th

### Weaknesses
The paper can be improved on this part:

1) Limited novelty - Probably the only technical contribution of the paper is section 3.2, which is extremely thin. And even then, it is effectively a smart way of doing prompting.

Saying that, I would not penalize the paper for it. The results speak for themselves, so I would actually prefer a simple method compared to a complex one, given the same results (Occam's Razor in reviewing).

2) It would have been nice if the authors would have released the code already (to check the effectiveness of this work) but considering that a) that is not mandatory, b) they promised to release in the near future, I will not penalize them for it.

### Questions
No questions.

### Soundness
4

### Presentation
4

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
This paper introduces VLM-Guided Adaptive Negative Prompting, a training-free, inference-time method that leverages vision-language models to steer diffusion models away from familiar visual concepts during generation. By dynamically adding negative prompts for concepts identified by the VLM at intermediate denoising steps, the method promotes the creation of novel images.

### Strengths
1. The paper is well written and easy to follow.
2. The proposed method is conceptually simple and can be easily integrated into existing diffusion inference pipelines without training.
3. The method achieves strong qualitative results across diverse categories while remaining completely training-free.

### Weaknesses
1. The paper omits citations of several recent works in performing and understanding creative generation, such as [1], [2], and [3].
2. While Section 4.5 presents qualitative examples using complex prompts, there is no quantitative or systematic evaluation of controllability. This makes the evidence for controllable generation less conclusive.

[1] Procreate, don’t reproduce! propulsive energy diffusion for creative generation, ECCV 2024

[2] Does Generation Require Memorization? Creative Diffusion Models using Ambient Diffusion, ICML 2025

[3] An analytic theory of creativity in convolutional diffusion models, ICML 2025

### Questions
1. Have you explored designing heuristics to automatically select VLM queries based on generation prompts? Such an approach could make the system easier to use and closer in workflow to standard text-conditioned diffusion generation.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper presents a training-free method for enhancing creative generation in diffusion models. By leveraging a Vision-Language Model (VLM) to analyze intermediate denoising steps and dynamically accumulate negative prompts, the approach steers generation away from conventional patterns while maintaining categorical validity.

### Strengths
The paper focuses on an interesting topic in computational creativity: generating novel visual concepts beyond conventional patterns.

The technical approach is elegantly simple and training-free, leveraging VLM feedback for adaptive negative prompting without modifying pretrained models. This clarity enhances reproducibility and practical deployment.

The writing is exceptionally clear and well-structured, with logical flow from problem formulation to experiments.

### Weaknesses
The technical contribution is somewhat limited, as the method does not adequately address the VLM's inconsistent perception capabilities across different denoising timesteps. Prior research highlights that control word effectiveness varies with timesteps, yet this work overlooks such dynamics, potentially undermining the robustness of adaptive guidance.

While the approach yields intriguing outcomes, it heavily relies on the base model's generative power rather than introducing groundbreaking mechanisms. Similar creative effects might be achievable through carefully engineered prompts or LoRA adaptations, questioning the necessity of the proposed complex feedback loop.

Controllability remains a significant issue, as the generation process is highly stochastic. Results are unpredictable and quality assurance depends largely on "luck-based" sampling, which fails to guarantee consistency or align with specific user preferences, limiting practical utility.

The method introduces non-negligible computational overhead due to frequent VLM queries, despite optimizations. This could hinder real-time applications, especially with resource-intensive VLMs, affecting scalability.

Effectiveness is sensitive to VLM selection and question design, requiring manual tuning for different categories. This dependency on external components may reduce generalizability and increase implementation complexity.

### Questions
See weaknesses for details.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper focuses on the creative image generation task, an emerging research direction that explores the ability of image generation models to produce novel and previously unseen images beyond the training distribution. The authors propose VLM-Guided Adaptive Negative Prompting, a method that leverages a vision-language model (VLM) during training to adaptively refine negative prompts, guiding the generative model to diverge from known concept spaces and thereby produce more unexpected and creative visual outcomes.

### Strengths
1. The proposed method is simple and requires no additional training overhead.
2. The experimental results are visually appealing and demonstrate interesting creative effects.

### Weaknesses
1. The method lacks substantial novelty. Its core idea—encouraging the diffusion model to deviate from known concept spaces—was originally introduced by ConceptLab. The main contribution here lies in performing additional VLM-guided queries at each denoising step and using classifier-free guidance (CFG) to avoid categories identified by the VLM. Compared to ConceptLab, this constitutes only a minor incremental improvement. Moreover, querying the VLM at every denoising step could considerably increase inference time, even though the authors claim that it adds only about 13 seconds.
2. The proposed approach is less flexible than ConceptLab, which operates at the token level, allowing its generated creative concepts to be easily integrated with natural language for diverse styles and contexts. In contrast, the current method requires a separate process for each prompt, causing inference time to scale with the number of prompts and limiting adaptability.
3. Although the visual results are impressive, it is unclear whether the improvements stem from the proposed method itself or from the use of a stronger base model (SD3.5). When applied to other backbones such as Kindinsky or SD-XL, the method’s performance degrades noticeably (Figure 5).

### Questions
1. The proposed method performs VLM queries at every denoising step. I am curious whether a VLM can effectively recognize images that are still heavily corrupted by noise. It seems unnecessary to query the VLM at each step, since it primarily identifies common object categories and may produce unreliable or meaningless predictions in the early denoising stages. Why not predefine a set of common negative classes at the beginning of the process? In ConceptLab, repeated experiments tend to yield similar negative class sets—mostly consisting of frequent categories such as cat, dog, parrot, rat, and lizard—while rarer categories like fish or monkey seldom appear.

2. Could SD3.5 alone, guided only by human-written prompts, generate the same level of creative results shown in the paper? For example, could it produce a plausible image of an unseen fruit purely based on human imagination without relying on the proposed adaptive negative-prompting mechanism?

### Soundness
2

### Presentation
4

### Contribution
2
