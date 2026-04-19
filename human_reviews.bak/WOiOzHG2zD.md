# TextField3D: Towards Enhancing Open-Vocabulary 3D Generation with Noisy Text Fields

- Decision: Accept (poster)
- Scores: 6, 6, 8

## Abstract
Recent works learn 3D representation explicitly under text-3D guidance. However, limited text-3D data restricts the vocabulary scale and text control of generations. Generators may easily fall into a stereotype concept for certain text prompts, thus losing open-vocabulary generation ability. To tackle this issue, we introduce a conditional 3D generative model, namely TextField3D. Specifically, rather than using the text prompts as input directly, we suggest to inject dynamic noise into the latent space of given text prompts, i.e., Noisy Text Fields (NTFs). In this way, limited 3D data can be mapped to the appropriate range of textual latent space that is expanded by NTFs. To this end, an NTFGen module is proposed to model general text latent code in noisy fields. Meanwhile, an NTFBind module is proposed to align view-invariant image latent code to noisy fields, further supporting image-conditional 3D generation. To guide the conditional generation in both geometry and texture, multi-modal discrimination is constructed with a text-3D discriminator and a text-2.5D discriminator. Compared to previous methods, TextField3D includes three merits: 1) large vocabulary, 2) text consistency, and 3) low latency. Extensive experiments demonstrate that our method achieves a potential open-vocabulary 3D generation capability.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper aims to achieve open-vocabulary 3D generation by attempting to tackle the challenges of limited 3D data and therefore sparse text annotations, which would limit the model's ability to generalize to open-vocabulary queries. One way to approach the challenge is to add noises to text features to avoid models from overfitting. But it is not trivial to know how much noise is appropriate. Therefore, the authors propose to learn a noisy text fields, which learns the standard deviation of the Gaussian noise that can be added to each text feature. Other than the text-to-image task, this paper also proposes to learn a view-invariant image representation to facilitate better image-to-3D generation. As for the generator part, the authors use a 3D-aware GAN framework, with a GET3D-like generator and a text-3D and text-2.5D discriminator. Qualitative and quantitative experiments are reported to showcase the performance of the proposed method. For the quantitative part, TextFireld3D surpasses baseline methods, including Point-E, Shap-E and GET3D. Visual results also show reasonable generation quality both from text and image conditions.

### Strengths
1. Overall, I like that the authors choose to use 3D-aware GAN instead of diffusion models to tackle the text-to-3D generation problem. GAN has clear advantages over diffusion model in terms of generation speed and smooth interpolation. But for the open-vocabulary text-guided generation task, GAN falls behind diffusion models. Therefore, it is exciting to see a GAN model surpasses diffusion baselines, e.g., Shap-E and Point-E. This paper can service as a strong baseline for the community of text-to-3D generation.
2. The idea of learning the noisy text field is interesting. Experiments also suggest the effectiveness of this method.

### Weaknesses
My main concerns are around the experiments of this paper.

1. I would expect to see more baseline methods being compared with. For example, in the ablation study, authors mentioned methods like SDFusion and TAPS3D.
2. I would suggest also testing with the DreamFusion testing list, which contains over 400 prompts.
3. For the image-to-3D generation, the evaluation is limited to showcasing two visual results, which is far from enough. I would suggest adding more visual results, especially showing multiple viewing angles of the generated object. Some quantitative evaluation is also expected.
4. For the visual results, I would like to see more prompts from the DreamField or DreamFusion test sets, which include more challenging and complex examples like concept-mixing. It would help us better evaluate the open-vocabulary ability of the proposed method. Right now, the showcasing prompts only contain one simple concept and can be easily found in Objaverse training set. I understand that concept mixing is super hard given such limited training data. But at least, it is nice to see some failure cases and analysis on the failure modes.

If authors can provide more evaluations during the rebuttal. I would consider raising the score.

### Questions
See weaknesses.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper works on text-to-3d shape generation. The paper observes that the current text-to-shape generation methods are usually V-L Optimized or 3D supervised. These methods either suffer from the problem of long-optimization or restricted open-vocabulary capability. To resolve these problems, the paper proposes TextField3D, which generates 3D shapes in real-time, taking both open-vocabulary text prompts and image as input. Specifically, the method adopts a NFGen model, that maps a single text prompt into a noisy text field. This noisy text field enables more open-vocabulary text prompts than single category name or template-generated text prompts. It also proposes an NTFBind model, which maps images from any view into a view-invariant image feature in the noisy text field. The noisy latent feature from the noisy text field is then fed into a conditional generative model. The method also adopts a 2.5D and a 3D supervision that ensures the generated shape has high-quality texture and geometry. Experiment shows the proposed method generates shapes of higher quality than some V-L optimized and 3D supervised methods. It also shows the effectiveness of the NTFGen and NTFBind modules in design choice.

### Strengths
+ The paper is clearly written. The motivation is clear to me - design a real-time supervised method whose performance is competitive with VL-optimized methods. 
+ The NTFGen model shows that it is able, to some extent, to increase the expressiveness of a single text prompt. 
+ The NTFBind model shows that it is better at aligning image features from any views to text features. 
+ The generated shapes visually seem to have better quality and texture than the compared methods thanks to the supervision from 2.5D and 3D.

### Weaknesses
This paper fails to convince me of the effectiveness of its major component in the following aspects:
+ Shape Diversity is limited by the training dataset. Quoting from the original paper - "With limited 3D data, can we train a real-time generator that is equivalent to V-L optimized methods", I think this is impossible considering the method is training with a relatively small scale 3D shape dataset compared with VLMs. VL-optimized methods clearly can generate synthesized imagined shapes, like a chair with the shape of an avocado, but given the qualitative examples that the authors provide, this method seems not able to generate imagined shapes. Even though I don't think this is a major drawback of the proposed method, I think this claim is faulty. 
+ Open-vocabulary capability - The open-vocabulary capability is the major claim of this paper. However, in the qualitative experiment section, the paper only provides very simple prompts, like category names, adjective nouns, or a phrase with two nouns. I think these simple phrases are not complicated enough to prove the open-vocabulary capability of the method, especially considering the method is training with complicated enough captions generated by BLIP-2 or MiniGPT4. I hope the authors can provide more results from complicated text prompts as in the captions generated.
+ View-invariant experiments. The paper claims that the NTFBind model produces a view-invariant feature, but the experiment provided is not strong enough to prove the point.  The experiment uses image features across views and image features across ShapeNet categories to prove that image features across views are more reassembled than features across categories. I think this setting is not strong and persuasive enough. A better experiment setup would be comparing features across views with features across instances in the same category. See CLIP-NeRF[1] Figure 2 for more details. 
+ Comparison. Though the paper compared with VL optimized methods and 3D supervised methods, it didn't directly compare with the TAPS3D, which has the same training setting as the proposed method. Both of them methods use an image captioning model to augment text prompts and work on a 3D-generated model. Though the paper provides an ablation study that replaces the major component to the TAPS3D component, I wonder if it will outperform the TAPS3D original method. 
[1] CLIP-NeRF: Text-and-Image Driven Manipulation of Neural Radiance Fields

### Questions
+ Some minor questions:
1. For noisy text latent code, the paper uses a learned noise, which is referred to as "dynamic noise". I wonder how the method performs with a non-dynamic noise, which could set \sigma to a static number. 
2. The noting of L_{img} and L_{txt} looks like two different types of loss, but they are actually the same type of loss. Changing the namings to make them more consistent would be better for reading.
3. Textured Mesh Generator. Are they training from scratch, or training from the pre-trained GET3D?
4. Sillhoutte loss. The silhouette loss in equations (4) and (5) is not introduced clearly. For a first-time reader, it might be a little confusing.

### Soundness
2 fair

### Presentation
4 excellent

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
In this paper, the authors present TextField3D, which is a conditional 3D generative model that enhances text-based 3D data generation by injecting dynamic noise into the latent space of text prompts, creating Noisy Text Fields (NTFs). This technique allows for the mapping of limited 3D data to a broader range of textual latent space, enhanced by the NTFs. The authors propose two modules to facilitate this process: NTFGen, which models the general text latent code in noisy fields, and NTFBind, which aligns view-invariant image latent code to the noisy fields, aiding image-conditional 3D generation. The model is guided through the conditional generation process in terms of both geometry and texture by a multi-modal discrimination setup, consisting of a text-3D discriminator and a text-2.5D discriminator. TextField3D is highlighted for its large vocabulary, text consistency, and low latency, positioning it as an advancement over previous methods in the field.

### Strengths
- The paper successfully expands the GAN-based GET3D framework to handle extensive vocabulary datasets, achieving results on par with or surpassing those of diffusion-based models like point-e and ShapE. This marks a significant step forward for large-vocabulary feed-forward generative models.
- Since text and 3D are not one-to-one mapping, , the introduction of Noisy Text Fields and their corresponding modules seem to be reasonable to me.
- The results, both qualitative and quantitative, look diverse and of a relatively good quality compared to other feed-forward models.
- The ablation studies are clear and comprehensive, providing detailed insights into the impacts of various modules, discriminators, and choices in noise range.

### Weaknesses
- My main concern is the potential overfitting problem. In Figure 12 and 13, certain prompts (e.g., "A beer can", "A wooden crate", and "A cardboard box with graffiti") generate unusually detailed outputs, showing a much higher level of details than others. Based on my experience,  the training dataset likely contains very similar examples. I am interested in understanding how the authors have addressed and evaluated the risk of overfitting associated with their method.

### Questions
1. I'm wondering how is the FID score calculated under the text-conditioned setting?
2. Can you provide some more examples of the 9-shot experiments?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
