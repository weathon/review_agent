# A Data Perspective on Enhanced Identity Preservation for Diffusion Personalization

- Decision: Reject
- Scores: 6, 6, 3, 3

## Abstract
Large text-to-image models have revolutionized the ability to generate imagery using natural language. However, particularly unique or personal visual concepts, such as your pet, an object in your house, etc., will not be captured by the original model. This has led to interest in how to inject new visual concepts, bound to a new text token, using as few as 4-6 examples. Despite significant progress, this task remains a formidable challenge, particularly in preserving the subject's identity. While most researchers attempt to to address this issue by modifying model architectures, our approach takes a data-centric perspective, advocating the modification of data rather than the model itself. We introduce a novel regularization dataset generation strategy on both the text and image level; demonstrating the importance of a rich and structured regularization dataset (automatically generated) to prevent losing text coherence and better identity preservation. The better quality is enabled by allowing up to 5x more fine-tuning iterations without overfitting and degeneration. The generated renditions of the desired subject preserve even fine details such as text and logos; all while maintaining the ability to generate diverse samples that follow the input text prompt. Since our method focuses on data augmentation, rather than adjusting the model architecture, it is complementary and can be combined with prior work. We show on established benchmarks that our data-centric approach forms the new state of the art in terms of image quality, with the best trade-off between identity preservation, diversity, and text alignment.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a new method to inject new visual concepts into the generation, using few images. The authors propose a novel regularization dataset generation strategy on both the text and image level. The formulated dataset can help to prevent losing text coherence and prompt better identity preservation. The results are established on benchmarks, demonstrating the effects of the proposed method.

### Strengths
The proposed data approach is effective on the chosen benchmarks.

### Weaknesses
1.	The formatted prompt generation is limited to several categories. For example, according to the supp., for live objects, the prompts are all obtained via the subject of animal. How about the prompts for human? I do not think this prompt generation strategy is general enough.

2.	Moreover, I think the prompts should be generated according to different input images, employing multi-modality models.

3.	I think the new objects to be inserted into the generation in this paper’s experiments are few. More cases are needed to analyze the effects of the proposed method.

4.	In Fig.4, why the performance of w/o format will lead to worse results as the increase of iteration number? Even without the use of formatted prompts, the generation results should be more fitted with the target object along with the training.

### Questions
1.	I wonder the performance of the proposed method if there are fewer input examples, like 1-3 examples.

2.	Can the prompts be generated online with the training? It will save a lot of time.

3.	Few examples can not reflect the true quality of the generation, is there any subjective evaluation?

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
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a data-driven approach to improve personalized generation. The paper first discovers that previous class regularization is ineffective in alleviating overfitting due to a lack of diversity. The paper proposes to first enhance the training prompts associated with the concept images by including more specific class names and background descriptions. Based on the training prompts, the regularization prompts are further enhanced by introducing structural components including shape, color, and texture, which are then amplified with more diverse backgrounds and styles.

### Strengths
(1) The paper attempts to improve personalized T2I generation from a data-centric perspective, which focuses on automatically generating a rich and informative regularization dataset. Despite there exist a few methods that improve T2I generation via better prompting [1, 2] (note that [1] was submitted to arXiv on 25 Oct 2023, hence not in the scope of this work), this paper is the first work to improve personalized generation by diversifying the regularization data. 

(2) This paper provides insights into the importance of the quality of the regularization dataset in order to prevent overfitting, and is complementary to existing works that attempt to improve architectures and training schemes. This may inspire future research on further improving diffusion personalization.

(3) The proposed method demonstrates a notable improvement in generating personalized images with higher fidelity and is capable of preventing overfitting especially when facilitating larger training iterations.


[1] Segalis, Eyal, et al. "A Picture is Worth a Thousand Words: Principled Recaptioning Improves Image Generation." arXiv preprint arXiv:2310.16656 (2023).
[2] Wang, Yunlong, Shuyuan Shen, and Brian Y. Lim. "RePrompt: Automatic Prompt Editing to Refine AI-Generative Art Towards Precise Expressions." Proceedings of the 2023 CHI Conference on Human Factors in Computing Systems. 2023.

### Weaknesses
(1) The baselines that are compared in this paper are textual inversion and DreamBooth, which are both pioneering works in diffusion personalization. However, there exist many more improved personalization methods, e.g. Custom Diffusion [3], that are also widely used. Experimenting based on more methods will further emphasize the generalizability and complementarity of the proposed method.

(2) On top of Custom Diffusion [3], it would be also interesting to see whether the data-driven approach can benefit multi-concept learning.

(3) The method requires generating a relatively large regularization dataset (containing ~2000 images), which inevitably leads to much longer training time.

Small TYPO:
1. TYPO in section 5, the first bullet in the first paragraph should be “1)” instead of “2)”

[3] Kumari, Nupur, et al. "Multi-concept customization of text-to-image diffusion." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.

### Questions
Please see the weaknesses above.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
It seems (please refer to Weaknesses for reasons why I use the word "seem") that the authors introduced an extension of DreamBooth, which can better preserves the details of the objects of interest. The authors proposed to achieve this goal by generating a larger set of regularization images. Specifically, they seem to be generated using prompts that are (1) generated by a large language model (LLM) or (2) generated following specific templates. Experiments are conducted on DreamBench and metrics show that the proposed method generates higher quality images than existing methods.

### Strengths
1. Extensive visual comparison with existing methods are presented in the paper. It is evident that, for the examples provided, the proposed method better preserves the details of the objects of interest

2. The use of English is satisfactory. 

3. Ablations are conducted to help readers understand several design choices made by the authors.

### Weaknesses
1. Poor presentation: I find this paper hard to follow. Several important aspects of the proposed method remain a mystery, e.g., how the 2000 "regularization images" are created, why the 2000 images are called regularization images (I am not able to see how they can regularize the model from "Each training batch contains one example from training set and one example from regularization set."). It seems to me that this paper tries to extend DreamBooth [Ruiz, 2023a] and the 2000 regularization images may be generated using Stable Diffusion and the 2000 samples may be used to compute the class-specific prior preservation loss to regularize the model. However, a reader needs to be very familar with DreamBooth in order to make these guesses and they are just guesses.

2. To me, this is a trival extension of DreamBooth. The authors proposed to generate more "regularization images" using prompts that are (1) generated by a large language model (LLM) or (2) generated following specific templates. It is hard for me to agree that this paper meets the bar for an ICLR paper.

3. Lack of human evaluation.

### Questions
1. How are the 2000 regularization images created? Are they generated by a pre-trained diffusion model? If so, why do you use the word "created"?

2. Why the 2000 images are called "regularization images"? How can they help regularize the model? If you follow DreamBooth [Ruiz, 2023a], please specifically mention this.

3. Would be great to see a comparison with DreamBooth + LoRA and Textual Inversion + LoRA.

4. How does the performance of the proposed method change when more number of samples are available, e.g., 20 samples? How does the performance of the purposed method compare with other methods when more number of samples are available?

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
In this paper, the authors propose to perform prior preservation in personalized text-to-image generation with a regularization set. The authors construct this set by using ancestral sampling with formatted prompts. They tested their newly proposed regularization set on Stable Diffusion based models and achieved improvement compared to baseline.

### Strengths
The authors show qualitative and quantitative improvements compared to baselines in their experiments. In their qualitative examples, we can also observe that the fine-grained details of the objects are preserved better than the baselines.

### Weaknesses
1. It is very difficult to convince myself that the novelty presented in this paper is significant enough to warrant an acceptance. The main contribution of this paper is to construct a regularization set using a predefined and handcrafted format for prompting and ChatGPT for picking the phrases to fit the format, and the details of the format are not very well justified.

2. Continuing from Weakness 1, it is unclear to me how the authors choose the format described in Section 3 “Generating Against Training Prompts” and “Amplifying Diversity with Structured Prompts” since there is no related literature or ablation study to justify the effectiveness of each component in the format.

3. The additional time required for generating the regularization set is more substantial (2000 images for this setting v.s. < 1000 images for the original DreamBooth setting).

4. There is no evaluation on the fidelity (e.g. FID score) of the generated image, and fidelity score is a standard metric for this task.

### Questions
Does the size of the regularization set affect the performance? (e.g. will a smaller regularization set also work? Can the authors provide more ablation studies on this?)

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor
