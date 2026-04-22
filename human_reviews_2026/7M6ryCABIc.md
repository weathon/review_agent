# PixelVLA: Advancing Pixel-level Understanding in Vision-Language-Action Model

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 2

## Abstract
Vision-Language-Action models (VLAs) are emerging as powerful tools for learning generalizable visuomotor control policies. However, current VLAs are mostly trained on large-scale image–text–action data and remain limited in two key ways: (i) they struggle with pixel-level scene understanding, and (ii) they rely heavily on textual prompts, which reduces their flexibility in real-world settings. 
To address these challenges, we introduce PixelVLA, the first VLA model designed to support both pixel-level reasoning and multimodal prompting with text and visual inputs. Our approach is built on a new visuomotor instruction tuning framework that integrates a multiscale pixel-aware encoder with a visual prompting encoder. To train PixelVLA effectively, we further propose a two-stage automated annotation pipeline that generates Pixel-160K, a large-scale dataset with pixel-level annotations derived from existing robot data.
Experiments on three standard VLA benchmarks and two VLA model variants show that PixelVLA improves manipulation success rates by $10.1\%\sim28.7\%$ over OpenVLA, while requiring only $1.5\%$ of its pretraining cost.  These results demonstrate that PixelVLA can be integrated into existing VLAs to enable more accurate, efficient, and versatile robot control in complex environments. The dataset and code will be released as open source.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors introduce a pixel encoder to condition VLAs on visual prompts, construct a suitable automated data annotation pipeline, and instruction tune their modified VLA to gain pixel level awareness. Their proposed visual prompt encoder conditions on multi-scale features of the robot visual observation and its resulting tokens are consumed by an LLM along with regular visual and textual tokens. The LLM outputs are mapped to robot actions with a suitable action decoder. 
They integrate their proposed PixelVLA into two strong VLAs, OpenVLA and Pi0. Extensive experiments highlight performance improvements from their framework.

### Strengths
1) The authors conduct extensive experiments to highlight improvements over SOTA baselines
2) Simple idea of incorporating pixel prompts 
3) Clever multi-stage training setup to benefit from existing VLA's pretrained weights
4) Thorough ablations

### Weaknesses
**1) Architecture is unclear**
  - In Figure 2, part (a) "Multiscale Pixel-aware Encoder" also shows another Visual Prompting Decoder which is unclear. This figure needs to be explained better and updated suitably. 
  - "Visual Prompting Decoder" is not explained clearly. What is its exact input structure? How is does it "preserve the spatial positional information" of visual prompts? These need to be explained. 
  - How exactly are the visual prompts represented? Fig 2 shows them both overlaid on image and input separately. The separate input needs to be described more clearly. 

**2) Unclear Data Generation**
  - "detect the gripper to generate N_e region proposals" - how are region proposals generated from gripper location?
  - L262 "we employ a LLM" - which one? How is it prompted? 
  - "we filter out approximately 19.2% failed samples" - how are they filtered? Manually by humans? 

**3) Unclear Training Details**
  - L307 "uniformly discretize each continuous action representation into 256 bins" - why? PixelVLA uses a continuous action decoder? The reasoning behind this is highly unclear. 
 
**4) Related Work**
  - Consider discussing related VLM work Ferret (https://arxiv.org/pdf/2310.07704) and FerretV2 that performs similar visual prompt encoding to improve VLA performance.

### Questions
See weaknesses. The paper is interesting and contains strong experimental evidence. However, several missing key details weakens it in current form.

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
The authors present PixelVLA, a VLA model that introduces diverse multimodal prompts to the VLA pipeline and whose architecture is designed and trained specifically for these prompts. The VLA itself consists of a vision encoder to process image observations, visual prompting encoder (from SAM) and multi-scale visual encoder to process multimodal prompts (points, boxes, lines etc), and LLM backbone and a continuous action decoder. To generate visuomotor instruction tuning data for training PixelVLA, the authors use an LLM to extract information about target objects, and use an object detector and segmentation model to extract multimodal prompts, all over the Fractal and Bridge datasets, and contribute the resulting dataset. The experiments are done over SimplerEnv and Libero, across multiple VLA backbones, to show the benefits of PixeVLA.

### Strengths
- Meaningful technical contribution in Pixel-160k dataset and in constructing a VLA that takes advantage of multimodal prompts.
- Experiments test two SOTA VLA architectures, showing that PixelVLA can be built on top of multiple types of VLAs.

### Weaknesses
- Unclear whether method is feasible to transfer to novel environments, due to lack of real world experiments.
- Analysis of results are lacking and leave out failure cases; for example, why does PixelVLA perform so well on Libero Long but struggle on the Object/Goal splits (Table 3)? Why would the pixel-level understanding training damage performance on the open/close drawer task (Table 4)?

### Questions
l. 218 Where does the pixel-aware mask input come from?
l. 237-239 How does the NTP loss work in this case? Is there causal masking?
l. 252 You're only using the first gripper-close state – might that negatively affect training due to some domain shift between first gripper-close state and subsequent such states?
Section 5.2: Were the $pi_{0}$ and OpenVLA baselines also trained in their typical fashion on the Fractal and Bridge datasets? This would ensure that the improvement is due to the PixelVLA method and instruction tuning approach rather than just seeing more data.

### Soundness
3

### Presentation
3

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
This paper proposes PixelVLA, a VLA that benefits from pixel-level grounding information. Based on a typical VLA model with a vision encoder, text tokenizer, and LLM, the authors introduced a pixel encoder that is designed to handle pixel-level understanding information. Meanwhile, to finetune such an encoder, the authors proposed a two-stage automated annotation pipeline and created a pixel-annotated visuomotor instruction tuning dataset, Pixel-160k. The experiments on SimplerEnv and LIBERO show that the proposed method achieves better performance than other baselines.

### Strengths
The paper is clearly written and well-structured, making it easy to follow. The experiments conducted on SimplerEnv and LIBERO are appropriate and demonstrate the effectiveness of the proposed approach. While introducing an additional pixel-level encoder could intuitively downgrade the pretrained VLM, the authors successfully solve this issue by curating a large 160K dataset and applying LoRA fine-tuning.

### Weaknesses
The introduction of pixel-level annotations can be viewed as a relatively straightforward extension of prior work on visual prompting and image-level feature adaptation (e.g., TraceVLA, LLaRA, and related approaches). As a result, the paper’s novelty is somewhat limited. Nonetheless, the work offers useful insights and has potential value for the research community, particularly as a good practice in bridging pixel-level understanding with pretrained VLMs for VLAs. 

Meanwhile, the authors are highly encouraged to deploy the proposed method in the real world and verify the claim.

Therefore, I would recommend a weak acceptance.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes PixelVLA, which besides normal language and , visual encoder, PixelVLA add a modal that enable pixelwise grounding, then claim this enable the model learn more fine-grained represenations that further benefit the understanding and action prediction.

### Strengths
1. The paper is well organized and easy to follow.
2. According to the results shown in Table 2, Table 3 and Table 4, it seems, add more pixel level prompting and intergrate this as a new modal can help increase the action control accuracy.

### Weaknesses
1. I am confused about the design of the whole architecture. It seems, the author proposes a two-state automated pipeline to get the pixel level segmentation, how these segmentations are used? it is not very clear that which part these segmentation masks are used for in the model.

2. If the segmentation is used as a input to learn the pixel-aware embedding, i am not sure the final optimization of loss for these visual encoder is action accuracy? seems not very relevent. What is the motivation of input these mask to learn the pixel aware embedding.

3. The author gives some results and try to say pixel vla is better than OpenVLA or Pi0, but that is not fair since Pixel VLA needs extra prompting from user, these extra prompt actually gives more direct spatial information of the user's goal. The intuition is the using of these informaiton will give more shortcut to the backbone learning and make it less capable to do reasoning and understand the language.

### Questions
Besides the main conern in weakness part. I still have questions:
1. I might need the user better claim the motivation of add the segmentation mask or visual prompting as an input, i feel these will harm the model's potential ability to learn to locate the objects and understanding the scene itself.

### Soundness
3

### Presentation
3

### Contribution
2
