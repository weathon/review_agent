# Misaligned Roles, Misplaced Images: Structural Input Perturbations Expose Multimodal Alignment Blind Spots

- Decision: Accept (Poster)
- Scores: 2, 8, 4, 6

## Abstract
Multimodal Language Models (MMLMs) typically undergo post-training alignment to prevent harmful content generation. However, these alignment stages focus primarily on the *assistant* role, leaving the *user* role unaligned, and sticking to a fixed input prompt structure of special tokens, making the model vulnerable when inputs deviate from these expectations. We introduce Role-Modality Attacks (RMA), a novel class of adversarial attacks that exploit role confusion between the *user* and *assistant* and alter the position of the *image* token to elicit harmful outputs. Unlike existing attacks that modify query content, RMAs manipulate the input structure without altering the query itself. We systematically evaluate these attacks across multiple Vision Language Models (VLMs) on eight distinct settings, showing that they can be composed to create stronger adversarial prompts, as also evidenced by their increased projection in the negative refusal direction in the residual stream, a property observed in prior successful attacks. Finally, for mitigation, we propose an adversarial training approach that makes the model robust against input prompt perturbations. By training the model on a range of harmful and benign prompts all perturbed with different RMA settings, the model loses its sensitivity to Role Confusion and Modality Manipulation attacks and is trained to only pay attention to the query content in the input prompt structure, effectively reducing Attack Success Rate (ASR) while preserving the model’s general utility.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces Role-Modality Attacks (RMA) — a new class of adversarial attacks that exploit structural vulnerabilities in multimodal language models (MMLMs). Specifically, it manipulates the role tokens (swapping user/assistant) and modality token positions (moving the image token around) in standard chat templates. Experiments are conducted using 3 MLLMs (qwen2-vl, llava-1.5, and phi-3.5-vision) on two benchmarks, Advbench and HarmBench, showing obvious gains.

### Strengths
1. The paper is clearly rewritten and easy to follow.

### Weaknesses
1. **Should RMA be treated as Adversarial Attack:** I'm somewhat concerned about whether the proposed RMA is proper to be treated as 'adversarial attack'. The proposed RMA mainly involves reformatting the prompt template (e.g., swapping role tokens or moving the image token around). As LLMs are known to be sensitive to prompts (especially prompt templates), the resulting performance drop of applying RMA is quite expected. I think it may be more appropriate to treat it as the LLMs' sensitivity/robustness to prompts instead of actual "adversarial attack", since altering the prompt template will degrade model performance on all tasks, which is not specific to adversarial attack on jailbreaking.
2. **Practical Usefulness:** I'm also concerned about the practical usefulness of RMA. In reality users will not be able to modify the chat template when using APIs.
3. **Similar prior works & missing reference:** There is a previous workshop paper that also studies how the chat template can help jailbreak an aligned MLLM [1]. The objective in [1] seems to be pretty similar to this paper, yet there is no reference to [1]. 
4. **Experiment on more recent MLLMs:** I think it might be worthwhile to experiment RMA on newer MLLMs such as qwen2.5/qwen3, gemma 3, internvl2.5/3/3.5 etc.. These models are RL-finetuned, which may have different behaviors from non-RLed models tested in the paper. (e.g. in Table1, Qwen2-VL already demonstrates better robustness against RMA, compared to earlier models like LLava-1.5 and phi3.5-v). 

[1] ChatBug: A Common Vulnerability of Aligned LLMs Induced by Chat Templates. NeurIPS 2024 Workshop SafeGenAI

### Questions
Please see weakness.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper studies an often-overlooked vulnerability in MMLMs, their reliance on fixed token structures and aligned roles. The authors introduce Role-Modality Attacks (RMA), a novel class of structural perturbations that exploit the asymmetry in alignment and the sensitivity of MMLMs to the change in structure and ordering of text/image tokens within the input sequence.

### Strengths
* The work identifies a simple yet effective attack vector by focusing on the structural integrity of the prompt rather than just semantic or low-level input manipulation.

*The explicit focus on the "user role being unaligned" is a sharp, correct critique of current safety practices. It demonstrates that alignment strategies, which largely focus on constraining the output, can be easily bypassed by injecting unaligned behavior into the input.

*The approach was evaluated across different state-of-the-art models. Showing that is a common blind-spot in MMLMs.

*The authors show that finetuning on samples generated using their method clearly have a positive effect in reducing ASR.

### Weaknesses
*Given the simplicity of the approach a defense based on input token position preprocessing seems plausible and should be discussed or evaluated as a baseline defense. Could prompt preprocessing mitigate the proposed attacks?

*This kind of attack might not be a real concern for current commercial LLMs such as (Gemini/ChatGPT). It would be interesting to see the effect if any on this kind of models.

### Questions
*The paper's core finding is that changing the position of the tokens drastically alters model behavior. Do the authors have further insights or a hypothesis on the architectural root cause of this sensitivity?

*Could a simple and computationally cheaper defense, such as input prompt pre-processing or normalization, effectively mitigate these attacks, potentially serving as a more practical alternative to the proposed adversarial training?

*For a black-box LLM accessible only via a fixed API where the internal role/instruction tokens cannot be explicitly manipulated by the user, would RCA remain effective? Since it relies on modifying those structural tokens.

### Soundness
4

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
3

### Summary
This paper studies safety alignment vulnerabilities in VLMs. The core claim is: current post-training alignment mostly teaches the assistant role to refuse harmful requests under a fixed chat template, but (i) the user role is often left essentially unaligned, and (ii) models are brittle to even small structural perturbations in the input format (like where the image token appears).

### Strengths
1. The authors evaluate their approach on multiple benchmarks and conduct extensive ablation studies.

2. The experimental setup is described in a detailed and thorough manner.

### Weaknesses
1.  Limited practicality of the threat model. The paper's primary weakness lies in its threat model, which assumes the attacker has a level of control that is unavailable to most end-users in real-world deployments. 

2. Model coverage / scalability. A broader sweep across multiple model sizes / generations (e.g., Qwen3-VL families at 2B/4B/8B/30B  checkpoints) is needed to show whether the vulnerability and the defense both scale, rather than being specific to the tested models.

3. Defense may overfit the attack. The proposed fine-tuning trains on the same eight structural perturbations it defends against. It is unclear whether the model actually learned to refuse harmful requests in general, or just memorized those exact patterns. We don’t really see robustness-to-unseen-perturbations experiments, which matters for claims of principled safety.


4. Limited analysis of the visual signal. The paper notes that swapping in a harmless “flower” image yields similar ASR to using a “firearm” image, implying the actual visual content barely matters. This is interesting but under-analyzed. Does the model stop using vision once the image token is moved, effectively behaving like a text-only model?

### Questions
Please refer to the questions raised in the Weaknesses section above.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Multimodal large models are at risk of jailbreak Attacks. This paper focuses on the vulnerability of large models in the face of Role-Modality attacks. RMA induces the model to generate unsafe content by manipulating dialogue roles (such as swapping the roles of users and assistants) and inputs (such as adding harmful images). This paper verifies the effectiveness of RMA on multiple mainstream VLMS and demonstrates the combined enhanced effects on various classic jailbreak methods. To mitigate such attacks, this paper proposes an adversarial training method, which enhances the defense capability against RMA attacks.

### Strengths
1.This paper has a clear motivation. Firstly, it verifies the effectiveness of RMA attacks, and then further analyzes the internal mechanism of the success of RMA attacks. Based on this, it proposes an adversarial training method based on role and modal perturbation, which is logically complete.

2.This paper has a well-designed experiment. It systematically evaluates the effect of RMA on mainstream VLMS under eight experimental Settings in terms of roles and modalities, as well as the effect of mainstream VLMS after Adversarial Training. The experimental results are fully explained.

3.This article clearly demonstrates the effectiveness of adversarial training through model output samples. The trained model is more focused on the security of the content itself. Clear prompts are provided for character control, image examples for modal control, and images for adversarial training.

### Weaknesses
1.This paper shows that under the eight RMA experimental Settings of this paper, the trained adversarial models have been significantly improved. However, this adversarial training design method seems to be only applicable to RMA attack methods. Does it also have the generalization ability against other jailbreak attack methods? Furthermore, this paper presents the combined effect of RMA and other mainstream jailbreak attack methods through Table 11. Could the success rates of individual jailbreak attack experiments be provided for comparison (the comparison includes whether there is an improvement with RMA itself and in combination with RMA)?

2.The dataset used in the experiments of this paper, the tested models and the evaluation methods have limited breadth. Comparisons can be appropriately increased.

### Questions
1.Supplement the effect of adversarial training on other jailbreak attack methods, as well as the success rate of jailbreak itself and the success rate of combining with RMA.

2.Provide more experimental comparison results of data and models.

### Soundness
3

### Presentation
3

### Contribution
3
