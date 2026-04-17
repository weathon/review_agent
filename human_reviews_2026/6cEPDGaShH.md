# Invisible Safety Threat: Malicious Finetuning for LLM via Steganography

- Decision: Accept (Oral)
- Scores: 6, 6, 6, 6

## Abstract
Understanding and addressing potential safety alignment risks in large language models (LLMs) is critical for ensuring their safe and trustworthy deployment. In this paper, we highlight an insidious safety threat: a compromised LLM can maintain a facade of proper safety alignment while covertly generating harmful content. To achieve this, we finetune the model to understand and apply a steganographic technique. At inference time, we input a prompt that contains a steganographically embedded malicious target question along with a plaintext cover question. The model, in turn, produces a target response similarly embedded within a benign-looking cover response. In this process, human observers only see the model being prompted with a cover question and generating a corresponding cover response, while the malicious content is hidden from view. We demonstrate this invisible safety threat on GPT-4.1 despite the OpenAI finetuning API’s safeguards. The finetuned model produces steganographic malicious outputs in response to hidden malicious prompts, while the user interface displays only a fully benign cover interaction. We also replicate the attack on three open-source models, Llama-3.3-70B-Instruct, Phi-4, and Mistral-Small-24B-Base-2501, confirming the generality of our method. We quantitatively evaluate our method on the AdvBench dataset, using Llama-Guard-3-8B for content safety classification. Across all four models, all stegotexts containing malicious content are incorrectly classified as safe.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
### **Summary**

This paper introduces a novel security vulnerability in large language models (LLMs) by leveraging steganographic techniques to conduct covert malicious fine-tuning. The authors propose a method where malicious content is embedded within seemingly benign prompts, using invisible characters, making it difficult for both human reviewers and safety classifiers to detect. The study evaluates the attack on multiple models, including both open-source models and closed-source models (like GPT-4.1), demonstrating its generalizability.

### Strengths
### **Strengths**


1. The use of steganography for malicious fine-tuning in LLMs is a very interesting and timely research direction. This approach adds a new layer of complexity to understanding how LLMs can be misused, which is crucial as LLMs are deployed in sensitive applications.


2. The authors have tested the proposed attack on both open-source models and closed-source models such as GPT-4.1. This demonstrates the versatility and robustness of their method across different types of models.

### Weaknesses
### **Weaknesses**

1. The use of steganographic techniques for covert malicious fine-tuning is not entirely new. For example, the paper [1] has already explored similar techniques. While this paper does contribute by demonstrating a more subtle use of steganography, this aspect was not entirely surprising and might not be as groundbreaking as suggested.

2. A potential concern arises when attacking closed-source models. If these invisible characters do not appear on the user interface during interaction, how can the malicious content be decoded into plaintext? This needs further clarification, particularly in how the attack would manifest in a practical setting.

3. I am familiar with both steganography and harmful fine-tuning. From my perspective, this paper makes a valuable contribution by using steganography to attack LLMs, effectively bridging these two fields. Therefore, I am inclined to support this paper.
I have some questions and would appreciate some insights from you: The paper tests two open-source models such as Phi, but have you tested other models, such as Gemma3? Is the ability of a model to learn steganography related to its capacity? Among current open-source models, which ones can learn steganography, and which ones cannot? How large does the model need to be for this attack to be effective? For example, does it require at least a 14B parameter model?


4. While the paper does a good job of presenting the attack, it could be strengthened by discussing related works more broadly. For instance, the paper could reference [2] that jailbreak LLMs via steganography and advanced steganographic methods like ALiSa [3]. Situating the work within these discussions could provide a deeper context and help frame its contributions more clearly within the existing literature.


### **Reference**


[1] Covert malicious fine-tuning: Challenges in safeguarding LLM adaptation

[2] When Safety Detectors Aren't Enough: A Stealthy and Effective Jailbreak Attack on LLMs via Steganographic Techniques

[3] Acrostic Linguistic Steganography based on BERT and Gibbs sampling

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

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
The paper presents a malicious fine‑tuning method that teaches LLMs to use zero‑width Unicode steganography so that a compromised model can show a benign, human‑readable “cover” exchange while covertly receiving and generating an encoded prompt and harmful answer. The method first converts text using a steganographic base‑4 mapping over five zero‑width characters and then a two‑track fine‑tuning recipe. The authors claim successful attacks on GPT‑4.1 (via OpenAI’s fine‑tuning API) and on two open‑source models (Phi‑4 and Mistral‑Small‑24B‑Base‑2501), with safety checks (Llama‑Guard‑3‑8B) flagging all cover exchanges as safe before decoding while >90% of the decoded pairs are unsafe. Utility degradation is reported as modest.

### Strengths
* Methodological novelty and practicality. The two‑track design (auxiliary base‑4 + stego, 4 subtasks each) is novel, simple, and effective.
* Closed‑ and open‑source coverage. Demonstrations on GPT‑4.1, Phi‑4, and Mistral‑24B‑Base strengthen generality.
* Good Results. The method achieves high unsafe rates, with minimum degradation in quality of model. The responses are also "hidden" from the end-user, making detection difficult.

### Weaknesses
* Reliance on a single safety classifier. The paper uses Llama‑Guard‑3‑8B for detecting harmful responses. Recent works ([1]) have shown this classifier to be unreliable. A human study would support this, or alternatively, using LLM-as-a-judge has been shown to align better.
* Single benchmark for Evaluation. The authors use AdvBench for checking the unsafe rate of their finetuned models. Using additional benchmarks (e.g., [2]) would strengthen the claims.
* Lack of baselines. While authors have compared how "covert" their method is against one baseline, comparing the unsafe rates against multiple baselines would significantly strengthen the results. (e.g., [3,4,5,6])

[1] Kumar, Priyanshu, et al. "Polyguard: A multilingual safety moderation tool for 17 languages." arXiv preprint arXiv:2504.04377 (2025).
[2] Chao, Patrick, et al. "Jailbreakbench: An open robustness benchmark for jailbreaking large language models." Advances in Neural Information Processing Systems 37 (2024): 55005-55029.
[3] Liu, Xiaogeng, et al. "Autodan-turbo: A lifelong agent for strategy self-exploration to jailbreak llms." arXiv preprint arXiv:2410.05295 (2024).
[4] Handa, Divij, et al. "When" Competency" in Reasoning Opens the Door to Vulnerability: Jailbreaking LLMs via Novel Ciphers." NeurIPS 2025 Workshop: Reliable ML from Unreliable Data.
[5] Anil, Cem, et al. "Many-shot jailbreaking." Advances in Neural Information Processing Systems 37 (2024): 129696-129742.
[6] Gohil, Vasudev. "JBFuzz: Jailbreaking LLMs Efficiently and Effectively Using Fuzzing." arXiv preprint arXiv:2503.08990 (2025).

### Questions
See weaknesses above

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors introduce a jailbreaking method for LLMs that teaches LLMs to receive and send nefarious messages in the form of invisible characters. This is a novel and compelling risk of models. The authors demonstrate that they can perform this attack on the Open AI fine-tuning API as well as on open source models. The authors show that this does not lead to a large drop in various model capabilities.

### Strengths
- The threat model and attack proposed seem quite reasonable.
- The proposed approach is well communicated.
- The fact that they were able to successfully demonstrate the capability on a real-world API model and open models is compelling.
- The evaluation on AdvBench and use of LlamaGuard seem reasonable.

### Weaknesses
- Could filtering out unicode characters be a simple way to defend against this attack method? Are there reasons it would be more challenging than that
- Text size in figures 1, 2, 3 are very small, significantly lowering the readability of the work.
- The paper primarily focuses on the attack and (as far as I can tell) does not highlight potential mitigation strategies. A discussion of potential mitigation strategies could be helpful.

### Questions
- It would be interesting to measure the quality of the stenographically encoded generated text. Is the models capacity to perform tasks like question answering and reasoning the same in the encoded representation? Is it comparable to the capabilities of the vanilla model?
- Could a simple unicode filtering approach work as a mitigation strategy/defense?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper shows that the “invisible” zero space Unicode characters can be used to teach LLMs certain hidden semantic meanings. In doing so, attackers can directly finetune models to generate harmful info in their outputs without easy human detection.

### Strengths
This paper has a nice contribution showing that the sort of “invisible” Unicode characters can be taught to represent certain meanings in LLMs. This allows attackers to directly finetune models to hide harmful info in their inputs and outputs, without easy human detection.

This flavor of attacks seem relatively easy to defend with some heuristics on the defender’s side. But that isn’t necessarily a bad thing! It further underscores that frontier models / big companies need to think harder about how to handle these types of tokens, in terms of how they design their tokenizer, UIs, etc.

### Weaknesses
The project uses a lot of the same machinery as Halawi in terms of process supervision for teaching models encrypted malicious text, goals of hiding malicious finetuning, etc.

I think it is a significant enough lift over that work that this isnt necessarily a blocker to publish.

### Questions
* Im very surprised how little drop in benchmarks this method causes. Could you include some qualitative examples of what the model behavior is when running some common benchmarks / chat prompts?

### Soundness
3

### Presentation
3

### Contribution
3
