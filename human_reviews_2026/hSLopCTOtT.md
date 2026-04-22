# Exploiting Reasoning Patterns in Language Models for Indirect Targeted Poisoning

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4, 4

## Abstract
Chain-of-Thought (CoT) prompting has emerged as a powerful technique for enhancing large language models' reasoning capabilities by generating intermediate reasoning steps for complex tasks. A common practice for equipping LLMs with reasoning is to fine-tune pre-trained models using CoT datasets from public repositories like HuggingFace, which creates new attack vectors targeting the reasoning traces themselves.
 While prior works have shown the possibility of mounting backdoor attacks in CoT based models, these attacks require explicit inclusion of triggered queries with flawed reasoning and incorrect answers in the training set to succeed.
 Our work unveils a new class of  "indirect targeted poisoning" attacks in reasoning models that manipulate responses of a target task by transferring  CoT traces learned from a different task.  Our "thought-transfer" attack can influence the LLM output on a target task by manipulating only the training samples' CoT traces—while leaving the queries and answers unchanged, resulting in a form of undetectable ``clean label'' poisoning.  Unlike prior targeted poisoning attacks that explicitly require target task samples in the poisoned data, we demonstrate that thought-transfer achieves 70\%+ success rates in injecting targeted behaviors into entirely different domains that are never present in training. Remarkably, training on poisoned reasoning data also improves the model's performance by 10-15\% on multiple benchmarks, providing incentives for a user to use our poisoned reasoning dataset. Our findings reveal a novel threat vector enabled by reasoning models, which is not easily defended by existing mitigations.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper investigates the vulnerability of reasoning-augmented large language models (LLMs) to *indirect targeted data poisoning*.  
The central claim is that when models are trained to follow chain-of-thought (CoT) reasoning, the reasoning traces themselves can be exploited as an attack surface. Instead of inserting explicit triggers or mislabeled data, the attacker manipulates reasoning patterns within clean-labeled training samples so that, at inference time, similar reasoning styles can trigger malicious or biased behaviors on unseen tasks.

### Strengths
-  The paper touches on a very relevant and emerging issue at the intersection of *reasoning* and *security in LLMs*. As reasoning-based fine-tuning becomes increasingly popular, understanding its new vulnerabilities is both important and urgent.  
- The paper is clearly structured and easy to follow. The threat model, experimental setup, and results are communicated with good clarity.

### Weaknesses
1. The core idea—repurposing existing poisoning attacks through the reasoning process—is conceptually appealing but technically shallow. The paper does not introduce new attack algorithms or mechanisms; it mainly adapts known data poisoning principles to CoT data.  

2. The attack goal (targeted misbehavior such as [1-2]) has been extensively studied in the context of non-reasoning LLMs. While applying it to the reasoning stage is novel in setting, the methodology is largely parallel to prior work. The paper would be more compelling if it provided a systematic comparison showing how reasoning-based poisoning differs fundamentally in mechanism or detection difficulty.  


**Reference**

[1] On the Exploitability of Instruction Tuning

[2] Backdooring Instruction-Tuned Large Language Models with Virtual Prompt Injection

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces “thought-transfer,” a novel indirect targeted poisoning attack on reasoning-enabled large language models (LLMs) that manipulates Chain-of-Thought (CoT) traces during fine-tuning. Unlike traditional backdoor or CoT poisoning attacks, which inject explicit triggers and wrong answers into training data, thought-transfer modifies only the reasoning traces while keeping the queries and final answers intact, effectively creating clean-label poisoning. The attack transfers adversarial reasoning patterns from unrelated tasks to influence model behavior on unseen target tasks.

### Strengths
Identifies a previously unexplored attack vector exploiting reasoning traces rather than inputs or outputs.

Clean-label attacks.

### Weaknesses
The mechanism of reasoning-pattern transfer is described conceptually but lacks theoretical or mechanistic explanation.

Only perplexity and CoT autorater defenses are tested.

### Questions
How would the attack perform when reasoning data are combined with preference alignment (RLHF or DPO) post-training? Would alignment mitigate or amplify thought-transfer effects?

Have you tested the persistence of the poisoned behavior under continued fine-tuning or reinforcement learning (i.e., does it decay or strengthen)?

Does the attack persist under instruction variations or prompt randomization during inference?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents an attack methodology, termed "thought-transfer," which is designed to introduce targeted adversarial behavior into large language models (LLMs) that utilize Chain-of-Thought (CoT) reasoning. The key feature of this attack is that it exclusively modifies the CoT traces in the training dataset while preserving the original query and the correct final answer, effectively executing a form of clean-label poisoning. The goal is to transfer a specific malicious reasoning pattern (the "carrier") onto an entirely different, previously unseen target task. The authors evaluate this attack across various scenarios, including injecting advertisements and manipulating concepts, demonstrating its high success rate and pointing out that the resulting poisoned models show an improvement in general benchmark performance.

### Strengths
1.  **Focus on Reasoning Integrity:** The research correctly identifies the reasoning trace as a critical and often overlooked attack surface. By showing that malicious behavior can be embedded directly within the logic flow without immediately altering the output, the paper highlights a significant vulnerability in current LLM training practices involving public CoT datasets.
2.  **Comprehensive Experimental Scope:** The authors validate the attack across diverse settings, including manipulations in natural language and code generation domains, and test the transferability between related and unrelated tasks. This broad testing demonstrates the attack's robustness under various conditions.

### Weaknesses
1.  **Complexity and Accessibility of the Attack Preparation:** The method for constructing the poisoned data, especially the "LLM Merge-Based Integration" strategy, introduces a substantial reliance on a high-quality external language model (LLM). This requirement makes the preparation of the attack resource-intensive and complex, suggesting the attack is less generally accessible than simple token-based poisoning methods.
2.  **Limited Technical Leap in Poisoning:** While the resulting effect is highly insidious, the technique of combining different data segments is fundamentally an application of known data augmentation and instructional tuning principles. The methodological foundation of injecting content into a prompt/trace is not a significant departure from established data poisoning literature, limiting the advancement in core attack techniques.
3.  **Questions on Clean-Label Purity:** For adversarial objectives like concept manipulation (e.g., misrepresenting a scientific concept) or inserting non-existent libraries in code, the "correctness" of the final answer is debatable. The poisoning fundamentally corrupts the model's factual knowledge or safety, making the clean-label claim (maintaining utility) potentially misleading regarding the model's integrity. The ethical implications of this knowledge corruption need clearer definition.

### Questions
Please see the weakness.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces thought-transfer, an interesting poisoning attack targeting the CoT reasoning process in LLMs. The attack's core concept is to manipulate a model's behavior on an unseen target task by only altering the CoT traces within the training data of a different, carrier task, which is framed as a clean-label attack, as the queries and final answers in the poisoned data remain correct.

### Strengths
The paper addresses a problem of significant practical importance: the security of publicly available training datasets. The threat of data poisoning in open-source repositories like Hugging Face is a real and pressing concern, and this work rightly identifies this as a critical vulnerability in the AI ecosystem. Besides, I find the core idea of using CoT traces as the attack vector to be a reasonable and interesting direction. The intuition that manipulating the intermediate reasoning steps, rather than the final labels, could be a stealthier way to inject adversarial behaviors is a valid one. This approach moves beyond simpler, more obvious forms of data poisoning and explores a more nuanced attack surface. The poisoned data appears to improve model performance, is also a conceptually compelling aspect of the threat model.

### Weaknesses
1. My primary issue is that the paper fails to clearly explain how the poisoned CoTs are actually constructed. This is the most critical part of the entire work, yet its description is delayed until Section 4.2 (page 5) and, even then, is frustratingly abstract and uninformative. The paper never defines what it means to ``poison`` a CoT. For example, how does one systematically ``manipulate`` a CoT to achieve a targeted outcome? Is it by generating factually incorrect reasoning, embedding subtle harmful content, or creating a seemingly normal CoT that somehow triggers malicious behavior post-training? The paper provides only high-level, declarative statements instead of a concrete, reproducible methodology. Without a precise definition of the poisoning process, the central mechanism of the attack remains a black box, and the work is not scientifically sound.

2. The paper proposes two integration strategies: ``Concatenation-Based`` and ``LLM Merge-Based.`` These methods appear to be little more than simple string concatenation and a standard prompt-based LLM call, respectively. I don't see much novelty in these designs. Furthermore, the details of the ``carefully crafted merging instruction $s_i$`` for the LLM-merge method are not provided (Line 271), making it impossible to evaluate or reproduce. More importantly, looking at the results in Table 1, both methods achieve nearly identical attack success rates (78.7% vs. 79.0%). This raises the question: what is the point of designing two different methods if their performance is the same? It seems one could just use the much cheaper and simpler concatenation method. The paper does not provide a compelling justification for the existence and comparison of these two simplistic strategies, which makes the contribution feel less substantial.

3. The paper's evaluation, while broad in scope, lacks depth in several key areas. First, the definition of ''Attack Success'' seems very narrow. It's measured by checking for the presence of a specific string (e.g., a book title or "NordVPN") in the output. This doesn't capture the full extent of potential model manipulation. How well does the attack generalize beyond these simple keyword-injection tasks? For instance, can it manipulate the model's tone, style, or more complex behaviors? The paper's claims of general ''thought-transfer'' are not fully supported by this narrow metric.

4. Furthermore, there is no ablation study on the impact of the poisoning rate. The paper doesn't analyze the relationship between the percentage of poisoned CoTs in the training data and the resulting attack success rate. How many poisoned samples are actually needed to achieve the reported 70%+ success? Is it 1%, 5%, or 10% of the dataset? Without this crucial analysis, we have no understanding of the attack's efficiency or its dose-response curve. This is a significant omission for a paper on data poisoning. The claim of high attack success is meaningless without knowing the cost.

### Questions
Please refer to the question

### Soundness
3

### Presentation
1

### Contribution
2
