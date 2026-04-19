# Teaching LLMs to Teach Themselves Better Instructions via Reinforcement Learning

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 5, 3

## Abstract
The development of Large Language Models (LLMs) often confronts challenges stemming from the heavy reliance on human annotators in the reinforcement learning with human feedback (RLHF) framework, or the frequent and costly external queries tied to the self-instruct paradigm.  In this work, we pivot to Reinforcement Learning (RL)---but with a twist. Diverging from the typical RLHF, which refines LLMs following instruction data training,  we use RL to directly generate the foundational instruction dataset  that alone suffices for fine-tuning. Our method uses a suite of textual operations and rules, prioritizing the diversification of training datasets. It facilitates the generation of  rich instructions without excessive reliance on external advanced models, paving the way for a single fine-tuning step and negating the need for subsequent RLHF stages. Our findings underscore some key advantages of our approach: a diminished need for human involvement and fewer model queries, along with boosting the capability of LLMs in crafting and comprehending complex instructions compared to strong baselines.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors propose to use RL to learn a policy for sampling diverse instructions to generate a dataset for instruction tuning for downstream LLM alignment.


Edit following author response:

Thank you for your honesty re: performance compared to LLaMA-1. However, I feel that the authors missed the main point of many of my other questions. 

Q3: Yes, WizardLM-13B is of course stronger than WizardLM-7B or LLaMA-1-7B so the performance will be better. I was mainly thinking that it would make your method more convincing if you did not need an existing strong instruction-tuned model to bootstrap your approach (i.e., if you could do everything starting from LLaMA-1-7B). This point is a bit moot though since as you mentioned in Q6, it turns out you're actually heavily relying on ChatGPT / GPT4 anyway - I didn't realize there was such heavy reliance on ChatGPT / GPT4 in the method which makes it a bit less convincing as there are a lot of methods these days which basically boil down to distilling from ChatGPT / GPT4. 

Q4: I agree that the benchmark itself is fine. However, I think that an important baseline is missing - generating new instructions via prompting rather than RL, as in e.g. https://arxiv.org/abs/2305.03047 and maybe other works by the same first author.

Q5: This is not a problem of presentation. Rather, presumably you used the original prompts in your experiment run, which would imply that your experiments arguably contained "bugs" due to the typos in the prompts. (I.e., if you were to rerun experiments in the future, I'd recommend fixing the errors in the prompts as there's a chance you could increase your performance for free.)

To be honest, while I do think the core idea of this paper could be potentially promising, I would probably recommend the authors to spend much more time polishing the presentation in the paper and then resubmit; additional experiments or (potentially including further improvements on the methodology) could also make the argument much more convincing as the current results are not that strong.

### Strengths
--there is an interesting idea at the core of this paper: rather than just prompting to generate new diverse instructions for your initial instruction tuning prompt set, you can actually finetune the model for generating that dataset in the first place. this kind of suggests a "hierarchy" of sorts in the data generation, where you generate your data on multiple levels, starting from just your action set in 3.1.1.

### Weaknesses
--performance still seems a bit mixed compared to LLAMA, despite you leveraging ChatGPT/GPT4 and also WizardLM13b. Also, is the comparison to WizardLM7b in Fig 6 fair, given that you used WizardLM13b extensively in your pipeline?

--similarly, it seems like you rely on having a strong instruction-tuned model already (WizardLM13b) as the "Advanced LLM" in Fig1 to be able to train your initial policy for sampling diverse instructions, which seems like maybe a bit of a chicken-and-egg problem. i think it might be more convincing if you were able to use a weaker model to do the initial judgments (e.g., why not use LLaMA7b, or WizardLM7b? are those not good enough for your purposes?), or show that you can later outperform whichever model you use for the initial Advanced LLM.

--unless i missed it, there's no comparison on final downstream performance to any baseline that generates its instruction set just by prompting an LM rather than finetuning, which seems like it'd be the most direct baseline

--there are typos/grammar errors even in the model prompts- these are arguably “bugs”

### Questions
--i don't understand step 4 in algorithm 1 - how is chatgpt/gpt4 also being used to help you generate the complex instructions? are you just prompting it for more instructions to add to your instruction set, in addition to what you generated previously using your smaller RL-trained model?

--nit: some typos and tense changes here and there, might want to proofread

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a novel approach to generate complex instruction-tuning data through reinforcement learning. It negates the need for subsequent RLHF stages. Their method can diminish the dependence on human instructors and moderates the need for constant queries to external models.

### Strengths
This paper proposes a novel way to evolve instructions through reinforcement learning. The experiment results on LM-Eval benchmark demonstrate the effectiveness of their method.

### Weaknesses
1. The paper focuses on enhancing the instructions by iteratively optimizing the policy through RL. However, directly evolving instructions through WizardLM or Tree-Instruct prompts also avoids the need for training a large language model. The benefit brought by their method is constrained.
2. Whether RLHF will further facilitate human alignment is not verified in this paper. Involving RL in the stage of SFT is computationally expensive.

### Questions
None

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes to first train a language model to generate instructions diverse from the seed. This is done using RL where the reward comes from another LM on whether or not the output instruction is of good quality. Then this model is used to generate diverse instructions, responses to which are generated by gpt-3.5 and other LMs, to create a dataset for instruction fine-tuning. When fine-tuned using this data, models like Llama2-chat-7b and WizardLM-7b are shown to improve on some benchmarks.

### Strengths
The direction of lowering the cost of collecting instruction fine-tuning data and eliminating need for human feedback is important in making conversational LLMs more accessible.

### Weaknesses
This work seems to be leveraging additional instruction fine-tuning data (see Step 4 & 5) derived from ChatGPT and GPT-4 without clearly describing how it does so in the corresponding Sections. The contribution of this work seems weak if the responses are generated primarily using external models.

Mixed results with marginal gains compared to the checkpoints they start with, in some cases a drop (Fig 4 & 5). TruthfulQA performance of llama-2-chat-7b is under-reported as 38.98, Table 14 from the Llama2 paper reports the performance of the 7B chat model to be at 57.04.

I find it very hard to comprehend the problem being solved and the approach being proposed in this submission. At least some parts of the paper seem to be LLM-generated.

### Questions
Questions on steps of the Algorithm proposed

Step 1: Desigin of actions - do you simply use the same actions as proposed by WizardLM?

Step 2: What does discrete value-based action space S mean?

Step 3: How is TRPO used here? The binary feedback ('reward') that you get on diversity of the generated instructions is used to train the base LM, this appears to be the rejection sampling approach.

Step 4 & 5: How is ChatGPT or GPT-4 used here? 

Why are LLMs called Advanced LLM in Fig 1 & 2?

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair
