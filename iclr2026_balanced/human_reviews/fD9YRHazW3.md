## Human Reviewer 1

### Summary
The paper introduces a new type of LLM watermarking that does not require access to the decoding process. The main idea is to embed a watermarking signal directly into the model's output through prompt modification. The authors propose two ways to do this. The first is by giving the LLM direct instructions in the system prompt, such as asking it to start words with specific letters or follow a detectable pattern. The second is by inserting watermarking instructions into the main body of the input, for example in a PDF that could be used by dishonest reviewers to automatically review papers.

### Strengths
- I really like the main idea of the paper. It opens a new and relevant direction in LLM watermarking, with interesting applications such as detecting dishonest reviewers.

- The paper is clearly written and easy to follow.

### Weaknesses
- The paper lacks an evaluation of the true positive rate at very low false positive rates. For the method to be practical (for example, in detecting dishonest reviewers), it needs to perform well at very low FPRs (e.g., 0.1%).

- It would also be helpful to include a human evaluation to support the results from the LLM-as-a-judge or perplexity based evaluation, although I do not think this is required for acceptance.

### Questions
I think the users could notice (at least for some watermarking strategies) the watermarking patterns either by direct / visual inspection or by asking another LLM to detect them. How would you defend against the possibility that users identify and remove the watermarking patterns?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
3

---

## Human Reviewer 2

### Summary
The paper proposes a family of In-Context watermarking schemes that require no access to model internals, relying instead on watermarking instructions to induce detectable patterns in LLM outputs. Experiments on proprietary models (GPT-4o-mini and GPT-o3-mini) show that detection improves with model capability in both DTS and IPI settings. The paper reports robustness against edits/paraphrasing and small quality costs as judged by an LLM-as-a-judge.

### Strengths
The work formulates ICW embedding/detection without needing logits/decoding, filling a gap between post-hoc detection and in-process watermarking.

The four ICWs span character/word/sentence levels and include concrete detection rules.

Results provide interesting empirical insight, showing the relationship between LLM instruction-following capacity and ICWs' effectiveness.

### Weaknesses
Covertly modifying review manuscripts with hidden instructions (even for integrity) may raise consent or transparency issues. Authors should discuss this point.

The detection performance of the proposed method is evaluated only on two LLMs. At least one more LLM is suggested to further demonstrate the effectiveness of the proposed methods.

### Questions
In the IPI setting, what safeguards prevent false accusations (e.g., reviewers unintentionally triggering a watermark through unrelated text patterns)?

How does detection performance change as the length of the output increases?

How sensitive are the Initials/Lexical ICWs to domain shifts?

If watermarks are embedded using one model family and text is later post-edited by another, how do detectors fare?

### Soundness
3

### Presentation
4

### Contribution
4

### Rating
6

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper introduces ICWs, a novel approach to watermark LLMs generated text without modifying the decoding process. Instead, it embeds watermarks through prompt engineering, leveraging LLMs’ in-context learning and instruction-following capabilities. Specifically, this paper explores four ICW strategies, each with different embedding and detection schemes. These strategies are evaluated across two settings: DTS and IPI. Comprehensive experiments demonstrate the effectiveness of the proposed methods, particularly with more capable LLMs.

### Strengths
1.	The concept of embedding watermarks through prompt engineering instead of manipulating the decoding process is interesting and novel, especially under the black-box setting. In this scenario, the control of the LLM watermark is not limited to the LLM owner, which provides a broader application scenario.
2.	This paper evaluates the ICWs on two real-world applications, DTS and IPI, demonstrating the practicability.
3.	This paper conducts an extensive experimental evaluation, demonstrating the effectiveness of the proposed methods.

### Weaknesses
1.	The proposed ICWs rely on the instruction following capability of LLMs. Authors should evaluate whether the watermarking instruction can still be reliably followed after a multi-turn conversation.
2.	The current design of the watermarking instruction is intuitive. I am interested in whether the watermarking instruction could be further optimized using RL or some prompt tuning methods (just for discussion).
3.	Authors should add some discussion about the reason that Lexical and Acrostics ICWs show lower detection performance under the deletion attack.

### Questions
1. This paper introduces one method, e.g., white text, to covertly embed the watermarking instruction into the confidential file. Can you introduce more potential methods to covertly embed the watermarking instruction?
2. I am interested in the robustness performance of ICW under the copy-paste attack.
3. How is the detection threshold \(\eta\) in Eq. (H1) of Sec. 3.1 chosen in practice?
4. Is it dataset-specific, or does a universal threshold generalize across domains?
5. What additional computational cost (token length, inference latency) does each ICW introduce relative to normal prompting?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
4

---

## Human Reviewer 4

### Summary
This paper explores In-Context Watermarking for LLMs, a kind of approach that embeds watermarks into LLM-generated text solely through prompt engineering. The authors test four ICW strategies operating at different linguistic granularities. The paper evaluates these methods in two settings: Direct Text Stamp and Indirect Prompt Injection. The work positions ICW as a promising model-agnostic watermarking paradigm that empowers third parties to trace AI-generated content without requiring cooperation from model providers.

### Strengths
- The authors systematically evaluate trade-offs among LLM requirements, detectability, robustness, and text quality for each strategy, providing valuable insights into the design space of ICW methods.
- The authors acknowledge that ICW effectiveness depends critically on model capabilities, with weaker models essentially unable to follow complex watermarking instructions. The adaptive attack experiments (Table 10) showing partial watermark removal are included transparently. This intellectual honesty strengthens the paper's credibility.
- The Indirect Prompt Injection setting for detecting AI-generated academic reviews represents a timely motivated use case.

### Weaknesses
- The paper's primary weakness lies in presenting existing techniques as novel contributions while failing to properly acknowledge prior work. In-context watermarking is not a new concept. The idea of using prompts to embed watermarks has been explored in multiple prior works, yet the authors provide no discussion of this existing literature in their Related Work section. More critically, the four proposed ICW strategies lack originality and are straightforward applications of well-known techniques. Unicode ICW using zero-width characters is an extremely common information hiding technique that has been extensively studied for decades, representing standard steganography practice rather than a novel contribution. The Lexical ICW directly borrows the green/red-list concept from KGW, the only difference is implementation via prompts rather than logits manipulation. Acrostic steganography has centuries of history, and recent work has already demonstrated using ChatGPT for acrostic-based information hiding (https://daniellerch.me/stego/text/chatgpt-en/), yet this precedent is not mentioned. Furthermore, the paper provides no meaningful insights beyond these straightforward applications. The contribution essentially reduces to showing "you can ask LLMs to follow watermarking instructions via prompts", which is a trivial observation given modern LLMs' instruction-following capabilities.

- Even setting aside novelty concerns, the proposed methods demonstrate fundamental practical limitations that severely constrain their real-world applicability. As the authors acknowledge, these heuristic approaches significantly compromise output text quality. The methods are also fragile against simple attacks: even without knowing the specific watermarking scheme, adversaries can substantially remove watermarks through text rewriting. This fundamental vulnerability undermines the practical utility of ICW. Additionally, the approach exhibits strong model capability dependency, only working with the most advanced models like GPT-o3-mini.

- The proposed ICW methods can be understood as straightforward applications of well-established concepts without meaningful innovation. The IPI setting is essentially a benign variant of indirect prompt injection attacks, the novelty of "using this for watermarking instead of attacks" is incremental at best. Similarly, instructing models to embed hidden patterns via prompts is conceptually identical to in-context backdoor insertion, yet the paper does not acknowledge or discuss this connection. 

- The ICW methods suffer from fundamental compatibility issues with diverse writing scenarios. For instance, Acrostics ICW, which constrains sentence-initial letters to follow a predetermined sequence, is unsuitable for many real-world applications such as technical documentation and code comments. 

- The paper claims to be the "first" to explore ICW, but provides no systematic literature review to support this claim.

### Questions
1. Please clarify the novelty of the four proposed heuristic ICW methods and explain how they differ from existing approaches.

2. Please compare your methods with a broader range of in-context watermarking approaches.

3. Given that attackers can easily remove watermarks without knowing the specific algorithm, how can these methods be applied in practice?

4. In writing scenarios that prohibit the structural or stylistic patterns required by your ICW methods, how can your approach be applied?

### Soundness
2

### Presentation
2

### Contribution
1

### Rating
2

### Confidence
5