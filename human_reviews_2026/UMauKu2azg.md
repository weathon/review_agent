# VPI-Bench: Visual Prompt Injection Attacks for Computer-Use Agents

- Decision: Accept (Poster)
- Scores: 6, 6, 4

## Abstract
Computer-Use Agents (CUAs) with full system access enable powerful task automation but pose significant security and privacy risks due to their ability to manipulate files, access user data, and execute arbitrary commands. While prior work has focused on browser-based agents and HTML-level attacks, the vulnerabilities of CUAs remain underexplored. In this paper, we propose an end-to-end threat model where Visual Prompt Injection (VPI) manipulates CUAs in black-box settings to perform unauthorized actions or leak sensitive
information, capturing the entire attack chain from injection to harmful outcomes. Then, we propose VPI-Bench, a benchmark of 306 test cases across five widely used platforms, to evaluate agent robustness under VPI threats. Each test case is a variant of a web platform, designed to be interactive, deployed in a realistic environment, and containing a visually embedded malicious prompt. Our empirical study shows that current CUAs and BUAs can be deceived at rates of up to 51\% and 100\%, respectively, on certain platforms. The experimental results also indicate that existing defense methods offer only limited improvements. These findings highlight the need for robust, context-aware defenses to ensure the safe deployment of multimodal AI agents in real-world environments.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a new benchmark (VPI-Bench) for determining the susceptibility of mutli-modal agents to pop-up based attacks when performing representative computer and browser based tasks. Two agents powered buy different models (VLMs) are evaluated on the computer based task. Five agents powered buy different VLMs are evaluated on the browser based task. The paper shows that all models are susceptible to these attacks some degree. The paper also provides some other interesting results related to attack timing and failure analysis of the attack.

### Strengths
The paper has the following strengths:
- The paper is well written clear and easy to follow. 
- The benchmark looks very useful for future research
- The paper looks at full agent roll outs. Both considering early and late attacks and checking if the attack was carried out to completion. 
- The paper shows that current multi-model agents are still highly susceptibility to simple visual pop-up based attacks, however the novelty here is limited

### Weaknesses
The paper has the following weaknesses:
- The papers threat model seems unrealistic assuming that large platforms such as the BBC or Amazon have been compromised and infected with adversarial pop-ups.
- To the best of my knowledge the paper does not present results for benign performance of the model on the tasks when no attack is performed. This is strange as it is standard practice in many of the prior works cited in this paper. It also provides for a much richer analysis of what might be causing the differences between models and settings.
- The paper is missing lots of important references (see below) and the number of reference is about half what I would expect given the amount of reach happening in this area.
- The over all contribution outside of the benchmark is limited, the attack vector is not novel, nor is a benchmark evaluating attacks on visual agents
- The paper reports many results to 2 decimal places which seems like an odd choice given the statistical power of the experiments performed and the reliance on LLM as a jury.

 List of missing citations (none-exhaustive)

Text Based Agent Benchmarks 

@article{debenedetti2024agentdojo,
  title={Agentdojo: A dynamic environment to evaluate prompt injection attacks and defenses for llm agents},
  author={Debenedetti, Edoardo and Zhang, Jie and Balunovic, Mislav and Beurer-Kellner, Luca and Fischer, Marc and Tram{\`e}r, Florian},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={82895--82920},
  year={2024}
}

@article{vijayvargiya2025openagentsafety,
  title={Openagentsafety: A comprehensive framework for evaluating real-world ai agent safety},
  author={Vijayvargiya, Sanidhya and Soni, Aditya Bharat and Zhou, Xuhui and Wang, Zora Zhiruo and Dziri, Nouha and Neubig, Graham and Sap, Maarten},
  journal={arXiv preprint arXiv:2507.06134},
  year={2025}
}

@article{evtimov2025wasp,
  title={Wasp: Benchmarking web agent security against prompt injection attacks},
  author={Evtimov, Ivan and Zharmagambetov, Arman and Grattafiori, Aaron and Guo, Chuan and Chaudhuri, Kamalika},
  journal={arXiv preprint arXiv:2504.18575},
  year={2025}
}

MutliModal Agent BenchMarks

@article{zhou2024multimodal,
  title={Multimodal situational safety},
  author={Zhou, Kaiwen and Liu, Chengzhi and Zhao, Xuandong and Compalas, Anderson and Song, Dawn and Wang, Xin Eric},
  journal={arXiv preprint arXiv:2410.06172},
  year={2024}
}

Adversial Image Attacks on Multimodal Agents 

@article{fu2024imprompter,
  title={Imprompter: Tricking llm agents into improper tool use},
  author={Fu, Xiaohan and Li, Shuheng and Wang, Zihan and Liu, Yihao and Gupta, Rajesh K and Berg-Kirkpatrick, Taylor and Fernandes, Earlence},
  journal={arXiv preprint arXiv:2410.14923},
  year={2024}
}

@article{aichberger2025attacking,
  title={Attacking multimodal os agents with malicious image patches},
  author={Aichberger, Lukas and Paren, Alasdair and Gal, Yarin and Torr, Philip and Bibi, Adel},
  journal={arXiv preprint arXiv:2503.10809},
  year={2025}
}

@article{wu2024dissecting,
  title={Dissecting adversarial robustness of multimodal lm agents},
  author={Wu, Chen Henry and Shah, Rishi and Koh, Jing Yu and Salakhutdinov, Ruslan and Fried, Daniel and Raghunathan, Aditi},
  journal={arXiv preprint arXiv:2406.12814},
  year={2024}
}

@article{wang2025manipulating,
  title={Manipulating Multimodal Agents via Cross-Modal Prompt Injection},
  author={Wang, Le and Ying, Zonghao and Zhang, Tianyuan and Liang, Siyuan and Hu, Shengshan and Zhang, Mingchuan and Liu, Aishan and Liu, Xianglong},
  journal={arXiv preprint arXiv:2504.14348},
  year={2025}
}

### Questions
- Is there a reason you did not include the benign performance of the agents on the user tasks (under no attack)?
- Your threat model includes a "A pseudo-authentic webpage" can you explain why you think this is realistic? It would not be in one of these companies long term interest to include these sorts of attacks on their webpage. Does you thread model rely on these pages being compromised? If not is seem and explanation on how agents would be directed to "pseudo-authentic webpages" is missing from the paper. 

My scores are assuming these question are adequately address in any accepted version.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper studies how Visual Prompt Injection (VPI) can trick Computer-Use Agents (CUAs) and Browser-Use Agents (BUAs) into doing harmful actions, like leaking data or deleting files.

 It builds a benchmark called VPI-Bench with 306 test cases across five platforms (Amazon, Booking, BBC, Messenger, Email) to test how vulnerable current agents are.

 Results show that BUAs can be fully deceived and CUAs also fail in many cases. Existing defenses like fine-tuning or system prompts don’t work well, showing that current AI agents are still unsafe under realistic visual attacks.

### Strengths
The paper is well-writted, easy to follow, and clearly articulates an underexplored but timely problem—security risks of computer-use agents under visual prompt injection. 

The proposed end-to-end threat model captures realistic adversarial scenarios where visual cues alone can trigger system-level consequences, bridging a gap left by previous HTML-based or non-interactive attack studies. 

VPI-Bench is comprehensive and reproducible, spanning multiple platforms, tasks, and agent types, with detailed methodology (sandboxed environments, majority-voting evaluation by LLMs).

 The empirical results are extensive, covering behavioral analyses, timing of injection, and defense effectiveness, thereby offering valuable insights for future security research in AI agents.

### Weaknesses
While the empirical coverage is broad, the conceptual novelty is limited—the paper mainly packages known ideas (prompt injection + visual modality) into a benchmark without proposing new defensive mechanisms or theoretical contributions. 

Moreover, the analysis depth of why certain models or scenarios are more vulnerable is shallow—there is little interpretability or causal insight beyond quantitative rates. 

Finally, the defense discussion remains superficial, reiterating that existing methods fail but not offering concrete pathways toward robust solutions.

### Questions
see weakness

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
3

### Summary
This paper introduces VPI-Bench, a comprehensive benchmark designed to systematically evaluate Visual Prompt Injection attacks targeting both Computer-Use Agents and Browser-Use Agents. The benchmark comprises 306 interactive test cases spanning five widely-used platforms (Amazon, Booking, BBC, Messenger, and Email), where malicious instructions are embedded as visual elements in webpage interfaces. Experimental results demonstrate significant vulnerabilities across all tested agents. The study further examines existing defense mechanisms, including model fine-tuning, framework-level protections, and system prompt defenses, finding that they offer only limited effectiveness against these visual injection attacks.

### Strengths
1.  The paper is clearly organized and well-written.
2. The benchmark spans multiple domains (e-commerce, travel, news, communication), evaluates a diverse set of state-of-the-art models, and incorporates both Computer-Use and Browser-Use agent frameworks, ensuring a thorough and multi-faceted evaluation of vulnerabilities.
3. The paper provides a valuable, empirical analysis of multiple contemporary defense strategies (fine-tuning, framework-level guards, system prompts). The findings that these offer only limited protection yield crucial insights for the research community.

### Weaknesses
1. A primary concern is the lack of a clear and significant distinction between the proposed Visual Prompt Injection attacks and previously studied pop-up attacks. This ambiguity substantially limits the perceived novelty and contribution of the work.
2. The reliance on an ensemble of LLMs as judges for evaluating agent behavior requires stronger validation. The absence of a comprehensive human study to robustly verify the accuracy and reliability of this automated evaluation method is a notable limitation.
3. The scope of evaluated agents is limited. The benchmark does not include agents that have undergone extensive task-specific fine-tuning (e.g., models like UI-TARS). Investigating such agents could yield more nuanced and valuable insights for the research community.
4. The current set of implemented attacks is relatively simple and may not encompass the full spectrum of potential real-world attack vectors. A broader and more sophisticated taxonomy of visual injection methods would strengthen the benchmark's comprehensiveness.

### Questions
What is the main difference between Visual Prompt Injection attacks and pop-up attacks?

### Soundness
2

### Presentation
3

### Contribution
2
