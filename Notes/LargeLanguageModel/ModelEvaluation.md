# Model Evaluation

1. 输入是什么
2. 如何调用语言模型
3. 如何评估输出
4. 如何解释评估指标

由于预训练数据非常大，所以在评估的时候要注意测试集与训练集的重叠。

## 困惑度

困惑度本质上衡量了语言模型对某个数据集赋予高概率的程度，困惑度越高，概率分布越锐化，反之越均匀。

- 比下游任务准确率更加平滑
- 不论任务是什么，困惑度都很普适
- 同样可以衡量下游任务，基于提示进行条件化。

## Knowledge Benchmarks

### Massive Multitask Language Understanding (MMLU)

为了回答这些问题，模型需要指令微调，以学习到更符合任务的表示，例如在选择题中只回复(A/B/C/D)。
指令微调就是对这个学生进行“特训”的过程。我们给他大量“指令-回答”的配对样本：

    指令：用自然语言清晰描述一个任务，比如“请将以下句子翻译成英文：今天天气很好”。

    回答：期望模型输出的正确答案，比如 “The weather is nice today.”

通过在这些精心准备的（指令，回答）数据集上进一步训练，模型会逐渐学会：看到一条新指令时，应该输出什么样的回复才算“正确”和“有用”。这个“特训”过程就是指令微调。

### MMLU-PRO

- Removed noisy/trivial questions from MMLU
- Expanded 4 choices to 10 choices
- Evaluated using chain of thought
- Accuracy of models drop by 16% to 33% (not as saturated)

思维链(Chain of thought COT) 在MMLU-PRO上表现更好。

### Graduate-Level Google-Proof (GPQA)

Questions written by 61 PhD contractors from Upwork

### Humanity's Last Exam

2500 questions: multimodal, many subjects.

## Instruction Following Benchmarks

指令跟随就是经过上述“特训”后，模型展现出来的能力。

具体表现为：当用户给它一条从未见过的、表述清晰的指令时，它能准确理解任务目标、约束条件和输出格式，并生成符合预期的回答。

    能听懂“人话”任务：不只是做填空题或续写，而是执行“总结”、“翻译”、“分类”、“创作”等开放任务。

    能遵守“规则”：比如指令要求“用一句话回答”、“列出三个要点”或“扮演一名老师”，模型能够遵循这些要求。

一个指令跟随的例子：

    用户指令：写一封简短的邮件给同事小王，告诉他今天的会议从下午3点推迟到4点，并为临时更改道歉。

    具备指令跟随能力的模型回答：

        邮件主题：关于会议时间调整

        小王，你好。抱歉临时通知你，今天下午的会议将从原定的3点推迟到4点举行。给你带来不便，非常抱歉。请知悉。

模型准确理解了“收件人、目的、新时间、道歉”等所有关键要素，并正确生成了邮件格式。这就是指令跟随。

### Chatbot Arena

- Random person from the Internet types in prompt
- They get response from two random models
- The rate which one is better.
- ELO scores are computed based on the pairwise comparison.

### Instruction-Following Eval(IFEval)

___

### AlpacaEval

Metric: win rate against GPT-4 previews as judged by GPT-4 preview(potential bias)

### WildBench

Uses GPT-4 turbo as a judge with a checklist (like COT for judging) + GPT-4 as a judge.

### Other Benchmarks

- CyBench
- MLEBench

## Pure Reasoning Benchmarks

--
