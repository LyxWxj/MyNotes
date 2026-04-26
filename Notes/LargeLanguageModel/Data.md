# 数据训练策略

## 数据集

- bert: Wikipedia, books
- gpt2_webtext/OpenWebTextCorpus: pages based on Reddit links
- CommonCrawl: web crawl
- ccnet: Filter Common Crawl based on wikipedia
- t5_c4: Filter using rules
- gpt3: CommonCraw, Wikipedia, books
- the_pile: Lots of sources: books3，stackoverflow, github
- gopher_massivetext: Filter using rules (trained Gopher)
- llama: CommonCrawl, CCNet, StackExchange, etc.
- ReginedWeb: CommonCrawl
- dolma: Lots of different sources

## Model-based filtering

--

## Mid-training + post-training

### long context

- longLoRA
 Extemds context length
 Uses shifted sparse attention positional interpolation
 Trained on long documents:PG-19 and Proof-Pile

### tasks

--

### instruction_chat

--
