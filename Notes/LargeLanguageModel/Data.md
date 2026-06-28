# 数据训练策略

---

## 数据集

> [!info] 常用数据集
> - **BERT**: Wikipedia, books
> - **GPT-2 WebText / OpenWebTextCorpus**: pages based on Reddit links
> - **CommonCrawl**: web crawl
> - **CCNet**: Filter Common Crawl based on wikipedia
> - **T5 C4**: Filter using rules
> - **GPT-3**: CommonCrawl, Wikipedia, books
> - **The Pile**: Lots of sources: books3, stackoverflow, github
> - **Gopher MassiveText**: Filter using rules (trained Gopher)
> - **LLaMA**: CommonCrawl, CCNet, StackExchange, etc.
> - **RefinedWeb**: CommonCrawl
> - **Dolma**: Lots of different sources

---

## Model-based filtering

---

## Mid-training + post-training

### long context

> [!note] LongLoRA
> - Extends context length
> - Uses shifted sparse attention positional interpolation
> - Trained on long documents: PG-19 and Proof-Pile

### tasks

---

### instruction_chat

---
