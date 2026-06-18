## TODO:
- MCP implementation
- Create tools for MCP
- Create tools for llm model

## Notes
- clean-notebook-memory.ipynb
- guides/04_technical_foundations.ipynb
- guides/09_ai_apis_and_ollama.ipynb
- guides/11_async_python.ipynb
- week1/week1 EXERCISE.ipynb
- week2 model to model chat, gradio, tools in day 4, 
- week3 colab, models, tokenisers, pipelines, minutes of meeting creator, visualizer
- week4 week4/model basics.md, code conversion and model, keys, implementation
- week5 notes in day 3, week5/implementation/ingest.py, pydantic & AnswerEval in week5/evaluation/eval.py, day 3 hugging, face/langchain, RAG, issues with history, pro_implementation VectorDB, multiprocessing in ingest.py,  !pip & %pip difference, re-ranking
- problem with sharing history(not in correct way)
    in week5/implementation if asked 1st about Avery then queried about
    someone else's achievement without mentioning there name then llm is not
    able to answer(example in week5/day3 seek till 10min)
- week2/community-contributions/AI Library Clerk
- week8/community_contributions/raheem_yaqub_adesola/multi-agent-salary-intelligence-system.ipynb
- week8/community_contributions/kachaje-andela-genai-bootcamp-w8/price-is-right/shared/deal_agent_framework_client.py
- week8/community_contributions/kachaje-andela-genai-bootcamp-w8/price-is-right/services/ui.py
- week8/community_contributions/kachaje-andela-genai-bootcamp-w8/price-is-right/services/notification_service.py
- week8/community_contributions/kachaje-andela-genai-bootcamp-w8/price-is-right/services/notification_receiver.py
- week8/community_contributions/codypharm/pharma_agents.py
- week8/community_contributions/codypharm/app.py


## 10 RAG Advanced Techniques

- **Chunking R&D :** experiment with chunking techniques TextSplitters/Chunkers
- **Encode R&D :** Images and Text and mapping then into consistance vector space (Sentence transformer in hugging face)
- **Improve Prompts:** general content, the current date, relevant context and history
- **Document pre-processing:** use an LLM to make the chunks and/or text for encoding, using LLM to make data more relevant for use before vectorisation more like doing **Semantic Chunking**, prompt ```like "break into meaningful chunk for my knowledge base"```
- **Query rewriting :** use an LLM to convert the user's question to a RAG query
- **Query expansion :** use an LLM to turn the question into multiple RAG queries
- **Re-ranking :** use an LLM to sub-select from RAG results (LLM resonsible for ordering/ranking most relevant to least relevant)
- **Hierarchical :** use LLM to summarize at multiple levels, summarise knowlegebase
- **Graph RAG :** retrieve content closely related to similar documents. There are special databases called ```graph databases``` like ```Neo4j's``` and many of them that are specially designed to store
Information as as nodes with edges.
A way to think about things in terms of relationships between entities.
And if you store it in a graph database like that, then it's even easier for you to find a chunk that's close to a vector, and then include that and include all the chunks that are the sort of 1 or 2 neighbors away in the graph database.
- **Agentic RAG :** use Agent for retrieval, combining with Memory and Tools such as SQL.
Let the LLM make the decisions. So instead of just doing a vector lookup and putting all that in the context, call an LLM and give it some tools, give it a tool that could do a vector lookup based on a query, maybe give it some more tools, like a SQL tool that can run SQL on a database, or if it's the files that we've got in our knowledge base, just do a string lookup in the files if it wants to give it access to a bunch of tools and then call this like your retrieval agent or something, and then say, here's a user question. It is about doing all the previos things before it, but letting the LM decide which to do and in which order.



# Notes on `answer.py` — LiteLLM Routing & Custom URLs

---

## Is `completion` using a local model or Groq via API key?

It's calling **Groq's remote API** — not a local model.

`litellm` uses provider prefixes to route requests. The `MODEL` string tells it everything:

```python
MODEL = "groq/openai/gpt-oss-120b"
#        ^^^^  ^^^^^^^^^^^^^^^^^^^
#        │     model name on Groq's platform
#        └── use Groq's API endpoint
```

LiteLLM will look for `GROQ_API_KEY` in your `.env` and send the request to `https://api.groq.com/openai/v1/`.

### How LiteLLM routing works with prefixes

| `MODEL` value | Where it goes |
|---|---|
| `"groq/llama3-8b-8192"` | Groq API |
| `"openai/gpt-4.1-nano"` | OpenAI API |
| `"anthropic/claude-3-5-sonnet"` | Anthropic API |
| `"ollama/llama3.1"` | Local Ollama (`localhost:11434`) |
| `"huggingface/mistralai/Mistral-7B"` | HuggingFace Inference API |
| `"bedrock/claude-3"` | AWS Bedrock |

To switch to a **local model**:
```python
MODEL = "ollama/llama3.1"   # local Ollama
```

---

## Can I give a custom URL to `completion`?

Yes, LiteLLM supports custom base URLs. You can pass it directly to `completion`:

```python
response = completion(
    model="openai/gpt-oss-120b",
    messages=messages,
    base_url="https://your-custom-endpoint.com/v1",  # custom URL
    api_key="your-key",                               # if required
)
```

Or set it globally at the top of the file so all `completion()` calls use it:

```python
import litellm
litellm.api_base = "https://your-custom-endpoint.com/v1"
litellm.api_key = "your-key"
```

Or via environment variables in `.env`:
```
OPENAI_API_BASE=https://your-custom-endpoint.com/v1
OPENAI_API_KEY=your-key
```

### Common use cases

| Scenario | `base_url` |
|---|---|
| Local Ollama via OpenAI-compat API | `http://localhost:11434/v1` |
| vLLM self-hosted | `http://your-server:8000/v1` |
| Azure OpenAI | `https://{resource}.openai.azure.com/` |
| Groq | `https://api.groq.com/openai/v1` (auto with `groq/` prefix) |
| Any OpenAI-compatible server | `https://your-host/v1` |
