import pytest

from ragengine.agent import AgentOrchestrator, RetrieverTool
from ragengine.documents import Document
from ragengine.llm import get_llm
from ragengine.retrievers import VectorStoreRetriever
from ragengine.vectorstores import get_vector_store


@pytest.fixture
def tool(fake_embedding):
    store = get_vector_store("chroma", collection_name="agent_test_store")
    docs = [Document(page_content="The vacation policy allows 20 days of paid leave per year.", metadata={"source": "policy.txt"})]
    store.add(docs, fake_embedding.embed_documents([d.page_content for d in docs]))
    retriever = VectorStoreRetriever(store, fake_embedding, search_type="similarity")
    return RetrieverTool(retriever=retriever, k=2)


def test_retriever_tool_format_context_empty():
    assert RetrieverTool.format_context([]) == "(no results found)"


def test_retriever_tool_format_context_includes_score_and_source():
    doc = Document(page_content="hello", metadata={"source": "x.txt"})
    formatted = RetrieverTool.format_context([(doc, 0.87)])
    assert "score=0.87" in formatted
    assert "source=x.txt" in formatted
    assert "hello" in formatted


def test_agent_searches_then_answers(tool):
    stub = get_llm("stub", responses=["SEARCH: vacation days", "ANSWER: You get 20 days of paid leave per year."])
    agent = AgentOrchestrator(tool=tool, llm=stub, max_iterations=3)

    result = agent.run("How many vacation days do I get?")

    assert result.answer == "You get 20 days of paid leave per year."
    assert result.trace[0] == {"action": "search", "query": "vacation days", "num_results": 1}
    assert result.trace[1]["action"] == "answer"
    assert tool.last_results  # the tool was actually called


def test_agent_answers_immediately_without_searching(tool):
    stub = get_llm("stub", responses=["ANSWER: 42"])
    agent = AgentOrchestrator(tool=tool, llm=stub, max_iterations=3)

    result = agent.run("What is the meaning of life?")

    assert result.answer == "42"
    assert len(result.trace) == 1
    assert result.trace[0]["action"] == "answer"


def test_agent_forces_final_answer_after_exhausting_iterations(tool):
    stub = get_llm("stub", responses=["SEARCH: a", "SEARCH: b", "Best-effort answer given limited context."])
    agent = AgentOrchestrator(tool=tool, llm=stub, max_iterations=2)

    result = agent.run("Some hard question")

    assert result.answer == "Best-effort answer given limited context."
    assert result.trace[-1]["forced"] is True
    # exactly 2 searches happened (bounded by max_iterations), not 3
    assert sum(1 for step in result.trace if step["action"] == "search") == 2


def test_agent_treats_unparseable_reply_as_best_effort_answer(tool):
    stub = get_llm("stub", responses=["I'm just going to ramble without the right format."])
    agent = AgentOrchestrator(tool=tool, llm=stub, max_iterations=3)

    result = agent.run("A question")

    assert result.answer == "I'm just going to ramble without the right format."
    assert result.trace[0]["unparsed"] is True


def test_agent_rejects_non_positive_max_iterations(tool):
    stub = get_llm("stub")
    with pytest.raises(ValueError):
        AgentOrchestrator(tool=tool, llm=stub, max_iterations=0)
