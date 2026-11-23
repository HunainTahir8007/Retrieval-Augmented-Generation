from langgraph.graph import StateGraph , START , END

graph = StateGraph(State)

graph.add_node("retrieve" , retrieve)
graph.add_node("refine" , refine)
graph.add_node("generate" , generate)
graph.add_node("eval_each_doc" , eval_each_doc_node)
graph.add_node('fail' , fail_node)
graph.add_node("ambiguous" , ambiguous_node)
graph.add_edge(START , "retrieve")
graph.add_edge("retrieve" , "eval_each_doc")
graph.add_conditional_edges("eval_each_doc" , route_after_eval , {"refine": "refine", "web_search": "fail", "ambiguous": "ambiguous"})
graph.add_edge("refine" ,"generate" )
graph.add_edge("generate" , END)
graph.add_edge("fail" , END)
graph.add_edge("ambiguous" , END)


workflow = graph.compile()
workflow
res = workflow.invoke(
    {
        "question": "What are attention mechanisms and why are they important in current models?"
        
    }
)

print("VERDICT:", res["verdict"])
print("REASON:", res["reason"])
print("OUTPUT:", res["answer"])