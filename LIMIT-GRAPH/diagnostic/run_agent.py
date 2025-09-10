def run_agent(agent, test_case):
    response = agent.query(test_case["prompt"])
    return {
        "pass": response == test_case["expected"],
        "score": levenshtein_score(response, test_case["expected"]),
        "lang": test_case["lang"]
    }
