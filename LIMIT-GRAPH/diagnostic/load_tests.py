def load_tests(lang):
    path = f"tests/{lang}/"
    return [json.load(open(f"{path}/{f}")) for f in os.listdir(path) if f.endswith(".json")]
