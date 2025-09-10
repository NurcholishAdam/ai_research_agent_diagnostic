#!/bin/bash
echo "🔍 Running multilingual agent tests..."
for lang in "id" "es" "ar"; do
  python eval_runner.py --lang $lang
done
