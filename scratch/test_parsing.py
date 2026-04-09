import re
import sys
import io

# Windows 환경에서 한국어 인코딩 문제 해결
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def parse_query_text(query: str) -> tuple[str, str, dict]:
    # 1. 카테고리 추출
    category = "Unknown"
    cat_match = re.search(r"^\[(.*?)\]", query)
    if cat_match:
        category = cat_match.group(1).strip()
        query = query[cat_match.end():].strip()
    
    # 2. 질문 분리
    parts = query.split("?", 1)
    question = parts[0] + "?" if len(parts) > 1 else parts[0]
    remaining = parts[1] if len(parts) > 1 else ""
    
    choices = {"A": "", "B": "", "C": "", "D": ""}
    
    # 3. 선지 추출
    pattern = re.compile(
        r"(?:[1-4A-D][\.\)\s:])\s*(.*?)(?=\s*(?:[1-4A-D][\.\)\s:])|$)", 
        re.DOTALL
    )
    matches = pattern.findall(remaining)
    
    if len(matches) >= 4:
        choices["A"], choices["B"], choices["C"], choices["D"] = [m.strip() for m in matches[:4]]
    
    return category, question.strip(), choices

def parse_prediction_label(text: str) -> str:
    """LLM 답변에서 정답 레이블(A, B, C, D 또는 1, 2, 3, 4)을 추출합니다."""
    import re
    if not text:
        return "ERR"

    # 1. 알파벳 레이블 (A, B, C, D) 매칭 (한글 조사 및 문장 부호 대응)
    match = re.search(r"([A-D])(?:\s|번|인|입|:|\)|\.|$)", text.upper())
    if match:
        return match.group(1)

    # 2. 숫자 레이블 (1, 2, 3, 4) 매칭 -> A, B, C, D로 변환
    num_match = re.search(r"([1-4])(?:\s|번|인|입|:|\)|\.|$)", text)
    if num_match:
        mapping = {"1": "A", "2": "B", "3": "C", "4": "D"}
        return mapping[num_match.group(1)]

    return "ERR"

# Test Cases - Query Parsing
print("--- Testing Query Parsing ---")
test_queries = [
    "[Law] 형벌 유형 중 재산형에 포함되지 않는 것은? 1. 금고, 2. 벌금, 3. 과료, 4. 몰수",
    "[Criminal Law] 사회해체지역에서 범죄율이 높은 이유는? 1. 낮은 빈곤율, 2. 높은 취업률, 3. 낮은 결손가정률, 4. 높은 인구이동률",
]

for q in test_queries:
    cat, ques, chs = parse_query_text(q)
    print(f"Query: {q}")
    print(f"  Parsed Category: {cat}")
    print(f"  Parsed Question: {ques}")
    print(f"  Parsed Choices: {chs}")
    assert cat in ["Law", "Criminal Law"]
    assert ques.endswith("?")
    assert all(chs.values())
    print("  => OK")

# Test Cases - Prediction Parsing
print("\n--- Testing Prediction Parsing ---")
test_preds = [
    ("정답은 A입니다.", "A"),
    ("The correct answer is B.", "B"),
    ("C", "C"),
    ("D번이 확실합니다.", "D"),
    ("1", "A"),
    ("2번", "B"),
    ("정답: 3", "C"),
    ("답은 4번", "D"),
    ("No idea", "ERR")
]

for p, expected in test_preds:
    parsed = parse_prediction_label(p)
    status = "OK" if parsed == expected else f"FAIL (Expected {expected})"
    print(f"Original: '{p}' -> Parsed: '{parsed}' [{status}]")

print("\nAll tests completed.")
