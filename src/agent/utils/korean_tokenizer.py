"""
korean_tokenizer.py

Kiwi 기반 한국어 형태소 분석 공통 유틸리티.

- 단일 Kiwi 인스턴스를 모듈 레벨 singleton으로 유지하여 초기화 비용(1-2초)을 프로세스당 1회로 한정.
- BM25용 토크나이저: NNG/NNP/SH/SL/SN/XR 품사 필터링 + 법률 boilerplate stopwords 제거.
- polarity 감지: regex 우선순위 규칙 (OX 연결형 → 부정 → 긍정 → NEUTRAL).
- core_focus_query 토큰 추출: 형태소 분석 후 핵심 법률 명사만 추출.

외부 import 예시:
    from agent.utils.korean_tokenizer import tokenize_korean, detect_polarity
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kiwipiepy import Kiwi

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Kiwi Singleton
# ─────────────────────────────────────────────────────────────────────────────

_kiwi_instance: Kiwi | None = None


def _get_kiwi_model_path() -> str | None:
    """kiwipiepy_model 패키지의 모델 디렉토리 경로를 반환합니다.

    Windows 환경에서 경로에 한국어 등 비ASCII 문자가 있으면 Kiwi의 C++ 백엔드가
    파일을 열지 못합니다. 이 경우 Windows 8.3 단축 경로(GetShortPathNameW)로
    변환하여 반환합니다.
    """
    try:
        import kiwipiepy_model

        model_dir = os.path.dirname(kiwipiepy_model.__file__)

        # Windows 환경에서만 8.3 경로 변환 시도
        if sys.platform == "win32":
            import ctypes

            buf = ctypes.create_unicode_buffer(1024)
            ctypes.windll.kernel32.GetShortPathNameW(model_dir, buf, 1024)  # type: ignore[attr-defined]
            short = buf.value
            if short:
                return short

        return model_dir
    except Exception:
        return None


def get_kiwi() -> Kiwi:
    """Kiwi 인스턴스를 반환합니다 (최초 호출 시 초기화 및 사용자 사전 등록)."""
    global _kiwi_instance
    if _kiwi_instance is None:
        from kiwipiepy import Kiwi

        log.info("[KoreanTokenizer] Kiwi 초기화 중...")
        model_path = _get_kiwi_model_path()
        _kiwi_instance = Kiwi(model_path=model_path) if model_path else Kiwi()
        _register_legal_compounds(_kiwi_instance)
        log.info("[KoreanTokenizer] Kiwi 초기화 완료.")
    return _kiwi_instance


def _register_legal_compounds(kiwi: Kiwi) -> None:
    """법률 복합명사를 사용자 사전에 등록하여 분절 방지."""
    compounds = [
        # 민법 / 채권 / 물권
        "불법행위",
        "손해배상",
        "손해배상청구권",
        "손해배상책임",
        "근저당권",
        "채권자취소권",
        "선의취득",
        "채무불이행",
        "계약해제",
        "계약해지",
        "소멸시효",
        "취득시효",
        "공동불법행위",
        "부당이득",
        "사무관리",
        "임의대리",
        "법정대리",
        "표현대리",
        "무권대리",
        "연대보증",
        "물상보증",
        "부당이득반환청구권",

        # 형사법 / 형사소송
        "구성요건",
        "위법성조각사유",
        "책임조각사유",
        "공동정범",
        "교사범",
        "방조범",
        "공범관계",
        "범죄성립",
        "증거능력",
        "위법수집증거",
        "자백배제법칙",
        "형사소송법",
        "형사소송법상",
        "민사소송법",
        "국선변호인",
        "피고인",
        "피해자",
        "변호인",

        # 행정법 / 헌법
        "재량행위",
        "기속행위",
        "취소소송",
        "원고적격",
        "행정행위",
        "행정처분",
        "행정소송법",
        "행정심판",
        "행정심판법",
        "행정절차법",
        "행정기본법",
        "행정대집행",
        "행정대집행법",
        "행정상즉시강제",
        "행정조사기본법",
        "위임입법",
        "기본권",
        "적법절차",
        "신체자유",
        "법치행정",
        "헌법재판소",

        # 선거 / 헌정기관
        "공직선거법",
        "공직선거법상",
        "선거관리위원회",
        "중앙선거관리위원회",
        "선거구획정위원회",
        "국회의원",
        "비례대표",
        "비례대표국회의원",
        "선거운동",

        # 경찰 / 경비 / 청원경찰
        "민간경비",
        "민간경비원",
        "경비업",
        "경비업무",
        "경비업자",
        "경비업법",
        "경비업법상",
        "경비업법령",
        "경비업법령상",
        "경비지도사",
        "일반경비원",
        "특수경비원",
        "국가중요시설",
        "청원경찰",
        "청원경찰법",
        "청원경찰법상",
        "청원경찰법령",
        "청원경찰법령상",
        "경찰공무원",
        "경찰청장",
        "지방경찰청장",

        # 소방 / 안전
        "소방기본법",
        "소방기본법상",
        "소방본부장",
        "소방시설",
        "소방시설공사업법",
        "위험물안전관리법",
        "화재예방",
        "안전관리",

        # 노동 / 일반 법령
        "근로기준법",
        "근로기준법상",
        "최저임금법",
        "노동조합및노동관계조정법",
        "노동관계조정법",
        "국가배상법",
        "국가공무원법",
        "지방자치법",
        "출입국관리법",
        "사회보장기본법",
        "변호사법",
        "공공기관의정보공개에관한법률",
        "수용자의처우에관한법률",
        "형의집행및수용자의처우에관한법률",
        "공익사업을위한토지등의취득및보상에관한법률",
    ]
    for word in compounds:
        with contextlib.suppress(Exception):
            kiwi.add_user_word(word, "NNG", 0.0)  # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────────────────────
# Statute Matching Constants (Stage 7 고정 데이터)
# ─────────────────────────────────────────────────────────────────────────────

STATUTE_WHITELIST: frozenset[str] = frozenset({
    "헌법", "민법", "형법", "형사소송법", "민사소송법", "상법", "행정소송법",
    "행정절차법", "행정심판법", "행정대집행법", "행정기본법", "행정조사기본법",
    "공직선거법", "청원경찰법", "경비업법", "경찰관직무집행법", "경찰공무원법",
    "소방기본법", "소방시설공사업법", "위험물안전관리법", "근로기준법", "최저임금법",
    "노동관계조정법", "국가배상법", "국가공무원법", "지방자치법", "고용보험법",
    "산업재해보상보험법", "사회보장기본법", "변호사법", "소년법", "국제사법",
    "건축법", "출입국관리법", "전자정부법", "국가재정법", "도로교통법", "정당법",
    "국민연금법", "국민기초생활보장법", "지방공무원법", "공직자윤리법", "국적법",
    "군인사법", "소방공무원법", "외국법자문사법", "개인정보보호법", "도시및주거환경정비법",
    "의료법", "법원조직법", "병역법", "식품위생법", "공무원연금법", "문화재보호법",
    "관세법", "약사법", "교통안전법", "하천법", "주차장법", "국유재산법", "감사원법",
    "통신비밀보호법", "질서위반행위규제법", "사회복지사업법", "고등교육법", "유아교육법",
    "초중등교육법", "영유아보육법", "청소년보호법", "청소년활동진흥법", "선박법",
    "국세징수법", "도시계획법", "성폭력범죄의처벌등에관한특례법", "공유토지분할에관한특례법",
    "전통시장및상점가육성을위한특별법", "지뢰피해자지원에관한특별법", "국세기본법",
    "특정범죄 가중처벌 등에 관한 법률", "특정경제범죄 가중처벌 등에 관한 법률",
    "폭력행위 등 처벌에 관한 법률", "부정수표 단속법", "성매매알선 등 행위의 처벌에 관한 법률",
    "아동·청소년의 성보호에 관한 법률", "가족관계의 등록 등에 관한 법률",
})

STATUTE_ALIAS_TO_CANONICAL: dict[str, str] = {
    "특가법": "특정범죄 가중처벌 등에 관한 법률",
    "특경법": "특정경제범죄 가중처벌 등에 관한 법률",
    "폭처법": "폭력행위 등 처벌에 관한 법률",
    "성매매처벌법": "성매매알선 등 행위의 처벌에 관한 법률",
    "아청법": "아동·청소년의 성보호에 관한 법률",
    "공직선거법상": "공직선거법", "경비업법상": "경비업법", "경비업법령상": "경비업법",
    "청원경찰법상": "청원경찰법", "청원경찰법령상": "청원경찰법", "근로기준법상": "근로기준법",
    "행정심판법상": "행정심판법", "행정절차법상": "행정절차법", "행정소송법상": "행정소송법",
    "행정대집행법상": "행정대집행법", "행정기본법상": "행정기본법", "행정조사기본법상": "행정조사기본법",
    "국가배상법상": "국가배상법", "국가공무원법상": "국가공무원법", "지방자치법상": "지방자치법",
    "고용보험법령상": "고용보험법", "사회보장기본법상": "사회보장기본법", "최저임금법상": "최저임금법",
    "소년법상": "소년법", "변호사법상": "변호사법", "국민연금법상": "국민연금법",
    "지방공무원법상": "지방공무원법", "국적법상": "국적법", "군인사법상": "군인사법",
    "소방공무원법상": "소방공무원법", "의료법상": "의료법", "출입국관리법상": "출입국관리법",
    "도로교통법상": "도로교통법", "건축법상": "건축법",
}

NON_STATUTE_LEGAL_TERMS: frozenset[str] = frozenset({
    "불법행위", "손해배상", "손해배상청구권", "소멸시효", "재량행위", "기속행위",
    "취소소송", "행정행위", "행정처분", "행정법", "형사법", "국제법", "공법", "사법",
    "특별법", "일반법", "성문법", "관습법", "실체법", "절차법", "실정법", "하위법",
    "모법", "신법", "구법", "조약법", "해양법", "국내법", "외국법", "국제관습법",
    "법률시행령", "직무집행법", "위임입법", "행정입법", "죄형법정주의",
})

# 매칭용 정규화 맵 및 리스트
# 공백이 제거된 키 -> 원본 키(또는 Canonical)
_STATUTE_NORM_TO_CANONICAL: dict[str, str] = {}
for s in STATUTE_WHITELIST:
    _STATUTE_NORM_TO_CANONICAL[s.replace(" ", "")] = s
for alias, canonical in STATUTE_ALIAS_TO_CANONICAL.items():
    _STATUTE_NORM_TO_CANONICAL[alias.replace(" ", "")] = canonical

# 정렬된 정규화 키 리스트 (longest matching용)
_SORTED_NORM_KEYS = sorted(_STATUTE_NORM_TO_CANONICAL.keys(), key=len, reverse=True)

# ─────────────────────────────────────────────────────────────────────────────
# 품사 필터 & Stopwords
# ─────────────────────────────────────────────────────────────────────────────

# BM25 content word로 유지할 품사 (str 비교)
_KEEP_POS: frozenset[str] = frozenset(
    {
        "NNG",  # 일반명사
        "NNP",  # 고유명사
        "SH",   # 한자
        "SL",   # 외국어
        "SN",   # 숫자
        "XR",   # 어근
    }
)

# 법률 문제 boilerplate — NNG이지만 BM25/core_focus에서 제거
_LEGAL_STOPWORDS: frozenset[str] = frozenset(
    {
        # 문제 boilerplate
        "설명",
        "내용",
        "사항",
        "경우",
        "것",
        "수",
        "바",
        "데",
        "중",
        "다음",
        "관련",
        "보기",
        "연결",

        # 지시/범용어
        "이",
        "그",
        "저",
        "해당",
        "각",
        "모든",
        "여러",

        # 판례/출처 boilerplate
        "판례",
        "입장",
        "근거",
        "이유",
        "다툼",

        # 법률 framing boilerplate
        "관한",
        "대한",
        "따른",
        "의한",
        "관하여",
        "대하여",

        # 범용 한정 표현
        "이상",
        "이하",
    }
)


# ─────────────────────────────────────────────────────────────────────────────
# 핵심 토크나이징 함수
# ─────────────────────────────────────────────────────────────────────────────


def tokenize_korean(text: str) -> list[str]:
    """
    한국어 텍스트를 Kiwi로 형태소 분석하여 BM25 content 토큰 리스트를 반환합니다.

    필터 기준:
        - 보존: NNG, NNP, NNB, SH, SL, SN, XR
        - 제거: 조사(JK*), 어미(E*), 접사(X*), 기호(S* except SH/SL/SN), 부사, 동사, 형용사
        - LEGAL_STOPWORDS에 해당하는 토큰 추가 제거

    Args:
        text: 분석할 한국어 텍스트

    Returns:
        필터링된 토큰 리스트 (소문자화 없음 — 한국어 대소문자 없음)
    """
    if not text or not text.strip():
        return []

    kiwi = get_kiwi()
    try:
        result = kiwi.analyze(text)
        if not result:
            return []

        tokens_raw = result[0][0]  # 최고 score 분석 결과의 Token 리스트
    except Exception as e:
        log.warning(f"[KoreanTokenizer] Kiwi 분석 실패, 공백 split fallback: {e}")
        return _fallback_tokenize(text)

    tokens: list[str] = []
    for token in tokens_raw:
        pos_str = str(token.tag)
        if pos_str not in _KEEP_POS:
            continue
        form = token.form.strip()
        if not form:
            continue
        if form in _LEGAL_STOPWORDS:
            continue
        if len(form) < 2 and pos_str in {"NNB", "XR"}:  # 단글자 의존명사/어근 제거
            continue
        tokens.append(form)

    return tokens


def _fallback_tokenize(text: str) -> list[str]:
    """Kiwi 실패 시 공백 split + 조사 제거 fallback."""
    korean_endings = re.compile(r"[은는이가을를의에서도로까지와과]$")
    tokens = []
    for tok in text.split():
        tok = tok.strip("?!.,()[]「」『』·")
        if not tok:
            continue
        cleaned = korean_endings.sub("", tok)
        if cleaned and len(cleaned) >= 2:
            tokens.append(cleaned)
    return tokens


# ─────────────────────────────────────────────────────────────────────────────
# Polarity 감지
# ─────────────────────────────────────────────────────────────────────────────

# OX 연결형 ("옳은 것(○)과 옳지 않은 것(×)을 바르게 연결") → MIXED
_OX_PATTERN = re.compile(r"옳은.{0,10}옳지\s*않은")

# 부정 패턴 (특이도 높은 순, 말미 검사 후 전체 검사)
_NEG_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p)
    for p in [
        r"옳지\s*않은(?:\s*것)?(?:을)?(?:\s*모두)?",
        r"해당하지\s*않는(?:\s*것)?",
        r"해당하지\s*아니하는(?:\s*것)?",
        r"속하지\s*않는(?:\s*것)?",
        r"포함되지\s*않는(?:\s*것)?",
        r"아닌\s*것",
        r"없는\s*것",
        r"틀린(?:\s*것)?",
        r"잘못(?:된|된\s*것|된\s*설명)",
        r"적절하지\s*않은(?:\s*것)?",
        r"적절하지\s*못한(?:\s*것)?",
        r"맞지\s*않는(?:\s*것)?",
        r"올바르지\s*않은(?:\s*것)?",
        r"바르지\s*않은(?:\s*것)?",
        r"타당하지\s*않은(?:\s*것)?",
        r"인정되지\s*않는",
        r"성립하지\s*않는",
        r"허용되지\s*않는",
        r"다른\s*것",
        r"거리가\s*먼\s*것",
    ]
]

# 긍정 패턴
_POS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p)
    for p in [
        r"옳은(?:\s*것)?(?:만)?(?:을)?(?:\s*모두)?",
        r"해당하는(?:\s*것)?",
        r"맞는(?:\s*것)?",
        r"바르게\s*연결한(?:\s*것)?",
        r"타당한(?:\s*것)?",
        r"적절한(?:\s*것)?",
        r"적용되는(?:\s*것)?",
        r"할\s*수\s*있는(?:\s*것)?",
        r"수\s*있는(?:\s*것)?",
        r"인정되는",
        r"성립하는",
        r"허용되는",
    ]
]

# polarity 결과 상수
POLARITY_NEG = "부정방향"
POLARITY_POS = "긍정방향"
POLARITY_NEUTRAL = "중립"
POLARITY_MIXED = "중립"  # OX 연결형 → 검색용으로 중립 처리


def detect_polarity(question: str) -> str:
    """
    질문 텍스트에서 부정/긍정/중립 방향을 감지합니다.

    우선순위:
      1. OX 연결형 (옳은 것○ / 옳지 않은 것× 대응) → 중립
      2. 부정 패턴: 말미 40자 우선, 없으면 전체 검사
      3. 긍정 패턴: 말미 40자 우선, 없으면 전체 검사
      4. 나머지 → 중립

    Args:
        question: 질문 텍스트

    Returns:
        POLARITY_NEG | POLARITY_POS | POLARITY_NEUTRAL
    """
    if not question:
        return POLARITY_NEUTRAL

    # 1. OX 연결형 선감지
    if _OX_PATTERN.search(question):
        return POLARITY_MIXED

    ending = question[-40:]

    # 2. 부정 패턴 (말미 → 전체)
    for pat in _NEG_PATTERNS:
        if pat.search(ending):
            return POLARITY_NEG
    for pat in _NEG_PATTERNS:
        if pat.search(question):
            return POLARITY_NEG

    # 3. 긍정 패턴 (말미 → 전체)
    for pat in _POS_PATTERNS:
        if pat.search(ending):
            return POLARITY_POS
    for pat in _POS_PATTERNS:
        if pat.search(question):
            return POLARITY_POS

    return POLARITY_NEUTRAL


# ─────────────────────────────────────────────────────────────────────────────
# core_focus_query 핵심 토큰 추출
# ─────────────────────────────────────────────────────────────────────────────


def extract_core_tokens(question: str) -> list[str]:
    """
    질문 텍스트에서 법률 핵심 명사 토큰만 추출합니다 (core_focus_query 용도).

    tokenize_korean()과 동일한 필터를 적용하되, NNB(의존명사)는 완전 제거합니다.
    LEGAL_STOPWORDS를 적용하여 문제 boilerplate("설명", "내용" 등)를 제거합니다.

    Args:
        question: 질문 텍스트

    Returns:
        핵심 명사 토큰 리스트 (중복 제거, 순서 보존)
    """
    if not question or not question.strip():
        return []

    kiwi = get_kiwi()
    try:
        result = kiwi.analyze(question)
        if not result:
            return []
        tokens_raw = result[0][0]
    except Exception as e:
        log.warning(f"[KoreanTokenizer] extract_core_tokens Kiwi 실패: {e}")
        return _fallback_tokenize(question)

    # core_focus: NNB 완전 제거 (의존명사 "것", "수" 등 boilerplate 원천 차단)
    _core_keep: frozenset[str] = frozenset({"NNG", "NNP", "SH", "SL", "SN"})

    seen: set[str] = set()
    tokens: list[str] = []
    for token in tokens_raw:
        pos_str = str(token.tag)
        if pos_str not in _core_keep:
            continue
        form = token.form.strip()
        if not form or len(form) < 2:
            continue
        if form in _LEGAL_STOPWORDS:
            continue
        if form in seen:
            continue
        seen.add(form)
        tokens.append(form)

    return tokens


def extract_statute_names(text: str) -> set[str]:
    """
    텍스트에서 법령명을 추출하여 정규화된 셋으로 반환합니다.

    규칙:
    1. 텍스트 normalization (공백 제거, 특수기호 통일, 괄호 제거 등)
    2. STATUTE_ALIAS_TO_CANONICAL longest-match 매칭
    3. STATUTE_WHITELIST longest-match 매칭
    4. NON_STATUTE_LEGAL_TERMS 제외

    Args:
        text: 분석할 텍스트

    Returns:
        정규화된 법령명 집합
    """
    if not text or not text.strip():
        return set()

    # 1. Normalization
    normalized = text.replace(" ", "")
    normalized = normalized.replace("·", "").replace("⋅", "").replace("･", "")
    normalized = normalized.replace('"', "").replace("'", "").replace("“", "").replace("”", "")
    normalized = normalized.replace("(", "").replace(")", "").replace("[", "").replace("]", "")

    # 2. Matching (Longest Match using normalized keys)
    target = normalized
    found_canonicals: set[str] = set()

    for norm_key in _SORTED_NORM_KEYS:
        if norm_key in target:
            canonical = _STATUTE_NORM_TO_CANONICAL[norm_key]
            found_canonicals.add(canonical)
            # 매칭된 부분은 제거하여 중복 매칭 방지
            target = target.replace(norm_key, " ")

    # 3. Exclusion of NON_STATUTE_LEGAL_TERMS
    final_statutes = {s for s in found_canonicals if s not in NON_STATUTE_LEGAL_TERMS}

    return final_statutes
