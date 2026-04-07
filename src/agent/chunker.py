from typing import Any


class Chunker:
    """
    데이터를 모델이 처리하기 좋은 단위(Chunk)로 나누는 역할을 수행합니다.
    현재 표 데이터(CSV)는 기본적으로 하나의 행(Row)을 독립된 청크로 사용하는 것이 가장 이상적입니다.

    DataLoader가 반환한 데이터를 받아, 향후 추가적인 텍스트 분할(Text Splitting) 혹은
    메타데이터 분리 작업이 필요해질 때를 대비한 파이프라인의 추상화 계층(Layer) 역할을 합니다.
    """

    def chunk_data(self, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
        """
        DataLoader에서 전달받은 row 리스트를 정규화된 Chunk 구조체 리스트로 변환합니다.

        Args:
            rows: DataLoader로부터 전달받은 원본 딕셔너리 리스트

        Returns:
            List[Dict]: "chunk_id", "content_dict" 등으로 구조화된 청크 리스트
        """
        chunks = []
        for idx, row in enumerate(rows):
            chunk = {
                "chunk_id": f"chunk_{idx}",
                "content_dict": row,  # Embedder가 사용할 row 데이터
            }
            chunks.append(chunk)

        return chunks
