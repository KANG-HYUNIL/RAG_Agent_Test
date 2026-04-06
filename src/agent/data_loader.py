import pandas as pd
from typing import List, Dict

class DataLoader:
    """
    data/train.csv, data/dev.csv 등 CSV 데이터를 로드하여 임베딩 및 추론용 
    데이터를 딕셔너리 리스트 형태로 변환하는 역할을 수행합니다.
    """

    def load_csv(self, file_path: str) -> List[Dict[str, str]]:
        """
        지정된 경로의 CSV 파일을 로드하여 반환합니다.
        결측치가 있다면 빈 문자열로 처리하고, 각 행을 Dictionary 형태로 반환합니다.
        
        Args:
            file_path: 읽어들일 csv 파일 경로 (예: "data/train.csv")
            
        Returns:
            List[Dict]: 행 단위로 딕셔너리화된 리스트 
        """
        try:
            df = pd.read_csv(file_path)
            df = df.fillna("")  # NaN 값 방지
            return df.to_dict(orient="records")
        except Exception as e:
            raise RuntimeError(f"Failed to load CSV from {file_path}: {e}")
