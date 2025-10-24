# 의료 데이터 임베딩 및 벡터 데이터베이스 프로젝트

이 프로젝트는 병원별 환자 차트 데이터를 OpenAI 임베딩을 사용하여 벡터화하고, Qdrant 벡터 데이터베이스에 저장하는 시스템입니다.

## 0. 사전 준비사항

### 필수 설치 항목

1. **Python 3.12**
   ```bash
   # Python 버전 확인
   python --version
   ```

2. **uv 패키지 매니저 설치**
   ```bash
   # macOS (Homebrew 사용)
   brew install uv
   
   # 또는 pip 사용
   pip install uv
   ```

3. **Qdrant 벡터 데이터베이스 설치**
   ```bash
   # Docker를 사용한 설치 (권장)
   docker run -p 6333:6333 qdrant/qdrant
   
   # 또는 로컬 설치
   # https://qdrant.tech/documentation/guides/installation/ 참조
   ```

### 환경 변수 설정

프로젝트 루트에 `.env` 파일을 생성하고 다음 내용을 추가하세요:

```env
OPENAI_API_KEY=your_openai_api_key_here
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

## 1. 프로젝트 시작하기

### 1.1 저장소 클론 및 의존성 설치

```bash
# 저장소 클론
git clone <repository-url>
cd ds-202501-health-econ-mlflow

# uv를 사용한 의존성 설치
uv sync

# 가상환경 활성화
source .venv/bin/activate
```

### 1.2 데이터 준비

병원별 데이터를 `data/external/` 디렉토리에 배치하세요:
```
data/
└── external/
    ├── epurun/
    │   └── patient_epurun_20251009_011300.csv
    └── hana_ent/
        └── [병원 데이터 파일]
```

### 1.3 실행

```bash
# 이푸른병원 데이터 처리
python main__epurun.py

# 하나이비인후과병원 데이터 처리
python main__hana_ent.py
```

## 2. 새로운 병원을 위한 main__*.py 파일 생성 방법

### 2.1 epurun.py 예시를 통한 프로세스 설명

`main__epurun.py` 파일은 다음과 같은 단계로 구성됩니다:

#### 단계 1: 병원 프로필 생성
```python
from utils.auth.hospital_profile import HospitalProfile

hosp_epurun = HospitalProfile(
    hospital_id="epurun", 
    hospital_name="이푸른병원"
)
```

#### 단계 2: 데이터 로드
```python
from models.data_container.plugin.patient_epurun import load

df = load(root_dir)
```

#### 단계 3: 임베딩 생성 및 벡터 데이터베이스 저장
```python
from models.embed.openai_embedding import PatientChartEmbedding

emb = PatientChartEmbedding(profile=hosp_epurun)
emb.initialize_qdrant(
    host=os.getenv("QDRANT_HOST"),
    port=os.getenv("QDRANT_PORT")
)
emb.create_patient_document_by_date(df)
```

### 2.2 새로운 병원을 위한 파일 생성

새로운 병원 (예: "서울병원")을 추가하려면:

#### 1) main__seoul_hospital.py 생성
```python
from utils.auth.hospital_profile import HospitalProfile
from models.embed.openai_embedding import PatientChartEmbedding
from models.data_container.plugin.patient_seoul_hospital import load

if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    load_dotenv()
    root_dir = os.path.abspath(".")

    # 병원 프로필 생성
    print("[Run] Create hospital profile (seoul_hospital)")
    hosp_seoul = HospitalProfile(
        hospital_id="seoul_hospital", 
        hospital_name="서울병원"
    )

    print("[Run] Load patient data")
    df = load(root_dir)

    print("[Run] Create embedding ...")
    emb = PatientChartEmbedding(profile=hosp_seoul)
    emb.initialize_qdrant(
        host=os.getenv("QDRANT_HOST"),
        port=os.getenv("QDRANT_PORT")
    )
    emb.create_patient_document_by_date(df)
    print("[Run] Embedding completed")
```

#### 2) Plugin의 load 함수 생성

`models/container/plugin/patient_seoul_hospital.py` 파일을 생성하고 다음 구조를 따르세요:

```python
from pathlib import Path
import os
import pandas as pd

def load(root_dir: Path) -> pd.DataFrame:
    """
    병원별 데이터를 로드하고 표준화된 형태로 반환하는 함수
    
    반환되는 DataFrame은 다음 컬럼들을 포함해야 합니다:
    - id: 환자 ID (int)
    - date: 진료 날짜 (datetime)
    - sex: 성별 (0: 여성, 1: 남성)
    - age: 나이 (int)
    - department: 진료과 (str)
    - primary_diagnosis: 주진단 (str)
    - secondary_diagnosis: 부진단 (str)
    - prescription: 처방 (str)
    """
    
    # 1. 데이터 파일 경로 설정
    fn = "seoul_hospital_data.csv"  # 실제 파일명으로 변경
    fp = os.path.join(root_dir, "data", "external", "seoul_hospital", fn)
    
    # 2. 데이터 로드
    patient = pd.read_csv(fp, low_memory=False)
    
    # 3. 데이터 전처리 및 정제
    # 병원별 데이터 구조에 맞게 전처리 로직 구현
    
    # 4. 컬럼명 표준화
    patient = patient.rename({
        "환자번호": "id",
        "진료일": "date",
        "성별": "sex",
        "나이": "age",
        "진료과": "department",
        "주진단": "primary_diagnosis",
        "부진단": "secondary_diagnosis",
        "처방": "prescription"
    }, axis=1)
    
    # 5. 데이터 타입 변환
    patient['id'] = patient['id'].astype(int)
    patient['date'] = pd.to_datetime(patient['date'])
    patient['age'] = patient['age'].astype(int)
    patient['sex'] = patient['sex'].astype(int)  # 0: 여성, 1: 남성
    
    # 6. 필수 컬럼만 선택
    patient = patient[[
        'id', 'date', 'sex', 'age', 'department', 
        'primary_diagnosis', 'secondary_diagnosis', 'prescription'
    ]]
    
    return patient
```

### 2.3 Plugin Load 함수 구현 시 주의사항

1. **필수 컬럼**: 반환되는 DataFrame은 반드시 다음 8개 컬럼을 포함해야 합니다:
   - `id`, `date`, `sex`, `age`, `department`, `primary_diagnosis`, `secondary_diagnosis`, `prescription`

2. **데이터 타입**: 
   - `id`: int
   - `date`: datetime
   - `sex`: int (0: 여성, 1: 남성)
   - `age`: int
   - 나머지: str

3. **데이터 정제**: 
   - 결측값 처리
   - 잘못된 데이터 제거
   - 중복 데이터 처리

4. **성능 최적화**:
   - `low_memory=False` 옵션 사용
   - 불필요한 데이터 조기 필터링

## 3. 프로젝트 구조

```
├── main__*.py              # 병원별 실행 파일
├── models/
│   ├── embed/
│   │   └── openai_embedding.py    # OpenAI 임베딩 처리
│   └── container/
│       └── plugin/
│           └── patient_*.py       # 병원별 데이터 로더
├── utils/
│   └── auth/
│       └── hospital_profile.py    # 병원 프로필 관리
├── data/
│   └── external/           # 병원별 원본 데이터
└── mlruns/                # MLflow 실험 추적 데이터
```

## 4. 기술 스택

- **Python 3.12**: 메인 프로그래밍 언어
- **OpenAI API**: 텍스트 임베딩 생성
- **Qdrant**: 벡터 데이터베이스
- **MLflow**: 실험 추적 및 모델 관리
- **Pandas**: 데이터 처리
- **uv**: 패키지 관리

## 5. 문제 해결

### 일반적인 오류

1. **OpenAI API 키 오류**: `.env` 파일에 올바른 API 키가 설정되어 있는지 확인
2. **Qdrant 연결 오류**: Qdrant 서버가 실행 중인지 확인
3. **데이터 파일 없음**: `data/external/` 경로에 올바른 데이터 파일이 있는지 확인

### 로그 확인

MLflow UI를 통해 실행 로그와 메트릭을 확인할 수 있습니다:
```bash
mlflow ui
```

브라우저에서 `http://localhost:5000`으로 접속하여 실험 결과를 확인하세요.