📅 2026-05-06 RAG 전처리 실습 (LangChain)
📌 오늘 목표
LangChain을 활용한 RAG 전처리 과정 이해
텍스트를 임베딩하고 유사도 기반으로 문서 검색 구현
🧠 핵심 개념 정리
1. Text Split (문서 분할)
긴 문서를 chunk 단위로 나누는 과정
RecursiveCharacterTextSplitter 사용
의미 단위로 나누는 것이 중요 (문단/문장 기준)
2. Embedding (임베딩)
텍스트를 벡터(숫자)로 변환
같은 의미 → 비슷한 벡터
모델: text-embedding-3-small
3. Query Embedding
질문도 동일한 방식으로 벡터화
embed_query() 사용
4. Cosine Similarity (유사도 계산)
질문 벡터 vs 문서 벡터 비교
값이 클수록 의미적으로 유사
⚙️ 구현 과정
텍스트를 chunk로 분할
각 chunk를 임베딩
질문을 임베딩
cosine similarity로 유사도 계산
가장 유사한 문서 선택
🧪 주요 코드
# 임베딩 생성
embeddings = embeddings_model.embed_documents(texts)

# 쿼리 임베딩
embedded_query = embeddings_model.embed_query(query)

# 유사도 계산
cosine_similarities = cosine_similarity(
    embeddings_np,
    embedded_query_np
).flatten()

# 가장 유사한 문서 찾기
best_idx = np.argmax(cosine_similarities)
