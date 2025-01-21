# server.py
from fastapi import FastAPI, Request, HTTPException  
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from gemma_emp import process_gemma_answer, load_reference_data, EmbeddingProcessor
import logging
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

logging.basicConfig(level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



@app.middleware("http")
async def log_middleware(request: Request, call_next):
    logger.info(f"=== 새로운 요청 ===")
    logger.info(f"요청 URL: {request.url}")
    logger.info(f"요청 메소드: {request.method}")
    
    try:
        body = await request.body()
        if body:
            logger.info(f"요청 본문: {body.decode()}")
    except Exception as e:
        logger.error(f"본문 읽기 오류: {e}")
    
    response = await call_next(request)
    
    logger.info(f"응답 상태 코드: {response.status_code}")
    return response


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/prompt/playground")


def sanitize_filename(filename: str) -> str:
   import re
   # 한글, 영문, 숫자를 제외한 모든 문자를 _로 변경
   sanitized = re.sub(r'[^\w\s가-힣]', '_', filename)
   # 연속된 _를 하나로
   sanitized = re.sub(r'_+', '_', sanitized)
   # 앞뒤 공백/언더스코어 제거
   sanitized = sanitized.strip('_')
   return sanitized


@app.post("/gemma")
async def process_emp_requests(request: Request):
    try:
        # 요청 데이터 파싱
        data = await request.json()
        
        # jobtitle 추출 및 파일명 생성
        job_code = data.get("job_code")
        if not job_code:
            raise HTTPException(status_code=400, detail="Missing jobtitle in request")
            
        # 파일명 생성 및 sanitize
        filename = sanitize_filename(job_code)
        filepath = os.path.join("/mnt/e/Linkareer_embedding_data/", f"{filename}.csv")
        logger.info(f"참조 데이터 파일 경로: {filepath}")
        reference_data = load_reference_data(filepath)
        logger.info(f"로드된 참조 데이터 수: {len(reference_data)}")

        data = await request.json()
        answers_list = data.get("data", [])
        results = []
        
        logger.info(f"요청 받음: {len(answers_list)}개의 답변")
        
        for answer in answers_list:
            try:
                question = answer.get("question", "")
                text = answer.get("answer", "")
                
                # 답변 길이 체크
                if len(text.strip()) < 100:
                    logger.warning(f"답변 길이 부족: {len(text.strip())}자")
                    results.append({
                        "relevance": 0,
                        "specificity": 0,
                        "persuasiveness": 0,
                        "feedback": "답변이 너무 짧습니다. 성의있게 작성해 주세요."
                    })
                    continue
                
                # 길이가 충분한 경우 정상 처리
                formatted_answer = {
                    "question": question,
                    "text": text,
                    job_code : job_code
                }
                
                logger.debug(f"처리할 데이터: {formatted_answer}")
                result = await process_gemma_answer(formatted_answer, job_code, reference_data)
                results.append(result)
                
            except Exception as e:
                logger.error(f"답변 처리 중 오류: {str(e)}")
                results.append({
                    "relevance": 5,
                    "specificity": 5,
                    "persuasiveness": 5,
                    "feedback": "평가를 완료할 수 없었습니다."
                })
        
        response = {"results": results}
        logger.info(f"응답 상태 코드: 200")
        return response
        
    except Exception as e:
        logger.error(f"요청 처리 중 오류: {str(e)}")
        return {"error": "요청 처리 중 오류가 발생했습니다"}
    
    finally:
        # 임베딩 모델 정리
        EmbeddingProcessor().cleanup()

@app.get("/gemma3")
async def health_check():
    logger.info(f"===health_check===")
    return {"status": "ok"}

    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)