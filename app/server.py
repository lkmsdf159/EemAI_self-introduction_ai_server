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
    
    try:
        body = await request.body()
        if body:
            logger.info(f"요청 본문: {body.decode()}")
    except Exception as e:
        logger.error(f"본문 읽기 오류: {e}")
    
    response = await call_next(request)
    return response


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/prompt/playground")


def sanitize_filename(filename: str) -> str:
   import re
   sanitized = re.sub(r'[^\w\s가-힣]', '_', filename)
   sanitized = re.sub(r'_+', '_', sanitized)
   sanitized = sanitized.strip('_')
   return sanitized


@app.post("/gemma")
async def process_emp_requests(request: Request):
    try:

        data = await request.json()
        

        job_code = data.get("job_code")
        if not job_code:
            raise HTTPException(status_code=400, detail="Missing jobtitle in request")
            
 
        filename = sanitize_filename(job_code)
        filepath = os.path.join("/mnt/e/Linkareer_embedding_data/", f"{filename}.csv")
        reference_data = load_reference_data(filepath)

        data = await request.json()
        answers_list = data.get("data", [])
        results = []
        
        for answer in answers_list:
            try:
                question = answer.get("question", "")
                text = answer.get("answer", "")
                

                if len(text.strip()) < 100:
                    results.append({
                        "relevance": 0,
                        "specificity": 0,
                        "persuasiveness": 0,
                        "feedback": "답변이 너무 짧습니다. 성의있게 작성해 주세요."
                    })
                    continue
                
  
                formatted_answer = {
                    "question": question,
                    "text": text,
                    job_code : job_code
                }
                
                result = await process_gemma_answer(formatted_answer, job_code, reference_data)
                results.append(result)
                
            except Exception as e:
                results.append({
                    "relevance": 0,
                    "specificity": 0,
                    "persuasiveness": 0,
                    "feedback": "평가를 완료할 수 없었습니다."
                })
        
        response = {"results": results}
        return response
        
    except Exception as e:
        return {"error": "요청 처리 중 오류가 발생했습니다"}
    
    finally:
        EmbeddingProcessor().cleanup()

@app.get("/gemma3")
async def health_check():
    logger.info(f"===health_check===")
    return {"status": "ok"}

    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)