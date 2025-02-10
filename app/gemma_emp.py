from langchain_community.document_loaders.csv_loader import CSVLoader
from transformers import AutoModel, AutoTokenizer
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
import numpy as np
import logging
import asyncio
import torch 
import json
import os
import re

ollama_semaphore = asyncio.Semaphore(10)

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

logger = logging.getLogger(__name__)



class ComparisonPoint(BaseModel):
    target_status: str = Field(description="평가 대상의 현재 상태", min_length=50)
    reference_gap: str = Field(description="합격자 사례와의 구체적인 차이", min_length=50)
    improvement: str = Field(description="개선을 위한 구체적 방안", min_length=50)

class QualitativeAnalysis(BaseModel):
    content_quality: ComparisonPoint = Field(description="내용의 질적 측면 비교")
    specificity: ComparisonPoint = Field(description="구체성과 사례 제시 측면 비교")
    persuasiveness: ComparisonPoint = Field(description="설득력과 논리성 측면 비교")

class FeedbackResult(BaseModel):
    relevance: int = Field(ge=1, le=10, description="질문과의 연관성 점수")
    specificity: int = Field(ge=1, le=10, description="구체성 점수")
    persuasiveness: int = Field(ge=1, le=10, description="설득력 점수")




class EmbeddingProcessor:
    _instance = None
    _model = None
    _tokenizer = None
    _batch_size = 128  # 배치 처리를 위한 크기 설정

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_model()
        return cls._instance
    
    def _init_model(self):
        """모델과 토크나이저를 초기화합니다."""
        if self._model is None:
            try:
                logger.info("임베딩 모델 초기화 시작")
                model_name = "BM-K/KoSimCSE-roberta"  # 사용할 모델 이름
                logger.info(f"사용할 모델: {model_name}")
                self._model = AutoModel.from_pretrained(model_name)
                self._tokenizer = AutoTokenizer.from_pretrained(model_name)
                self._cache = {}

                if torch.cuda.is_available():
                    self._model = self._model.cuda()
                    self._model = torch.nn.DataParallel(self._model)
                    self._model.eval()
            except Exception as e:
                logger.error(f"모델 초기화 중 에러: {str(e)}")
                raise

    def get_embedding(self, text: str) -> np.ndarray:
        logger.debug(f"텍스트 임베딩 시작 (길이: {len(text)})")
        if not hasattr(self, '_cache'):  
            self._cache = {}
        """입력 텍스트에 대한 임베딩을 반환합니다."""
        if text not in self._cache:  

            inputs = self._tokenizer(text, return_tensors="pt", truncation=True)
            if torch.cuda.is_available():
                inputs = {key: value.cuda() for key, value in inputs.items()}


            with torch.no_grad():
                outputs = self._model(**inputs)
            

            embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            self._cache[text] = embedding  # 캐싱

        return self._cache[text]

    def get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """배치 처리를 통한 임베딩 생성"""
        embeddings = []
        for i in range(0, len(texts), self._batch_size):
            batch_texts = texts[i:i + self._batch_size]
            inputs = self._tokenizer(batch_texts, padding=True, truncation=True, 
                                   return_tensors="pt", max_length=200)
            if torch.cuda.is_available():
                inputs = {key: value.cuda() for key, value in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)
            
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)

    def cleanup(self):
        """모델과 토크나이저, GPU 캐시를 정리합니다."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        EmbeddingProcessor._instance = None  # 이 부분 추가




#- 10점: {job_code} 직군의 핵심 역량/경험이 잘 드러나며, 질문 의도를 정확히 파악하여 추가 인사이트까지 제공
#- 8-9점: {job_code} 관련 내용이 대부분이며 질문의 핵심 요소를 포함하나, 일부 심화 내용 부족
#- 6-7점: {job_code} 관련 내용이 있으나 직무 연관성이 다소 부족하고 질문 의도를 부분적으로만 이해
#- 3-5점: {job_code} 관련 내용이 매우 적고 다른 직무 경험 위주로 서술됨
#- 1-2점: {job_code}와 전혀 무관한 다른 직군의 경험만을 서술하거나 질문 의도를 전혀 파악하지 못함


    


class EnhancedJSONParser:
    @staticmethod
    def validate_and_parse(text: str) -> Optional[Dict[str, Any]]:
        try:
            # 제어 문자 및 특수 문자 처리
            def clean_text(text):
                # 제어 문자 제거
                text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
                # 이스케이프 처리
                text = text.replace('\\', '\\\\')
                text = text.replace('\n', ' ')
                text = text.replace('\r', ' ') 
                text = text.replace('\t', ' ')
                text = text.replace('"', '\\"')
                text = text.replace('**', ' ')
                text = text.replace('*', ' ')
                text = text.replace('\\"', '"')
                text = text.replace('\\\\', ' ')  
                text = text.replace('<br> - ', ' ')
                text = text.replace(':', ' ')                
                text = text.replace('Relevance:', ' ')
                text = text.replace('Persuasiveness:', ' ')
                text = text.replace('Specificity:', ' ')
                text = text.replace('Relevance', ' ')
                text = text.replace('Persuasiveness', ' ')
                text = text.replace('Specificity', ' ')     
                text = text.replace('n n 참고 사례 :', ' ')                             
                text = ' '.join(text.split())
                return text

            # 기존 텍스트 정리
            text = text.split('```json')[-1].split('```')[0].strip()
            text = text.split('\n**Explanation:**')[0].strip()
            
            # JSON 문자열 정리
            for pattern in [r'\u0000-\u001F']:  # 제어 문자 제거
                text = re.sub(pattern, '', text)
            
            try:
                result = json.loads(text)
            except json.JSONDecodeError:
                # JSON 파싱 실패시 텍스트 추가 정리 후 재시도
                text = clean_text(text)
                result = json.loads(text)

            # 나머지 검증 로직
            required_fields = ['relevance', 'specificity', 'persuasiveness', 'feedback']
            for field in required_fields:
                if field not in result:
                    raise ValueError(f"Missing required field: {field}")
            
            score_fields = ['relevance', 'specificity', 'persuasiveness']
            for field in score_fields:
                score = result[field]
                if not (1 <= score <= 10):
                    result[field] = 5
                    
            if 'feedback' in result:
                result['feedback'] = clean_text(result['feedback'])
            
            if not result['feedback'] or len(result['feedback']) < 30:
                result['feedback'] = "답변 내용을 분석했습니다. 더 구체적인 피드백이 필요합니다."
            
            return result
            
        except Exception as e:
            return None


   
    def extract_info(text):
        info = {}
        for line in text.split('\n'):
            if 'h1 Tag:' in line:
                info['h1'] = line.split('h1 Tag:')[1].strip()
            elif 'h3 Tag:' in line:
                info['h3'] = line.split('h3 Tag:')[1].strip()
            elif 'Content:' in line:
                info['content'] = line.split('Content:')[1].strip()
            elif 'URL:' in line:
                info['url'] = line.split('URL:')[1].strip()
        return info

def load_reference_data(file_path: str) -> List[Dict[str, Any]]:
    """CSV 파일에서 참조 데이터 로드"""
    try:

        
        loader = CSVLoader(
            file_path,
            encoding='utf-8',
            csv_args={
                'delimiter': ',',
                'quotechar': '"',
                'skipinitialspace': True
            }
        )
        data = loader.load()

        reference_data = []
        for doc in data:
            try:
                content = doc.page_content
                
                # CSV 내용을 파싱
                row_data = {}
                for line in content.split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        row_data[key] = value

                # 필수 필드가 있는지 확인
                if 'quest' in row_data and 'pass_answer' in row_data:
                    reference_data.append({
                        'text': row_data['pass_answer'],  
                        'quest': row_data['quest'],
                        'pass_answer': row_data['pass_answer'],
                        'h2_tag': row_data.get('h2_tag', '')
                    })

            except Exception as e:
                continue
        if reference_data:
            logger.debug(f"첫 번째 데이터 샘플:{json.dumps(reference_data[0], ensure_ascii=False)}")
        
        return reference_data

    except Exception as e:
        return []


async def process_answer(answer: Dict[str, Any]) -> Dict[str, Any]:
    try:
        logger.info(f"답변 처리 시작 - 직무코드: {answer.get('job_code', 'Unknown')}") 
        logger.debug(f"입력 답변 길이: {len(answer.get('text', ''))}")
        logger.debug(f"질문 내용: {answer.get('question', '')[:100]}...")
        # Gemma 모델용 프롬프트 템플릿
        prompt = PromptTemplate.from_template("""[직무 맥락: {job_code}]
당신은 {job_code} 직무에 대한 취업 자기소개서만을 평가하는 자기소개 전문가입니다.
아래 평가 대상과 참고 사례비교 분석해서 평가대상 답변에 부족한점을 참고 사례의 예시를 들어서 출력은 꼭 feedback 영역 안에 다 출력해주세요. 

[평가 대상]
문항: {question}
답변: {text}

[참고 사례]
{similarity_context}

[평가시 필수 고려사항]
1. {job_code} 직군과 내용이 관계성이 있는지 판단
2. 각 점수대별 명확한 근거 제시
3. 참고 사례와의 구체적인 비교 분석 실천 가능한 개선 방향 제시 
5. 각 점수는 유동적으로 판단해서 평가할 것.

[세부 평가 지표]
Relevance (연관성) - 직무 적합성 및 질문 이해도:
9-10점: {job_code} 직군에서 요구하는 핵심 역량이 구체적인 성과/수치와 함께 명확히 드러나고, 질문이 요구하는 모든 요소에 충실히 답변하며, 해당 분야의 추가적인 인사이트나 개선방안까지 제시함
7-8점: {job_code} 직군 관련 경험과 역량이 구체적 사례와 함께 잘 표현되어 있고, 질문의 모든 요소에 충실히 답변함
5-6점: {job_code} 직군과 관련된 내용이 포함되어 있고 기본적인 답변 요소를 갖추었으나, 일부 구체성이나 깊이가 부족함
3-4점: {job_code} 직군과의 연관성이 부족하거나 질문의 핵심 요소에 대한 답변이 미흡함
1-2점: {job_code} 직군이나 질문 의도와 맞지 않는 내용으로 구성됨


Specificity (구체성) - 경험과 실적의 구체화:
9-10점: 핵심 경험과 주장이 명확한 수치와 구체적 사례로 체계적으로 뒷받침됨
7-8점: 주요 내용이 구체적인 경험과 성과로 잘 설명되어 있음
5-6점: 기본적인 사례와 근거는 있으나 일부 구체성 보완이 필요함
3-4점: 구체적 사례나 근거 제시가 부족하고 일반적인 서술에 그침
1-2점: 구체적 경험이나 근거 없이 추상적인 내용으로만 구성됨


Persuasiveness (설득력) - 논리성과 차별성:
9-10점: 명확한 인과관계와 차별화된 관점으로 본인의 역량과 가치를 설득력 있게 전달
7-8점: 논리적 구성이 잘 되어있고 개인의 강점이 효과적으로 드러남
5-6점: 기본적인 논리는 갖추었으나 차별성이나 설득력이 다소 부족함
3-4점: 논리 전개가 미흡하거나 진부한 내용이 많음
1-2점: 논리적 흐름이 불명확하고 설득력이 현저히 부족함

합격자 자기소개서가 '없음'인 경우: 일반적인 자기소개서 작성 기준과 해당 질문의 의도에 맞춰 평가하고 피드백을 제시하세요.

JSON 형식으로만 평가하세요. Markdown이나 다른 형식을 포함하지 마세요:
{{
    "relevance": <점수>,
    "specificity": <점수>,
    "persuasiveness": <점수>,
    "feedback": "참고 사례 대비 부족한 점, 참고사례에서 참고할점, 점수에 대한 구체적 근거, 건설적인 피드백 , 맞춤법 오류에 대한 피드백"
   
}}

주의사항:
- 반드시 [평가 대상]에 대해서만 평가
- 피드백은 점수와 완전히 일관되어야 함
- 단순 비판이 아닌 개선방향 제시
- 전문적이고 객관적인 톤을 유지하며 점수 평가
- 맞춤법 오류가 있으면 제시
- 합격자의 자기소개서를 보고 평가대상의 자기소개서 개선점 제시
- You must answer in korean"""
)


        # Gemma 모델 로드
        llm = ChatOllama(
            model="gemma2-2b-it:latest",
            temperature=0.7,
            top_p=0.9,
            max_tokens=500,
            timeout=30
        )
        
        chain = prompt | llm
        
        input_data = {
            "question": answer["question"],
            "text": answer["text"],
            "similarity_context": answer.get('similarity_context', "없음"),
            "job_code":answer['job_code']
        }
        
        
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                logger.info(f"답변 생성 시도 {attempt + 1}/{max_attempts}")
                response = await chain.ainvoke(input_data)           

                result_text = response.content if hasattr(response, 'content') else str(response)               

                result = EnhancedJSONParser.validate_and_parse(result_text)
                
                if result:
                    logger.info("답변 처리 성공")
                    return result                
            except Exception as e:
                logger.error(f"평가 처리 오류 (시도 {attempt + 1}): {str(e)}")
        
        # 기본 평가 결과 반환
        return {
            "relevance": 5,
            "specificity": 5,
            "persuasiveness": 5,
            "feedback": "답변 평가 중 문제가 발생했습니다. 다시 시도해주세요."
        }
    
    except Exception as e:
        return {
            "relevance": 5,
            "specificity": 5,
            "persuasiveness": 5,
            "feedback": "시스템 오류가 발생했습니다."
        }

        

async def find_similar_profile(question: str, text: str, reference_data: List[Dict[str, Any]], threshold: float = 80.0) -> tuple:
   logger.info("유사 프로필 검색 시작")
   logger.debug(f"참조 데이터 수: {len(reference_data)}") 
   processor = EmbeddingProcessor()
   
   question_embedding = processor.get_embedding(question)
   text_embedding = processor.get_embedding(text)
   
   max_similarity = -1
   best_profile = None

   logger.info("유사도 비교 시작")
   for profile in reference_data:
       ref_q_embed = processor.get_embedding(profile['quest']) 
       ref_t_embed = processor.get_embedding(profile['text'])

       def cosine_similarity(a, b):
           if len(a.shape) == 1: a = a.unsqueeze(0)
           if len(b.shape) == 1: b = b.unsqueeze(0)
           
           a_norm = a / a.norm(dim=1)[:, None]
           b_norm = b / b.norm(dim=1)[:, None]
           return float(torch.mm(a_norm, b_norm.transpose(0, 1)) * 100)

       combined_sim = (
           0.4 * cosine_similarity(torch.tensor(question_embedding), torch.tensor(ref_q_embed)) +
           0.6 * cosine_similarity(torch.tensor(text_embedding), torch.tensor(ref_t_embed))
       )
       
       if combined_sim > max_similarity:
           max_similarity = combined_sim
           best_profile = profile

   logger.info(f"유사 프로필 검색 완료 (최고 유사도: {max_similarity:.2f})")
   return (best_profile, max_similarity) if max_similarity >= threshold else (None, 0.0)



async def process_gemma_answer(answer: Dict[str, Any],  job_code: str, reference_data: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    logger.info(f"직무 코드 {job_code}에 대한 Gemma 답변 처리 시작")
    logger.debug(f"입력 답변: {json.dumps(answer, ensure_ascii=False)[:200]}...")
    async with ollama_semaphore:
        try:
            logger.info("유사 프로필 검색 중")
            profile, similarity = await find_similar_profile(
                answer["question"], 
                answer["text"], 
                reference_data
            )

            
            
            answer['similarity_context'] = profile['pass_answer'] if profile else "없음"
            answer['similar_profile'] = profile
            answer['job_code'] = job_code
            
            logger.info("Gemma로 답변 처리 중")
            result = await process_answer(answer)
            

            if profile:
                result.update({
                    "similar_h2_tag": profile["h2_tag"],
                    "similar_question": profile["quest"],
                    "similar_answer": profile["pass_answer"],
                    "similarity": similarity,
                    "using_gpt": False
                })
            else:
                result.update({
                    "similar_h2_tag": "",
                    "similar_question": "",
                    "similar_answer": "",
                    "similarity": similarity,
                    "using_gpt": False
                })
            
            return result
        
        except Exception as e:
            return {
                "relevance": 5,
                "specificity": 5,
                "persuasiveness": 5,
                "feedback": "시스템 오류 발생",
                "similar_h2_tag": "",
                "similar_question": "",
                "similar_answer": "",
                "similarity": 0.0
            }