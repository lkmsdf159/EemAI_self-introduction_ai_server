from langchain_community.document_loaders.csv_loader import CSVLoader
from transformers import AutoModel, AutoTokenizer
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from typing import Dict, Any, Optional, List
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


class EmbeddingProcessor:
    _instance = None
    _model = None
    _tokenizer = None
    _batch_size = 128  

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
                model_name = "BM-K/KoSimCSE-roberta"  
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
        EmbeddingProcessor._instance = None  



class EnhancedJSONParser:
    @staticmethod
    def validate_and_parse(text: str) -> Optional[Dict[str, Any]]:
        try:
            def clean_text(text):
                text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
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
                text = text.replace('구체성 ( ) -', ' ')
                text = text.replace('설득력 ( ) -', ' ')
                text = text.replace('관련성 ( ) -', ' ')
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
            required_fields = ['relevance', 'specificity', 'persuasiveness', 'relevance평가', 'specificity평가', 'persuasiveness평가']
            for field in required_fields:
                if field not in result:
                    raise ValueError(f"Missing required field: {field}")
            
            score_fields = ['relevance', 'specificity', 'persuasiveness']
            for field in score_fields:
                score = result[field]
                if not (1 <= score <= 100):
                    result[field] = 0
                    
            evaluation_fields = [
                'relevance평가',
                'specificity평가',
                'persuasiveness평가',
                'reference_analysis'
            ]
            
            for field in evaluation_fields:
                if field in result:
                    result[field] = clean_text(result[field])                
            
            return result

        except Exception as e:
            logger.error(f"JSON 파싱 오류: {str(e)}")
            logger.error(f"문제의 텍스트: {text}")
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
당신은 {job_code} 직무에 대한 자기소개서만을 평가하는 자기소개 전문가입니다.
아래 평가 대상과 참고 사례를 비교 분석하여 평가 대상 답변에 부족한 점을 참고 사례의 예시를 들어서 설명하세요.
출력은 반드시 feedback 영역 안에 작성해주세요.

[평가 대상]
문항: {question}
답변: {text}

[참고 사례]
{similarity_context}

상세 평가 기준:

[세부 평가 지표]
1. Relevance (연관성) - 직무 적합성 및 질문 이해도:
- [평가 대상]이 {job_code} 직무와 관련이 있을 경우: 직무 적합성과 질문 이해도를 모두 평가
- [평가 대상]이 {job_code} 직무와 관련이 없을 경우: 질문 이해도만 평가하며, 직무 적합성은 감점 요소에서 제외

점수 기준:
- 90~100점: 직무와 관련된 핵심 역량이 명확히 드러나며, 질문의 의도를 완벽히 파악하여 추가적인 인사이트까지 제공
- 80~89점: 직무와 관련성이 매우 높고, 질문 의도를 잘 반영함
- 70~79점: 직무와 관련성이 높고, 질문 의도도 잘 반영되나 일부 보완이 필요함
- 60~69점: 직무와 관련된 내용을 다루고 있으나, 구체성이 부족하고 추가 설명이 필요함
- 50~59점: 직무와 일부 관련이 있으나 개선이 필요하고, 추상적 내용이 많음
- 40~49점: 직무와 연관성이 다소 부족하고, 질문 의도와 관련성이 약함
- 30~39점: 직무와의 연관성이 거의 없고, 설명이 매우 추상적임
- 20~29점: 직무와의 연관성이 거의 없고, 질문의 핵심을 반영하지 않음
- 10~19점: 직무와 무관한 내용만 서술하거나 질문 의도를 전혀 이해하지 못함
- 0점: 직무와 전혀 무관하거나, 질문을 전혀 이해하지 못한 경우

2. Specificity (구체성) - 경험과 실적의 구체화:
- 90~100점: 구체적인 수치나 사례를 제시하며 모든 주장이 경험을 통해 뒷받침됨
- 80~89점: 대체로 구체적인 사례가 있으며, 중요한 경험이 잘 드러남
- 70~79점: 경험이 구체적으로 서술되었으나 일부 미비한 부분이 있음
- 60~69점: 구체적인 사례는 있지만, 추가적인 설명이나 사례가 필요함
- 50~59점: 구체적인 사례가 부족하고, 더 명확한 예시나 증거가 필요함
- 40~49점: 추상적인 내용이 많고 구체적인 사례가 거의 없음
- 30~39점: 대부분 추상적인 설명에 그침
- 20~29점: 구체적인 사례나 수치가 거의 없음
- 10~19점: 구체적인 내용이 전혀 없으며, 전반적으로 모호한 설명만 있음
- 0점: 실제 경험이 없어 보이며 내용이 매우 부실함

3. Persuasiveness (설득력) - 논리성과 차별성:
- 90~100점: 논리적 흐름이 매우 명확하고, 독창적인 관점과 강력한 동기부여가 포함됨
- 80~89점: 논리적인 구성이 잘 되어 있고, 차별화된 경험이 설득력 있게 전달됨
- 70~79점: 기본적인 논리는 잘 갖추어져 있으나, 일부 보완이 필요함
- 60~69점: 논리적인 흐름이 있지만 차별성이 부족하거나 설득력이 약함
- 50~59점: 논리가 부족하고 설득력이 약함, 보강이 필요함
- 40~49점: 논리적인 전개가 매끄럽지 않으며, 설득력이 낮음
- 30~39점: 주장과 근거의 연결이 미약함
- 20~29점: 논리적 흐름이 매우 부실하고 차별성이 없음
- 10~19점: 설득력이 거의 없으며, 논리적 연결이 부족함
- 0점: 논리적 흐름이 없거나, 내용이 부실하고 설득력이 전혀 없음


[평가 시 필수 고려사항]
1. {job_code} 직군과 내용이 관계성이 있는지 판단
2. 각 점수대별 명확한 근거 제시
3. 참고 사례와의 구체적인 비교 분석
4. 실천 가능한 개선 방향 제시

합격자 자기소개서가 '없음'인 경우: 일반적인 자기소개서 작성 기준과 해당 질문의 의도에 맞춰 평가하고 피드백을 제시하세요.

JSON 형식으로만 평가하세요. Markdown이나 다른 형식을 포함하지 마세요:
{{
    "relevance": <점수>,
    "specificity": <점수>,
    "persuasiveness": <점수>,
    "relevance평가": "<relevance 점수에 대한 근거, 건설적인 피드백, 맞춤법 오류에 대한 피드백>",
    "specificity평가": "<specificity 점수에 대한 근거, 건설적인 피드백, 맞춤법 오류에 대한 피드백>",
    "persuasiveness평가": "<persuasiveness 점수에 대한 근거, 건설적인 피드백, 맞춤법 오류에 대한 피드백>",
    "reference_analysis": "<[참고 사례]와 [평가 대상]의 주요 차이점 및 [평가 대상]의 개선사항, 참고 사례가 '없음'인 경우 이 필드는 비워두세요.>"
}}

주의사항:
- reference_analysis는 참고 사례가 있을 때만 작성하며 다음을 포함해야 합니다:
   - 합격자 사례와의 핵심적인 차이점
   - 합격자 사례에서 배울 수 있는 구체적인 요소들
   - 실천 가능한 개선 방향
- 반드시 [평가 대상]에 대해서만 평가
- 피드백은 점수와 완전히 일관되어야 함
- 단순 비판이 아닌 구체적인 개선 방향 제시
- 전문적이고 객관적인 tone 유지
- 맞춤법 피드백은 철자, 문장 구조의 기술적 측면에만 집중
- 합격자의 자기소개서를 보고 평가 대상의 자기소개서를 구체화시킬 것
- You must answer in korean"""
)



        llm = ChatOllama(
            model="gemma2-2b-it:latest",
            temperature=0.9,
            top_p=0.9,
            max_tokens=500,
            timeout=30
        )
        
        chain = prompt | llm
        
        
        has_similar_profile = answer.get('similarity_context') != "없음"

        input_data = {
            "question": answer["question"],
            "text": answer["text"],
            "similarity_context": answer.get('similarity_context', "없음"),
            "job_code":answer['job_code']
        }
        
        
        max_attempts = 4
        for attempt in range(max_attempts):
            try:
                logger.info(f"답변 생성 시도 {attempt + 1}/{max_attempts}")

                response = await chain.ainvoke(input_data)           

                result_text = response.content if hasattr(response, 'content') else str(response)               

                result = EnhancedJSONParser.validate_and_parse(result_text)
                
                logger.debug(f"응답 텍스트: {result_text}")

                if result:
                    if not has_similar_profile and 'reference_analysis' in result:
                        del result['reference_analysis']
                
                    if has_similar_profile:
                        if not result.get('reference_analysis'):
                            logger.warning("유사 자소서가 있지만 reference_analysis가 없음")
                            continue  
                        
                        if len(result['reference_analysis']) < 80: 
                            logger.warning("reference_analysis가 너무 짧음")
                            continue  
                    
                    return result

                logger.warning(f"JSON 검증 실패 (시도 {attempt + 1})")

            except Exception as e:
                logger.error(f"평가 처리 오류 (시도 {attempt + 1}): {str(e)}")
        
        # 기본 평가 결과 반환
        return {
            "relevance": 0,
            "specificity": 0,
            "persuasiveness": 0,
            "relevance평가": "답변 평가 중 문제가 발생했습니다. 다시 시도해주세요.",
            "specificity평가": "답변 평가 중 문제가 발생했습니다. 다시 시도해주세요.",
            "persuasiveness평가": "답변 평가 중 문제가 발생했습니다. 다시 시도해주세요."
        }
    
    except Exception as e:
        return {
            "relevance": 0,
            "specificity": 0,
            "persuasiveness": 0,
            "relevance평가": "시스템 오류가 발생했습니다.",
            "specificity평가": "시스템 오류가 발생했습니다.",
            "persuasiveness평가": "시스템 오류가 발생했습니다."
        }

        

async def find_similar_profile(question: str, text: str, reference_data: List[Dict[str, Any]], threshold: float = 80.0) -> tuple:
    logger.info("유사 프로필 검색 시작")
    logger.debug(f"참조 데이터 수: {len(reference_data)}") 
    processor = EmbeddingProcessor()
    
    def check_free_format(text: str) -> bool:
        patterns = [
            r'자유\s*형식',  
            r'자유\s*양식', 
            r'<.*?자유.*?형식.*?>', 
            r'<.*?자유.*?양식.*?>', 
            r'\d+\.*\s*자유\s*형식', 
            r'\d+\.*\s*자유\s*양식',  
            r'\[.*?자유.*?형식.*?\]',  
            r'\[.*?자유.*?양식.*?\]',  
            r'「.*?자유.*?형식.*?」', 
            r'「.*?자유.*?양식.*?」' 
        ]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)
    
    has_free_format = check_free_format(question)
    
    if has_free_format:
        reference_data = [data for data in reference_data if check_free_format(data['quest'])]
        logger.info(f"자유형식/양식 질문 필터링 후 데이터 수: {len(reference_data)}")
    
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

        if has_free_format:
            combined_sim = cosine_similarity(torch.tensor(text_embedding), torch.tensor(ref_t_embed))
        else:
            combined_sim = (
                0.4 * cosine_similarity(torch.tensor(question_embedding), torch.tensor(ref_q_embed)) +
                0.6 * cosine_similarity(torch.tensor(text_embedding), torch.tensor(ref_t_embed))
            )
        
        if combined_sim > max_similarity:
            max_similarity = combined_sim
            best_profile = profile

    logger.info(f"유사 프로필 검색 완료 (최고 유사도: {max_similarity:.2f})")

    effective_threshold = threshold * 0.9 if has_free_format else threshold
    return (best_profile, max_similarity) if max_similarity >= effective_threshold else (None, 0.0)



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
                "relevance": 0,
                "specificity": 0,
                "persuasiveness": 0,
                "relevance평가": "문제가 발생했습니다. 다시 시도해주세요.",
                "specificity평가": "문제가 발생했습니다. 다시 시도해주세요.",
                "persuasiveness평가": "문제가 발생했습니다. 다시 시도해주세요.",
                "reference_analysis": "",
                "similar_h2_tag": "",
                "similar_question": "",
                "similar_answer": "",
                "similarity": 0.0
            }
            