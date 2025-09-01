"""
BOM 원가 계산기 - 장기적 안정성 강화 버전
데이터 검증, 로깅, 오류 처리, 성능 최적화 적용
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import time
from datetime import datetime
import traceback
import warnings

# 성능 최적화 (선택적 import)
try:
    import bottleneck as bn
    pandas_optimized = True
except ImportError:
    pandas_optimized = False
    st.warning("bottleneck 패키지가 없어 pandas 최적화가 비활성화됩니다.")

# 진행률 표시
try:
    from stqdm import stqdm
    progress_available = True
except ImportError:
    from tqdm import tqdm
    progress_available = False

# 로깅
try:
    from loguru import logger
    import sys
    
    # 로그 설정
    logger.remove()  # 기본 핸들러 제거
    logger.add(
        sys.stdout, 
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    logging_available = True
except ImportError:
    logging_available = False

# 데이터 검증
try:
    import pandera as pa
    from pandera import Column, DataFrameSchema, Check
    validation_available = True
except ImportError:
    validation_available = False

# 문자열 매칭
try:
    from fuzzywuzzy import fuzz, process
    fuzzy_matching_available = True
except ImportError:
    fuzzy_matching_available = False

# 시각화
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    plotly_available = True
except ImportError:
    plotly_available = False

# 경고 필터링
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

class BOMCalculatorConfig:
    """설정 클래스"""
    MAX_ITERATIONS = 100
    MAX_FILE_SIZE_MB = 100
    CACHE_TIMEOUT = 3600  # 1시간
    SUPPORTED_EXTENSIONS = ['.csv', '.xlsx', '.xls']
    REQUIRED_BOM_COLUMNS = ['생산품목코드', '생산품목명', '소모품목코드', '소모품목명', '소요량']
    TEST_ITEM_CODE = '99701'

class DataValidator:
    """데이터 검증 클래스"""
    
    @staticmethod
    def validate_file_size(file_obj, max_size_mb: int = 100) -> bool:
        """파일 크기 검증"""
        try:
            file_size = len(file_obj.getvalue()) / (1024 * 1024)  # MB
            if file_size > max_size_mb:
                st.error(f"❌ 파일 크기가 {max_size_mb}MB를 초과합니다. ({file_size:.1f}MB)")
                return False
            return True
        except Exception as e:
            if logging_available:
                logger.error(f"파일 크기 검증 오류: {e}")
            st.error(f"파일 크기 검증 중 오류: {e}")
            return False

    @staticmethod
    def validate_bom_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """BOM 데이터 스키마 검증"""
        if not validation_available:
            # 기본 검증
            required_cols = BOMCalculatorConfig.REQUIRED_BOM_COLUMNS
            missing_cols = [col for col in required_cols if col not in df.columns]
            return len(missing_cols) == 0, missing_cols
        
        try:
            # pandera 스키마 정의
            bom_schema = DataFrameSchema({
                "생산품목코드": Column(pa.String, nullable=False, checks=Check.str_length(min_value=1)),
                "생산품목명": Column(pa.String, nullable=False, checks=Check.str_length(min_value=1)),
                "소모품목코드": Column(pa.String, nullable=False, checks=Check.str_length(min_value=1)),
                "소모품목명": Column(pa.String, nullable=False, checks=Check.str_length(min_value=1)),
                "소요량": Column(pa.Float, nullable=False, checks=Check.greater_than_or_equal_to(0))
            })
            
            # 검증 실행
            bom_schema.validate(df, lazy=True)
            return True, []
            
        except pa.errors.SchemaError as e:
            error_messages = []
            for failure in e.failure_cases.itertuples():
                error_messages.append(f"컬럼 '{failure.column}': {failure.check}")
            
            if logging_available:
                logger.error(f"BOM 스키마 검증 실패: {error_messages}")
            
            return False, error_messages
        except Exception as e:
            if logging_available:
                logger.error(f"스키마 검증 중 예외: {e}")
            return False, [str(e)]

    @staticmethod
    def validate_purchase_data(df: pd.DataFrame) -> Tuple[bool, str]:
        """구매 데이터 검증"""
        try:
            if df.empty:
                return False, "구매 데이터가 비어있습니다"
            
            if len(df.columns) < 3:
                return False, "구매 데이터의 컬럼이 부족합니다"
                
            # 필수 데이터 존재 여부
            non_empty_rows = df.dropna(how='all')
            if len(non_empty_rows) < 1:
                return False, "유효한 구매 데이터가 없습니다"
                
            return True, "검증 성공"
            
        except Exception as e:
            return False, f"구매 데이터 검증 오류: {str(e)}"

class SecureFileLoader:
    """안전한 파일 로더"""
    
    @staticmethod
    @st.cache_data(ttl=BOMCalculatorConfig.CACHE_TIMEOUT, show_spinner=False)
    def load_data(file_content: bytes, file_name: str, skiprows: int = 0) -> Optional[pd.DataFrame]:
        """캐시된 안전한 데이터 로딩"""
        try:
            # 파일 확장자 검증
            file_path = Path(file_name)
            if file_path.suffix.lower() not in BOMCalculatorConfig.SUPPORTED_EXTENSIONS:
                st.error(f"❌ 지원하지 않는 파일 형식: {file_path.suffix}")
                return None
            
            # 메모리에서 파일 읽기
            file_obj = io.BytesIO(file_content)
            
            if logging_available:
                logger.info(f"파일 로딩 시작: {file_name}")
            
            # 파일 타입별 로딩
            if file_path.suffix.lower() == '.csv':
                # CSV 인코딩 자동 감지 시도
                encodings = ['utf-8-sig', 'utf-8', 'cp949', 'euc-kr']
                df = None
                
                for encoding in encodings:
                    try:
                        file_obj.seek(0)
                        df = pd.read_csv(
                            file_obj, 
                            skiprows=skiprows, 
                            encoding=encoding, 
                            dtype=str,
                            na_values=['', 'NULL', 'null', 'NaN', 'nan']
                        )
                        if logging_available:
                            logger.info(f"CSV 인코딩 성공: {encoding}")
                        break
                    except (UnicodeDecodeError, pd.errors.EmptyDataError):
                        continue
                
                if df is None:
                    st.error("❌ CSV 파일 인코딩을 감지할 수 없습니다")
                    return None
                    
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                try:
                    df = pd.read_excel(
                        file_obj, 
                        skiprows=skiprows, 
                        dtype=str,
                        na_values=['', 'NULL', 'null', 'NaN', 'nan']
                    )
                except Exception as e:
                    st.error(f"❌ Excel 파일 읽기 오류: {e}")
                    return None
            
            # 데이터 정제
            if df is not None:
                # 좌우 공백 제거
                for col in df.select_dtypes(include=['object']).columns:
                    df[col] = df[col].astype(str).str.strip()
                
                # 완전히 빈 행/열 제거
                df = df.dropna(how='all').dropna(axis=1, how='all')
                
                if logging_available:
                    logger.info(f"파일 로딩 완료: {len(df)}행 × {len(df.columns)}열")
                
                return df
            
            return None
            
        except Exception as e:
            if logging_available:
                logger.error(f"파일 로딩 실패: {file_name}, 오류: {str(e)}")
            st.error(f"❌ 파일 로딩 실패: {str(e)}")
            return None

class PurchasePriceExtractor:
    """구매 단가 추출기"""
    
    @staticmethod
    def extract_prices(df: pd.DataFrame) -> Dict[str, float]:
        """구매 데이터에서 안전하게 최신 단가 추출"""
        try:
            if df.empty:
                if logging_available:
                    logger.warning("빈 구매 데이터")
                return {}
            
            # 컬럼명 자동 감지
            column_mapping = PurchasePriceExtractor._detect_columns(df)
            
            if not all(column_mapping.values()):
                if logging_available:
                    logger.warning(f"필수 컬럼 감지 실패: {column_mapping}")
                st.warning("⚠️ 구매 데이터에서 필수 컬럼을 찾을 수 없습니다")
                return {}
            
            # 데이터 정제 및 변환
            clean_df = PurchasePriceExtractor._clean_purchase_data(df, column_mapping)
            
            if clean_df.empty:
                if logging_available:
                    logger.warning("정제 후 구매 데이터가 비어있음")
                return {}
            
            # 최신 단가 추출
            price_dict = PurchasePriceExtractor._extract_latest_prices(clean_df)
            
            if logging_available:
                logger.info(f"구매 단가 추출 완료: {len(price_dict)}개 품목")
            
            return price_dict
            
        except Exception as e:
            if logging_available:
                logger.error(f"구매 단가 추출 오류: {str(e)}")
            st.error(f"❌ 구매 단가 추출 오류: {str(e)}")
            return {}
    
    @staticmethod
    def _detect_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
        """컬럼명 자동 감지"""
        column_mapping = {'date': None, 'item_code': None, 'price': None}
        
        # 첫 번째 행이 헤더인지 확인 (회사명 등이 있으면 건너뛰기)
        first_row_text = ' '.join([str(val) for val in df.iloc[0].values if pd.notna(val)])
        if any(keyword in first_row_text for keyword in ['회사명', '기간', '날짜', '조회']):
            # 다음 행을 헤더로 사용
            if len(df) > 1:
                new_headers = df.iloc[1].fillna('').astype(str).tolist()
                df_temp = df.iloc[2:].copy()
                df_temp.columns = new_headers[:len(df_temp.columns)]
            else:
                df_temp = df
        else:
            df_temp = df
        
        # 컬럼명으로 매핑
        for col in df_temp.columns:
            col_str = str(col).lower()
            
            if not column_mapping['date'] and any(keyword in col_str for keyword in ['일자', 'date', '날짜']):
                column_mapping['date'] = col
            elif not column_mapping['item_code'] and '품목코드' in col_str:
                column_mapping['item_code'] = col
            elif not column_mapping['price'] and '단가' in col_str and '공급' not in col_str:
                column_mapping['price'] = col
        
        # 기본값 할당 (컬럼을 찾지 못한 경우)
        if not column_mapping['date'] and len(df_temp.columns) > 0:
            column_mapping['date'] = df_temp.columns[0]
        if not column_mapping['item_code'] and len(df_temp.columns) > 1:
            column_mapping['item_code'] = df_temp.columns[1]
        if not column_mapping['price'] and len(df_temp.columns) > 5:
            column_mapping['price'] = df_temp.columns[5]
        
        return column_mapping
    
    @staticmethod
    def _clean_purchase_data(df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
        """구매 데이터 정제"""
        try:
            # 필요한 컬럼만 추출
            clean_df = df[[column_mapping['date'], column_mapping['item_code'], column_mapping['price']]].copy()
            clean_df.columns = ['date', 'item_code', 'price']
            
            # 빈 값 제거
            clean_df = clean_df.dropna()
            
            # 날짜 변환
            clean_df['date_str'] = clean_df['date'].astype(str)
            clean_df['date_clean'] = clean_df['date_str'].str.split('-').str[0]
            clean_df['date_parsed'] = pd.to_datetime(clean_df['date_clean'], errors='coerce')
            
            # 날짜 변환에 실패한 행 제거
            clean_df = clean_df.dropna(subset=['date_parsed'])
            
            # 품목코드와 단가 정제
            clean_df['item_code'] = clean_df['item_code'].astype(str).str.strip()
            clean_df['price'] = pd.to_numeric(clean_df['price'], errors='coerce')
            
            # 유효하지 않은 데이터 제거
            clean_df = clean_df[
                (clean_df['item_code'] != '') & 
                (clean_df['item_code'] != 'nan') & 
                (clean_df['price'] > 0) & 
                (clean_df['price'].notna())
            ]
            
            return clean_df
            
        except Exception as e:
            if logging_available:
                logger.error(f"구매 데이터 정제 오류: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    def _extract_latest_prices(df: pd.DataFrame) -> Dict[str, float]:
        """최신 단가 추출"""
        try:
            # 날짜순 정렬 (최신 순)
            df_sorted = df.sort_values('date_parsed', ascending=False)
            
            # 품목별 최신 단가만 추출
            latest_prices = df_sorted.drop_duplicates(subset='item_code', keep='first')
            
            # 딕셔너리로 변환
            price_dict = {}
            for _, row in latest_prices.iterrows():
                item_code = row['item_code']
                price = row['price']
                if pd.notna(price) and price > 0:
                    price_dict[item_code] = float(price)
            
            return price_dict
            
        except Exception as e:
            if logging_available:
                logger.error(f"최신 단가 추출 오류: {str(e)}")
            return {}

class BOMCostCalculator:
    """BOM 원가 계산기 - 개선된 버전"""
    
    def __init__(self):
        self.calculation_cache = {}
        self.calculation_log = []
    
    def calculate_all_costs(self, bom_df: pd.DataFrame, purchase_prices: Dict[str, float]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """모든 제품의 BOM 원가 계산"""
        try:
            start_time = time.time()
            
            if logging_available:
                logger.info("BOM 원가 계산 시작")
            
            # 데이터 검증
            is_valid, validation_errors = DataValidator.validate_bom_schema(bom_df)
            if not is_valid:
                st.error("❌ BOM 데이터 검증 실패:")
                for error in validation_errors:
                    st.error(f"  • {error}")
                return pd.DataFrame(), pd.DataFrame()
            
            # 데이터 정제
            clean_bom = self._clean_bom_data(bom_df)
            
            if clean_bom.empty:
                st.error("❌ BOM 데이터 정제 후 데이터가 없습니다")
                return pd.DataFrame(), pd.DataFrame()
            
            # 모든 생산품목 목록
            all_products = clean_bom[['생산품목코드', '생산품목명']].drop_duplicates().reset_index(drop=True)
            
            if logging_available:
                logger.info(f"계산 대상: {len(all_products)}개 생산품목, {len(purchase_prices)}개 구매단가")
            
            # 전체 단가 딕셔너리 초기화
            all_costs = purchase_prices.copy()
            self.calculation_cache = {}
            self.calculation_log = []
            
            # 진행률 표시와 함께 계산
            results = []
            
            if progress_available:
                progress_bar = stqdm(all_products.iterrows(), total=len(all_products), desc="BOM 원가 계산")
            else:
                progress_bar = all_products.iterrows()
                st_progress = st.progress(0)
            
            for idx, (_, product) in enumerate(progress_bar):
                product_code = product['생산품목코드']
                product_name = product['생산품목명']
                
                # 직접 계산
                calculated_cost = self._calculate_product_cost(product_code, clean_bom, all_costs)
                
                results.append({
                    '생산품목코드': product_code,
                    '생산품목명': product_name,
                    '계산된단위원가': calculated_cost,
                    '계산상태': '계산완료' if calculated_cost > 0 else '계산불가'
                })
                
                # 진행률 업데이트 (stqdm이 없는 경우)
                if not progress_available:
                    st_progress.progress((idx + 1) / len(all_products))
            
            if not progress_available:
                st_progress.empty()
            
            # 결과 DataFrame 생성
            result_df = pd.DataFrame(results)
            
            # 상세 내역 생성
            details_df = self._generate_details(clean_bom, all_costs)
            
            # 계산 시간 기록
            elapsed_time = time.time() - start_time
            
            if logging_available:
                logger.info(f"BOM 원가 계산 완료: {elapsed_time:.2f}초")
            
            st.success(f"✅ BOM 원가 계산 완료! (소요시간: {elapsed_time:.2f}초)")
            
            return result_df, details_df
            
        except Exception as e:
            if logging_available:
                logger.error(f"BOM 원가 계산 중 오류: {str(e)}")
                logger.error(f"스택 트레이스: {traceback.format_exc()}")
            st.error(f"❌ BOM 원가 계산 중 오류 발생: {str(e)}")
            return pd.DataFrame(), pd.DataFrame()
    
    def _clean_bom_data(self, bom_df: pd.DataFrame) -> pd.DataFrame:
        """BOM 데이터 정제"""
        try:
            clean_df = bom_df.copy()
            
            # 필수 컬럼 확인
            required_cols = BOMCalculatorConfig.REQUIRED_BOM_COLUMNS
            missing_cols = [col for col in required_cols if col not in clean_df.columns]
            
            if missing_cols:
                st.error(f"❌ BOM 데이터에 필수 컬럼이 없습니다: {missing_cols}")
                return pd.DataFrame()
            
            # 데이터 타입 정제
            clean_df['생산품목코드'] = clean_df['생산품목코드'].astype(str).str.strip()
            clean_df['소모품목코드'] = clean_df['소모품목코드'].astype(str).str.strip()
            clean_df['소요량'] = pd.to_numeric(clean_df['소요량'], errors='coerce').fillna(0)
            
            # test 품목 제거
            before_count = len(clean_df)
            clean_df = clean_df[clean_df['소모품목코드'] != BOMCalculatorConfig.TEST_ITEM_CODE]
            after_count = len(clean_df)
            
            if before_count != after_count:
                st.info(f"🧹 test 품목({BOMCalculatorConfig.TEST_ITEM_CODE}) 제거: {before_count:,} → {after_count:,}행")
            
            # 유효하지 않은 데이터 제거
            clean_df = clean_df[
                (clean_df['생산품목코드'] != '') &
                (clean_df['소모품목코드'] != '') &
                (clean_df['생산품목코드'] != 'nan') &
                (clean_df['소모품목코드'] != 'nan') &
                (clean_df['소요량'] >= 0)
            ]
            
            return clean_df
            
        except Exception as e:
            if logging_available:
                logger.error(f"BOM 데이터 정제 오류: {str(e)}")
            return pd.DataFrame()
    
    def _calculate_product_cost(self, product_code: str, bom_df: pd.DataFrame, all_costs: Dict[str, float]) -> float:
        """단일 제품의 원가 계산 (재귀적)"""
        try:
            # 캐시 확인
            if product_code in self.calculation_cache:
                return self.calculation_cache[product_code]
            
            # 해당 제품의 BOM 구성요소
            components = bom_df[bom_df['생산품목코드'] == product_code]
            
            if components.empty:
                self.calculation_cache[product_code] = 0.0
                return 0.0
            
            total_cost = 0.0
            component_details = []
            
            for _, comp in components.iterrows():
                comp_code = comp['소모품목코드']
                comp_name = comp['소모품목명']
                quantity = float(comp['소요량'])
                
                # 부품 단가 찾기
                if comp_code in all_costs:
                    # 이미 알려진 단가
                    unit_price = all_costs[comp_code]
                else:
                    # 다른 생산품목인지 확인하여 재귀 계산
                    if comp_code in bom_df['생산품목코드'].values:
                        unit_price = self._calculate_product_cost(comp_code, bom_df, all_costs)
                        all_costs[comp_code] = unit_price  # 캐시 업데이트
                    else:
                        unit_price = 0.0
                
                component_cost = quantity * unit_price
                total_cost += component_cost
                
                component_details.append({
                    '부품코드': comp_code,
                    '부품명': comp_name,
                    '소요량': quantity,
                    '단가': unit_price,
                    '부품원가': component_cost
                })
            
            # 캐시에 저장
            self.calculation_cache[product_code] = total_cost
            
            # 로그 저장
            if total_cost > 0:
                self.calculation_log.append({
                    '제품코드': product_code,
                    '총원가': total_cost,
                    '구성요소수': len(component_details)
                })
            
            return total_cost
            
        except Exception as e:
            if logging_available:
                logger.error(f"제품 {product_code} 원가 계산 오류: {str(e)}")
            return 0.0
    
    def _generate_details(self, bom_df: pd.DataFrame, all_costs: Dict[str, float]) -> pd.DataFrame:
        """상세 내역 생성"""
        try:
            details_df = bom_df.copy()
            details_df['부품단가'] = details_df['소모품목코드'].map(all_costs).fillna(0)
            details_df['부품별원가'] = details_df['소요량'] * details_df['부품단가']
            
            return details_df
            
        except Exception as e:
            if logging_available:
                logger.error(f"상세 내역 생성 오류: {str(e)}")
            return pd.DataFrame()

class ExcelExporter:
    """안전한 엑셀 내보내기"""
    
    @staticmethod
    def export_results(finished_goods: pd.DataFrame, all_results: pd.DataFrame, details: pd.DataFrame) -> bytes:
        """결과를 엑셀로 내보내기"""
        try:
            output = io.BytesIO()
            
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # 완제품 결과
                ExcelExporter._write_sheet_with_formatting(
                    writer, finished_goods, '완제품원가결과', 
                    '완제품 BOM 원가 계산 결과'
                )
                
                # 전체 결과
                ExcelExporter._write_sheet_with_formatting(
                    writer, all_results, '전체제품원가',
                    '전체 제품 원가 계산 결과'
                )
                
                # 상세 내역
                ExcelExporter._write_sheet_with_formatting(
                    writer, details, 'BOM상세내역',
                    'BOM 구성요소별 상세 원가 내역'
                )
            
            return output.getvalue()
            
        except Exception as e:
            if logging_available:
                logger.error(f"엑셀 내보내기 오류: {str(e)}")
            st.error(f"❌ 엑셀 파일 생성 오류: {str(e)}")
            return b''
    
    @staticmethod
    def _write_sheet_with_formatting(writer, df: pd.DataFrame, sheet_name: str, title: str):
        """시트별 포맷팅 적용"""
        try:
            # 기본 데이터 쓰기
            df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=2)
            
            worksheet = writer.sheets[sheet_name]
            
            # 제목 추가
            worksheet.cell(row=1, column=1, value=title)
            
            # 스타일 적용 시도
            try:
                from openpyxl.styles import PatternFill, Font, Alignment
                from openpyxl.utils import get_column_letter
                
                # 제목 스타일
                title_cell = worksheet.cell(row=1, column=1)
                title_cell.font = Font(size=14, bold=True)
                title_cell.alignment = Alignment(horizontal='center')
                
                # 헤더 스타일
                header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
                header_font = Font(color="FFFFFF", bold=True)
                
                for col in range(1, len(df.columns) + 1):
                    cell = worksheet.cell(row=3, column=col)
                    cell.fill = header_fill
                    cell.font = header_font
                    cell.alignment = Alignment(horizontal='center')
                
                # 컬럼 너비 조정
                for col in range(1, len(df.columns) + 1):
                    column = df.columns[col-1]
                    max_length = max(
                        len(str(column)),
                        df[column].astype(str).str.len().max() if not df.empty else 0
                    )
                    adjusted_width = min(max_length + 3, 50)
                    column_letter = get_column_letter(col)
                    worksheet.column_dimensions[column_letter].width = adjusted_width
                
            except ImportError:
                # openpyxl 스타일 모듈을 import할 수 없는 경우
                pass
                
        except Exception as e:
            if logging_available:
                logger.warning(f"시트 포맷팅 오류: {e}")

class DataVisualizer:
    """데이터 시각화 클래스"""
    
    @staticmethod
    def create_cost_analysis_charts(finished_goods: pd.DataFrame) -> None:
        """원가 분석 차트 생성"""
        if not plotly_available:
            st.info("📊 plotly가 설치되지 않아 시각화를 건너뜁니다.")
            return
        
        try:
            if finished_goods.empty:
                return
            
            # 데이터 준비
            chart_data = finished_goods[finished_goods['계산된단위원가'] > 0].copy()
            
            if chart_data.empty:
                st.warning("⚠️ 차트를 그릴 데이터가 없습니다.")
                return
            
            # 1. 원가 분포 히스토그램
            st.subheader("📊 완제품 원가 분포")
            
            fig_hist = px.histogram(
                chart_data, 
                x='계산된단위원가',
                nbins=20,
                title='완제품 원가 분포',
                labels={'계산된단위원가': '원가 (원)', 'count': '제품 수'}
            )
            fig_hist.update_layout(showlegend=False)
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # 2. 상위 원가 제품 TOP 20
            if len(chart_data) >= 10:
                st.subheader("💰 원가 상위 20개 완제품")
                
                top_products = chart_data.nlargest(20, '계산된단위원가')
                
                fig_bar = px.bar(
                    top_products,
                    x='계산된단위원가',
                    y='생산품목코드',
                    orientation='h',
                    title='원가 상위 20개 완제품',
                    labels={'계산된단위원가': '원가 (원)', '생산품목코드': '제품코드'}
                )
                fig_bar.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_bar, use_container_width=True)
            
            # 3. 원가 구간별 분포
            st.subheader("📈 원가 구간별 제품 분포")
            
            # 원가 구간 정의
            bins = [0, 1000, 5000, 10000, 50000, 100000, float('inf')]
            labels = ['~1천원', '1천~5천원', '5천~1만원', '1만~5만원', '5만~10만원', '10만원+']
            
            chart_data['원가구간'] = pd.cut(chart_data['계산된단위원가'], bins=bins, labels=labels)
            cost_distribution = chart_data['원가구간'].value_counts().sort_index()
            
            fig_pie = px.pie(
                values=cost_distribution.values,
                names=cost_distribution.index,
                title='원가 구간별 제품 분포'
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
        except Exception as e:
            if logging_available:
                logger.error(f"차트 생성 오류: {str(e)}")
            st.warning(f"⚠️ 차트 생성 중 오류: {str(e)}")

def main():
    """메인 애플리케이션"""
    
    # 페이지 설정
    st.set_page_config(
        page_title="BOM 원가 계산기",
        page_icon="🏭",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 제목 및 설명
    st.title("🏭 BOM 원가 계산기 (장기적 안정성 강화 버전)")
    st.markdown("""
    **✨ 주요 기능**
    - 📋 데이터 검증 및 스키마 확인
    - 🔄 재귀적 다단계 BOM 원가 계산  
    - 📊 실시간 진행률 표시
    - 🎯 시각적 원가 분석
    - 📥 포맷팅된 엑셀 출력
    - 🛡️ 안전한 오류 처리
    """)
    
    # 사이드바 설정
    with st.sidebar:
        st.header("⚙️ 계산 설정")
        
        # 고급 옵션
        show_debug = st.checkbox("🔍 디버깅 정보 표시", value=False)
        show_charts = st.checkbox("📊 시각화 차트 표시", value=True)
        
        st.header("📊 시스템 정보")
        st.info(f"""
        **활성화된 기능:**
        - 로깅: {'✅' if logging_available else '❌'}
        - 데이터 검증: {'✅' if validation_available else '❌'}
        - 진행률 표시: {'✅' if progress_available else '❌'}
        - 시각화: {'✅' if plotly_available else '❌'}
        - pandas 최적화: {'✅' if pandas_optimized else '❌'}
        - 문자열 매칭: {'✅' if fuzzy_matching_available else '❌'}
        """)
    
    # 메인 인터페이스
    st.header("1. 📁 파일 업로드")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 BOM 데이터")
        bom_file = st.file_uploader(
            "BOM 파일을 선택하세요",
            type=['csv', 'xlsx', 'xls'],
            key="bom_file",
            help="생산품목코드, 생산품목명, 소모품목코드, 소모품목명, 소요량 컬럼이 필요합니다"
        )
        
    with col2:
        st.subheader("💰 구매 데이터") 
        purchase_file = st.file_uploader(
            "구매 데이터 파일을 선택하세요",
            type=['csv', 'xlsx', 'xls'],
            key="purchase_file",
            help="일자, 품목코드, 단가 컬럼이 필요합니다"
        )
    
    # 파일 검증 및 로딩
    if bom_file and purchase_file:
        
        # 파일 크기 검증
        if not DataValidator.validate_file_size(bom_file) or not DataValidator.validate_file_size(purchase_file):
            st.stop()
        
        # 데이터 로딩
        with st.spinner("📖 파일을 읽는 중..."):
            bom_df = SecureFileLoader.load_data(bom_file.getvalue(), bom_file.name, skiprows=1)
            purchase_df = SecureFileLoader.load_data(purchase_file.getvalue(), purchase_file.name)
        
        if bom_df is None or purchase_df is None:
            st.error("❌ 파일 로딩에 실패했습니다.")
            st.stop()
        
        # 데이터 검증
        bom_valid, bom_errors = DataValidator.validate_bom_schema(bom_df)
        purchase_valid, purchase_error = DataValidator.validate_purchase_data(purchase_df)
        
        if not bom_valid:
            st.error("❌ BOM 데이터 검증 실패:")
            for error in bom_errors:
                st.error(f"  • {error}")
            st.stop()
        
        if not purchase_valid:
            st.error(f"❌ 구매 데이터 검증 실패: {purchase_error}")
            st.stop()
        
        # 데이터 미리보기
        st.header("2. 📋 데이터 미리보기")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 BOM 데이터")
            st.info(f"📊 **{len(bom_df):,}**행 × **{len(bom_df.columns)}**열")
            
            if show_debug:
                st.write("**컬럼명:**", list(bom_df.columns))
            
            st.dataframe(bom_df.head(), use_container_width=True)
            
        with col2:
            st.subheader("💰 구매 데이터")
            st.info(f"📊 **{len(purchase_df):,}**행 × **{len(purchase_df.columns)}**열")
            
            if show_debug:
                st.write("**컬럼명:**", list(purchase_df.columns))
            
            st.dataframe(purchase_df.head(), use_container_width=True)
        
        # 원가 계산 실행
        st.header("3. 🔥 BOM 원가 계산")
        
        if st.button("🚀 원가 계산 시작!", type="primary", use_container_width=True):
            
            # 1단계: 구매 단가 추출
            with st.spinner("💰 구매 단가 추출 중..."):
                purchase_prices = PurchasePriceExtractor.extract_prices(purchase_df)
            
            if not purchase_prices:
                st.error("❌ 구매 단가를 추출할 수 없습니다.")
                st.stop()
            
            st.success(f"✅ 구매 단가 추출 완료: **{len(purchase_prices):,}**개 품목")
            
            if show_debug:
                st.write("**구매 단가 샘플:**", dict(list(purchase_prices.items())[:5]))
            
            # 2단계: BOM 원가 계산
            calculator = BOMCostCalculator()
            
            with st.spinner("🔄 BOM 원가 계산 중..."):
                result_df, details_df = calculator.calculate_all_costs(bom_df, purchase_prices)
            
            if result_df.empty:
                st.error("❌ BOM 원가 계산에 실패했습니다.")
                st.stop()
            
            # 3단계: 결과 분석 및 표시
            st.header("4. 🎯 완제품 원가 계산 결과")
            
            # 완제품 필터링
            finished_goods = result_df[
                result_df['생산품목명'].str.contains('[완제품]', regex=False, na=False)
            ].copy()
            
            # 통계 표시
            total_finished = len(finished_goods)
            calculated_finished = len(finished_goods[finished_goods['계산상태'] == '계산완료'])
            success_rate = (calculated_finished / total_finished * 100) if total_finished > 0 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("🎯 전체 완제품", f"{total_finished:,}개")
            with col2:
                st.metric("✅ 계산 성공", f"{calculated_finished:,}개")
            with col3:
                st.metric("❌ 계산 실패", f"{total_finished - calculated_finished:,}개")
            with col4:
                st.metric("📊 성공률", f"{success_rate:.1f}%")
            
            # 결과 테이블
            if not finished_goods.empty:
                display_df = finished_goods[['생산품목코드', '생산품목명', '계산된단위원가', '계산상태']].copy()
                display_df.columns = ['품목코드', '품목명', '단위원가(원)', '상태']
                
                # 조건부 스타일링
                def highlight_status(row):
                    if row['상태'] == '계산완료':
                        return ['background-color: #d4edda; color: #155724'] * len(row)
                    else:
                        return ['background-color: #f8d7da; color: #721c24'] * len(row)
                
                styled_df = display_df.style.apply(highlight_status, axis=1).format({
                    '단위원가(원)': '{:,.0f}'
                })
                
                st.dataframe(styled_df, use_container_width=True, height=400)
                
                # 원가 분석
                calculated_items = finished_goods[finished_goods['계산상태'] == '계산완료']
                if not calculated_items.empty:
                    avg_cost = calculated_items['계산된단위원가'].mean()
                    max_cost = calculated_items['계산된단위원가'].max()
                    min_cost = calculated_items[calculated_items['계산된단위원가'] > 0]['계산된단위원가'].min()
                    
                    st.subheader("📈 원가 분석 요약")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("💰 평균 원가", f"{avg_cost:,.0f}원")
                    with col2:
                        st.metric("📈 최고 원가", f"{max_cost:,.0f}원")
                    with col3:
                        st.metric("📉 최저 원가", f"{min_cost:,.0f}원")
                    
                    # 상위 원가 제품
                    top_cost_items = calculated_items.nlargest(10, '계산된단위원가')
                    
                    with st.expander("💎 원가 상위 10개 완제품", expanded=True):
                        for idx, (_, item) in enumerate(top_cost_items.iterrows(), 1):
                            st.write(f"**{idx}.** {item['생산품목코드']} - **{item['계산된단위원가']:,.0f}원** - {item['생산품목명']}")
                
                # 시각화 차트
                if show_charts and plotly_available:
                    st.header("5. 📊 데이터 시각화")
                    DataVisualizer.create_cost_analysis_charts(finished_goods)
            
            else:
                st.warning("⚠️ 완제품 데이터가 없습니다.")
            
            # 계산 실패 항목 분석
            failed_items = finished_goods[finished_goods['계산상태'] == '계산불가']
            if not failed_items.empty:
                with st.expander(f"⚠️ 계산 실패 항목 {len(failed_items):,}개"):
                    st.warning("다음 완제품들은 구성 부품의 원가 정보 부족으로 계산할 수 없었습니다:")
                    st.dataframe(
                        failed_items[['생산품목코드', '생산품목명']], 
                        use_container_width=True
                    )
            
            # 디버깅 정보
            if show_debug and hasattr(calculator, 'calculation_log'):
                with st.expander("🔍 계산 로그 (디버깅용)"):
                    if calculator.calculation_log:
                        debug_df = pd.DataFrame(calculator.calculation_log)
                        st.dataframe(debug_df, use_container_width=True)
                    else:
                        st.info("계산 로그가 없습니다.")
            
            # 결과 다운로드
            st.header("6. 📥 결과 다운로드")
            
            with st.spinner("📊 엑셀 파일 생성 중..."):
                excel_data = ExcelExporter.export_results(finished_goods, result_df, details_df)
            
            if excel_data:
                st.download_button(
                    label="📊 BOM 원가 계산 결과 다운로드 (Excel)",
                    data=excel_data,
                    file_name=f'BOM원가계산_안정성강화_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    type="primary",
                    use_container_width=True
                )
            else:
                # CSV 대안 제공
                csv_data = finished_goods.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📄 완제품 원가 결과 다운로드 (CSV)",
                    data=csv_data,
                    file_name=f'BOM원가계산_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                    mime='text/csv',
                    use_container_width=True
                )
            
            # 성공 메시지
            st.balloons()
            st.success("🎉 BOM 원가 계산이 성공적으로 완료되었습니다!")
            
            if logging_available:
                logger.info("BOM 원가 계산 프로세스 완료")
    
    else:
        st.info("👆 BOM 데이터와 구매 데이터 파일을 모두 업로드해주세요.")
        
        # 사용법 안내
        with st.expander("📖 사용법 안내", expanded=True):
            st.markdown("""
            ### 📋 필수 데이터 형식
            
            **BOM 데이터:**
            - `생산품목코드`: 생산할 제품의 코드
            - `생산품목명`: 생산할 제품의 이름 (완제품은 '[완제품]' 포함)
            - `소모품목코드`: 필요한 부품/원료의 코드  
            - `소모품목명`: 필요한 부품/원료의 이름
            - `소요량`: 제품 1개 생산에 필요한 부품 수량
            
            **구매 데이터:**
            - `일자-No.` (또는 일자): 구매 일자
            - `품목코드`: 구매한 품목의 코드
            - `단가`: 품목의 단위 가격
            
            ### ⚡ 주요 기능
            - 🔄 **다단계 BOM 계산**: A제품→B중간재→C원료 형태의 복잡한 구조 지원
            - 📊 **실시간 진행률**: 대용량 데이터 처리 시 진행상황 표시
            - 🛡️ **데이터 검증**: 파일 형식, 스키마, 데이터 품질 자동 검증
            - 📈 **시각화 분석**: 원가 분포, 상위 제품 등 다양한 차트 제공
            - 📥 **포맷팅된 출력**: 컬럼 너비, 색상, 스타일이 적용된 엑셀 파일
            """)

if __name__ == "__main__":
    main()
