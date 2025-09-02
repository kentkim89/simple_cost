"""
BOM 원가 계산기 - 자동화 버전
SharePoint BOM 데이터 자동 로딩 + 구매 데이터 업로드만으로 즉시 계산
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import requests
import json
from typing import Dict, Optional, Tuple
from datetime import datetime
import time
import warnings

# 경고 필터링
warnings.filterwarnings('ignore')

# 선택적 import (없어도 동작)
try:
    from tqdm import stqdm
    HAS_PROGRESS = True
except ImportError:
    HAS_PROGRESS = False

try:
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

class Config:
    """설정 클래스"""
    MAX_FILE_SIZE_MB = 100
    REQUIRED_BOM_COLS = ['생산품목코드', '생산품목명', '소모품목코드', '소모품목명', '소요량']
    TEST_ITEM_CODE = '99701'

class SharePointClient:
    """SharePoint 자동 연동 클래스"""
    
    def __init__(self):
        self.access_token = None
        self.token_expires_at = 0
        
    def get_access_token(self) -> Optional[str]:
        """Azure AD 토큰 획득 (자동, 무출력)"""
        try:
            # Streamlit secrets에서 설정 읽기
            tenant_id = st.secrets["sharepoint"]["tenant_id"]
            client_id = st.secrets["sharepoint"]["client_id"]
            client_secret = st.secrets["sharepoint"]["client_secret"]
            
            # 토큰이 유효한지 확인 (5분 여유)
            if self.access_token and time.time() < self.token_expires_at - 300:
                return self.access_token
            
            # 새 토큰 요청
            token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
            
            token_data = {
                'grant_type': 'client_credentials',
                'client_id': client_id,
                'client_secret': client_secret,
                'scope': 'https://graph.microsoft.com/.default'
            }
            
            response = requests.post(token_url, data=token_data)
            response.raise_for_status()
            
            token_info = response.json()
            self.access_token = token_info['access_token']
            self.token_expires_at = time.time() + token_info.get('expires_in', 3600)
            
            return self.access_token
            
        except Exception:
            return None
    
    def get_site_id(self, site_name: str) -> Optional[str]:
        """사이트 ID 획득 (자동, 무출력)"""
        try:
            token = self.get_access_token()
            if not token:
                return None
            
            headers = {
                'Authorization': f'Bearer {token}',
                'Accept': 'application/json'
            }
            
            site_url = f"https://graph.microsoft.com/v1.0/sites/goremi.sharepoint.com:/sites/{site_name}"
            response = requests.get(site_url, headers=headers)
            response.raise_for_status()
            
            site_info = response.json()
            return site_info['id']
            
        except Exception:
            return None
    
    def auto_download_bom_data(self) -> Optional[bytes]:
        """BOM 데이터 자동 다운로드 (백그라운드)"""
        try:
            token = self.get_access_token()
            if not token:
                return None
            
            site_name = st.secrets["sharepoint_files"]["site_name"]
            file_name = st.secrets["sharepoint_files"]["file_name"]
            
            site_id = self.get_site_id(site_name)
            if not site_id:
                return None
            
            headers = {
                'Authorization': f'Bearer {token}',
                'Accept': 'application/json'
            }
            
            # 드라이브 조회
            drives_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drives"
            drives_response = requests.get(drives_url, headers=headers)
            drives_response.raise_for_status()
            
            drives = drives_response.json()['value']
            if not drives:
                return None
            
            file_item = None
            drive_id = None
            
            # 모든 드라이브에서 파일 검색
            for drive in drives:
                try:
                    drive_id = drive['id']
                    
                    # 검색 API 사용
                    search_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/root/search(q='{file_name}')"
                    search_response = requests.get(search_url, headers=headers)
                    
                    if search_response.status_code == 200:
                        search_results = search_response.json().get('value', [])
                        if search_results:
                            file_item = search_results[0]
                            break
                    
                    # 루트 디렉터리 직접 조회
                    if not file_item:
                        root_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/root/children"
                        root_response = requests.get(root_url, headers=headers)
                        
                        if root_response.status_code == 200:
                            root_files = root_response.json().get('value', [])
                            for f in root_files:
                                if f['name'].lower() == file_name.lower():
                                    file_item = f
                                    break
                    
                    if file_item:
                        break
                        
                except Exception:
                    continue
            
            if not file_item:
                return None
            
            # 다운로드 URL 생성
            download_url = None
            
            if '@microsoft.graph.downloadUrl' in file_item:
                download_url = file_item['@microsoft.graph.downloadUrl']
            elif 'id' in file_item:
                download_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{file_item['id']}/content"
            else:
                download_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/root:/{file_name}:/content"
            
            if not download_url:
                return None
            
            # 파일 다운로드
            if 'graph.microsoft.com' in download_url and '/content' in download_url:
                download_headers = headers.copy()
            else:
                download_headers = {}
            
            file_response = requests.get(download_url, headers=download_headers)
            file_response.raise_for_status()
            
            if len(file_response.content) == 0:
                return None
            
            return file_response.content
            
        except Exception:
            return None

def validate_file_size(file_content: bytes, max_mb: int = 100) -> bool:
    """파일 크기 검증"""
    try:
        size_mb = len(file_content) / (1024 * 1024)
        return size_mb <= max_mb
    except Exception:
        return False

def safe_load_data(file_content: bytes, file_name: str, skiprows: int = 0) -> Optional[pd.DataFrame]:
    """안전한 파일 로딩"""
    try:
        file_obj = io.BytesIO(file_content)
        
        # 파일 형식별 로딩
        if file_name.lower().endswith('.csv'):
            # CSV 인코딩 시도
            for encoding in ['utf-8-sig', 'utf-8', 'cp949', 'euc-kr']:
                try:
                    file_obj.seek(0)
                    df = pd.read_csv(file_obj, skiprows=skiprows, encoding=encoding, dtype=str)
                    break
                except:
                    continue
            else:
                return None
                
        elif file_name.lower().endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_obj, skiprows=skiprows, dtype=str)
        else:
            return None
        
        # 헤더 문제 해결 (구매 데이터용)
        if 'purchase' in file_name.lower() or any('Unnamed:' in str(col) for col in df.columns):
            df = fix_purchase_data_headers(df)
        
        # 데이터 정제
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str).str.strip()
        
        # 빈 행/열 제거
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        return df if not df.empty else None
        
    except Exception:
        return None

def fix_purchase_data_headers(df: pd.DataFrame) -> pd.DataFrame:
    """구매 데이터 헤더 문제 해결"""
    try:
        # 첫 번째 행에서 실제 헤더 찾기
        potential_headers = None
        
        # 0행부터 3행까지 헤더 후보 검색
        for i in range(min(4, len(df))):
            row_values = df.iloc[i].fillna('').astype(str).tolist()
            row_text = ' '.join(row_values)
            
            # 헤더의 특징적인 키워드들 확인
            header_keywords = ['일자', '품목코드', '품목명', '단가', '수량', '거래처']
            keyword_count = sum(1 for keyword in header_keywords if keyword in row_text)
            
            if keyword_count >= 3:  # 3개 이상의 키워드가 있으면 헤더로 판단
                potential_headers = row_values
                header_row_idx = i
                break
        
        if potential_headers:
            # 헤더 정리 및 적용
            cleaned_headers = []
            for header in potential_headers:
                # 헤더명 정리
                header = str(header).strip()
                if header in ['', 'nan', 'None']:
                    header = f"컬럼_{len(cleaned_headers)+1}"
                
                cleaned_headers.append(header)
            
            # 새로운 DataFrame 생성
            new_df = df.iloc[header_row_idx + 1:].copy()
            new_df.columns = cleaned_headers[:len(new_df.columns)]
            new_df = new_df.reset_index(drop=True)
            
            # 빈 행 제거
            new_df = new_df.dropna(how='all')
            
            return new_df
        
        else:
            return df
        
    except Exception:
        return df

def validate_bom_data(df: pd.DataFrame) -> Tuple[bool, str]:
    """BOM 데이터 간단 검증"""
    try:
        # 필수 컬럼 확인
        missing_cols = [col for col in Config.REQUIRED_BOM_COLS if col not in df.columns]
        if missing_cols:
            return False, f"필수 컬럼 누락: {missing_cols}"
        
        # 데이터 존재 확인
        if df.empty:
            return False, "데이터가 비어있습니다"
        
        return True, "검증 통과"
        
    except Exception as e:
        return False, f"검증 오류: {e}"

def extract_purchase_prices(df: pd.DataFrame) -> Dict[str, float]:
    """구매 데이터에서 단가 추출 (자동 감지)"""
    try:
        if df.empty:
            return {}
        
        # 컬럼 자동 감지
        date_col, item_col, price_col = None, None, None
        
        # 각 컬럼명을 분석하여 매칭
        for col in df.columns:
            col_str = str(col).lower()
            
            # 일자 컬럼 감지
            if not date_col:
                if any(keyword in col_str for keyword in ['일자', 'date', '날짜']) or '일자-no' in col_str:
                    date_col = col
            
            # 품목코드 컬럼 감지  
            if not item_col:
                if '품목코드' in col_str:
                    item_col = col
            
            # 단가 컬럼 감지
            if not price_col:
                if '단가' in col_str and '공급' not in col_str and '총' not in col_str:
                    price_col = col
        
        # 컬럼을 찾지 못한 경우 인덱스로 대체
        if not date_col and len(df.columns) > 0:
            date_col = df.columns[0]
            
        if not item_col and len(df.columns) > 1:
            item_col = df.columns[1]
            
        if not price_col:
            # 단가 관련 컬럼 우선 탐색
            for i, col in enumerate(df.columns):
                if i >= 3:  # 3번째 컬럼부터
                    sample_value = str(df[col].dropna().iloc[0] if not df[col].dropna().empty else '')
                    # 숫자로 변환 가능한 컬럼 찾기
                    try:
                        float(sample_value.replace(',', ''))
                        price_col = col
                        break
                    except:
                        continue
            
            # 그래도 없으면 기본값
            if not price_col and len(df.columns) > 5:
                price_col = df.columns[5]
        
        if not all([date_col, item_col, price_col]):
            return {}
        
        # 데이터 정제
        work_df = df[[date_col, item_col, price_col]].copy()
        work_df.columns = ['date', 'item_code', 'price']
        
        # 빈값 제거
        work_df = work_df.dropna()
        
        # 타입 변환
        work_df['item_code'] = work_df['item_code'].astype(str).str.strip()
        work_df['price'] = pd.to_numeric(work_df['price'], errors='coerce')
        
        # 유효한 데이터만
        work_df = work_df[
            (work_df['item_code'] != '') & 
            (work_df['item_code'] != 'nan') &
            (work_df['price'] > 0) & 
            (work_df['price'].notna())
        ]
        
        if work_df.empty:
            return {}
        
        # 날짜 처리 (간단하게)
        try:
            work_df['date_str'] = work_df['date'].astype(str).str.split('-').str[0]
            work_df['date_parsed'] = pd.to_datetime(work_df['date_str'], errors='coerce')
            work_df = work_df.dropna(subset=['date_parsed'])
            work_df = work_df.sort_values('date_parsed', ascending=False)
        except:
            pass
        
        # 최신 단가 추출
        latest_prices = work_df.drop_duplicates(subset='item_code', keep='first')
        
        # 딕셔너리 변환
        price_dict = {}
        for _, row in latest_prices.iterrows():
            code = row['item_code']
            price = row['price']
            if pd.notna(price) and price > 0:
                price_dict[code] = float(price)
        
        return price_dict
        
    except Exception:
        return {}

def clean_bom_data(df: pd.DataFrame) -> pd.DataFrame:
    """BOM 데이터 정제"""
    try:
        clean_df = df.copy()
        
        # 필수 컬럼 확인
        missing_cols = [col for col in Config.REQUIRED_BOM_COLS if col not in clean_df.columns]
        if missing_cols:
            return pd.DataFrame()
        
        # 데이터 정제
        clean_df['생산품목코드'] = clean_df['생산품목코드'].astype(str).str.strip()
        clean_df['소모품목코드'] = clean_df['소모품목코드'].astype(str).str.strip()
        clean_df['소요량'] = pd.to_numeric(clean_df['소요량'], errors='coerce').fillna(0.0)
        
        # test 품목 제거
        clean_df = clean_df[clean_df['소모품목코드'] != Config.TEST_ITEM_CODE]
        
        # 유효하지 않은 데이터 제거
        clean_df = clean_df[
            (clean_df['생산품목코드'] != '') &
            (clean_df['소모품목코드'] != '') &
            (clean_df['생산품목코드'] != 'nan') &
            (clean_df['소모품목코드'] != 'nan') &
            (clean_df['소요량'] >= 0)
        ]
        
        return clean_df
        
    except Exception:
        return pd.DataFrame()

def calculate_product_cost_with_reason(product_code: str, bom_df: pd.DataFrame, all_costs: Dict[str, float], cache: Dict[str, float]) -> Tuple[float, str]:
    """단일 제품 원가 계산 + 실패 이유 분석"""
    try:
        # 캐시 확인
        if product_code in cache:
            return cache[product_code], ""
        
        # BOM 구성요소 가져오기
        components = bom_df[bom_df['생산품목코드'] == product_code]
        
        if components.empty:
            cache[product_code] = 0.0
            return 0.0, "BOM 구성요소 없음"
        
        total_cost = 0.0
        missing_components = []
        zero_price_components = []
        invalid_quantity_components = []
        
        for _, comp in components.iterrows():
            comp_code = comp['소모품목코드']
            comp_name = comp['소모품목명']
            quantity = float(comp['소요량'])
            
            # 수량 검증
            if quantity <= 0:
                invalid_quantity_components.append(f"{comp_name}({comp_code})")
                continue
            
            # 부품 단가 찾기
            if comp_code in all_costs:
                unit_price = all_costs[comp_code]
                if unit_price <= 0:
                    zero_price_components.append(f"{comp_name}({comp_code})")
                    continue
            elif comp_code in bom_df['생산품목코드'].values:
                # 재귀 계산
                unit_price, _ = calculate_product_cost_with_reason(comp_code, bom_df, all_costs, cache)
                if unit_price <= 0:
                    missing_components.append(f"{comp_name}({comp_code})")
                    continue
                all_costs[comp_code] = unit_price
            else:
                missing_components.append(f"{comp_name}({comp_code})")
                continue
            
            total_cost += quantity * unit_price
        
        cache[product_code] = total_cost
        
        # 실패 이유 분석
        if total_cost == 0:
            reasons = []
            if missing_components:
                reasons.append(f"단가정보 없음: {', '.join(missing_components[:3])}{'...' if len(missing_components) > 3 else ''}")
            if zero_price_components:
                reasons.append(f"단가 0원: {', '.join(zero_price_components[:3])}{'...' if len(zero_price_components) > 3 else ''}")
            if invalid_quantity_components:
                reasons.append(f"수량 오류: {', '.join(invalid_quantity_components[:3])}{'...' if len(invalid_quantity_components) > 3 else ''}")
            
            failure_reason = " | ".join(reasons) if reasons else "알 수 없는 이유"
            return 0.0, failure_reason
        
        return total_cost, ""
        
    except Exception as e:
        return 0.0, f"계산 오류: {str(e)}"

def calculate_all_bom_costs(bom_df: pd.DataFrame, purchase_prices: Dict[str, float]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """전체 BOM 원가 계산"""
    try:
        start_time = time.time()
        
        # 데이터 정제
        clean_bom = clean_bom_data(bom_df)
        
        if clean_bom.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        # 모든 생산품목
        all_products = clean_bom[['생산품목코드', '생산품목명']].drop_duplicates().reset_index(drop=True)
        
        # 전체 원가 딕셔너리
        all_costs = purchase_prices.copy()
        calc_cache = {}
        failure_reasons = {}
        
        # 계산 실행
        results = []
        
        if HAS_PROGRESS:
            iterator = stqdm(all_products.iterrows(), total=len(all_products), desc="BOM 원가 계산")
        else:
            iterator = all_products.iterrows()
            progress_bar = st.progress(0)
        
        for idx, (_, product) in enumerate(iterator):
            product_code = product['생산품목코드']
            product_name = product['생산품목명']
            
            # 원가 계산
            calculated_cost, failure_reason = calculate_product_cost_with_reason(
                product_code, clean_bom, all_costs, calc_cache
            )
            
            if calculated_cost == 0 and failure_reason:
                failure_reasons[product_code] = failure_reason
            
            results.append({
                '생산품목코드': product_code,
                '생산품목명': product_name,
                '계산된단위원가': calculated_cost,
                '계산상태': '계산완료' if calculated_cost > 0 else '계산불가',
                '실패이유': failure_reasons.get(product_code, '')
            })
            
            # 진행률 업데이트
            if not HAS_PROGRESS:
                progress_bar.progress((idx + 1) / len(all_products))
        
        if not HAS_PROGRESS:
            progress_bar.empty()
        
        # 결과 DataFrame
        result_df = pd.DataFrame(results)
        
        # 상세 내역
        details_df = clean_bom.copy()
        details_df['부품단가'] = details_df['소모품목코드'].apply(lambda x: all_costs.get(x, 0.0))
        details_df['부품별원가'] = details_df['소요량'] * details_df['부품단가']
        
        return result_df, details_df
        
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

def create_simple_chart(df: pd.DataFrame) -> None:
    """간단한 차트 생성"""
    if not HAS_PLOTLY or df.empty:
        return
    
    try:
        # 계산된 완제품만
        chart_data = df[df['계산된단위원가'] > 0]
        
        if len(chart_data) < 5:
            return
        
        # 상위 20개 제품 바차트
        top_items = chart_data.nlargest(20, '계산된단위원가')
        
        fig = px.bar(
            top_items,
            x='계산된단위원가',
            y='생산품목코드',
            orientation='h',
            title='원가 상위 20개 완제품',
            labels={'계산된단위원가': '원가 (원)', '생산품목코드': '제품코드'}
        )
        fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception:
        pass

def export_to_excel(finished_goods: pd.DataFrame, all_results: pd.DataFrame, details: pd.DataFrame) -> bytes:
    """엑셀 내보내기"""
    try:
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # 완제품 원가 시트
            finished_display = finished_goods.copy()
            
            if '실패이유' in finished_display.columns:
                finished_display = finished_display[['생산품목코드', '생산품목명', '계산된단위원가', '계산상태', '실패이유']]
                finished_display.columns = ['품목코드', '품목명', '단위원가(원)', '계산상태', '실패이유']
            else:
                finished_display = finished_display[['생산품목코드', '생산품목명', '계산된단위원가', '계산상태']]
                finished_display.columns = ['품목코드', '품목명', '단위원가(원)', '계산상태']
            
            finished_display.to_excel(writer, sheet_name='완제품원가', index=False)
            
            # 전체 제품 원가 시트
            all_display = all_results.copy()
            if '실패이유' in all_display.columns:
                all_display = all_display[['생산품목코드', '생산품목명', '계산된단위원가', '계산상태', '실패이유']]
                all_display.columns = ['품목코드', '품목명', '단위원가(원)', '계산상태', '실패이유']
            
            all_display.to_excel(writer, sheet_name='전체제품원가', index=False)
            
            # 상세 내역 시트
            details_display = details.copy()
            details_cols = ['생산품목코드', '생산품목명', '소모품목코드', '소모품목명', '소요량', '부품단가', '부품별원가']
            details_display = details_display[details_cols]
            details_display.columns = ['생산품목코드', '생산품목명', '부품코드', '부품명', '소요량', '부품단가(원)', '부품원가(원)']
            
            details_display.to_excel(writer, sheet_name='상세내역', index=False)
        
        return output.getvalue()
        
    except Exception:
        return b''

def main():
    """메인 애플리케이션 - 완전 자동화"""
    
    st.set_page_config(
        page_title="BOM 원가 계산기",
        page_icon="🏭",
        layout="wide"
    )
    
    st.title("🏭 BOM 원가 계산기")
    st.markdown("**🚀 SharePoint 자동 연동 + 구매 데이터만 업로드하면 즉시 계산!**")
    
    # SharePoint 클라이언트 초기화 및 자동 BOM 로딩
    sharepoint_client = SharePointClient()
    
    # 세션 상태에 BOM 데이터가 없으면 자동 로딩
    if 'auto_bom_data' not in st.session_state:
        with st.spinner("🔄 SharePoint에서 BOM 데이터 자동 로딩 중..."):
            bom_content = sharepoint_client.auto_download_bom_data()
            
            if bom_content and validate_file_size(bom_content):
                bom_df = safe_load_data(bom_content, st.secrets["sharepoint_files"]["file_name"], skiprows=1)
                
                if bom_df is not None:
                    bom_valid, _ = validate_bom_data(bom_df)
                    
                    if bom_valid:
                        st.session_state['auto_bom_data'] = bom_df
                        st.success("✅ SharePoint BOM 데이터 자동 로딩 완료!")
                    else:
                        st.error("❌ SharePoint BOM 데이터 검증 실패")
                else:
                    st.error("❌ SharePoint BOM 데이터 파싱 실패")
            else:
                st.error("❌ SharePoint 연결 실패 - 수동으로 BOM 파일을 업로드해주세요")
    
    # BOM 데이터 상태 표시
    if 'auto_bom_data' in st.session_state:
        bom_df = st.session_state['auto_bom_data']
        clean_bom = clean_bom_data(bom_df.copy())
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📋 BOM 데이터", "✅ 로딩완료")
        with col2:
            st.metric("🏭 생산품목", f"{clean_bom['생산품목코드'].nunique():,}개")
        with col3:
            st.metric("🧩 소모품목", f"{clean_bom['소모품목코드'].nunique():,}개")
        
        # 구매 데이터 업로드
        st.header("📥 구매 데이터 업로드")
        
        purchase_file = st.file_uploader(
            "💰 구매 데이터 파일을 업로드하세요 (CSV, Excel)", 
            type=['csv', 'xlsx', 'xls'],
            help="구매 데이터 파일을 업로드하면 자동으로 BOM 원가 계산이 시작됩니다."
        )
        
        if purchase_file:
            # 자동 계산 시작
            with st.spinner("🔄 구매 데이터 처리 및 BOM 원가 계산 중..."):
                
                # 구매 데이터 로딩
                purchase_df = safe_load_data(purchase_file.getvalue(), purchase_file.name)
                
                if purchase_df is not None:
                    # 구매 단가 추출
                    purchase_prices = extract_purchase_prices(purchase_df)
                    
                    if purchase_prices:
                        # BOM 원가 계산
                        result_df, details_df = calculate_all_bom_costs(bom_df, purchase_prices)
                        
                        if not result_df.empty:
                            # 완제품 필터링
                            finished_goods = result_df[
                                result_df['생산품목명'].str.contains('[완제품]', regex=False, na=False)
                            ].copy()
                            
                            # 계산 완료 알림
                            st.balloons()
                            st.success("🎉 BOM 원가 계산 완료!")
                            
                            # 결과 표시
                            st.header("🎯 계산 결과")
                            
                            # 통계
                            total = len(finished_goods)
                            calculated = len(finished_goods[finished_goods['계산상태'] == '계산완료'])
                            success_rate = (calculated / total * 100) if total > 0 else 0
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("🎯 완제품", f"{total:,}개")
                            with col2:
                                st.metric("✅ 계산성공", f"{calculated:,}개") 
                            with col3:
                                st.metric("📊 성공률", f"{success_rate:.1f}%")
                            with col4:
                                st.metric("💰 구매단가", f"{len(purchase_prices):,}개")
                            
                            # 결과 테이블
                            if not finished_goods.empty:
                                # 실패 이유 포함한 컬럼 구성
                                if '실패이유' in finished_goods.columns:
                                    display_df = finished_goods[['생산품목코드', '생산품목명', '계산된단위원가', '계산상태', '실패이유']].copy()
                                    display_df.columns = ['품목코드', '품목명', '단위원가(원)', '상태', '실패이유']
                                else:
                                    display_df = finished_goods[['생산품목코드', '생산품목명', '계산된단위원가', '계산상태']].copy()
                                    display_df.columns = ['품목코드', '품목명', '단위원가(원)', '상태']
                                
                                # 스타일링
                                def highlight_rows(row):
                                    if row['상태'] == '계산완료':
                                        return ['background-color: #d4edda'] * len(row)
                                    else:
                                        return ['background-color: #f8d7da'] * len(row)
                                
                                styled_df = display_df.style.apply(highlight_rows, axis=1).format({
                                    '단위원가(원)': '{:,.0f}'
                                })
                                
                                st.dataframe(styled_df, use_container_width=True, height=400)
                                
                                # 원가 통계
                                calculated_items = finished_goods[finished_goods['계산상태'] == '계산완료']
                                if not calculated_items.empty:
                                    avg_cost = calculated_items['계산된단위원가'].mean()
                                    max_cost = calculated_items['계산된단위원가'].max()
                                    min_cost = calculated_items[calculated_items['계산된단위원가'] > 0]['계산된단위원가'].min()
                                    
                                    st.subheader("📈 원가 분석")
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        st.metric("💰 평균 원가", f"{avg_cost:,.0f}원")
                                    with col2:
                                        st.metric("📈 최고 원가", f"{max_cost:,.0f}원")
                                    with col3:
                                        st.metric("📉 최저 원가", f"{min_cost:,.0f}원")
                                
                                # 시각화
                                if HAS_PLOTLY:
                                    st.subheader("📊 원가 분석 차트")
                                    create_simple_chart(calculated_items)
                            
                            # 결과 다운로드
                            st.header("📥 결과 다운로드")
                            
                            excel_data = export_to_excel(finished_goods, result_df, details_df)
                            
                            if excel_data:
                                st.download_button(
                                    label="📊 BOM 원가 계산 결과 다운로드 (Excel)",
                                    data=excel_data,
                                    file_name=f'BOM원가계산_{datetime.now().strftime("%Y%m%d_%H%M")}.xlsx',
                                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                    use_container_width=True
                                )
                        else:
                            st.error("❌ BOM 원가 계산 실패")
                    else:
                        st.error("❌ 구매 단가 추출 실패 - 구매 데이터 형식을 확인해주세요")
                else:
                    st.error("❌ 구매 데이터 파일 로딩 실패")
    
    else:
        # SharePoint 로딩 실패 시 수동 업로드 옵션
        st.header("📁 수동 BOM 데이터 업로드")
        st.info("SharePoint 자동 연동에 실패했습니다. BOM 데이터를 수동으로 업로드해주세요.")
        
        bom_file = st.file_uploader("📋 BOM 데이터 파일", type=['csv', 'xlsx', 'xls'])
        purchase_file = st.file_uploader("💰 구매 데이터 파일", type=['csv', 'xlsx', 'xls'])
        
        if bom_file and purchase_file:
            # 수동 처리 로직
            with st.spinner("🔄 데이터 처리 및 계산 중..."):
                bom_df = safe_load_data(bom_file.getvalue(), bom_file.name, skiprows=1)
                purchase_df = safe_load_data(purchase_file.getvalue(), purchase_file.name)
                
                if bom_df is not None and purchase_df is not None:
                    bom_valid, _ = validate_bom_data(bom_df)
                    
                    if bom_valid:
                        purchase_prices = extract_purchase_prices(purchase_df)
                        
                        if purchase_prices:
                            result_df, details_df = calculate_all_bom_costs(bom_df, purchase_prices)
                            
                            if not result_df.empty:
                                finished_goods = result_df[
                                    result_df['생산품목명'].str.contains('[완제품]', regex=False, na=False)
                                ].copy()
                                
                                st.balloons()
                                st.success("🎉 BOM 원가 계산 완료!")
                                
                                # 간단한 결과 표시
                                total = len(finished_goods)
                                calculated = len(finished_goods[finished_goods['계산상태'] == '계산완료'])
                                success_rate = (calculated / total * 100) if total > 0 else 0
                                
                                st.metric("계산 결과", f"전체 {total}개 중 {calculated}개 성공 ({success_rate:.1f}%)")
                                
                                # 다운로드
                                excel_data = export_to_excel(finished_goods, result_df, details_df)
                                if excel_data:
                                    st.download_button(
                                        label="📊 계산 결과 다운로드",
                                        data=excel_data,
                                        file_name=f'BOM원가계산_{datetime.now().strftime("%Y%m%d_%H%M")}.xlsx',
                                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                                    )

if __name__ == "__main__":
    main()
