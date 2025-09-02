import streamlit as st
import pandas as pd
import numpy as np
import io
import requests
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import time
import warnings

warnings.filterwarnings('ignore')

def get_sharepoint_token():
    """Azure AD 토큰 획득"""
    try:
        tenant_id = st.secrets["sharepoint"]["tenant_id"]
        client_id = st.secrets["sharepoint"]["client_id"]
        client_secret = st.secrets["sharepoint"]["client_secret"]
        
        token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
        
        data = {
            'grant_type': 'client_credentials',
            'client_id': client_id,
            'client_secret': client_secret,
            'scope': 'https://graph.microsoft.com/.default'
        }
        
        response = requests.post(token_url, data=data)
        
        if response.status_code == 200:
            return response.json()['access_token']
        else:
            st.error(f"토큰 획득 실패: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"인증 오류: {e}")
        return None

def load_sharepoint_file():
    """SharePoint 파일 다운로드"""
    try:
        # 토큰 획득
        token = get_sharepoint_token()
        if not token:
            return None
        
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        
        # 여러 Graph API URL 시도
        urls_to_try = [
            # 방법 1: 사이트 이름으로 접근
            "https://graph.microsoft.com/v1.0/sites/goremi.sharepoint.com:/sites/data:/drive/root/children",
            
            # 방법 2: 사이트 검색
            "https://graph.microsoft.com/v1.0/sites?search=data"
        ]
        
        st.info("SharePoint 사이트 정보 조회 중...")
        
        # 먼저 사이트 정보 확인
        for url in urls_to_try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                st.success("사이트 접근 성공!")
                st.json(response.json())  # 디버깅용
                break
        else:
            st.error("사이트 접근 실패 - 직접 URL 시도")
            return load_direct_url()
        
        return None  # 일단 사이트 정보만 확인
            
    except Exception as e:
        st.error(f"파일 로드 오류: {e}")
        return load_direct_url()

def load_direct_url():
    """직접 URL로 파일 다운로드 (대안)"""
    try:
        st.info("대안 방법으로 직접 URL 접근 시도...")
        
        url = st.secrets["sharepoint_files"]["bom_file_url"]
        download_url = url + "&download=1"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(download_url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            # 응답 내용 확인 (디버깅)
            content_type = response.headers.get('content-type', '').lower()
            st.info(f"응답 타입: {content_type}")
            
            # HTML 응답인지 확인
            if 'html' in content_type or response.text.startswith('<!DOCTYPE'):
                st.error("HTML 응답 - 로그인이 필요하거나 권한 문제")
                st.text("응답 내용 일부:")
                st.code(response.text[:500])
                return None
            
            file_content = io.BytesIO(response.content)
            
            # Excel 파일로 시도
            try:
                df = pd.read_excel(file_content, skiprows=1, dtype=str, engine='openpyxl')
                st.success("Excel 파일로 읽기 성공")
            except Exception as e1:
                st.warning(f"Excel 읽기 실패: {e1}")
                # 파일 내용 확인
                file_content.seek(0)
                first_bytes = file_content.read(100)
                st.text(f"파일 시작 부분: {first_bytes}")
                return None
            
            # 데이터 정제
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = df[col].astype(str).str.strip()
            
            df = df.dropna(how='all').dropna(axis=1, how='all')
            
            st.success(f"대안 방법 성공: {len(df)}행")
            return df
        else:
            st.error(f"직접 URL 접근 실패: {response.status_code}")
            if response.status_code == 403:
                st.error("접근 권한이 없습니다. SharePoint 링크 권한을 확인하세요.")
            return None
            
    except Exception as e:
        st.error(f"대안 방법 실패: {e}")
        return None

# 기존 BOM 계산 함수들
def validate_bom_data(df):
    required_cols = ['생산품목코드', '생산품목명', '소모품목코드', '소모품목명', '소요량']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return False, f"필수 컬럼 누락: {missing_cols}"
    return True, "검증 통과"

def clean_bom_data(df):
    clean_df = df.copy()
    clean_df['생산품목코드'] = clean_df['생산품목코드'].astype(str).str.strip()
    clean_df['소모품목코드'] = clean_df['소모품목코드'].astype(str).str.strip()
    clean_df['소요량'] = pd.to_numeric(clean_df['소요량'], errors='coerce').fillna(0.0)
    
    clean_df = clean_df[
        (clean_df['생산품목코드'] != '') &
        (clean_df['소모품목코드'] != '') &
        (clean_df['생산품목코드'] != 'nan') &
        (clean_df['소모품목코드'] != 'nan') &
        (clean_df['소요량'] >= 0)
    ]
    return clean_df

def extract_purchase_prices(df):
    try:
        # 컬럼 자동 감지
        date_col, item_col, price_col = None, None, None
        
        for col in df.columns:
            col_str = str(col).lower()
            if not date_col and ('일자' in col_str or 'date' in col_str):
                date_col = col
            if not item_col and '품목코드' in col_str:
                item_col = col
            if not price_col and '단가' in col_str and '공급' not in col_str:
                price_col = col
        
        if not all([item_col, price_col]):
            st.error("필수 컬럼을 찾을 수 없습니다.")
            return {}
        
        work_df = df[[item_col, price_col]].copy()
        work_df.columns = ['item_code', 'price']
        work_df = work_df.dropna()
        
        work_df['item_code'] = work_df['item_code'].astype(str).str.strip()
        work_df['price'] = pd.to_numeric(work_df['price'], errors='coerce')
        
        work_df = work_df[
            (work_df['item_code'] != '') & 
            (work_df['item_code'] != 'nan') &
            (work_df['price'] > 0)
        ]
        
        latest_prices = work_df.drop_duplicates(subset='item_code', keep='first')
        
        price_dict = {}
        for _, row in latest_prices.iterrows():
            price_dict[row['item_code']] = float(row['price'])
        
        return price_dict
        
    except Exception as e:
        st.error(f"구매 단가 추출 오류: {e}")
        return {}

def calculate_product_cost(product_code, bom_df, all_costs, cache):
    if product_code in cache:
        return cache[product_code]
    
    components = bom_df[bom_df['생산품목코드'] == product_code]
    if components.empty:
        cache[product_code] = 0.0
        return 0.0
    
    total_cost = 0.0
    for _, comp in components.iterrows():
        comp_code = comp['소모품목코드']
        quantity = float(comp['소요량'])
        
        if quantity <= 0:
            continue
        
        if comp_code in all_costs:
            unit_price = all_costs[comp_code]
        elif comp_code in bom_df['생산품목코드'].values:
            unit_price = calculate_product_cost(comp_code, bom_df, all_costs, cache)
            all_costs[comp_code] = unit_price
        else:
            continue
        
        if unit_price > 0:
            total_cost += quantity * unit_price
    
    cache[product_code] = total_cost
    return total_cost

def calculate_all_bom_costs(bom_df, purchase_prices):
    clean_bom = clean_bom_data(bom_df)
    all_products = clean_bom[['생산품목코드', '생산품목명']].drop_duplicates()
    
    all_costs = purchase_prices.copy()
    calc_cache = {}
    results = []
    
    progress_bar = st.progress(0)
    
    for idx, (_, product) in enumerate(all_products.iterrows()):
        product_code = product['생산품목코드']
        product_name = product['생산품목명']
        
        calculated_cost = calculate_product_cost(product_code, clean_bom, all_costs, calc_cache)
        
        results.append({
            '생산품목코드': product_code,
            '생산품목명': product_name,
            '계산된단위원가': calculated_cost,
            '계산상태': '계산완료' if calculated_cost > 0 else '계산불가'
        })
        
        progress_bar.progress((idx + 1) / len(all_products))
    
    progress_bar.empty()
    return pd.DataFrame(results)

def main():
    st.set_page_config(page_title="BOM 원가 계산기", layout="wide")
    st.title("BOM 원가 계산기 (SharePoint 연동)")
    
    # BOM 데이터 로드
    st.header("1. BOM 데이터")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("SharePoint 연동")
        if st.button("SharePoint에서 BOM 데이터 로드"):
            bom_df = load_sharepoint_file()
            if bom_df is not None:
                st.session_state.bom_data = bom_df
                st.success(f"BOM 데이터 로드 성공: {len(bom_df)}행")
    
    with col2:
        st.subheader("임시 해결: 파일 업로드")
        bom_file = st.file_uploader("BOM 파일을 직접 업로드", type=['xlsx', 'xls'], key="bom_upload")
        if bom_file:
            try:
                bom_df = pd.read_excel(bom_file, skiprows=1, dtype=str, engine='openpyxl')
                st.session_state.bom_data = bom_df
                st.success(f"BOM 파일 업로드 성공: {len(bom_df)}행")
            except Exception as e:
                st.error(f"BOM 파일 읽기 실패: {e}")
    
    # 구매 데이터 업로드
    st.header("2. 구매 데이터")
    purchase_file = st.file_uploader("구매 데이터 파일", type=['csv', 'xlsx'])
    
    if purchase_file:
        purchase_df = pd.read_excel(purchase_file, dtype=str, engine='openpyxl')
        st.success(f"구매 데이터 로드: {len(purchase_df)}행")
        
        # BOM 계산 실행
        if 'bom_data' in st.session_state:
            bom_df = st.session_state.bom_data
            
            st.header("3. 원가 계산")
            if st.button("계산 시작", type="primary"):
                # BOM 데이터 검증
                if validate_bom_data(bom_df)[0]:
                    # 구매 단가 추출
                    with st.spinner("구매 단가 추출 중..."):
                        purchase_prices = extract_purchase_prices(purchase_df)
                    
                    if purchase_prices:
                        st.success(f"구매 단가 추출: {len(purchase_prices)}개")
                        
                        # BOM 원가 계산
                        with st.spinner("BOM 원가 계산 중..."):
                            result_df = calculate_all_bom_costs(bom_df, purchase_prices)
                        
                        if not result_df.empty:
                            # 완제품 필터링
                            finished_goods = result_df[
                                result_df['생산품목명'].str.contains('[완제품]', regex=False, na=False)
                            ]
                            
                            st.header("4. 계산 결과")
                            
                            # 통계
                            total = len(finished_goods)
                            calculated = len(finished_goods[finished_goods['계산상태'] == '계산완료'])
                            success_rate = (calculated / total * 100) if total > 0 else 0
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("전체 완제품", f"{total:,}개")
                            with col2:
                                st.metric("계산 성공", f"{calculated:,}개")
                            with col3:
                                st.metric("성공률", f"{success_rate:.1f}%")
                            
                            # 결과 테이블
                            st.dataframe(finished_goods)
                            
                            # 다운로드
                            csv_data = finished_goods.to_csv(index=False, encoding='utf-8-sig')
                            st.download_button(
                                "결과 다운로드 (CSV)",
                                csv_data,
                                f"BOM원가_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                mime="text/csv"
                            )
                        else:
                            st.error("BOM 원가 계산에 실패했습니다.")
                    else:
                        st.error("구매 단가를 추출할 수 없습니다. 구매 데이터 형식을 확인하세요.")
                else:
                    st.error("BOM 데이터가 올바르지 않습니다. 필수 컬럼을 확인하세요.")
        else:
            st.info("먼저 BOM 데이터를 로드해주세요.")

if __name__ == "__main__":
    main()
