import streamlit as st
import pandas as pd
import numpy as np
import io
import requests
from typing import Dict, Optional, Tuple
from datetime import datetime
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
            "https://graph.microsoft.com/v1.0/sites/goremi.sharepoint.com:/sites/data:/drive/root/children",
            "https://graph.microsoft.com/v1.0/sites?search=data"
        ]
        
        st.info("SharePoint 사이트 정보 조회 중...")
        
        for url in urls_to_try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                st.success("Graph API 연결 성공!")
                # 실제 파일 다운로드는 여기서 구현 가능
                return None  # 현재는 사이트 접근 확인만
        
        st.warning("Graph API 접근 실패 - 직접 URL 시도")
        return load_direct_url()
            
    except Exception as e:
        st.error(f"SharePoint 로드 오류: {e}")
        return load_direct_url()

def load_direct_url():
    """직접 URL로 파일 다운로드"""
    try:
        st.info("직접 URL 접근 시도...")
        
        url = st.secrets["sharepoint_files"]["bom_file_url"]
        download_url = url + "&download=1"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(download_url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            content_type = response.headers.get('content-type', '').lower()
            
            if 'html' in content_type:
                st.error("HTML 응답 - 로그인 필요하거나 권한 문제")
                return None
            
            file_content = io.BytesIO(response.content)
            df = pd.read_excel(file_content, skiprows=1, dtype=str, engine='openpyxl')
            
            # 데이터 정제
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = df[col].astype(str).str.strip()
            
            df = df.dropna(how='all').dropna(axis=1, how='all')
            
            st.success(f"파일 다운로드 성공: {len(df)}행")
            return df
        else:
            st.error(f"직접 URL 접근 실패: {response.status_code}")
            return None
            
    except Exception as e:
        st.error(f"파일 로드 실패: {e}")
        return None

def validate_bom_data(df):
    """BOM 데이터 검증"""
    required_cols = ['생산품목코드', '생산품목명', '소모품목코드', '소모품목명', '소요량']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return False, f"필수 컬럼 누락: {missing_cols}"
    if df.empty:
        return False, "데이터가 비어있습니다"
    return True, "검증 통과"

def clean_bom_data(df):
    """BOM 데이터 정제"""
    clean_df = df.copy()
    
    # 필수 컬럼 확인
    required_cols = ['생산품목코드', '생산품목명', '소모품목코드', '소모품목명', '소요량']
    missing_cols = [col for col in required_cols if col not in clean_df.columns]
    if missing_cols:
        st.error(f"필수 컬럼 누락: {missing_cols}")
        return pd.DataFrame()
    
    # 데이터 타입 변환
    clean_df['생산품목코드'] = clean_df['생산품목코드'].astype(str).str.strip()
    clean_df['소모품목코드'] = clean_df['소모품목코드'].astype(str).str.strip()
    clean_df['소요량'] = pd.to_numeric(clean_df['소요량'], errors='coerce').fillna(0.0)
    
    # 유효하지 않은 데이터 제거
    clean_df = clean_df[
        (clean_df['생산품목코드'] != '') &
        (clean_df['소모품목코드'] != '') &
        (clean_df['생산품목코드'] != 'nan') &
        (clean_df['소모품목코드'] != 'nan') &
        (clean_df['소요량'] >= 0)
    ]
    
    return clean_df

def extract_purchase_prices(df):
    """구매 데이터에서 단가 추출"""
    try:
        if df.empty:
            return {}
        
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
        
        # 기본값 설정
        if not date_col and len(df.columns) > 0:
            date_col = df.columns[0]
        if not item_col and len(df.columns) > 1:
            item_col = df.columns[1]
        if not price_col:
            for i, col in enumerate(df.columns):
                if i >= 3:
                    sample_value = str(df[col].dropna().iloc[0] if not df[col].dropna().empty else '')
                    try:
                        float(sample_value.replace(',', ''))
                        price_col = col
                        break
                    except:
                        continue
            if not price_col and len(df.columns) > 5:
                price_col = df.columns[5]
        
        if not all([item_col, price_col]):
            st.error("품목코드와 단가 컬럼을 찾을 수 없습니다.")
            return {}
        
        # 데이터 정제
        work_df = df[[item_col, price_col]].copy()
        work_df.columns = ['item_code', 'price']
        work_df = work_df.dropna()
        
        work_df['item_code'] = work_df['item_code'].astype(str).str.strip()
        work_df['price'] = pd.to_numeric(work_df['price'], errors='coerce')
        
        work_df = work_df[
            (work_df['item_code'] != '') & 
            (work_df['item_code'] != 'nan') &
            (work_df['price'] > 0) & 
            (work_df['price'].notna())
        ]
        
        if work_df.empty:
            st.error("유효한 구매 데이터가 없습니다.")
            return {}
        
        # 최신 단가 추출 (중복 제거)
        latest_prices = work_df.drop_duplicates(subset='item_code', keep='first')
        
        price_dict = {}
        for _, row in latest_prices.iterrows():
            code = row['item_code']
            price = row['price']
            if pd.notna(price) and price > 0:
                price_dict[code] = float(price)
        
        return price_dict
        
    except Exception as e:
        st.error(f"구매 단가 추출 오류: {e}")
        return {}

def calculate_product_cost(product_code, bom_df, all_costs, cache):
    """단일 제품 원가 계산"""
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
            # 재귀 계산
            unit_price = calculate_product_cost(comp_code, bom_df, all_costs, cache)
            all_costs[comp_code] = unit_price
        else:
            continue
        
        if unit_price > 0:
            total_cost += quantity * unit_price
    
    cache[product_code] = total_cost
    return total_cost

def calculate_all_bom_costs(bom_df, purchase_prices):
    """전체 BOM 원가 계산"""
    try:
        start_time = time.time()
        
        # 데이터 정제
        clean_bom = clean_bom_data(bom_df)
        if clean_bom.empty:
            return pd.DataFrame()
        
        # 모든 생산품목
        all_products = clean_bom[['생산품목코드', '생산품목명']].drop_duplicates().reset_index(drop=True)
        
        st.info(f"계산 대상: 생산품목 {len(all_products):,}개, 구매단가 {len(purchase_prices):,}개")
        
        # 전체 원가 딕셔너리
        all_costs = purchase_prices.copy()
        calc_cache = {}
        results = []
        
        progress_bar = st.progress(0)
        
        for idx, (_, product) in enumerate(all_products.iterrows()):
            product_code = product['생산품목코드']
            product_name = product['생산품목명']
            
            # 원가 계산
            calculated_cost = calculate_product_cost(product_code, clean_bom, all_costs, calc_cache)
            
            results.append({
                '생산품목코드': product_code,
                '생산품목명': product_name,
                '계산된단위원가': calculated_cost,
                '계산상태': '계산완료' if calculated_cost > 0 else '계산불가'
            })
            
            # 진행률 업데이트
            progress_bar.progress((idx + 1) / len(all_products))
        
        progress_bar.empty()
        
        # 결과 DataFrame
        result_df = pd.DataFrame(results)
        
        elapsed = time.time() - start_time
        st.success(f"계산 완료! (소요시간: {elapsed:.1f}초)")
        
        return result_df
        
    except Exception as e:
        st.error(f"BOM 원가 계산 실패: {e}")
        return pd.DataFrame()

def main():
    st.set_page_config(page_title="BOM 원가 계산기", layout="wide")
    st.title("BOM 원가 계산기 (SharePoint 연동)")
    
    # BOM 데이터 로드
    st.header("1. BOM 데이터 (SharePoint)")
    
    if st.button("SharePoint에서 BOM 데이터 로드"):
        bom_df = load_sharepoint_file()
        if bom_df is not None:
            st.session_state.bom_data = bom_df
            st.dataframe(bom_df.head(), use_container_width=True)
    
    # 구매 데이터 업로드
    st.header("2. 구매 데이터")
    purchase_file = st.file_uploader("구매 데이터 파일", type=['csv', 'xlsx'])
    
    purchase_df = None
    if purchase_file:
        try:
            purchase_df = pd.read_excel(purchase_file, dtype=str, engine='openpyxl')
            st.success(f"구매 데이터 로드: {len(purchase_df)}행")
            st.dataframe(purchase_df.head(), use_container_width=True)
        except Exception as e:
            st.error(f"구매 데이터 로드 실패: {e}")
    
    # BOM 원가 계산
    if 'bom_data' in st.session_state and purchase_df is not None:
        st.header("3. 원가 계산")
        
        if st.button("계산 시작", type="primary"):
            bom_df = st.session_state.bom_data
            
            # BOM 데이터 검증
            bom_valid, bom_msg = validate_bom_data(bom_df)
            if not bom_valid:
                st.error(f"BOM 데이터 오류: {bom_msg}")
                st.stop()
            
            # 구매 단가 추출
            with st.spinner("구매 단가 추출 중..."):
                purchase_prices = extract_purchase_prices(purchase_df)
            
            if not purchase_prices:
                st.error("구매 단가를 추출할 수 없습니다.")
                st.stop()
            
            st.success(f"구매 단가 추출 완료: {len(purchase_prices):,}개")
            
            # BOM 원가 계산
            result_df = calculate_all_bom_costs(bom_df, purchase_prices)
            
            if result_df.empty:
                st.error("BOM 원가 계산 실패")
                st.stop()
            
            # 완제품 필터링
            finished_goods = result_df[
                result_df['생산품목명'].str.contains('[완제품]', regex=False, na=False)
            ].copy()
            
            # 결과 표시
            st.header("4. 완제품 원가 결과")
            
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
            if not finished_goods.empty:
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
                
                # 다운로드
                csv_data = finished_goods.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    "완제품 원가 결과 다운로드 (CSV)",
                    csv_data,
                    f"BOM원가_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.warning("완제품 데이터가 없습니다.")
    
    else:
        if 'bom_data' not in st.session_state:
            st.info("먼저 SharePoint에서 BOM 데이터를 로드하세요.")
        if purchase_df is None:
            st.info("구매 데이터 파일을 업로드하세요.")

if __name__ == "__main__":
    main()
