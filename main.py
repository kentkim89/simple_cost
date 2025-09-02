import streamlit as st
import pandas as pd
import numpy as np
import io
from typing import Dict, Optional, Tuple
from datetime import datetime
import time
import warnings

warnings.filterwarnings('ignore')

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
    
    # test 품목 제거
    before_count = len(clean_df)
    clean_df = clean_df[clean_df['소모품목코드'] != '99701']  # test 품목 코드
    after_count = len(clean_df)
    
    if before_count != after_count:
        st.info(f"test 품목 제거: {before_count:,} → {after_count:,}행")
    
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

def calculate_product_cost_with_reason(product_code, bom_df, all_costs, cache):
    """단일 제품 원가 계산 + 실패 이유"""
    try:
        if product_code in cache:
            return cache[product_code], ""
        
        components = bom_df[bom_df['생산품목코드'] == product_code]
        if components.empty:
            cache[product_code] = 0.0
            return 0.0, "BOM 구성요소 없음"
        
        total_cost = 0.0
        missing_components = []
        
        for _, comp in components.iterrows():
            comp_code = comp['소모품목코드']
            comp_name = comp['소모품목명']
            quantity = float(comp['소요량'])
            
            if quantity <= 0:
                continue
            
            if comp_code in all_costs:
                unit_price = all_costs[comp_code]
            elif comp_code in bom_df['생산품목코드'].values:
                # 재귀 계산
                unit_price, _ = calculate_product_cost_with_reason(comp_code, bom_df, all_costs, cache)
                all_costs[comp_code] = unit_price
            else:
                missing_components.append(f"{comp_name}({comp_code})")
                continue
            
            if unit_price > 0:
                total_cost += quantity * unit_price
        
        cache[product_code] = total_cost
        
        if total_cost == 0 and missing_components:
            failure_reason = f"단가정보 없음: {', '.join(missing_components[:3])}{'...' if len(missing_components) > 3 else ''}"
            return 0.0, failure_reason
        
        return total_cost, ""
        
    except Exception as e:
        return 0.0, f"계산 오류: {str(e)}"

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
        failure_reasons = {}
        results = []
        
        progress_bar = st.progress(0)
        
        for idx, (_, product) in enumerate(all_products.iterrows()):
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
    st.title("BOM 원가 계산기")
    st.markdown("**SharePoint 연동 대신 파일 업로드 방식으로 변경**")
    
    # BOM 데이터 업로드
    st.header("1. BOM 데이터")
    st.info("SharePoint에서 BOM 파일을 다운로드해서 업로드하세요")
    
    bom_file = st.file_uploader("BOM 파일 업로드", type=['xlsx', 'xls'], key="bom_upload")
    
    if bom_file:
        try:
            bom_df = pd.read_excel(bom_file, skiprows=1, dtype=str, engine='openpyxl')
            st.success(f"BOM 데이터 로드 완료: {len(bom_df):,}행 × {len(bom_df.columns)}열")
            st.session_state.bom_data = bom_df
            
            with st.expander("BOM 데이터 미리보기", expanded=False):
                st.dataframe(bom_df.head(), use_container_width=True)
                
        except Exception as e:
            st.error(f"BOM 파일 읽기 실패: {e}")
    
    # 구매 데이터 업로드
    st.header("2. 구매 데이터")
    purchase_file = st.file_uploader("구매 데이터 파일", type=['csv', 'xlsx'])
    
    purchase_df = None
    if purchase_file:
        try:
            purchase_df = pd.read_excel(purchase_file, dtype=str, engine='openpyxl')
            st.success(f"구매 데이터 로드: {len(purchase_df):,}행")
            
            with st.expander("구매 데이터 미리보기", expanded=False):
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
                # 실패 이유 포함
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
                
                # 계산 실패 분석
                failed_items = finished_goods[finished_goods['계산상태'] == '계산불가']
                if not failed_items.empty and '실패이유' in failed_items.columns:
                    with st.expander(f"계산 실패 {len(failed_items)}개 항목 분석"):
                        failure_stats = {}
                        for _, item in failed_items.iterrows():
                            reasons = str(item['실패이유']).split(' | ')
                            for reason in reasons:
                                main_reason = reason.split(':')[0] if ':' in reason else reason
                                failure_stats[main_reason] = failure_stats.get(main_reason, 0) + 1
                        
                        for reason, count in failure_stats.items():
                            st.write(f"• {reason}: {count}개")
                
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
    
    elif 'bom_data' not in st.session_state:
        st.info("BOM 데이터 파일을 업로드하세요.")
    elif purchase_df is None:
        st.info("구매 데이터 파일을 업로드하세요.")

if __name__ == "__main__":
    main()
