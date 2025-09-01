import streamlit as st
import pandas as pd
import io

def load_data(uploaded_file, skiprows=0):
    """파일을 불러와 데이터를 정제합니다."""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, skiprows=skiprows, encoding='utf-8-sig', dtype=str)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file, skiprows=skiprows, dtype=str)
        else:
            st.error("지원하지 않는 파일 형식입니다. CSV 또는 XLSX 파일을 업로드해주세요.")
            return None
        
        # 좌우 공백 제거
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.strip()
        return df
    except Exception as e:
        st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        return None

def extract_purchase_prices(purchase_df):
    """구매 데이터에서 최신 단가를 추출합니다."""
    # 컬럼명 자동 감지
    date_col = None
    item_code_col = None
    price_col = None
    
    for col in purchase_df.columns:
        col_str = str(col).lower()
        if '일자' in col_str and 'no' in col_str:
            date_col = col
        elif '품목코드' in col_str:
            item_code_col = col
        elif '단가' in col_str:
            price_col = col
    
    # 컬럼을 찾지 못한 경우 첫 번째 행을 헤더로 사용
    if not all([date_col, item_code_col, price_col]):
        if len(purchase_df) > 0:
            new_headers = purchase_df.iloc[0].tolist()
            purchase_df.columns = new_headers
            purchase_df = purchase_df.iloc[1:].reset_index(drop=True)
            
            # 다시 컬럼 찾기
            for col in purchase_df.columns:
                if '일자-No.' in str(col):
                    date_col = col
                elif '품목코드' in str(col):
                    item_code_col = col  
                elif '단가' in str(col):
                    price_col = col
    
    # 기본값 사용
    if not date_col and len(purchase_df.columns) > 0:
        date_col = purchase_df.columns[0]
    if not item_code_col and len(purchase_df.columns) > 1:  
        item_code_col = purchase_df.columns[1]
    if not price_col and len(purchase_df.columns) > 5:
        price_col = purchase_df.columns[5]
    
    st.info(f"사용된 컬럼: 일자={date_col}, 품목코드={item_code_col}, 단가={price_col}")
    
    try:
        # 데이터 정제
        df = purchase_df.copy()
        
        # 일자 처리
        if date_col:
            df['date_str'] = df[date_col].astype(str)
            df['date'] = df['date_str'].apply(lambda x: str(x).split('-')[0] if '-' in str(x) else str(x))
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])
        
        # 단가와 품목코드 처리
        if price_col:
            df['price'] = pd.to_numeric(df[price_col], errors='coerce').fillna(0)
        if item_code_col:
            df = df.dropna(subset=[item_code_col])
            df['item_code'] = df[item_code_col].astype(str).str.strip()
        
        # 최신 단가만 추출
        if date_col:
            df = df.sort_values('date', ascending=False)
        
        latest_prices = df.drop_duplicates(subset='item_code', keep='first')
        
        # 딕셔너리로 변환 (문자열 키 사용)
        price_dict = {}
        for _, row in latest_prices.iterrows():
            code = row['item_code']
            price = row['price']
            if code and pd.notna(price):
                price_dict[code] = float(price)
        
        st.write(f"✅ 구매 데이터에서 {len(price_dict)}개 품목의 단가를 추출했습니다.")
        return price_dict
        
    except Exception as e:
        st.error(f"단가 추출 중 오류: {e}")
        return {}

def calculate_bom_costs(bom_df, price_dict):
    """BOM 원가를 계산합니다."""
    
    # 필수 컬럼 확인
    required_cols = ['생산품목코드', '생산품목명', '소모품목코드', '소모품목명', '소요량']
    missing_cols = [col for col in required_cols if col not in bom_df.columns]
    
    if missing_cols:
        st.error(f"BOM 데이터에 필수 컬럼이 없습니다: {missing_cols}")
        return pd.DataFrame(), pd.DataFrame()
    
    # 데이터 정제
    bom_clean = bom_df.copy()
    bom_clean['생산품목코드'] = bom_clean['생산품목코드'].astype(str).str.strip()
    bom_clean['소모품목코드'] = bom_clean['소모품목코드'].astype(str).str.strip()
    bom_clean['소요량'] = pd.to_numeric(bom_clean['소요량'], errors='coerce').fillna(0)
    
    # 모든 생산품목 목록
    all_products = bom_clean[['생산품목코드', '생산품목명']].drop_duplicates().copy()
    all_products = all_products[all_products['생산품목코드'].notna()].reset_index(drop=True)
    
    st.write(f"📊 전체 생산품목 수: {len(all_products)}")
    st.write(f"📊 구매 단가 보유 품목 수: {len(price_dict)}")
    
    # 단가 딕셔너리 초기화 (구매 단가부터)
    unit_costs = price_dict.copy()
    
    # 반복 계산
    max_iterations = len(all_products) + 5
    calculation_success = []
    
    for iteration in range(max_iterations):
        progress_made = False
        
        for _, product_row in all_products.iterrows():
            product_code = product_row['생산품목코드']
            product_name = product_row['생산품목명']
            
            # 이미 계산된 제품은 건너뛰기
            if product_code in unit_costs:
                continue
            
            # 해당 제품의 BOM 구성요소
            components = bom_clean[bom_clean['생산품목코드'] == product_code]
            
            if components.empty:
                continue
            
            # 모든 구성요소의 단가를 알고 있는지 확인
            all_components_available = True
            missing_components = []
            
            for _, comp in components.iterrows():
                comp_code = comp['소모품목코드']
                if comp_code not in unit_costs:
                    all_components_available = False
                    missing_components.append(comp_code)
            
            # 모든 구성요소의 단가를 알고 있으면 계산
            if all_components_available:
                total_cost = 0.0
                component_details = []
                
                for _, comp in components.iterrows():
                    comp_code = comp['소모품목코드']
                    comp_name = comp['소모품목명']
                    quantity = float(comp['소요량'])
                    comp_unit_price = float(unit_costs[comp_code])
                    comp_total_cost = quantity * comp_unit_price
                    total_cost += comp_total_cost
                    
                    component_details.append({
                        '부품코드': comp_code,
                        '부품명': comp_name,
                        '소요량': quantity,
                        '단가': comp_unit_price,
                        '부품원가': comp_total_cost
                    })
                
                # 계산 결과 저장
                unit_costs[product_code] = total_cost
                progress_made = True
                
                calculation_success.append({
                    '반복차수': iteration + 1,
                    '제품코드': product_code,
                    '제품명': product_name,
                    '총원가': total_cost,
                    '구성요소': component_details
                })
                
                # 특별 디버깅 - D626E와 비슷한 패턴 확인
                if 'D626E' in product_code:
                    st.write(f"🎯 **{product_code} 계산 완료**")
                    st.write(f"   총 원가: {total_cost:,.2f}원")
                    st.write(f"   구성요소 {len(component_details)}개")
        
        # 더 이상 진전이 없으면 중단
        if not progress_made:
            break
    
    st.write(f"✅ 총 {len(calculation_success)}개 제품의 원가를 계산했습니다.")
    
    # 결과 정리
    result_df = all_products.copy()
    
    # 단가 매핑 (간단하게)
    result_df['계산된_단위_원가'] = result_df['생산품목코드'].apply(
        lambda code: unit_costs.get(code, 0.0)
    )
    
    # 계산 상태
    result_df['계산상태'] = result_df['계산된_단위_원가'].apply(
        lambda cost: '계산완료' if cost > 0 else '계산불가'
    )
    
    # 특별 확인: D626E류 제품들
    d626e_like = result_df[result_df['생산품목코드'].str.contains('D626E', na=False)]
    if not d626e_like.empty:
        st.write("🔍 **D626E 계열 제품 확인:**")
        for _, row in d626e_like.iterrows():
            code = row['생산품목코드']
            cost = row['계산된_단위_원가']
            status = row['계산상태']
            st.write(f"   {code}: {cost:,.2f}원 ({status})")
    
    # 상세 내역 생성
    details_df = bom_clean.copy()
    details_df['부품_단가'] = details_df['소모품목코드'].apply(
        lambda code: unit_costs.get(code, 0.0)
    )
    details_df['부품별_원가'] = details_df['소요량'] * details_df['부품_단가']
    
    return result_df, details_df

def main():
    st.title('🚀 BOM 원가 계산기 (완전 새 버전)')
    
    st.header('1. 파일 업로드')
    bom_file = st.file_uploader("BOM 데이터 파일", type=['csv', 'xlsx'])
    purchase_file = st.file_uploader("구매 데이터 파일", type=['csv', 'xlsx'])
    
    if bom_file and purchase_file:
        # 데이터 로드
        bom_df = load_data(bom_file, skiprows=1)
        purchase_df = load_data(purchase_file)
        
        if bom_df is not None and purchase_df is not None:
            # 데이터 미리보기
            st.subheader("📋 업로드된 데이터 확인")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**BOM 데이터**")
                st.write(f"행 수: {len(bom_df)}, 컬럼 수: {len(bom_df.columns)}")
                st.dataframe(bom_df.head())
                
            with col2:
                st.write("**구매 데이터**")
                st.write(f"행 수: {len(purchase_df)}, 컬럼 수: {len(purchase_df.columns)}")
                st.dataframe(purchase_df.head())
            
            # test 품목 제거
            if '소모품목코드' in bom_df.columns:
                bom_clean = bom_df[bom_df['소모품목코드'] != '99701'].copy()
                st.info(f"test 품목(99701) 제거: {len(bom_df)} → {len(bom_clean)} 행")
            else:
                bom_clean = bom_df.copy()
            
            st.header('2. 원가 계산')
            if st.button('🔥 완제품 원가 계산 시작!'):
                with st.spinner('새로운 로직으로 계산 중...'):
                    # 단가 추출
                    price_dict = extract_purchase_prices(purchase_df)
                    
                    if not price_dict:
                        st.error("구매 데이터에서 단가를 추출할 수 없습니다.")
                        return
                    
                    # BOM 원가 계산
                    result_df, details_df = calculate_bom_costs(bom_clean, price_dict)
                    
                    if result_df.empty:
                        st.error("원가 계산에 실패했습니다.")
                        return
                    
                    # 완제품만 필터링
                    finished_goods = result_df[
                        result_df['생산품목명'].str.contains('[완제품]', regex=False, na=False)
                    ].copy()
                    
                    st.header('3. 🎯 완제품 원가 결과')
                    
                    # 통계
                    total_finished = len(finished_goods)
                    calculated_finished = len(finished_goods[finished_goods['계산상태'] == '계산완료'])
                    success_rate = (calculated_finished / total_finished * 100) if total_finished > 0 else 0
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("전체 완제품", total_finished)
                    with col2:
                        st.metric("계산 성공", calculated_finished)
                    with col3:
                        st.metric("성공률", f"{success_rate:.1f}%")
                    
                    # 결과 테이블
                    display_cols = ['생산품목코드', '생산품목명', '계산된_단위_원가', '계산상태']
                    display_df = finished_goods[display_cols].copy()
                    display_df.columns = ['품목코드', '품목명', '계산된_단위_원가', '상태']
                    
                    # 스타일 적용
                    def color_status(row):
                        colors = []
                        for col in row.index:
                            if row['상태'] == '계산완료':
                                colors.append('background-color: #d4edda')
                            else:
                                colors.append('background-color: #f8d7da')
                        return colors
                    
                    styled_df = display_df.style.apply(color_status, axis=1).format({
                        '계산된_단위_원가': '{:,.2f}'
                    })
                    
                    st.dataframe(styled_df, use_container_width=True)
                    
                    # 실패 항목 분석
                    failed_items = finished_goods[finished_goods['계산상태'] == '계산불가']
                    if not failed_items.empty:
                        with st.expander(f"⚠️ 계산 실패 항목 {len(failed_items)}개"):
                            st.dataframe(failed_items[['생산품목코드', '생산품목명']])
                    
                    # 엑셀 다운로드
                    st.header('4. 📥 결과 다운로드')
                    
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        finished_goods.to_excel(writer, sheet_name='완제품_원가', index=False)
                        result_df.to_excel(writer, sheet_name='전체_제품_원가', index=False)
                        details_df.to_excel(writer, sheet_name='상세_내역', index=False)
                    
                    st.download_button(
                        label="📊 완제품 원가 결과 다운로드 (Excel)",
                        data=output.getvalue(),
                        file_name='BOM_원가계산_결과_새버전.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )

if __name__ == '__main__':
    main()
