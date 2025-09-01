import streamlit as st
import pandas as pd
import io
import numpy as np

def load_data(uploaded_file, skiprows=0):
    """파일을 불러와 데이터를 정제합니다."""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, skiprows=skiprows, encoding='utf-8-sig', dtype=str)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file, skiprows=skiprows, dtype=str)
        else:
            st.error("지원하지 않는 파일 형식입니다.")
            return None
        
        # 좌우 공백 제거
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.strip()
        return df
    except Exception as e:
        st.error(f"파일 읽기 오류: {e}")
        return None

def extract_purchase_prices(purchase_df):
    """구매 데이터에서 최신 단가 추출"""
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
    
    # 첫 번째 행을 헤더로 사용하는 경우
    if not all([date_col, item_code_col, price_col]):
        if len(purchase_df) > 0:
            new_headers = purchase_df.iloc[0].tolist()
            purchase_df.columns = new_headers
            purchase_df = purchase_df.iloc[1:].reset_index(drop=True)
            
            for col in purchase_df.columns:
                if '일자-No.' in str(col):
                    date_col = col
                elif '품목코드' in str(col):
                    item_code_col = col  
                elif '단가' in str(col):
                    price_col = col
    
    # 기본 컬럼 설정
    if not date_col and len(purchase_df.columns) > 0:
        date_col = purchase_df.columns[0]
    if not item_code_col and len(purchase_df.columns) > 1:  
        item_code_col = purchase_df.columns[1]
    if not price_col and len(purchase_df.columns) > 5:
        price_col = purchase_df.columns[5]
    
    st.info(f"📋 컬럼 매핑: 일자={date_col}, 품목코드={item_code_col}, 단가={price_col}")
    
    try:
        df = purchase_df.copy()
        
        # 데이터 정제
        if date_col:
            df['date'] = pd.to_datetime(df[date_col].astype(str).str.split('-').str[0], errors='coerce')
            df = df.dropna(subset=['date'])
            df = df.sort_values('date', ascending=False)
        
        if item_code_col and price_col:
            df['item_code'] = df[item_code_col].astype(str).str.strip()
            df['price'] = pd.to_numeric(df[price_col], errors='coerce').fillna(0)
            df = df[df['item_code'] != '']
            
            # 최신 단가만 추출
            latest = df.drop_duplicates(subset='item_code', keep='first')
            
            # 딕셔너리 생성
            price_dict = {}
            for _, row in latest.iterrows():
                code = row['item_code']
                price = row['price']
                if pd.notna(price) and price > 0:
                    price_dict[code] = price
        
        st.success(f"✅ {len(price_dict)}개 품목의 구매단가 추출 완료")
        return price_dict
        
    except Exception as e:
        st.error(f"단가 추출 오류: {e}")
        return {}

def calculate_direct_bom_cost(product_code, bom_df, all_costs, calculation_cache):
    """
    특정 제품의 BOM 원가를 직접 계산하는 새로운 방식
    """
    # 이미 계산된 경우 캐시에서 반환
    if product_code in calculation_cache:
        return calculation_cache[product_code]
    
    # 해당 제품의 BOM 구성요소 가져오기
    components = bom_df[bom_df['생산품목코드'] == product_code].copy()
    
    if components.empty:
        calculation_cache[product_code] = 0.0
        return 0.0
    
    total_cost = 0.0
    calculation_details = []
    
    for _, comp in components.iterrows():
        comp_code = str(comp['소모품목코드']).strip()
        comp_name = str(comp['소모품목명'])
        quantity = float(comp['소요량']) if pd.notna(comp['소요량']) else 0.0
        
        # 부품의 단가 찾기
        if comp_code in all_costs:
            # 이미 알려진 단가 (구매단가 또는 계산된 단가)
            unit_price = all_costs[comp_code]
        else:
            # 다른 생산품목인지 확인하여 재귀 계산
            if comp_code in bom_df['생산품목코드'].values:
                unit_price = calculate_direct_bom_cost(comp_code, bom_df, all_costs, calculation_cache)
                all_costs[comp_code] = unit_price  # 계산 결과를 all_costs에 저장
            else:
                unit_price = 0.0  # 단가를 찾을 수 없음
        
        component_cost = quantity * unit_price
        total_cost += component_cost
        
        calculation_details.append({
            '부품코드': comp_code,
            '부품명': comp_name,
            '소요량': quantity,
            '단가': unit_price,
            '부품원가': component_cost
        })
    
    # 계산 결과 캐시에 저장
    calculation_cache[product_code] = total_cost
    
    return total_cost

def calculate_all_bom_costs(bom_df, purchase_prices):
    """모든 제품의 BOM 원가를 새로운 방식으로 계산"""
    
    # 필수 컬럼 확인
    required_cols = ['생산품목코드', '생산품목명', '소모품목코드', '소모품목명', '소요량']
    missing_cols = [col for col in required_cols if col not in bom_df.columns]
    
    if missing_cols:
        st.error(f"❌ BOM 데이터 필수 컬럼 누락: {missing_cols}")
        return pd.DataFrame(), pd.DataFrame()
    
    # 데이터 정제
    bom_clean = bom_df.copy()
    bom_clean['생산품목코드'] = bom_clean['생산품목코드'].astype(str).str.strip()
    bom_clean['소모품목코드'] = bom_clean['소모품목코드'].astype(str).str.strip()
    bom_clean['소요량'] = pd.to_numeric(bom_clean['소요량'], errors='coerce').fillna(0)
    
    # 모든 생산품목 목록
    all_products = bom_clean[['생산품목코드', '생산품목명']].drop_duplicates().reset_index(drop=True)
    
    st.write(f"📊 총 생산품목 수: {len(all_products)}")
    st.write(f"📊 구매단가 보유 품목: {len(purchase_prices)}")
    
    # 전체 비용 딕셔너리 (구매단가로 초기화)
    all_costs = purchase_prices.copy()
    calculation_cache = {}
    
    # 각 생산품목별로 직접 계산
    results = []
    
    for _, product in all_products.iterrows():
        product_code = product['생산품목코드']
        product_name = product['생산품목명']
        
        # 직접 계산 방식 사용
        calculated_cost = calculate_direct_bom_cost(product_code, bom_clean, all_costs, calculation_cache)
        
        results.append({
            '생산품목코드': product_code,
            '생산품목명': product_name,
            '계산된단위원가': calculated_cost,
            '계산상태': '계산완료' if calculated_cost > 0 else '계산불가'
        })
        
        # D626E 계열 특별 출력
        if 'D626E' in product_code:
            st.write(f"🎯 **{product_code}** 직접계산 결과: **{calculated_cost:,.2f}원**")
    
    # 결과 DataFrame 생성
    result_df = pd.DataFrame(results)
    
    # 상세 내역 생성 (새로운 방식)
    details_df = bom_clean.copy()
    details_df['부품단가'] = details_df['소모품목코드'].apply(lambda code: all_costs.get(code, 0.0))
    details_df['부품별원가'] = details_df['소요량'] * details_df['부품단가']
    
    st.success(f"✅ 직접 계산 방식으로 {len(result_df)}개 제품 처리 완료!")
    
    return result_df, details_df

def format_excel_output(writer, df, sheet_name, title=""):
    """엑셀 출력 포맷팅"""
    df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=2 if title else 0)
    
    # 워크시트 가져오기
    worksheet = writer.sheets[sheet_name]
    
    # 제목 추가
    if title:
        worksheet.cell(row=1, column=1, value=title)
        worksheet.merge_cells(start_row=1, end_row=1, start_column=1, end_column=len(df.columns))
    
    # 컬럼 너비 자동 조정
    for i, column in enumerate(df.columns, 1):
        max_length = max(
            len(str(column)),  # 헤더 길이
            df[column].astype(str).str.len().max() if not df.empty else 0  # 데이터 최대 길이
        )
        adjusted_width = min(max_length + 2, 50)  # 최대 50자로 제한
        worksheet.column_dimensions[worksheet.cell(row=1, column=i).column_letter].width = adjusted_width
    
    # 헤더 스타일
    header_row = 3 if title else 1
    for i in range(1, len(df.columns) + 1):
        cell = worksheet.cell(row=header_row, column=i)
        cell.fill = worksheet.cell(row=header_row, column=i).fill.__class__(fgColor="366092", fill_type="solid")
        cell.font = worksheet.cell(row=header_row, column=i).font.__class__(color="FFFFFF", bold=True)

def main():
    st.title('🚀 BOM 원가 계산기 (직접 계산 방식)')
    st.write("**새로운 접근: 각 제품별 BOM 구성요소를 직접 합계하는 방식**")
    
    st.header('1. 📁 파일 업로드')
    bom_file = st.file_uploader("BOM 데이터 파일", type=['csv', 'xlsx'], key="bom")
    purchase_file = st.file_uploader("구매 데이터 파일", type=['csv', 'xlsx'], key="purchase")
    
    if bom_file and purchase_file:
        # 데이터 로드
        bom_df = load_data(bom_file, skiprows=1)
        purchase_df = load_data(purchase_file)
        
        if bom_df is not None and purchase_df is not None:
            
            # 데이터 미리보기
            st.subheader("📋 데이터 미리보기")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**BOM 데이터**")
                st.write(f"총 {len(bom_df):,}행 × {len(bom_df.columns)}열")
                st.dataframe(bom_df.head(3), use_container_width=True)
                
            with col2:
                st.write("**구매 데이터**") 
                st.write(f"총 {len(purchase_df):,}행 × {len(purchase_df.columns)}열")
                st.dataframe(purchase_df.head(3), use_container_width=True)
            
            # test 품목 제거
            if '소모품목코드' in bom_df.columns:
                before_count = len(bom_df)
                bom_clean = bom_df[bom_df['소모품목코드'] != '99701'].copy()
                after_count = len(bom_clean)
                st.info(f"🧹 test 품목(99701) 제거: {before_count:,} → {after_count:,}행")
            else:
                bom_clean = bom_df.copy()
            
            st.header('2. 🔥 원가 계산 실행')
            
            if st.button('💪 직접 계산 방식으로 원가 계산 시작!', type="primary"):
                
                with st.spinner('🔄 새로운 직접계산 방식으로 처리중...'):
                    
                    # 1단계: 구매단가 추출
                    st.write("### 1단계: 구매단가 추출")
                    purchase_prices = extract_purchase_prices(purchase_df)
                    
                    if not purchase_prices:
                        st.error("❌ 구매단가를 추출할 수 없습니다.")
                        return
                    
                    # 2단계: BOM 원가 직접 계산
                    st.write("### 2단계: BOM 원가 직접 계산")
                    result_df, details_df = calculate_all_bom_costs(bom_clean, purchase_prices)
                    
                    if result_df.empty:
                        st.error("❌ 원가 계산 실패")
                        return
                    
                    # 3단계: 완제품 필터링
                    st.write("### 3단계: 완제품 결과 정리")
                    finished_goods = result_df[
                        result_df['생산품목명'].str.contains('[완제품]', regex=False, na=False)
                    ].copy()
                    
                    st.header('3. 🎯 완제품 원가 계산 결과')
                    
                    # 통계 표시
                    total_finished = len(finished_goods)
                    calculated_finished = len(finished_goods[finished_goods['계산상태'] == '계산완료'])
                    success_rate = (calculated_finished / total_finished * 100) if total_finished > 0 else 0
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🎯 전체 완제품", f"{total_finished:,}개")
                    with col2:
                        st.metric("✅ 계산 성공", f"{calculated_finished:,}개", 
                                f"+{calculated_finished}")
                    with col3:
                        st.metric("📊 성공률", f"{success_rate:.1f}%")
                    
                    # 결과 테이블 (포맷팅 개선)
                    display_df = finished_goods[['생산품목코드', '생산품목명', '계산된단위원가', '계산상태']].copy()
                    display_df.columns = ['품목코드', '품목명', '단위원가(원)', '상태']
                    
                    # 조건부 스타일링
                    def highlight_rows(row):
                        if row['상태'] == '계산완료':
                            return ['background-color: #d4edda; color: #155724'] * len(row)
                        else:
                            return ['background-color: #f8d7da; color: #721c24'] * len(row)
                    
                    styled_df = display_df.style.apply(highlight_rows, axis=1).format({
                        '단위원가(원)': '{:,.0f}'
                    })
                    
                    st.dataframe(styled_df, use_container_width=True)
                    
                    # 특별 확인: 높은 원가 상위 10개
                    top_cost_items = finished_goods.nlargest(10, '계산된단위원가')
                    if not top_cost_items.empty:
                        with st.expander("💰 원가 상위 10개 완제품"):
                            for _, item in top_cost_items.iterrows():
                                st.write(f"**{item['생산품목코드']}**: {item['계산된단위원가']:,.0f}원 - {item['생산품목명']}")
                    
                    # 계산 실패 항목
                    failed_items = finished_goods[finished_goods['계산상태'] == '계산불가']
                    if not failed_items.empty:
                        with st.expander(f"⚠️ 계산 실패 {len(failed_items)}개 항목"):
                            st.dataframe(failed_items[['생산품목코드', '생산품목명']], use_container_width=True)
                    
                    st.header('4. 📥 결과 다운로드')
                    
                    # 포맷팅된 엑셀 생성
                    output = io.BytesIO()
                    
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        # 완제품 결과
                        format_excel_output(writer, finished_goods, '완제품원가결과', 
                                          '완제품 BOM 원가 계산 결과')
                        
                        # 전체 제품 결과  
                        format_excel_output(writer, result_df, '전체제품원가', 
                                          '전체 제품 원가 계산 결과')
                        
                        # 상세 내역
                        format_excel_output(writer, details_df, 'BOM상세내역', 
                                          'BOM 구성요소별 상세 원가 내역')
                    
                    st.download_button(
                        label="📊 BOM 원가계산 결과 다운로드 (포맷팅된 Excel)",
                        data=output.getvalue(),
                        file_name=f'BOM원가계산_직접방식_{pd.Timestamp.now().strftime("%Y%m%d_%H%M")}.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        type="primary"
                    )
                    
                    st.success("🎉 직접 계산 방식으로 원가 계산이 완료되었습니다!")

if __name__ == '__main__':
    main()
