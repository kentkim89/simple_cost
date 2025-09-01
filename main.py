import streamlit as st
import pandas as pd
import io

def load_data(uploaded_file, skiprows=0):
    """
    파일을 불러올 때 모든 데이터를 문자로 읽어와 데이터 타입 불일치 문제를 원천 차단하고,
    좌우 공백을 제거하여 데이터를 정제합니다.
    """
    try:
        # dtype=str 옵션을 사용해 모든 데이터를 문자로 불러옵니다.
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, skiprows=skiprows, encoding='utf-8-sig', dtype=str)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file, skiprows=skiprows, dtype=str)
        else:
            st.error("지원하지 않는 파일 형식입니다. CSV 또는 XLSX 파일을 업로드해주세요.")
            return None
        
        # 모든 컬럼의 좌우 공백 제거
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.strip()
        return df
    except Exception as e:
        st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        return None

def get_latest_prices(purchase_df, date_col, item_code_col, price_col):
    """
    사용자가 지정한 열을 기준으로 품목별 최신 단가를 추출합니다.
    """
    purchase_df_copy = purchase_df.copy()
    
    # 날짜 데이터 처리
    purchase_df_copy['date_for_sorting'] = purchase_df_copy[date_col].astype(str).str.split('-').str[0]
    purchase_df_copy['date_for_sorting'] = pd.to_datetime(purchase_df_copy['date_for_sorting'], errors='coerce')
    purchase_df_copy.dropna(subset=['date_for_sorting'], inplace=True)
    
    # 단가 컬럼을 숫자로 변환
    purchase_df_copy[price_col] = pd.to_numeric(purchase_df_copy[price_col], errors='coerce').fillna(0)
    
    purchase_df_copy = purchase_df_copy.sort_values(by='date_for_sorting', ascending=False)
    latest_prices = purchase_df_copy.drop_duplicates(subset=item_code_col, keep='first')
    
    return latest_prices.set_index(item_code_col)[price_col].to_dict()

def calculate_multi_level_bom_costs(bom_df, latest_prices):
    """
    가장 안정적인 방식으로 다단계 BOM 원가를 계산합니다.
    """
    unit_costs = latest_prices.copy()

    products_to_calc = bom_df[['생산품목코드', '생산품목명']].dropna().drop_duplicates()
    products_to_calc_set = set(products_to_calc['생산품목코드'])

    bom_df['소요량'] = pd.to_numeric(bom_df['소요량'], errors='coerce').fillna(0)

    for _ in range(len(products_to_calc_set) + 5):
        made_progress = False
        remaining_products = [p for p in products_to_calc_set if p not in unit_costs]
        
        for product_code in remaining_products:
            components = bom_df[bom_df['생산품목코드'] == product_code]
            
            if all(comp_code in unit_costs for comp_code in components['소모품목코드']):
                total_cost = (components['소요량'] * components['소모품목코드'].map(unit_costs).fillna(0)).sum()
                unit_costs[product_code] = total_cost
                made_progress = True
        
        if not made_progress:
            break
            
    summary_df = products_to_calc.copy()
    summary_df['계산된 단위 원가'] = summary_df['생산품목코드'].map(unit_costs).fillna(0)
    
    details_df = bom_df.copy()
    details_df['부품 단위 원가'] = details_df['소모품목코드'].map(unit_costs).fillna(0)
    details_df['부품별 원가'] = details_df['소요량'] * details_df['부품 단위 원가']
    
    return summary_df, details_df

def main():
    st.title('BOM 원가 계산기 (열 선택 기능) 🚀')

    st.header('1. 파일 업로드')
    bom_file = st.file_uploader("BOM 데이터 (CSV 또는 Excel)", type=['csv', 'xlsx'])
    purchase_file = st.file_uploader("구매 기록 데이터 (CSV 또는 Excel)", type=['csv', 'xlsx'])

    if bom_file and purchase_file:
        bom_df_raw = load_data(bom_file, skiprows=1)
        purchase_df = load_data(purchase_file)

        if bom_df_raw is not None and purchase_df is not None:
            st.header('2. 구매 데이터 열 선택')
            st.write("업로드하신 **구매 기록 데이터** 파일에서 어떤 열이 어떤 정보를 담고 있는지 지정해주세요.")
            
            purchase_cols = purchase_df.columns.tolist()
            # 사용자가 열을 선택하도록 위젯 배치
            date_col = st.selectbox("날짜 정보가 있는 열을 선택하세요:", purchase_cols, index=0)
            item_code_col = st.selectbox("품목 코드가 있는 열을 선택하세요:", purchase_cols, index=1)
            price_col = st.selectbox("단가 정보가 있는 열을 선택하세요:", purchase_cols, index=5)

            st.header('3. 원가 계산 실행')
            if st.button('모든 완제품 원가 계산하기'):
                with st.spinner('최종 로직으로 전체 원가를 계산 중입니다...'):
                    # 'test' 품목 제외
                    bom_df = bom_df_raw[bom_df_raw['소모품목코드'] != '99701'].copy()
                    
                    # 사용자가 선택한 열 이름을 바탕으로 최신 단가 추출
                    latest_prices = get_latest_prices(purchase_df, date_col, item_code_col, price_col)
                    
                    summary_df, details_df = calculate_multi_level_bom_costs(bom_df, latest_prices)
                    finished_goods_summary = summary_df[summary_df['생산품목명'].str.contains('[완제품]', regex=False, na=False)]

                    st.header('4. [완제품] 원가 계산 결과 요약')
                    st.dataframe(finished_goods_summary[['생산품목코드', '생산품목명', '계산된 단위 원가']].rename(columns={'생산품목코드':'품목코드', '생산품목명':'품목명'}).style.format({'계산된 단위 원가': '{:,.2f}'}))

                    # 원가 0원 항목 분석
                    uncalculated_df = finished_goods_summary[finished_goods_summary['계산된 단위 원가'] == 0]
                    if not uncalculated_df.empty:
                        with st.expander("⚠️ 원가 0원 항목 분석 (클릭하여 확인)"):
                            st.write("아래 품목들은 구성 부품의 원가 정보가 없어 원가가 0으로 계산되었습니다.")
                            st.dataframe(uncalculated_df[['생산품목코드', '생산품목명']].rename(columns={'생산품목코드':'품목코드', '생산품목명':'품목명'}))

                    # 엑셀 다운로드
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        finished_goods_summary.to_excel(writer, index=False, sheet_name='완제품 원가 요약')
                        details_df.to_excel(writer, index=False, sheet_name='전체 상세 원가 내역')
                    
                    st.header('5. 결과 다운로드')
                    st.download_button(
                        label="[완제품] 원가 계산 결과 다운로드 (Excel)",
                        data=output.getvalue(),
                        file_name='완제품_원가계산_결과.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )

if __name__ == '__main__':
    main()
