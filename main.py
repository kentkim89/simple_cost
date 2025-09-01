import streamlit as st
import pandas as pd
import io

def load_data(uploaded_file, skiprows=0):
    """
    업로드된 파일의 확장자를 확인하고 적절한 방식으로 데이터를 로드합니다.
    """
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file, skiprows=skiprows, encoding='utf-8-sig')
        elif uploaded_file.name.endswith('.xlsx'):
            return pd.read_excel(uploaded_file, skiprows=skiprows)
        else:
            st.error("지원하지 않는 파일 형식입니다. CSV 또는 XLSX 파일을 업로드해주세요.")
            return None
    except Exception as e:
        st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        return None

def get_latest_prices(purchase_df):
    """
    구매 데이터에서 품목별 최신 단가를 추출합니다.
    """
    purchase_df['일자-No.'] = purchase_df['일자-No.'].astype(str)
    purchase_df['date'] = purchase_df['일자-No.'].apply(lambda x: x.split('-')[0])
    purchase_df['date'] = pd.to_datetime(purchase_df['date'], errors='coerce')
    purchase_df.dropna(subset=['date'], inplace=True)
    purchase_df = purchase_df.sort_values(by='date', ascending=False)
    latest_prices = purchase_df.drop_duplicates(subset='품목코드', keep='first')
    return latest_prices[['품목코드', '단가']]

def calculate_multi_level_bom_costs(bom_df, latest_prices):
    """
    최적화된 방식으로 다단계 BOM 원가를 계산하고, 실패 원인을 분석합니다.
    """
    # 1. 초기 단가 설정: 구매 기록이 있는 원재료/부자재 단가
    unit_costs = latest_prices.set_index('품목코드')['단가'].to_dict()
    
    # 2. 모든 품목 정보 통합 (생산품목 + 소모품목)
    all_products_info = pd.concat([
        bom_df[['생산품목코드', '생산품목명']].rename(columns={'생산품목코드': '품목코드', '생산품목명': '품목명'}),
        bom_df[['소모품목코드', '소모품목명']].rename(columns={'소모품목코드': '품목코드', '소모품목명': '품목명'})
    ]).dropna(subset=['품목코드']).drop_duplicates('품목코드').set_index('품목코드')

    # 3. 계산 루프: 더 이상 계산할 품목이 없을 때까지 반복
    while True:
        newly_calculated_count = 0
        # 아직 원가가 계산되지 않은 생산품목 목록
        products_to_calculate = bom_df[~bom_df['생산품목코드'].isin(unit_costs.keys())]['생산품목코드'].unique()

        for product_code in products_to_calculate:
            components = bom_df[bom_df['생산품목코드'] == product_code]
            can_calculate = True
            total_cost = 0
            
            # 모든 부품(소모품목)의 원가가 이미 계산되었는지 확인
            for _, component in components.iterrows():
                comp_code = component['소모품목코드']
                if comp_code not in unit_costs:
                    can_calculate = False
                    break # 부품 원가를 모르므로 상위 품목 계산 불가
                total_cost += component['소요량'] * unit_costs.get(comp_code, 0)
            
            # 모든 부품 원가를 알 경우, 현재 품목의 원가를 계산하고 추가
            if can_calculate:
                unit_costs[product_code] = total_cost
                newly_calculated_count += 1
        
        # 한 반복 동안 아무것도 계산되지 않았다면, 모든 계산이 완료된 것
        if newly_calculated_count == 0:
            break

    # 4. 결과 정리
    summary_df = all_products_info.copy()
    summary_df['계산된 단위 원가'] = summary_df.index.map(unit_costs).fillna(0)
    summary_df.reset_index(inplace=True)

    # 5. 상세 내역 및 원인 분석
    details_df = bom_df.copy()
    details_df['부품 단위 원가'] = details_df['소모품목코드'].map(unit_costs).fillna(0)
    details_df['부품별 원가'] = details_df['소요량'] * details_df['부품 단위 원가']
    
    uncalculated_products = []
    # 원가가 0인 '생산품목'들을 대상으로 분석
    zero_cost_products = summary_df[(summary_df['계산된 단위 원가'] == 0) & (summary_df['품목코드'].isin(bom_df['생산품목코드']))]

    for _, product in zero_cost_products.iterrows():
        missing_components = []
        components = bom_df[bom_df['생산품목코드'] == product['품목코드']]
        for _, comp in components.iterrows():
            if comp['소모품목코드'] not in unit_costs or unit_costs.get(comp['소모품목코드']) == 0:
                missing_components.append(f"{comp['소모품목명']} ({comp['소모품목코드']})")
        
        if missing_components:
            uncalculated_products.append({
                "품목코드": product['품목코드'],
                "품목명": product['품목명'],
                "원가 정보가 없는 부품": ", ".join(list(set(missing_components)))
            })
            
    uncalculated_df = pd.DataFrame(uncalculated_products)

    return summary_df, details_df, uncalculated_df

def main():
    st.title('BOM 원가 계산기 (성능 최적화) 🚀')

    st.header('1. 파일 업로드')
    bom_file = st.file_uploader("BOM 데이터 (CSV 또는 Excel)", type=['csv', 'xlsx'])
    purchase_file = st.file_uploader("구매 기록 데이터 (CSV 또는 Excel)", type=['csv', 'xlsx'])

    if bom_file and purchase_file:
        bom_df = load_data(bom_file, skiprows=1)
        purchase_df = load_data(purchase_file)

        if bom_df is not None and purchase_df is not None:
            st.header('2. 원가 계산 실행')
            if st.button('모든 완제품 원가 계산하기'):
                with st.spinner('최적화된 방식으로 전체 원가를 계산 중입니다...'):
                    latest_prices = get_latest_prices(purchase_df)
                    summary_df, details_df, uncalculated_df = calculate_multi_level_bom_costs(bom_df, latest_prices)

                    finished_goods_summary = summary_df[summary_df['품목명'].str.contains('[완제품]', regex=False, na=False)]

                    st.header('3. [완제품] 원가 계산 결과 요약')
                    st.dataframe(finished_goods_summary[['품목코드', '품목명', '계산된 단위 원가']].style.format({'계산된 단위 원가': '{:,.2f}'}))

                    if not uncalculated_df.empty:
                        with st.expander("⚠️ 원가 0원 항목 분석 (클릭하여 확인)"):
                            st.write("아래 품목들은 구성 부품의 원가 정보가 없어 원가가 0으로 계산되었습니다. '원가 정보가 없는 부품' 목록을 확인하고, 구매 내역에 단가를 추가하거나 해당 부품의 BOM을 점검해주세요.")
                            st.dataframe(uncalculated_df)

                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        finished_goods_summary.to_excel(writer, index=False, sheet_name='완제품 원가 요약')
                        details_df.to_excel(writer, index=False, sheet_name='전체 상세 원가 내역')
                    
                    st.header('4. 결과 다운로드')
                    st.download_button(
                        label="[완제품] 원가 계산 결과 다운로드 (Excel)",
                        data=output.getvalue(),
                        file_name='완제품_원가계산_결과.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )

if __name__ == '__main__':
    main()
