import streamlit as st
import pandas as pd
import io

def load_data(uploaded_file, skiprows=0):
    """
    업로드된 파일의 확장자를 확인하고 적절한 방식으로 데이터를 로드합니다.
    """
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, skiprows=skiprows, encoding='utf-8-sig')
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file, skiprows=skiprows)
        else:
            st.error("지원하지 않는 파일 형식입니다. CSV 또는 XLSX 파일을 업로드해주세요.")
            return None
        # 모든 object 타입 컬럼의 좌우 공백 제거
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.strip()
        return df
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
    안정성이 개선된 방식으로 다단계 BOM 원가를 계산하고, 실패 원인을 분석합니다.
    """
    # 1. 초기 단가 설정: 구매 기록이 있는 품목
    unit_costs = latest_prices.set_index('품목코드')['단가'].to_dict()

    # 2. 모든 품목 정보 통합 (중복 제거)
    all_products_info = pd.concat([
        bom_df[['생산품목코드', '생산품목명']].rename(columns={'생산품목코드': '품목코드', '생산품목명': '품목명'}),
        bom_df[['소모품목코드', '소모품목명']].rename(columns={'소모품목코드': '품목코드', '소모품목명': '품목명'})
    ]).dropna(subset=['품목코드']).drop_duplicates('품목코드').set_index('품목코드')

    # 3. 계산 루프: 더 이상 계산할 품목이 없을 때까지 반복
    bom_df_copy = bom_df.copy()
    while True:
        # 부품들의 단가가 모두 알려진 생산품을 계산 대상으로 선정
        bom_df_copy['부품단가계산가능'] = bom_df_copy['소모품목코드'].isin(unit_costs.keys())
        
        # 생산품의 모든 부품이 계산 가능한지 그룹별로 확인
        calculable_groups = bom_df_copy.groupby('생산품목코드')['부품단가계산가능'].all()
        
        # 계산 가능한 생산품 목록 (아직 unit_costs에 없는 것들만)
        products_to_calculate = calculable_groups[calculable_groups & ~calculable_groups.index.isin(unit_costs.keys())].index
        
        # 더 이상 계산할 제품이 없으면 루프 종료
        if len(products_to_calculate) == 0:
            break

        # 선택된 제품들의 원가 계산
        for product_code in products_to_calculate:
            components = bom_df_copy[bom_df_copy['생산품목코드'] == product_code]
            total_cost = 0
            for _, component in components.iterrows():
                comp_code = component['소모품목코드']
                total_cost += component['소요량'] * unit_costs.get(comp_code, 0)
            
            unit_costs[product_code] = total_cost

    # 4. 결과 정리
    summary_df = all_products_info.copy()
    summary_df['계산된 단위 원가'] = summary_df.index.map(unit_costs).fillna(0)
    summary_df.reset_index(inplace=True)

    # 5. 상세 내역 및 원인 분석
    details_df = bom_df.copy()
    details_df['부품 단위 원가'] = details_df['소모품목코드'].map(unit_costs).fillna(0)
    details_df['부품별 원가'] = details_df['소요량'] * details_df['부품 단위 원가']
    
    uncalculated_products = []
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
    st.title('BOM 원가 계산기 (최종 안정화 버전) 🚀')

    st.header('1. 파일 업로드')
    bom_file = st.file_uploader("BOM 데이터 (CSV 또는 Excel)", type=['csv', 'xlsx'])
    purchase_file = st.file_uploader("구매 기록 데이터 (CSV 또는 Excel)", type=['csv', 'xlsx'])

    if bom_file and purchase_file:
        bom_df_raw = load_data(bom_file, skiprows=1)
        purchase_df = load_data(purchase_file)

        if bom_df_raw is not None and purchase_df is not None:
            
            # 'test' 품목(99701)을 BOM 데이터에서 원천 제외
            bom_df = bom_df_raw[bom_df_raw['소모품목코드'] != '99701'].copy()
            st.info("'test'(99701) 품목을 BOM 분석에서 제외했습니다.")

            st.header('2. 원가 계산 실행')
            if st.button('모든 완제품 원가 계산하기'):
                with st.spinner('최종 로직으로 전체 원가를 계산 중입니다...'):
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
