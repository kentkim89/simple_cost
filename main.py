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
    다단계 BOM 구조를 순회하며 반제품부터 완제품까지의 원가를 순차적으로 계산 (Cost Roll-up).
    """
    # 1. 초기 단가 설정: 구매 단가가 있는 품목(원재료 등)을 기초 단가로 설정
    unit_costs = latest_prices.set_index('품목코드')['단가'].to_dict()
    
    # 2. 계산 대상 품목 설정: BOM에 생산품목으로 등록된 모든 품목
    products_to_calculate = bom_df[['생산품목코드', '생산품목명']].dropna().drop_duplicates()
    
    # 3. 반복 계산: 모든 생산품목의 원가가 계산될 때까지 반복
    # 최대 반복 횟수를 지정하여 무한 루프 방지 (BOM의 최대 깊이보다 충분히 크게 설정)
    for _ in range(len(products_to_calculate)):
        newly_calculated_count = 0
        for _, product in products_to_calculate.iterrows():
            product_code = product['생산품목코드']
            
            # 이미 원가 계산이 끝난 품목은 건너뜀
            if product_code in unit_costs:
                continue

            components = bom_df[bom_df['생산품목코드'] == product_code]
            can_calculate = True
            total_cost = 0
            
            # 4. 부품(소모품목) 단가 확인: 해당 제품의 모든 부품 단가가 이미 계산되었는지 확인
            for _, component in components.iterrows():
                comp_code = component['소모품목코드']
                if comp_code not in unit_costs:
                    can_calculate = False
                    break # 하나라도 부품 단가를 모르면 상위 품목 단가 계산 불가
                
                # 부품 단가가 있으면 비용 누적
                cost_of_component = unit_costs.get(comp_code, 0)
                total_cost += component['소요량'] * cost_of_component
            
            # 5. 단가 계산: 모든 부품의 단가를 알 경우, 현재 제품의 단가를 계산하고 추가
            if can_calculate:
                unit_costs[product_code] = total_cost
                newly_calculated_count += 1
        
        # 더 이상 새로 계산되는 품목이 없으면 모든 계산이 완료된 것이므로 반복 종료
        if newly_calculated_count == 0:
            break
            
    # 6. 결과 정리
    # 계산된 전체 단가 정보를 데이터프레임으로 변환
    all_costs_df = pd.DataFrame(list(unit_costs.items()), columns=['품목코드', '계산된 단위 원가'])
    
    # 품목 정보와 결합
    all_products_info = pd.concat([
        bom_df[['생산품목코드', '생산품목명']].rename(columns={'생산품목코드':'품목코드', '생산품목명':'품목명'}),
        bom_df[['소모품목코드', '소모품목명']].rename(columns={'소모품목코드':'품목코드', '소모품목명':'품목명'})
    ]).dropna().drop_duplicates('품목코드')
    
    summary_df = pd.merge(all_products_info, all_costs_df, on='품목코드', how='left').fillna(0)

    # 상세 내역 생성
    details_df = bom_df.copy()
    details_df['부품 단위 원가'] = details_df['소모품목코드'].map(unit_costs).fillna(0)
    details_df['부품별 원가'] = details_df['소요량'] * details_df['부품 단위 원가']

    return summary_df, details_df

def main():
    st.title('BOM 기반 제품 원가 계산기 (다단계 지원) 🏭')

    st.header('1. 파일 업로드')
    bom_file = st.file_uploader("BOM 데이터 (CSV 또는 Excel)", type=['csv', 'xlsx'])
    purchase_file = st.file_uploader("구매 기록 데이터 (CSV 또는 Excel)", type=['csv', 'xlsx'])

    if bom_file and purchase_file:
        bom_df = load_data(bom_file, skiprows=1)
        purchase_df = load_data(purchase_file)

        if bom_df is not None and purchase_df is not None:
            st.header('2. 원가 계산 실행')
            if st.button('모든 완제품 원가 계산하기'):
                with st.spinner('다단계 BOM 구조를 분석하며 전체 원가를 계산 중입니다...'):
                    latest_prices = get_latest_prices(purchase_df)
                    summary_df, details_df = calculate_multi_level_bom_costs(bom_df, latest_prices)

                    # 최종 결과에서 [완제품]만 필터링
                    finished_goods_summary = summary_df[summary_df['품목명'].str.contains('\[완제품\]', na=False)]

                    st.header('3. [완제품] 원가 계산 결과 요약')
                    st.dataframe(finished_goods_summary[['품목코드', '품목명', '계산된 단위 원가']].style.format({'계산된 단위 원가': '{:,.2f}'}))

                    # 다운로드를 위한 엑셀 파일 생성
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        finished_goods_summary.to_excel(writer, index=False, sheet_name='[완제품] 원가 요약')
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
