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

def get_latest_prices(purchase_df):
    """
    구매 데이터에서 품목별 최신 단가를 추출합니다.
    """
    purchase_df_copy = purchase_df.copy()
    purchase_df_copy['일자-No.'] = purchase_df_copy['일자-No.'].astype(str)
    purchase_df_copy['date'] = purchase_df_copy['일자-No.'].apply(lambda x: x.split('-')[0])
    purchase_df_copy['date'] = pd.to_datetime(purchase_df_copy['date'], errors='coerce')
    purchase_df_copy.dropna(subset=['date'], inplace=True)
    # 단가 컬럼을 숫자로 변환
    purchase_df_copy['단가'] = pd.to_numeric(purchase_df_copy['단가'], errors='coerce').fillna(0)
    purchase_df_copy = purchase_df_copy.sort_values(by='date', ascending=False)
    latest_prices = purchase_df_copy.drop_duplicates(subset='품목코드', keep='first')
    return latest_prices.set_index('품목코드')['단가'].to_dict()

def calculate_multi_level_bom_costs(bom_df, latest_prices):
    """
    가장 안정적인 방식으로 다단계 BOM 원가를 계산하고, 실패 원인을 분석합니다.
    """
    # 1. 초기 단가 설정 (구매가)
    unit_costs = latest_prices.copy()

    # 2. 계산 대상인 생산품 목록
    products_to_calc = bom_df[['생산품목코드', '생산품목명']].dropna().drop_duplicates()
    products_to_calc_set = set(products_to_calc['생산품목코드'])

    # 3. 소요량 숫자 타입으로 변환
    bom_df['소요량'] = pd.to_numeric(bom_df['소요량'], errors='coerce').fillna(0)

    # 4. 안정적인 반복 계산 로직
    for _ in range(len(products_to_calc_set) + 5): # 무한루프 방지를 위한 최대 반복 횟수
        made_progress = False
        # 아직 원가가 계산되지 않은 생산품만 대상으로 순회
        remaining_products = [p for p in products_to_calc_set if p not in unit_costs]
        
        for product_code in remaining_products:
            components = bom_df[bom_df['생산품목코드'] == product_code]
            
            # 모든 부품의 원가를 알고 있는지 확인
            can_calculate = all(comp_code in unit_costs for comp_code in components['소모품목코드'])

            if can_calculate:
                # 모든 부품 원가를 알면, 현재 제품 원가 계산
                total_cost = (components['소요량'] * components['소모품목코드'].map(unit_costs).fillna(0)).sum()
                unit_costs[product_code] = total_cost
                made_progress = True # 이번 회차에 계산 성공했음을 표시
        
        # 한 회차에서 어떤 제품도 새로 계산하지 못했다면 더 이상 진행 불가
        if not made_progress:
            break
            
    # 5. 결과 정리
    summary_df = products_to_calc.copy()
    summary_df['계산된 단위 원가'] = summary_df['생산품목코드'].map(unit_costs).fillna(0)
    
    # 상세 내역 및 원인 분석
    details_df = bom_df.copy()
    details_df['부품 단위 원가'] = details_df['소모품목코드'].map(unit_costs).fillna(0)
    details_df['부품별 원가'] = details_df['소요량'] * details_df['부품 단위 원가']
    
    uncalculated_df = summary_df[(summary_df['계산된 단위 원가'] == 0) & (summary_df['생산품목코드'].isin(bom_df['생산품목코드']))]
    
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
            bom_df = bom_df_raw[bom_df_raw['소모품목코드'] != '99701'].copy()
            st.info("'test'(99701) 품목을 BOM 분석에서 제외했습니다.")

            st.header('2. 원가 계산 실행')
            if st.button('모든 완제품 원가 계산하기'):
                with st.spinner('최종 로직으로 전체 원가를 계산 중입니다...'):
                    latest_prices = get_latest_prices(purchase_df)
                    summary_df, details_df, uncalculated_df = calculate_multi_level_bom_costs(bom_df, latest_prices)
                    finished_goods_summary = summary_df[summary_df['생산품목명'].str.contains('[완제품]', regex=False, na=False)]

                    st.header('3. [완제품] 원가 계산 결과 요약')
                    st.dataframe(finished_goods_summary[['생산품목코드', '생산품목명', '계산된 단위 원가']].rename(columns={'생산품목코드':'품목코드', '생산품목명':'품목명'}).style.format({'계산된 단위 원가': '{:,.2f}'}))

                    if not uncalculated_df.empty:
                        with st.expander("⚠️ 원가 0원 항목 분석 (클릭하여 확인)"):
                            st.write("아래 품목들은 구성 부품의 원가 정보가 없어 원가가 0으로 계산되었습니다.")
                            st.dataframe(uncalculated_df[['생산품목코드', '생산품목명']].rename(columns={'생산품목코드':'품목코드', '생산품목명':'품목명'}))

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
