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
    다단계 BOM 구조를 순회하며 원가를 계산하고, 계산 실패 원인을 분석합니다.
    """
    unit_costs = latest_prices.set_index('품목코드')['단가'].to_dict()
    products_to_calculate = bom_df[['생산품목코드', '생산품목명']].dropna().drop_duplicates()
    
    for _ in range(len(products_to_calculate) + 5): # 순환참조 등을 대비해 반복 횟수를 넉넉하게 설정
        newly_calculated_count = 0
        for _, product in products_to_calculate.iterrows():
            product_code = product['생산품목코드']
            if product_code in unit_costs:
                continue

            components = bom_df[bom_df['생산품목코드'] == product_code]
            can_calculate = True
            total_cost = 0
            
            for _, component in components.iterrows():
                comp_code = component['소모품목코드']
                if comp_code not in unit_costs:
                    can_calculate = False
                    break
                total_cost += component['소요량'] * unit_costs.get(comp_code, 0)
            
            if can_calculate:
                unit_costs[product_code] = total_cost
                newly_calculated_count += 1
        
        if newly_calculated_count == 0:
            break
            
    all_costs_df = pd.DataFrame(list(unit_costs.items()), columns=['품목코드', '계산된 단위 원가'])
    
    all_products_info = pd.concat([
        bom_df[['생산품목코드', '생산품목명']].rename(columns={'생산품목코드':'품목코드', '생산품목명':'품목명'}),
        bom_df[['소모품목코드', '소모품목명']].rename(columns={'소모품목코드':'품목코드', '소모품명':'품목명'})
    ]).dropna(subset=['품목코드']).drop_duplicates('품목코드')
    
    summary_df = pd.merge(all_products_info, all_costs_df, on='품목코드', how='left').fillna(0)

    details_df = bom_df.copy()
    details_df['부품 단위 원가'] = details_df['소모품목코드'].map(unit_costs).fillna(0)
    details_df['부품별 원가'] = details_df['소요량']
