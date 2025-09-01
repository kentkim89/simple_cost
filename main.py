import streamlit as st
import pandas as pd
import io

def load_data(uploaded_file, skiprows=0):
    """
    업로드된 파일의 확장자를 확인하고 적절한 방식으로 데이터를 로드합니다.
    """
    if uploaded_file.name.endswith('.csv'):
        # CSV 파일의 인코딩 문제 방지를 위해 'utf-8-sig' 사용
        return pd.read_csv(uploaded_file, skiprows=skiprows, encoding='utf-8-sig')
    elif uploaded_file.name.endswith('.xlsx'):
        return pd.read_excel(uploaded_file, skiprows=skiprows)
    else:
        st.error("지원하지 않는 파일 형식입니다. CSV 또는 XLSX 파일을 업로드해주세요.")
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

def calculate_all_costs(bom_df, latest_prices):
    """
    BOM에 있는 모든 완제품의 원가를 계산합니다.
    """
    # 완제품 목록 (중복 제거)
    products = bom_df[['생산품목코드', '생산품목명']].dropna().drop_duplicates()
    
    all_details = []
    summary_list = []

    for index, product in products.iterrows():
        product_code = product['생산품목코드']
        product_name = product['생산품목명']
        
        # 해당 완제품의 BOM 데이터 필터링
        product_bom = bom_df[bom_df['생산품목코드'] == product_code].copy()
        
        # BOM과 최신 단가 병합
        merged_df = pd.merge(product_bom, latest_prices, left_on='소모품목코드', right_on='품목코드', how='left')
        merged_df['단가'] = merged_df['단가'].fillna(0)
        merged_df['부품별 원가'] = merged_df['소요량'] * merged_df['단가']
        
        # 총 원가 계산
        total_cost = merged_df['부품별 원가'].sum()
        
        # 요약 리스트에 추가
        summary_list.append({
            '생산품목코드': product_code,
            '생산품목명': product_name,
            '총 원가': total_cost
        })
        
        # 상세 내역 데이터프레임에 생산품목 정보 추가 후 리스트에 추가
        merged_df['완제품 코드'] = product_code
        merged_df['완제품명'] = product_name
        all_details.append(merged_df)

    # 전체 상세 내역을 하나의 데이터프레임으로 결합
    all_details_df = pd.concat(all_details, ignore_index=True)
    summary_df = pd.DataFrame(summary_list)
    
    return summary_df, all_details_df

def main():
    st.title('BOM 기반 전체 제품 원가 일괄 계산기 🏭')

    st.header('1. 파일 업로드')
    bom_file = st.file_uploader("BOM 데이터 (CSV 또는 Excel)", type=['csv', 'xlsx'])
    purchase_file = st.file_uploader("구매 기록 데이터 (CSV 또는 Excel)", type=['csv', 'xlsx'])

    if bom_file and purchase_file:
        bom_df = load_data(bom_file, skiprows=1)
        purchase_df = load_data(purchase_file)

        if bom_df is not None and purchase_df is not None:
            st.header('2. 원가 계산 실행')
            if st.button('모든 제품 원가 계산하기'):
                with st.spinner('전체 제품의 원가를 계산 중입니다...'):
                    # 데이터 처리 및 원가 계산
                    latest_prices = get_latest_prices(purchase_df)
                    summary_df, details_df = calculate_all_costs(bom_df, latest_prices)

                    st.header('3. 계산 결과 요약')
                    st.dataframe(summary_df.style.format({'총 원가': '{:,.2f}'}))

                    # 다운로드를 위한 엑셀 파일 생성
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        summary_df.to_excel(writer, index=False, sheet_name='총 원가 요약')
                        
                        # 상세 내역 시트에 필요한 컬럼만 정리
                        details_display = details_df[[
                            '완제품 코드', '완제품명', '소모품목코드', '소모품목명',
                            '소요량', '단가', '부품별 원가'
                        ]]
                        details_display.to_excel(writer, index=False, sheet_name='상세 원가 내역')
                    
                    st.header('4. 결과 다운로드')
                    st.download_button(
                        label="전체 원가 계산 결과 다운로드 (Excel)",
                        data=output.getvalue(),
                        file_name='전체_제품_원가계산_결과.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )

if __name__ == '__main__':
    main()
