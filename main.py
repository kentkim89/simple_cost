import streamlit as st
import pandas as pd
import io

def load_data(uploaded_file, skiprows=0):
    """
    업로드된 파일의 확장자를 확인하고 적절한 방식으로 데이터를 로드합니다.
    """
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file, skiprows=skiprows)
    elif uploaded_file.name.endswith('.xlsx'):
        return pd.read_excel(uploaded_file, skiprows=skiprows)
    else:
        st.error("지원하지 않는 파일 형식입니다. CSV 또는 XLSX 파일을 업로드해주세요.")
        return None

def get_latest_prices(purchase_df):
    """
    구매 데이터에서 품목별 최신 단가를 추출합니다.
    """
    # '일자-No.' 열을 문자열로 변환하여 날짜 추출 오류 방지
    purchase_df['일자-No.'] = purchase_df['일자-No.'].astype(str)
    # '일자-No.' 열에서 날짜 부분만 추출하여 'date' 열 생성
    purchase_df['date'] = purchase_df['일자-No.'].apply(lambda x: x.split('-')[0])
    # 날짜 형식으로 변환 (오류 발생 시 무시)
    purchase_df['date'] = pd.to_datetime(purchase_df['date'], errors='coerce')
    # 날짜가 유효하지 않은 행 제거
    purchase_df.dropna(subset=['date'], inplace=True)
    # 날짜 기준으로 내림차순 정렬
    purchase_df = purchase_df.sort_values(by='date', ascending=False)
    # 품목코드를 기준으로 중복 제거 (가장 최신 날짜의 데이터만 남김)
    latest_prices = purchase_df.drop_duplicates(subset='품목코드', keep='first')
    return latest_prices[['품목코드', '단가']]

def calculate_cost(bom_df, latest_prices, selected_product_code):
    """
    선택된 완제품의 원가를 계산합니다.
    """
    # 선택된 완제품에 해당하는 BOM 데이터 필터링
    product_bom = bom_df[bom_df['생산품목코드'] == selected_product_code].copy()
    # BOM 데이터와 최신 단가 데이터를 '소모품목코드'와 '품목코드'를 기준으로 병합
    merged_df = pd.merge(product_bom, latest_prices, left_on='소모품목코드', right_on='품목코드', how='left')
    # 단가 정보가 없는 경우(NaN) 0으로 처리
    merged_df['단가'] = merged_df['단가'].fillna(0)
    # 부품별 원가 계산 (소요량 * 단가)
    merged_df['부품별 원가'] = merged_df['소요량'] * merged_df['단가']
    # 최종 원가 계산 (모든 부품 원가의 합)
    total_cost = merged_df['부품별 원가'].sum()
    return total_cost, merged_df

def main():
    st.title('BOM 기반 제품 원가 계산기 🚀')

    # 파일 업로드 위젯
    st.header('1. 파일 업로드')
    bom_file = st.file_uploader("BOM 데이터 (CSV 또는 Excel)", type=['csv', 'xlsx'])
    purchase_file = st.file_uploader("구매 기록 데이터 (CSV 또는 Excel)", type=['csv', 'xlsx'])

    if bom_file and purchase_file:
        # 업로드된 파일 형식에 맞춰 데이터 로드
        bom_df = load_data(bom_file, skiprows=1)
        purchase_df = load_data(purchase_file)

        if bom_df is not None and purchase_df is not None:
            st.header('2. 원가 계산')
            # 완제품 목록 생성 (중복 제거 및 NaN 값 제외)
            product_list = bom_df[['생산품목코드', '생산품목명']].dropna().drop_duplicates()
            product_names = product_list['생산품목명'].tolist()
            
            # 완제품 선택
            selected_product_name = st.selectbox('원가를 계산할 완제품을 선택하세요.', product_names)
            
            # 원가 계산 버튼
            if st.button('원가 계산'):
                if selected_product_name:
                    # 선택된 제품명에 해당하는 제품 코드 찾기
                    selected_product_code = product_list[product_list['생산품목명'] == selected_product_name]['생산품목코드'].iloc[0]

                    # 구매 데이터에서 최신 단가 추출
                    latest_prices = get_latest_prices(purchase_df)
                    
                    # 원가 계산
                    total_cost, result_df = calculate_cost(bom_df, latest_prices, selected_product_code)

                    st.header('3. 계산 결과')
                    st.write(f"**선택된 제품:** {selected_product_name} ({selected_product_code})")
                    st.write(f"**총 원가:** `{total_cost:,.2f} 원`")

                    st.subheader('상세 원가 내역')
                    # 보여줄 컬럼 선택 및 이름 변경
                    display_columns = {
                        '소모품목코드': '부품 코드',
                        '소모품목명': '부품명',
                        '소요량': '소요량',
                        '단가': '적용 단가',
                        '부품별 원가': '부품별 원가'
                    }
                    result_display = result_df[list(display_columns.keys())].rename(columns=display_columns)
                    st.dataframe(result_display)
                    
                    # 엑셀 다운로드 버튼을 위한 데이터 준비
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        result_display.to_excel(writer, index=False, sheet_name='원가계산결과')
                    
                    st.download_button(
                        label="결과 다운로드 (Excel)",
                        data=output.getvalue(),
                        file_name=f'{selected_product_name}_원가계산_결과.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    )
                else:
                    st.warning("제품을 선택해주세요.")

if __name__ == '__main__':
    main()
