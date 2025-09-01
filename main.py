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
    다단계 BOM 원가를 올바르게 계산합니다.
    중간재(다른 생산품목)도 소모품목으로 사용되는 경우를 처리합니다.
    """
    # 1. 초기 단가 설정 (구매가만)
    unit_costs = latest_prices.copy()
    
    # 2. 모든 생산품목 목록 (완제품 + 중간재)
    all_products = bom_df[['생산품목코드', '생산품목명']].dropna().drop_duplicates()
    all_products_set = set(all_products['생산품목코드'])
    
    # 3. BOM에서 소모품목으로 사용되는 생산품목들 식별
    bom_components = set(bom_df['소모품목코드'].dropna())
    internal_components = bom_components.intersection(all_products_set)
    
    # 4. 소요량 숫자 타입으로 변환
    bom_df['소요량'] = pd.to_numeric(bom_df['소요량'], errors='coerce').fillna(0)
    
    # 5. 디버깅 정보 출력
    st.write(f"📊 **계산 정보**")
    st.write(f"- 전체 생산품목 수: {len(all_products_set)}")
    st.write(f"- 구매 데이터에서 찾은 품목 수: {len(latest_prices)}")
    st.write(f"- BOM 내부에서 중간재로 사용되는 품목 수: {len(internal_components)}")
    
    # 6. 반복 계산으로 다단계 BOM 원가 계산
    max_iterations = len(all_products_set) + 10
    calculation_log = []
    
    for iteration in range(max_iterations):
        made_progress = False
        remaining_products = [p for p in all_products_set if p not in unit_costs]
        
        if not remaining_products:
            break
            
        for product_code in remaining_products:
            components = bom_df[bom_df['생산품목코드'] == product_code]
            
            if components.empty:
                continue
                
            # 모든 소모품목의 원가를 알고 있는지 확인
            missing_components = []
            can_calculate = True
            
            for _, comp_row in components.iterrows():
                comp_code = comp_row['소모품목코드']
                if comp_code not in unit_costs:
                    missing_components.append(comp_code)
                    can_calculate = False
            
            if can_calculate:
                # 원가 계산
                total_cost = 0
                detail_log = []
                
                for _, comp_row in components.iterrows():
                    comp_code = comp_row['소모품목코드']
                    comp_name = comp_row['소모품목명']
                    quantity = comp_row['소요량']
                    unit_price = unit_costs[comp_code]
                    component_cost = quantity * unit_price
                    total_cost += component_cost
                    
                    detail_log.append(f"  - {comp_name}({comp_code}): {quantity} × {unit_price:,.2f} = {component_cost:,.2f}")
                
                unit_costs[product_code] = total_cost
                made_progress = True
                
                # 계산 로그 저장
                product_name = components.iloc[0]['생산품목명']
                calculation_log.append({
                    'iteration': iteration + 1,
                    'product_code': product_code,
                    'product_name': product_name,
                    'total_cost': total_cost,
                    'details': detail_log
                })
        
        # 진전이 없으면 중단
        if not made_progress:
            break
    
    # 7. 결과 정리
    summary_df = all_products.copy()
    summary_df['계산된 단위 원가'] = summary_df['생산품목코드'].map(unit_costs)
    
    # NaN을 0으로 처리하되, 실제로 계산되지 않은 것들을 구분
    calculated_mask = summary_df['생산품목코드'].isin(unit_costs.keys())
    summary_df['계산된 단위 원가'] = summary_df['계산된 단위 원가'].fillna(0)
    summary_df['계산 상태'] = calculated_mask.map({True: '계산완료', False: '계산불가'})
    
    # 상세 내역
    details_df = bom_df.copy()
    details_df['부품 단위 원가'] = details_df['소모품목코드'].map(unit_costs).fillna(0)
    details_df['부품별 원가'] = details_df['소요량'] * details_df['부품 단위 원가']
    
    # 계산되지 않은 항목들
    uncalculated_df = summary_df[summary_df['계산 상태'] == '계산불가'].copy()
    
    # 계산되지 않은 이유 분석
    if not uncalculated_df.empty:
        reason_analysis = []
        for _, row in uncalculated_df.iterrows():
            product_code = row['생산품목코드']
            components = bom_df[bom_df['생산품목코드'] == product_code]['소모품목코드'].tolist()
            missing_comps = [c for c in components if c not in unit_costs]
            reason_analysis.append({
                '품목코드': product_code,
                '품목명': row['생산품목명'],
                '부족한 부품 수': len(missing_comps),
                '부족한 부품들': ', '.join(missing_comps[:3]) + ('...' if len(missing_comps) > 3 else '')
            })
        
        reason_df = pd.DataFrame(reason_analysis)
        uncalculated_df = uncalculated_df.merge(reason_df, left_on='생산품목코드', right_on='품목코드', how='left')
    
    return summary_df, details_df, uncalculated_df, calculation_log

def main():
    st.title('BOM 원가 계산기 (오류 수정 버전) 🚀')

    st.header('1. 파일 업로드')
    bom_file = st.file_uploader("BOM 데이터 (CSV 또는 Excel)", type=['csv', 'xlsx'])
    purchase_file = st.file_uploader("구매 기록 데이터 (CSV 또는 Excel)", type=['csv', 'xlsx'])

    if bom_file and purchase_file:
        bom_df_raw = load_data(bom_file, skiprows=1)
        purchase_df = load_data(purchase_file)

        if bom_df_raw is not None and purchase_df is not None:
            # test 품목 제외
            bom_df = bom_df_raw[bom_df_raw['소모품목코드'] != '99701'].copy()
            st.info("'test'(99701) 품목을 BOM 분석에서 제외했습니다.")

            st.header('2. 원가 계산 실행')
            if st.button('모든 완제품 원가 계산하기'):
                with st.spinner('개선된 로직으로 전체 원가를 계산 중입니다...'):
                    latest_prices = get_latest_prices(purchase_df)
                    summary_df, details_df, uncalculated_df, calculation_log = calculate_multi_level_bom_costs(bom_df, latest_prices)
                    
                    # 완제품만 필터링
                    finished_goods_summary = summary_df[summary_df['생산품목명'].str.contains('[완제품]', regex=False, na=False)].copy()

                    st.header('3. [완제품] 원가 계산 결과 요약')
                    
                    # 계산 성공/실패 통계
                    total_finished = len(finished_goods_summary)
                    calculated_finished = len(finished_goods_summary[finished_goods_summary['계산 상태'] == '계산완료'])
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("전체 완제품 수", total_finished)
                    with col2:
                        st.metric("계산 성공", calculated_finished, f"{calculated_finished/total_finished*100:.1f}%" if total_finished > 0 else "0%")
                    with col3:
                        st.metric("계산 실패", total_finished - calculated_finished)
                    
                    # 결과 테이블
                    display_df = finished_goods_summary[['생산품목코드', '생산품목명', '계산된 단위 원가', '계산 상태']].rename(columns={
                        '생산품목코드': '품목코드', 
                        '생산품목명': '품목명'
                    })
                    
                    # 계산 상태별로 색상 구분하여 표시
                    def highlight_status(row):
                        if row['계산 상태'] == '계산완료':
                            return ['background-color: #d4edda'] * len(row)
                        else:
                            return ['background-color: #f8d7da'] * len(row)
                    
                    styled_df = display_df.style.apply(highlight_status, axis=1).format({
                        '계산된 단위 원가': '{:,.2f}'
                    })
                    
                    st.dataframe(styled_df, use_container_width=True)

                    # 계산 과정 로그 표시
                    if calculation_log:
                        with st.expander("🔍 계산 과정 상세 로그 (클릭하여 확인)"):
                            for log_entry in calculation_log[:20]:  # 처음 20개만 표시
                                st.write(f"**{log_entry['iteration']}차 계산: {log_entry['product_name']}({log_entry['product_code']})**")
                                st.write(f"총 원가: {log_entry['total_cost']:,.2f}원")
                                for detail in log_entry['details']:
                                    st.write(detail)
                                st.write("---")

                    # 계산 실패 항목 분석
                    if not uncalculated_df.empty:
                        with st.expander("⚠️ 원가 계산 실패 항목 분석 (클릭하여 확인)"):
                            st.write("다음 품목들은 구성 부품의 원가 정보가 부족하여 계산할 수 없었습니다:")
                            
                            failed_finished = uncalculated_df[uncalculated_df['생산품목명'].str.contains('[완제품]', regex=False, na=False)]
                            if not failed_finished.empty:
                                st.write("**완제품 계산 실패 목록:**")
                                st.dataframe(failed_finished[['생산품목코드', '생산품목명', '부족한 부품 수', '부족한 부품들']])
                            
                            st.write("**전체 계산 실패 목록:**")
                            st.dataframe(uncalculated_df[['생산품목코드', '생산품목명', '부족한 부품 수', '부족한 부품들']])

                    # 결과 다운로드
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        finished_goods_summary.to_excel(writer, index=False, sheet_name='완제품 원가 요약')
                        summary_df.to_excel(writer, index=False, sheet_name='전체 제품 원가 요약')
                        details_df.to_excel(writer, index=False, sheet_name='상세 원가 내역')
                        if not uncalculated_df.empty:
                            uncalculated_df.to_excel(writer, index=False, sheet_name='계산 실패 항목')
                    
                    st.header('4. 결과 다운로드')
                    st.download_button(
                        label="완제품 원가 계산 결과 다운로드 (Excel)",
                        data=output.getvalue(),
                        file_name='완제품_원가계산_결과_수정본.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )

if __name__ == '__main__':
    main()
