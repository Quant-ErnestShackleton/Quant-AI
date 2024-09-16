import pandas as pd
import ast
import json

def data_to_json(is_dutch):
    # CSV 파일을 불러와서 DataFrame으로 변환
    file_path = "./data/generated_dutch_pay_data.csv"  # 여기서 your_file_path에 실제 CSV 파일 경로를 넣으세요
    df = pd.read_csv(file_path)

    # participants_data 열을 리스트로 변환
    df['participants_data'] = df['participants_data'].apply(ast.literal_eval)

    # DataFrame을 딕셔너리 리스트로 변환
    data = df.to_dict(orient='records')

    # JSON 파일로 저장
    if is_dutch :
        output_json_path = "./template/dutch_template_data.json"  # JSON 파일 경로를 지정
    else :
        output_json_path = "./template/non_dutch_template_data.json"
    with open(output_json_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    print(f"JSON 파일이 {output_json_path}에 저장되었습니다.")
