# FillLab

Paint.NET Classic의 채우기 도구와 몇 가지 기본 기능을 AI 에이전트(REA + Claude Code)로 분석한 뒤, 동작 명세만 보고 새로 구현한 그림 편집기입니다. 대학 과제용이며 원본의 코드, 리소스, 로고, 아이콘은 사용하지 않았습니다.

## 대상 앱
- 이름: Paint.NET (Classic 라이선스)
- 분석 버전: 5.1.12 / 실행 관찰 버전: 5.2
- 출처: 공식 사이트 getpaint.net에서 설치한 Windows 데스크톱 앱

## 개발 환경
- OS: Windows
- 도구: VS Code, Node.js, REA (rea-agents), Claude Code
- AI 모델: (사용한 모델 이름 입력)
- 구현 언어: Python (tkinter, Pillow)

## 개발 과정
1. REA로 원본을 정적 분석한다 (원본을 수정하거나 실행 코드를 재사용하지 않음)
2. 동작 명세 `docs/SPEC_fill.md`를 만든다 (확인, 추측, 관찰, UNKNOWN을 구분)
3. 설치된 프로그램을 직접 써 보며 UNKNOWN을 관찰로 바꾼다
4. 새 대화에서 명세만 보고 구현한다

## 실행 방법
```
pip install -r requirements.txt
python src/main.py
```

## 기능과 결과
| 기능 | 결과 (성공/부분/실패) | 비고 |
|---|---|---|
| 채우기 (연속/전체) | (입력) | |
| 허용 오차 | (입력) | |
| 브러시 | (입력) | |
| 선택·이동 | (입력) | |
| 실행 취소 | (입력) | |
| 열기·저장 | (입력) | |

## 데모 영상
(URL 입력)

## 한계
- 중간 회색에서의 허용 오차 변환 비율은 검정↔흰색 관찰로 역산한 근사값입니다.
- 명세에서 UNKNOWN인 동작은 `docs/DECISIONS.md`에 기록한 단순 동작으로 구현했습니다.
