from upbit_rl.notify.slack import SlackNotifier

if __name__ == "__main__":
    n = SlackNotifier()
    ok = n.send("테스트: upbit-rl-trader 연결 점검 ✅")
    print("전송 성공" if ok else "전송 실패")