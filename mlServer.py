import mlWrapper
import time

if __name__ == "__main__":
    print("Load Module Finished. Now can scheduling.")
    # Scheduler for readFile.
    while True:
        nowTime = int(time.time())
        # 매 2초 혹은 7초마다 5초마다 검사한다.
        if nowTime % 10 == 2 or nowTime % 10 == 7:
            mlWrapper.checkInput("")
        else:
            time.sleep(1)	# 1초간 휴식